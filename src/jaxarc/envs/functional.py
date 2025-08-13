"""
Functional API for JaxARC environments with Hydra integration.

This module provides pure functional implementations of the ARC environment
that work with both typed configs and Hydra DictConfig objects.
"""

from __future__ import annotations

import time
from typing import Any, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from jaxarc.configs import JaxArcConfig
from jaxarc.utils.jax_types import EPISODE_MODE_TEST, EPISODE_MODE_TRAIN

from ..state import ArcEnvState
from ..types import JaxArcTask
from ..utils.jax_types import (
    NUM_OPERATIONS,
    EpisodeDone,
    ObservationArray,
    PRNGKey,
    RewardValue,
    get_action_record_fields,
)
from ..utils.state_utils import (
    increment_step_count,
    set_episode_done,
    update_selection,
)
from ..utils.validation import validate_action, validate_state_consistency
from .action_history import ActionHistoryTracker, HistoryConfig
from .action_space_controller import ActionSpaceController
from .actions import (
    BboxAction,
    MaskAction,
    PointAction,
    StructuredAction,
    bbox_handler,
    create_mask_action,
    mask_handler,
    point_handler,
)
from .grid_initialization import initialize_working_grids
from .grid_operations import compute_grid_similarity, execute_grid_operation
from .observation import create_observation
from .reward import _calculate_reward
from .termination import _is_episode_done

# JAX-compatible step info structure - replaces dict for performance.
class StepInfo(eqx.Module):
    """Step info as an Equinox Module for PyTree compatibility."""
    similarity: jax.Array
    similarity_improvement: jax.Array
    operation_type: jax.Array
    step_count: jax.Array
    episode_done: jax.Array
    episode_mode: jax.Array
    current_pair_index: jax.Array
    available_demo_pairs: jax.Array
    available_test_pairs: jax.Array
    action_history_length: jax.Array
    success: jax.Array

# Type aliases for cleaner signatures
ConfigType = Union[JaxArcConfig, DictConfig]


def _ensure_config(config: ConfigType) -> JaxArcConfig:
    """Convert config to typed JaxArcConfig if needed."""
    if isinstance(config, DictConfig):
        return JaxArcConfig.from_hydra(config)
    return config

def _initialize_grids(
    task_data: JaxArcTask,
    selected_pair_idx: jnp.ndarray,
    episode_mode: int,
    config: JaxArcConfig,
    key: PRNGKey | None = None,
    initial_pair_idx: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize grids with diverse initialization strategies - enhanced helper function.

    This enhanced helper function sets up the initial, target, and mask grids based on the
    episode mode, selected pair, and diverse initialization configuration. It supports
    multiple initialization modes including demo, permutation, empty, and random grids.

    Args:
        task_data: JaxArcTask containing demonstration and test pair data
        selected_pair_idx: JAX array with the index of the selected pair
        episode_mode: Episode mode (0=train, 1=test) determining grid initialization
        config: Environment configuration containing grid initialization settings
        key: Optional JAX PRNG key for diverse initialization (required for non-demo modes)
        initial_pair_idx: Optional specific pair index for demo-based initialization.
                         If None, uses random selection. If specified, uses that pair.

    Returns:
        Tuple containing:
        - initial_grid: Starting grid for the episode (JAX array)
        - target_grid: Target grid (visible in train mode, masked in test mode)
        - initial_mask: Boolean mask indicating valid grid cells

    Examples:
        ```python
        # Training mode with diverse initialization and specific pair
        init_grid, target, mask = _initialize_grids(
            task, idx, 0, config, key, initial_pair_idx=2
        )

        # Test mode initialization (target masked) with random selection
        init_grid, masked_target, mask = _initialize_grids(task, idx, 1, config, key)
        ```

    Note:
        Uses the new diverse grid initialization engine when grid_initialization
        config is not in demo mode. Falls back to original behavior for demo mode
        or when key is not provided. Respects initial_pair_idx for demo-based modes.
    """

    # Get target grid and mask based on episode mode
    def get_train_target():
        target_grid = task_data.output_grids_examples[selected_pair_idx]
        target_mask = task_data.output_masks_examples[selected_pair_idx]
        return target_grid, target_mask

    def get_test_target():
        # In test mode, target grid is masked (set to background color) to prevent cheating
        background_color = config.dataset.background_color
        initial_grid = task_data.test_input_grids[selected_pair_idx]
        target_grid = jnp.full_like(initial_grid, background_color)
        target_mask = jnp.zeros_like(task_data.test_input_masks[selected_pair_idx])
        return target_grid, target_mask

    target_grid, target_mask = jax.lax.cond(
        episode_mode == EPISODE_MODE_TRAIN, get_train_target, get_test_target
    )

    # Always use diverse initialization engine when config present and key provided
    use_diverse_init = hasattr(config, "grid_initialization") and key is not None

    if use_diverse_init:
        # Use diverse initialization engine with initial_pair_idx support
        initial_grid, initial_mask = initialize_working_grids(
            task_data,
            config.grid_initialization,
            key,
            batch_size=1,
            initial_pair_idx=initial_pair_idx,
        )
        # Remove batch dimension (squeeze first axis)
        initial_grid = jnp.squeeze(initial_grid, axis=0)
        initial_mask = jnp.squeeze(initial_mask, axis=0)
    else:
        msg = "grid_initialization config missing or PRNG key not provided"
        raise ValueError(msg)

    return initial_grid, target_grid, initial_mask, target_mask


def _create_initial_state(
    task_data: JaxArcTask,
    initial_grid: jnp.ndarray,
    target_grid: jnp.ndarray,
    initial_mask: jnp.ndarray,
    target_mask: jnp.ndarray,
    selected_pair_idx: jnp.ndarray,
    episode_mode: int,
    config: JaxArcConfig,
) -> ArcEnvState:
    """Create initial state - focused helper function.

    This helper function constructs the complete initial ArcEnvState with all
    required fields properly initialized. It handles enhanced functionality
    including action history, operation masks, and completion tracking.

    Args:
        task_data: JaxArcTask containing complete task information
        initial_grid: Starting grid configuration (JAX array)
        target_grid: Target grid (visible or masked based on mode)
        initial_mask: Boolean mask for valid grid cells
        selected_pair_idx: Index of the currently selected pair
        episode_mode: Episode mode (0=train, 1=test)
        config: Environment configuration for state initialization parameters

    Returns:
        ArcEnvState: Complete initial environment state with all fields properly
                    initialized including action history, completion tracking,
                    and operation masks.

    Examples:
        ```python
        # Create initial state for training
        state = _create_initial_state(task, grid, target, mask, target_mask, idx, 0, config)

        # Create initial state for testing
        state = _create_initial_state(
            task, grid, masked_target, mask, empty_mask, idx, 1, config
        )
        ```

    Note:
        Initializes enhanced features like action history storage, completion
        status tracking, and dynamic operation control for full functionality.
    """
    # Calculate initial similarity (will be 0.0 in test mode due to masked target)
    initial_similarity = compute_grid_similarity(
        initial_grid, initial_mask, target_grid, target_mask
    )

    # Initialize grids based on episode mode using JAX-compatible operations
    # Get available pairs and completion status for enhanced functionality
    available_demo_pairs = task_data.get_available_demo_pairs()
    available_test_pairs = task_data.get_available_test_pairs()
    demo_completion_status = jnp.zeros_like(available_demo_pairs)
    test_completion_status = jnp.zeros_like(available_test_pairs)

    # Initialize action history with dynamic sizing based on configuration
    max_history_length = getattr(config, "max_history_length", 1000)
    num_operations = NUM_OPERATIONS

    # Calculate optimal action record fields based on selection format and dataset
    action_record_fields = get_action_record_fields(
        config.action.selection_format,
        config.dataset.max_grid_height,
        config.dataset.max_grid_width,
    )

    # Initialize action history storage with proper dimensions
    action_history = jnp.zeros(
        (max_history_length, action_record_fields), dtype=jnp.float32
    )
    action_history_length = jnp.array(0, dtype=jnp.int32)
    action_history_write_pos = jnp.array(0, dtype=jnp.int32)

    # Initialize allowed operations mask (all operations allowed by default)
    allowed_operations_mask = jnp.ones(num_operations, dtype=jnp.bool_)

    # Create enhanced initial state with all new fields
    return ArcEnvState(
        # Core ARC state (unchanged)
        task_data=task_data,
        working_grid=initial_grid,
        working_grid_mask=initial_mask,
        target_grid=target_grid,
        target_grid_mask=initial_mask,  # Same mask as working grid
        step_count=jnp.array(0, dtype=jnp.int32),
        episode_done=jnp.array(False),
        current_example_idx=selected_pair_idx,
        selected=jnp.zeros_like(initial_grid, dtype=jnp.bool_),
        clipboard=jnp.zeros_like(initial_grid, dtype=jnp.int32),
        similarity_score=initial_similarity,
        # Enhanced functionality fields
        episode_mode=episode_mode,
        available_demo_pairs=available_demo_pairs,
        available_test_pairs=available_test_pairs,
        demo_completion_status=demo_completion_status,
        test_completion_status=test_completion_status,
        action_history=action_history,
        action_history_length=action_history_length,
        action_history_write_pos=action_history_write_pos,
        allowed_operations_mask=allowed_operations_mask,
    )


@eqx.filter_jit
def arc_reset(
    key: PRNGKey,
    config: ConfigType,
    task_data: JaxArcTask | None = None,
    episode_mode: int = 0,  # 0=train, 1=test (JAX-compatible integers)
    initial_pair_idx: int | None = None,
) -> tuple[ArcEnvState, ObservationArray]:
    """
    Reset ARC environment with enhanced multi-demonstration support.

    This enhanced reset function provides comprehensive support for multi-demonstration
    training and test pair evaluation with proper mode-specific initialization,
    action history setup, and dynamic operation control. The function has been
    decomposed into focused helper functions for better maintainability while
    preserving JAX compatibility and performance.

    The reset process involves:
    1. Task data acquisition or demo task creation
    2. Initial pair selection based on mode and strategy
    3. Grid initialization with proper target masking
    4. Complete state creation with enhanced features

    Args:
        key: JAX PRNG key for reproducible randomization
        config: Environment configuration (typed JaxArcConfig or Hydra DictConfig).
               Automatically converted to typed config if needed.
        task_data: Optional specific task data. If None, will use parser from config
                  or create demo task as fallback.
        episode_mode: Episode mode (0=train, 1=test) for JAX-compatible initialization.
                     Determines pair type selection and target visibility.
        initial_pair_idx: Optional explicit pair index specification. If None,
                         uses episode manager selection strategy.

    Returns:
        Tuple of (initial_state, initial_observation) with enhanced functionality
        including action history, completion tracking, and operation control.

    Examples:
        ```python
        # Reset in training mode with random demo pair selection
        state, obs = arc_reset(key, config, task_data, episode_mode=0)

        # Reset in test mode with specific test pair
        state, obs = arc_reset(key, config, task_data, episode_mode=1, initial_pair_idx=1)

        # Reset with explicit pair selection in training mode
        state, obs = arc_reset(key, config, task_data, episode_mode=0, initial_pair_idx=2)

        # Using typed config (preferred)
        from jaxarc.configs import JaxArcConfig

        typed_config = JaxArcConfig.from_hydra(hydra_config)
        state, obs = arc_reset(key, typed_config, task_data)
        ```
    """
    # Ensure we have a typed config
    typed_config = _ensure_config(config)

    if not task_data and not initial_pair_idx:
        # Raise error if no task data or initial pair index provided
        msg = (
            "Either task_data or initial_pair_idx must be provided. "
            "For training, provide task_data or set initial_pair_idx to a valid pair index."
        )
        raise ValueError(msg)

    # Validate episode mode (0=train, 1=test) using JAX-compatible integer check
    if episode_mode not in [EPISODE_MODE_TRAIN, EPISODE_MODE_TEST]:
        msg = (
            f"episode_mode must be {EPISODE_MODE_TRAIN} (train) or {EPISODE_MODE_TEST} (test), "
            f"got '{episode_mode}'"
        )
        raise ValueError(msg)
    # Split key for different operations
    _, init_key = jax.random.split(key)

    # Select first pair index, if not provided
    selected_pair_idx = jnp.array(0, dtype=jnp.int32)
    if initial_pair_idx is not None:
        # Use explicit pair index if provided
        selected_pair_idx = jnp.array(initial_pair_idx, dtype=jnp.int32)

        # Validate the explicit pair index using JAX-compatible conditional logic
        selection_successful = jax.lax.cond(
            episode_mode == EPISODE_MODE_TRAIN,
            lambda: task_data.is_demo_pair_available(initial_pair_idx),
            lambda: task_data.is_test_pair_available(initial_pair_idx),
        )
        if not selection_successful:
            msg = (
                f"Initial pair index {initial_pair_idx} is not available in "
                f"{['demo', 'test'][episode_mode]} pairs."
            )
            raise ValueError(msg)

    # Initialize grids based on episode mode using JAX-compatible operations
    initial_grid, target_grid, initial_mask, target_mask = _initialize_grids(
        task_data,
        selected_pair_idx,
        episode_mode,
        typed_config,
        init_key,
        initial_pair_idx,
    )

    # Create enhanced initial state with all new fields
    state = _create_initial_state(
        task_data,
        initial_grid,
        target_grid,
        initial_mask,
        target_mask,
        selected_pair_idx,
        episode_mode,
        typed_config,
    )

    # Validate initial state consistency
    validated_state = validate_state_consistency(state)

    # Create initial observation using the enhanced create_observation function
    observation = create_observation(validated_state, typed_config)

    # Logging removed for performance - callbacks cause 10-50x slowdown
    # All logging information is now available in the returned state for external logging

    return validated_state, observation
def _process_action(
    state: ArcEnvState,
    action: StructuredAction | dict[str, Any],
    config: JaxArcConfig,
) -> tuple[ArcEnvState, StructuredAction]:
    """Process action and return updated state.

    This function handles the complete action processing pipeline supporting both
    structured actions and dictionary actions for backward compatibility. It supports
    both grid operations (0-34) and control operations (35-41) with proper validation
    and execution.

    Args:
        state: Current environment state before action execution
        action: Action to execute (StructuredAction or dict with 'operation' and 'selection')
        config: Environment configuration for action processing settings

    Returns:
        Tuple containing:
        - new_state: Updated environment state after action execution
        - validated_action: Validated structured action

    Examples:
        ```python
        # Process structured point action
        action = PointAction(operation=15, row=5, col=10)
        new_state, validated_action, is_control = _process_action(state, action, config)

        # Process dictionary action (backward compatibility)
        action = {"operation": 15, "selection": jnp.array([5, 10])}
        new_state, validated_action, is_control = _process_action(state, action, config)
        ```

    Raises:
        ValueError: If action format is invalid or operation is out of range

    Note:
        Supports both structured actions and dictionary actions for backward compatibility.
        Automatically validates action parameters and clips to valid ranges.
    """
    # Get grid shape for validation
    grid_shape = state.working_grid.shape

    # Convert dictionary action to structured action if needed
    if isinstance(action, dict):
        # Handle dictionary action for backward compatibility
        operation = action["operation"]
        selection = action["selection"]

        # Convert to MaskAction (most general format)
        if selection.ndim == 2:
            # Already a 2D mask
            validated_action = create_mask_action(operation, selection)
        elif selection.ndim == 1 and len(selection) == 2:
            # Point format [row, col] - convert to mask
            mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
            row = jnp.clip(selection[0].astype(jnp.int32), 0, grid_shape[0] - 1)
            col = jnp.clip(selection[1].astype(jnp.int32), 0, grid_shape[1] - 1)
            mask = mask.at[row, col].set(True)
            validated_action = create_mask_action(operation, mask)
        elif selection.ndim == 1 and len(selection) == 4:
            # Bbox format [r1, c1, r2, c2] - convert to mask
            r1, c1, r2, c2 = selection.astype(jnp.int32)
            r1 = jnp.clip(r1, 0, grid_shape[0] - 1)
            c1 = jnp.clip(c1, 0, grid_shape[1] - 1)
            r2 = jnp.clip(r2, 0, grid_shape[0] - 1)
            c2 = jnp.clip(c2, 0, grid_shape[1] - 1)

            min_r, max_r = jnp.minimum(r1, r2), jnp.maximum(r1, r2)
            min_c, max_c = jnp.minimum(c1, c2), jnp.maximum(c1, c2)

            rows = jnp.arange(grid_shape[0])
            cols = jnp.arange(grid_shape[1])
            row_mesh, col_mesh = jnp.meshgrid(rows, cols, indexing="ij")

            mask = (
                (row_mesh >= min_r)
                & (row_mesh <= max_r)
                & (col_mesh >= min_c)
                & (col_mesh <= max_c)
            )
            validated_action = create_mask_action(operation, mask)
        elif selection.ndim == 1 and len(selection) == grid_shape[0] * grid_shape[1]:
            # Flattened mask - reshape to 2D
            mask = selection.reshape(grid_shape).astype(jnp.bool_)
            validated_action = create_mask_action(operation, mask)
        else:
            msg = f"Unsupported selection format: shape {selection.shape}"
            raise ValueError(msg)
    else:
        # Handle structured action
        validated_action = action.validate(grid_shape, max_operations=NUM_OPERATIONS)

    # Extract operation from validated action
    operation = validated_action.operation

    # Apply dynamic action space validation and filtering if enabled
    if (
        hasattr(config.action, "dynamic_action_filtering")
        and config.action.dynamic_action_filtering
    ):
        controller = ActionSpaceController()

        # Filter invalid operation according to policy
        operation = controller.filter_invalid_operation_jax(
            operation, state, config.action
        )

        # Update validated action with filtered operation
        if isinstance(validated_action, PointAction):
            validated_action = PointAction(
                operation=operation, row=validated_action.row, col=validated_action.col
            )
        elif isinstance(validated_action, BboxAction):
            validated_action = BboxAction(
                operation=operation,
                r1=validated_action.r1,
                c1=validated_action.c1,
                r2=validated_action.r2,
                c2=validated_action.c2,
            )
        elif isinstance(validated_action, MaskAction):
            validated_action = MaskAction(
                operation=operation, selection=validated_action.selection
            )

    # Only grid operations remain (0-34). Execute directly.
    if isinstance(validated_action, PointAction):
        selection_mask = point_handler(validated_action, state.working_grid_mask)
    elif isinstance(validated_action, BboxAction):
        selection_mask = bbox_handler(validated_action, state.working_grid_mask)
    elif isinstance(validated_action, MaskAction):
        selection_mask = mask_handler(validated_action, state.working_grid_mask)
    else:
        selection_mask = validated_action.to_selection_mask(grid_shape)
        selection_mask = selection_mask & state.working_grid_mask

    new_state = update_selection(state, selection_mask)
    new_state = execute_grid_operation(new_state, operation)
    return new_state, validated_action

def _update_state(
    _old_state: ArcEnvState,
    new_state: ArcEnvState,
    action: StructuredAction,
    config: JaxArcConfig,
) -> ArcEnvState:
    """Update state with action history and step count - focused helper function.

    This helper function handles post-action state updates including action history
    tracking, step count incrementation, and other bookkeeping operations that
    need to occur after action execution.

    Args:
        old_state: Environment state before action execution
        new_state: Environment state after action execution but before updates
        action: Structured action that was executed
        config: Environment configuration for history and update settings

    Returns:
        ArcEnvState: Updated state with action history, incremented step count,
                    and other post-action updates applied.

    Examples:
        ```python
        # Update state after action processing
        updated_state = _update_state(old_state, new_state, action, config)
        ```

    Note:
        Conditionally applies action history tracking based on configuration.
        Always increments step count and handles other required state updates.
    """

    # Add action history tracking for each step with memory optimization
    # Use JAX-compatible conditional for history tracking
    def add_to_history(state):
        """Add action to history if enabled."""
        # Create history config from main config or use defaults
        history_config = HistoryConfig(
            enabled=getattr(config.history, "enabled", True),
            max_history_length=getattr(config.history, "max_history_length", 1000),
            store_selection_data=getattr(config.history, "store_selection_data", True),
            compress_repeated_actions=getattr(
                config.history, "compress_repeated_actions", True
            ),
        )

        # Add action to history
        tracker = ActionHistoryTracker()
        return tracker.add_action(
            state,
            action,
            history_config,
            config.action.selection_format,
            config.dataset.max_grid_height,
            config.dataset.max_grid_width,
        )

    # Check if history is enabled and apply conditionally
    history_enabled = hasattr(config, "history") and getattr(
        config.history, "enabled", True
    )
    updated_state = jax.lax.cond(
        history_enabled,
        add_to_history,
        lambda s: s,  # No-op if history disabled
        new_state,
    )

    # Update step count using PyTree utilities
    return increment_step_count(updated_state)

def _calculate_reward_and_done(
    old_state: ArcEnvState,
    new_state: ArcEnvState,
    config: JaxArcConfig,
) -> tuple[RewardValue, EpisodeDone, ArcEnvState]:
    """Compute reward and termination status.

    Simplified version after removal of control operations. Delegates reward
    computation to _calculate_enhanced_reward with control flag fixed False.
    """
    reward = _calculate_reward(old_state, new_state, config)

    # Check if episode is done with enhanced termination logic
    done = _is_episode_done(new_state, config)

    # Update episode_done flag using PyTree utilities
    final_state = set_episode_done(new_state, done)

    return reward, done, final_state


@eqx.filter_jit(donate="all")
def _arc_step_unsafe(
    state: ArcEnvState,
    action: StructuredAction,
    config: ConfigType,
) -> tuple[ArcEnvState, ObservationArray, RewardValue, EpisodeDone, StepInfo]:
    """
    Execute single step in ARC environment with comprehensive functionality.

    This enhanced step function provides comprehensive action processing with
    support for both grid operations (0-34) and control operations (35-41).
    The function has been decomposed into focused helper functions for better
    maintainability while preserving JAX compatibility and performance.

    The step process involves:
    1. Action processing and validation
    2. State updates with history tracking
    Features:
    - Support for enhanced non-parametric control operations (35-41)
    - Action history tracking for each step with memory optimization
    - Dynamic action space validation and filtering
    - Mode-specific reward calculation logic
    - Non-parametric pair switching logic (next/prev/first_unsolved)
    - Focused agent observation generation

    Args:
        state: Current environment state with enhanced functionality including
               action history, completion tracking, and operation control
        state: ArcEnvState,
        action: StructuredAction,
        config: ConfigType,
     ) -> tuple[ArcEnvState, ObservationArray, RewardValue, EpisodeDone, dict[str, Any]]:
               Contains operation ID and selection data in type-safe format.
        config: Environment configuration (typed JaxArcConfig or Hydra DictConfig).
               Automatically converted to typed config if needed.

    Returns:
        Tuple of (new_state, agent_observation, reward, done, info) where:
        - new_state: Updated environment state after action execution
        - agent_observation: Focused observation for agent (currently working grid)
        - reward: Calculated reward value with mode-specific logic
        - done: Boolean indicating episode termination
        - info: JAX-compatible StepInfo structure with step information and context

    Examples:
        ```python
        # Point-based grid operation
        action = PointAction(operation=15, row=5, col=10)
        new_state, obs, reward, done, info = _arc_step_unsafe(state, action, config)

        # Bounding box grid operation
        action = BboxAction(operation=10, r1=2, c1=3, r2=5, c2=7)
        new_state, obs, reward, done, info = _arc_step_unsafe(state, action, config)

        # Mask-based grid operation
        mask = jnp.zeros((30, 30), dtype=jnp.bool_).at[5:10, 5:10].set(True)
        action = MaskAction(operation=20, selection=mask)
        new_state, obs, reward, done, info = arc_step(state, action, config)

        # Control operation - switch to next demo pair
        action = PointAction(operation=35, row=0, col=0)  # Coordinates ignored for control ops
        new_state, obs, reward, done, info = arc_step(state, action, config)

        # Using typed config (preferred)
        from jaxarc.configs import JaxArcConfig

        typed_config = JaxArcConfig.from_hydra(hydra_config)
        new_state, obs, reward, done, info = _arc_step_unsafe(state, action, typed_config)
        ```

    Note:
        Function has been decomposed into helper functions (_process_action,
        _update_state, _calculate_reward_and_done) for better maintainability
        while preserving performance and JAX compatibility.
    """
    # Ensure we have a typed config
    typed_config = _ensure_config(config)

    # Validate action and state before processing
    validated_action = validate_action(action, typed_config)
    validated_state = validate_state_consistency(state)

    # Process action and get updated state
    new_state, standardized_action = _process_action(
        validated_state, validated_action, typed_config
    )

    # Update state with action history and step count
    updated_state = _update_state(
        validated_state, new_state, standardized_action, typed_config
    )

    # Calculate reward and done status
    reward, done, final_state = _calculate_reward_and_done(
        validated_state, updated_state, typed_config
    )

    # Use create_observation function to generate focused agent view
    observation = create_observation(final_state, typed_config)

    # Create JAX-compatible info structure - replaces dict for performance
    info = StepInfo(
        similarity=final_state.similarity_score,
        similarity_improvement=final_state.similarity_score - state.similarity_score,
        operation_type=standardized_action.operation,
        step_count=final_state.step_count,
        episode_done=done,
        episode_mode=final_state.episode_mode,
        current_pair_index=final_state.current_example_idx,
        available_demo_pairs=jnp.sum(final_state.available_demo_pairs),
        available_test_pairs=jnp.sum(final_state.available_test_pairs),
        action_history_length=final_state.get_action_history_length(),
        success=final_state.similarity_score >= 1.0,
    )

    # All logging and visualization callbacks removed for performance
    # Callbacks cause 10-50x slowdown due to device-to-host transfers
    # All information is now available in the StepInfo structure for external logging

    return final_state, observation, reward, done, info


@eqx.filter_jit
def validate_action_jax(
    action: StructuredAction, state: ArcEnvState, _config: ConfigType
) -> jax.Array:
    """JAX-friendly validation returning a boolean predicate.

    This avoids raising inside JIT and can be used with lax.cond.
    """
    # Operation bounds check
    max_ops = jnp.asarray(NUM_OPERATIONS, dtype=jnp.int32)
    op = jnp.asarray(action.operation, dtype=jnp.int32)
    op_valid = (op >= 0) & (op < max_ops)

    # Grid bounds from static shapes
    grid_h: int = state.working_grid.shape[0]
    grid_w: int = state.working_grid.shape[1]

    # Selection-specific checks
    def _check_point(a):
        return (a.row >= 0) & (a.row < grid_h) & (a.col >= 0) & (a.col < grid_w)

    def _check_bbox(a):
        # All coords must be within grid (ordering handled during execution)
        return (
            (a.r1 >= 0)
            & (a.r1 < grid_h)
            & (a.c1 >= 0)
            & (a.c1 < grid_w)
            & (a.r2 >= 0)
            & (a.r2 < grid_h)
            & (a.c2 >= 0)
            & (a.c2 < grid_w)
        )

    def _check_mask(a):
        sel_shape = a.selection.shape
        # Compare static shapes; coerce to boolean scalar array
        shape_ok = (sel_shape[0] == grid_h) and (sel_shape[1] == grid_w)
        return jnp.asarray(shape_ok, dtype=jnp.bool_)

    # Since action is an Equinox Module, isinstance checks are static-friendly here
    selection_valid = jnp.asarray(True, dtype=jnp.bool_)
    if isinstance(action, PointAction):
        selection_valid = _check_point(action)
    elif isinstance(action, BboxAction):
        selection_valid = _check_bbox(action)
    elif isinstance(action, MaskAction):
        selection_valid = _check_mask(action)

    return jnp.asarray(op_valid & selection_valid, dtype=jnp.bool_)


@eqx.filter_jit
def arc_step(
    state: ArcEnvState,
    action: StructuredAction,
    config: ConfigType,
) -> tuple[ArcEnvState, ObservationArray, RewardValue, EpisodeDone, StepInfo]:
    """Step with JAX-native error handling.

    Uses lax.cond to avoid raising in hot path. On invalid actions, returns a
    safe fallback with negative reward and unchanged state.
    """
    typed_config = _ensure_config(config)

    is_valid = validate_action_jax(action, state, typed_config)

    def _valid_step():
        return _arc_step_unsafe(state, action, typed_config)

    def _invalid_step():
        obs = create_observation(state, typed_config)
        info = StepInfo(
            similarity=state.similarity_score,
            similarity_improvement=jnp.asarray(0.0, dtype=jnp.float32),
            operation_type=jnp.asarray(-1, dtype=jnp.int32),
            step_count=state.step_count,
            episode_done=jnp.asarray(False),
            episode_mode=state.episode_mode,
            current_pair_index=state.current_example_idx,
            available_demo_pairs=jnp.sum(state.available_demo_pairs),
            available_test_pairs=jnp.sum(state.available_test_pairs),
            action_history_length=state.get_action_history_length(),
            success=jnp.asarray(False),
        )
        return (
            state,
            obs,
            jnp.asarray(-1.0, dtype=jnp.float32),
            jnp.asarray(False),
            info,
        )

    return jax.lax.cond(is_valid, _valid_step, _invalid_step)


# =========================================================================
# Batch Processing Functions (Task 4.3 - Filtered Transformations with vmap)
# =========================================================================


@eqx.filter_jit
def batch_reset(
    keys: jnp.ndarray, config: ConfigType, task_data: JaxArcTask | None = None
 ) -> tuple[ArcEnvState, ObservationArray]:
    """Reset multiple environments in parallel using vmap.

    This function provides efficient batch processing for environment resets
    using JAX's vmap transformation over the filtered JIT compiled arc_reset function.

    Args:
        keys: Array of PRNG keys with shape (batch_size,) for parallel resets
        config: Environment configuration (JaxArcConfig or DictConfig)
        task_data: Optional task data. If None, demo tasks will be created.

    Returns:
        Tuple containing:
        - Batched ArcEnvState with batch dimension as first axis
        - Batched observations with shape (batch_size, ...)

    Examples:
        ```python
        import jax.random as jrandom

        # Create batch of PRNG keys
        batch_size = 8
        keys = jrandom.split(jrandom.PRNGKey(42), batch_size)

        # Reset environments in parallel
        states, observations = batch_reset(keys, config, task_data)

        # states.working_grid.shape[0] == batch_size
        # observations.shape[0] == batch_size
        ```

    Note:
        This function uses jax.vmap internally for efficient vectorization.
        All environments will use the same config and task_data but different
        PRNG keys for proper randomization.
    """
    # Use vmap to vectorize over batch dimension
    vectorized_reset = jax.vmap(arc_reset, in_axes=(0, None, None))
    return vectorized_reset(keys, config, task_data)


@eqx.filter_jit
def batch_step(
    states: ArcEnvState, actions: StructuredAction, config: ConfigType
 ) -> tuple[ArcEnvState, ObservationArray, jnp.ndarray, jnp.ndarray, StepInfo]:
    """Step multiple environments in parallel using vmap.

    This function provides efficient batch processing for environment steps
    using JAX's vmap transformation over the filtered JIT compiled arc_step function.

    Args:
        states: Batched ArcEnvState with batch dimension as first axis
        actions: Batched StructuredAction with batch dimension as first axis
        config: Environment configuration (JaxArcConfig or DictConfig)

    Returns:
        Tuple containing:
        - Batched new ArcEnvState with batch dimension as first axis
        - Batched observations with shape (batch_size, ...)
        - Batched rewards with shape (batch_size,)
        - Batched done flags with shape (batch_size,)
        - Batched StepInfo structures

    Examples:
        ```python
        # Create batched actions
        batch_size = 8
        batched_actions = PointAction(
            operation=jnp.array([0] * batch_size),
            row=jnp.array([3] * batch_size),
            col=jnp.array([3] * batch_size),
        )

        # Step environments in parallel
        new_states, observations, rewards, dones, infos = batch_step(
            states, batched_actions, config
        )

        # All outputs have batch_size as first dimension
        ```

    Note:
        This function uses jax.vmap internally for efficient vectorization.
        All environments will use the same config but different states and actions.
    """
    # Use vmap to vectorize over batch dimension
    vectorized_step = jax.vmap(arc_step, in_axes=(0, 0, None))
    return vectorized_step(states, actions, config)


def create_batch_episode_runner(
    config: ConfigType,
    task_data: JaxArcTask | None = None,
) -> callable:
    """Create a JIT-compiled batch episode runner function.

    This function returns a JIT-compiled function that can run complete episodes
    for multiple environments in parallel, demonstrating the full power of
    filtered transformations with batch processing.

    Args:
        config: Environment configuration
        task_data: Optional task data for all environments
        max_steps: Maximum steps per episode (uses config default if None)

    Returns:
        JIT-compiled function that takes (keys, num_steps) and returns episode results

    Examples:
        ```python
        # Create batch episode runner
        runner = create_batch_episode_runner(config, task_data)

        # Run batch episodes
        keys = jrandom.split(jrandom.PRNGKey(42), batch_size)
        final_states, episode_rewards, episode_lengths = runner(keys, 50)
        ```
    """
    _ensure_config(config)

    @eqx.filter_jit
    def run_batch_episodes(
        keys: jnp.ndarray, num_steps: int
    ) -> tuple[ArcEnvState, jnp.ndarray, jnp.ndarray]:
        """Run complete episodes for multiple environments."""

        # Initialize environments
        states, _ = batch_reset(keys, config, task_data)
        batch_size = keys.shape[0]

        # Track episode statistics
        episode_rewards = jnp.zeros(batch_size, dtype=jnp.float32)
        episode_lengths = jnp.zeros(batch_size, dtype=jnp.int32)

        def step_fn(carry, step_idx):
            states, episode_rewards, episode_lengths = carry

            # Create simple actions for demonstration (fill operation at step position)
            actions = PointAction(
                operation=jnp.zeros(batch_size, dtype=jnp.int32),  # Fill operation
                row=jnp.full(batch_size, 2 + (step_idx % 5), dtype=jnp.int32),
                col=jnp.full(batch_size, 2 + (step_idx % 5), dtype=jnp.int32),
            )

            # Step all environments
            new_states, _, rewards, dones, _ = batch_step(states, actions, config)

            # Update episode statistics
            episode_rewards += rewards
            episode_lengths = jnp.where(~dones, episode_lengths + 1, episode_lengths)

            return (new_states, episode_rewards, episode_lengths), None

        # Run episode steps
        final_carry, _ = jax.lax.scan(
            step_fn, (states, episode_rewards, episode_lengths), jnp.arange(num_steps)
        )

        final_states, final_rewards, final_lengths = final_carry
        return final_states, final_rewards, final_lengths

    return run_batch_episodes


# Utility functions for batch processing analysis

def analyze_batch_performance(
    config: ConfigType,
    task_data: JaxArcTask | None = None,
    batch_sizes: list[int] | None = None,
    num_steps: int = 10,
 ) -> dict[str, Any]:
    """Analyze batch processing performance across different batch sizes.

    This function provides comprehensive performance analysis for batch processing
    with filtered transformations, helping optimize batch sizes for different use cases.

    Args:
        config: Environment configuration
        task_data: Optional task data
        batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8, 16, 32])
        num_steps: Number of steps to run for each batch size

    Returns:
        Dictionary containing performance metrics for each batch size

    Examples:
        ```python
        # Analyze performance
        results = analyze_batch_performance(config, task_data)

        # Print results
        for batch_size, metrics in results["batch_metrics"].items():
            print(f"Batch {batch_size}: {metrics['steps_per_second']:.1f} steps/sec")
        ```
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]

    results = {"batch_metrics": {}, "optimal_batch_size": None, "peak_throughput": 0.0}

    for batch_size in batch_sizes:
        # Create batch episode runner
        runner = create_batch_episode_runner(config, task_data)

        # Generate keys
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

        # Warm up JIT compilation
        _ = runner(keys, 1)

        # Time actual execution
        start_time = time.perf_counter()
        _, rewards, lengths = runner(keys, num_steps)
        end_time = time.perf_counter()

        # Calculate metrics
        total_time = end_time - start_time
        total_steps = batch_size * num_steps
        steps_per_second = total_steps / total_time
        time_per_env = total_time / batch_size

        metrics = {
            "batch_size": batch_size,
            "total_time": total_time,
            "time_per_env": time_per_env,
            "steps_per_second": steps_per_second,
            "avg_reward": float(jnp.mean(rewards)),
            "avg_length": float(jnp.mean(lengths)),
        }

        results["batch_metrics"][batch_size] = metrics

        # Track optimal batch size
        if steps_per_second > results["peak_throughput"]:
            results["peak_throughput"] = steps_per_second
            results["optimal_batch_size"] = batch_size

    return results


# PRNG Key Management Utilities for Batch Processing

def create_batch_keys(key: PRNGKey, batch_size: int) -> jnp.ndarray:
    """Create array of PRNG keys for batch processing.

    This utility function splits a single PRNG key into multiple keys
    for parallel batch processing, ensuring deterministic behavior
    across batch elements.

    Args:
        key: Base PRNG key to split
        batch_size: Number of keys to generate

    Returns:
        Array of PRNG keys with shape (batch_size, 2)

    Examples:
        ```python
        # Create keys for batch processing
        base_key = jax.random.PRNGKey(42)
        batch_keys = create_batch_keys(base_key, 8)

        # Use with batch_reset
        states, obs = batch_reset(batch_keys, config, task_data)
        ```
    """
    return jax.random.split(key, batch_size)

def split_key_for_batch_step(key: PRNGKey, batch_size: int) -> jnp.ndarray:
    """Split PRNG key for batch step operations.

    This function provides deterministic key splitting for batch step
    operations, ensuring reproducible behavior when stepping multiple
    environments in parallel.

    Args:
        key: PRNG key to split for batch operations
        batch_size: Number of environments in the batch

    Returns:
        Array of PRNG keys for batch step operations

    Examples:
        ```python
        # Split key for batch stepping
        step_key = jax.random.PRNGKey(123)
        batch_keys = split_key_for_batch_step(step_key, 16)

        # Keys can be used for any random operations during stepping
        # (though current step function doesn't use keys directly)
        ```
    """
    return jax.random.split(key, batch_size)


def validate_batch_keys(keys: jnp.ndarray, expected_batch_size: int) -> bool:
    """Validate that PRNG keys array has correct shape for batch processing.

    This utility function validates that the provided keys array has the
    correct shape and dtype for batch processing operations.

    Args:
        keys: Array of PRNG keys to validate
        expected_batch_size: Expected batch size

    Returns:
        True if keys are valid for batch processing

    Examples:
        ```python
        # Validate keys before batch processing
        keys = create_batch_keys(jax.random.PRNGKey(42), 8)
        is_valid = validate_batch_keys(keys, 8)  # Returns True

        # Invalid keys
        bad_keys = jnp.array([1, 2, 3])
        is_valid = validate_batch_keys(bad_keys, 8)  # Returns False
        ```
    """
    if not isinstance(keys, jnp.ndarray):
        return False

    # Check shape: should be (batch_size, 2) for JAX PRNG keys
    if len(keys.shape) != 2 or keys.shape[1] != 2:
        return False

    # Check batch size matches
    if keys.shape[0] != expected_batch_size:
        return False

    # Check dtype (JAX PRNG keys are uint32)
    return keys.dtype == jnp.uint32


def ensure_deterministic_batch_keys(
    base_key: PRNGKey, batch_size: int, step_count: int = 0
) -> jnp.ndarray:
    """Ensure deterministic PRNG key generation for reproducible batch processing.

    This function creates deterministic PRNG keys for batch processing by
    incorporating step count and batch size into the key generation process,
    ensuring reproducible behavior across training runs.

    Args:
        base_key: Base PRNG key for deterministic generation
        batch_size: Number of environments in the batch
        step_count: Current step count for temporal determinism

    Returns:
        Array of deterministic PRNG keys for batch processing

    Examples:
        ```python
        # Create deterministic keys for training
        base_key = jax.random.PRNGKey(42)
        keys = ensure_deterministic_batch_keys(base_key, 8, step_count=100)

        # Same inputs always produce same keys
        keys2 = ensure_deterministic_batch_keys(base_key, 8, step_count=100)
        assert jnp.array_equal(keys, keys2)  # True
        ```
    """
    # Create deterministic seed by combining base key with step count
    step_key = jax.random.fold_in(base_key, step_count)

    # Split into batch keys
    return jax.random.split(step_key, batch_size)


def prng_key_splitting_demo(batch_sizes: list[int] | None = None) -> dict[str, Any]:
    """Test PRNG key splitting with different batch sizes.

    This function tests the PRNG key management utilities with various
    batch sizes to ensure proper functionality and deterministic behavior.

    Args:
        batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8, 16, 32, 64])

    Returns:
        Dictionary containing test results and validation metrics

    Examples:
        ```python
        # Test PRNG key splitting
        results = test_prng_key_splitting()

        # Check if all tests passed
        all_passed = all(
            results["batch_results"][bs]["valid"] for bs in results["batch_results"]
        )
        ```
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    base_key = jax.random.PRNGKey(42)
    results = {"batch_results": {}, "determinism_test": True, "validation_test": True}

    for batch_size in batch_sizes:
        # Test key creation
        keys = create_batch_keys(base_key, batch_size)

        # Test validation
        is_valid = validate_batch_keys(keys, batch_size)

        # Test deterministic generation
        keys2 = ensure_deterministic_batch_keys(base_key, batch_size, step_count=0)
        keys3 = ensure_deterministic_batch_keys(base_key, batch_size, step_count=0)
        is_deterministic = jnp.array_equal(keys2, keys3)

        # Test uniqueness within batch
        unique_keys = jnp.unique(keys.reshape(-1, 2), axis=0)
        is_unique = unique_keys.shape[0] == batch_size

        results["batch_results"][batch_size] = {
            "valid": is_valid,
            "deterministic": is_deterministic,
            "unique": is_unique,
            "shape": keys.shape,
            "dtype": str(keys.dtype),
        }

        # Update overall results
        if not is_valid:
            results["validation_test"] = False
        if not is_deterministic:
            results["determinism_test"] = False

    return results
