"""
Functional API for JaxARC environments with Hydra integration.

This module provides pure functional implementations of the ARC environment
that work with both typed configs and Hydra DictConfig objects.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig

from jaxarc.utils.visualization import (
    _clear_output_directory,
    jax_save_step_visualization,
)

from ..state import ArcEnvState
from ..types import JaxArcTask
from ..utils.jax_types import (
    EpisodeDone,
    ObservationArray,
    OperationId,
    PRNGKey,
    get_action_record_fields,
    RewardValue,
)
from .config import JaxArcConfig
from .grid_operations import compute_grid_similarity, execute_grid_operation
from .structured_actions import StructuredAction, PointAction, BboxAction, MaskAction

# Type aliases for cleaner signatures
ConfigType = Union[JaxArcConfig, DictConfig]


def _ensure_config(config: ConfigType) -> JaxArcConfig:
    """Convert config to typed JaxArcConfig if needed."""
    if isinstance(config, DictConfig):
        return JaxArcConfig.from_hydra(config)
    return config


def _validate_operation(operation: Any, config: JaxArcConfig) -> OperationId:
    """Validate and normalize operation value with enhanced range (0-41)."""
    if isinstance(operation, (int, jnp.integer)):
        operation = jnp.array(operation, dtype=jnp.int32)
    elif not isinstance(operation, jnp.ndarray):
        raise ValueError(f"Operation must be int or jnp.ndarray, got {type(operation)}")

    # Enhanced operation range validation (0-41 for control operations)
    max_operations = 42  # Updated to include enhanced control operations (0-41)

    # Validate operation range (JAX-compatible)
    if config.action.validate_actions and not config.action.allow_invalid_actions:
        operation = jnp.clip(operation, 0, max_operations - 1)

    return operation


def _get_observation(state: ArcEnvState, config: JaxArcConfig) -> ObservationArray:
    """Extract observation from state."""
    # For now, just return the working grid
    # Future: Could include additional channels for selection, target, etc.
    return state.working_grid


def create_observation(state: ArcEnvState, config: JaxArcConfig) -> ObservationArray:
    """Create agent observation from environment state.

    This function extracts relevant information from the full environment state
    and constructs a focused observation for the agent, hiding internal
    implementation details and providing configurable observation formats.

    The observation provides agents with:
    - Core grid information (working grid and mask)
    - Episode context (mode, current pair, step count)
    - Progress tracking (completion status for pairs)
    - Action space information (allowed operations)
    - Target information (only in training mode)

    Args:
        state: Current environment state with enhanced functionality
        config: Environment configuration

    Returns:
        Observation array for the agent (currently working grid, future: structured observation)

    Examples:
        ```python
        # Create observation for agent
        observation = create_observation(state, config)

        # In training mode, observation includes target information
        train_obs = create_observation(train_state, config)

        # In test mode, target information is masked
        test_obs = create_observation(test_state, config)
        ```
    """
    # Current implementation: return working grid as observation
    # This maintains backward compatibility while the enhanced observation system
    # is being developed. Future versions will return structured ArcObservation
    # with configurable fields based on ObservationConfig.

    # The structured observation will include:
    # - working_grid: Current grid being modified
    # - working_grid_mask: Valid cells mask
    # - episode_mode: Training (0) or test (1) mode
    # - current_pair_idx: Which pair is currently active
    # - step_count: Number of steps taken
    # - demo_completion_status: Which demo pairs are solved
    # - test_completion_status: Which test pairs are solved
    # - allowed_operations_mask: Currently allowed operations
    # - target_grid: Only available in train mode (masked in test mode)
    # - recent_actions: Optional recent action history

    return _get_observation(state, config)


def _calculate_reward(
    old_state: ArcEnvState, new_state: ArcEnvState, config: JaxArcConfig
) -> RewardValue:
    """Calculate reward based on state transition and config."""
    reward_cfg = config.reward

    # Reward for similarity improvement
    similarity_improvement = new_state.similarity_score - old_state.similarity_score
    similarity_reward = reward_cfg.similarity_weight * similarity_improvement

    # Progress bonus (if enabled)
    progress_reward = jnp.where(
        similarity_improvement > 0, reward_cfg.progress_bonus, 0.0
    )

    # Step penalty
    step_penalty = jnp.array(reward_cfg.step_penalty, dtype=jnp.float32)

    # Success bonus
    success_bonus = jnp.where(
        new_state.similarity_score >= 1.0, reward_cfg.success_bonus, 0.0
    )

    # Calculate total reward
    total_reward = similarity_reward + progress_reward + step_penalty + success_bonus

    # Apply reward-on-submit-only logic
    if reward_cfg.reward_on_submit_only:
        # Only give full reward if episode is done (submit operation)
        submit_only_reward = step_penalty + success_bonus
        reward = jnp.where(new_state.episode_done, total_reward, submit_only_reward)
    else:
        reward = total_reward

    return reward


def _calculate_enhanced_reward(
    old_state: ArcEnvState,
    new_state: ArcEnvState,
    config: JaxArcConfig,
    is_control_operation: bool,
) -> RewardValue:
    """Calculate reward with mode-specific logic and control operation handling.

    This enhanced reward function provides:
    - Training mode reward calculation with configurable frequency
    - Evaluation mode reward calculation with target masking
    - Proper similarity scoring for different pair types
    - Different reward structures based on configuration
    - Control operation penalty/reward handling
    - JIT-compilable and efficient implementation

    Args:
        old_state: Previous environment state
        new_state: New environment state after action
        config: Environment configuration
        is_control_operation: Whether the action was a control operation

    Returns:
        JAX scalar array containing the calculated reward
    """
    reward_cfg = config.reward

    # Get episode configuration for mode-specific logic
    episode_mode = new_state.episode_mode
    is_training = episode_mode == 0
    is_test = episode_mode == 1

    # Get episode configuration from config or use defaults
    episode_config = getattr(config, "episode", None)
    if episode_config is not None:
        training_reward_frequency = episode_config.training_reward_frequency
        evaluation_reward_frequency = episode_config.evaluation_reward_frequency
    else:
        # Fallback to defaults if episode config not available
        training_reward_frequency = "step"
        evaluation_reward_frequency = "submit"

    # Calculate similarity improvement for proper scoring
    similarity_improvement = new_state.similarity_score - old_state.similarity_score

    # Training mode reward calculation with configurable frequency
    training_reward = jax.lax.cond(
        training_reward_frequency == "submit",
        lambda: _calculate_training_submit_reward(
            old_state, new_state, reward_cfg, similarity_improvement
        ),
        lambda: _calculate_training_step_reward(
            old_state, new_state, reward_cfg, similarity_improvement
        ),
    )

    # Evaluation mode reward calculation with target masking
    evaluation_reward = jax.lax.cond(
        evaluation_reward_frequency == "submit",
        lambda: _calculate_evaluation_submit_reward(old_state, new_state, reward_cfg),
        lambda: _calculate_evaluation_step_reward(old_state, new_state, reward_cfg),
    )

    # Mode-specific reward selection using JAX-compatible conditional
    mode_reward = jax.lax.cond(
        is_training, lambda: training_reward, lambda: evaluation_reward
    )

    # Control operation adjustments with proper JAX operations
    control_adjustment = jax.lax.cond(
        is_control_operation,
        lambda: _calculate_control_operation_reward(new_state, reward_cfg),
        lambda: jnp.array(0.0, dtype=jnp.float32),
    )

    # Final reward calculation
    final_reward = mode_reward + control_adjustment

    return final_reward


def _calculate_training_submit_reward(
    old_state: ArcEnvState,
    new_state: ArcEnvState,
    reward_cfg,
    similarity_improvement: RewardValue,
) -> RewardValue:
    """Calculate training mode reward for submit-only frequency.

    In submit-only mode, full rewards are only given when the episode ends
    (submit operation), otherwise only step penalty is applied.
    """
    # Full reward calculation for submit operations with enhanced bonuses
    similarity_reward = reward_cfg.training_similarity_weight * similarity_improvement
    progress_bonus = jnp.where(
        similarity_improvement > 0, reward_cfg.progress_bonus, 0.0
    )
    step_penalty = jnp.array(reward_cfg.step_penalty, dtype=jnp.float32)

    # Enhanced success bonus calculation
    is_solved = new_state.similarity_score >= 1.0
    base_success_bonus = jnp.where(is_solved, reward_cfg.success_bonus, 0.0)
    demo_bonus = jnp.where(is_solved, reward_cfg.demo_completion_bonus, 0.0)

    # Efficiency bonus for solving within threshold
    efficiency_bonus = jnp.where(
        jnp.logical_and(
            is_solved, new_state.step_count <= reward_cfg.efficiency_bonus_threshold
        ),
        reward_cfg.efficiency_bonus,
        0.0,
    )

    full_reward = (
        similarity_reward
        + progress_bonus
        + step_penalty
        + base_success_bonus
        + demo_bonus
        + efficiency_bonus
    )
    submit_only_reward = step_penalty  # Only step penalty between submits

    # Return full reward if episode is done (submit), otherwise just step penalty
    return jnp.where(new_state.episode_done, full_reward, submit_only_reward)


def _calculate_training_step_reward(
    old_state: ArcEnvState,
    new_state: ArcEnvState,
    reward_cfg,
    similarity_improvement: RewardValue,
) -> RewardValue:
    """Calculate training mode reward for step-by-step frequency.

    In step mode, rewards are calculated and given on every step with
    enhanced pair-type specific bonuses and efficiency considerations.
    """
    # Use training-specific similarity weight
    similarity_reward = reward_cfg.training_similarity_weight * similarity_improvement
    progress_bonus = jnp.where(
        similarity_improvement > 0, reward_cfg.progress_bonus, 0.0
    )
    step_penalty = jnp.array(reward_cfg.step_penalty, dtype=jnp.float32)

    # Enhanced success bonus with pair-type specific bonuses
    is_solved = new_state.similarity_score >= 1.0
    base_success_bonus = jnp.where(is_solved, reward_cfg.success_bonus, 0.0)

    # Add demo completion bonus for demonstration pairs
    demo_bonus = jnp.where(is_solved, reward_cfg.demo_completion_bonus, 0.0)

    # Add efficiency bonus if solved within threshold
    efficiency_bonus = jnp.where(
        jnp.logical_and(
            is_solved, new_state.step_count <= reward_cfg.efficiency_bonus_threshold
        ),
        reward_cfg.efficiency_bonus,
        0.0,
    )

    return (
        similarity_reward
        + progress_bonus
        + step_penalty
        + base_success_bonus
        + demo_bonus
        + efficiency_bonus
    )


def _calculate_evaluation_submit_reward(
    old_state: ArcEnvState, new_state: ArcEnvState, reward_cfg
) -> RewardValue:
    """Calculate evaluation mode reward for submit-only frequency.

    In evaluation mode with submit-only frequency, agents receive step penalties
    and enhanced success bonuses for test pairs (no similarity rewards due to target masking).
    """
    step_penalty = jnp.array(reward_cfg.step_penalty, dtype=jnp.float32)

    # Enhanced success bonus for test pairs
    is_solved = new_state.similarity_score >= 1.0
    base_success_bonus = jnp.where(is_solved, reward_cfg.success_bonus, 0.0)
    test_bonus = jnp.where(is_solved, reward_cfg.test_completion_bonus, 0.0)

    # Efficiency bonus for solving test pairs quickly
    efficiency_bonus = jnp.where(
        jnp.logical_and(
            is_solved, new_state.step_count <= reward_cfg.efficiency_bonus_threshold
        ),
        reward_cfg.efficiency_bonus,
        0.0,
    )

    submit_reward = step_penalty + base_success_bonus + test_bonus + efficiency_bonus

    # Return submit reward if episode is done, otherwise just step penalty
    return jnp.where(new_state.episode_done, submit_reward, step_penalty)


def _calculate_evaluation_step_reward(
    old_state: ArcEnvState, new_state: ArcEnvState, reward_cfg
) -> RewardValue:
    """Calculate evaluation mode reward for step-by-step frequency.

    In evaluation mode with step frequency, agents receive step penalties
    and enhanced success bonuses on every step (no similarity rewards due to target masking).
    """
    step_penalty = jnp.array(reward_cfg.step_penalty, dtype=jnp.float32)

    # Enhanced success bonus for test pairs
    is_solved = new_state.similarity_score >= 1.0
    base_success_bonus = jnp.where(is_solved, reward_cfg.success_bonus, 0.0)
    test_bonus = jnp.where(is_solved, reward_cfg.test_completion_bonus, 0.0)

    # Efficiency bonus for solving test pairs quickly
    efficiency_bonus = jnp.where(
        jnp.logical_and(
            is_solved, new_state.step_count <= reward_cfg.efficiency_bonus_threshold
        ),
        reward_cfg.efficiency_bonus,
        0.0,
    )

    return step_penalty + base_success_bonus + test_bonus + efficiency_bonus


def _calculate_control_operation_reward(
    new_state: ArcEnvState, reward_cfg
) -> RewardValue:
    """Calculate reward adjustment for control operations.

    Control operations (pair switching, reset) receive context-aware rewards:
    - Small penalty for basic control operations to encourage efficiency
    - Bonus for beneficial pair switching (to unsolved pairs)
    - Neutral reward for necessary operations (reset)
    """
    # Base control operation penalty
    base_penalty = jnp.array(reward_cfg.control_operation_penalty, dtype=jnp.float32)

    # Pair switching bonus for beneficial operations
    # This is a simplified implementation - could be enhanced with operation-specific logic
    switching_bonus = jnp.array(reward_cfg.pair_switching_bonus, dtype=jnp.float32)

    # For now, apply base penalty with potential for switching bonus
    # Future enhancement: analyze specific control operation and context
    # to determine if it was beneficial (e.g., switching to unsolved pair)

    return base_penalty + switching_bonus


def _calculate_similarity_score_for_pair_type(
    working_grid: Array, target_grid: Array, pair_type: str, episode_mode: int
) -> RewardValue:
    """Calculate proper similarity scoring for different pair types.

    This function provides specialized similarity calculation based on:
    - Pair type (demonstration vs test)
    - Episode mode (training vs evaluation)
    - Target masking in evaluation mode

    Args:
        working_grid: Current working grid
        target_grid: Target grid (may be masked in test mode)
        pair_type: Type of pair ("demo" or "test")
        episode_mode: Episode mode (0=train, 1=test)

    Returns:
        Similarity score appropriate for the pair type and mode
    """
    from .grid_operations import compute_grid_similarity

    # In training mode, use full similarity calculation
    training_similarity = compute_grid_similarity(working_grid, target_grid)

    # In test mode, similarity is limited due to target masking
    # We can only measure structural consistency, not correctness
    test_similarity = jnp.array(0.0, dtype=jnp.float32)  # Masked similarity

    # Return appropriate similarity based on episode mode
    return jnp.where(episode_mode == 0, training_similarity, test_similarity)


def _is_episode_done(state: ArcEnvState, config: JaxArcConfig) -> EpisodeDone:
    """Check if episode should terminate."""
    # Episode ends if:
    # 1. Task is solved (perfect similarity)
    # 2. Maximum steps reached
    # 3. Submit operation was used (sets episode_done=True)

    task_solved = state.similarity_score >= 1.0
    max_steps_reached = state.step_count >= config.environment.max_episode_steps
    submitted = state.episode_done  # Set by submit operation

    return task_solved | max_steps_reached | submitted


def _is_episode_done_enhanced(state: ArcEnvState, config: JaxArcConfig) -> EpisodeDone:
    """Check if episode should terminate with enhanced multi-pair logic.

    This enhanced termination function provides:
    - Multi-pair episode management
    - Mode-specific termination criteria
    - Configurable continuation policies
    - Pair completion tracking

    Args:
        state: Current environment state
        config: Environment configuration

    Returns:
        JAX boolean scalar indicating if episode should terminate
    """
    # Basic termination conditions
    basic_done = _is_episode_done(state, config)

    # For now, use basic termination logic to maintain JAX compatibility
    # Enhanced termination logic with episode manager would require
    # more complex JAX-compatible implementation

    # Future enhancement: implement JAX-compatible episode manager logic
    # using jax.lax.cond and pure functions instead of try/except and imports

    return basic_done


def _create_demo_task(config: JaxArcConfig) -> JaxArcTask:
    """Create a simple demo task for testing when no task sampler is available."""
    from ..types import JaxArcTask

    logger.warning(
        "No task sampler provided - creating demo task. "
        "For real training, provide a task_sampler in config or task_data directly."
    )

    # Create simple demo grids using dataset-appropriate size
    # Use full configured grid size for proper testing, with reasonable upper bounds
    demo_height = min(30, config.dataset.max_grid_height)
    demo_width = min(30, config.dataset.max_grid_width)
    grid_shape = (demo_height, demo_width)

    # Use dataset background color
    bg_color = config.dataset.background_color
    input_grid = jnp.full(grid_shape, bg_color, dtype=jnp.int32)

    # Add a small pattern (ensure it fits and uses valid colors)
    pattern_size = min(3, demo_height - 2, demo_width - 2)
    if pattern_size > 0:
        start_row = (demo_height - pattern_size) // 2
        start_col = (demo_width - pattern_size) // 2

        # Use color 1 if available, otherwise background + 1
        pattern_color = (
            1
            if config.dataset.max_colors > 1
            else (bg_color + 1) % config.dataset.max_colors
        )
        input_grid = input_grid.at[
            start_row : start_row + pattern_size, start_col : start_col + pattern_size
        ].set(pattern_color)

    # Target: change pattern to different color
    target_grid = input_grid.copy()
    if pattern_size > 0:
        target_color = (
            2
            if config.dataset.max_colors > 2
            else (pattern_color + 1) % config.dataset.max_colors
        )
        target_grid = target_grid.at[
            start_row : start_row + pattern_size, start_col : start_col + pattern_size
        ].set(target_color)

    # Create masks
    mask = jnp.ones(grid_shape, dtype=jnp.bool_)

    # Pad to max size
    max_shape = (config.dataset.max_grid_height, config.dataset.max_grid_width)
    padded_input = jnp.full(max_shape, -1, dtype=jnp.int32)
    padded_target = jnp.full(max_shape, -1, dtype=jnp.int32)
    padded_mask = jnp.zeros(max_shape, dtype=jnp.bool_)

    padded_input = padded_input.at[: grid_shape[0], : grid_shape[1]].set(input_grid)
    padded_target = padded_target.at[: grid_shape[0], : grid_shape[1]].set(target_grid)
    padded_mask = padded_mask.at[: grid_shape[0], : grid_shape[1]].set(mask)

    return JaxArcTask(
        input_grids_examples=jnp.expand_dims(padded_input, 0),
        output_grids_examples=jnp.expand_dims(padded_target, 0),
        input_masks_examples=jnp.expand_dims(padded_mask, 0),
        output_masks_examples=jnp.expand_dims(padded_mask, 0),
        num_train_pairs=1,
        test_input_grids=jnp.expand_dims(padded_input, 0),
        test_input_masks=jnp.expand_dims(padded_mask, 0),
        true_test_output_grids=jnp.expand_dims(padded_target, 0),
        true_test_output_masks=jnp.expand_dims(padded_mask, 0),
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )


def _get_or_create_task_data(
    task_data: JaxArcTask | None, config: JaxArcConfig
) -> JaxArcTask:
    """Get task data or create demo task - focused helper function.
    
    This helper function ensures that valid task data is available for environment
    initialization. If no task data is provided, it creates a simple demo task
    suitable for testing and development purposes.
    
    Args:
        task_data: Optional JaxArcTask data. If None, a demo task will be created.
        config: Environment configuration used for demo task creation parameters
               including grid dimensions, colors, and dataset settings.
    
    Returns:
        JaxArcTask: Valid task data ready for environment initialization.
                   Either the provided task_data or a newly created demo task.
    
    Examples:
        ```python
        # With existing task data
        task = _get_or_create_task_data(existing_task, config)
        
        # Without task data (creates demo)
        demo_task = _get_or_create_task_data(None, config)
        ```
    
    Note:
        Demo tasks are created with simple patterns suitable for testing.
        For production training, always provide real task data from parsers.
    """
    if task_data is None:
        # Create demo task as fallback
        # Parsers are handled separately in the new system
        task_data = _create_demo_task(config)
        if config.logging.log_operations:
            logger.info(f"Created demo task for dataset {config.dataset.dataset_name}")
    return task_data


def _select_initial_pair(
    key: PRNGKey,
    task_data: JaxArcTask,
    episode_mode: int,
    initial_pair_idx: int | None,
) -> Tuple[jnp.ndarray, bool]:
    """Select initial pair based on mode and configuration - focused helper function.
    
    This helper function handles the selection of which task pair to use for episode
    initialization. It supports both explicit pair specification and automatic
    selection based on episode mode and configuration strategy.
    
    Args:
        key: JAX PRNG key for random pair selection when needed
        task_data: JaxArcTask containing available demonstration and test pairs
        episode_mode: Episode mode (0=train, 1=test) determining pair type selection
        initial_pair_idx: Optional explicit pair index. If None, uses episode manager
                         selection strategy based on configuration.
    
    Returns:
        Tuple containing:
        - selected_pair_idx: JAX array with selected pair index (int32)
        - selection_successful: Boolean indicating if selection was valid
    
    Examples:
        ```python
        # Explicit pair selection
        pair_idx, success = _select_initial_pair(key, task, 0, 2)
        
        # Automatic selection based on mode
        pair_idx, success = _select_initial_pair(key, task, 1, None)
        ```
    
    Note:
        Uses episode manager for intelligent pair selection strategies.
        Falls back to index 0 if selection fails for any reason.
    """
    # Import episode manager for pair selection
    from .episode_manager import (
        ArcEpisodeConfig,
        ArcEpisodeManager,
        EPISODE_MODE_TRAIN,
        EPISODE_MODE_TEST,
    )

    # JAX-compliant integer-only episode mode validation
    if episode_mode not in [EPISODE_MODE_TRAIN, EPISODE_MODE_TEST]:
        raise ValueError(
            f"episode_mode must be {EPISODE_MODE_TRAIN} (train) or {EPISODE_MODE_TEST} (test), got '{episode_mode}'"
        )

    # Use integer episode mode directly for JAX compatibility
    episode_config = ArcEpisodeConfig(episode_mode=episode_mode)

    # Select initial pair based on mode and configuration strategy
    if initial_pair_idx is not None:
        # Use explicit pair index if provided
        selected_pair_idx = jnp.array(initial_pair_idx, dtype=jnp.int32)

        # Validate the explicit pair index using JAX-compatible conditional logic
        selection_successful = jax.lax.cond(
            episode_mode == EPISODE_MODE_TRAIN,
            lambda: task_data.is_demo_pair_available(initial_pair_idx),
            lambda: task_data.is_test_pair_available(initial_pair_idx),
        )
    else:
        # Use episode manager for pair selection based on configuration strategy
        selected_pair_idx, selection_successful = ArcEpisodeManager.select_initial_pair(
            key, task_data, episode_config
        )

    # Handle selection failure using JAX-compatible operations
    fallback_idx = jnp.array(0, dtype=jnp.int32)
    selected_pair_idx = jnp.where(selection_successful, selected_pair_idx, fallback_idx)

    return selected_pair_idx, selection_successful


def _initialize_grids(
    task_data: JaxArcTask,
    selected_pair_idx: jnp.ndarray,
    episode_mode: int,
    config: JaxArcConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize grids based on episode mode - focused helper function.
    
    This helper function sets up the initial, target, and mask grids based on the
    episode mode and selected pair. It handles the critical difference between
    training mode (with target access) and test mode (with target masking).
    
    Args:
        task_data: JaxArcTask containing demonstration and test pair data
        selected_pair_idx: JAX array with the index of the selected pair
        episode_mode: Episode mode (0=train, 1=test) determining grid initialization
        config: Environment configuration containing dataset settings like background_color
    
    Returns:
        Tuple containing:
        - initial_grid: Starting grid for the episode (JAX array)
        - target_grid: Target grid (visible in train mode, masked in test mode)
        - initial_mask: Boolean mask indicating valid grid cells
    
    Examples:
        ```python
        # Training mode initialization
        init_grid, target, mask = _initialize_grids(task, idx, 0, config)
        
        # Test mode initialization (target masked)
        init_grid, masked_target, mask = _initialize_grids(task, idx, 1, config)
        ```
    
    Note:
        In test mode, target grids are filled with background color to prevent
        cheating while maintaining proper evaluation conditions.
    """
    from .episode_manager import EPISODE_MODE_TRAIN

    # JAX-compliant mode-specific grid initialization using conditional logic
    def get_train_grids():
        # Training mode: use demonstration pair with target access
        initial_grid = task_data.input_grids_examples[selected_pair_idx]
        target_grid = task_data.output_grids_examples[selected_pair_idx]
        initial_mask = task_data.input_masks_examples[selected_pair_idx]
        return initial_grid, target_grid, initial_mask

    def get_test_grids():
        # Test mode: use test pair with target masking for proper evaluation
        initial_grid = task_data.test_input_grids[selected_pair_idx]
        initial_mask = task_data.test_input_masks[selected_pair_idx]
        # In test mode, target grid is masked (set to background color) to prevent cheating
        background_color = config.dataset.background_color
        target_grid = jnp.full_like(
            initial_grid, background_color
        )  # Masked target for evaluation
        return initial_grid, target_grid, initial_mask

    # Use JAX conditional to select grids based on episode mode
    initial_grid, target_grid, initial_mask = jax.lax.cond(
        episode_mode == EPISODE_MODE_TRAIN, get_train_grids, get_test_grids
    )

    return initial_grid, target_grid, initial_mask


def _create_initial_state(
    task_data: JaxArcTask,
    initial_grid: jnp.ndarray,
    target_grid: jnp.ndarray,
    initial_mask: jnp.ndarray,
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
        state = _create_initial_state(task, grid, target, mask, idx, 0, config)
        
        # Create initial state for testing
        state = _create_initial_state(task, grid, masked_target, mask, idx, 1, config)
        ```
    
    Note:
        Initializes enhanced features like action history storage, completion
        status tracking, and dynamic operation control for full functionality.
    """
    # Calculate initial similarity (will be 0.0 in test mode due to masked target)
    initial_similarity = compute_grid_similarity(initial_grid, target_grid)

    # Initialize grids based on episode mode using JAX-compatible operations
    # Get available pairs and completion status for enhanced functionality
    available_demo_pairs = task_data.get_available_demo_pairs()
    available_test_pairs = task_data.get_available_test_pairs()
    demo_completion_status = jnp.zeros_like(available_demo_pairs)
    test_completion_status = jnp.zeros_like(available_test_pairs)

    # Initialize action history with dynamic sizing based on configuration
    max_history_length = getattr(config, "max_history_length", 1000)
    num_operations = 42  # Updated to include enhanced control operations (0-41)

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
    state = ArcEnvState(
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

    return state


@eqx.filter_jit
def arc_reset(
    key: PRNGKey,
    config: ConfigType,
    task_data: JaxArcTask | None = None,
    episode_mode: int = 0,  # 0=train, 1=test (JAX-compatible integers)
    initial_pair_idx: int | None = None,
) -> Tuple[ArcEnvState, ObservationArray]:
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
        from jaxarc.envs.config import JaxArcConfig
        typed_config = JaxArcConfig.from_hydra(hydra_config)
        state, obs = arc_reset(key, typed_config, task_data)
        ```

    Note:
        Function has been decomposed into helper functions (_get_or_create_task_data,
        _select_initial_pair, _initialize_grids, _create_initial_state) for better
        maintainability while preserving performance and JAX compatibility.
    """
    # Ensure we have a typed config
    typed_config = _ensure_config(config)

    # Get or create task data
    task_data = _get_or_create_task_data(task_data, typed_config)

    # Select initial pair based on mode and configuration strategy
    selected_pair_idx, selection_successful = _select_initial_pair(
        key, task_data, episode_mode, initial_pair_idx
    )

    # Initialize grids based on episode mode using JAX-compatible operations
    initial_grid, target_grid, initial_mask = _initialize_grids(
        task_data, selected_pair_idx, episode_mode, typed_config
    )

    # Create enhanced initial state with all new fields
    state = _create_initial_state(
        task_data,
        initial_grid,
        target_grid,
        initial_mask,
        selected_pair_idx,
        episode_mode,
        typed_config,
    )

    # Create initial observation using the enhanced create_observation function
    observation = create_observation(state, typed_config)

    # Optional logging with enhanced information
    if typed_config.logging.log_operations:
        jax.debug.callback(
            lambda mode, idx, sim, avail_demo, avail_test: logger.info(
                f"Reset ARC environment in {['train', 'test'][int(mode)]} mode, "
                f"pair {int(idx)}, similarity: {float(sim):.3f}, "
                f"available demos: {int(avail_demo)}, available tests: {int(avail_test)}"
            ),
            episode_mode,
            selected_pair_idx,
            state.similarity_score,
            jnp.sum(state.available_demo_pairs),
            jnp.sum(state.available_test_pairs),
        )

    return state, observation


def _process_action(
    state: ArcEnvState,
    action: Union[StructuredAction, Dict[str, Any]],
    config: JaxArcConfig,
) -> Tuple[ArcEnvState, StructuredAction, bool]:
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
        - is_control_operation: Boolean indicating if this was a control operation
    
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
    # Import episode manager for control operations
    from .episode_manager import ArcEpisodeConfig, ArcEpisodeManager
    from .structured_actions import PointAction, BboxAction, MaskAction, create_mask_action

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
            raise ValueError(f"Unsupported selection format: shape {selection.shape}")
    else:
        # Handle structured action
        validated_action = action.validate(grid_shape, max_operations=42)

    # Extract operation from validated action
    operation = validated_action.operation

    # Apply dynamic action space validation and filtering if enabled
    if (
        hasattr(config.action, "dynamic_action_filtering")
        and config.action.dynamic_action_filtering
    ):
        from .action_space_controller import ActionSpaceController

        controller = ActionSpaceController()

        # Filter invalid operation according to policy
        operation = controller.filter_invalid_operation_jax(
            operation, state, config.action
        )
        
        # Update validated action with filtered operation
        if isinstance(validated_action, PointAction):
            validated_action = PointAction(
                operation=operation,
                row=validated_action.row,
                col=validated_action.col
            )
        elif isinstance(validated_action, BboxAction):
            validated_action = BboxAction(
                operation=operation,
                r1=validated_action.r1,
                c1=validated_action.c1,
                r2=validated_action.r2,
                c2=validated_action.c2
            )
        elif isinstance(validated_action, MaskAction):
            validated_action = MaskAction(
                operation=operation,
                selection=validated_action.selection
            )

    # Check if this is a control operation (35-41) or grid operation (0-34)
    is_control_operation = operation >= 35

    # Define functions for control and grid operations
    def handle_control_operation(state, action, operation, config):
        """Handle control operations (35-41) using episode manager."""
        # Create episode configuration from main config using integer episode mode
        episode_mode_int = jax.lax.cond(
            state.is_training_mode(),
            lambda: 0,  # EPISODE_MODE_TRAIN = 0 = train mode
            lambda: 1,  # EPISODE_MODE_TEST = 1 = test mode
        )

        episode_config = ArcEpisodeConfig(
            episode_mode=episode_mode_int,
            demo_selection_strategy=getattr(
                config.episode, "demo_selection_strategy", "random"
            ),
            allow_demo_switching=getattr(config.episode, "allow_demo_switching", True),
            allow_test_switching=getattr(config.episode, "allow_test_switching", False),
        )

        # Execute pair control operation using episode manager
        new_state = ArcEpisodeManager.execute_pair_control_operation(
            state, operation, episode_config
        )

        # For control operations, we need to update grids if pair was switched
        is_pair_switching_op = jnp.isin(operation, jnp.array([35, 36, 37, 38, 40, 41]))

        def update_grids_for_pair_switch(state):
            # Update working grid and target based on new pair
            def get_train_grids(state):
                input_grid, target_grid, input_mask = (
                    state.task_data.get_demo_pair_data(state.current_example_idx)
                )
                return input_grid, target_grid, input_mask

            def get_test_grids(state):
                input_grid, input_mask = state.task_data.get_test_pair_data(
                    state.current_example_idx
                )
                # In test mode, mask target grid
                background_color = getattr(config.dataset, "background_color", 0)
                target_grid = jnp.full_like(input_grid, background_color)
                return input_grid, target_grid, input_mask

            # Use JAX conditional to get grids based on mode
            input_grid, target_grid, input_mask = jax.lax.cond(
                state.is_training_mode(), get_train_grids, get_test_grids, state
            )

            # Update state with new grids
            updated_state = eqx.tree_at(
                lambda s: (
                    s.working_grid,
                    s.working_grid_mask,
                    s.target_grid,
                    s.selected,
                    s.clipboard,
                ),
                state,
                (
                    input_grid,
                    input_mask,
                    target_grid,
                    jnp.zeros_like(state.selected),
                    jnp.zeros_like(state.clipboard),
                ),
            )

            # Recalculate similarity for new pair
            new_similarity = compute_grid_similarity(
                updated_state.working_grid, updated_state.target_grid
            )
            updated_state = eqx.tree_at(
                lambda s: s.similarity_score, updated_state, new_similarity
            )

            return updated_state

        # Use JAX conditional to update grids only if needed
        new_state = jax.lax.cond(
            is_pair_switching_op,
            update_grids_for_pair_switch,
            lambda s: s,  # No-op if not a pair switching operation
            new_state,
        )

        return new_state

    def handle_grid_operation(state, action, operation, config):
        """Handle grid operations (0-34) using structured actions and action handlers."""
        from .actions import point_handler, bbox_handler, mask_handler
        from .structured_actions import PointAction, BboxAction, MaskAction
        
        # Use appropriate action handler based on action type
        if isinstance(action, PointAction):
            selection_mask = point_handler(action, state.working_grid_mask)
        elif isinstance(action, BboxAction):
            selection_mask = bbox_handler(action, state.working_grid_mask)
        elif isinstance(action, MaskAction):
            selection_mask = mask_handler(action, state.working_grid_mask)
        else:
            # Fallback to the action's own method if handler not available
            selection_mask = action.to_selection_mask(grid_shape)
            # Constrain to working grid area
            selection_mask = selection_mask & state.working_grid_mask

        # Update selection in state
        new_state = eqx.tree_at(
            lambda s: s.selected, state, selection_mask
        )

        # Execute grid operation
        new_state = execute_grid_operation(new_state, operation)

        return new_state

    # Use JAX-compatible conditional to choose between control and grid operations
    new_state = jax.lax.cond(
        is_control_operation,
        lambda: handle_control_operation(state, validated_action, operation, config),
        lambda: handle_grid_operation(state, validated_action, operation, config),
    )

    return new_state, validated_action, is_control_operation


def _update_state(
    old_state: ArcEnvState,
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
        from .action_history import ActionHistoryTracker, HistoryConfig

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

    # Update step count using Equinox tree_at
    updated_state = eqx.tree_at(
        lambda s: s.step_count, updated_state, old_state.step_count + 1
    )

    return updated_state


def _calculate_reward_and_done(
    old_state: ArcEnvState,
    new_state: ArcEnvState,
    config: JaxArcConfig,
    is_control_operation: bool,
) -> Tuple[RewardValue, EpisodeDone, ArcEnvState]:
    """Calculate reward and done status - focused helper function.
    
    This helper function computes the reward signal and episode termination
    status based on state transitions, configuration settings, and operation type.
    It handles mode-specific reward calculation and enhanced termination logic.
    
    Args:
        old_state: Environment state before action execution
        new_state: Environment state after action execution
        config: Environment configuration for reward and termination settings
        is_control_operation: Whether the executed action was a control operation
    
    Returns:
        Tuple containing:
        - reward: Calculated reward value (JAX scalar)
        - done: Boolean indicating if episode should terminate
        - final_state: State with episode_done flag properly updated
    
    Examples:
        ```python
        # Calculate reward and termination for grid operation
        reward, done, state = _calculate_reward_and_done(old, new, config, False)
        
        # Calculate reward and termination for control operation
        reward, done, state = _calculate_reward_and_done(old, new, config, True)
        ```
    
    Note:
        Uses enhanced reward calculation with mode-specific logic and control
        operation adjustments. Updates episode_done flag in returned state.
    """
    # Mode-specific reward calculation logic
    reward = _calculate_enhanced_reward(
        old_state, new_state, config, is_control_operation
    )

    # Check if episode is done with enhanced termination logic
    done = _is_episode_done_enhanced(new_state, config)

    # Update episode_done flag using Equinox tree_at
    final_state = eqx.tree_at(lambda s: s.episode_done, new_state, done)

    return reward, done, final_state


@eqx.filter_jit
def arc_step(
    state: ArcEnvState,
    action: StructuredAction,
    config: ConfigType,
) -> Tuple[ArcEnvState, ObservationArray, RewardValue, EpisodeDone, Dict[str, Any]]:
    """
    Execute single step in ARC environment with comprehensive functionality.

    This enhanced step function provides comprehensive action processing with
    support for both grid operations (0-34) and control operations (35-41).
    The function has been decomposed into focused helper functions for better
    maintainability while preserving JAX compatibility and performance.

    The step process involves:
    1. Action processing and validation
    2. State updates with history tracking
    3. Reward calculation with mode-specific logic
    4. Episode termination checking
    5. Observation generation for agent

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
        action: Structured action to execute (PointAction, BboxAction, or MaskAction).
               Contains operation ID and selection data in type-safe format.
        config: Environment configuration (typed JaxArcConfig or Hydra DictConfig).
               Automatically converted to typed config if needed.

    Returns:
        Tuple of (new_state, agent_observation, reward, done, info) where:
        - new_state: Updated environment state after action execution
        - agent_observation: Focused observation for agent (currently working grid)
        - reward: Calculated reward value with mode-specific logic
        - done: Boolean indicating episode termination
        - info: Dictionary with step information and context

    Examples:
        ```python
        # Point-based grid operation
        action = PointAction(operation=15, row=5, col=10)
        new_state, obs, reward, done, info = arc_step(state, action, config)

        # Bounding box grid operation
        action = BboxAction(operation=10, r1=2, c1=3, r2=5, c2=7)
        new_state, obs, reward, done, info = arc_step(state, action, config)

        # Mask-based grid operation
        mask = jnp.zeros((30, 30), dtype=jnp.bool_).at[5:10, 5:10].set(True)
        action = MaskAction(operation=20, selection=mask)
        new_state, obs, reward, done, info = arc_step(state, action, config)

        # Control operation - switch to next demo pair
        action = PointAction(operation=35, row=0, col=0)  # Coordinates ignored for control ops
        new_state, obs, reward, done, info = arc_step(state, action, config)
        
        # Using typed config (preferred)
        from jaxarc.envs.config import JaxArcConfig
        typed_config = JaxArcConfig.from_hydra(hydra_config)
        new_state, obs, reward, done, info = arc_step(state, action, typed_config)
        ```

    Note:
        Function has been decomposed into helper functions (_process_action,
        _update_state, _calculate_reward_and_done) for better maintainability
        while preserving performance and JAX compatibility.
    """
    # Ensure we have a typed config
    typed_config = _ensure_config(config)

    # Process action and get updated state
    new_state, standardized_action, is_control_operation = _process_action(
        state, action, typed_config
    )

    # Update state with action history and step count
    updated_state = _update_state(state, new_state, standardized_action, typed_config)

    # Calculate reward and done status
    reward, done, final_state = _calculate_reward_and_done(
        state, updated_state, typed_config, is_control_operation
    )

    # Use create_observation function to generate focused agent view
    observation = create_observation(final_state, typed_config)

    # Create enhanced info dict with additional context (JAX-compatible)
    info = {
        "success": final_state.similarity_score >= 1.0,
        "similarity": final_state.similarity_score,
        "step_count": final_state.step_count,
        "similarity_improvement": final_state.similarity_score - state.similarity_score,
        "is_control_operation": is_control_operation,  # Keep as JAX array
        "operation_type": jax.lax.cond(
            is_control_operation, lambda: 1, lambda: 0
        ),  # 1=control, 0=grid
        "episode_mode": final_state.episode_mode,  # Already integer (0=train, 1=test)
        "current_pair_index": final_state.current_example_idx,
        "available_demo_pairs": final_state.get_available_demo_count(),
        "available_test_pairs": final_state.get_available_test_count(),
        "action_history_length": final_state.get_action_history_length(),
    }

    # Optional logging with enhanced information (JAX-compatible)
    def log_operations():
        """Log operations if enabled."""
        jax.debug.callback(
            lambda step, op, sim, rew, mode, pair: logger.info(
                f"Step {int(step)}: op={int(op)} ({'control' if int(op) >= 35 else 'grid'}), "
                f"sim={float(sim):.3f}, reward={float(rew):.3f}, "
                f"mode={'train' if int(mode) == 0 else 'test'}, pair={int(pair)}"
            ),
            final_state.step_count,
            standardized_action.operation,
            final_state.similarity_score,
            reward,
            final_state.episode_mode,
            final_state.current_example_idx,
        )

    def log_rewards():
        """Log rewards if enabled."""
        jax.debug.callback(
            lambda rew, imp: logger.info(
                f"Reward: {float(rew):.3f} (improvement: {float(imp):.3f})"
            ),
            reward,
            info["similarity_improvement"],
        )

    # Use JAX-compatible conditionals for logging
    jax.lax.cond(typed_config.logging.log_operations, log_operations, lambda: None)

    jax.lax.cond(typed_config.logging.log_rewards, log_rewards, lambda: None)

    # Optional visualization callback with enhanced information (JAX-compatible)
    def handle_visualization():
        """Handle visualization if enabled."""

        # Clear output directory at episode start (step 0) - using JAX-compatible conditional
        def clear_if_needed():
            # Note: This still has a Python conditional, but it's inside a callback
            # so it won't be traced by JAX
            jax.debug.callback(
                lambda output_dir, should_clear: _clear_output_directory(output_dir)
                if should_clear
                else None,
                typed_config.storage.base_output_dir,
                typed_config.storage.clear_output_on_start,
            )

        # Use lax.cond for JAX-compatible conditional on traced values
        jax.lax.cond(state.step_count == 0, clear_if_needed, lambda: None)

        # Use enhanced JAX callback (simplified for JAX compatibility)
        jax.debug.callback(
            jax_save_step_visualization,
            state,
            standardized_action,
            final_state,
            reward,
            info,
            typed_config.storage.base_output_dir,
        )

    # Use JAX-compatible conditional for visualization
    jax.lax.cond(typed_config.visualization.enabled, handle_visualization, lambda: None)

    return final_state, observation, reward, done, info



# =========================================================================
# Batch Processing Functions (Task 4.3 - Filtered Transformations with vmap)
# =========================================================================

@eqx.filter_jit
def batch_reset(
    keys: jnp.ndarray, 
    config: ConfigType, 
    task_data: JaxArcTask | None = None
) -> Tuple[ArcEnvState, ObservationArray]:
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
    states: ArcEnvState, 
    actions: StructuredAction, 
    config: ConfigType
) -> Tuple[ArcEnvState, ObservationArray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
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
        - Batched info dictionaries
    
    Examples:
        ```python
        # Create batched actions
        batch_size = 8
        batched_actions = PointAction(
            operation=jnp.array([0] * batch_size),
            row=jnp.array([3] * batch_size), 
            col=jnp.array([3] * batch_size)
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
    max_steps: int | None = None
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
    typed_config = _ensure_config(config)
    episode_max_steps = max_steps or typed_config.environment.max_episode_steps
    
    @eqx.filter_jit
    def run_batch_episodes(
        keys: jnp.ndarray, 
        num_steps: int
    ) -> Tuple[ArcEnvState, jnp.ndarray, jnp.ndarray]:
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
                col=jnp.full(batch_size, 2 + (step_idx % 5), dtype=jnp.int32)
            )
            
            # Step all environments
            new_states, _, rewards, dones, _ = batch_step(states, actions, config)
            
            # Update episode statistics
            episode_rewards += rewards
            episode_lengths = jnp.where(
                ~dones, 
                episode_lengths + 1, 
                episode_lengths
            )
            
            return (new_states, episode_rewards, episode_lengths), None
        
        # Run episode steps
        final_carry, _ = jax.lax.scan(
            step_fn, 
            (states, episode_rewards, episode_lengths),
            jnp.arange(num_steps)
        )
        
        final_states, final_rewards, final_lengths = final_carry
        return final_states, final_rewards, final_lengths
    
    return run_batch_episodes


# Utility functions for batch processing analysis

def analyze_batch_performance(
    config: ConfigType,
    task_data: JaxArcTask | None = None,
    batch_sizes: list[int] | None = None,
    num_steps: int = 10
) -> Dict[str, Any]:
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
        for batch_size, metrics in results['batch_metrics'].items():
            print(f"Batch {batch_size}: {metrics['steps_per_second']:.1f} steps/sec")
        ```
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]
    
    results = {
        'batch_metrics': {},
        'optimal_batch_size': None,
        'peak_throughput': 0.0
    }
    
    for batch_size in batch_sizes:
        # Create batch episode runner
        runner = create_batch_episode_runner(config, task_data, num_steps)
        
        # Generate keys
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
        
        # Warm up JIT compilation
        _ = runner(keys, 1)
        
        # Time actual execution
        import time
        start_time = time.perf_counter()
        final_states, rewards, lengths = runner(keys, num_steps)
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = end_time - start_time
        total_steps = batch_size * num_steps
        steps_per_second = total_steps / total_time
        time_per_env = total_time / batch_size
        
        metrics = {
            'batch_size': batch_size,
            'total_time': total_time,
            'time_per_env': time_per_env,
            'steps_per_second': steps_per_second,
            'avg_reward': float(jnp.mean(rewards)),
            'avg_length': float(jnp.mean(lengths))
        }
        
        results['batch_metrics'][batch_size] = metrics
        
        # Track optimal batch size
        if steps_per_second > results['peak_throughput']:
            results['peak_throughput'] = steps_per_second
            results['optimal_batch_size'] = batch_size
    
    return results