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
    save_rl_step_visualization,
)

from ..state import ArcEnvState
from ..types import ARCLEAction, JaxArcTask
from ..utils.jax_types import (
    ACTION_RECORD_FIELDS,
    EpisodeDone,
    ObservationArray,
    OperationId,
    PRNGKey,
    get_action_record_fields,
    RewardValue,
)
from .actions import get_action_handler
from .config import JaxArcConfig
from .grid_operations import compute_grid_similarity, execute_grid_operation

# Type aliases for cleaner signatures
ConfigType = Union[JaxArcConfig, DictConfig]
ActionType = Union[Dict[str, Any], ARCLEAction]




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
    is_control_operation: bool
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
    episode_config = getattr(config, 'episode', None)
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
        lambda: _calculate_training_submit_reward(old_state, new_state, reward_cfg, similarity_improvement),
        lambda: _calculate_training_step_reward(old_state, new_state, reward_cfg, similarity_improvement)
    )
    
    # Evaluation mode reward calculation with target masking
    evaluation_reward = jax.lax.cond(
        evaluation_reward_frequency == "submit",
        lambda: _calculate_evaluation_submit_reward(old_state, new_state, reward_cfg),
        lambda: _calculate_evaluation_step_reward(old_state, new_state, reward_cfg)
    )
    
    # Mode-specific reward selection using JAX-compatible conditional
    mode_reward = jax.lax.cond(
        is_training,
        lambda: training_reward,
        lambda: evaluation_reward
    )
    
    # Control operation adjustments with proper JAX operations
    control_adjustment = jax.lax.cond(
        is_control_operation,
        lambda: _calculate_control_operation_reward(new_state, reward_cfg),
        lambda: jnp.array(0.0, dtype=jnp.float32)
    )
    
    # Final reward calculation
    final_reward = mode_reward + control_adjustment
    
    return final_reward


def _calculate_training_submit_reward(
    old_state: ArcEnvState,
    new_state: ArcEnvState,
    reward_cfg,
    similarity_improvement: RewardValue
) -> RewardValue:
    """Calculate training mode reward for submit-only frequency.
    
    In submit-only mode, full rewards are only given when the episode ends
    (submit operation), otherwise only step penalty is applied.
    """
    # Full reward calculation for submit operations with enhanced bonuses
    similarity_reward = reward_cfg.training_similarity_weight * similarity_improvement
    progress_bonus = jnp.where(similarity_improvement > 0, reward_cfg.progress_bonus, 0.0)
    step_penalty = jnp.array(reward_cfg.step_penalty, dtype=jnp.float32)
    
    # Enhanced success bonus calculation
    is_solved = new_state.similarity_score >= 1.0
    base_success_bonus = jnp.where(is_solved, reward_cfg.success_bonus, 0.0)
    demo_bonus = jnp.where(is_solved, reward_cfg.demo_completion_bonus, 0.0)
    
    # Efficiency bonus for solving within threshold
    efficiency_bonus = jnp.where(
        jnp.logical_and(is_solved, new_state.step_count <= reward_cfg.efficiency_bonus_threshold),
        reward_cfg.efficiency_bonus,
        0.0
    )
    
    full_reward = similarity_reward + progress_bonus + step_penalty + base_success_bonus + demo_bonus + efficiency_bonus
    submit_only_reward = step_penalty  # Only step penalty between submits
    
    # Return full reward if episode is done (submit), otherwise just step penalty
    return jnp.where(new_state.episode_done, full_reward, submit_only_reward)


def _calculate_training_step_reward(
    old_state: ArcEnvState,
    new_state: ArcEnvState,
    reward_cfg,
    similarity_improvement: RewardValue
) -> RewardValue:
    """Calculate training mode reward for step-by-step frequency.
    
    In step mode, rewards are calculated and given on every step with
    enhanced pair-type specific bonuses and efficiency considerations.
    """
    # Use training-specific similarity weight
    similarity_reward = reward_cfg.training_similarity_weight * similarity_improvement
    progress_bonus = jnp.where(similarity_improvement > 0, reward_cfg.progress_bonus, 0.0)
    step_penalty = jnp.array(reward_cfg.step_penalty, dtype=jnp.float32)
    
    # Enhanced success bonus with pair-type specific bonuses
    is_solved = new_state.similarity_score >= 1.0
    base_success_bonus = jnp.where(is_solved, reward_cfg.success_bonus, 0.0)
    
    # Add demo completion bonus for demonstration pairs
    demo_bonus = jnp.where(is_solved, reward_cfg.demo_completion_bonus, 0.0)
    
    # Add efficiency bonus if solved within threshold
    efficiency_bonus = jnp.where(
        jnp.logical_and(is_solved, new_state.step_count <= reward_cfg.efficiency_bonus_threshold),
        reward_cfg.efficiency_bonus,
        0.0
    )
    
    return similarity_reward + progress_bonus + step_penalty + base_success_bonus + demo_bonus + efficiency_bonus


def _calculate_evaluation_submit_reward(
    old_state: ArcEnvState,
    new_state: ArcEnvState,
    reward_cfg
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
        jnp.logical_and(is_solved, new_state.step_count <= reward_cfg.efficiency_bonus_threshold),
        reward_cfg.efficiency_bonus,
        0.0
    )
    
    submit_reward = step_penalty + base_success_bonus + test_bonus + efficiency_bonus
    
    # Return submit reward if episode is done, otherwise just step penalty
    return jnp.where(new_state.episode_done, submit_reward, step_penalty)


def _calculate_evaluation_step_reward(
    old_state: ArcEnvState,
    new_state: ArcEnvState,
    reward_cfg
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
        jnp.logical_and(is_solved, new_state.step_count <= reward_cfg.efficiency_bonus_threshold),
        reward_cfg.efficiency_bonus,
        0.0
    )
    
    return step_penalty + base_success_bonus + test_bonus + efficiency_bonus


def _calculate_control_operation_reward(
    new_state: ArcEnvState,
    reward_cfg
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
    working_grid: Array,
    target_grid: Array,
    pair_type: str,
    episode_mode: int
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
            1 if config.dataset.max_colors > 1 else (bg_color + 1) % config.dataset.max_colors
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


def arc_reset(
    key: PRNGKey,
    config: ConfigType,
    task_data: JaxArcTask | None = None,
    episode_mode: str = "train",
    initial_pair_idx: int | None = None,
) -> Tuple[ArcEnvState, ObservationArray]:
    """
    Reset ARC environment with enhanced multi-demonstration support.

    This enhanced reset function provides comprehensive support for multi-demonstration
    training and test pair evaluation with proper mode-specific initialization,
    action history setup, and dynamic operation control.

    Args:
        key: JAX PRNG key for reproducible randomization
        config: Environment configuration (typed or Hydra DictConfig)
        task_data: Optional specific task data. If None, will use parser from config
                      or create demo task as fallback.
        episode_mode: Episode mode ("train" or "test") for initialization
        initial_pair_idx: Optional explicit pair index specification

    Returns:
        Tuple of (initial_state, initial_observation) with enhanced functionality

    Examples:
        ```python
        # Reset in training mode with random demo pair selection
        state, obs = arc_reset(key, config, task_data, episode_mode="train")
        
        # Reset in test mode with specific test pair
        state, obs = arc_reset(key, config, task_data, episode_mode="test", initial_pair_idx=1)
        
        # Reset with explicit pair selection in training mode
        state, obs = arc_reset(key, config, task_data, episode_mode="train", initial_pair_idx=2)
        ```
    """
    # Ensure we have a typed config
    typed_config = _ensure_config(config)

    # Get or create task data
    if task_data is None:
        # Create demo task as fallback
        # Parsers are handled separately in the new system
        task_data = _create_demo_task(typed_config)
        if typed_config.logging.log_operations:
            logger.info(
                f"Created demo task for dataset {typed_config.dataset.dataset_name}"
            )

    # Import episode manager for pair selection
    from .episode_manager import ArcEpisodeConfig, ArcEpisodeManager

    # Create episode configuration with proper mode validation
    if episode_mode not in ["train", "test"]:
        raise ValueError(f"episode_mode must be 'train' or 'test', got '{episode_mode}'")
    
    episode_config = ArcEpisodeConfig(episode_mode=episode_mode)

    # Select initial pair based on mode and configuration strategy
    if initial_pair_idx is not None:
        # Use explicit pair index if provided
        selected_pair_idx = jnp.array(initial_pair_idx, dtype=jnp.int32)
        
        # Validate the explicit pair index using task data methods
        if episode_mode == "train":
            selection_successful = task_data.is_demo_pair_available(initial_pair_idx)
        else:
            selection_successful = task_data.is_test_pair_available(initial_pair_idx)
    else:
        # Use episode manager for pair selection based on configuration strategy
        selected_pair_idx, selection_successful = ArcEpisodeManager.select_initial_pair(
            key, task_data, episode_config
        )

    # Handle selection failure using JAX-compatible operations
    fallback_idx = jnp.array(0, dtype=jnp.int32)
    selected_pair_idx = jnp.where(selection_successful, selected_pair_idx, fallback_idx)

    # Initialize grids based on episode mode using JAX-compatible operations
    # Get available pairs and completion status for enhanced functionality
    available_demo_pairs = task_data.get_available_demo_pairs()
    available_test_pairs = task_data.get_available_test_pairs()
    demo_completion_status = jnp.zeros_like(available_demo_pairs)
    test_completion_status = jnp.zeros_like(available_test_pairs)
    
    # Mode-specific grid initialization
    if episode_mode == "train":
        # Training mode: use demonstration pair with target access
        initial_grid = task_data.input_grids_examples[selected_pair_idx]
        target_grid = task_data.output_grids_examples[selected_pair_idx]
        initial_mask = task_data.input_masks_examples[selected_pair_idx]
        episode_mode_int = jnp.array(0, dtype=jnp.int32)  # 0 = train
        
    else:
        # Test mode: use test pair with target masking for proper evaluation
        initial_grid = task_data.test_input_grids[selected_pair_idx]
        initial_mask = task_data.test_input_masks[selected_pair_idx]
        # In test mode, target grid is masked (set to background color) to prevent cheating
        # Use background color instead of zeros to avoid artificially high similarity
        background_color = typed_config.dataset.background_color
        target_grid = jnp.full_like(initial_grid, background_color)  # Masked target for evaluation
        episode_mode_int = jnp.array(1, dtype=jnp.int32)  # 1 = test

    # Calculate initial similarity (will be 0.0 in test mode due to masked target)
    initial_similarity = compute_grid_similarity(initial_grid, target_grid)

    # Initialize action history with dynamic sizing based on configuration
    max_history_length = getattr(typed_config, 'max_history_length', 1000)
    num_operations = 42  # Updated to include enhanced control operations (0-41)
    
    # Calculate optimal action record fields based on selection format and dataset
    action_record_fields = get_action_record_fields(
        typed_config.action.selection_format,
        typed_config.dataset.max_grid_height,
        typed_config.dataset.max_grid_width
    )
    
    # Initialize action history storage with proper dimensions
    action_history = jnp.zeros((max_history_length, action_record_fields), dtype=jnp.float32)
    action_history_length = jnp.array(0, dtype=jnp.int32)
    
    # Initialize allowed operations mask (all operations allowed by default)
    allowed_operations_mask = jnp.ones(num_operations, dtype=jnp.bool_)

    # Create enhanced initial state with all new fields
    state = ArcEnvState(
        # Core ARC state (unchanged)
        task_data=task_data,
        working_grid=initial_grid,
        working_grid_mask=initial_mask,
        target_grid=target_grid,
        step_count=jnp.array(0, dtype=jnp.int32),
        episode_done=jnp.array(False),
        current_example_idx=selected_pair_idx,
        selected=jnp.zeros_like(initial_grid, dtype=jnp.bool_),
        clipboard=jnp.zeros_like(initial_grid, dtype=jnp.int32),
        similarity_score=initial_similarity,
        
        # Enhanced functionality fields
        episode_mode=episode_mode_int,
        available_demo_pairs=available_demo_pairs,
        available_test_pairs=available_test_pairs,
        demo_completion_status=demo_completion_status,
        test_completion_status=test_completion_status,
        action_history=action_history,
        action_history_length=action_history_length,
        allowed_operations_mask=allowed_operations_mask,
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
            episode_mode_int,
            selected_pair_idx,
            initial_similarity,
            jnp.sum(available_demo_pairs),
            jnp.sum(available_test_pairs),
        )

    return state, observation


def arc_step(
    state: ArcEnvState,
    action: ActionType,
    config: ConfigType,
) -> Tuple[ArcEnvState, ObservationArray, RewardValue, EpisodeDone, Dict[str, Any]]:
    """
    Execute single step in ARC environment with comprehensive functionality.

    This enhanced step function provides:
    - Support for enhanced non-parametric control operations (35-41)
    - Action history tracking for each step with memory optimization
    - Dynamic action space validation and filtering
    - Mode-specific reward calculation logic
    - Non-parametric pair switching logic (next/prev/first_unsolved)
    - Focused agent observation generation

    Args:
        state: Current environment state with enhanced functionality
        action: Action to execute (dict or ARCLEAction)
        config: Environment configuration (typed or Hydra DictConfig)

    Returns:
        Tuple of (new_state, agent_observation, reward, done, info)

    Examples:
        ```python
        # Standard grid operation
        action = {"selection": mask, "operation": 15}
        new_state, obs, reward, done, info = arc_step(state, action, config)
        
        # Control operation - switch to next demo pair
        action = {"selection": jnp.zeros_like(mask), "operation": 35}
        new_state, obs, reward, done, info = arc_step(state, action, config)
        
        # Control operation - reset current pair
        action = {"selection": jnp.zeros_like(mask), "operation": 39}
        new_state, obs, reward, done, info = arc_step(state, action, config)
        ```
    """
    # Ensure we have a typed config
    typed_config = _ensure_config(config)

    # Convert ARCLEAction to dict format if needed
    if isinstance(action, ARCLEAction):
        action = {
            "selection": action.selection,
            "operation": action.operation,
        }

    # Validate action format
    if not isinstance(action, dict):
        raise ValueError("Action must be a dictionary")

    if "operation" not in action:
        raise ValueError("Action must contain 'operation' field")

    # Validate and normalize operation with enhanced range (0-41)
    operation = _validate_operation(action["operation"], typed_config)

    # Apply dynamic action space validation and filtering if enabled
    if hasattr(typed_config.action, 'dynamic_action_filtering') and typed_config.action.dynamic_action_filtering:
        from .action_space_controller import ActionSpaceController
        controller = ActionSpaceController()
        
        # Filter invalid operation according to policy
        operation = controller.filter_invalid_operation_jax(operation, state, typed_config.action)

    # Check if this is a control operation (35-41) or grid operation (0-34)
    is_control_operation = operation >= 35

    # Handle control operations using episode manager
    if is_control_operation:
        # Import episode manager for control operations
        from .episode_manager import ArcEpisodeConfig, ArcEpisodeManager
        
        # Create episode configuration from main config
        episode_config = ArcEpisodeConfig(
            episode_mode="train" if state.is_training_mode() else "test",
            demo_selection_strategy=getattr(typed_config, 'demo_selection_strategy', 'random'),
            allow_demo_switching=getattr(typed_config, 'allow_demo_switching', True),
            allow_test_switching=getattr(typed_config, 'allow_test_switching', False),
        )
        
        # Execute pair control operation
        new_state = ArcEpisodeManager.execute_pair_control_operation(
            state, int(operation), episode_config
        )
        
        # For control operations, we need to update grids if pair was switched
        if int(operation) in [35, 36, 37, 38, 40, 41]:  # Pair switching operations
            # Update working grid and target based on new pair
            if new_state.is_training_mode():
                input_grid, target_grid, input_mask = new_state.task_data.get_demo_pair_data(
                    int(new_state.current_example_idx)
                )
            else:
                input_grid, input_mask = new_state.task_data.get_test_pair_data(
                    int(new_state.current_example_idx)
                )
                # In test mode, mask target grid
                background_color = getattr(typed_config.dataset, 'background_color', 0)
                target_grid = jnp.full_like(input_grid, background_color)
            
            # Update state with new grids
            new_state = eqx.tree_at(
                lambda s: (s.working_grid, s.working_grid_mask, s.target_grid, s.selected, s.clipboard),
                new_state,
                (input_grid, input_mask, target_grid, jnp.zeros_like(state.selected), jnp.zeros_like(state.clipboard))
            )
            
            # Recalculate similarity for new pair
            from .grid_operations import compute_grid_similarity
            new_similarity = compute_grid_similarity(new_state.working_grid, new_state.target_grid)
            new_state = eqx.tree_at(lambda s: s.similarity_score, new_state, new_similarity)
        
        # Create standardized action for history tracking (control operations use empty selection)
        standardized_action = {
            "selection": jnp.zeros_like(state.selected),
            "operation": operation
        }
        
    else:
        # Handle grid operations (0-34) using existing logic
        
        # Get the appropriate action handler based on configuration
        handler = get_action_handler(typed_config.action.selection_format)

        # Extract action data based on selection format
        if typed_config.action.selection_format == "point":
            if "point" not in action:
                raise ValueError("Action must contain 'point' field")
            action_data = jnp.array(action["point"])
        elif typed_config.action.selection_format == "bbox":
            if "bbox" not in action:
                raise ValueError("Action must contain 'bbox' field")
            action_data = jnp.array(action["bbox"])
        elif typed_config.action.selection_format == "mask":
            if "mask" in action:
                mask = action["mask"]
                # Handle both 2D (grid shape) and 1D (flattened) masks
                if len(mask.shape) == 2:
                    # 2D mask - validate shape
                    if mask.shape != state.working_grid_mask.shape:
                        raise ValueError(
                            f"Selection shape {mask.shape} doesn't match grid shape {state.working_grid_mask.shape}"
                        )
                    action_data = mask.flatten()
                elif len(mask.shape) == 1:
                    # 1D mask - validate size
                    expected_size = state.working_grid_mask.size
                    if mask.size != expected_size:
                        raise ValueError(
                            f"Selection size {mask.size} doesn't match grid size {expected_size}"
                        )
                    action_data = mask
                else:
                    raise ValueError(f"Mask must be 1D or 2D array, got shape {mask.shape}")
            elif "selection" in action:
                selection = action["selection"]
                # Handle both 2D (grid shape) and 1D (flattened) selections
                if len(selection.shape) == 2:
                    # 2D selection - validate shape
                    if selection.shape != state.working_grid_mask.shape:
                        raise ValueError(
                            f"Selection shape {selection.shape} doesn't match grid shape {state.working_grid_mask.shape}"
                        )
                    action_data = selection.flatten()
                elif len(selection.shape) == 1:
                    # 1D selection - validate size
                    expected_size = state.working_grid_mask.size
                    if selection.size != expected_size:
                        raise ValueError(
                            f"Selection size {selection.size} doesn't match grid size {expected_size}"
                        )
                    action_data = selection
                else:
                    raise ValueError(
                        f"Selection must be 1D or 2D array, got shape {selection.shape}"
                    )
            else:
                raise ValueError("Action must contain 'selection' field")
        else:
            raise ValueError(
                f"Unknown selection format: {typed_config.action.selection_format}"
            )

        # Handler creates standardized selection mask
        selection_mask = handler(action_data, state.working_grid_mask)

        # Create standardized action dictionary
        standardized_action = {"selection": selection_mask, "operation": operation}

        # Update selection in state using Equinox tree_at for better performance
        new_state = eqx.tree_at(lambda s: s.selected, state, standardized_action["selection"])

        # Execute grid operation using existing grid operations
        new_state = execute_grid_operation(new_state, standardized_action["operation"])

    # Add action history tracking for each step with memory optimization
    if hasattr(typed_config, 'history') and getattr(typed_config.history, 'enabled', True):
        from .action_history import ActionHistoryTracker, HistoryConfig
        
        # Create history config from main config or use defaults
        history_config = HistoryConfig(
            enabled=getattr(typed_config.history, 'enabled', True),
            max_history_length=getattr(typed_config.history, 'max_history_length', 1000),
            store_selection_data=getattr(typed_config.history, 'store_selection_data', True),
            compress_repeated_actions=getattr(typed_config.history, 'compress_repeated_actions', True)
        )
        
        # Add action to history
        tracker = ActionHistoryTracker()
        new_state = tracker.add_action(
            new_state,
            action,
            history_config,
            typed_config.action.selection_format,
            typed_config.dataset.max_grid_height,
            typed_config.dataset.max_grid_width
        )

    # Update step count using Equinox tree_at
    new_state = eqx.tree_at(lambda s: s.step_count, new_state, state.step_count + 1)

    # Mode-specific reward calculation logic
    reward = _calculate_enhanced_reward(state, new_state, typed_config, is_control_operation)

    # Check if episode is done with enhanced termination logic
    done = _is_episode_done_enhanced(new_state, typed_config)

    # Update episode_done flag using Equinox tree_at
    new_state = eqx.tree_at(lambda s: s.episode_done, new_state, done)

    # Use create_observation function to generate focused agent view
    observation = create_observation(new_state, typed_config)

    # Create enhanced info dict with additional context
    info = {
        "success": new_state.similarity_score >= 1.0,
        "similarity": new_state.similarity_score,
        "step_count": new_state.step_count,
        "similarity_improvement": new_state.similarity_score - state.similarity_score,
        "is_control_operation": bool(is_control_operation),
        "operation_type": "control" if is_control_operation else "grid",
        "episode_mode": "train" if new_state.is_training_mode() else "test",
        "current_pair_index": int(new_state.current_example_idx),
        "available_demo_pairs": int(new_state.get_available_demo_count()),
        "available_test_pairs": int(new_state.get_available_test_count()),
        "action_history_length": int(new_state.get_action_history_length()),
    }

    # Optional logging with enhanced information
    if typed_config.logging.log_operations:
        jax.debug.callback(
            lambda step, op, sim, rew, mode, pair: logger.info(
                f"Step {int(step)}: op={int(op)} ({'control' if int(op) >= 35 else 'grid'}), "
                f"sim={float(sim):.3f}, reward={float(rew):.3f}, "
                f"mode={'train' if int(mode) == 0 else 'test'}, pair={int(pair)}"
            ),
            new_state.step_count,
            standardized_action["operation"],
            new_state.similarity_score,
            reward,
            new_state.episode_mode,
            new_state.current_example_idx,
        )

    if typed_config.logging.log_rewards:
        jax.debug.callback(
            lambda rew, imp: logger.info(
                f"Reward: {float(rew):.3f} (improvement: {float(imp):.3f})"
            ),
            reward,
            info["similarity_improvement"],
        )

    # Optional visualization callback with enhanced information
    if typed_config.visualization.enabled:
        # Clear output directory at episode start (step 0) - using JAX-compatible conditional
        def clear_if_needed():
            if typed_config.storage.clear_output_on_start:
                jax.debug.callback(
                    lambda output_dir: _clear_output_directory(output_dir),
                    typed_config.storage.base_output_dir,
                )

        # Use lax.cond for JAX-compatible conditional on traced values
        jax.lax.cond(
            state.step_count == 0,
            clear_if_needed,
            lambda: None
        )

        # Use enhanced JAX callback if available, otherwise fallback to legacy
        try:
            jax.debug.callback(
                jax_save_step_visualization,
                state,
                standardized_action,
                new_state,
                reward,
                info,
                typed_config.storage.base_output_dir,
            )
        except Exception:
            # Fallback to legacy visualization
            jax.debug.callback(
                save_rl_step_visualization,
                state,
                standardized_action,
                new_state,
                typed_config.debug.rl_steps_output_dir,
            )

    return new_state, observation, reward, done, info


# Convenience functions for common use cases


def arc_reset_with_hydra(
    key: PRNGKey,
    hydra_config: DictConfig,
    task_data: JaxArcTask | None = None,
) -> Tuple[ArcEnvState, ObservationArray]:
    """Reset with explicit Hydra config (for type clarity)."""
    return arc_reset(key, hydra_config, task_data)


def arc_step_with_hydra(
    state: ArcEnvState,
    action: ActionType,
    hydra_config: DictConfig,
) -> Tuple[ArcEnvState, ObservationArray, RewardValue, EpisodeDone, Dict[str, Any]]:
    """Step with explicit Hydra config (for type clarity)."""
    return arc_step(state, action, hydra_config)


# Enhanced reset functions for mode-specific initialization


def arc_reset_training_mode(
    key: PRNGKey,
    config: ConfigType,
    task_data: JaxArcTask | None = None,
    initial_pair_idx: int | None = None,
) -> Tuple[ArcEnvState, ObservationArray]:
    """Reset ARC environment in training mode with demonstration pairs.
    
    Args:
        key: JAX PRNG key
        config: Environment configuration
        task_data: Optional task data
        initial_pair_idx: Optional explicit demonstration pair index
        
    Returns:
        Tuple of (initial_state, initial_observation) configured for training
        
    Examples:
        ```python
        # Reset with random demo pair selection
        state, obs = arc_reset_training_mode(key, config, task_data)
        
        # Reset with specific demo pair
        state, obs = arc_reset_training_mode(key, config, task_data, initial_pair_idx=2)
        ```
    """
    return arc_reset(key, config, task_data, episode_mode="train", initial_pair_idx=initial_pair_idx)


def arc_reset_evaluation_mode(
    key: PRNGKey,
    config: ConfigType,
    task_data: JaxArcTask | None = None,
    initial_pair_idx: int | None = None,
) -> Tuple[ArcEnvState, ObservationArray]:
    """Reset ARC environment in evaluation mode with test pairs.
    
    Args:
        key: JAX PRNG key
        config: Environment configuration
        task_data: Optional task data
        initial_pair_idx: Optional explicit test pair index
        
    Returns:
        Tuple of (initial_state, initial_observation) configured for evaluation
        
    Examples:
        ```python
        # Reset with first test pair
        state, obs = arc_reset_evaluation_mode(key, config, task_data)
        
        # Reset with specific test pair
        state, obs = arc_reset_evaluation_mode(key, config, task_data, initial_pair_idx=1)
        ```
    """
    return arc_reset(key, config, task_data, episode_mode="test", initial_pair_idx=initial_pair_idx)
