"""
Functional API for JaxARC environments with Hydra integration.

This module provides pure functional implementations of the ARC environment
that work with both typed configs and Hydra DictConfig objects.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig

from jaxarc.utils.visualization import (
    _clear_output_directory,
    save_rl_step_visualization,
    jax_save_step_visualization,
    jax_log_episode_summary,
)

from ..state import ArcEnvState
from ..types import ARCLEAction, JaxArcTask
from ..utils.jax_types import (
    EpisodeDone,
    GridArray,
    ObservationArray,
    OperationId,
    PRNGKey,
    RewardValue,
    SelectionArray,
    SimilarityScore,
)
from .actions import get_action_handler
from .config import ArcEnvConfig
from .grid_operations import compute_grid_similarity, execute_grid_operation

# Type aliases for cleaner signatures
ConfigType = Union[ArcEnvConfig, DictConfig]
ActionType = Union[Dict[str, Any], ARCLEAction]


def _ensure_config(config: ConfigType) -> ArcEnvConfig:
    """Convert config to typed ArcEnvConfig if needed."""
    if isinstance(config, DictConfig):
        return ArcEnvConfig.from_hydra(config)
    return config


def _validate_operation(operation: Any, config: ArcEnvConfig) -> OperationId:
    """Validate and normalize operation value."""
    if isinstance(operation, (int, jnp.integer)):
        operation = jnp.array(operation, dtype=jnp.int32)
    elif not isinstance(operation, jnp.ndarray):
        raise ValueError(f"Operation must be int or jnp.ndarray, got {type(operation)}")
    
    # Validate operation range (JAX-compatible)
    if config.action.validate_actions and config.action.clip_invalid_actions:
        operation = jnp.clip(operation, 0, config.action.num_operations - 1)
    
    return operation


def _get_observation(state: ArcEnvState, config: ArcEnvConfig) -> ObservationArray:
    """Extract observation from state."""
    # For now, just return the working grid
    # Future: Could include additional channels for selection, target, etc.
    return state.working_grid


def _calculate_reward(
    old_state: ArcEnvState, new_state: ArcEnvState, config: ArcEnvConfig
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


def _is_episode_done(state: ArcEnvState, config: ArcEnvConfig) -> EpisodeDone:
    """Check if episode should terminate."""
    # Episode ends if:
    # 1. Task is solved (perfect similarity)
    # 2. Maximum steps reached
    # 3. Submit operation was used (sets episode_done=True)

    task_solved = state.similarity_score >= 1.0
    max_steps_reached = state.step_count >= config.max_episode_steps
    submitted = state.episode_done  # Set by submit operation

    return task_solved | max_steps_reached | submitted


def _create_demo_task(config: ArcEnvConfig) -> JaxArcTask:
    """Create a simple demo task for testing when no task sampler is available."""
    from ..types import JaxArcTask

    logger.warning(
        "No task sampler provided - creating demo task. "
        "For real training, provide a task_sampler in config or task_data directly."
    )

    # Create simple demo grids using dataset-appropriate size
    # Use full configured grid size for proper testing, with reasonable upper bounds
    demo_height = min(30, config.grid.max_grid_height)
    demo_width = min(30, config.grid.max_grid_width)
    grid_shape = (demo_height, demo_width)

    # Use dataset background color
    bg_color = config.grid.background_color
    input_grid = jnp.full(grid_shape, bg_color, dtype=jnp.int32)

    # Add a small pattern (ensure it fits and uses valid colors)
    pattern_size = min(3, demo_height - 2, demo_width - 2)
    if pattern_size > 0:
        start_row = (demo_height - pattern_size) // 2
        start_col = (demo_width - pattern_size) // 2

        # Use color 1 if available, otherwise background + 1
        pattern_color = (
            1 if config.grid.max_colors > 1 else (bg_color + 1) % config.grid.max_colors
        )
        input_grid = input_grid.at[
            start_row : start_row + pattern_size, start_col : start_col + pattern_size
        ].set(pattern_color)

    # Target: change pattern to different color
    target_grid = input_grid.copy()
    if pattern_size > 0:
        target_color = (
            2
            if config.grid.max_colors > 2
            else (pattern_color + 1) % config.grid.max_colors
        )
        target_grid = target_grid.at[
            start_row : start_row + pattern_size, start_col : start_col + pattern_size
        ].set(target_color)

    # Create masks
    mask = jnp.ones(grid_shape, dtype=jnp.bool_)

    # Pad to max size
    max_shape = config.grid.max_grid_size
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
) -> Tuple[ArcEnvState, ObservationArray]:
    """
    Reset ARC environment with functional API.

    Args:
        key: JAX PRNG key
        config: Environment configuration (typed or Hydra DictConfig)
        task_data: Optional specific task data. If None, will use parser from config
                      or create demo task as fallback.

    Returns:
        Tuple of (initial_state, initial_observation)
    """
    # Ensure we have a typed config
    typed_config = _ensure_config(config)

    # Get or create task data
    if task_data is None:
        if typed_config.parser is not None:
            # Use provided parser
            try:
                task_data = typed_config.parser.get_random_task(key)
                if typed_config.log_operations:
                    logger.info(
                        f"Sampled task from {typed_config.dataset.dataset_name}"
                    )
            except Exception as e:
                logger.warning(f"Parser failed: {e}. Falling back to demo task.")
                task_data = _create_demo_task(typed_config)
        else:
            # Create demo task as fallback
            task_data = _create_demo_task(typed_config)

    # Initialize working grid from first training example
    initial_grid = task_data.input_grids_examples[0]
    target_grid = task_data.output_grids_examples[0]
    initial_mask = task_data.input_masks_examples[0]

    # Calculate initial similarity
    initial_similarity = compute_grid_similarity(initial_grid, target_grid)

    # Create initial state
    state = ArcEnvState(
        task_data=task_data,
        working_grid=initial_grid,
        working_grid_mask=initial_mask,
        target_grid=target_grid,
        step_count=0,
        episode_done=False,
        current_example_idx=0,
        selected=jnp.zeros_like(initial_grid, dtype=jnp.bool_),
        clipboard=jnp.zeros_like(initial_grid, dtype=jnp.int32),
        similarity_score=initial_similarity,
    )

    # Get initial observation
    observation = _get_observation(state, typed_config)

    if typed_config.log_operations:
        jax.debug.callback(
            lambda x: logger.info(
                f"Reset ARC environment with similarity: {float(x):.3f}"
            ),
            initial_similarity,
        )

    return state, observation


def arc_step(
    state: ArcEnvState,
    action: ActionType,
    config: ConfigType,
) -> Tuple[ArcEnvState, ObservationArray, RewardValue, EpisodeDone, Dict[str, Any]]:
    """
    Execute single step in ARC environment with functional API.

    Args:
        state: Current environment state
        action: Action to execute (dict or ARCLEAction)
        config: Environment configuration (typed or Hydra DictConfig)

    Returns:
        Tuple of (new_state, observation, reward, done, info)
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
                raise ValueError(f"Selection must be 1D or 2D array, got shape {selection.shape}")
        else:
            raise ValueError("Action must contain 'selection' field")
    else:
        raise ValueError(f"Unknown selection format: {typed_config.action.selection_format}")

    # Handler creates standardized selection mask
    selection_mask = handler(action_data, state.working_grid_mask)
    
    # Validate and normalize operation
    operation = _validate_operation(action["operation"], typed_config)
    
    # Create standardized action dictionary
    standardized_action = {
        "selection": selection_mask,
        "operation": operation
    }

    # Update selection in state using Equinox tree_at for better performance
    state = eqx.tree_at(lambda s: s.selected, state, standardized_action["selection"])

    # Execute operation using existing grid operations
    new_state = execute_grid_operation(state, standardized_action["operation"])

    # Update step count using Equinox tree_at
    new_state = eqx.tree_at(lambda s: s.step_count, new_state, state.step_count + 1)

    # Calculate reward
    reward = _calculate_reward(state, new_state, typed_config)

    # Check if episode is done
    done = _is_episode_done(new_state, typed_config)

    # Update episode_done flag using Equinox tree_at
    new_state = eqx.tree_at(lambda s: s.episode_done, new_state, done)

    # Get observation
    observation = _get_observation(new_state, typed_config)

    # Create info dict
    info = {
        "success": new_state.similarity_score >= 1.0,
        "similarity": new_state.similarity_score,
        "step_count": new_state.step_count,
        "similarity_improvement": new_state.similarity_score - state.similarity_score,
    }

    # Optional logging
    if typed_config.log_operations:
        jax.debug.callback(
            lambda step, op, sim, rew: logger.info(
                f"Step {int(step)}: op={int(op)}, sim={float(sim):.3f}, reward={float(rew):.3f}"
            ),
            new_state.step_count,
            standardized_action["operation"],
            new_state.similarity_score,
            reward,
        )

    if typed_config.log_rewards:
        jax.debug.callback(
            lambda rew, imp: logger.info(
                f"Reward: {float(rew):.3f} (improvement: {float(imp):.3f})"
            ),
            reward,
            info["similarity_improvement"],
        )

    # Optional visualization callback
    if typed_config.debug.log_rl_steps:
        # Clear output directory at episode start (step 0)
        if state.step_count == 0 and typed_config.debug.clear_output_dir:
            jax.debug.callback(
                lambda output_dir: _clear_output_directory(output_dir),
                typed_config.debug.rl_steps_output_dir,
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
                typed_config.debug.rl_steps_output_dir,
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
