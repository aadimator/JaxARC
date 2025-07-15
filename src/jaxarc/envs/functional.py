"""
Functional API for JaxARC environments with Hydra integration.

This module provides pure functional implementations of the ARC environment
that work with both typed configs and Hydra DictConfig objects.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig

from ..types import ARCLEAction, JaxArcTask, Grid
from .actions import get_action_handler
from .config import ArcEnvConfig
from .grid_operations import compute_grid_similarity, execute_grid_operation
from jaxarc.utils.visualization import save_rl_step_visualization, _clear_output_directory

# Type aliases for cleaner signatures
ConfigType = Union[ArcEnvConfig, DictConfig]
ActionType = Union[Dict[str, Any], ARCLEAction]


@chex.dataclass
class ArcEnvState:
    """ARC environment state with full grid operations compatibility."""

    # Core ARC state
    task_data: JaxArcTask
    working_grid: jnp.ndarray  # Current grid being modified
    working_grid_mask: jnp.ndarray  # Valid cells mask
    target_grid: jnp.ndarray  # Goal grid for current example

    # Episode management
    step_count: int
    episode_done: bool
    current_example_idx: int  # Which training example we're working on

    # Grid operations fields
    selected: jnp.ndarray  # Selection mask for operations
    clipboard: jnp.ndarray  # For copy/paste operations
    similarity_score: jnp.ndarray  # Grid similarity to target

    def __post_init__(self) -> None:
        """Validate ARC environment state structure."""
        # Skip validation during JAX transformations
        if not hasattr(self.working_grid, "shape"):
            return

        try:
            # Validate grid shapes and types
            chex.assert_rank(self.working_grid, 2)
            chex.assert_rank(self.working_grid_mask, 2)
            chex.assert_rank(self.target_grid, 2)
            chex.assert_rank(self.selected, 2)
            chex.assert_rank(self.clipboard, 2)

            chex.assert_type(self.working_grid, jnp.integer)
            chex.assert_type(self.working_grid_mask, jnp.bool_)
            chex.assert_type(self.target_grid, jnp.integer)
            chex.assert_type(self.selected, jnp.bool_)
            chex.assert_type(self.clipboard, jnp.integer)
            chex.assert_type(self.similarity_score, jnp.floating)

            # Check consistent shapes
            chex.assert_shape(self.working_grid_mask, self.working_grid.shape)
            chex.assert_shape(self.target_grid, self.working_grid.shape)
            chex.assert_shape(self.selected, self.working_grid.shape)
            chex.assert_shape(self.clipboard, self.working_grid.shape)

        except (AttributeError, TypeError):
            # Skip validation during JAX transformations
            pass


def _ensure_config(config: ConfigType) -> ArcEnvConfig:
    """Convert config to typed ArcEnvConfig if needed."""
    if isinstance(config, DictConfig):
        return ArcEnvConfig.from_hydra(config)
    return config


def _validate_and_transform_action(
    action: ActionType, config: ArcEnvConfig, grid_shape: Tuple[int, int]
) -> Dict[str, Any]:
    """Validate and transform action to standard format."""
    if isinstance(action, ARCLEAction):
        # Convert ARCLEAction to standard format
        return {
            "selection": action.selection,
            "operation": action.operation,
        }

    if isinstance(action, dict):
        # Validate required fields
        if "selection" not in action:
            raise ValueError("Action must contain 'selection' field")
        if "operation" not in action:
            raise ValueError("Action must contain 'operation' field")

        # Validate and transform selection
        selection = action["selection"]
        if isinstance(selection, jnp.ndarray):
            # Continuous selection - validate shape and values
            if selection.shape != grid_shape:
                raise ValueError(
                    f"Selection shape {selection.shape} doesn't match grid shape {grid_shape}"
                )

            # Convert to boolean selection if needed
            if config.action.action_format == "selection_operation":
                if selection.dtype == jnp.bool_:
                    validated_selection = selection
                else:
                    # Convert continuous to discrete using threshold
                    validated_selection = selection >= config.action.selection_threshold
            else:
                validated_selection = selection
        else:
            raise ValueError(f"Selection must be jnp.ndarray, got {type(selection)}")

        # Validate operation
        operation = action["operation"]
        if isinstance(operation, (int, jnp.integer)):
            operation = jnp.array(operation, dtype=jnp.int32)

        if not isinstance(operation, jnp.ndarray):
            raise ValueError(
                f"Operation must be int or jnp.ndarray, got {type(operation)}"
            )

        # Validate operation range (JAX-compatible)
        if config.action.validate_actions:
            if config.action.clip_invalid_actions:
                # Always clip to valid range - JAX-compatible
                operation = jnp.clip(operation, 0, config.action.num_operations - 1)
            else:
                # For non-clipping validation, we can't check concrete values during tracing
                # So we'll rely on runtime errors from grid_operations if invalid
                pass

        # Check if operation is in allowed list (if restricted)
        if config.action.allowed_operations is not None:
            # For JAX compatibility, we can't do complex validation during tracing
            # Just clip to ensure it's in valid range - the actual restriction
            # should be enforced at the action generation level
            operation = jnp.clip(operation, 0, config.action.num_operations - 1)

        return {
            "selection": validated_selection,
            "operation": operation,
        }

    raise ValueError(f"Action must be dict or ARCLEAction, got {type(action)}")


def _transform_point_action(
    action: Dict[str, Any], grid_shape: Tuple[int, int]
) -> Dict[str, Any]:
    """Transform point-based action to selection-operation format."""
    if "point" not in action:
        raise ValueError("Point action must contain 'point' field")

    point = action["point"]
    if not isinstance(point, (tuple, list, jnp.ndarray)) or len(point) != 2:
        raise ValueError(f"Point must be (row, col) tuple, got {point}")

    row, col = int(point[0]), int(point[1])

    # Create selection mask with single point
    selection = jnp.zeros(grid_shape, dtype=jnp.bool_)
    selection = selection.at[row, col].set(True)

    return {
        "selection": selection,
        "operation": action["operation"],
    }


def _transform_bbox_action(
    action: Dict[str, Any], grid_shape: Tuple[int, int]
) -> Dict[str, Any]:
    """Transform bbox-based action to selection-operation format."""
    if "bbox" not in action:
        raise ValueError("Bbox action must contain 'bbox' field")

    bbox = action["bbox"]
    if not isinstance(bbox, (tuple, list, jnp.ndarray)) or len(bbox) != 4:
        raise ValueError(f"Bbox must be (row1, col1, row2, col2) tuple, got {bbox}")

    row1, col1, row2, col2 = map(int, bbox)

    # Ensure valid bbox
    row1, row2 = min(row1, row2), max(row1, row2)
    col1, col2 = min(col1, col2), max(col1, col2)

    # Create selection mask for bbox region
    selection = jnp.zeros(grid_shape, dtype=jnp.bool_)
    selection = selection.at[row1 : row2 + 1, col1 : col2 + 1].set(True)

    return {
        "selection": selection,
        "operation": action["operation"],
    }


def _get_observation(state: ArcEnvState, config: ArcEnvConfig) -> jnp.ndarray:
    """Extract observation from state."""
    # For now, just return the working grid
    # Future: Could include additional channels for selection, target, etc.
    return state.working_grid


def _calculate_reward(
    old_state: ArcEnvState, new_state: ArcEnvState, config: ArcEnvConfig
) -> jnp.ndarray:
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


def _is_episode_done(state: ArcEnvState, config: ArcEnvConfig) -> jnp.ndarray:
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
    key: chex.PRNGKey,
    config: ConfigType,
    task_data: JaxArcTask | None = None,
) -> Tuple[ArcEnvState, jnp.ndarray]:
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
) -> Tuple[ArcEnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
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

    # Validate action format
    if not isinstance(action, dict):
        raise ValueError("Action must be a dictionary")

    if "operation" not in action:
        raise ValueError("Action must contain 'operation' field")

    # Check for selection field (either direct or format-specific)
    has_selection = "selection" in action
    has_format_specific = (
        (typed_config.action.action_format == "point" and "point" in action) or
        (typed_config.action.action_format == "bbox" and "bbox" in action)
    )

    if not has_selection and not has_format_specific:
        raise ValueError("Action must contain 'selection' field")

    # Handle action transformation using new handler system
    if isinstance(action, dict) and "selection" in action and "operation" in action:
        # Check if action is already in standardized format (selection mask + operation)
        if (isinstance(action["selection"], jnp.ndarray) and
            len(action["selection"].shape) == 2 and
            action["selection"].shape == state.working_grid_mask.shape):
            # Already standardized format from environment class
            validated_action = action
        else:
            # Validate selection shape if it's a 2D array
            if (isinstance(action["selection"], jnp.ndarray) and
                len(action["selection"].shape) == 2 and
                action["selection"].shape != state.working_grid_mask.shape):
                raise ValueError(f"Selection shape {action['selection'].shape} doesn't match grid shape {state.working_grid_mask.shape}")

            # Transform using appropriate handler
            handler = get_action_handler(typed_config.action.action_format)
            selection_mask = handler(action["selection"], state.working_grid_mask)
            validated_action = {
                "selection": selection_mask,
                "operation": action["operation"]
            }
    else:
        # Legacy action format - transform using appropriate handler
        handler = get_action_handler(typed_config.action.action_format)
        if typed_config.action.action_format == "point":
            if isinstance(action, dict) and "point" in action:
                action_data = jnp.array(action["point"])
                operation = action["operation"]
            else:
                raise ValueError("Point action must be dict with 'point' and 'operation' keys")
        elif typed_config.action.action_format == "bbox":
            if isinstance(action, dict) and "bbox" in action:
                action_data = jnp.array(action["bbox"])
                operation = action["operation"]
            else:
                raise ValueError("Bbox action must be dict with 'bbox' and 'operation' keys")
        else:  # mask or selection_operation
            if isinstance(action, dict) and "selection" in action:
                action_data = action["selection"].flatten()
                operation = action["operation"]
            else:
                raise ValueError("Selection action must be dict with 'selection' and 'operation' keys")

        selection_mask = handler(action_data, state.working_grid_mask)
        validated_action = {
            "selection": selection_mask,
            "operation": operation
        }

    # Update selection in state
    state = state.replace(selected=validated_action["selection"])

    # Execute operation using existing grid operations
    new_state = execute_grid_operation(state, validated_action["operation"])

    # Update step count
    new_state = new_state.replace(step_count=state.step_count + 1)

    # Calculate reward
    reward = _calculate_reward(state, new_state, typed_config)

    # Check if episode is done
    done = _is_episode_done(new_state, typed_config)

    # Update episode_done flag
    new_state = new_state.replace(episode_done=done)

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
            validated_action["operation"],
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

    # Optional visualization callback (add before return)
    if typed_config.debug.log_rl_steps:
        # Clear output directory at episode start (step 0)
        if state.step_count == 0 and typed_config.debug.clear_output_dir:
            jax.debug.callback(
                lambda output_dir: _clear_output_directory(output_dir),
                typed_config.debug.rl_steps_output_dir,
            )

        # Save step visualization
        jax.debug.callback(
            save_rl_step_visualization,
            state,
            validated_action,
            new_state,
            typed_config.debug.rl_steps_output_dir,
        )

    return new_state, observation, reward, done, info


# Convenience functions for common use cases


def arc_reset_with_hydra(
    key: chex.PRNGKey,
    hydra_config: DictConfig,
    task_data: JaxArcTask | None = None,
) -> Tuple[ArcEnvState, jnp.ndarray]:
    """Reset with explicit Hydra config (for type clarity)."""
    return arc_reset(key, hydra_config, task_data)


def arc_step_with_hydra(
    state: ArcEnvState,
    action: ActionType,
    hydra_config: DictConfig,
) -> Tuple[ArcEnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """Step with explicit Hydra config (for type clarity)."""
    return arc_step(state, action, hydra_config)
