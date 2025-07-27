"""
ARC Base Environment - Single-Agent Reinforcement Learning for ARC tasks.

This module provides a clean, single-agent implementation for ARC task solving.
It focuses on learning and iteration over complex architecture, and can be extended
for multi-task or hierarchical reinforcement learning approaches.
"""

from __future__ import annotations

import chex
import equinox as eqx
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from ..parsers.arc_agi import ArcAgiParser
from ..state import ArcEnvState
from ..types import JaxArcTask
from .grid_operations import compute_grid_similarity, execute_grid_operation


def create_enhanced_state(task_data, working_grid, working_grid_mask, target_grid, **kwargs):
    """Helper function to create ArcEnvState with all required enhanced fields."""
    from ..utils.jax_types import (
        DEFAULT_MAX_TRAIN_PAIRS, DEFAULT_MAX_TEST_PAIRS, MAX_HISTORY_LENGTH, 
        ACTION_RECORD_FIELDS, NUM_OPERATIONS
    )
    
    # Default values for enhanced functionality fields
    defaults = {
        'episode_mode': jnp.array(0, dtype=jnp.int32),  # Training mode
        'available_demo_pairs': jnp.array([True] + [False] * (DEFAULT_MAX_TRAIN_PAIRS - 1), dtype=bool),
        'available_test_pairs': jnp.array([True] + [False] * (DEFAULT_MAX_TEST_PAIRS - 1), dtype=bool),
        'demo_completion_status': jnp.zeros(DEFAULT_MAX_TRAIN_PAIRS, dtype=bool),
        'test_completion_status': jnp.zeros(DEFAULT_MAX_TEST_PAIRS, dtype=bool),
        'action_history': jnp.zeros((MAX_HISTORY_LENGTH, ACTION_RECORD_FIELDS), dtype=jnp.float32),
        'action_history_length': jnp.array(0, dtype=jnp.int32),
        'allowed_operations_mask': jnp.ones(NUM_OPERATIONS, dtype=bool),
        'step_count': jnp.array(0, dtype=jnp.int32),
        'episode_done': jnp.array(False, dtype=jnp.bool_),
        'current_example_idx': jnp.array(0, dtype=jnp.int32),
        'selected': jnp.zeros_like(working_grid, dtype=jnp.bool_),
        'clipboard': jnp.zeros_like(working_grid, dtype=jnp.int32),
        'similarity_score': jnp.array(0.0, dtype=jnp.float32),
    }
    
    # Override defaults with any provided kwargs
    defaults.update(kwargs)
    
    return ArcEnvState(
        task_data=task_data,
        working_grid=working_grid,
        working_grid_mask=working_grid_mask,
        target_grid=target_grid,
        target_grid_mask=working_grid_mask,  # Same mask as working grid
        **defaults
    )


class ArcEnvironment:
    """Clean ARC environment implementation for single-agent reinforcement learning."""

    def __init__(self, env_config: DictConfig, dataset_config: DictConfig):
        """Initialize ARC environment with configuration.

        Args:
            env_config: Environment-specific configuration (rewards, episode settings, etc.)
            dataset_config: Dataset configuration (grid dimensions, parser settings, etc.)
        """

        self.env_config = env_config
        self.dataset_config = dataset_config

        # Extract environment settings from env_config
        self.max_episode_steps = env_config.get("max_episode_steps", 100)

        # Extract dataset settings from dataset_config
        self.max_grid_height = dataset_config.get("max_grid_height", 30)
        self.max_grid_width = dataset_config.get("max_grid_width", 30)
        self.max_grid_size = (self.max_grid_height, self.max_grid_width)

        # Reward settings from env_config
        self.reward_on_submit_only = env_config.get("reward_on_submit_only", True)
        self.step_penalty = env_config.get("step_penalty", -0.01)
        self.success_bonus = env_config.get("success_bonus", 10.0)

        # Debug settings from env_config
        self.log_operations = env_config.get("log_operations", False)

        logger.info("Initializing ARC Environment with separated configs:")
        logger.info(f"Environment config: {OmegaConf.to_yaml(self.env_config)}")
        logger.info(f"Dataset config keys: {list(self.dataset_config.keys())}")
        logger.info(f"Max grid size from dataset: {self.max_grid_size}")
        logger.info(f"Max episode steps from environment: {self.max_episode_steps}")

        try:
            self.task_parser = ArcAgiParser(dataset_config)
            logger.info("Task parser initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize task parser: {e}")
            logger.info("Creating mock task parser for demo purposes")
            self.task_parser = None

    def reset(
        self, key: chex.PRNGKey, task_data: JaxArcTask | None = None
    ) -> tuple[ArcEnvState, jnp.ndarray]:
        """Reset with clean ARC environment state, optionally with specific task."""
        if task_data is None:
            if self.task_parser is not None:
                try:
                    # Default behavior: sample from configured task distribution
                    task_data = self.task_parser.get_random_task(key)
                except RuntimeError as e:
                    logger.warning(f"Failed to get task from parser: {e}")
                    logger.info("Falling back to demo task")
                    task_data = self._create_demo_task()
            else:
                # Create a simple demo task for testing
                task_data = self._create_demo_task()

        # Initialize working grid from first training example
        initial_grid = task_data.input_grids_examples[0]
        target_grid = task_data.output_grids_examples[0]
        initial_mask = task_data.input_masks_examples[0]

        # Calculate initial similarity
        initial_similarity = compute_grid_similarity(initial_grid, target_grid)

        state = create_enhanced_state(
            task_data=task_data,
            working_grid=initial_grid,
            working_grid_mask=initial_mask,
            target_grid=target_grid,
            similarity_score=initial_similarity,
        )

        observation = self._get_observation(state)
        return state, observation

    def step(
        self, state: ArcEnvState, action
    ) -> tuple[ArcEnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """Execute single step with structured action."""
        # Convert structured action to selection mask
        grid_shape = state.working_grid.shape
        selection_mask = action.to_selection_mask(grid_shape)
        
        # Update selection in state using Equinox tree_at for better performance
        state = eqx.tree_at(lambda s: s.selected, state, selection_mask)

        # Execute operation using existing grid_operations
        new_state = execute_grid_operation(state, action.operation)

        # Update step count using Equinox tree_at
        new_state = eqx.tree_at(lambda s: s.step_count, new_state, state.step_count + 1)

        # Calculate reward and check termination
        reward = self._calculate_reward(state, new_state)
        done = self._is_episode_done(new_state)

        # Update episode_done flag using Equinox tree_at
        new_state = eqx.tree_at(lambda s: s.episode_done, new_state, done)

        # Get observation
        observation = self._get_observation(new_state)

        # Create info dict (JAX-compatible)
        info = {
            "success": new_state.similarity_score >= 1.0,
            "similarity": new_state.similarity_score,
            "step_count": new_state.step_count,
        }

        return new_state, observation, reward, done, info

    def _get_observation(self, state: ArcEnvState) -> jnp.ndarray:
        """Extract observation from state - simple version just returns working grid."""
        return state.working_grid

    def _calculate_reward(
        self, old_state: ArcEnvState, new_state: ArcEnvState
    ) -> jnp.ndarray:
        """Calculate reward based on similarity improvement and step penalty."""
        # Reward for similarity improvement
        similarity_improvement = new_state.similarity_score - old_state.similarity_score

        # Apply step penalty
        step_penalty = self.step_penalty

        # Bonus for solving the task (JAX-compatible conditional)
        success_bonus = jnp.where(
            new_state.similarity_score >= 1.0, self.success_bonus, 0.0
        )

        # Calculate full reward
        full_reward = similarity_improvement + step_penalty + success_bonus

        # Only use step penalty if reward_on_submit_only and not episode_done
        submit_only_reward = jnp.array(step_penalty, dtype=jnp.float32)

        # Use JAX-compatible conditional logic
        reward_on_submit_only_flag = jnp.array(
            self.reward_on_submit_only, dtype=jnp.bool_
        )
        should_use_submit_only = reward_on_submit_only_flag & ~new_state.episode_done

        return jnp.where(should_use_submit_only, submit_only_reward, full_reward)

    def _is_episode_done(self, state: ArcEnvState) -> jnp.ndarray:
        """Check if episode should terminate."""
        # Episode ends if:
        # 1. Task is solved (perfect similarity)
        # 2. Maximum steps reached
        # 3. Submit operation was used (operation 34 sets episode_done=True)

        task_solved = state.similarity_score >= 1.0
        max_steps_reached = state.step_count >= self.max_episode_steps
        submitted = state.episode_done  # Set by submit operation

        # JAX-compatible boolean operations
        return task_solved | max_steps_reached | submitted

    def _create_demo_task(self) -> JaxArcTask:
        """Create a simple demo task for testing when no parser is available."""
        from ..types import JaxArcTask

        # Create simple demo grids with white background for better visibility
        grid_shape = (8, 8)
        input_grid = jnp.full(grid_shape, 5, dtype=jnp.int32)  # White background
        input_grid = input_grid.at[2:5, 2:5].set(1)  # Blue square (3x3)

        target_grid = jnp.full(grid_shape, 5, dtype=jnp.int32)  # White background
        target_grid = target_grid.at[2:5, 2:5].set(2)  # Same square in red

        # Create masks (all valid for demo)
        mask = jnp.ones(grid_shape, dtype=jnp.bool_)

        # Pad to max size with -1 for invalid areas
        max_shape = (self.max_grid_height, self.max_grid_width)
        padded_input = jnp.full(max_shape, -1, dtype=jnp.int32)
        padded_target = jnp.full(max_shape, -1, dtype=jnp.int32)
        padded_mask = jnp.zeros(max_shape, dtype=jnp.bool_)

        padded_input = padded_input.at[: grid_shape[0], : grid_shape[1]].set(input_grid)
        padded_target = padded_target.at[: grid_shape[0], : grid_shape[1]].set(
            target_grid
        )
        padded_mask = padded_mask.at[: grid_shape[0], : grid_shape[1]].set(mask)

        logger.info(
            "Created demo task: change blue 3x3 square to red 3x3 square (on white background)"
        )

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
