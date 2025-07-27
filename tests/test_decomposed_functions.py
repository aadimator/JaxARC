"""Tests for decomposed functions in functional.py.

This module tests that the decomposed helper functions maintain the same behavior
as the original implementations and are JAX-compliant.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig

from jaxarc.envs.config import JaxArcConfig
from jaxarc.envs.functional import (
    _calculate_reward_and_done,
    _create_initial_state,
    _get_or_create_task_data,
    _initialize_grids,
    _process_action,
    _select_initial_pair,
    _update_state,
    arc_reset,
    arc_step,
)
from jaxarc.state import ArcEnvState
from jaxarc.types import ARCLEAction, JaxArcTask


class TestDecomposedFunctions:
    """Test decomposed helper functions from functional.py."""

    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        return JaxArcConfig(
            dataset=DictConfig({
                "dataset_path": "test/path",
                "task_split": "train",
                "max_grid_height": 5,
                "max_grid_width": 5,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
                "max_train_pairs": 3,
                "max_test_pairs": 1,
            }),
            action=DictConfig({
                "action_format": "mask",
                "validate_actions": True,
                "allow_invalid_actions": False,
            }),
            reward=DictConfig({
                "correct_pixel": 1.0,
                "incorrect_pixel": -0.1,
                "completion_bonus": 10.0,
                "step_penalty": -0.01,
            }),
            environment=DictConfig({
                "max_episode_steps": 100,
                "episode_mode": 0,
            }),
        )

    @pytest.fixture
    def test_task(self):
        """Create a test task."""
        # Create simple 3x3 grids for testing
        input_grid = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        output_grid = jnp.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        
        return JaxArcTask(
            input_grids_examples=jnp.expand_dims(input_grid, 0),
            input_masks_examples=jnp.ones((1, 3, 3), dtype=bool),
            output_grids_examples=jnp.expand_dims(output_grid, 0),
            output_masks_examples=jnp.ones((1, 3, 3), dtype=bool),
            num_train_pairs=1,
            test_input_grids=jnp.expand_dims(input_grid, 0),
            test_input_masks=jnp.ones((1, 3, 3), dtype=bool),
            true_test_output_grids=jnp.expand_dims(output_grid, 0),
            true_test_output_masks=jnp.ones((1, 3, 3), dtype=bool),
            num_test_pairs=1,
            task_index=jnp.array(42, dtype=jnp.int32),  # Scalar task index
        )

    def test_get_or_create_task_data_with_existing_task(self, test_config, test_task):
        """Test _get_or_create_task_data with existing task data."""
        result = _get_or_create_task_data(test_task, test_config)
        
        # Should return the same task
        assert result is test_task
        assert jnp.array_equal(result.input_grids_examples, test_task.input_grids_examples)

    def test_get_or_create_task_data_with_none(self, test_config):
        """Test _get_or_create_task_data with None (creates demo task)."""
        result = _get_or_create_task_data(None, test_config)
        
        # Should create a demo task
        assert isinstance(result, JaxArcTask)
        assert result.num_train_pairs > 0
        assert result.num_test_pairs > 0

    def test_select_initial_pair_train_mode(self, test_task, test_config):
        """Test _select_initial_pair in training mode."""
        key = jax.random.PRNGKey(42)
        
        result = _select_initial_pair(
            key, test_task, episode_mode=0, initial_pair_idx=None, config=test_config
        )
        
        # Should return a valid pair index
        assert isinstance(result, jnp.ndarray)
        assert 0 <= result < test_task.num_train_pairs

    def test_select_initial_pair_test_mode(self, test_task, test_config):
        """Test _select_initial_pair in test mode."""
        key = jax.random.PRNGKey(42)
        
        result = _select_initial_pair(
            key, test_task, episode_mode=1, initial_pair_idx=None, config=test_config
        )
        
        # Should return a valid test pair index
        assert isinstance(result, jnp.ndarray)
        assert 0 <= result < test_task.num_test_pairs

    def test_select_initial_pair_with_specified_index(self, test_task, test_config):
        """Test _select_initial_pair with specified initial pair index."""
        key = jax.random.PRNGKey(42)
        specified_idx = 0
        
        result = _select_initial_pair(
            key, test_task, episode_mode=0, initial_pair_idx=specified_idx, config=test_config
        )
        
        # Should return the specified index
        assert result == specified_idx

    def test_initialize_grids_train_mode(self, test_task, test_config):
        """Test _initialize_grids in training mode."""
        selected_pair_idx = jnp.array(0)
        
        initial_grid, target_grid, initial_mask = _initialize_grids(
            test_task, selected_pair_idx, episode_mode=0, config=test_config
        )
        
        # Should return valid grids and mask
        assert isinstance(initial_grid, jnp.ndarray)
        assert isinstance(target_grid, jnp.ndarray)
        assert isinstance(initial_mask, jnp.ndarray)
        assert initial_grid.shape == target_grid.shape
        assert initial_mask.dtype == bool

    def test_initialize_grids_test_mode(self, test_task, test_config):
        """Test _initialize_grids in test mode."""
        selected_pair_idx = jnp.array(0)
        
        initial_grid, target_grid, initial_mask = _initialize_grids(
            test_task, selected_pair_idx, episode_mode=1, config=test_config
        )
        
        # Should return valid grids and mask
        assert isinstance(initial_grid, jnp.ndarray)
        assert isinstance(target_grid, jnp.ndarray)
        assert isinstance(initial_mask, jnp.ndarray)

    def test_create_initial_state(self, test_task, test_config):
        """Test _create_initial_state creates valid state."""
        initial_grid = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        target_grid = jnp.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        initial_mask = jnp.ones((3, 3), dtype=bool)
        selected_pair_idx = jnp.array(0)
        episode_mode = 0
        
        state = _create_initial_state(
            test_task, initial_grid, target_grid, initial_mask,
            selected_pair_idx, episode_mode, test_config
        )
        
        # Should return valid ArcEnvState
        assert isinstance(state, ArcEnvState)
        assert jnp.array_equal(state.working_grid, initial_grid)
        assert jnp.array_equal(state.target_grid, target_grid)
        assert state.episode_mode == episode_mode
        assert state.current_pair_idx == selected_pair_idx

    def test_process_action_valid_action(self, test_config):
        """Test _process_action with valid action."""
        # Create a simple state with correct field names
        state = ArcEnvState(
            task_data=None,
            working_grid=jnp.zeros((3, 3), dtype=jnp.int32),
            working_grid_mask=jnp.ones((3, 3), dtype=bool),
            target_grid=jnp.ones((3, 3), dtype=jnp.int32),
            target_grid_mask=jnp.ones((3, 3), dtype=bool),
            step_count=jnp.array(0),
            episode_done=jnp.array(False),
            current_example_idx=jnp.array(0),
            selected=jnp.zeros((3, 3), dtype=bool),
            clipboard=jnp.zeros((3, 3), dtype=jnp.int32),
            similarity_score=jnp.array(0.0),
            episode_mode=jnp.array(0),
            available_demo_pairs=jnp.ones(3, dtype=bool),
            available_test_pairs=jnp.ones(1, dtype=bool),
            demo_completion_status=jnp.zeros(3, dtype=bool),
            test_completion_status=jnp.zeros(1, dtype=bool),
            action_history=jnp.zeros((1000, 10)),  # MAX_HISTORY_LENGTH, ACTION_RECORD_FIELDS
            action_history_length=jnp.array(0),
            allowed_operations_mask=jnp.ones(42, dtype=bool),  # NUM_OPERATIONS
        )
        
        action = ARCLEAction(
            operation=1,  # Some valid operation
            selection_mask=jnp.zeros((3, 3), dtype=bool),
            color=1,
        )
        
        new_state, action_record = _process_action(state, action, test_config)
        
        # Should return updated state and action record
        assert isinstance(new_state, ArcEnvState)
        assert isinstance(action_record, dict)
        assert new_state.step_count == state.step_count + 1

    def test_update_state_increments_step(self, test_config):
        """Test _update_state increments step count."""
        old_state = ArcEnvState(
            task_data=None,
            working_grid=jnp.zeros((3, 3), dtype=jnp.int32),
            working_grid_mask=jnp.ones((3, 3), dtype=bool),
            target_grid=jnp.ones((3, 3), dtype=jnp.int32),
            target_grid_mask=jnp.ones((3, 3), dtype=bool),
            step_count=jnp.array(5),
            episode_done=jnp.array(False),
            current_example_idx=jnp.array(0),
            selected=jnp.zeros((3, 3), dtype=bool),
            clipboard=jnp.zeros((3, 3), dtype=jnp.int32),
            similarity_score=jnp.array(0.0),
            episode_mode=jnp.array(0),
            available_demo_pairs=jnp.ones(3, dtype=bool),
            available_test_pairs=jnp.ones(1, dtype=bool),
            demo_completion_status=jnp.zeros(3, dtype=bool),
            test_completion_status=jnp.zeros(1, dtype=bool),
            action_history=jnp.zeros((1000, 10)),
            action_history_length=jnp.array(0),
            allowed_operations_mask=jnp.ones(42, dtype=bool),
        )
        
        new_state = ArcEnvState(
            task_data=None,
            working_grid=jnp.ones((3, 3), dtype=jnp.int32),
            working_grid_mask=jnp.ones((3, 3), dtype=bool),
            target_grid=jnp.ones((3, 3), dtype=jnp.int32),
            target_grid_mask=jnp.ones((3, 3), dtype=bool),
            step_count=jnp.array(5),  # Will be incremented
            episode_done=jnp.array(False),
            current_example_idx=jnp.array(0),
            selected=jnp.zeros((3, 3), dtype=bool),
            clipboard=jnp.zeros((3, 3), dtype=jnp.int32),
            similarity_score=jnp.array(0.0),
            episode_mode=jnp.array(0),
            available_demo_pairs=jnp.ones(3, dtype=bool),
            available_test_pairs=jnp.ones(1, dtype=bool),
            demo_completion_status=jnp.zeros(3, dtype=bool),
            test_completion_status=jnp.zeros(1, dtype=bool),
            action_history=jnp.zeros((1000, 10)),
            action_history_length=jnp.array(0),
            allowed_operations_mask=jnp.ones(42, dtype=bool),
        )
        
        updated_state = _update_state(old_state, new_state, test_config)
        
        # Should increment step count
        assert updated_state.step_count == 6

    def test_calculate_reward_and_done(self, test_config):
        """Test _calculate_reward_and_done calculates rewards correctly."""
        old_state = ArcEnvState(
            task_data=None,
            working_grid=jnp.zeros((3, 3), dtype=jnp.int32),
            working_grid_mask=jnp.ones((3, 3), dtype=bool),
            target_grid=jnp.ones((3, 3), dtype=jnp.int32),
            target_grid_mask=jnp.ones((3, 3), dtype=bool),
            step_count=jnp.array(0),
            episode_done=jnp.array(False),
            current_example_idx=jnp.array(0),
            selected=jnp.zeros((3, 3), dtype=bool),
            clipboard=jnp.zeros((3, 3), dtype=jnp.int32),
            similarity_score=jnp.array(0.0),
            episode_mode=jnp.array(0),
            available_demo_pairs=jnp.ones(3, dtype=bool),
            available_test_pairs=jnp.ones(1, dtype=bool),
            demo_completion_status=jnp.zeros(3, dtype=bool),
            test_completion_status=jnp.zeros(1, dtype=bool),
            action_history=jnp.zeros((1000, 10)),
            action_history_length=jnp.array(0),
            allowed_operations_mask=jnp.ones(42, dtype=bool),
        )
        
        new_state = ArcEnvState(
            task_data=None,
            working_grid=jnp.ones((3, 3), dtype=jnp.int32),  # Matches target
            working_grid_mask=jnp.ones((3, 3), dtype=bool),
            target_grid=jnp.ones((3, 3), dtype=jnp.int32),
            target_grid_mask=jnp.ones((3, 3), dtype=bool),
            step_count=jnp.array(1),
            episode_done=jnp.array(False),
            current_example_idx=jnp.array(0),
            selected=jnp.zeros((3, 3), dtype=bool),
            clipboard=jnp.zeros((3, 3), dtype=jnp.int32),
            similarity_score=jnp.array(0.0),
            episode_mode=jnp.array(0),
            available_demo_pairs=jnp.ones(3, dtype=bool),
            available_test_pairs=jnp.ones(1, dtype=bool),
            demo_completion_status=jnp.zeros(3, dtype=bool),
            test_completion_status=jnp.zeros(1, dtype=bool),
            action_history=jnp.zeros((1000, 10)),
            action_history_length=jnp.array(0),
            allowed_operations_mask=jnp.ones(42, dtype=bool),
        )
        
        reward, done = _calculate_reward_and_done(old_state, new_state, test_config)
        
        # Should return positive reward for matching target
        assert isinstance(reward, (float, jnp.ndarray))
        assert isinstance(done, (bool, jnp.ndarray))
        assert reward > 0  # Should be positive for correct pixels


class TestJAXCompliance:
    """Test JAX compliance of decomposed functions."""

    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        return JaxArcConfig(
            dataset=DictConfig({
                "dataset_path": "test/path",
                "task_split": "train",
                "max_grid_height": 5,
                "max_grid_width": 5,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
                "max_train_pairs": 3,
                "max_test_pairs": 1,
            }),
            action=DictConfig({
                "action_format": "mask",
                "validate_actions": True,
                "allow_invalid_actions": False,
            }),
            reward=DictConfig({
                "correct_pixel": 1.0,
                "incorrect_pixel": -0.1,
                "completion_bonus": 10.0,
                "step_penalty": -0.01,
            }),
            environment=DictConfig({
                "max_episode_steps": 100,
                "episode_mode": 0,
            }),
        )

    def test_select_initial_pair_jit_compilation(self, test_config):
        """Test that _select_initial_pair can be JIT compiled."""
        from jaxarc.types import JaxArcTask
        
        # Create a simple task
        task = JaxArcTask(
            input_grids_examples=jnp.zeros((2, 3, 3), dtype=jnp.int32),
            input_masks_examples=jnp.ones((2, 3, 3), dtype=bool),
            output_grids_examples=jnp.ones((2, 3, 3), dtype=jnp.int32),
            output_masks_examples=jnp.ones((2, 3, 3), dtype=bool),
            num_train_pairs=2,
            test_input_grids=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, 3, 3), dtype=bool),
            true_test_output_grids=jnp.ones((1, 3, 3), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, 3, 3), dtype=bool),
            num_test_pairs=1,
            task_index=jnp.array(42, dtype=jnp.int32),  # Scalar task index
        )
        
        # JIT compile the function
        jit_select_pair = jax.jit(_select_initial_pair, static_argnames=['episode_mode', 'config'])
        
        key = jax.random.PRNGKey(42)
        result = jit_select_pair(key, task, episode_mode=0, initial_pair_idx=None, config=test_config)
        
        assert isinstance(result, jnp.ndarray)
        assert 0 <= result < task.num_train_pairs

    def test_initialize_grids_jit_compilation(self, test_config):
        """Test that _initialize_grids can be JIT compiled."""
        from jaxarc.types import JaxArcTask
        
        # Create a simple task
        task = JaxArcTask(
            input_grids_examples=jnp.zeros((2, 3, 3), dtype=jnp.int32),
            input_masks_examples=jnp.ones((2, 3, 3), dtype=bool),
            output_grids_examples=jnp.ones((2, 3, 3), dtype=jnp.int32),
            output_masks_examples=jnp.ones((2, 3, 3), dtype=bool),
            num_train_pairs=2,
            test_input_grids=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, 3, 3), dtype=bool),
            true_test_output_grids=jnp.ones((1, 3, 3), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, 3, 3), dtype=bool),
            num_test_pairs=1,
            task_index=jnp.array(42, dtype=jnp.int32),  # Scalar task index
        )
        
        # JIT compile the function
        jit_init_grids = jax.jit(_initialize_grids, static_argnames=['episode_mode', 'config'])
        
        selected_pair_idx = jnp.array(0)
        initial_grid, target_grid, initial_mask = jit_init_grids(
            task, selected_pair_idx, episode_mode=0, config=test_config
        )
        
        assert isinstance(initial_grid, jnp.ndarray)
        assert isinstance(target_grid, jnp.ndarray)
        assert isinstance(initial_mask, jnp.ndarray)

    def test_calculate_reward_and_done_jit_compilation(self, test_config):
        """Test that _calculate_reward_and_done can be JIT compiled."""
        # Create simple states
        old_state = ArcEnvState(
            task_data=None,
            working_grid=jnp.zeros((3, 3), dtype=jnp.int32),
            working_grid_mask=jnp.ones((3, 3), dtype=bool),
            target_grid=jnp.ones((3, 3), dtype=jnp.int32),
            target_grid_mask=jnp.ones((3, 3), dtype=bool),
            step_count=jnp.array(0),
            episode_done=jnp.array(False),
            current_example_idx=jnp.array(0),
            selected=jnp.zeros((3, 3), dtype=bool),
            clipboard=jnp.zeros((3, 3), dtype=jnp.int32),
            similarity_score=jnp.array(0.0),
            episode_mode=jnp.array(0),
            available_demo_pairs=jnp.ones(3, dtype=bool),
            available_test_pairs=jnp.ones(1, dtype=bool),
            demo_completion_status=jnp.zeros(3, dtype=bool),
            test_completion_status=jnp.zeros(1, dtype=bool),
            action_history=jnp.zeros((1000, 10)),
            action_history_length=jnp.array(0),
            allowed_operations_mask=jnp.ones(42, dtype=bool),
        )
        
        import equinox as eqx
        new_state = eqx.tree_at(lambda s: s.step_count, old_state, jnp.array(1))
        
        # JIT compile the function
        jit_calc_reward = jax.jit(_calculate_reward_and_done, static_argnames=['config'])
        
        reward, done = jit_calc_reward(old_state, new_state, config=test_config)
        
        assert isinstance(reward, (float, jnp.ndarray))
        assert isinstance(done, (bool, jnp.ndarray))

    def test_decomposed_functions_maintain_behavior(self, test_config):
        """Test that decomposed functions maintain the same behavior as original."""
        # This is an integration test to ensure the decomposed functions
        # work together the same way as the original monolithic functions
        
        key = jax.random.PRNGKey(42)
        
        # Test arc_reset (which uses decomposed functions internally)
        state, observation = arc_reset(key, test_config)
        
        assert isinstance(state, ArcEnvState)
        assert isinstance(observation, jnp.ndarray)
        assert state.step_count == 0
        assert not state.episode_done
        
        # Test arc_step (which uses decomposed functions internally)
        action = ARCLEAction(
            operation=1,
            selection_mask=jnp.zeros_like(state.working_grid, dtype=bool),
            color=1,
        )
        
        new_state, new_observation, reward, done, info = arc_step(
            state, action, test_config
        )
        
        assert isinstance(new_state, ArcEnvState)
        assert isinstance(new_observation, jnp.ndarray)
        assert isinstance(reward, (float, jnp.ndarray))
        assert isinstance(done, (bool, jnp.ndarray))
        assert isinstance(info, dict)
        assert new_state.step_count == state.step_count + 1