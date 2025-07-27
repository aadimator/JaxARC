"""Simplified tests for decomposed functions in functional.py.

This module tests the core decomposed helper functions with minimal setup.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig

from jaxarc.envs.config import JaxArcConfig
from jaxarc.envs.functional import (
    _get_or_create_task_data,
    _select_initial_pair,
    _initialize_grids,
)
from jaxarc.types import JaxArcTask


class TestDecomposedFunctionsSimple:
    """Test decomposed helper functions with minimal setup."""

    @pytest.fixture
    def simple_config(self):
        """Create a simple test configuration."""
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
                "selection_format": "mask",
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
    def simple_task(self):
        """Create a simple test task."""
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

    def test_get_or_create_task_data_with_existing_task(self, simple_config, simple_task):
        """Test _get_or_create_task_data with existing task data."""
        result = _get_or_create_task_data(simple_task, simple_config)
        
        # Should return the same task
        assert result is simple_task
        assert jnp.array_equal(result.input_grids_examples, simple_task.input_grids_examples)

    def test_get_or_create_task_data_with_none(self, simple_config):
        """Test _get_or_create_task_data with None (creates demo task)."""
        result = _get_or_create_task_data(None, simple_config)
        
        # Should create a demo task
        assert isinstance(result, JaxArcTask)
        assert result.num_train_pairs > 0
        assert result.num_test_pairs > 0

    def test_select_initial_pair_train_mode(self, simple_task, simple_config):
        """Test _select_initial_pair in training mode."""
        key = jax.random.PRNGKey(42)
        
        result = _select_initial_pair(
            key, simple_task, episode_mode=0, initial_pair_idx=None, config=simple_config
        )
        
        # Should return a valid pair index
        assert isinstance(result, jnp.ndarray)
        assert 0 <= result < simple_task.num_train_pairs

    def test_select_initial_pair_test_mode(self, simple_task, simple_config):
        """Test _select_initial_pair in test mode."""
        key = jax.random.PRNGKey(42)
        
        result = _select_initial_pair(
            key, simple_task, episode_mode=1, initial_pair_idx=None, config=simple_config
        )
        
        # Should return a valid test pair index
        assert isinstance(result, jnp.ndarray)
        assert 0 <= result < simple_task.num_test_pairs

    def test_select_initial_pair_with_specified_index(self, simple_task, simple_config):
        """Test _select_initial_pair with specified initial pair index."""
        key = jax.random.PRNGKey(42)
        specified_idx = 0
        
        result = _select_initial_pair(
            key, simple_task, episode_mode=0, initial_pair_idx=specified_idx, config=simple_config
        )
        
        # Should return the specified index
        assert result == specified_idx

    def test_initialize_grids_train_mode(self, simple_task, simple_config):
        """Test _initialize_grids in training mode."""
        selected_pair_idx = jnp.array(0)
        
        initial_grid, target_grid, initial_mask = _initialize_grids(
            simple_task, selected_pair_idx, episode_mode=0, config=simple_config
        )
        
        # Should return valid grids and mask
        assert isinstance(initial_grid, jnp.ndarray)
        assert isinstance(target_grid, jnp.ndarray)
        assert isinstance(initial_mask, jnp.ndarray)
        assert initial_grid.shape == target_grid.shape
        assert initial_mask.dtype == bool

    def test_initialize_grids_test_mode(self, simple_task, simple_config):
        """Test _initialize_grids in test mode."""
        selected_pair_idx = jnp.array(0)
        
        initial_grid, target_grid, initial_mask = _initialize_grids(
            simple_task, selected_pair_idx, episode_mode=1, config=simple_config
        )
        
        # Should return valid grids and mask
        assert isinstance(initial_grid, jnp.ndarray)
        assert isinstance(target_grid, jnp.ndarray)
        assert isinstance(initial_mask, jnp.ndarray)


class TestJAXComplianceSimple:
    """Test JAX compliance of decomposed functions with minimal setup."""

    @pytest.fixture
    def simple_config(self):
        """Create a simple test configuration."""
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
                "selection_format": "mask",
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

    def test_select_initial_pair_jit_compilation(self, simple_config):
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
        result = jit_select_pair(key, task, episode_mode=0, initial_pair_idx=None, config=simple_config)
        
        assert isinstance(result, jnp.ndarray)
        assert 0 <= result < task.num_train_pairs

    def test_initialize_grids_jit_compilation(self, simple_config):
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
            task, selected_pair_idx, episode_mode=0, config=simple_config
        )
        
        assert isinstance(initial_grid, jnp.ndarray)
        assert isinstance(target_grid, jnp.ndarray)
        assert isinstance(initial_mask, jnp.ndarray)

    def test_functions_are_pure(self, simple_config):
        """Test that decomposed functions are pure (no side effects)."""
        from jaxarc.types import JaxArcTask
        
        # Create a simple task
        task = JaxArcTask(
            input_grids_examples=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            input_masks_examples=jnp.ones((1, 3, 3), dtype=bool),
            output_grids_examples=jnp.ones((1, 3, 3), dtype=jnp.int32),
            output_masks_examples=jnp.ones((1, 3, 3), dtype=bool),
            num_train_pairs=1,
            test_input_grids=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, 3, 3), dtype=bool),
            true_test_output_grids=jnp.ones((1, 3, 3), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, 3, 3), dtype=bool),
            num_test_pairs=1,
            task_index=jnp.array(42, dtype=jnp.int32),
        )
        
        key = jax.random.PRNGKey(42)
        
        # Call functions multiple times with same inputs
        result1 = _select_initial_pair(key, task, episode_mode=0, initial_pair_idx=None, config=simple_config)
        result2 = _select_initial_pair(key, task, episode_mode=0, initial_pair_idx=None, config=simple_config)
        
        # Should get same results (pure functions)
        assert jnp.array_equal(result1, result2)
        
        # Test grid initialization
        selected_pair_idx = jnp.array(0)
        grids1 = _initialize_grids(task, selected_pair_idx, episode_mode=0, config=simple_config)
        grids2 = _initialize_grids(task, selected_pair_idx, episode_mode=0, config=simple_config)
        
        # Should get same results
        assert jnp.array_equal(grids1[0], grids2[0])  # initial_grid
        assert jnp.array_equal(grids1[1], grids2[1])  # target_grid
        assert jnp.array_equal(grids1[2], grids2[2])  # initial_mask