"""
Tests for ARC Base Environment.

This module tests the ArcEnvironment and ArcEnvState components.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig
from pyprojroot import here

from jaxarc.envs.arc_base import ArcEnvironment, ArcEnvState
from jaxarc.parsers.arc_agi import ArcAgiParser
from jaxarc.types import JaxArcTask


class TestArcEnvState:
    """Test ArcEnvState dataclass functionality."""

    def test_arc_env_state_creation(self):
        """Test basic ArcEnvState creation and validation."""
        # Create dummy task data
        task_data = self._create_dummy_task_data()

        # Create dummy grids
        working_grid = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        working_grid_mask = jnp.array([[True, True], [True, True]], dtype=jnp.bool_)
        target_grid = jnp.array([[5, 6], [7, 8]], dtype=jnp.int32)

        # Create state
        state = ArcEnvState(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            step_count=0,
            episode_done=False,
            current_example_idx=0,
            selected=jnp.zeros((2, 2), dtype=jnp.bool_),
            clipboard=jnp.zeros((2, 2), dtype=jnp.int32),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
        )

        # Verify state fields
        assert state.step_count == 0
        assert state.episode_done is False
        assert state.current_example_idx == 0
        chex.assert_shape(state.working_grid, (2, 2))
        chex.assert_type(state.working_grid, jnp.int32)

    def test_arc_env_state_validation(self):
        """Test ArcEnvState validation catches shape mismatches."""
        task_data = self._create_dummy_task_data()

        working_grid = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        working_grid_mask = jnp.array(
            [[True, True, True]], dtype=jnp.bool_
        )  # Wrong shape
        target_grid = jnp.array([[5, 6], [7, 8]], dtype=jnp.int32)

        # This should raise an error due to shape mismatch
        with pytest.raises((ValueError, AssertionError)):
            ArcEnvState(
                task_data=task_data,
                working_grid=working_grid,
                working_grid_mask=working_grid_mask,
                target_grid=target_grid,
                step_count=0,
                episode_done=False,
                current_example_idx=0,
                selected=jnp.zeros((2, 2), dtype=jnp.bool_),
                clipboard=jnp.zeros((2, 2), dtype=jnp.int32),
                similarity_score=jnp.array(0.0, dtype=jnp.float32),
            )

    def _create_dummy_task_data(self) -> JaxArcTask:
        """Create dummy JaxArcTask for testing."""
        return JaxArcTask(
            input_grids_examples=jnp.zeros((1, 2, 2), dtype=jnp.int32),
            input_masks_examples=jnp.ones((1, 2, 2), dtype=jnp.bool_),
            output_grids_examples=jnp.ones((1, 2, 2), dtype=jnp.int32),
            output_masks_examples=jnp.ones((1, 2, 2), dtype=jnp.bool_),
            num_train_pairs=1,
            test_input_grids=jnp.zeros((1, 2, 2), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, 2, 2), dtype=jnp.bool_),
            true_test_output_grids=jnp.ones((1, 2, 2), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, 2, 2), dtype=jnp.bool_),
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )


class TestArcEnvironment:
    """Test ArcEnvironment functionality."""

    def _split_config(self, config):
        """Helper method to split config into environment and dataset parts."""
        env_config = DictConfig(
            {
                "max_episode_steps": config.get("max_episode_steps", 50),
                "reward_on_submit_only": config.get("reward_on_submit_only", True),
                "step_penalty": config.get("step_penalty", -0.01),
                "success_bonus": config.get("success_bonus", 10.0),
                "mask_threshold": config.get("mask_threshold", 0.5),
                "log_operations": config.get("log_operations", False),
            }
        )

        dataset_config = DictConfig(
            {
                "max_grid_height": config.get("max_grid_height", 10),
                "max_grid_width": config.get("max_grid_width", 10),
                "max_train_pairs": config.get("max_train_pairs", 5),
                "max_test_pairs": config.get("max_test_pairs", 2),
                "training": config.get(
                    "training", {"challenges": "dummy_path", "solutions": "dummy_path"}
                ),
            }
        )

        return env_config, dataset_config

    def test_environment_creation(self):
        """Test basic environment instantiation."""
        config = DictConfig(
            {
                "max_episode_steps": 50,
                "max_grid_height": 10,
                "max_grid_width": 10,
                "max_train_pairs": 5,
                "max_test_pairs": 2,
                "max_action_params": 8,
                "training": {"challenges": "dummy_path", "solutions": "dummy_path"},
            }
        )

        # Split config and create environment
        env_config, dataset_config = self._split_config(config)

        # This might fail due to missing data files, but should test basic instantiation
        try:
            env = ArcEnvironment(env_config, dataset_config)
            assert env.max_episode_steps == 50
            assert env.max_grid_size == (10, 10)
        except (FileNotFoundError, RuntimeError):
            # Expected if data files don't exist
            pytest.skip("Skipping test due to missing data files")

    def test_reset_with_provided_task_data(self):
        """Test reset functionality with provided task data."""
        config = DictConfig(
            {
                "max_episode_steps": 50,
                "max_grid_height": 10,
                "max_grid_width": 10,
                "max_train_pairs": 5,
                "max_test_pairs": 2,
                "max_action_params": 8,
                "training": {"challenges": "dummy_path", "solutions": "dummy_path"},
            }
        )

        # Split config and create environment
        env_config, dataset_config = self._split_config(config)

        # Create dummy environment (this will fail when trying to load real data)
        try:
            env = ArcEnvironment(env_config, dataset_config)
        except (FileNotFoundError, RuntimeError):
            pytest.skip("Skipping test due to missing data files")

        # Create dummy task data
        task_data = JaxArcTask(
            input_grids_examples=jnp.array([[[1, 2], [3, 4]]], dtype=jnp.int32),
            input_masks_examples=jnp.array(
                [[[True, True], [True, True]]], dtype=jnp.bool_
            ),
            output_grids_examples=jnp.array([[[5, 6], [7, 8]]], dtype=jnp.int32),
            output_masks_examples=jnp.array(
                [[[True, True], [True, True]]], dtype=jnp.bool_
            ),
            num_train_pairs=1,
            test_input_grids=jnp.array([[[0, 0], [0, 0]]], dtype=jnp.int32),
            test_input_masks=jnp.array([[[True, True], [True, True]]], dtype=jnp.bool_),
            true_test_output_grids=jnp.array([[[1, 1], [1, 1]]], dtype=jnp.int32),
            true_test_output_masks=jnp.array(
                [[[True, True], [True, True]]], dtype=jnp.bool_
            ),
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Test reset with provided task data
        key = jax.random.PRNGKey(42)
        state, observation = env.reset(key, task_data=task_data)

        # Verify state
        assert isinstance(state, ArcEnvState)
        assert state.step_count == 0
        assert state.episode_done is False
        assert state.current_example_idx == 0

        # Verify observation
        chex.assert_shape(observation, (2, 2))
        chex.assert_type(observation, jnp.int32)

        # Verify grids match expected values
        expected_input = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        expected_target = jnp.array([[5, 6], [7, 8]], dtype=jnp.int32)

        chex.assert_trees_all_equal(state.working_grid, expected_input)
        chex.assert_trees_all_equal(state.target_grid, expected_target)
        chex.assert_trees_all_equal(observation, expected_input)

    def test_step_with_action(self):
        """Test step functionality with grid operations."""
        config = DictConfig(
            {
                "max_episode_steps": 50,
                "max_grid_height": 10,
                "max_grid_width": 10,
                "max_train_pairs": 5,
                "max_test_pairs": 2,
                "max_action_params": 8,
                "training": {"challenges": "dummy_path", "solutions": "dummy_path"},
            }
        )

        # Split config and create environment
        env_config, dataset_config = self._split_config(config)

        try:
            env = ArcEnvironment(env_config, dataset_config)
        except (FileNotFoundError, RuntimeError):
            pytest.skip("Skipping test due to missing data files")

        # Create dummy task data
        task_data = JaxArcTask(
            input_grids_examples=jnp.array([[[1, 2], [3, 4]]], dtype=jnp.int32),
            input_masks_examples=jnp.array(
                [[[True, True], [True, True]]], dtype=jnp.bool_
            ),
            output_grids_examples=jnp.array([[[5, 6], [7, 8]]], dtype=jnp.int32),
            output_masks_examples=jnp.array(
                [[[True, True], [True, True]]], dtype=jnp.bool_
            ),
            num_train_pairs=1,
            test_input_grids=jnp.array([[[0, 0], [0, 0]]], dtype=jnp.int32),
            test_input_masks=jnp.array([[[True, True], [True, True]]], dtype=jnp.bool_),
            true_test_output_grids=jnp.array([[[1, 1], [1, 1]]], dtype=jnp.int32),
            true_test_output_masks=jnp.array(
                [[[True, True], [True, True]]], dtype=jnp.bool_
            ),
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Reset environment
        key = jax.random.PRNGKey(42)
        state, observation = env.reset(key, task_data=task_data)

        # Create action
        selection = jnp.zeros((2, 2), dtype=jnp.bool_)
        selection = selection.at[0, 0].set(True)

        action = {
            "selection": selection,
            "operation": jnp.array(2, dtype=jnp.int32),  # Fill with color 2
        }

        # Execute step (no key parameter needed)
        new_state, new_observation, reward, done, info = env.step(state, action)

        # Verify step executed
        assert isinstance(new_state, ArcEnvState)
        assert new_state.step_count == 1
        assert isinstance(reward, (float, jnp.ndarray))
        assert isinstance(done, (bool, jnp.bool_, jnp.ndarray))
        assert isinstance(info, dict)

        # Check that action was executed (selected cell should be filled with color 2)
        assert new_state.working_grid[0, 0] == 2

    def test_jax_step_compilation(self):
        """Test that step can be JIT compiled."""
        config = DictConfig(
            {
                "max_episode_steps": 50,
                "max_grid_height": 5,
                "max_grid_width": 5,
                "max_train_pairs": 5,
                "max_test_pairs": 2,
                "max_action_params": 8,
                "training": {"challenges": "dummy_path", "solutions": "dummy_path"},
            }
        )

        # Split config and create environment
        env_config, dataset_config = self._split_config(config)

        try:
            env = ArcEnvironment(env_config, dataset_config)
        except (FileNotFoundError, RuntimeError):
            pytest.skip("Skipping test due to missing data files")

        # Create dummy task data
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((1, 5, 5), dtype=jnp.int32),
            input_masks_examples=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            output_grids_examples=jnp.ones((1, 5, 5), dtype=jnp.int32),
            output_masks_examples=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            num_train_pairs=1,
            test_input_grids=jnp.zeros((1, 5, 5), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            true_test_output_grids=jnp.ones((1, 5, 5), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Get initial state
        key = jax.random.PRNGKey(42)
        state, _ = env.reset(key, task_data=task_data)

        # Create a function that can be JIT compiled
        def step_fn(state, selection, operation):
            action = {"selection": selection, "operation": operation}
            return env.step(state, action)

        # This should compile without errors
        jit_step = jax.jit(step_fn)

        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        operation = jnp.array(1, dtype=jnp.int32)

        new_state, observation, reward, done, info = jit_step(
            state, selection, operation
        )

        # Verify it still works
        assert isinstance(new_state, ArcEnvState)
        chex.assert_shape(observation, (5, 5))

    def test_jax_compilation(self):
        """Test that reset can be JIT compiled."""
        config = DictConfig(
            {
                "max_episode_steps": 50,
                "max_grid_height": 5,
                "max_grid_width": 5,
                "max_train_pairs": 5,
                "max_test_pairs": 2,
                "max_action_params": 8,
                "training": {"challenges": "dummy_path", "solutions": "dummy_path"},
            }
        )

        # Split config and create environment
        env_config, dataset_config = self._split_config(config)

        try:
            env = ArcEnvironment(env_config, dataset_config)
        except (FileNotFoundError, RuntimeError):
            pytest.skip("Skipping test due to missing data files")

        # Create dummy task data
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((1, 5, 5), dtype=jnp.int32),
            input_masks_examples=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            output_grids_examples=jnp.ones((1, 5, 5), dtype=jnp.int32),
            output_masks_examples=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            num_train_pairs=1,
            test_input_grids=jnp.zeros((1, 5, 5), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            true_test_output_grids=jnp.ones((1, 5, 5), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Create a function that can be JIT compiled
        def reset_fn(key):
            return env.reset(key, task_data=task_data)

        # This should compile without errors
        jit_reset = jax.jit(reset_fn)

        key = jax.random.PRNGKey(42)
        state, observation = jit_reset(key)

        # Verify it still works
        assert isinstance(state, ArcEnvState)
        chex.assert_shape(observation, (5, 5))

    def test_with_real_config(self):
        """Test with real project configuration structure."""

        from hydra import compose, initialize_config_dir

        # Get config directory path
        config_dir = here() / "conf"

        # Skip if config directory doesn't exist
        if not config_dir.exists():
            pytest.skip("Config directory not found")

        try:
            # Initialize Hydra with the project's config directory
            with initialize_config_dir(
                config_dir=str(config_dir.absolute()), version_base=None
            ):
                # Load the actual project configuration
                cfg = compose(config_name="config")

                # Extract dataset config for the environment
                dataset_cfg = cfg.dataset

                # Extract environment config from the full config
                env_cfg = cfg.environment

                # Try to create environment with real config
                try:
                    env = ArcEnvironment(env_cfg, dataset_cfg)

                    # If we get here, environment creation worked
                    assert env.max_grid_size == (
                        dataset_cfg.grid.max_grid_height,
                        dataset_cfg.grid.max_grid_width,
                    )
                    assert hasattr(env, "task_parser")
                    assert isinstance(env.task_parser, ArcAgiParser)

                    # Try to test reset with random task if data is available
                    try:
                        key = jax.random.PRNGKey(42)
                        state, observation = env.reset(key)

                        # If we get here, data loading worked too
                        assert isinstance(state, ArcEnvState)
                        assert state.step_count == 0
                        assert state.episode_done is False
                        chex.assert_rank(observation, 2)

                    except (FileNotFoundError, RuntimeError):
                        # Data files not available, but environment creation worked
                        pytest.skip(
                            "Real ARC data files not available, but config integration works"
                        )

                except (FileNotFoundError, RuntimeError):
                    # Expected if data files don't exist
                    pytest.skip("Skipping real config test due to missing data files")

        except ImportError:
            pytest.skip("Hydra not available for config testing")
