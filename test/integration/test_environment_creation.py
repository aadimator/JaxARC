"""
Integration tests for environment creation workflows.

This module tests the complete environment setup process from configuration
to ready state, including parser-to-environment data flow and registration
system functionality.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

from jaxarc import JaxArcConfig
from jaxarc.envs import Environment
from jaxarc.registration import available_task_ids, make, register, register_subset
from jaxarc.types import EnvParams


class TestEnvironmentCreation:
    """Test complete environment setup from configuration to ready state."""

    def test_basic_environment_creation(self):
        """Test basic environment creation using registration system."""
        # Create configuration
        config = JaxArcConfig()

        # Get available task IDs
        task_ids = available_task_ids("Mini", config=config)
        assert len(task_ids) > 0, "Should have available Mini tasks"

        # Create environment using registration system
        task_id = task_ids[0]
        env, env_params = make(f"Mini-{task_id}", config=config)

        # Verify environment instance
        assert isinstance(env, Environment)
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

        # Verify environment parameters
        assert isinstance(env_params, EnvParams)
        assert hasattr(env_params, "buffer")
        assert hasattr(env_params, "dataset")
        assert hasattr(env_params, "max_episode_steps")
        assert hasattr(env_params, "reward")

        # Verify buffer is properly initialized
        assert env_params.buffer is not None
        # Buffer is a JaxArcTask with batched arrays
        assert hasattr(env_params.buffer, "input_grids_examples")
        assert (
            env_params.buffer.input_grids_examples.shape[0] > 0
        )  # Should have at least one task

    def test_environment_creation_with_all_datasets(self):
        """Test environment creation works with all supported datasets."""
        config = JaxArcConfig()
        datasets = ["Mini", "Concept", "AGI1", "AGI2"]

        for dataset in datasets:
            try:
                # Get available task IDs
                task_ids = available_task_ids(dataset, config=config)
                if not task_ids:
                    pytest.skip(f"No tasks available for {dataset}")

                # Create environment
                task_id = task_ids[0]
                env, env_params = make(f"{dataset}-{task_id}", config=config)

                # Basic verification
                assert isinstance(env, Environment)
                assert isinstance(env_params, EnvParams)
                assert env_params.buffer is not None

            except Exception as e:
                # Some datasets might not be available in test environment
                pytest.skip(f"Dataset {dataset} not available: {e}")

    def test_environment_creation_with_custom_config(self):
        """Test environment creation with custom configuration parameters."""
        # Create custom configuration
        config = JaxArcConfig()

        # Modify some configuration parameters
        import equinox as eqx

        from jaxarc.configs.environment_config import EnvironmentConfig
        from jaxarc.configs.reward_config import RewardConfig

        # Update environment config
        env_config = EnvironmentConfig(max_episode_steps=50)
        config = eqx.tree_at(lambda c: c.environment, config, env_config)

        # Update reward config
        reward_config = RewardConfig(
            similarity_weight=0.5, success_bonus=2.0, step_penalty=-0.02
        )
        config = eqx.tree_at(lambda c: c.reward, config, reward_config)

        # Create environment with custom config
        task_ids = available_task_ids("Mini", config=config)
        if task_ids:
            task_id = task_ids[0]
            env, env_params = make(f"Mini-{task_id}", config=config)

            # Verify custom configuration is applied
            assert env_params.max_episode_steps == 50
            assert env_params.reward.similarity_weight == 0.5
            assert env_params.reward.success_bonus == 2.0
            assert env_params.reward.step_penalty == -0.02

    def test_environment_creation_with_episode_modes(self):
        """Test environment creation with different episode modes."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)

        if not task_ids:
            pytest.skip("No Mini tasks available")

        task_id = task_ids[0]

        # Test training mode (episode_mode=0)
        env_train, params_train = make(f"Mini-{task_id}", config=config, episode_mode=0)
        assert params_train.episode_mode == 0

        # Test evaluation mode (episode_mode=1)
        env_eval, params_eval = make(f"Mini-{task_id}", config=config, episode_mode=1)
        assert params_eval.episode_mode == 1

        # Both should be valid environments
        assert isinstance(env_train, Environment)
        assert isinstance(env_eval, Environment)
        assert isinstance(params_train, EnvParams)
        assert isinstance(params_eval, EnvParams)

    def test_environment_spaces_initialization(self):
        """Test that environment spaces are properly initialized."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)

        if not task_ids:
            pytest.skip("No Mini tasks available")

        task_id = task_ids[0]
        env, env_params = make(f"Mini-{task_id}", config=config)

        # Test observation space
        obs_space = env.observation_space(env_params)
        assert hasattr(obs_space, "shape")
        assert obs_space.shape[0] > 0
        assert obs_space.shape[1] > 0
        assert obs_space.shape[2] > 0

        # Test action space
        action_space = env.action_space(env_params)
        assert hasattr(action_space, "_spaces")
        assert "operation" in action_space._spaces
        assert "selection" in action_space._spaces

        # Test reward space
        reward_space = env.reward_space(env_params)
        assert hasattr(reward_space, "shape")
        assert hasattr(reward_space, "dtype")
        assert reward_space.shape == ()
        assert reward_space.dtype == jnp.float32

        # Test discount space
        discount_space = env.discount_space(env_params)
        assert hasattr(discount_space, "shape")
        assert hasattr(discount_space, "dtype")
        assert discount_space.shape == ()
        assert discount_space.dtype == jnp.float32


class TestParserEnvironmentIntegration:
    """Test parser-to-environment data flow integration."""

    def test_parser_to_buffer_integration(self):
        """Test that parser data is correctly converted to environment buffer."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)

        if not task_ids:
            pytest.skip("No Mini tasks available")

        # Create environment with single task
        task_id = task_ids[0]
        env, env_params = make(f"Mini-{task_id}", config=config)

        # Verify buffer structure
        buffer = env_params.buffer
        assert hasattr(buffer, "input_grids_examples")
        assert hasattr(buffer, "output_grids_examples")
        assert hasattr(buffer, "num_train_pairs")
        assert hasattr(buffer, "num_test_pairs")

        # Verify task structure - buffer is a single JaxArcTask with batched arrays
        assert buffer.input_grids_examples.shape[0] == 1  # Single task

        # Verify task has required fields
        assert hasattr(buffer, "input_grids_examples")
        assert hasattr(buffer, "output_grids_examples")
        assert hasattr(buffer, "num_train_pairs")
        assert hasattr(buffer, "num_test_pairs")

        # Verify grid shapes are consistent
        input_grids = buffer.input_grids_examples
        output_grids = buffer.output_grids_examples

        # Check that grids have valid shapes
        assert input_grids.shape[-2:] == (
            env_params.dataset.max_grid_height,
            env_params.dataset.max_grid_width,
        )
        assert output_grids.shape[-2:] == (
            env_params.dataset.max_grid_height,
            env_params.dataset.max_grid_width,
        )

    def test_multiple_tasks_buffer_integration(self):
        """Test parser integration with multiple tasks."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)

        if len(task_ids) < 2:
            pytest.skip("Need at least 2 Mini tasks for this test")

        # Create environment with multiple tasks
        selected_ids = task_ids[:3] if len(task_ids) >= 3 else task_ids
        env, env_params = make("Mini-all", config=config)

        # Verify buffer contains multiple tasks
        buffer = env_params.buffer
        assert buffer.input_grids_examples.shape[0] >= len(selected_ids)

        # Verify all tasks have consistent structure
        num_tasks = buffer.input_grids_examples.shape[0]
        for i in range(min(3, num_tasks)):
            # Each task should have valid train and test pairs
            assert buffer.num_train_pairs[i] > 0
            assert buffer.num_test_pairs[i] > 0

    def test_parser_error_handling(self):
        """Test error handling in parser-to-environment integration."""
        config = JaxArcConfig()

        # Test with invalid task ID
        with pytest.raises(ValueError, match="Unknown selector .* for Mini"):
            make("Mini-nonexistent_task_id", config=config)

        # Test with invalid dataset
        with pytest.raises(ValueError, match="not registered"):
            make("InvalidDataset-task", config=config)

    def test_task_data_consistency(self):
        """Test that task data maintains consistency through parser integration."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)

        if not task_ids:
            pytest.skip("No Mini tasks available")

        task_id = task_ids[0]
        env, env_params = make(f"Mini-{task_id}", config=config)

        # Get task from buffer
        buffer = env_params.buffer

        # Verify data types are correct
        chex.assert_type(buffer.input_grids_examples, jnp.int32)
        chex.assert_type(buffer.output_grids_examples, jnp.int32)
        chex.assert_type(buffer.input_masks_examples, jnp.bool_)
        chex.assert_type(buffer.output_masks_examples, jnp.bool_)

        # Verify shapes are consistent
        input_shape = buffer.input_grids_examples.shape
        output_shape = buffer.output_grids_examples.shape
        assert input_shape[-2:] == output_shape[-2:]  # Same grid dimensions

        # Verify masks match grid shapes
        assert buffer.input_masks_examples.shape == input_shape
        assert buffer.output_masks_examples.shape == output_shape

        # Verify values are in valid range (-1 for padding, 0-9 for ARC colors)
        assert jnp.all(buffer.input_grids_examples >= -1)
        assert jnp.all(buffer.input_grids_examples <= 9)
        assert jnp.all(buffer.output_grids_examples >= -1)
        assert jnp.all(buffer.output_grids_examples <= 9)


class TestRegistrationSystem:
    """Test environment registration and factory functions."""

    def test_registration_system_basic(self):
        """Test basic registration system functionality."""
        # Test that default environments are registered
        from jaxarc.registration import _registry

        # Check that basic datasets are registered
        assert "Mini" in _registry._specs
        assert "Concept" in _registry._specs
        assert "AGI1" in _registry._specs
        assert "AGI2" in _registry._specs

    def test_custom_environment_registration(self):
        """Test registering custom environment specifications."""
        # Register a custom environment spec
        register(
            id="TestEnv",
            env_entry="jaxarc.envs:Environment",
            max_episode_steps=200,
            custom_param="test_value",
        )

        # Verify registration
        from jaxarc.registration import _registry

        assert "TestEnv" in _registry._specs

        spec = _registry._specs["TestEnv"]
        assert spec.max_episode_steps == 200
        assert spec.kwargs.get("custom_param") == "test_value"

    def test_subset_registration(self):
        """Test named subset registration functionality."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)

        if len(task_ids) < 3:
            pytest.skip("Need at least 3 Mini tasks for subset testing")

        # Register a custom subset
        subset_ids = task_ids[:2]
        register_subset("Mini", "test_subset", subset_ids)

        # Test that subset can be used
        env, env_params = make("Mini-test_subset", config=config)

        # Verify environment was created successfully
        assert isinstance(env, Environment)
        assert isinstance(env_params, EnvParams)

        # Verify buffer contains correct number of tasks
        assert env_params.buffer.input_grids_examples.shape[0] == len(subset_ids)

    def test_factory_function_parameters(self):
        """Test factory function parameter handling."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)

        if not task_ids:
            pytest.skip("No Mini tasks available")

        task_id = task_ids[0]

        # Test with explicit config
        env1, params1 = make(f"Mini-{task_id}", config=config)
        assert isinstance(env1, Environment)
        assert isinstance(params1, EnvParams)

        # Test with episode mode
        env2, params2 = make(f"Mini-{task_id}", config=config, episode_mode=1)
        assert params2.episode_mode == 1

        # Test with auto_download flag
        env3, params3 = make(f"Mini-{task_id}", config=config, auto_download=True)
        assert isinstance(env3, Environment)
        assert isinstance(params3, EnvParams)

    def test_environment_entry_point_override(self):
        """Test overriding environment entry point."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)

        if not task_ids:
            pytest.skip("No Mini tasks available")

        task_id = task_ids[0]

        # Test with explicit environment entry point
        env, params = make(
            f"Mini-{task_id}", config=config, env_entry="jaxarc.envs:Environment"
        )

        assert isinstance(env, Environment)
        assert isinstance(params, EnvParams)

    def test_available_task_ids_functionality(self):
        """Test available_task_ids function across datasets."""
        config = JaxArcConfig()

        # Test Mini dataset
        mini_ids = available_task_ids("Mini", config=config)
        assert isinstance(mini_ids, list)
        if mini_ids:  # Only test if tasks are available
            assert all(isinstance(tid, str) for tid in mini_ids)

        # Test with auto_download
        mini_ids_auto = available_task_ids("Mini", config=config, auto_download=True)
        assert isinstance(mini_ids_auto, list)

        # Results should be consistent
        if mini_ids and mini_ids_auto:
            assert set(mini_ids) == set(mini_ids_auto)

    def test_error_handling_in_registration(self):
        """Test error handling in registration system."""
        config = JaxArcConfig()

        # Test invalid environment ID
        with pytest.raises(ValueError):
            make("NonExistentDataset-task", config=config)

        # Test invalid task selector
        with pytest.raises(ValueError):
            make("Mini-nonexistent_task", config=config)

        # Test invalid subset registration
        with pytest.raises(ValueError):
            register_subset("Mini", "", ["task1", "task2"])  # Empty subset name
