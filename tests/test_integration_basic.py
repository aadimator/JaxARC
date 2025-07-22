"""
Basic integration tests for JaxARC using current API.

This module tests the integration between parsers, environment, and configuration
using the current Hydra-based configuration system.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc.envs import ArcEnvironment
from jaxarc.envs.config_factory import ConfigFactory
from jaxarc.state import ArcEnvState
from jaxarc.types import JaxArcTask
from jaxarc.utils.config import get_config


class TestBasicIntegration:
    """Test basic integration between components."""

    def test_config_to_environment_integration(self):
        """Test that Hydra config can create a working environment."""
        # Get current configuration
        cfg = get_config()

        # Create environment config using the correct factory
        from jaxarc.envs.config_factory import ConfigFactory

        env_config = ConfigFactory.from_hydra(cfg)

        # Create environment
        env = ArcEnvironment(env_config)

        # Test basic properties
        assert env.config is not None
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

    def test_simple_environment_workflow(self):
        """Test a simple environment workflow without complex task creation."""
        # Get configuration and create environment
        cfg = get_config()
        env_config = ConfigFactory.from_hydra(cfg)

        # Disable all logging to avoid callback issues
        import equinox as eqx

        env_config = eqx.tree_at(
            lambda c: (
                c.logging.log_operations,
                c.logging.log_grid_changes,
                c.logging.log_rewards,
            ),
            env_config,
            (False, False, False),
        )

        env = ArcEnvironment(env_config)

        # Test that environment can be created and has expected attributes
        assert isinstance(env, ArcEnvironment)
        assert hasattr(env, "config")
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

        # Test configuration is properly set
        assert env.config is not None

    def test_parser_to_environment_workflow(self):
        """Test complete workflow from parser to environment execution."""
        # Create a simple mock task for testing
        train_input = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        train_output = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)
        test_input = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)

        # Create task with proper padding
        max_height, max_width = 30, 30
        max_train_pairs, max_test_pairs = 10, 4

        # Pad grids to max dimensions
        padded_train_input = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        padded_train_input = padded_train_input.at[:2, :2].set(train_input)

        padded_train_output = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        padded_train_output = padded_train_output.at[:2, :2].set(train_output)

        padded_test_input = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        padded_test_input = padded_test_input.at[:2, :2].set(test_input)

        # Create masks
        train_input_mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)
        train_input_mask = train_input_mask.at[:2, :2].set(True)

        train_output_mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)
        train_output_mask = train_output_mask.at[:2, :2].set(True)

        test_input_mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)
        test_input_mask = test_input_mask.at[:2, :2].set(True)

        # Create arrays with proper dimensions
        input_grids = jnp.zeros(
            (max_train_pairs, max_height, max_width), dtype=jnp.int32
        )
        input_grids = input_grids.at[0].set(padded_train_input)

        input_masks = jnp.zeros(
            (max_train_pairs, max_height, max_width), dtype=jnp.bool_
        )
        input_masks = input_masks.at[0].set(train_input_mask)

        output_grids = jnp.zeros(
            (max_train_pairs, max_height, max_width), dtype=jnp.int32
        )
        output_grids = output_grids.at[0].set(padded_train_output)

        output_masks = jnp.zeros(
            (max_train_pairs, max_height, max_width), dtype=jnp.bool_
        )
        output_masks = output_masks.at[0].set(train_output_mask)

        test_input_grids = jnp.zeros(
            (max_test_pairs, max_height, max_width), dtype=jnp.int32
        )
        test_input_grids = test_input_grids.at[0].set(padded_test_input)

        test_input_masks = jnp.zeros(
            (max_test_pairs, max_height, max_width), dtype=jnp.bool_
        )
        test_input_masks = test_input_masks.at[0].set(test_input_mask)

        test_output_grids = jnp.zeros(
            (max_test_pairs, max_height, max_width), dtype=jnp.int32
        )
        test_output_masks = jnp.zeros(
            (max_test_pairs, max_height, max_width), dtype=jnp.bool_
        )

        task = JaxArcTask(
            input_grids_examples=input_grids,
            input_masks_examples=input_masks,
            output_grids_examples=output_grids,
            output_masks_examples=output_masks,
            num_train_pairs=1,
            test_input_grids=test_input_grids,
            test_input_masks=test_input_masks,
            true_test_output_grids=test_output_grids,
            true_test_output_masks=test_output_masks,
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Get configuration and create environment
        cfg = get_config()
        env_config = ConfigFactory.from_hydra(cfg)

        # Disable logging to avoid visualization callback issues
        import equinox as eqx

        env_config = eqx.tree_at(
            lambda c: (
                c.logging.log_operations,
                c.logging.log_grid_changes,
                c.logging.log_rewards,
            ),
            env_config,
            (False, False, False),
        )

        env = ArcEnvironment(env_config)

        # Reset with task
        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key, task_data=task)

        # Verify state
        assert isinstance(state, ArcEnvState)
        assert state.task_data is not None

        # Take a simple action
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        mask = mask.at[0, 0].set(True)
        action = {
            "selection": mask,
            "operation": 1,  # Fill with blue
        }

        # Step environment
        new_state, new_obs, reward, info = env.step(action)

        # Verify step results
        assert isinstance(new_state, ArcEnvState)
        assert isinstance(reward, (int, float, jnp.ndarray))
        assert isinstance(info, dict)

    def test_jax_transformations_integration(self):
        """Test that the integrated system works with JAX transformations."""
        # Get configuration and create environment
        cfg = get_config()
        env_config = ConfigFactory.from_hydra(cfg)

        # Disable logging to avoid visualization callback issues
        import equinox as eqx

        env_config = eqx.tree_at(
            lambda c: (
                c.logging.log_operations,
                c.logging.log_grid_changes,
                c.logging.log_rewards,
                c.visualization.enabled,
            ),
            env_config,
            (False, False, False, False),
        )

        env = ArcEnvironment(env_config)

        # Create simple task
        train_input = jnp.array([[0, 1]], dtype=jnp.int32)
        train_output = jnp.array([[1, 0]], dtype=jnp.int32)
        test_input = jnp.array([[0, 1]], dtype=jnp.int32)

        # Create task with proper padding (reuse the same logic)
        max_height, max_width = 30, 30
        max_train_pairs, max_test_pairs = 10, 4

        # Pad grids to max dimensions
        padded_train_input = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        padded_train_input = padded_train_input.at[:1, :2].set(train_input)

        padded_train_output = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        padded_train_output = padded_train_output.at[:1, :2].set(train_output)

        padded_test_input = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        padded_test_input = padded_test_input.at[:1, :2].set(test_input)

        # Create masks
        train_input_mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)
        train_input_mask = train_input_mask.at[:1, :2].set(True)

        train_output_mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)
        train_output_mask = train_output_mask.at[:1, :2].set(True)

        test_input_mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)
        test_input_mask = test_input_mask.at[:1, :2].set(True)

        # Create arrays with proper dimensions
        input_grids = jnp.zeros(
            (max_train_pairs, max_height, max_width), dtype=jnp.int32
        )
        input_grids = input_grids.at[0].set(padded_train_input)

        input_masks = jnp.zeros(
            (max_train_pairs, max_height, max_width), dtype=jnp.bool_
        )
        input_masks = input_masks.at[0].set(train_input_mask)

        output_grids = jnp.zeros(
            (max_train_pairs, max_height, max_width), dtype=jnp.int32
        )
        output_grids = output_grids.at[0].set(padded_train_output)

        output_masks = jnp.zeros(
            (max_train_pairs, max_height, max_width), dtype=jnp.bool_
        )
        output_masks = output_masks.at[0].set(train_output_mask)

        test_input_grids = jnp.zeros(
            (max_test_pairs, max_height, max_width), dtype=jnp.int32
        )
        test_input_grids = test_input_grids.at[0].set(padded_test_input)

        test_input_masks = jnp.zeros(
            (max_test_pairs, max_height, max_width), dtype=jnp.bool_
        )
        test_input_masks = test_input_masks.at[0].set(test_input_mask)

        test_output_grids = jnp.zeros(
            (max_test_pairs, max_height, max_width), dtype=jnp.int32
        )
        test_output_masks = jnp.zeros(
            (max_test_pairs, max_height, max_width), dtype=jnp.bool_
        )

        task = JaxArcTask(
            input_grids_examples=input_grids,
            input_masks_examples=input_masks,
            output_grids_examples=output_grids,
            output_masks_examples=output_masks,
            num_train_pairs=1,
            test_input_grids=test_input_grids,
            test_input_masks=test_input_masks,
            true_test_output_grids=test_output_grids,
            true_test_output_masks=test_output_masks,
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Test JIT compilation using functional API directly
        from jaxarc.envs.functional import arc_reset, arc_step

        @jax.jit
        def jit_reset(key):
            return arc_reset(key, env_config, task)

        @jax.jit
        def jit_step(state, action):
            return arc_step(state, action, env_config)

        key = jax.random.PRNGKey(42)
        state, obs = jit_reset(key)

        # Verify JIT worked
        assert isinstance(state, ArcEnvState)

        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        mask = mask.at[0, 0].set(True)
        action = {"selection": mask, "operation": 1}

        new_state, new_obs, reward, done, info = jit_step(state, action)
        assert isinstance(new_state, ArcEnvState)


class TestConfigurationIntegration:
    """Test configuration system integration."""

    def test_hydra_config_loading(self):
        """Test that Hydra configuration loads correctly."""
        cfg = get_config()

        # Test required sections exist
        assert hasattr(cfg, "dataset")
        assert hasattr(cfg, "action")
        assert hasattr(cfg, "environment")
        assert hasattr(cfg, "reward")

        # Test configuration can create environment config
        env_config = ConfigFactory.from_hydra(cfg)
        assert env_config is not None

    def test_config_validation(self):
        """Test that configuration validation works."""
        cfg = get_config()

        # Should not raise for valid config
        env_config = ConfigFactory.from_hydra(cfg)
        env = ArcEnvironment(env_config)

        # Basic validation
        assert env.config.action is not None
        assert env.config.reward is not None
        assert env.config.dataset is not None
