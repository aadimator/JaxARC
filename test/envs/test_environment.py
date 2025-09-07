"""
Tests for Environment class in jaxarc.envs.environment.

This module tests the Environment class initialization, methods,
state transitions, and episode management.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc import JaxArcConfig
from jaxarc.envs.actions import create_action
from jaxarc.envs.environment import Environment
from jaxarc.envs.spaces import ARCActionSpace, BoundedArraySpace, GridSpace
from jaxarc.state import State
from jaxarc.types import EnvParams, JaxArcTask, StepType, TimeStep
from jaxarc.utils.buffer import stack_task_list


def create_mock_buffer():
    """Create a minimal mock buffer for testing."""
    # Create simple 3x3 grids for testing
    grid_shape = (3, 3)

    # Single task with one training pair
    input_grid = jnp.array([[1, 0, 2], [0, 1, 0], [2, 0, 1]], dtype=jnp.int32)
    output_grid = jnp.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]], dtype=jnp.int32)
    mask = jnp.ones(grid_shape, dtype=jnp.bool_)

    # Create a proper JaxArcTask first
    task = JaxArcTask(
        input_grids_examples=input_grid[None, ...],  # Add pair dim
        input_masks_examples=mask[None, ...],
        output_grids_examples=output_grid[None, ...],
        output_masks_examples=mask[None, ...],
        num_train_pairs=jnp.array(1, dtype=jnp.int32),  # JAX int32 scalar
        test_input_grids=input_grid[None, ...],
        test_input_masks=mask[None, ...],
        true_test_output_grids=output_grid[None, ...],
        true_test_output_masks=mask[None, ...],
        num_test_pairs=jnp.array(1, dtype=jnp.int32),  # JAX int32 scalar
        task_index=jnp.array(0, dtype=jnp.int32),  # JAX int32 scalar
    )

    # Create buffer using the proper utility
    return stack_task_list([task])


class TestEnvironmentInitialization:
    """Test Environment class initialization."""

    def test_environment_creation(self):
        """Test that Environment can be created."""
        env = Environment()
        assert env is not None
        assert isinstance(env, Environment)

    def test_default_params(self):
        """Test default_params method."""
        env = Environment()
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()

        env_params = env.default_params(
            config=config, buffer=mock_buffer, episode_mode=0, subset_indices=None
        )

        assert isinstance(env_params, EnvParams)
        assert env_params.episode_mode == 0
        assert env_params.buffer is not None

    def test_observation_shape(self):
        """Test observation_shape method."""
        env = Environment()
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()

        env_params = env.default_params(
            config=config, buffer=mock_buffer, episode_mode=0
        )

        obs_shape = env.observation_shape(env_params)
        assert isinstance(obs_shape, tuple)
        assert len(obs_shape) == 2  # (height, width)
        assert all(isinstance(dim, int) for dim in obs_shape)


class TestEnvironmentMethods:
    """Test Environment class methods."""

    def test_reset_method(self, prng_key):
        """Test Environment reset method."""
        env = Environment()
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()

        env_params = env.default_params(
            config=config, buffer=mock_buffer, episode_mode=0
        )

        # Test reset
        timestep = env.reset(env_params, prng_key)

        # Verify return type
        assert isinstance(timestep, TimeStep)
        assert timestep.step_type == StepType.FIRST
        assert isinstance(timestep.reward, jax.Array)
        assert isinstance(timestep.discount, jax.Array)
        assert isinstance(timestep.observation, jax.Array)
        assert isinstance(timestep.state, State)

    def test_step_method(self, prng_key):
        """Test Environment step method."""
        env = Environment()
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()

        env_params = env.default_params(
            config=config, buffer=mock_buffer, episode_mode=0
        )

        # Get initial timestep
        timestep = env.reset(env_params, prng_key)

        # Create test action
        action = create_action(
            operation=jnp.array(0, dtype=jnp.int32),
            selection=jnp.ones((3, 3), dtype=jnp.bool_),
        )

        # Test step
        new_timestep = env.step(env_params, timestep, action)

        # Verify return type
        assert isinstance(new_timestep, TimeStep)
        assert isinstance(new_timestep.reward, jax.Array)
        assert isinstance(new_timestep.discount, jax.Array)
        assert isinstance(new_timestep.observation, jax.Array)
        assert isinstance(new_timestep.state, State)

        # Verify state progression
        assert new_timestep.state.step_count == 1
        assert timestep.state.step_count == 0  # Original unchanged


class TestEnvironmentSpaces:
    """Test Environment space methods."""

    def test_observation_space(self):
        """Test observation_space method."""
        env = Environment()
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()

        env_params = env.default_params(
            config=config, buffer=mock_buffer, episode_mode=0
        )

        obs_space = env.observation_space(env_params)
        assert isinstance(obs_space, GridSpace)
        assert hasattr(obs_space, "max_height")
        assert hasattr(obs_space, "max_width")

    def test_action_space(self):
        """Test action_space method."""
        env = Environment()
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()

        env_params = env.default_params(
            config=config, buffer=mock_buffer, episode_mode=0
        )

        action_space = env.action_space(env_params)
        assert isinstance(action_space, ARCActionSpace)
        assert hasattr(action_space, "max_height")
        assert hasattr(action_space, "max_width")

    def test_reward_space(self):
        """Test reward_space method."""
        env = Environment()
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()

        env_params = env.default_params(
            config=config, buffer=mock_buffer, episode_mode=0
        )

        reward_space = env.reward_space(env_params)
        assert isinstance(reward_space, BoundedArraySpace)
        assert reward_space.shape == ()
        assert reward_space.dtype == jnp.float32

    def test_discount_space(self):
        """Test discount_space method."""
        env = Environment()
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()

        env_params = env.default_params(
            config=config, buffer=mock_buffer, episode_mode=0
        )

        discount_space = env.discount_space(env_params)
        assert isinstance(discount_space, BoundedArraySpace)
        assert discount_space.shape == ()
        assert discount_space.dtype == jnp.float32


class TestEnvironmentInterface:
    """Test Environment interface compliance."""

    def test_unwrapped_property(self):
        """Test unwrapped property."""
        env = Environment()
        assert env.unwrapped is env

    def test_close_method(self):
        """Test close method."""
        env = Environment()
        # Should not raise an exception
        env.close()

    def test_episode_workflow(self, prng_key):
        """Test complete episode workflow."""
        env = Environment()
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()

        env_params = env.default_params(
            config=config, buffer=mock_buffer, episode_mode=0
        )

        # Reset
        timestep = env.reset(env_params, prng_key)
        assert timestep.step_type == StepType.FIRST
        assert timestep.state.step_count == 0

        # Take several steps
        for i in range(3):
            action = create_action(
                operation=jnp.array(i, dtype=jnp.int32),
                selection=jnp.ones((3, 3), dtype=jnp.bool_),
            )
            timestep = env.step(env_params, timestep, action)

            # Verify step progression
            assert timestep.state.step_count == i + 1

            # Verify step type progression
            if i < 2:  # Not terminal yet
                assert timestep.step_type == StepType.MID

    def test_state_transitions(self, prng_key):
        """Test state transitions are handled correctly."""
        env = Environment()
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()

        env_params = env.default_params(
            config=config, buffer=mock_buffer, episode_mode=0
        )

        # Reset
        timestep1 = env.reset(env_params, prng_key)
        initial_state = timestep1.state

        # Step
        action = create_action(
            operation=jnp.array(1, dtype=jnp.int32),
            selection=jnp.ones((3, 3), dtype=jnp.bool_),
        )
        timestep2 = env.step(env_params, timestep1, action)
        new_state = timestep2.state

        # Verify state immutability
        assert initial_state.step_count == 0
        assert new_state.step_count == 1
        assert initial_state is not new_state

        # Verify state consistency
        assert initial_state.working_grid.shape == new_state.working_grid.shape
        assert initial_state.target_grid.shape == new_state.target_grid.shape
