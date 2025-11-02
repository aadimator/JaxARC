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
from jaxarc.envs.spaces import ARCActionSpace, BoundedArraySpace
from jaxarc.state import State
from jaxarc.types import JaxArcTask, StepType, TimeStep
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
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env = Environment(config, mock_buffer)
        assert env is not None
        assert isinstance(env, Environment)

    def test_observation_shape(self):
        """Test observation_shape method."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env = Environment(config, mock_buffer)

        obs_shape = env.observation_shape()
        assert isinstance(obs_shape, tuple)
        assert len(obs_shape) == 3  # (height, width, 1)
        assert all(isinstance(dim, int) for dim in obs_shape)


class TestEnvironmentMethods:
    """Test Environment class methods."""

    def test_reset_method(self, prng_key):
        """Test Environment reset method."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env = Environment(config, mock_buffer)

        # Test reset
        state, timestep = env.reset(prng_key)

        # Verify return type
        assert isinstance(timestep, TimeStep)
        assert timestep.step_type == StepType.FIRST
        assert isinstance(timestep.reward, jax.Array)
        assert isinstance(timestep.discount, jax.Array)
        assert isinstance(timestep.observation, jax.Array)
        assert isinstance(state, State)

    def test_step_method(self, prng_key):
        """Test Environment step method."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env = Environment(config, mock_buffer)

        # Get initial timestep
        state, timestep = env.reset(prng_key)

        # Create test action
        action = create_action(
            operation=jnp.array(0, dtype=jnp.int32),
            selection=jnp.ones((3, 3), dtype=jnp.bool_),
        )

        # Test step
        new_state, new_timestep = env.step(state, action)

        # Verify return type
        assert isinstance(new_timestep, TimeStep)
        assert isinstance(new_timestep.reward, jax.Array)
        assert isinstance(new_timestep.discount, jax.Array)
        assert isinstance(new_timestep.observation, jax.Array)
        assert isinstance(new_state, State)

        # Verify state progression
        assert new_state.step_count == 1
        assert state.step_count == 0  # Original unchanged


class TestEnvironmentSpaces:
    """Test Environment space methods."""

    def test_observation_space(self):
        """Test observation_space method."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env = Environment(config, mock_buffer)

        obs_space = env.observation_space()
        assert isinstance(obs_space, BoundedArraySpace)
        assert hasattr(obs_space, "shape")
        assert len(obs_space.shape) == 3  # Should be (height, width, 1)
        assert obs_space.shape[-1] == 1

    def test_action_space(self):
        """Test action_space method."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env = Environment(config, mock_buffer)

        action_space = env.action_space()
        assert isinstance(action_space, ARCActionSpace)
        # ARCActionSpace is a DictSpace, so it doesn't have shape directly
        # But it should have the expected structure
        assert "operation" in action_space._spaces
        assert "selection" in action_space._spaces

    def test_reward_space(self):
        """Test reward_space method."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env = Environment(config, mock_buffer)

        reward_space = env.reward_space()
        assert isinstance(reward_space, BoundedArraySpace)
        assert reward_space.shape == ()
        assert reward_space.dtype == jnp.float32

    def test_discount_space(self):
        """Test discount_space method."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env = Environment(config, mock_buffer)

        discount_space = env.discount_space()
        assert isinstance(discount_space, BoundedArraySpace)
        assert discount_space.shape == ()
        assert discount_space.dtype == jnp.float32


class TestEnvironmentInterface:
    """Test Environment interface compliance."""

    def test_unwrapped_property(self):
        """Test unwrapped property."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env = Environment(config, mock_buffer)
        assert env.unwrapped is env

    def test_close_method(self):
        """Test close method."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env = Environment(config, mock_buffer)
        # Should not raise an exception
        env.close()

    def test_episode_workflow(self, prng_key):
        """Test complete episode workflow."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env = Environment(config, mock_buffer)

        # Reset
        state, timestep = env.reset(prng_key)
        assert timestep.step_type == StepType.FIRST
        assert state.step_count == 0

        # Take several steps
        for i in range(3):
            action = create_action(
                operation=jnp.array(i, dtype=jnp.int32),
                selection=jnp.ones((3, 3), dtype=jnp.bool_),
            )
            state, timestep = env.step(state, action)

            # Verify step progression
            assert state.step_count == i + 1

            # Verify step type progression
            if i < 2:  # Not terminal yet
                assert timestep.step_type == StepType.MID

    def test_state_transitions(self, prng_key):
        """Test state transitions are handled correctly."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env = Environment(config, mock_buffer)

        # Reset
        state1, timestep1 = env.reset(prng_key)
        initial_state = state1

        # Step
        action = create_action(
            operation=jnp.array(1, dtype=jnp.int32),
            selection=jnp.ones((3, 3), dtype=jnp.bool_),
        )
        state2, timestep2 = env.step(state1, action)
        new_state = state2

        # Verify state immutability
        assert initial_state.step_count == 0
        assert new_state.step_count == 1
        assert initial_state is not new_state

        # Verify state consistency
        assert initial_state.working_grid.shape == new_state.working_grid.shape
        assert initial_state.target_grid.shape == new_state.target_grid.shape
