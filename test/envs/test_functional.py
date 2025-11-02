"""
Tests for functional API in jaxarc.envs.functional.

This module tests the pure functional environment operations (reset, step)
for JAX compatibility, pure function behavior, and state immutability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc import JaxArcConfig
from jaxarc.envs.actions import create_action
from jaxarc.envs.functional import reset, step
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


class TestFunctionalReset:
    """Test the functional reset function."""

    def test_reset_signature_and_return_type(self, prng_key):
        """Test reset function signature and return type."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env_params = EnvParams.from_config(
            config=config, episode_mode=0, buffer=mock_buffer, subset_indices=None
        )

        # Test function signature
        state, timestep = reset(env_params, prng_key)

        # Verify return type
        assert isinstance(timestep, TimeStep)
        assert timestep.step_type == StepType.FIRST
        assert isinstance(timestep.reward, jax.Array)
        assert isinstance(timestep.discount, jax.Array)
        assert isinstance(timestep.observation, jax.Array)
        assert isinstance(state, State)

    def test_reset_jax_compatibility(self, prng_key):
        """Test that reset function is JAX-compatible."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env_params = EnvParams.from_config(
            config=config, episode_mode=0, buffer=mock_buffer, subset_indices=None
        )

        # Test JIT compilation
        jitted_reset = jax.jit(reset)
        state, timestep = jitted_reset(env_params, prng_key)

        # Verify result is valid
        assert isinstance(timestep, TimeStep)
        assert timestep.step_type == StepType.FIRST


class TestFunctionalStep:
    """Test the functional step function."""

    def test_step_signature_and_return_type(self, prng_key):
        """Test step function signature and return type."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env_params = EnvParams.from_config(
            config=config, episode_mode=0, buffer=mock_buffer, subset_indices=None
        )

        # Get initial timestep
        state, timestep = reset(env_params, prng_key)

        # Create test action
        action = create_action(
            operation=jnp.array(0, dtype=jnp.int32),  # Fill with color 0
            selection=jnp.ones((3, 3), dtype=jnp.bool_),
        )

        # Test step function
        new_state, new_timestep = step(env_params, state, action)
        # Verify return type
        assert isinstance(new_timestep, TimeStep)
        assert isinstance(new_timestep.reward, jax.Array)
        assert isinstance(new_timestep.discount, jax.Array)
        assert isinstance(new_timestep.observation, jax.Array)
        assert isinstance(new_state, State)

    def test_step_jax_compatibility(self, prng_key):
        """Test that step function is JAX-compatible."""
        config = JaxArcConfig()
        mock_buffer = create_mock_buffer()
        env_params = EnvParams.from_config(
            config=config, episode_mode=0, buffer=mock_buffer, subset_indices=None
        )

        state, timestep = reset(env_params, prng_key)
        action = create_action(
            operation=jnp.array(1, dtype=jnp.int32),
            selection=jnp.ones((3, 3), dtype=jnp.bool_),
        )

        # Test JIT compilation
        jitted_step = jax.jit(lambda params, state, action: step(params, state, action))
        new_state, new_timestep = jitted_step(env_params, state, action)
        # Verify result is valid
        assert isinstance(new_timestep, TimeStep)
        assert new_state.step_count == 1
