"""
Tests for the ARC environment implementation.

This module contains basic tests for the ARC environment, focusing on
core functionality without complex state manipulation.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.envs import ArcEnvironment
from jaxarc.types import ParsedTaskData


@pytest.fixture
def arc_config():
    """Return a basic configuration for ARC environment tests."""
    return {
        "max_grid_size": [10, 10],
        "max_episode_steps": 20,
        "reward": {
            "reward_on_submit_only": True,
            "similarity_threshold": 0.95,
            "success_bonus": 1.0,
            "step_penalty": 0.0,
        },
    }


@pytest.fixture
def arc_environment(arc_config):
    """Create an ARC environment for testing."""
    return ArcEnvironment(arc_config)


@pytest.fixture
def sample_task_data():
    """Create sample task data for testing."""
    max_h, max_w = 10, 10

    # Create simple input and output grids
    input_grid = jnp.zeros((max_h, max_w), dtype=jnp.int32)
    input_grid = input_grid.at[2:5, 2:5].set(1)  # 3x3 square of 1's

    output_grid = jnp.zeros((max_h, max_w), dtype=jnp.int32)
    output_grid = output_grid.at[2:5, 2:5].set(2)  # Same square but color 2

    return ParsedTaskData(
        input_grids_examples=jnp.expand_dims(input_grid, 0),
        input_masks_examples=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        output_grids_examples=jnp.expand_dims(output_grid, 0),
        output_masks_examples=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        num_train_pairs=1,
        test_input_grids=jnp.expand_dims(input_grid, 0),
        test_input_masks=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        true_test_output_grids=jnp.expand_dims(output_grid, 0),
        true_test_output_masks=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )


def test_environment_init(arc_environment):
    """Test ARC environment initialization."""
    # Check the environment is properly initialized
    assert arc_environment is not None
    assert arc_environment.agents == ["agent_0"]
    assert arc_environment.num_agents == 1

    # Check action and observation spaces
    agent_id = arc_environment.agents[0]
    assert agent_id in arc_environment.action_spaces
    assert agent_id in arc_environment.observation_spaces

    # Check action space structure
    action_space = arc_environment.action_spaces[agent_id]
    assert "selection" in action_space.spaces
    assert "operation" in action_space.spaces

    # Check observation space
    obs_space = arc_environment.observation_spaces[agent_id]
    assert obs_space.shape[0] > 0


def test_environment_reset(arc_environment, sample_task_data):
    """Test ARC environment reset functionality."""
    # Reset the environment
    key = jax.random.PRNGKey(0)
    obs, state = arc_environment.reset(key, sample_task_data)

    # Check observations and state
    assert "agent_0" in obs
    agent_obs = obs["agent_0"]
    chex.assert_shape(agent_obs, arc_environment.observation_spaces["agent_0"].shape)

    # Check state structure
    assert hasattr(state, 'done')
    assert hasattr(state, 'step')
    assert hasattr(state, 'working_grid')
    assert state.step == 0
    assert not state.done

    # Check grids
    chex.assert_rank(state.working_grid, 2)
    chex.assert_rank(state.selected, 2)
    chex.assert_rank(state.clipboard, 2)


def test_environment_step(arc_environment, sample_task_data):
    """Test ARC environment step functionality."""
    # Reset the environment
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs, state = arc_environment.reset(subkey, sample_task_data)

    # Create a simple action
    h, w = state.working_grid.shape
    selection = jnp.zeros((h, w), dtype=jnp.float32)
    selection = selection.at[2:5, 2:5].set(1.0)  # Select the 3x3 square

    action = {
        "agent_0": {
            "selection": selection,
            "operation": jnp.array(1),  # Fill with color 1
        }
    }

    key, subkey = jax.random.split(key)
    obs, next_state, rewards, dones, info = arc_environment.step_env(
        subkey, state, action
    )

    # Check state updates
    assert next_state.step == state.step + 1
    assert "agent_0" in obs
    assert "agent_0" in rewards
    assert "agent_0" in dones


def test_submit_operation(arc_environment, sample_task_data):
    """Test submit operation terminates the episode."""
    # Reset the environment
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs, state = arc_environment.reset(subkey, sample_task_data)

    # Create a submit action
    h, w = state.working_grid.shape
    selection = jnp.zeros((h, w), dtype=jnp.float32)

    submit_action = {
        "agent_0": {
            "selection": selection,
            "operation": jnp.array(34),  # Submit operation
        }
    }

    key, subkey = jax.random.split(key)
    obs, next_state, rewards, dones, info = arc_environment.step_env(
        subkey, state, submit_action
    )

    # Should be done after submit
    assert dones["agent_0"]
    assert dones["__all__"]  # JaxMARL requires __all__ key


def test_jit_compatibility(arc_environment, sample_task_data):
    """Test JIT compatibility of environment functions."""
    # Test JIT compatibility of reset
    jit_reset = jax.jit(lambda key: arc_environment.reset(key, sample_task_data))

    key = jax.random.PRNGKey(0)
    obs, state = jit_reset(key)

    assert "agent_0" in obs
    assert hasattr(state, 'working_grid')

    # Test JIT compatibility of step
    def step_fn(key, state, action):
        return arc_environment.step_env(key, state, action)

    jit_step = jax.jit(step_fn)

    # Create a simple action
    h, w = state.working_grid.shape
    selection = jnp.zeros((h, w), dtype=jnp.float32)
    action = {
        "agent_0": {
            "selection": selection,
            "operation": jnp.array(1),  # Fill color
        }
    }

    key, subkey = jax.random.split(key)
    obs, next_state, rewards, dones, info = jit_step(subkey, state, action)

    assert "agent_0" in obs
    assert "agent_0" in rewards
    assert "agent_0" in dones


def test_multiple_operations(arc_environment, sample_task_data):
    """Test multiple different operations work."""
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    obs, state = arc_environment.reset(subkey, sample_task_data)

    h, w = state.working_grid.shape
    selection = jnp.zeros((h, w), dtype=jnp.float32)
    selection = selection.at[4:6, 4:6].set(1.0)  # Small 2x2 selection

    # Test various operations
    operations_to_test = [0, 1, 2, 10, 20, 24, 28, 31]  # Sample of different operation types

    for op_id in operations_to_test:
        action = {
            "agent_0": {
                "selection": selection,
                "operation": jnp.array(op_id),
            }
        }

        key, subkey = jax.random.split(key)
        obs, next_state, rewards, dones, info = arc_environment.step_env(
            subkey, state, action
        )

        # Basic checks - all operations should work without errors
        assert "agent_0" in obs
        assert "agent_0" in rewards
        assert "agent_0" in dones
        assert next_state.step == state.step + 1

        # Update state for next test (unless we hit a terminal state)
        if not dones["agent_0"]:
            state = next_state
        else:
            # Reset if we terminated
            key, subkey = jax.random.split(key)
            obs, state = arc_environment.reset(subkey, sample_task_data)
