"""
Tests for the ARCLE environment implementation.

This module contains tests for the ARCLE environment, which implements
the ARCLE approach with JAX optimizations.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxarc.envs import ARCLEEnvironment
from jaxarc.envs.arcle_env import ARCLEState
from jaxarc.envs.arcle_operations import (
    clear_grid,
    copy_to_clipboard,
    execute_arcle_operation,
    fill_color,
    paste_from_clipboard,
)
from jaxarc.types import ParsedTaskData


@pytest.fixture
def arcle_config():
    """Return a basic configuration for ARCLE environment tests."""
    return {
        "max_grid_size": [10, 10],
        "max_episode_steps": 20,
        "reward_on_submit_only": True,
    }


@pytest.fixture
def arcle_environment(arcle_config):
    """Create an ARCLE environment for testing."""
    return ARCLEEnvironment(arcle_config)


@pytest.fixture
def dummy_task_data():
    """Create dummy task data for testing."""
    max_h, max_w = 10, 10

    # Create simple input and output grids
    input_grid = np.zeros((2, max_h, max_w), dtype=np.int32)
    output_grid = np.zeros((2, max_h, max_w), dtype=np.int32)

    # Add some patterns to differentiate input and output
    input_grid[0, 2:5, 2:5] = 1  # 3x3 square of 1's in first training example
    output_grid[0, 3:6, 3:6] = 2  # 3x3 square of 2's in first example (shifted)

    # Different pattern for second example
    input_grid[1, 1:6, 1:3] = 3  # Rectangle of 3's
    output_grid[1, 1:6, 4:6] = 3  # Rectangle of 3's moved right

    return ParsedTaskData(
        input_grids_examples=jnp.array(input_grid),
        input_masks_examples=jnp.ones((2, max_h, max_w), dtype=jnp.bool_),
        output_grids_examples=jnp.array(output_grid),
        output_masks_examples=jnp.ones((2, max_h, max_w), dtype=jnp.bool_),
        num_train_pairs=2,
        test_input_grids=jnp.array(input_grid[:1]),
        test_input_masks=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        true_test_output_grids=jnp.array(output_grid[:1]),
        true_test_output_masks=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )


def test_environment_init(arcle_environment):
    """Test ARCLE environment initialization."""
    # Check the environment is properly initialized
    assert arcle_environment is not None
    assert arcle_environment.agents == ["agent_0"]
    assert arcle_environment.num_agents == 1

    # Check action and observation spaces
    agent_id = arcle_environment.agents[0]
    assert agent_id in arcle_environment.action_spaces
    assert agent_id in arcle_environment.observation_spaces

    # Check action space structure
    action_space = arcle_environment.action_spaces[agent_id]
    assert "selection" in action_space.spaces
    assert "operation" in action_space.spaces

    # Check observation space
    obs_space = arcle_environment.observation_spaces[agent_id]
    assert obs_space.shape[0] > 0


def test_environment_reset(arcle_environment):
    """Test ARCLE environment reset functionality."""
    # Reset the environment
    key = jax.random.PRNGKey(0)
    obs, state = arcle_environment.reset(key)

    # Check observations and state
    assert "agent_0" in obs
    agent_obs = obs["agent_0"]
    chex.assert_shape(agent_obs, arcle_environment.observation_spaces["agent_0"].shape)

    # Check state structure
    assert isinstance(state, ARCLEState)
    assert state.done == jnp.array(False)
    assert state.step == 0
    assert state.step_count == jnp.array(0)
    assert state.terminated == jnp.array(False)

    # Check grids
    chex.assert_rank(state.working_grid, 2)
    input_grid = state.task_data.input_grids_examples[state.active_train_pair_idx]
    target_grid = state.task_data.output_grids_examples[state.active_train_pair_idx]
    chex.assert_rank(input_grid, 2)
    chex.assert_rank(target_grid, 2)
    chex.assert_type(state.working_grid, jnp.int32)

    # Check dimensions
    assert state.grid_dim.shape == (2,)
    assert state.target_dim.shape == (2,)
    assert state.max_grid_dim.shape == (2,)


def test_environment_step(arcle_environment):
    """Test ARCLE environment step functionality with various operations."""
    # Reset the environment
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs, state = arcle_environment.reset(subkey)

    # Create a selection mask (select center 3x3 region)
    h, w = state.grid_dim
    selection = jnp.zeros((h, w), dtype=jnp.float32)
    selection = selection.at[h // 2 - 1 : h // 2 + 2, w // 2 - 1 : w // 2 + 2].set(1.0)

    # Test fill color operation (operation ID 1 = fill with color 1)
    action = {
        "agent_0": {
            "selection": selection,
            "operation": jnp.array(1),  # Fill with color 1
        }
    }

    key, subkey = jax.random.split(key)
    obs, next_state, rewards, dones, info = arcle_environment.step(
        subkey, state, action
    )

    # Check state updates
    assert next_state.step_count == state.step_count + 1
    assert not dones["agent_0"]  # Should not be done after this operation

    # The center 3x3 region should now be color 1
    center_y, center_x = h // 2, w // 2
    center_region = next_state.working_grid[
        center_y - 1 : center_y + 2, center_x - 1 : center_x + 2
    ]
    assert jnp.all(center_region == 1)

    # Test a few more operations
    operations_to_test = [
        # Flood fill
        {"op": 11, "desc": "Flood fill with color 1"},
        # Move object
        {"op": 20, "desc": "Move object up"},
        # Rotate object
        {"op": 24, "desc": "Rotate object 90°"},
        # Copy and paste
        {"op": 29, "desc": "Copy to clipboard"},
        {"op": 30, "desc": "Paste from clipboard"},
        # Clear grid
        {"op": 31, "desc": "Clear grid"},
    ]

    for op_info in operations_to_test:
        action = {
            "agent_0": {
                "selection": selection,
                "operation": jnp.array(op_info["op"]),
            }
        }

        key, subkey = jax.random.split(key)
        obs, next_state, rewards, dones, info = arcle_environment.step(
            subkey, state, action
        )

        # Basic checks for each operation
        assert next_state.step_count == state.step_count + 1
        assert not dones["agent_0"]  # Should not be done after these operations

        # Update state for next iteration
        state = next_state

    # Test submit operation (34)
    action = {
        "agent_0": {
            "selection": selection,
            "operation": jnp.array(34),  # Submit
        }
    }

    key, subkey = jax.random.split(key)
    obs, next_state, rewards, dones, info = arcle_environment.step(
        subkey, state, action
    )

    # Should be done after submit
    assert dones["agent_0"]
    assert dones["__all__"]  # JaxMARL requires __all__ key
    # Note: next_state.terminated will be False due to JaxMARL auto-reset behavior
    # The dones dict indicates termination, but the returned state is post-reset
    assert rewards["agent_0"] is not None  # Should have some reward value


def test_operations_execution():
    """Test individual ARCLE operations."""
    # Create a simple test state
    max_h, max_w = 10, 10
    grid = jnp.zeros((max_h, max_w), dtype=jnp.int32)
    grid = grid.at[2:5, 2:5].set(1)  # 3x3 square of 1's

    # Create dummy task data for operations testing
    dummy_task_data = ParsedTaskData(
        input_grids_examples=jnp.zeros((1, max_h, max_w), dtype=jnp.int32),
        input_masks_examples=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        output_grids_examples=jnp.zeros((1, max_h, max_w), dtype=jnp.int32),
        output_masks_examples=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        num_train_pairs=1,
        test_input_grids=jnp.zeros((1, max_h, max_w), dtype=jnp.int32),
        test_input_masks=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        true_test_output_grids=jnp.zeros((1, max_h, max_w), dtype=jnp.int32),
        true_test_output_masks=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        num_test_pairs=1,
        task_index=jnp.array(-1, dtype=jnp.int32),
    )

    test_state = ARCLEState(
        done=jnp.array(False),
        step=0,
        task_data=dummy_task_data,
        active_train_pair_idx=jnp.array(0, dtype=jnp.int32),
        working_grid=grid,
        working_grid_mask=jnp.ones((max_h, max_w), dtype=jnp.bool_),
        program=jnp.zeros((100, 10), dtype=jnp.int32),  # dummy program
        program_length=jnp.array(0, dtype=jnp.int32),
        active_agents=jnp.array([True], dtype=jnp.bool_),
        cumulative_rewards=jnp.array([0.0], dtype=jnp.float32),
        selected=jnp.zeros((max_h, max_w), dtype=jnp.bool_),
        clipboard=jnp.zeros((max_h, max_w), dtype=jnp.int32),
        grid_dim=jnp.array([max_h, max_w], dtype=jnp.int32),
        target_dim=jnp.array([max_h, max_w], dtype=jnp.int32),
        max_grid_dim=jnp.array([max_h, max_w], dtype=jnp.int32),
        step_count=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False, dtype=jnp.bool_),
        similarity_score=jnp.array(0.0, dtype=jnp.float32),
    )

    # Create selection for the 3x3 square
    selection = jnp.zeros((max_h, max_w), dtype=jnp.bool_)
    selection = selection.at[2:5, 2:5].set(True)

    # Test fill color
    new_state = fill_color(test_state, selection, 2)
    assert jnp.all(new_state.working_grid[2:5, 2:5] == 2)

    # Test copy and paste
    new_state = copy_to_clipboard(test_state, selection)
    assert jnp.all(new_state.clipboard[2:5, 2:5] == 1)

    # Test paste by overlapping selection with clipboard content
    # The paste operation only works where selection is True AND clipboard has content
    overlap_selection = jnp.zeros((max_h, max_w), dtype=jnp.bool_)
    overlap_selection = overlap_selection.at[2:5, 2:5].set(
        True
    )  # Same area as clipboard

    paste_state = paste_from_clipboard(new_state, overlap_selection)
    # Should still be 1's since we're pasting 1's over 1's
    assert jnp.all(paste_state.working_grid[2:5, 2:5] == 1)

    # Test reset grid
    reset_state = clear_grid(test_state, jnp.ones_like(selection))
    assert jnp.all(reset_state.working_grid == 0)


def test_execute_arcle_operation():
    """Test execute_arcle_operation function."""
    # Create a simple test state
    max_h, max_w = 10, 10
    grid = jnp.zeros((max_h, max_w), dtype=jnp.int32)

    # Create dummy task data for operations testing
    dummy_task_data = ParsedTaskData(
        input_grids_examples=jnp.zeros((1, max_h, max_w), dtype=jnp.int32),
        input_masks_examples=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        output_grids_examples=jnp.zeros((1, max_h, max_w), dtype=jnp.int32),
        output_masks_examples=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        num_train_pairs=1,
        test_input_grids=jnp.zeros((1, max_h, max_w), dtype=jnp.int32),
        test_input_masks=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        true_test_output_grids=jnp.zeros((1, max_h, max_w), dtype=jnp.int32),
        true_test_output_masks=jnp.ones((1, max_h, max_w), dtype=jnp.bool_),
        num_test_pairs=1,
        task_index=jnp.array(-1, dtype=jnp.int32),
    )

    test_state = ARCLEState(
        done=jnp.array(False),
        step=0,
        task_data=dummy_task_data,
        active_train_pair_idx=jnp.array(0, dtype=jnp.int32),
        working_grid=grid,
        working_grid_mask=jnp.ones((max_h, max_w), dtype=jnp.bool_),
        program=jnp.zeros((100, 10), dtype=jnp.int32),  # dummy program
        program_length=jnp.array(0, dtype=jnp.int32),
        active_agents=jnp.array([True], dtype=jnp.bool_),
        cumulative_rewards=jnp.array([0.0], dtype=jnp.float32),
        selected=jnp.zeros((max_h, max_w), dtype=jnp.bool_),
        clipboard=jnp.zeros((max_h, max_w), dtype=jnp.int32),
        grid_dim=jnp.array([max_h, max_w], dtype=jnp.int32),
        target_dim=jnp.array([max_h, max_w], dtype=jnp.int32),
        max_grid_dim=jnp.array([max_h, max_w], dtype=jnp.int32),
        step_count=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False, dtype=jnp.bool_),
        similarity_score=jnp.array(0.0, dtype=jnp.float32),
    )

    # Create selection for the 3x3 square
    selection = jnp.zeros((max_h, max_w), dtype=jnp.bool_)
    selection = selection.at[2:5, 2:5].set(True)

    # Test a few operations through the dispatch function
    operations_to_test = [
        (2, "fill_color_2"),
        (11, "flood_fill_1"),
        (20, "move_object_up"),
        (24, "rotate_object_90"),
        (34, "submit_answer"),
    ]

    for op_id, op_name in operations_to_test:
        # Set selection in state before executing operation
        state_with_selection = test_state.replace(selected=selection)

        new_state = execute_arcle_operation(
            state_with_selection, jnp.array(op_id, dtype=jnp.int32)
        )

        # Basic checks - execute_arcle_operation should set terminated for submit (34)
        if op_id == 34:
            assert new_state.terminated
        else:
            # For other operations, terminated status should remain unchanged
            assert new_state.terminated == test_state.terminated


def test_reward_computation(arcle_environment):
    """Test reward computation in ARCLE environment."""
    # Reset the environment
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs, state = arcle_environment.reset(subkey)

    # Make the working grid exactly match the target grid
    target_grid = state.task_data.output_grids_examples[state.active_train_pair_idx]
    perfect_state = state.replace(working_grid=target_grid)

    # Create a submit action
    action = {
        "agent_0": {
            "selection": jnp.zeros_like(state.working_grid, dtype=jnp.float32),
            "operation": jnp.array(34),  # Submit
        }
    }

    # Step with perfect match
    key, subkey = jax.random.split(key)
    _, _, perfect_rewards, _, _ = arcle_environment.step(subkey, perfect_state, action)

    # Step with original state (likely not a perfect match)
    key, subkey = jax.random.split(key)
    _, _, original_rewards, _, _ = arcle_environment.step(subkey, state, action)

    # Perfect match should have higher reward
    assert perfect_rewards["agent_0"] >= original_rewards["agent_0"]

    # Check if a non-submit action gives zero reward
    # Create a non-submit action for comparison
    non_submit_action = {
        "agent_0": {
            "selection": jnp.zeros_like(state.working_grid, dtype=jnp.float32),
            "operation": jnp.array(0),  # Fill with color 0
        }
    }

    if arcle_environment.reward_on_submit_only:
        key, subkey = jax.random.split(key)
        _, _, non_submit_rewards, _, _ = arcle_environment.step(
            subkey, state, non_submit_action
        )
        assert non_submit_rewards["agent_0"] == 0.0


def test_jit_compatibility(arcle_environment):
    """Test JIT compatibility of environment functions."""
    # Test JIT compatibility of reset
    jit_reset = jax.jit(lambda key: arcle_environment.reset(key))

    key = jax.random.PRNGKey(0)
    obs, state = jit_reset(key)

    assert "agent_0" in obs
    assert isinstance(state, ARCLEState)

    # Test JIT compatibility of step
    def step_fn(key, state, action):
        return arcle_environment.step(key, state, action)

    jit_step = jax.jit(step_fn)

    # Create a simple action
    action = {
        "agent_0": {
            "selection": jnp.zeros_like(state.working_grid, dtype=jnp.float32),
            "operation": jnp.array(0),  # Fill with color 0
        }
    }

    key, subkey = jax.random.split(key)
    obs, next_state, rewards, dones, info = jit_step(subkey, state, action)

    assert "agent_0" in obs
    assert isinstance(next_state, ARCLEState)
    assert "agent_0" in rewards
    assert "agent_0" in dones


def test_batch_processing():
    """Test batch processing capability."""
    # Create a batch of states
    batch_size = 4
    max_h, max_w = 10, 10

    # Create a batch of grids
    grids = jnp.zeros((batch_size, max_h, max_w), dtype=jnp.int32)

    # Create a batch of actions
    selections = jnp.zeros((batch_size, max_h, max_w), dtype=jnp.bool_)
    operations = jnp.arange(4, dtype=jnp.int32)  # Different operations

    # Define a batched operation function
    def batch_fill_color(grids, selections, colors):
        def single_fill(grid, selection, color):
            return jnp.where(selection, color, grid)

        return jax.vmap(single_fill)(grids, selections, colors)

    # Test batch processing
    colors = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    new_grids = batch_fill_color(grids, selections, colors)

    assert new_grids.shape == (batch_size, max_h, max_w)
