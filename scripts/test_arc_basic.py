#!/usr/bin/env python3
"""
Basic test script for ARC environment functionality.

This script tests the core functionality of the ARC environment to ensure
it works correctly despite any type checking issues.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import jax
import jax.numpy as jnp

from jaxarc.envs import ArcEnvironment
from jaxarc.types import ParsedTaskData
from jaxarc.utils.task_manager import create_jax_task_index


def create_dummy_task_data(grid_size=(10, 10)):
    """Create a simple dummy task for testing."""
    h, w = grid_size

    # Create simple input and output grids
    input_grid = jnp.zeros((1, h, w), dtype=jnp.int32)
    input_grid = input_grid.at[0, 2:4, 2:4].set(1)  # Small square of color 1

    output_grid = jnp.zeros((1, h, w), dtype=jnp.int32)
    output_grid = output_grid.at[0, 2:4, 2:4].set(2)  # Same square but color 2

    masks = jnp.ones((1, h, w), dtype=jnp.bool_)

    return ParsedTaskData(
        input_grids_examples=input_grid,
        input_masks_examples=masks,
        output_grids_examples=output_grid,
        output_masks_examples=masks,
        num_train_pairs=1,
        test_input_grids=input_grid,
        test_input_masks=masks,
        true_test_output_grids=output_grid,
        true_test_output_masks=masks,
        num_test_pairs=1,
        task_index=create_jax_task_index("test_task_001"),
    )


def test_environment_creation():
    """Test that we can create an ARC environment."""
    print("🧪 Testing environment creation...")

    try:
        env = ArcEnvironment(num_agents=1, max_grid_size=(10, 10), max_episode_steps=50)
        print("✅ Environment created successfully")
        print(f"   - Name: {env.name}")
        print(f"   - Agents: {env.agents}")
        print(f"   - Action spaces: {list(env.action_spaces.keys())}")
        print(f"   - Observation spaces: {list(env.observation_spaces.keys())}")
        return env
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        raise


def test_environment_reset(env):
    """Test environment reset functionality."""
    print("\n🧪 Testing environment reset...")

    try:
        # Create test task
        task_data = create_dummy_task_data((10, 10))

        # Reset environment
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key, task_data)

        print("✅ Environment reset successful")
        print(f"   - Observation keys: {list(obs.keys())}")
        print(f"   - Observation shape: {obs[env.agents[0]].shape}")
        print(f"   - State type: {type(state).__name__}")
        print(f"   - Initial step: {state.step}")
        print(f"   - Grid shape: {state.grid.shape}")
        print(f"   - Similarity score: {float(state.similarity_score):.3f}")

        return obs, state

    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_environment_step(env, initial_obs, initial_state):
    """Test environment step functionality."""
    print("\n🧪 Testing environment step...")

    try:
        key = jax.random.PRNGKey(123)

        # Create a simple action
        agent_id = env.agents[0]
        h, w = env.max_grid_size

        # Create selection mask (select the center area)
        selection = jnp.zeros((h, w), dtype=jnp.float32)
        selection = selection.at[2:4, 2:4].set(1.0)  # Select where the square is

        action = {
            "selection": selection,
            "operation": jnp.array(2, dtype=jnp.int32),  # Fill with color 2
        }
        actions = {agent_id: action}

        # Take a step
        obs, new_state, rewards, dones, infos = env.step_env(
            key, initial_state, actions
        )

        print("✅ Environment step successful")
        print(f"   - New step: {new_state.step}")
        print(f"   - Reward: {rewards[agent_id]:.3f}")
        print(f"   - Done: {dones[agent_id]}")
        print(f"   - New similarity: {float(new_state.similarity_score):.3f}")
        print(f"   - Info: {infos[agent_id]}")

        # Check if the grid was modified
        grid_changed = not jnp.array_equal(initial_state.grid, new_state.grid)
        print(f"   - Grid modified: {grid_changed}")

        return obs, new_state, rewards, dones, infos

    except Exception as e:
        print(f"❌ Environment step failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_multiple_steps(env):
    """Test multiple environment steps."""
    print("\n🧪 Testing multiple steps...")

    try:
        # Reset environment
        task_data = create_dummy_task_data((10, 10))
        key = jax.random.PRNGKey(456)
        obs, state = env.reset(key, task_data)

        agent_id = env.agents[0]
        h, w = env.max_grid_size

        # Take several steps with different actions
        for step_num in range(3):
            key, step_key = jax.random.split(key)

            # Random selection mask
            selection = jax.random.uniform(step_key, (h, w))
            selection = (selection > 0.8).astype(jnp.float32)  # Sparse selection

            action = {
                "selection": selection,
                "operation": jnp.array(
                    step_num % 3, dtype=jnp.int32
                ),  # Cycle through operations
            }
            actions = {agent_id: action}

            obs, state, rewards, dones, infos = env.step_env(step_key, state, actions)

            print(
                f"   Step {step_num + 1}: reward={rewards[agent_id]:.3f}, "
                f"similarity={float(state.similarity_score):.3f}, "
                f"done={dones[agent_id]}"
            )

            if dones[agent_id]:
                print("   Episode terminated early")
                break

        print("✅ Multiple steps completed successfully")

    except Exception as e:
        print(f"❌ Multiple steps test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_submit_action(env):
    """Test the submit action (operation 34)."""
    print("\n🧪 Testing submit action...")

    try:
        # Reset environment
        task_data = create_dummy_task_data((10, 10))
        key = jax.random.PRNGKey(789)
        obs, state = env.reset(key, task_data)

        agent_id = env.agents[0]
        h, w = env.max_grid_size

        # Create submit action
        selection = jnp.zeros((h, w), dtype=jnp.float32)  # Empty selection for submit
        action = {
            "selection": selection,
            "operation": jnp.array(34, dtype=jnp.int32),  # Submit operation
        }
        actions = {agent_id: action}

        obs, new_state, rewards, dones, infos = env.step_env(key, state, actions)

        print("✅ Submit action successful")
        print(f"   - Terminated: {bool(new_state.terminated)}")
        print(f"   - Done: {dones[agent_id]}")
        print(f"   - Final reward: {rewards[agent_id]:.3f}")

    except Exception as e:
        print(f"❌ Submit action test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """Run all tests."""
    print("🚀 Starting ARC Environment Basic Tests")
    print("=" * 50)

    try:
        # Test 1: Environment creation
        env = test_environment_creation()

        # Test 2: Environment reset
        obs, state = test_environment_reset(env)

        # Test 3: Single step
        test_environment_step(env, obs, state)

        # Test 4: Multiple steps
        test_multiple_steps(env)

        # Test 5: Submit action
        test_submit_action(env)

        print("\n" + "=" * 50)
        print("🎉 All tests passed! ARC environment is working correctly.")

    except Exception as e:
        print("\n" + "=" * 50)
        print(f"💥 Tests failed with error: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
