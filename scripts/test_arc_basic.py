#!/usr/bin/env python3
"""
Basic test script for ARC environment functionality with new config-based API.

This script tests the core functionality of the new config-based ARC environment
to ensure it works correctly with the functional API.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import jax
import jax.numpy as jnp

from jaxarc.envs import (
    arc_reset,
    arc_step,
    create_point_config,
    create_raw_config,
    create_standard_config,
    get_config_summary,
    validate_config,
)


def test_config_creation():
    """Test that we can create different configurations."""
    print("🧪 Testing configuration creation...")

    try:
        # Test standard config
        config = create_standard_config(
            max_episode_steps=50, success_bonus=10.0, log_operations=True
        )
        print("✅ Standard config created successfully")
        print(f"   - Max episodes: {config.max_episode_steps}")
        print(f"   - Action format: {config.action.action_format}")
        print(f"   - Operations: {config.action.num_operations}")
        print(f"   - Success bonus: {config.reward.success_bonus}")

        # Test raw config
        raw_config = create_raw_config(max_episode_steps=25)
        print("✅ Raw config created successfully")
        print(f"   - Max episodes: {raw_config.max_episode_steps}")
        print(f"   - Operations: {raw_config.action.num_operations}")

        # Test point config
        point_config = create_point_config(max_episode_steps=30)
        print("✅ Point config created successfully")
        print(f"   - Action format: {point_config.action.action_format}")

        # Validate configurations
        validate_config(config)
        validate_config(raw_config)
        validate_config(point_config)
        print("✅ All configurations validated successfully")

        return config

    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        raise


def test_environment_reset(config):
    """Test environment reset functionality with new API."""
    print("\n🧪 Testing environment reset...")

    try:
        # Reset environment using functional API
        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, config)

        print("✅ Environment reset successful")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - State type: {type(state).__name__}")
        print(f"   - Initial step: {state.step_count}")
        print(f"   - Grid shape: {state.working_grid.shape}")
        print(f"   - Similarity score: {float(state.similarity_score):.3f}")
        print(f"   - Task available: {state.task_data is not None}")

        return state, obs

    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_environment_step(config, initial_state):
    """Test environment step functionality with new API."""
    print("\n🧪 Testing environment step...")

    try:
        # Create a simple action
        h, w = initial_state.working_grid.shape

        # Create selection mask (select a small area)
        selection = jnp.zeros((h, w), dtype=jnp.bool_)
        selection = selection.at[2:4, 2:4].set(True)  # Select 2x2 area

        action = {
            "selection": selection,
            "operation": jnp.array(2, dtype=jnp.int32),  # Fill with color 2
        }

        # Take a step using functional API
        new_state, new_obs, reward, done, info = arc_step(initial_state, action, config)

        print("✅ Environment step successful")
        print(f"   - New step: {new_state.step_count}")
        print(f"   - Reward: {reward:.3f}")
        print(f"   - Done: {done}")
        print(f"   - New similarity: {float(new_state.similarity_score):.3f}")
        print(f"   - Info keys: {list(info.keys())}")

        # Check if the grid was modified
        grid_changed = not jnp.array_equal(
            initial_state.working_grid, new_state.working_grid
        )
        print(f"   - Grid modified: {grid_changed}")

        return new_state, new_obs, reward, done, info

    except Exception as e:
        print(f"❌ Environment step failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_multiple_steps(config):
    """Test multiple environment steps with new API."""
    print("\n🧪 Testing multiple steps...")

    try:
        # Reset environment
        key = jax.random.PRNGKey(456)
        state, obs = arc_reset(key, config)

        h, w = state.working_grid.shape

        # Take several steps with different actions
        for step_num in range(3):
            key, step_key = jax.random.split(key)

            # Create random selection mask
            selection_key, _ = jax.random.split(step_key)
            selection_probs = jax.random.uniform(selection_key, (h, w))
            selection = selection_probs > 0.8  # Sparse selection

            # Use different operations
            op_id = step_num % min(
                10, config.action.num_operations
            )  # Stay within available ops
            action = {
                "selection": selection,
                "operation": jnp.array(op_id, dtype=jnp.int32),
            }

            state, obs, reward, done, info = arc_step(state, action, config)

            print(
                f"   Step {step_num + 1}: reward={reward:.3f}, "
                f"similarity={float(state.similarity_score):.3f}, "
                f"done={done}"
            )

            if done:
                print("   Episode terminated early")
                break

        print("✅ Multiple steps completed successfully")

    except Exception as e:
        print(f"❌ Multiple steps test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_submit_action(config):
    """Test the submit action (operation 34)."""
    print("\n🧪 Testing submit action...")

    try:
        # Reset environment
        key = jax.random.PRNGKey(789)
        state, obs = arc_reset(key, config)

        h, w = state.working_grid.shape

        # Create submit action
        selection = jnp.zeros((h, w), dtype=jnp.bool_)  # Empty selection for submit
        action = {
            "selection": selection,
            "operation": jnp.array(34, dtype=jnp.int32),  # Submit operation
        }

        new_state, new_obs, reward, done, info = arc_step(state, action, config)

        print("✅ Submit action successful")
        print(f"   - Terminated: {done}")
        print(f"   - Final reward: {reward:.3f}")
        print(f"   - Final similarity: {info.get('similarity', 'N/A')}")

    except Exception as e:
        print(f"❌ Submit action test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_point_actions():
    """Test point-based actions."""
    print("\n🧪 Testing point-based actions...")

    try:
        # Create point config
        config = create_point_config(max_episode_steps=20)

        # Reset environment
        key = jax.random.PRNGKey(999)
        state, obs = arc_reset(key, config)

        # Create point action
        action = {
            "point": (3, 4),  # Point at row 3, col 4
            "operation": jnp.array(1, dtype=jnp.int32),  # Fill with color 1
        }

        new_state, new_obs, reward, done, info = arc_step(state, action, config)

        print("✅ Point action successful")
        print(f"   - Reward: {reward:.3f}")
        print(
            f"   - Grid changed: {not jnp.array_equal(state.working_grid, new_state.working_grid)}"
        )

    except Exception as e:
        print(f"❌ Point action test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_jax_compatibility(config):
    """Test JAX transformations with new API."""
    print("\n🧪 Testing JAX compatibility...")

    try:
        # Test JIT compilation
        @jax.jit
        def jitted_reset(key):
            return arc_reset(key, config)

        @jax.jit
        def jitted_step(state, action):
            return arc_step(state, action, config)

        # Use JIT-compiled functions
        key = jax.random.PRNGKey(111)
        state, obs = jitted_reset(key)

        h, w = state.working_grid.shape
        selection = jnp.zeros((h, w), dtype=jnp.bool_).at[1:3, 1:3].set(True)
        action = {
            "selection": selection,
            "operation": jnp.array(1, dtype=jnp.int32),
        }

        new_state, new_obs, reward, done, info = jitted_step(state, action)

        print("✅ JIT compilation successful")
        print(f"   - JIT reset worked: {state.step_count == 0}")
        print(f"   - JIT step worked: {new_state.step_count == 1}")

        # Test vmap
        def single_episode(key):
            state, obs = arc_reset(key, config)
            return state.similarity_score

        keys = jax.random.split(key, 3)
        batch_similarities = jax.vmap(single_episode)(keys)

        print("✅ vmap successful")
        print(f"   - Batch similarities: {batch_similarities}")

    except Exception as e:
        print(f"❌ JAX compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_config_summary(config):
    """Test configuration summary functionality."""
    print("\n🧪 Testing configuration summary...")

    try:
        summary = get_config_summary(config)
        print("✅ Configuration summary generated")
        print(f"   Summary preview: {summary[:100]}...")

    except Exception as e:
        print(f"❌ Configuration summary test failed: {e}")
        raise


def main():
    """Run all tests."""
    print("🚀 Starting ARC Environment Basic Tests (New Config-Based API)")
    print("=" * 60)

    try:
        # Test 1: Configuration creation
        config = test_config_creation()

        # Test 2: Environment reset
        state, obs = test_environment_reset(config)

        # Test 3: Single step
        test_environment_step(config, state)

        # Test 4: Multiple steps
        test_multiple_steps(config)

        # Test 5: Submit action
        test_submit_action(config)

        # Test 6: Point actions
        test_point_actions()

        # Test 7: JAX compatibility
        test_jax_compatibility(config)

        # Test 8: Configuration summary
        test_config_summary(config)

        print("\n" + "=" * 60)
        print(
            "🎉 All tests passed! New config-based ARC environment is working correctly."
        )

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"💥 Tests failed with error: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
