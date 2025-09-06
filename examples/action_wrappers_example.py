#!/usr/bin/env python3
"""
JaxARC Action Wrapper Example (Updated for Stoa API)

This example demonstrates how to use action wrappers and the spaces-based API
in JaxARC environments. It showcases:

Key Features Demonstrated:
- New spaces-based API (action_space, observation_space, reward_space)
- Enhanced TimeStep semantics (first(), mid(), last())
- BboxActionWrapper: {"operation": int, "r1": int, "c1": int, "r2": int, "c2": int} dict actions
- PointActionWrapper: {"operation": int, "row": int, "col": int} dict actions
- Space introspection and sampling
- JAX JIT/vmap/pmap compatibility

Usage:
    pixi run python examples/action_wrappers_example.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc.envs.action_wrappers import BboxActionWrapper, PointActionWrapper
from jaxarc.registration import make
from jaxarc.utils.visualization import log_grid_to_console


def basic_usage_example():
    """Demonstrate the basic desired API flow with spaces."""
    print("🚀 Basic Usage Example")
    print("=" * 50)

    # The exact API flow requested by the user
    env, env_params = make("Mini", auto_download=True)
    env = BboxActionWrapper(env)

    print(f"✅ Created wrapped environment: {type(env).__name__}")
    print(f"✅ Environment parameters: {type(env_params).__name__}")

    # Demonstrate space introspection
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    reward_space = env.reward_space(env_params)
    
    print(f"✅ Observation space: {obs_space}")
    print(f"✅ Action space: {action_space}")
    print(f"✅ Reward space: {reward_space}")

    # Reset and take a step
    key = jax.random.PRNGKey(42)
    timestep = env.reset(env_params, key)
    print(f"✅ Reset successful: {timestep.observation.shape}")
    print(f"✅ Initial timestep - first(): {timestep.first()}")

    # Simple bbox action - using dict format for BboxActionWrapper
    bbox_action = {"operation": 15, "r1": 1, "c1": 1, "r2": 3, "c2": 4}
    new_timestep = env.step(env_params, timestep, bbox_action)
    print(f"✅ Step successful with bbox action: {bbox_action}")
    print(f"✅ Next timestep - mid(): {new_timestep.mid()}")

    return env, env_params


def space_introspection_example():
    """Demonstrate the new space system."""
    print("\n🔍 Space System Demo")
    print("=" * 50)
    
    env, env_params = make("Mini", auto_download=True)
    
    # Demonstrate space introspection
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    reward_space = env.reward_space(env_params)
    discount_space = env.discount_space(env_params)
    
    print(f"✅ Observation space: {obs_space}")
    print(f"✅ Action space: {action_space}")
    print(f"✅ Reward space: {reward_space}")
    print(f"✅ Discount space: {discount_space}")
    
    # Sample from spaces
    key = jax.random.PRNGKey(42)
    sample_action = action_space.sample(key)
    print(f"✅ Sampled action: operation={sample_action['operation']}")
    print(f"   Selection shape: {sample_action['selection'].shape}")
    print(f"   Selection sum: {jnp.sum(sample_action['selection'])}")


def step_type_semantics_example():
    """Demonstrate enhanced TimeStep semantics."""
    print("\n📊 StepType Semantics Demo")
    print("=" * 50)
    
    env, env_params = make("Mini", auto_download=True)
    env = BboxActionWrapper(env)
    
    key = jax.random.PRNGKey(123)
    timestep = env.reset(env_params, key)
    
    print(f"✅ Initial step - first(): {timestep.first()}")
    print(f"   Step type: {timestep.step_type}")
    
    # Take some actions to see mid() transitions - using dict format for BboxActionWrapper
    actions = [
        {"operation": 1, "r1": 0, "c1": 0, "r2": 1, "c2": 1}, 
        {"operation": 2, "r1": 2, "c1": 2, "r2": 3, "c2": 3}
    ]
    
    for i, action in enumerate(actions):
        timestep = env.step(env_params, timestep, action)
        print(f"✅ After action {i+1} - mid(): {timestep.mid()}, last(): {timestep.last()}")
        if timestep.last():
            print("   Episode terminated!")
            break


def bbox_wrapper_demo():
    """Demonstrate BboxActionWrapper with visual output."""
    print("\n🎯 BboxActionWrapper Demo")
    print("=" * 50)

    env, env_params = make("Mini", auto_download=True)
    env = BboxActionWrapper(env)

    key = jax.random.PRNGKey(123)
    timestep = env.reset(env_params, key)

    print("🔹 Initial grid:")
    log_grid_to_console(timestep.observation, title="Working Grid")

    # Execute bbox action: fill rectangle using dict format
    bbox_action = {"operation": 2, "r1": 1, "c1": 2, "r2": 3, "c2": 5}  # Green rectangle from (1,2) to (3,5)
    timestep = env.step(env_params, timestep, bbox_action)

    print(f"\n🔹 After bbox action {bbox_action}:")
    log_grid_to_console(timestep.observation, title="After Rectangle Fill")
    print(f"   Reward: {timestep.reward}")


def point_wrapper_demo():
    """Demonstrate PointActionWrapper with visual output."""
    print("\n🎯 PointActionWrapper Demo")
    print("=" * 50)

    env, env_params = make("Mini", auto_download=True)
    env = PointActionWrapper(env)

    key = jax.random.PRNGKey(456)
    timestep = env.reset(env_params, key)

    print("🔹 Initial grid:")
    log_grid_to_console(timestep.observation, title="Working Grid")

    # Execute multiple point actions using dict format
    point_actions = [
        {"operation": 1, "row": 2, "col": 3},  # Red at (2,3)
        {"operation": 3, "row": 2, "col": 4},  # Blue at (2,4)
        {"operation": 4, "row": 3, "col": 3},  # Yellow at (3,3)
        {"operation": 4, "row": 3, "col": 4},  # Yellow at (3,4)
    ]

    for i, action in enumerate(point_actions):
        timestep = env.step(env_params, timestep, action)
        print(f"   Point action {i + 1}: {action}, reward: {timestep.reward}")

    print(f"\n🔹 After {len(point_actions)} point actions:")
    log_grid_to_console(timestep.observation, title="After Point Actions")


def multiple_actions_demo():
    """Demonstrate complex sequences of actions."""
    print("\n🎯 Multiple Actions Sequence Demo")
    print("=" * 50)

    env, env_params = make("Mini", auto_download=True)
    env = BboxActionWrapper(env)

    key = jax.random.PRNGKey(789)
    timestep = env.reset(env_params, key)

    # Create a pattern with multiple bbox actions using dict format
    actions_sequence = [
        {"operation": 1, "r1": 0, "c1": 0, "r2": 1, "c2": 1},  # Red square in top-left corner
        {"operation": 2, "r1": 0, "c1": 3, "r2": 1, "c2": 4},  # Green rectangle in top-right
        {"operation": 3, "r1": 3, "c1": 0, "r2": 4, "c2": 1},  # Blue rectangle in bottom-left
        {"operation": 4, "r1": 3, "c1": 3, "r2": 4, "c2": 4},  # Yellow square in bottom-right
        {"operation": 5, "r1": 2, "c1": 2, "r2": 2, "c2": 2},  # Gray point in center
    ]

    print("🔹 Executing action sequence:")
    total_reward = 0.0
    for i, action in enumerate(actions_sequence):
        timestep = env.step(env_params, timestep, action)
        total_reward += timestep.reward
        print(f"   Action {i + 1}: {action}, reward: {timestep.reward}")
        
        # Check if episode ended
        if timestep.last():
            print(f"   Episode ended after action {i + 1}")
            break

    print(f"\n🔹 Final result (total reward: {total_reward}):")
    log_grid_to_console(timestep.observation, title="Pattern Created")


def interface_compatibility_demo():
    """Show that wrappers provide full Environment interface."""
    print("\n🔧 Interface Compatibility Demo")
    print("=" * 50)

    env, env_params = make("Mini", auto_download=True)

    # Test both wrappers
    bbox_env = BboxActionWrapper(env)
    point_env = PointActionWrapper(env)

    # Check all methods exist and work
    methods_to_test = ["reset", "step", "observation_space", "action_space", "reward_space", "discount_space"]

    print("✅ BboxActionWrapper interface:")
    for method in methods_to_test:
        has_method = hasattr(bbox_env, method) and callable(getattr(bbox_env, method))
        print(f"   - {method}: {'✓' if has_method else '✗'}")

    print("\n✅ PointActionWrapper interface:")
    for method in methods_to_test:
        has_method = hasattr(point_env, method) and callable(getattr(point_env, method))
        print(f"   - {method}: {'✓' if has_method else '✗'}")

    # Test space methods
    obs_space = bbox_env.observation_space(env_params)
    action_space = bbox_env.action_space(env_params)
    reward_space = bbox_env.reward_space(env_params)
    
    print(f"\n✅ Observation space: {obs_space}")
    print(f"✅ Action space: {action_space}")
    print(f"✅ Reward space: {reward_space}")
    print("✅ Both wrappers provide complete Environment interface!")


def jax_compatibility_demo():
    """Demonstrate JAX JIT compilation compatibility."""
    print("\n⚡ JAX Compatibility Demo")
    print("=" * 50)

    # Test action space sampling with JIT
    env, env_params = make("Mini", auto_download=True)
    action_space = env.action_space(env_params)
    
    # JIT-compile action sampling
    @jax.jit
    def sample_actions(key):
        return action_space.sample(key)
    
    print("🔹 Testing JIT compilation of action sampling:")
    key = jax.random.PRNGKey(42)
    sample_action = sample_actions(key)
    print(f"   ✓ JIT action sampling: operation={sample_action['operation']}")
    print(f"   ✓ Selection sum: {jnp.sum(sample_action['selection'])}")
    
    # Test batch environment operations
    print("🔹 Testing batch operations:")
    batch_size = 4
    batch_keys = jax.random.split(key, batch_size)
    
    # vmap reset for batch processing  
    batch_reset = jax.vmap(env.reset, in_axes=(None, 0))
    batch_timesteps = batch_reset(env_params, batch_keys)
    print(f"   ✓ Batch reset: {batch_timesteps.observation.shape}")
    print(f"   ✓ All first timesteps: {jnp.all(batch_timesteps.first())}")
    
    print("   ✓ Environment operations are JAX JIT/vmap/pmap compatible")


def main():
    """Run all demonstration examples."""
    print("🎯 JaxARC Action Wrapper Examples (Stoa API)")
    print("=" * 60)

    try:
        # Run all examples
        basic_usage_example()
        space_introspection_example() 
        step_type_semantics_example()
        bbox_wrapper_demo()
        point_wrapper_demo()
        multiple_actions_demo()
        interface_compatibility_demo()
        jax_compatibility_demo()

        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
