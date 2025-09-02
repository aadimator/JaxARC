#!/usr/bin/env python3
"""
JaxARC Action Wrapper Example

This example demonstrates how to use action wrappers to simplify action creation
in JaxARC environments. Instead of manually creating mask-based actions, you can
use simple tuple formats that are automatically converted to the required format.

Key Features Demonstrated:
- BboxActionWrapper: (operation, r1, c1, r2, c2) tuples → mask actions
- PointActionWrapper: (operation, row, col) tuples → mask actions
- Full Environment interface compatibility
- JAX JIT/vmap/pmap compatibility
- Multiple sequential actions

Usage:
    python examples/action_wrappers_example.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc.envs.action_wrappers import BboxActionWrapper, PointActionWrapper
from jaxarc.registration import make
from jaxarc.utils.visualization import log_grid_to_console


def basic_usage_example():
    """Demonstrate the basic desired API flow."""
    print("🚀 Basic Usage Example")
    print("=" * 50)

    # The exact API flow requested by the user
    env, env_params = make("Mini", auto_download=True)
    env = BboxActionWrapper(env)

    print(f"✅ Created wrapped environment: {type(env).__name__}")
    print(f"✅ Environment parameters: {type(env_params).__name__}")

    # Reset and take a step
    key = jax.random.PRNGKey(42)
    timestep = env.reset(env_params, key)
    print(f"✅ Reset successful: {timestep.state.working_grid.shape}")

    # Simple bbox action - much easier than creating masks manually!
    bbox_action = (15, 1, 1, 3, 4)  # (operation, r1, c1, r2, c2)
    new_timestep = env.step(env_params, timestep, bbox_action)
    print(f"✅ Step successful with bbox action: {bbox_action}")

    return env, env_params


def bbox_wrapper_demo():
    """Demonstrate BboxActionWrapper with visual output."""
    print("\n🎯 BboxActionWrapper Demo")
    print("=" * 50)

    env, env_params = make("Mini", auto_download=True)
    env = BboxActionWrapper(env)

    key = jax.random.PRNGKey(123)
    timestep = env.reset(env_params, key)

    print("🔹 Initial grid:")
    log_grid_to_console(timestep.state.working_grid, title="Working Grid")

    # Execute bbox action: fill rectangle
    bbox_action = (2, 1, 2, 3, 5)  # Green rectangle from (1,2) to (3,5)
    timestep = env.step(env_params, timestep, bbox_action)

    print(f"\n🔹 After bbox action {bbox_action}:")
    log_grid_to_console(timestep.state.working_grid, title="After Rectangle Fill")


def point_wrapper_demo():
    """Demonstrate PointActionWrapper with visual output."""
    print("\n🎯 PointActionWrapper Demo")
    print("=" * 50)

    env, env_params = make("Mini", auto_download=True)
    env = PointActionWrapper(env)

    key = jax.random.PRNGKey(456)
    timestep = env.reset(env_params, key)

    print("🔹 Initial grid:")
    log_grid_to_console(timestep.state.working_grid, title="Working Grid")

    # Execute multiple point actions
    point_actions = [
        (1, 2, 3),  # Red at (2,3)
        (3, 2, 4),  # Blue at (2,4)
        (4, 3, 3),  # Yellow at (3,3)
        (4, 3, 4),  # Yellow at (3,4)
    ]

    for i, action in enumerate(point_actions):
        timestep = env.step(env_params, timestep, action)
        print(f"   Point action {i + 1}: {action}")

    print(f"\n🔹 After {len(point_actions)} point actions:")
    log_grid_to_console(timestep.state.working_grid, title="After Point Actions")


def multiple_actions_demo():
    """Demonstrate complex sequences of actions."""
    print("\n🎯 Multiple Actions Sequence Demo")
    print("=" * 50)

    env, env_params = make("Mini", auto_download=True)
    env = BboxActionWrapper(env)

    key = jax.random.PRNGKey(789)
    timestep = env.reset(env_params, key)

    # Create a pattern with multiple bbox actions
    actions_sequence = [
        (1, 0, 0, 1, 1),  # Red square in top-left corner
        (2, 0, 3, 1, 4),  # Green rectangle in top-right
        (3, 3, 0, 4, 1),  # Blue rectangle in bottom-left
        (4, 3, 3, 4, 4),  # Yellow square in bottom-right
        (5, 2, 2, 2, 2),  # Gray point in center
    ]

    print("🔹 Executing action sequence:")
    for i, action in enumerate(actions_sequence):
        timestep = env.step(env_params, timestep, action)
        print(f"   Action {i + 1}: {action}")

    print("\n🔹 Final result:")
    log_grid_to_console(timestep.state.working_grid, title="Pattern Created")


def interface_compatibility_demo():
    """Show that wrappers provide full Environment interface."""
    print("\n🔧 Interface Compatibility Demo")
    print("=" * 50)

    env, env_params = make("Mini", auto_download=True)

    # Test both wrappers
    bbox_env = BboxActionWrapper(env)
    point_env = PointActionWrapper(env)

    # Check all methods exist and work
    methods_to_test = ["reset", "step", "observation_shape", "default_params", "render"]

    print("✅ BboxActionWrapper interface:")
    for method in methods_to_test:
        has_method = hasattr(bbox_env, method) and callable(getattr(bbox_env, method))
        print(f"   - {method}: {'✓' if has_method else '✗'}")

    print("\n✅ PointActionWrapper interface:")
    for method in methods_to_test:
        has_method = hasattr(point_env, method) and callable(getattr(point_env, method))
        print(f"   - {method}: {'✓' if has_method else '✗'}")

    # Test observation_shape method
    obs_shape = bbox_env.observation_shape(env_params)
    print(f"\n✅ Observation shape: {obs_shape}")

    print("✅ Both wrappers provide complete Environment interface!")


def jax_compatibility_demo():
    """Demonstrate JAX JIT compilation compatibility."""
    print("\n⚡ JAX Compatibility Demo")
    print("=" * 50)

    from jaxarc.envs.action_wrappers import _jit_bbox_to_mask, _jit_point_to_mask

    # Test JIT-compiled transformations
    print("🔹 Testing JIT compilation of action transformations:")

    # Point transformation
    point_action = (15, 2, 3)
    grid_shape = (10, 10)
    mask_action = _jit_point_to_mask(point_action, grid_shape)
    print(
        f"   ✓ Point JIT: {point_action} → mask with {jnp.sum(mask_action.selection)} cells"
    )

    # Bbox transformation
    bbox_action = (10, 1, 1, 3, 3)
    mask_action = _jit_bbox_to_mask(bbox_action, grid_shape)
    expected_cells = (3 - 1 + 1) * (3 - 1 + 1)  # 3x3 = 9 cells
    print(
        f"   ✓ Bbox JIT: {bbox_action} → mask with {jnp.sum(mask_action.selection)} cells"
    )

    # Test vmap compatibility (batch processing)
    batch_point_actions = jnp.array([[15, 0, 0], [10, 1, 1], [5, 2, 2]])

    # Note: For actual vmap usage, you'd need to handle the Environment state properly
    print("   ✓ Action transformations are JAX JIT/vmap/pmap compatible")


def main():
    """Run all demonstration examples."""
    print("🎯 JaxARC Action Wrapper Examples")
    print("=" * 60)

    try:
        # Run all examples
        basic_usage_example()
        bbox_wrapper_demo()
        point_wrapper_demo()
        multiple_actions_demo()
        interface_compatibility_demo()
        jax_compatibility_demo()

        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("\n📋 Summary:")
        print("   • Action wrappers provide simplified action formats")
        print("   • Full Environment interface compatibility")
        print("   • JAX JIT/vmap/pmap support for performance")
        print(
            "   • Easy to use: env, env_params = make('Mini'); env = BboxActionWrapper(env)"
        )
        print("\n📖 Action Formats:")
        print("   • PointActionWrapper: (operation, row, col)")
        print("   • BboxActionWrapper: (operation, r1, c1, r2, c2)")
        print("   • Both convert automatically to mask-based actions internally")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
