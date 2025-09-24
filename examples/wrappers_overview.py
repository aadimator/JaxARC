#!/usr/bin/env python3
"""
Wrappers Overview: Simple and Comprehensive Demo

This single example consolidates the previous demos into one place. It shows:
- Spaces introspection and sampling
- BboxActionWrapper and PointActionWrapper dict actions
- FlattenActionWrapper over a dict-discrete action space
- AddChannelDimWrapper for observation shape adaptation
- JAX JIT/vmap compatibility for common flows

Run:
    pixi run python examples/wrappers_overview.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc.envs import (
    AddChannelDimWrapper,
    BboxActionWrapper,
    FlattenActionWrapper,
    PointActionWrapper,
)
from jaxarc.registration import make
from jaxarc.utils.visualization import log_grid_to_console


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def spaces_and_reset_demo() -> tuple[object, object]:
    section("Spaces and Reset Demo")
    env, env_params = make(
        "Mini-Most_Common_color_l6ab0lf3xztbyxsu3p", auto_download=True
    )

    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    reward_space = env.reward_space(env_params)
    discount_space = env.discount_space(env_params)

    print(f"Observation space: {obs_space}")
    print(f"Action space:      {action_space}")
    print(f"Reward space:      {reward_space}")
    print(f"Discount space:    {discount_space}")

    key = jax.random.PRNGKey(0)
    state, ts = env.reset(key, env_params=env_params)
    print(f"Reset OK. first(): {ts.first()} | obs shape: {ts.observation.shape}")

    # Sample a canonical (mask-based) action from the base env
    key = jax.random.PRNGKey(1)
    sample = action_space.sample(key)
    print(
        f"Sampled action -> operation={sample['operation']}, selection.shape={tuple(sample['selection'].shape)}"
    )
    return env, env_params


def bbox_wrapper_demo(base_env, env_params) -> None:
    section("BboxActionWrapper Demo (dict actions)")
    env = BboxActionWrapper(base_env)

    key = jax.random.PRNGKey(2)
    state, ts = env.reset(key, env_params=env_params)

    print("Initial grid (working):")
    log_grid_to_console(ts.observation, title="Working Grid")

    action = {"operation": 2, "r1": 1, "c1": 1, "r2": 3, "c2": 4}
    state, ts = env.step(state, action, env_params=env_params)
    print(f"Step with bbox dict action: {action} -> reward: {float(ts.reward):.3f}")

    print("After bbox action:")
    log_grid_to_console(ts.observation, title="After Bbox Action")


def point_wrapper_demo(base_env, env_params) -> None:
    section("PointActionWrapper Demo (dict actions)")
    env = PointActionWrapper(base_env)

    key = jax.random.PRNGKey(3)
    state, ts = env.reset(key, env_params=env_params)

    print("Initial grid (working):")
    log_grid_to_console(ts.observation, title="Working Grid")

    actions = [
        {"operation": 1, "row": 2, "col": 3},
        {"operation": 3, "row": 2, "col": 4},
        {"operation": 4, "row": 3, "col": 3},
    ]
    for i, a in enumerate(actions, start=1):
        state, ts = env.step(state, a, env_params=env_params)
        print(f"Point action {i}: {a} -> reward: {float(ts.reward):.3f}")

    print("After point actions:")
    log_grid_to_console(ts.observation, title="After Point Actions")


def flatten_wrapper_demo(base_env, env_params) -> None:
    section("FlattenActionWrapper Demo (over PointActionWrapper)")
    # Wrap a dict-discrete action space env; PointActionWrapper provides a dict action space
    env = PointActionWrapper(base_env)
    env = FlattenActionWrapper(env)

    flat_space = env.action_space(env_params)
    num_values = getattr(flat_space, "num_values", None)
    if num_values is None:
        msg = "Flat action space must expose `num_values`."
        raise RuntimeError(msg)
    print(f"Flat action space size: num_values={num_values}")

    key = jax.random.PRNGKey(4)
    state, ts = env.reset(key, env_params=env_params)

    # Sample a flat action index
    key = jax.random.PRNGKey(5)
    flat_action = jax.random.randint(key, shape=(), minval=0, maxval=num_values)
    state, ts = env.step(state, flat_action, env_params=env_params)

    # Show that wrappers propagated canonical mask-based info for logging/visualization
    extras = getattr(ts, "extras", {}) or {}
    canonical = extras.get("canonical_action")
    print(
        f"Step with flat action index: {int(flat_action)} -> reward: {float(ts.reward):.3f} | canonical present: {canonical is not None}"
    )
    if canonical is not None:
        sel = canonical.get("selection")
        op = canonical.get("operation")
        print(
            f"  canonical.operation={int(op)} | canonical.selection.shape={tuple(sel.shape) if hasattr(sel, 'shape') else 'N/A'}"
        )


def add_channel_dim_demo(base_env, env_params) -> None:
    section("AddChannelDimWrapper Demo")
    env = AddChannelDimWrapper(base_env)

    key = jax.random.PRNGKey(6)
    state, ts = env.reset(key, env_params=env_params)
    print(f"Obs shape with channel: {tuple(ts.observation.shape)} (expect H, W, 1)")

    # Check observation space matches actual shape
    obs_space = env.observation_space(env_params)
    print(f"Observation space shape: {getattr(obs_space, 'shape', None)}")


def jax_compat_demo(base_env, env_params) -> None:
    section("JAX Compatibility Demo (jit + vmap)")

    # JIT sample from base env's action space (canonical, mask-based)
    action_space = base_env.action_space(env_params)

    @jax.jit
    def sample_action(key):
        return action_space.sample(key)

    key = jax.random.PRNGKey(7)
    sample = sample_action(key)
    print(
        f"JIT sample ok -> operation={int(sample['operation'])}, selection.sum={int(jnp.sum(sample['selection']))}"
    )

    # vmap reset for batch processing
    batch_size = 4
    keys = jax.random.split(key, batch_size)

    batched_reset = jax.vmap(base_env.reset, in_axes=(0, None))
    states, timesteps = batched_reset(keys, env_params)
    print(
        f"Batch reset OK -> obs batch shape: {tuple(timesteps.observation.shape)} | all first(): {bool(jnp.all(timesteps.first()))}"
    )


def main() -> None:
    print("\nWrappers Overview: Simple and Comprehensive Demo")
    print("(This file supersedes previous wrapper demos.)")

    base_env, env_params = spaces_and_reset_demo()
    bbox_wrapper_demo(base_env, env_params)
    point_wrapper_demo(base_env, env_params)
    flatten_wrapper_demo(base_env, env_params)
    add_channel_dim_demo(base_env, env_params)
    jax_compat_demo(base_env, env_params)

    print("\nAll wrapper demos completed successfully.")


if __name__ == "__main__":
    main()
