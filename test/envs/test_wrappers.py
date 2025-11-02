from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc.envs import (
    BboxActionWrapper,
    FlattenActionWrapper,
    PointActionWrapper,
)
from jaxarc.registration import make


def _runner(env, params, num_steps: int = 3):
    action_space = env.action_space(params)

    def _single(key):
        reset_key, loop_key = jax.random.split(key)
        state, ts0 = env.reset(reset_key, env_params=params)

        def body(carry, _):
            state, ts, k = carry
            k_reset, k_action, k_next = jax.random.split(k, 3)

            def do_reset(_carry):
                _state, _ts = _carry
                return env.reset(k_reset, env_params=params)

            def keep(_carry):
                return _carry

            state, ts = jax.lax.cond(ts.last(), do_reset, keep, (state, ts))
            act = action_space.sample(k_action)
            new_state, new_ts = env.step(state, act, env_params=params)
            return (new_state, new_ts, k_next), ()

        (state_final, ts_final, _), _ = jax.lax.scan(
            body, (state, ts0, loop_key), xs=None, length=num_steps
        )
        return ts_final

    return jax.jit(_single)


def test_point_wrapper_extras_are_jax_compatible():
    env, params = make("Mini", auto_download=True)
    env = PointActionWrapper(env)

    fn = _runner(env, params, num_steps=3)
    key = jax.random.PRNGKey(0)
    ts_final = fn(key)

    # Ensure we got a valid timestep with JAX arrays
    assert hasattr(ts_final, "observation")
    assert isinstance(ts_final.observation, jax.Array)

    # Extras may be a pytree; ensure no string leaves are present
    leaves = jax.tree_util.tree_leaves(ts_final)
    for leaf in leaves:
        assert not isinstance(leaf, str)


def test_flatten_wrapper_extras_are_jax_compatible():
    env, params = make("Mini", auto_download=True)
    env = PointActionWrapper(env)
    env = FlattenActionWrapper(env)

    fn = _runner(env, params, num_steps=3)
    key = jax.random.PRNGKey(1)
    ts_final = fn(key)

    assert hasattr(ts_final, "observation")
    assert isinstance(ts_final.observation, jax.Array)

    leaves = jax.tree_util.tree_leaves(ts_final)
    for leaf in leaves:
        assert not isinstance(leaf, str)


def _grid_hw_from_reset(env, params):
    key = jax.random.PRNGKey(0)
    _state, ts = env.reset(key, env_params=params)
    h, w, _ = (
        int(ts.observation.shape[0]),
        int(ts.observation.shape[1]),
        int(ts.observation.shape[2]),
    )
    return h, w


def test_reset_extras_canonical_action_present_and_jax_arrays():
    env, params = make("Mini-Most_Common_color_l6ab0lf3xztbyxsu3p", auto_download=True)
    key = jax.random.PRNGKey(123)
    state, ts = env.reset(key, env_params=params)

    assert isinstance(ts.extras, dict)
    assert "canonical_action" in ts.extras
    assert "operation_id" in ts.extras

    ca = ts.extras["canonical_action"]
    assert isinstance(ca, dict)
    assert "operation" in ca
    assert "selection" in ca

    # JAX arrays, correct dtypes and shapes
    assert isinstance(ca["operation"], jax.Array)
    assert ca["operation"].dtype == jnp.int32
    assert isinstance(ca["selection"], jax.Array)
    assert ca["selection"].dtype == jnp.bool_

    h, w, _ = (
        int(ts.observation.shape[0]),
        int(ts.observation.shape[1]),
        int(ts.observation.shape[2]),
    )
    assert tuple(ca["selection"].shape) == (h, w)


def test_point_wrapper_single_point_mask_and_clipping():
    base_env, params = make(
        "Mini-Most_Common_color_l6ab0lf3xztbyxsu3p", auto_download=True
    )
    env = PointActionWrapper(base_env)

    h, w = _grid_hw_from_reset(env, params)

    key = jax.random.PRNGKey(42)
    state, ts = env.reset(key, env_params=params)

    # Deliberately out-of-bounds point to test clipping
    action = {"operation": 1, "row": -5, "col": w + 10}
    state, ts = env.step(state, action, env_params=params)

    ca = ts.extras["canonical_action"]
    sel = ca["selection"]
    op = ca["operation"]

    assert int(op) == 1
    assert sel.dtype == jnp.bool_
    assert int(sel.sum()) == 1
    # Expected clipped indices
    clipped_r, clipped_c = 0, w - 1
    assert bool(sel[clipped_r, clipped_c]) is True


def test_bbox_wrapper_rectangle_mask_and_ordering():
    base_env, params = make(
        "Mini-Most_Common_color_l6ab0lf3xztbyxsu3p", auto_download=True
    )
    env = BboxActionWrapper(base_env)

    h, w = _grid_hw_from_reset(env, params)

    key = jax.random.PRNGKey(7)
    state, ts = env.reset(key, env_params=params)

    # Provide reversed coords to test min/max ordering and inclusive bounds
    action = {"operation": 2, "r1": h + 5, "c1": w + 5, "r2": -3, "c2": -4}
    state, ts = env.step(state, action, env_params=params)

    ca = ts.extras["canonical_action"]
    sel = ca["selection"]

    # Expected rectangle is full grid after clipping
    assert sel.dtype == jnp.bool_
    assert int(sel.sum()) == h * w


def test_flatten_wrapper_action_space_and_step_point():
    base_env, params = make(
        "Mini-Most_Common_color_l6ab0lf3xztbyxsu3p", auto_download=True
    )
    # Flatten over point dict space
    dict_env = PointActionWrapper(base_env)
    env = FlattenActionWrapper(dict_env)

    # Validate action space size equals product of (operation, row, col)
    dict_space = dict_env.action_space(params)
    n_ops = int(getattr(dict_space.spaces["operation"], "num_values", 0))
    h = int(getattr(dict_space.spaces["row"], "num_values", 0))
    w = int(getattr(dict_space.spaces["col"], "num_values", 0))

    flat_space = env.action_space(params)
    num_values = int(getattr(flat_space, "num_values", 0))
    assert num_values == n_ops * h * w

    # Pick a known triple and compose flat index with the same radix order used in wrapper
    op, r, c = 1, min(2, h - 1), min(3, w - 1)
    # Flattening order follows component order in FlattenActionWrapper: [operation, row, col]
    # Flat index = ((op) * (h*w)) + (r * w) + c
    flat_index = (op * (h * w)) + (r * w) + c

    key = jax.random.PRNGKey(9)
    state, ts = env.reset(key, env_params=params)
    state, ts = env.step(
        state, jnp.asarray(flat_index, dtype=jnp.int32), env_params=params
    )

    ca = ts.extras["canonical_action"]
    sel = ca["selection"]
    got_op = int(ca["operation"])  # jax/numpy scalars coerce via int()

    assert got_op == op
    assert int(sel.sum()) == 1
    assert bool(sel[r, c]) is True


def test_extras_are_jax_compatible_after_step():
    # Sanity check to ensure no strings in jitted leaves
    base_env, params = make(
        "Mini-Most_Common_color_l6ab0lf3xztbyxsu3p", auto_download=True
    )
    env = PointActionWrapper(base_env)

    def runner(key):
        state, ts = env.reset(key, env_params=params)
        action = {"operation": 0, "row": 0, "col": 0}
        state, ts = env.step(state, action, env_params=params)
        return ts

    compiled = jax.jit(runner)
    ts_final = compiled(jax.random.PRNGKey(0))

    leaves = jax.tree_util.tree_leaves(ts_final)
    for leaf in leaves:
        assert not isinstance(leaf, str)
