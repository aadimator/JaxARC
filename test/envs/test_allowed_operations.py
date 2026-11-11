"""Tests that an agent can only choose from allowed_operations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from jaxarc import JaxArcConfig
from jaxarc.envs.actions import create_action
from jaxarc.registration import available_task_ids
from jaxarc.stoix_adapter import make_jaxarc_env
from jaxarc.types import StepType

RAW_PROFILE = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 33, 34)
RESIZE_INDEX, SUBMIT_INDEX = 10, 11
CELLS = 25  # Mini-all grids are 5x5.


def make_envs(allowed_operations=RAW_PROFILE, mode="point"):
    if not available_task_ids("Mini", config=JaxArcConfig()):
        pytest.skip("No Mini tasks available")
    action = {}
    if allowed_operations is not None:
        action["allowed_operations"] = list(allowed_operations)
    return make_jaxarc_env(
        OmegaConf.create(
            {
                "action": action,
                "env": {
                    "scenario": {"name": "Mini-all"},
                    "action": {"mode": mode},
                    "observation_wrappers": {},
                },
            }
        )
    )


def flat_action(operation_index, row, col):
    return jnp.asarray((operation_index * CELLS) + row * 5 + col, dtype=jnp.int32)


def test_action_count_matches_the_allowed_operations():
    train_env, eval_env = make_envs()

    assert int(train_env.action_space().num_values) == 11 * CELLS + 1
    assert int(eval_env.action_space().num_values) == 11 * CELLS + 1


def test_bbox_action_count_matches_the_allowed_operations():
    train_env, eval_env = make_envs(mode="bbox")

    assert int(train_env.action_space().num_values) == 11 * CELLS**2 + 1
    assert int(eval_env.action_space().num_values) == 11 * CELLS**2 + 1


def test_every_operation_stays_available_without_a_profile():
    train_env, eval_env = make_envs(allowed_operations=None)

    assert int(train_env.action_space().num_values) == 33 * CELLS + 2
    assert int(eval_env.action_space().num_values) == 33 * CELLS + 2


def test_every_action_runs_an_allowed_operation():
    env, _ = make_envs()
    base = env.unwrapped
    state, _ = env.reset(jax.random.PRNGKey(0))

    # Submit ignores the selection, so it gets one action, at cell (0, 0).
    per_cell = len(RAW_PROFILE) - 1
    operations = jnp.append(jnp.repeat(jnp.asarray(RAW_PROFILE[:-1]), CELLS), 34)
    rows = jnp.append(jnp.tile(jnp.repeat(jnp.arange(5), 5), per_cell), 0)
    cols = jnp.append(jnp.tile(jnp.arange(5), 5 * per_cell), 0)
    actions = jnp.arange(per_cell * CELLS + 1)
    selections = (
        jnp.zeros((actions.shape[0], *state.working_grid.shape), dtype=jnp.bool_)
        .at[actions, rows, cols]
        .set(True)
    )

    def by_flat_action(action):
        return env.step(state, action)

    def by_operation(operation, selection):
        return base.step(state, create_action(operation, selection))

    with jax.disable_jit(False):
        chosen, chosen_step = jax.jit(jax.vmap(by_flat_action))(actions)
        expected, expected_step = jax.jit(jax.vmap(by_operation))(
            operations, selections
        )

    assert jnp.array_equal(chosen.working_grid, expected.working_grid)
    assert jnp.array_equal(chosen.working_grid_mask, expected.working_grid_mask)
    assert jnp.array_equal(chosen_step.step_type, expected_step.step_type)
    assert jnp.array_equal(chosen_step.reward, expected_step.reward)


def test_resize_and_submit_are_the_last_two_choices():
    env, _ = make_envs()
    state, _ = env.reset(jax.random.PRNGKey(0))

    resized, _ = env.step(state, flat_action(RESIZE_INDEX, 0, 0))
    assert int(resized.working_grid_mask.sum()) == 1

    _, timestep = env.step(state, flat_action(SUBMIT_INDEX, 0, 0))
    assert int(timestep.step_type) == int(StepType.TERMINATED)
