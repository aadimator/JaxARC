"""Tests that Resize reaches any grid size through env.step."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxarc import JaxArcConfig
from jaxarc.envs.actions import create_action
from jaxarc.registration import available_task_ids, make
from jaxarc.types import StepType
from jaxarc.wrappers import PointActionWrapper

FILL_COLOR_1 = 1
RESIZE = 33
SUBMIT = 34


@pytest.fixture(scope="module")
def concept():
    """ConceptARC in test mode, with every pair whose output doesn't fit the input."""
    config = JaxArcConfig()
    if not available_task_ids("Concept", config=config):
        pytest.skip("No Concept tasks available")
    env, params = make("Concept-all", config=config, episode_mode=1)

    tasks, pairs = params.all_pairs()
    inputs = params.buffer.test_input_masks[tasks, pairs]
    outputs = params.buffer.true_test_output_masks[tasks, pairs]
    needs_growth = jnp.any(outputs & ~inputs, axis=(1, 2))
    return env, params, tasks[needs_growth], pairs[needs_growth]


def reset_pair(env, params, task, pair):
    state, _ = env.reset(jax.random.PRNGKey(0), env_params=params.with_pair(task, pair))
    return state


def corner(state, height, width):
    empty = jnp.zeros_like(state.working_grid_mask)
    return empty.at[height - 1, width - 1].set(True)


def rectangle(state, height, width):
    rows = jnp.arange(state.working_grid.shape[0])[:, None]
    cols = jnp.arange(state.working_grid.shape[1])[None, :]
    return (rows < height) & (cols < width)


def test_resize_grows_and_keeps_the_existing_cells(concept):
    env, params, tasks, pairs = concept
    state = reset_pair(env, params, tasks[0], pairs[0])
    height = int(state.working_grid_mask.any(axis=1).sum()) + 2
    width = int(state.working_grid_mask.any(axis=0).sum()) + 3

    grown, _ = env.step(state, create_action(RESIZE, corner(state, height, width)))

    old = state.working_grid_mask
    new_cells = grown.working_grid_mask & ~old
    assert jnp.array_equal(grown.working_grid_mask, rectangle(state, height, width))
    assert jnp.array_equal(grown.working_grid[old], state.working_grid[old])
    assert jnp.all(grown.working_grid[new_cells] == 0)


def test_resize_reaches_the_largest_grid(concept):
    env, params, tasks, pairs = concept
    state = reset_pair(env, params, tasks[0], pairs[0])
    height, width = state.working_grid.shape

    grown, _ = env.step(state, create_action(RESIZE, corner(state, height, width)))

    assert jnp.all(grown.working_grid_mask)


def test_resize_grows_through_the_point_wrapper(concept):
    env, params, tasks, pairs = concept
    wrapped = PointActionWrapper(env)
    state = reset_pair(env, params, tasks[0], pairs[0])
    height = int(state.working_grid_mask.any(axis=1).sum()) + 1
    width = int(state.working_grid_mask.any(axis=0).sum()) + 1

    grown, _ = wrapped.step(
        state,
        {
            "operation": jnp.array(RESIZE),
            "row": jnp.array(height - 1),
            "col": jnp.array(width - 1),
        },
    )

    assert jnp.array_equal(grown.working_grid_mask, rectangle(state, height, width))


def test_other_operations_ignore_cells_outside_the_grid(concept):
    env, params, tasks, pairs = concept
    state = reset_pair(env, params, tasks[0], pairs[0])
    outside = jnp.zeros_like(state.working_grid_mask).at[-1, -1].set(True)
    assert not state.working_grid_mask[-1, -1]

    painted, _ = env.step(state, create_action(FILL_COLOR_1, outside))

    assert jnp.array_equal(painted.working_grid, state.working_grid)
    assert jnp.array_equal(painted.working_grid_mask, state.working_grid_mask)


def test_pairs_needing_a_larger_grid_can_be_solved(concept):
    env, params, tasks, pairs = concept
    assert tasks.shape[0] > 0

    with jax.disable_jit(False):
        step = jax.jit(env.step)
        for task, pair in zip(tasks, pairs):
            answer = params.buffer.true_test_output_grids[task, pair]
            mask = params.buffer.true_test_output_masks[task, pair]
            state = reset_pair(env, params, task, pair)

            height = int(mask.any(axis=1).sum())
            width = int(mask.any(axis=0).sum())
            state, _ = step(state, create_action(RESIZE, corner(state, height, width)))
            for color in jnp.unique(answer[mask]).tolist():
                state, _ = step(state, create_action(color, (answer == color) & mask))
            final, timestep = step(state, create_action(SUBMIT, jnp.zeros_like(mask)))

            assert final.similarity_score == 1.0
            assert timestep.step_type == StepType.TERMINATED
            assert jnp.isclose(
                timestep.reward,
                params.reward.success_bonus + params.reward.step_penalty,
            )
