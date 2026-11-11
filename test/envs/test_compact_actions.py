"""Tests for the flat point and bbox action layouts built by make_jaxarc_env.

Operations that use the selection get one action per cell (point) or box
(bbox), op-major. Operations that ignore it (Copy input, Submit) come last,
with one action each instead of one per cell or box.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf
from stoa import AddRNGKey, AutoResetWrapper
from stoa.core_wrappers.vmap import VmapWrapper

from jaxarc import JaxArcConfig
from jaxarc.envs.actions import create_action
from jaxarc.registration import available_task_ids
from jaxarc.stoix_adapter import make_jaxarc_env
from jaxarc.types import StepType
from jaxarc.wrappers import (
    AllowedOperationsWrapper,
    BboxActionWrapper,
    FlattenActionWrapper,
    PointActionWrapper,
    SelectionAwareFlattenWrapper,
)

RAW = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 33, 34)
ALL = tuple(range(35))
COPY_INPUT, SUBMIT = 32, 34
SIZE = 5  # Mini-all grids are 5x5.
CELLS = SIZE * SIZE
CHOICES = {"point": CELLS, "bbox": CELLS**2}


def make_envs(allowed_operations=RAW, mode="point", scenario="Mini-all"):
    dataset = scenario.split("-")[0]
    if not available_task_ids(dataset, config=JaxArcConfig()):
        pytest.skip(f"No {dataset} tasks available")
    action = {}
    if allowed_operations is not None:
        action["allowed_operations"] = list(allowed_operations)
    return make_jaxarc_env(
        OmegaConf.create(
            {
                "action": action,
                "env": {
                    "scenario": {"name": scenario},
                    "action": {"mode": mode},
                    "observation_wrappers": {},
                },
            }
        )
    )


def per_choice_env(env, mode, operations):
    """The same base env with one action per (operation, cell or box)."""
    wrapped = {"point": PointActionWrapper, "bbox": BboxActionWrapper}[mode](
        env.unwrapped
    )
    if operations is not None:
        wrapped = AllowedOperationsWrapper(wrapped)
    return FlattenActionWrapper(wrapped)


def selection(mode, coords):
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    if mode == "point":
        mask[coords] = True
    else:
        r1, c1, r2, c2 = coords
        mask[min(r1, r2) : max(r1, r2) + 1, min(c1, c2) : max(c1, c2) + 1] = True
    return mask


def layout(mode, operations):
    """Operation, selection mask and per-choice action for each flat action.

    Operations that ignore the selection have mask None and match the
    per-choice action with every coordinate 0.
    """
    coords = list(itertools.product(range(SIZE), repeat=2 if mode == "point" else 4))
    actions = []
    for slot, op in enumerate(operations):
        if op not in (COPY_INPUT, SUBMIT):
            for k, c in enumerate(coords):
                actions.append((op, selection(mode, c), slot * len(coords) + k))
    for slot, op in enumerate(operations):
        if op in (COPY_INPUT, SUBMIT):
            actions.append((op, None, slot * len(coords)))
    return actions


def edited_state(env):
    """A reset state with cell (0, 0) changed, so Copy input has work to do."""
    state, _ = env.reset(jax.random.PRNGKey(0))
    colour = (int(state.working_grid[0, 0]) + 1) % 10
    mask = jnp.zeros_like(state.working_grid_mask).at[0, 0].set(True)
    state, _ = env.unwrapped.step(state, create_action(colour, mask))
    return state


def step_all(env, state, actions):
    with jax.disable_jit(False):
        return jax.jit(jax.vmap(lambda a: env.step(state, a)))(
            jnp.asarray(actions, dtype=jnp.int32)
        )


@pytest.mark.parametrize(
    ("mode", "operations", "count"),
    [
        ("point", RAW, 11 * CELLS + 1),
        ("point", None, 33 * CELLS + 2),
        ("bbox", RAW, 11 * CELLS**2 + 1),
        ("bbox", None, 33 * CELLS**2 + 2),
    ],
)
def test_action_count(mode, operations, count):
    train_env, eval_env = make_envs(operations, mode)

    assert int(train_env.action_space().num_values) == count
    assert int(eval_env.action_space().num_values) == count


def test_point_action_count_at_30x30():
    env, _ = make_envs(scenario="AGI1-train")

    assert int(env.action_space().num_values) == 11 * 900 + 1


@pytest.mark.parametrize("mode", ["point", "bbox"])
@pytest.mark.parametrize("operations", [RAW, ALL])
def test_every_action_decodes_to_its_operation_and_selection(mode, operations):
    env, _ = make_envs(operations, mode)
    state = edited_state(env)
    expected = layout(mode, operations)

    _, timestep = step_all(env, state, range(len(expected)))

    chosen = timestep.extras["canonical_action"]
    assert np.array_equal(chosen["operation"], [op for op, *_ in expected])
    uses_selection = [i for i, (_, mask, _) in enumerate(expected) if mask is not None]
    masks = np.stack([expected[i][1] for i in uses_selection])
    assert np.array_equal(np.asarray(chosen["selection"])[uses_selection], masks)


@pytest.mark.parametrize("mode", ["point", "bbox"])
@pytest.mark.parametrize("operations", [RAW, ALL])
def test_every_action_steps_like_its_per_choice_action(mode, operations):
    env, _ = make_envs(operations, mode)
    reference = per_choice_env(env, mode, operations)
    state = edited_state(env)
    expected = layout(mode, operations)

    flat, flat_step = step_all(env, state, range(len(expected)))
    ref, ref_step = step_all(reference, state, [i for *_, i in expected])

    assert jnp.array_equal(flat.working_grid, ref.working_grid)
    assert jnp.array_equal(flat.working_grid_mask, ref.working_grid_mask)
    assert jnp.array_equal(flat.selected, ref.selected)
    assert jnp.array_equal(flat_step.step_type, ref_step.step_type)
    assert jnp.array_equal(flat_step.reward, ref_step.reward)


@pytest.mark.parametrize("mode", ["point", "bbox"])
def test_submit_and_copy_input_have_one_action_each(mode):
    env, _ = make_envs(None, mode)
    state = edited_state(env)

    _, timestep = step_all(env, state, range(33 * CHOICES[mode] + 2))
    operations = timestep.extras["operation_id"]

    assert int(jnp.sum(operations == SUBMIT)) == 1
    assert int(jnp.sum(operations == COPY_INPUT)) == 1
    assert int(operations[-1]) == SUBMIT
    assert int(operations[-2]) == COPY_INPUT
    for op in (0, 31, 33):
        assert int(jnp.sum(operations == op)) == CHOICES[mode]


def test_the_last_point_actions_resize_and_submit_with_raw_operations():
    env, _ = make_envs()
    state, _ = env.reset(jax.random.PRNGKey(0))

    resized, _ = env.step(state, jnp.int32(10 * CELLS + 1 * SIZE + 2))
    assert int(resized.working_grid_mask.sum()) == 2 * 3

    _, timestep = env.step(state, jnp.int32(11 * CELLS))
    assert int(timestep.step_type) == int(StepType.TERMINATED)


def test_works_under_vmap_and_auto_reset():
    env, _ = make_envs()
    env = VmapWrapper(AutoResetWrapper(AddRNGKey(env)), num_envs=4)
    submit = jnp.full((4,), 11 * CELLS, dtype=jnp.int32)
    fill = jnp.full((4,), 3 * CELLS + 2 * SIZE + 1, dtype=jnp.int32)

    with jax.disable_jit(False):
        state, _ = jax.jit(env.reset)(jax.random.PRNGKey(0))
        filled, fill_step = jax.jit(env.step)(state, fill)
        _, submit_step = jax.jit(env.step)(filled, submit)

    assert jnp.all(fill_step.step_type == StepType.MID)
    assert jnp.all(fill_step.extras["operation_id"] == 3)
    assert jnp.all(filled.working_grid[:, 2, 1] == 3)
    assert jnp.all(submit_step.step_type == StepType.TERMINATED)
    assert jnp.all(submit_step.extras["operation_id"] == SUBMIT)


@pytest.mark.parametrize("filtered", [True, False])
def test_direct_chain_decodes_by_the_operations_it_wraps(filtered):
    """Without AllowedOperationsWrapper the slot is the operation ID itself."""
    env, _ = make_envs()
    inner = PointActionWrapper(env.unwrapped)
    if filtered:
        inner = AllowedOperationsWrapper(inner)
    flat = SelectionAwareFlattenWrapper(inner)
    operations = RAW if filtered else ALL
    expected = layout("point", operations)
    state = edited_state(env)

    assert int(flat.action_space().num_values) == len(expected)
    _, timestep = step_all(flat, state, range(len(expected)))
    assert np.array_equal(timestep.extras["operation_id"], [op for op, *_ in expected])
