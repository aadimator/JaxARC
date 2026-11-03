"""Tests that reset() starts each episode from the selected pair's input."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxarc import JaxArcConfig
from jaxarc.configs import GridInitializationConfig
from jaxarc.registration import available_task_ids, make

TRAIN, TEST = 0, 1
MODES = {
    "demo": {"demo_weight": 1.0},
    "permutation": {"permutation_weight": 1.0, "permutation_types": ("reflect",)},
    "empty": {"empty_weight": 1.0},
    "random": {"random_weight": 1.0},
}


def reset_batch(mode: str, episode_mode: int, n: int = 200):
    """Reset n environments on ConceptARC, whose grid sizes vary within a task."""
    weights = {
        "demo_weight": 0.0,
        "permutation_weight": 0.0,
        "empty_weight": 0.0,
        "random_weight": 0.0,
        **MODES[mode],
    }
    config = JaxArcConfig(grid_initialization=GridInitializationConfig(**weights))
    if not available_task_ids("Concept", config=config):
        pytest.skip("No Concept tasks available")
    env, params = make("Concept-all", config=config, episode_mode=episode_mode)

    keys = jax.random.split(jax.random.PRNGKey(0), n)
    state, _ = jax.vmap(lambda key: env.reset(key, env_params=params))(keys)

    buffer = params.buffer
    if episode_mode == TRAIN:
        # Guards against a vacuous pass: mismatches need tasks with several pairs.
        assert jnp.any(buffer.num_train_pairs[state.task_idx] > 1)
        inputs = buffer.input_grids_examples
        masks = buffer.input_masks_examples
    else:
        inputs = buffer.test_input_grids
        masks = buffer.test_input_masks
    return (
        state,
        inputs[state.task_idx, state.pair_idx],
        masks[state.task_idx, state.pair_idx],
        buffer,
    )


@pytest.mark.parametrize("episode_mode", [TRAIN, TEST])
def test_demo_starts_from_selected_input(episode_mode):
    state, selected_input, selected_mask, _ = reset_batch("demo", episode_mode)

    assert jnp.array_equal(state.working_grid, selected_input)
    assert jnp.array_equal(state.working_grid_mask, selected_mask)


def test_target_and_working_grid_come_from_same_pair():
    state, selected_input, _, buffer = reset_batch("demo", TRAIN)
    target = buffer.output_grids_examples[state.task_idx, state.pair_idx]

    assert jnp.array_equal(state.working_grid, selected_input)
    assert jnp.array_equal(state.target_grid, target)


@pytest.mark.parametrize("episode_mode", [TRAIN, TEST])
def test_permutation_transforms_selected_input(episode_mode):
    state, selected_input, _, _ = reset_batch("permutation", episode_mode)

    candidates = jnp.stack(
        [selected_input, selected_input[:, :, ::-1], selected_input[:, ::-1, :]]
    )
    matches = jnp.all(state.working_grid == candidates, axis=(2, 3))
    assert jnp.all(jnp.any(matches, axis=0))


@pytest.mark.parametrize("episode_mode", [TRAIN, TEST])
def test_empty_uses_selected_mask(episode_mode):
    state, _, selected_mask, _ = reset_batch("empty", episode_mode)

    assert jnp.all(state.working_grid == 0)
    assert jnp.array_equal(state.working_grid_mask, selected_mask)


@pytest.mark.parametrize("episode_mode", [TRAIN, TEST])
def test_random_uses_selected_mask(episode_mode):
    state, _, selected_mask, _ = reset_batch("random", episode_mode)

    assert jnp.array_equal(state.working_grid_mask, selected_mask)
    assert jnp.all(state.working_grid[~selected_mask] == 0)
