"""Tests that reset() starts each episode from the selected pair's input."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig

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


@pytest.mark.parametrize(
    "config",
    [JaxArcConfig(), JaxArcConfig.from_hydra(DictConfig({}))],
    ids=["python", "empty_hydra"],
)
def test_default_config_starts_from_selected_input(config):
    if not available_task_ids("Mini", config=config):
        pytest.skip("No Mini tasks available")
    env, params = make("Mini-all", config=config)

    keys = jax.random.split(jax.random.PRNGKey(0), 200)
    state, _ = jax.vmap(lambda key: env.reset(key, env_params=params))(keys)

    selected_input = params.buffer.input_grids_examples[state.task_idx, state.pair_idx]
    assert jnp.array_equal(state.working_grid, selected_input)


# --- Zero-weight modes are pruned at trace time --------------------------------------
# reset() runs under vmap, where lax.switch evaluates every branch for every env. A mode
# with weight 0 must not cost anything: the colour-remap permutation sorts 10 colours per
# env, and on jax 0.5.3 that sort was 94% of a step on an H100.

from jaxarc.envs.grid_initialization import initialize_working_grids  # noqa: E402

GRID = jnp.arange(25, dtype=jnp.int32).reshape(5, 5) % 10
MASK = jnp.ones((5, 5), dtype=bool)


def lowered_init(config: GridInitializationConfig, n: int = 8) -> str:
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    init = jax.vmap(lambda key: initialize_working_grids(GRID, MASK, config, key))
    return jax.jit(init).lower(keys).as_text()


def test_default_config_reset_has_no_sort():
    assert "sort" not in lowered_init(GridInitializationConfig())


def test_reset_keeps_the_sort_when_colour_remap_has_weight():
    config = GridInitializationConfig(
        demo_weight=1.0, permutation_weight=1.0, permutation_types=("color_remap",)
    )

    assert "sort" in lowered_init(config)


def test_zero_weight_modes_are_never_drawn_in_a_mixture():
    config = GridInitializationConfig(demo_weight=1.0, empty_weight=1.0)
    keys = jax.random.split(jax.random.PRNGKey(0), 200)
    grids, masks = jax.vmap(
        lambda key: initialize_working_grids(GRID, MASK, config, key)
    )(keys)
    grids = grids[:, 0]

    is_demo = jnp.all(grids == GRID, axis=(1, 2))
    is_empty = jnp.all(grids == 0, axis=(1, 2))
    assert jnp.all(is_demo | is_empty)
    assert jnp.any(is_demo)
    assert jnp.any(is_empty)
    assert jnp.all(masks)


def test_pruning_keeps_the_mode_draws_of_a_mixture():
    # The same key gives the same mode with or without a zero-weight mode in the config:
    # pruning removes work, not randomness.
    mixture = GridInitializationConfig(demo_weight=1.0, empty_weight=3.0)
    with_zero = GridInitializationConfig(
        demo_weight=1.0, empty_weight=3.0, random_weight=0.0, permutation_weight=0.0
    )
    keys = jax.random.split(jax.random.PRNGKey(1), 64)

    def init(config):
        return jax.vmap(lambda k: initialize_working_grids(GRID, MASK, config, k))(
            keys
        )[0]

    assert jnp.array_equal(init(mixture), init(with_zero))
