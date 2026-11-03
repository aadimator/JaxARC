"""Tests for starting an episode from a chosen task and pair."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxarc import JaxArcConfig
from jaxarc.configs import GridInitializationConfig
from jaxarc.envs.actions import create_action
from jaxarc.registration import available_task_ids, make
from jaxarc.wrappers import (
    AnswerObservationWrapper,
    ContextualObservationWrapper,
    FlattenActionWrapper,
    InputGridObservationWrapper,
    PointActionWrapper,
)

TRAIN, TEST = 0, 1
SUBMIT = 34


@pytest.fixture(scope="module", params=[TRAIN, TEST], ids=["train", "test"])
def concept_env(request):
    config = JaxArcConfig(
        grid_initialization=GridInitializationConfig(
            demo_weight=1.0,
            permutation_weight=0.0,
            empty_weight=0.0,
            random_weight=0.0,
        )
    )
    if not available_task_ids("Concept", config=config):
        pytest.skip("No Concept tasks available")
    env, params = make("Concept-all", config=config, episode_mode=request.param)
    return env, params


def pair_arrays(params):
    """The input and output arrays for the params' episode mode."""
    b = params.buffer
    if params.episode_mode == TRAIN:
        return (
            b.input_grids_examples,
            b.input_masks_examples,
            b.output_grids_examples,
            b.output_masks_examples,
        )
    return (
        b.test_input_grids,
        b.test_input_masks,
        b.true_test_output_grids,
        b.true_test_output_masks,
    )


def base_state(state):
    while hasattr(state, "base_env_state"):
        state = state.base_env_state
    return state


def test_all_pairs_lists_every_pair_once(concept_env):
    _, params = concept_env
    task_idx, pair_idx = params.all_pairs()

    b = params.buffer
    counts = b.num_train_pairs if params.episode_mode == TRAIN else b.num_test_pairs
    assert task_idx.shape == pair_idx.shape == (int(counts.sum()),)
    assert jnp.all(pair_idx < counts[task_idx])
    pairs = jnp.stack([task_idx, pair_idx], axis=1)
    assert jnp.unique(pairs, axis=0).shape[0] == pairs.shape[0]


def test_all_pairs_respects_subset(concept_env):
    _, params = concept_env
    subset = jnp.array([3, 7])
    subset_params = eqx.tree_at(
        lambda p: p.subset_indices, params, subset, is_leaf=lambda x: x is None
    )

    task_idx, _ = subset_params.all_pairs()

    assert set(task_idx.tolist()) == {3, 7}


def test_reset_starts_from_chosen_pair(concept_env):
    env, params = concept_env
    task_idx, pair_idx = params.all_pairs()
    keys = jax.random.split(jax.random.PRNGKey(0), task_idx.shape[0])

    def reset(key, t, p):
        return env.reset(key, params.with_pair(t, p))[0]

    with jax.disable_jit(False):
        state = jax.jit(jax.vmap(reset))(keys, task_idx, pair_idx)

    inputs, masks, _, _ = pair_arrays(params)
    assert jnp.array_equal(state.task_idx, task_idx)
    assert jnp.array_equal(state.pair_idx, pair_idx)
    assert jnp.array_equal(state.working_grid, inputs[task_idx, pair_idx])
    assert jnp.array_equal(state.working_grid_mask, masks[task_idx, pair_idx])


def test_correct_answer_on_every_chosen_pair_is_solved(concept_env):
    env, params = concept_env
    task_idx, pair_idx = params.all_pairs()
    _, _, outputs, output_masks = pair_arrays(params)
    keys = jax.random.split(jax.random.PRNGKey(0), task_idx.shape[0])
    submit = create_action(SUBMIT, jnp.zeros(outputs.shape[-2:], dtype=jnp.bool_))

    def solve(key, i):
        t, p = task_idx[i], pair_idx[i]
        state, _ = env.reset(key, params.with_pair(t, p))
        state = eqx.tree_at(
            lambda s: (s.working_grid, s.working_grid_mask),
            state,
            (outputs[t, p], output_masks[t, p]),
        )
        return env.step(state, submit)

    def solve_all():
        return jax.vmap(solve)(keys, jnp.arange(task_idx.shape[0]))

    # Nothing is passed in, so the pairs are constants under jit, as in an evaluator.
    with jax.disable_jit(False):
        state, timestep = jax.jit(solve_all)()

    # Train mode adds similarity shaping on top, so this is a lower bound there.
    lowest = params.reward.success_bonus + params.reward.step_penalty
    assert jnp.all(state.similarity_score == 1.0)
    assert jnp.all(timestep.reward >= lowest - 1e-5)


@pytest.mark.parametrize("wrapped", [False, True], ids=["bare", "wrapped"])
def test_chosen_pair_matches_random_reset_on_same_pair(concept_env, wrapped):
    env, params = concept_env
    if wrapped:
        env = FlattenActionWrapper(PointActionWrapper(env))
        env = AnswerObservationWrapper(env)
        env = InputGridObservationWrapper(env)
        env = ContextualObservationWrapper(env)
    keys = jax.random.split(jax.random.PRNGKey(5), 64)

    random_state, random_step = jax.vmap(env.reset)(keys)
    inner = base_state(random_state)
    chosen_state, chosen_step = jax.vmap(
        lambda key, t, p: env.reset(key, params.with_pair(t, p))
    )(keys, inner.task_idx, inner.pair_idx)

    for left, right in zip(
        jax.tree_util.tree_leaves((random_state, random_step.observation)),
        jax.tree_util.tree_leaves((chosen_state, chosen_step.observation)),
        strict=True,
    ):
        assert jnp.array_equal(left, right)


def test_reset_stays_random_when_no_pair_is_chosen(concept_env):
    env, params = concept_env
    assert params.task_idx is None
    assert params.pair_idx is None

    keys = jax.random.split(jax.random.PRNGKey(0), 200)
    state, _ = jax.vmap(env.reset)(keys)

    pairs = jnp.stack([state.task_idx, state.pair_idx], axis=1)
    assert jnp.unique(pairs, axis=0).shape[0] > 1
