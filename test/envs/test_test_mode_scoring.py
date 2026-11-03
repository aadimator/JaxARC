"""Tests that test-mode episodes are scored against the true test output."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxarc import JaxArcConfig
from jaxarc.configs import GridInitializationConfig
from jaxarc.envs.actions import create_action
from jaxarc.registration import available_task_ids, make
from jaxarc.types import StepType

SUBMIT = 34
FILL_COLOR_0 = 0
N = 200


@pytest.fixture(scope="module")
def test_env():
    """ConceptARC in test mode, with one batch of resets and their true answers."""
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
    env, params = make("Concept-all", config=config, episode_mode=1)

    keys = jax.random.split(jax.random.PRNGKey(0), N)
    state, _ = jax.vmap(env.reset)(keys)

    t, p = state.task_idx, state.pair_idx
    answer = params.buffer.true_test_output_grids[t, p]
    mask = params.buffer.true_test_output_masks[t, p]
    return env, params, state, answer, mask


def set_working(state, grid, mask):
    return eqx.tree_at(
        lambda s: (s.working_grid, s.working_grid_mask), state, (grid, mask)
    )


def step_all(env, state, operation):
    selection = jnp.zeros_like(state.working_grid[0], dtype=jnp.bool_)
    action = create_action(operation, selection)
    return jax.vmap(env.step, in_axes=(0, None))(state, action)


def test_correct_submission_is_solved(test_env):
    env, params, state, answer, mask = test_env

    new_state, timestep = step_all(env, set_working(state, answer, mask), SUBMIT)

    expected = params.reward.success_bonus + params.reward.step_penalty
    assert jnp.all(new_state.similarity_score == 1.0)
    assert jnp.all(timestep.step_type == StepType.TERMINATED)
    assert jnp.allclose(timestep.reward, expected)


def test_wrong_submission_is_not_solved(test_env):
    env, params, state, answer, mask = test_env

    # Change one cell of each answer. Every grid has a valid cell at (0, 0).
    assert jnp.all(mask[:, 0, 0])
    wrong = answer.at[:, 0, 0].set((answer[:, 0, 0] + 1) % 10)
    new_state, timestep = step_all(env, set_working(state, wrong, mask), SUBMIT)

    expected = params.reward.unsolved_submission_penalty + params.reward.step_penalty
    cells = mask.sum(axis=(1, 2))
    assert jnp.allclose(new_state.similarity_score, (cells - 1) / cells)
    assert jnp.all(timestep.step_type == StepType.TERMINATED)
    assert jnp.allclose(timestep.reward, expected)


def test_reset_scores_the_starting_grid(test_env):
    _, _, state, _, _ = test_env

    assert jnp.any(state.similarity_score > 0.0)


def test_answer_stays_hidden_from_state(test_env):
    env, _, state, answer, mask = test_env

    new_state, _ = step_all(env, set_working(state, answer, mask), FILL_COLOR_0)

    for s in (state, new_state):
        assert not jnp.any(s.target_grid_mask)
        assert jnp.all(s.target_grid == s.target_grid[0, 0, 0])


def test_reward_gives_no_hint_before_submit(test_env):
    env, params, state, answer, mask = test_env

    # An empty selection changes nothing, so only the step penalty should apply,
    # even though the working grid already holds the answer.
    _, timestep = step_all(env, set_working(state, answer, mask), FILL_COLOR_0)

    assert jnp.all(timestep.step_type == StepType.MID)
    assert jnp.allclose(timestep.reward, params.reward.step_penalty)


def test_train_mode_still_scores_against_demo_output():
    config = JaxArcConfig()
    if not available_task_ids("Mini", config=config):
        pytest.skip("No Mini tasks available")
    env, params = make("Mini-all", config=config, episode_mode=0)
    state, _ = env.reset(jax.random.PRNGKey(0))

    answer = params.buffer.output_grids_examples[state.task_idx, state.pair_idx]
    mask = params.buffer.output_masks_examples[state.task_idx, state.pair_idx]
    selection = jnp.zeros_like(answer, dtype=jnp.bool_)
    new_state, timestep = env.step(
        set_working(state, answer, mask), create_action(SUBMIT, selection)
    )

    assert new_state.similarity_score == 1.0
    assert timestep.step_type == StepType.TERMINATED
