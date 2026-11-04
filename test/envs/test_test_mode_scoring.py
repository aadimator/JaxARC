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

RESIZE = 33
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


def test_submitting_the_untouched_grid_only_solves_exact_matches():
    config = JaxArcConfig()
    if not available_task_ids("Concept", config=config):
        pytest.skip("No Concept tasks available")
    env, params = make("Concept-all", config=config, episode_mode=1)
    tasks, pairs = params.all_pairs()

    def submit_untouched(task, pair):
        state, _ = env.reset(
            jax.random.PRNGKey(0), env_params=params.with_pair(task, pair)
        )
        selection = jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
        return (state, *env.step(state, create_action(SUBMIT, selection)))

    with jax.disable_jit(False):
        start, final, timestep = jax.vmap(submit_untouched)(tasks, pairs)

    answer = params.buffer.true_test_output_grids[tasks, pairs]
    mask = params.buffer.true_test_output_masks[tasks, pairs]
    same_shape = jnp.all(start.working_grid_mask == mask, axis=(1, 2))
    same_cells = jnp.all((start.working_grid == answer) | ~mask, axis=(1, 2))
    oversized = jnp.all(mask <= start.working_grid_mask, axis=(1, 2)) & ~same_shape

    # Guards against a vacuous pass: these are the pairs that used to score 1.0.
    assert jnp.any(oversized & same_cells)
    assert jnp.array_equal(final.similarity_score == 1.0, same_shape & same_cells)
    assert jnp.array_equal(
        timestep.reward == params.reward.success_bonus + params.reward.step_penalty,
        same_shape & same_cells,
    )


def test_oversized_grid_can_be_resized_and_solved():
    config = JaxArcConfig()
    if not available_task_ids("Concept", config=config):
        pytest.skip("No Concept tasks available")
    env, params = make("Concept-all", config=config, episode_mode=1)
    tasks, pairs = params.all_pairs()

    with jax.disable_jit(False):
        input_masks = params.buffer.test_input_masks[tasks, pairs]
        output_masks = params.buffer.true_test_output_masks[tasks, pairs]
        shared_cell = jnp.any(
            (
                params.buffer.test_input_grids[tasks, pairs]
                == params.buffer.true_test_output_grids[tasks, pairs]
            )
            & output_masks,
            axis=(1, 2),
        )
        oversized = (
            jnp.all(output_masks <= input_masks, axis=(1, 2))
            & jnp.any(output_masks != input_masks, axis=(1, 2))
            & shared_cell
        )
        index = int(jnp.argmax(oversized))
        assert oversized[index]
        task, pair = tasks[index], pairs[index]

        answer = params.buffer.true_test_output_grids[task, pair]
        mask = output_masks[index]
        state, _ = env.reset(
            jax.random.PRNGKey(0), env_params=params.with_pair(task, pair)
        )
        start_score = state.similarity_score

        height = int(mask.any(axis=1).sum())
        width = int(mask.any(axis=0).sum())
        corner = jnp.zeros_like(mask).at[height - 1, width - 1].set(True)
        state, _ = env.step(state, create_action(RESIZE, corner))
        assert jnp.array_equal(state.working_grid_mask, mask)
        assert state.similarity_score > start_score

        for color in jnp.unique(answer[mask]).tolist():
            state, _ = env.step(state, create_action(color, (answer == color) & mask))

        final, timestep = env.step(
            state, create_action(SUBMIT, jnp.zeros_like(mask))
        )

    assert final.similarity_score == 1.0
    assert timestep.step_type == StepType.TERMINATED
    assert jnp.isclose(
        timestep.reward, params.reward.success_bonus + params.reward.step_penalty
    )
