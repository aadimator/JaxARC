"""Tests for what ExtendedMetrics reports at the end of an episode."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxarc import JaxArcConfig
from jaxarc.configs.environment_config import EnvironmentConfig
from jaxarc.envs.actions import create_action
from jaxarc.registration import available_task_ids, make
from jaxarc.types import StepType
from jaxarc.wrappers import ExtendedMetrics

RESIZE = 33
SUBMIT = 34
STEP_LIMIT = 15


@pytest.fixture(scope="module", params=[0, 1], ids=["train", "test"])
def game(request):
    """An env in each episode mode, reset on its first pair, with that pair's answer."""
    config = JaxArcConfig(environment=EnvironmentConfig(max_episode_steps=STEP_LIMIT))
    if not available_task_ids("Concept", config=config):
        pytest.skip("No Concept tasks available")
    base, params = make("Concept-Count", config=config, episode_mode=request.param)
    env = ExtendedMetrics(base)

    tasks, pairs = params.all_pairs()
    task, pair = int(tasks[0]), int(pairs[0])
    buffer = params.buffer
    if request.param == 0:
        answer = buffer.output_grids_examples[task, pair]
        answer_mask = buffer.output_masks_examples[task, pair]
    else:
        answer = buffer.true_test_output_grids[task, pair]
        answer_mask = buffer.true_test_output_masks[task, pair]

    state, timestep = env.reset(
        jax.random.PRNGKey(0), env_params=params.with_pair(task, pair)
    )
    return env, state, timestep, answer, answer_mask


def paint_answer(env, state, answer, answer_mask):
    """Resize to the answer's shape and fill it in, using real actions."""
    height = int(answer_mask.any(axis=1).sum())
    width = int(answer_mask.any(axis=0).sum())
    corner = jnp.zeros_like(answer_mask).at[height - 1, width - 1].set(True)
    state, timestep = env.step(state, create_action(RESIZE, corner))
    for color in range(10):
        selection = (answer == color) & answer_mask
        if selection.any():
            state, timestep = env.step(state, create_action(color, selection))
    assert float(state.base_env_state.similarity_score) == 1.0
    return state, timestep


def idle(env, state, timestep, answer_mask, until_step):
    nothing = jnp.zeros_like(answer_mask)
    while int(state.base_env_state.step_count) < until_step:
        state, timestep = env.step(state, create_action(0, nothing))
    return state, timestep


def submit(env, state, answer_mask):
    return env.step(state, create_action(SUBMIT, jnp.zeros_like(answer_mask)))


def metrics(timestep):
    return timestep.extras["episode_metrics"]


def test_submitting_the_answer_is_solved(game):
    env, state, _, answer, answer_mask = game
    state, _ = paint_answer(env, state, answer, answer_mask)
    state, timestep = submit(env, state, answer_mask)

    assert int(timestep.step_type) == int(StepType.TERMINATED)
    assert bool(metrics(timestep)["solved"])
    assert not bool(metrics(timestep)["was_truncated"])
    assert int(metrics(timestep)["steps_to_solve"]) == int(
        state.base_env_state.step_count
    )


def test_reaching_the_answer_without_submitting_is_not_solved(game):
    env, state, _, answer, answer_mask = game
    state, timestep = paint_answer(env, state, answer, answer_mask)
    _, timestep = idle(env, state, timestep, answer_mask, STEP_LIMIT)

    assert int(timestep.step_type) == int(StepType.TRUNCATED)
    assert float(metrics(timestep)["final_similarity"]) == 1.0
    assert not bool(metrics(timestep)["solved"])
    assert bool(metrics(timestep)["was_truncated"])


def test_breaking_the_answer_before_submitting_is_not_solved(game):
    env, state, _, answer, answer_mask = game
    state, _ = paint_answer(env, state, answer, answer_mask)

    row, col = (int(i) for i in jnp.argwhere(answer_mask)[0])
    wrong_color = (int(answer[row, col]) + 1) % 10
    cell = jnp.zeros_like(answer_mask).at[row, col].set(True)
    state, _ = env.step(state, create_action(wrong_color, cell))
    _, timestep = submit(env, state, answer_mask)

    assert int(timestep.step_type) == int(StepType.TERMINATED)
    assert float(metrics(timestep)["best_similarity"]) == 1.0
    assert not bool(metrics(timestep)["solved"])
    assert not bool(metrics(timestep)["was_truncated"])


def test_submitting_on_the_last_step_is_not_truncated(game):
    env, state, _, answer, answer_mask = game
    state, timestep = paint_answer(env, state, answer, answer_mask)
    state, _ = idle(env, state, timestep, answer_mask, STEP_LIMIT - 1)
    _, timestep = submit(env, state, answer_mask)

    assert int(timestep.step_type) == int(StepType.TERMINATED)
    assert bool(metrics(timestep)["solved"])
    assert not bool(metrics(timestep)["was_truncated"])


def test_metrics_match_when_compiled(game):
    env, state, _, answer, answer_mask = game
    state, _ = paint_answer(env, state, answer, answer_mask)
    with jax.disable_jit(False):
        step = jax.jit(env.step)
        nothing = jnp.zeros_like(answer_mask)

        _, submitted = step(state, create_action(SUBMIT, nothing))
        assert bool(metrics(submitted)["solved"])
        assert not bool(metrics(submitted)["was_truncated"])

        current, timestep = step(state, create_action(0, nothing))
        while not bool(timestep.done()):
            current, timestep = step(current, create_action(0, nothing))
        assert not bool(metrics(timestep)["solved"])
        assert bool(metrics(timestep)["was_truncated"])
