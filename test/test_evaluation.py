"""Tests for the held-out evaluator: scoring, the outcome wrapper, pair coverage."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from jaxarc import JaxArcConfig
from jaxarc.configs.environment_config import EnvironmentConfig
from jaxarc.envs.actions import create_action
from jaxarc.evaluation import make_eval_reset_fn, pass_at_k, score_grid, summarize
from jaxarc.registration import available_task_ids, make
from jaxarc.stoix_adapter import make_jaxarc_env
from jaxarc.types import StepType
from jaxarc.wrappers import EvalMetricsWrapper

RESIZE = 33
SUBMIT = 34
STEP_LIMIT = 15


def grid(rows):
    data = jnp.asarray(rows, dtype=jnp.int32)
    return jnp.zeros((4, 4), jnp.int32).at[: data.shape[0], : data.shape[1]].set(data)


def mask(height, width):
    return jnp.zeros((4, 4), bool).at[:height, :width].set(True)


# --- score_grid -------------------------------------------------------------


def test_matching_grid_is_exact():
    exact, accuracy = score_grid(
        grid([[1, 2], [3, 4]]), mask(2, 2), grid([[1, 2], [3, 4]]), mask(2, 2)
    )
    assert bool(exact)
    assert float(accuracy) == 1.0


def test_right_shape_scores_the_fraction_of_matching_cells():
    exact, accuracy = score_grid(
        grid([[1, 2], [3, 0]]), mask(2, 2), grid([[1, 2], [3, 4]]), mask(2, 2)
    )
    assert not bool(exact)
    assert float(accuracy) == 0.75


def test_wrong_shape_scores_zero_even_when_the_overlap_matches():
    working = grid([[1, 2, 0], [3, 4, 0]])
    exact, accuracy = score_grid(
        working, mask(2, 3), grid([[1, 2], [3, 4]]), mask(2, 2)
    )
    assert not bool(exact)
    assert float(accuracy) == 0.0


def test_cells_outside_the_masks_are_ignored():
    working = grid([[1, 2], [3, 4]]).at[3, 3].set(7)
    exact, accuracy = score_grid(
        working, mask(2, 2), grid([[1, 2], [3, 4]]), mask(2, 2)
    )
    assert bool(exact)
    assert float(accuracy) == 1.0


# --- EvalMetricsWrapper -----------------------------------------------------


@pytest.fixture(scope="module", params=[0, 1], ids=["train", "test"])
def game(request):
    """A wrapped env in each mode, reset on its first pair, with that pair's answer."""
    config = JaxArcConfig(environment=EnvironmentConfig(max_episode_steps=STEP_LIMIT))
    if not available_task_ids("Concept", config=config):
        pytest.skip("No Concept tasks available")
    base, params = make("Concept-Count", config=config, episode_mode=request.param)
    env = EvalMetricsWrapper(base)

    tasks, pairs = params.all_pairs()
    task, pair = int(tasks[-1]), int(pairs[-1])
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
    return env, state, timestep, answer, answer_mask, task, pair


def paint_answer(env, state, answer, answer_mask):
    """Resize to the answer's shape and fill it in, using real actions."""
    height = int(answer_mask.any(axis=1).sum())
    width = int(answer_mask.any(axis=0).sum())
    corner = jnp.zeros_like(answer_mask).at[height - 1, width - 1].set(True)
    state, _ = env.step(state, create_action(RESIZE, corner))
    for color in range(10):
        selection = (answer == color) & answer_mask
        if selection.any():
            state, _ = env.step(state, create_action(color, selection))
    return state


def submit(env, state, answer_mask):
    return env.step(state, create_action(SUBMIT, jnp.zeros_like(answer_mask)))


def outcome(timestep):
    return timestep.extras["eval_metrics"]


def test_submitting_the_answer_is_exact(game):
    env, state, _, answer, answer_mask, task, pair = game
    state = paint_answer(env, state, answer, answer_mask)
    _, timestep = submit(env, state, answer_mask)

    assert int(timestep.step_type) == int(StepType.TERMINATED)
    assert bool(outcome(timestep)["submitted"])
    assert bool(outcome(timestep)["exact"])
    assert float(outcome(timestep)["cell_accuracy"]) == 1.0
    assert int(outcome(timestep)["task_idx"]) == task
    assert int(outcome(timestep)["pair_idx"]) == pair


def test_a_wrong_cell_is_not_exact_and_costs_one_cell(game):
    env, state, _, answer, answer_mask, _, _ = game
    state = paint_answer(env, state, answer, answer_mask)
    row, col = (int(i) for i in jnp.argwhere(answer_mask)[0])
    cell = jnp.zeros_like(answer_mask).at[row, col].set(True)
    state, _ = env.step(state, create_action((int(answer[row, col]) + 1) % 10, cell))
    _, timestep = submit(env, state, answer_mask)

    assert bool(outcome(timestep)["submitted"])
    assert not bool(outcome(timestep)["exact"])
    expected = 1.0 - 1.0 / float(answer_mask.sum())
    assert float(outcome(timestep)["cell_accuracy"]) == pytest.approx(expected)


def test_reaching_the_answer_without_submitting_scores_nothing(game):
    env, state, _, answer, answer_mask, _, _ = game
    state = paint_answer(env, state, answer, answer_mask)
    nothing = jnp.zeros_like(answer_mask)
    timestep = None
    while timestep is None or not bool(timestep.done()):
        state, timestep = env.step(state, create_action(0, nothing))

    assert int(timestep.step_type) == int(StepType.TRUNCATED)
    assert not bool(outcome(timestep)["submitted"])
    assert not bool(outcome(timestep)["exact"])
    assert float(outcome(timestep)["cell_accuracy"]) == 0.0


def test_reset_and_step_extras_have_the_same_structure(game):
    # Stoix's evaluator carries the timestep through a while_loop.
    env, state, reset_timestep, _, answer_mask, _, _ = game
    _, step_timestep = env.step(state, create_action(0, jnp.zeros_like(answer_mask)))

    reset_leaves = jax.tree.leaves(reset_timestep.extras)
    step_leaves = jax.tree.leaves(step_timestep.extras)
    assert jax.tree.structure(reset_timestep.extras) == jax.tree.structure(
        step_timestep.extras
    )
    assert [x.dtype for x in reset_leaves] == [x.dtype for x in step_leaves]


def test_outcome_matches_when_compiled(game):
    env, state, _, answer, answer_mask, _, _ = game
    state = paint_answer(env, state, answer, answer_mask)
    with jax.disable_jit(False):
        _, timestep = jax.jit(env.step)(
            state, create_action(SUBMIT, jnp.zeros_like(answer_mask))
        )
    assert bool(outcome(timestep)["exact"])
    assert float(outcome(timestep)["cell_accuracy"]) == 1.0


# --- make_eval_reset_fn -----------------------------------------------------


def stoix_envs(scenario="Concept-Count", **env_options):
    if not available_task_ids(scenario.split("-")[0], config=JaxArcConfig()):
        pytest.skip("No tasks available")
    config = OmegaConf.create(
        {
            "env": {
                "scenario": {"name": scenario},
                "action": {"mode": "point"},
                **env_options,
            }
        }
    )
    return config, *make_jaxarc_env(config)


def test_eval_reset_covers_every_pair_once_per_attempt():
    config, _, eval_env = stoix_envs()
    tasks, pairs = eval_env.unwrapped.params.all_pairs()
    n = tasks.shape[0]

    reset_fn = make_eval_reset_fn(config, eval_env)
    with jax.disable_jit(False):
        state, timestep = jax.jit(reset_fn, static_argnums=1)(
            jax.random.PRNGKey(0), 2 * n
        )

    assert jnp.array_equal(state.task_idx, jnp.tile(tasks, 2))
    assert jnp.array_equal(state.pair_idx, jnp.tile(pairs, 2))
    assert jnp.array_equal(outcome(timestep)["task_idx"], jnp.tile(tasks, 2))


def test_eval_reset_follows_a_held_out_scenario():
    config, env, eval_env = stoix_envs(eval_scenario="Concept-Center")
    tasks, _ = eval_env.unwrapped.params.all_pairs()

    reset_fn = make_eval_reset_fn(config, eval_env)
    with jax.disable_jit(False):
        state, _ = jax.jit(reset_fn, static_argnums=1)(
            jax.random.PRNGKey(0), tasks.shape[0]
        )

    # task_idx indexes each env's own buffer; task_index names the task.
    eval_ids = eval_env.unwrapped.params.buffer.task_index[state.task_idx]
    train_ids = env.unwrapped.params.buffer.task_index
    assert set(eval_ids.tolist()) == set(
        eval_env.unwrapped.params.buffer.task_index.tolist()
    )
    assert not set(eval_ids.tolist()) & set(train_ids.tolist())


def test_eval_reset_rejects_a_batch_that_skips_pairs():
    config, _, eval_env = stoix_envs()
    n = eval_env.unwrapped.params.all_pairs()[0].shape[0]

    reset_fn = make_eval_reset_fn(config, eval_env)
    with pytest.raises(ValueError, match=str(n)):
        reset_fn(jax.random.PRNGKey(0), n + 1)


def test_stoix_eval_env_reports_outcomes_and_train_env_does_not():
    _, env, eval_env = stoix_envs()
    _, eval_timestep = eval_env.reset(jax.random.PRNGKey(0))
    _, train_timestep = env.reset(jax.random.PRNGKey(0))

    assert "eval_metrics" in eval_timestep.extras
    assert "eval_metrics" not in train_timestep.extras


# --- summarize --------------------------------------------------------------


def episodes():
    """Task 7 has two test pairs, task 9 has one, each run twice."""
    return {
        "task_idx": np.array([7, 7, 9, 7, 7, 9]),
        "pair_idx": np.array([0, 1, 0, 0, 1, 0]),
        "exact": np.array([1, 0, 0, 0, 0, 1], dtype=bool),
        "cell_accuracy": np.array([1.0, 0.5, 0.2, 0.6, 0.5, 1.0]),
        "submitted": np.ones(6, dtype=bool),
        "episode_return": np.zeros(6),
    }


def test_summary_averages_pairs_within_a_task_then_tasks():
    summary = summarize(episodes(), k=2)

    # Task 7: pair 0 exact once in two, pair 1 never -> 0.25. Task 9: 0.5.
    assert summary["exact"] == pytest.approx((0.25 + 0.5) / 2)
    # Task 7: pair 0 solved in one of two attempts, pair 1 not -> 0.5. Task 9: 1.
    assert summary["pass@2"] == pytest.approx((0.5 + 1.0) / 2)
    # Task 7: (0.8 + 0.5) / 2 = 0.65. Task 9: 0.6.
    assert summary["cell_accuracy"] == pytest.approx((0.65 + 0.6) / 2)
    assert summary["submit_rate"] == 1.0


def test_summary_keeps_per_task_scores_for_the_bootstrap():
    summary = summarize(episodes(), k=2)

    assert summary["per_task"]["task_idx"].tolist() == [7, 9]
    assert summary["per_task"]["exact"].tolist() == pytest.approx([0.25, 0.5])
    assert summary["per_task"]["pass@2"].tolist() == pytest.approx([0.5, 1.0])


def test_summary_accepts_stoix_device_and_batch_axes():
    batched = {k: v.reshape(1, -1) for k, v in episodes().items()}
    assert summarize(batched, k=2)["exact"] == summarize(episodes(), k=2)["exact"]


def test_summary_rejects_more_attempts_than_were_run():
    with pytest.raises(ValueError, match="attempts"):
        summarize(episodes(), k=3)


def test_pass_at_k_is_the_unbiased_estimator():
    assert pass_at_k(n=2, c=0, k=2) == 0.0
    assert pass_at_k(n=2, c=1, k=2) == 1.0
    assert pass_at_k(n=4, c=1, k=2) == pytest.approx(0.5)
    assert pass_at_k(n=4, c=1, k=1) == pytest.approx(0.25)
