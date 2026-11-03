"""Tests for the train and eval environments built by the Stoix adapter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from jaxarc import JaxArcConfig
from jaxarc.registration import available_task_ids
from jaxarc.stoix_adapter import make_jaxarc_env

TRAIN, TEST = 0, 1


def make_envs(scenario="Mini-all", wrappers=None, **env_options):
    dataset = scenario.split("-")[0]
    if not available_task_ids(dataset, config=JaxArcConfig()):
        pytest.skip(f"No {dataset} tasks available")
    return make_jaxarc_env(
        OmegaConf.create(
            {
                "env": {
                    "scenario": {"name": scenario},
                    "action": {"mode": "point"},
                    "observation_wrappers": wrappers or {},
                    **env_options,
                }
            }
        )
    )


def reset_batch(env):
    keys = jax.random.split(jax.random.PRNGKey(0), 32)
    # conftest disables jit, which makes full-size grids very slow here.
    with jax.disable_jit(False):
        return jax.jit(jax.vmap(env.reset))(keys)


def test_eval_env_uses_test_pairs_by_default():
    _, eval_env = make_envs()
    params = eval_env.unwrapped.params

    state, _ = reset_batch(eval_env)

    assert params.episode_mode == TEST
    expected = params.buffer.test_input_grids[state.task_idx, state.pair_idx]
    assert jnp.array_equal(state.working_grid, expected)


def test_eval_episode_mode_can_be_set():
    _, eval_env = make_envs(eval_episode_mode=TRAIN)

    assert eval_env.unwrapped.params.episode_mode == TRAIN


@pytest.mark.parametrize("scenario", ["Mini-all", "AGI1-eval"])
def test_train_env_always_uses_demo_pairs(scenario):
    env, _ = make_envs(scenario)
    params = env.unwrapped.params

    state, _ = reset_batch(env)

    assert params.episode_mode == TRAIN
    expected = params.buffer.output_grids_examples[state.task_idx, state.pair_idx]
    assert jnp.array_equal(state.target_grid, expected)
    assert jnp.all(jnp.any(state.target_grid_mask, axis=(1, 2)))


def test_agi1_eval_scenario_still_loads_the_evaluation_tasks():
    env, eval_env = make_envs("AGI1-eval")
    train_env_only, _ = make_envs("AGI1-train")

    tasks = env.unwrapped.params.buffer.input_grids_examples
    other_tasks = train_env_only.unwrapped.params.buffer.input_grids_examples
    assert jnp.array_equal(tasks, eval_env.unwrapped.params.buffer.input_grids_examples)
    assert tasks.shape != other_tasks.shape or not jnp.array_equal(tasks, other_tasks)


def test_eval_scenario_holds_out_tasks():
    env, eval_env = make_envs("Concept-Count", eval_scenario="Concept-Center")

    train_tasks = env.unwrapped.params.buffer.task_index
    eval_tasks = eval_env.unwrapped.params.buffer.task_index
    assert not set(train_tasks.tolist()) & set(eval_tasks.tolist())
    assert eval_env.unwrapped.params.episode_mode == TEST


def test_eval_scenario_defaults_to_the_train_scenario():
    env, eval_env = make_envs("Concept-Count")

    assert jnp.array_equal(
        env.unwrapped.params.buffer.task_index,
        eval_env.unwrapped.params.buffer.task_index,
    )


def test_answer_channel_is_off_by_default():
    env, eval_env = make_envs(wrappers={"input_grid": True, "contextual": False})

    _, timestep = reset_batch(env)
    _, eval_timestep = reset_batch(eval_env)

    # Working grid and input grid only
    assert timestep.observation.shape[-1] == 2
    assert eval_timestep.observation.shape[-1] == 2


def test_answer_channel_can_be_turned_on_for_both_envs():
    env, eval_env = make_envs(
        wrappers={"answer_grid": True, "input_grid": False, "contextual": False}
    )
    params = eval_env.unwrapped.params

    state, timestep = reset_batch(env)
    eval_state, eval_timestep = reset_batch(eval_env)

    assert jnp.array_equal(timestep.observation[..., -1], state.target_grid)
    expected = params.buffer.true_test_output_grids[
        eval_state.task_idx, eval_state.pair_idx
    ]
    assert jnp.array_equal(eval_timestep.observation[..., -1], expected)


@pytest.mark.parametrize("scenario", ["Mini-all", "Concept-all"])
def test_train_and_eval_observations_have_the_same_shape(scenario):
    wrappers = {"answer_grid": True, "input_grid": True, "contextual": True}
    env, eval_env = make_envs(scenario, wrappers=wrappers)

    _, timestep = reset_batch(env)
    _, eval_timestep = reset_batch(eval_env)

    assert timestep.observation.shape == eval_timestep.observation.shape
