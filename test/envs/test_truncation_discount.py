"""Tests that the discount tells a real ending from hitting the step limit."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxarc import JaxArcConfig
from jaxarc.configs.environment_config import EnvironmentConfig
from jaxarc.envs.actions import create_action
from jaxarc.registration import available_task_ids, make
from jaxarc.types import StepType

SUBMIT = 34
STEP_LIMIT = 3


@pytest.fixture(scope="module")
def game():
    config = JaxArcConfig(environment=EnvironmentConfig(max_episode_steps=STEP_LIMIT))
    if not available_task_ids("Concept", config=config):
        pytest.skip("No Concept tasks available")
    env, _ = make("Concept-Count", config=config)
    state, _ = env.reset(jax.random.PRNGKey(0))
    nothing = jnp.zeros_like(state.working_grid_mask)
    return env, state, create_action(0, nothing), create_action(SUBMIT, nothing)


def run_to_step(env, state, idle, step_count):
    for _ in range(step_count):
        state, timestep = env.step(state, idle)
    return state, timestep


def stoix_flags(timestep):
    """How Stoix's PPO reads the end of an episode."""
    done = timestep.discount == 0.0
    truncated = timestep.last() & (timestep.discount != 0.0)
    return bool(done), bool(truncated)


def test_a_normal_step_keeps_the_discount(game):
    env, state, idle, _ = game
    _, timestep = run_to_step(env, state, idle, 1)

    assert int(timestep.step_type) == int(StepType.MID)
    assert float(timestep.discount) == 1.0
    assert stoix_flags(timestep) == (False, False)


def test_submitting_ends_with_a_zero_discount(game):
    env, state, _, submit = game
    _, timestep = env.step(state, submit)

    assert int(timestep.step_type) == int(StepType.TERMINATED)
    assert float(timestep.discount) == 0.0
    assert stoix_flags(timestep) == (True, False)


def test_the_step_limit_keeps_the_discount(game):
    env, state, idle, _ = game
    _, timestep = run_to_step(env, state, idle, STEP_LIMIT)

    assert int(timestep.step_type) == int(StepType.TRUNCATED)
    assert float(timestep.discount) == 1.0
    assert stoix_flags(timestep) == (False, True)


def test_submitting_on_the_last_step_is_a_real_ending(game):
    env, state, idle, submit = game
    state, _ = run_to_step(env, state, idle, STEP_LIMIT - 1)
    _, timestep = env.step(state, submit)

    assert int(timestep.step_type) == int(StepType.TERMINATED)
    assert float(timestep.discount) == 0.0
    assert stoix_flags(timestep) == (True, False)


def test_discount_matches_when_compiled(game):
    env, state, idle, submit = game
    with jax.disable_jit(False):
        step = jax.jit(env.step)
        _, submitted = step(state, submit)
        assert float(submitted.discount) == 0.0

        current = state
        for _ in range(STEP_LIMIT):
            current, timestep = step(current, idle)
        assert int(timestep.step_type) == int(StepType.TRUNCATED)
        assert float(timestep.discount) == 1.0
