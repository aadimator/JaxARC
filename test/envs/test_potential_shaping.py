"""Tests for potential-based similarity shaping (Ng et al. 1999).

With shaping="potential" the shaping term is gamma * phi(s') - phi(s), with
phi = similarity, and phi(s') = 0 when the step ends the episode (a submit).
Truncation keeps phi(s'), since the learner bootstraps past it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxarc import JaxArcConfig
from jaxarc.configs import GridInitializationConfig
from jaxarc.configs.environment_config import EnvironmentConfig
from jaxarc.configs.reward_config import RewardConfig
from jaxarc.envs.actions import create_action
from jaxarc.registration import available_task_ids, make
from jaxarc.types import StepType

SUBMIT = 34
GAMMA = 0.9
WEIGHT = 2.0
STEP_LIMIT = 6

DEMO = GridInitializationConfig(
    demo_weight=1.0, permutation_weight=0.0, empty_weight=0.0, random_weight=0.0
)


def build(episode_mode=0, **reward):
    reward = {
        "shaping": "potential",
        "gamma": GAMMA,
        "similarity_weight": WEIGHT,
        "step_penalty": -0.01,
        "success_bonus": 10.0,
        "unsolved_submission_penalty": -1.0,
        **reward,
    }
    config = JaxArcConfig(
        environment=EnvironmentConfig(max_episode_steps=STEP_LIMIT),
        grid_initialization=DEMO,
        reward=RewardConfig(**reward),
    )
    if not available_task_ids("Mini", config=config):
        pytest.skip("No MiniARC tasks available")
    return make("Mini-all", config=config, episode_mode=episode_mode)


def fill_wrong_cell(state):
    """A fill that changes cell (0, 0) to a colour that is not the target's."""
    target = int(state.target_grid[0, 0])
    current = int(state.working_grid[0, 0])
    colour = next(c for c in range(10) if c not in (target, current))
    selection = jnp.zeros_like(state.working_grid_mask).at[0, 0].set(True)
    return create_action(colour, selection)


def fill_right_cell(state):
    """A fill that sets one wrong cell to its target colour."""
    wrong = np.argwhere(
        np.asarray(state.working_grid != state.target_grid) & np.asarray(state.target_grid_mask)
    )
    r, c = map(int, wrong[0])
    selection = jnp.zeros_like(state.working_grid_mask).at[r, c].set(True)
    return create_action(int(state.target_grid[r, c]), selection)


def submit(state):
    return create_action(SUBMIT, jnp.zeros_like(state.working_grid_mask))


def first_unsolved(env, params):
    """Reset until the start grid is not already the answer."""
    for seed in range(50):
        state, _ = env.reset(jax.random.PRNGKey(seed))
        if float(state.similarity_score) < 1.0:
            return state
    pytest.fail("every reset started solved")


def test_default_shaping_is_delta():
    assert RewardConfig().shaping == "delta"


def test_edit_step_reward_is_discounted_potential_difference():
    env, params = build()
    state = first_unsolved(env, params)

    new_state, timestep = env.step(state, fill_right_cell(state))

    expected = WEIGHT * (GAMMA * new_state.similarity_score - state.similarity_score) - 0.01
    assert timestep.step_type == StepType.MID
    assert float(new_state.similarity_score) > float(state.similarity_score)
    assert jnp.allclose(timestep.reward, expected)


def test_wrong_submit_drops_the_potential_to_zero():
    env, params = build()
    state = first_unsolved(env, params)

    _, timestep = env.step(state, submit(state))

    expected = WEIGHT * (0.0 - state.similarity_score) - 0.01 - 1.0
    assert timestep.step_type == StepType.TERMINATED
    assert jnp.allclose(timestep.reward, expected)


def test_correct_submit_gets_bonus_minus_full_potential():
    env, params = build()
    state = first_unsolved(env, params)
    while float(state.similarity_score) < 1.0:
        state, _ = env.step(state, fill_right_cell(state))

    _, timestep = env.step(state, submit(state))

    expected = 10.0 - WEIGHT * 1.0 - 0.01
    assert timestep.step_type == StepType.TERMINATED
    assert jnp.allclose(timestep.reward, expected)


def test_truncation_keeps_the_potential():
    env, params = build()
    state = first_unsolved(env, params)
    for _ in range(STEP_LIMIT - 1):
        state, _ = env.step(state, fill_wrong_cell(state))

    new_state, timestep = env.step(state, fill_wrong_cell(state))

    expected = WEIGHT * (GAMMA * new_state.similarity_score - state.similarity_score) - 0.01
    assert timestep.step_type == StepType.TRUNCATED
    assert jnp.allclose(timestep.reward, expected)


def test_shaping_telescopes_to_minus_start_potential_under_jit():
    """Any episode that ends in a submit sums (discounted) to -w * phi(s0) + bonus."""
    env, params = build(step_penalty=0.0, unsolved_submission_penalty=0.0, success_bonus=0.0)
    keys = jax.random.split(jax.random.PRNGKey(1), 64)

    def episode(key):
        state, _ = env.reset(key)
        start = state.similarity_score
        H, W = state.working_grid.shape

        def body(carry, k):
            state, ret, disc, alive, key = carry
            key, a, b, c = jax.random.split(key, 4)
            last = k == STEP_LIMIT - 1
            op = jnp.where(last, SUBMIT, jax.random.randint(a, (), 0, 10))
            sel = jnp.zeros((H, W), bool).at[
                jax.random.randint(b, (), 0, H), jax.random.randint(c, (), 0, W)
            ].set(True)
            new, ts = env.step(state, create_action(op, sel))
            ret = ret + alive * disc * ts.reward
            alive = alive * (ts.step_type == StepType.MID)
            return (new, ret, disc * GAMMA, alive, key), None

        init = (state, 0.0, 1.0, 1.0, key)
        (_, ret, _, _, _), _ = jax.lax.scan(body, init, jnp.arange(STEP_LIMIT))
        return ret, start

    with jax.disable_jit(False):
        ret, start = jax.jit(jax.vmap(episode))(keys)

    assert jnp.allclose(ret, -WEIGHT * start, atol=1e-5)


def test_test_mode_has_no_shaping():
    env, params = build(episode_mode=1)
    state = first_unsolved(env, params)

    _, timestep = env.step(state, submit(state))

    assert jnp.allclose(timestep.reward, -0.01 - 1.0)


@pytest.mark.parametrize(
    ("field", "value"), [("shaping", "potentail"), ("gamma", 0.0), ("gamma", 1.5)]
)
def test_invalid_shaping_settings_are_reported(field, value):
    assert RewardConfig(**{field: value}).validate()


def test_from_hydra_reads_shaping_fields():
    from omegaconf import DictConfig

    config = RewardConfig.from_hydra(DictConfig({"shaping": "potential", "gamma": 0.95}))
    assert config.shaping == "potential"
    assert config.gamma == 0.95
