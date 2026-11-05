"""Tests for the config make() uses."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import jaxarc
from jaxarc import JaxArcConfig
from jaxarc.configs.environment_config import EnvironmentConfig
from jaxarc.configs.reward_config import RewardConfig
from jaxarc.registration import available_task_ids, make
from jaxarc.registration.config_prep import prepare_config

REGISTERED_STEP_LIMIT = 100


@pytest.fixture(scope="module", autouse=True)
def mini_tasks():
    if not available_task_ids("Mini", config=JaxArcConfig()):
        pytest.skip("No Mini tasks available")


@pytest.fixture
def hydra_not_running():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def settings(params):
    return (
        params.action.allowed_operations,
        params.reward.step_penalty,
        params.reward.unsolved_submission_penalty,
        params.max_episode_steps,
    )


def test_no_config_is_the_same_with_or_without_hydra(hydra_not_running):
    _, without_hydra = make("Mini-all")

    conf_dir = Path(jaxarc.__file__).parent / "conf"
    with initialize_config_dir(str(conf_dir), version_base="1.2"):
        _, with_hydra = make("Mini-all")

    assert settings(without_hydra) == settings(with_hydra)


def test_no_config_uses_the_default_config(hydra_not_running):
    _, params = make("Mini-all")
    default = JaxArcConfig()

    assert params.action.allowed_operations == default.action.allowed_operations
    assert params.reward.step_penalty == default.reward.step_penalty
    assert (
        params.reward.unsolved_submission_penalty
        == default.reward.unsolved_submission_penalty
    )


def test_no_config_uses_the_registered_step_limit(hydra_not_running):
    _, params = make("Mini-all")

    assert params.max_episode_steps == REGISTERED_STEP_LIMIT
    assert prepare_config(None, 7, "Mini").environment.max_episode_steps == 7


def test_a_config_passed_in_is_used_as_given():
    config = JaxArcConfig(
        environment=EnvironmentConfig(max_episode_steps=12),
        reward=RewardConfig(step_penalty=-0.5),
    )

    _, params = make("Mini-all", config=config)

    assert params.max_episode_steps == 12
    assert params.reward.step_penalty == -0.5
    assert prepare_config(config, 7, "Mini") is config
