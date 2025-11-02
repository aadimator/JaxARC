"""
Unit tests for Observation Wrappers in JaxARC.

This module tests the functionality of the observation wrappers:
- `AnswerObservationWrapper`
- `ClipboardObservationWrapper`
- `InputGridObservationWrapper`
- `ContextualObservationWrapper`

Tests verify:
1.  Correct augmentation of the observation space.
2.  Correct content and shape of the observation tensor after wrapping.
3.  Correct behavior in both training and testing modes.
4.  Correct functionality when wrappers are stacked.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import jaxarc
from jaxarc.envs import (
    AnswerObservationWrapper,
    ClipboardObservationWrapper,
    ContextualObservationWrapper,
    InputGridObservationWrapper,
)

# Use a fixed task for reproducibility
TASK_ID = "Mini-Most_Common_color_l6ab0lf3xztbyxsu3p"


@pytest.fixture
def base_env_and_params():
    """Fixture to provide a base environment and its parameters."""
    env, env_params = jaxarc.make(TASK_ID)
    return env, env_params


def test_clipboard_observation_wrapper(base_env_and_params):
    """Test that ClipboardObservationWrapper correctly adds the clipboard channel."""
    env, env_params = base_env_and_params
    env = ClipboardObservationWrapper(env)

    # Check observation space
    obs_space = env.observation_space(env_params)
    assert obs_space.shape[-1] == 2  # Base (1) + Clipboard (1)

    # Reset env and check observation content
    key = jax.random.PRNGKey(0)
    state, timestep = env.reset(key)

    # Manually create a state with a known clipboard pattern
    clipboard_pattern = jnp.ones_like(state.clipboard) * 5
    state = eqx.tree_at(lambda s: s.clipboard, state, clipboard_pattern)

    # The wrapper's step/reset methods are what augment the observation
    action = {
        "operation": 0,
        "selection": jnp.zeros_like(state.working_grid, dtype=bool),
    }
    _, new_timestep = env.step(state, action)

    # The last channel should be the clipboard
    assert new_timestep.observation.shape[-1] == 2
    assert jnp.array_equal(new_timestep.observation[..., -1], clipboard_pattern)


def test_input_grid_observation_wrapper(base_env_and_params):
    """Test that InputGridObservationWrapper correctly adds the input_grid channel."""
    env, env_params = base_env_and_params
    env = InputGridObservationWrapper(env)

    obs_space = env.observation_space(env_params)
    assert obs_space.shape[-1] == 2  # Base (1) + Input Grid (1)

    key = jax.random.PRNGKey(1)
    state, timestep = env.reset(key)

    # The second channel should match the state's input_grid
    assert timestep.observation.shape[-1] == 2
    assert jnp.array_equal(timestep.observation[..., -1], state.input_grid)


@pytest.mark.parametrize("episode_mode, should_be_masked", [(0, False), (1, True)])
def test_answer_observation_wrapper(
    base_env_and_params, episode_mode, should_be_masked
):
    """Test AnswerObservationWrapper in both training and testing modes."""
    env, env_params = base_env_and_params

    # Create a new env_params for the desired mode
    mode_params = eqx.tree_at(lambda p: p.episode_mode, env_params, episode_mode)

    env = AnswerObservationWrapper(env)

    obs_space = env.observation_space(mode_params)
    assert obs_space.shape[-1] == 2  # Base (1) + Answer (1)

    key = jax.random.PRNGKey(2)
    state, timestep = env.reset(key, env_params=mode_params)

    assert timestep.observation.shape[-1] == 2
    answer_channel = timestep.observation[..., -1]

    if should_be_masked:
        # In test mode, the target grid is masked (filled with background color)
        background_color = mode_params.dataset.background_color
        assert jnp.all(answer_channel == background_color)
    else:
        # In training mode, it should match the state's target_grid
        assert jnp.array_equal(answer_channel, state.target_grid)


def test_contextual_observation_wrapper(base_env_and_params):
    """Test ContextualObservationWrapper for correct context pair selection."""
    env, env_params = base_env_and_params
    num_ctx_pairs = 2
    env = ContextualObservationWrapper(env, num_context_pairs=num_ctx_pairs)

    obs_space = env.observation_space(env_params)
    assert obs_space.shape[-1] == 1 + 2 * num_ctx_pairs

    key = jax.random.PRNGKey(3)
    state, timestep = env.reset(key, env_params=env_params)

    assert timestep.observation.shape[-1] == 1 + 2 * num_ctx_pairs

    # Check that the context channels are not empty (this task has enough pairs)
    context_channels = timestep.observation[..., 1:]
    assert not jnp.all(context_channels == 0)

    # More detailed check: ensure current pair is excluded from context
    task_data = jax.tree_util.tree_map(lambda x: x[state.task_idx], env_params.buffer)
    current_pair_input = task_data.input_grids_examples[state.pair_idx]

    is_present = False
    for i in range(num_ctx_pairs):
        # Context channels are interleaved: input, output
        ctx_input_channel = context_channels[..., 2 * i]
        if jnp.array_equal(ctx_input_channel, current_pair_input):
            is_present = True
            break
    assert not is_present, (
        "Current pair should be excluded from context in training mode."
    )


def test_stacked_wrappers(base_env_and_params):
    """Test that stacking multiple wrappers works as expected."""
    env, env_params = base_env_and_params
    num_ctx_pairs = 3

    # Stack all wrappers
    env = InputGridObservationWrapper(env)
    env = AnswerObservationWrapper(env)
    env = ClipboardObservationWrapper(env)
    env = ContextualObservationWrapper(env, num_context_pairs=num_ctx_pairs)

    # Check final observation space shape
    obs_space = env.observation_space(env_params)
    expected_channels = 1 + 1 + 1 + 1 + (2 * num_ctx_pairs)
    assert obs_space.shape[-1] == expected_channels

    # Reset and check the final observation shape
    key = jax.random.PRNGKey(4)
    state, timestep = env.reset(key, env_params=env_params)

    assert timestep.observation.shape[-1] == expected_channels

    # Verify the order of the first few channels
    # Channel 0: Base (working grid)
    assert jnp.array_equal(timestep.observation[..., 0], state.working_grid)
    # Channel 1: InputGrid wrapper
    assert jnp.array_equal(timestep.observation[..., 1], state.input_grid)
    # Channel 2: Answer wrapper
    assert jnp.array_equal(timestep.observation[..., 2], state.target_grid)
    # Channel 3: Clipboard wrapper (requires a step to populate observation)
    state = eqx.tree_at(
        lambda s: s.clipboard, state, jnp.ones_like(state.clipboard) * 8
    )
    action = {
        "operation": 0,
        "selection": jnp.zeros_like(state.working_grid, dtype=bool),
    }
    _, new_timestep = env.step(state, action)
    assert jnp.array_equal(new_timestep.observation[..., 3], state.clipboard)
