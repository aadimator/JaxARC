"""
Observation wrappers for JaxARC environments.

This module provides a set of composable wrappers to augment the base
environment's observation, allowing for flexible and extensible state
representation. Each wrapper adds one or more channels to the observation
array, such as clipboard contents, input grids, or contextual information
from other demonstration pairs.

- ClipboardObservationWrapper: Adds the agent's clipboard as a channel.
- InputGridObservationWrapper: Adds the original input grid of the current task pair.
- ContextualObservationWrapper: Adds channels for N other demonstration pairs
  from the same task, providing contextual examples.

Usage:
    ```python
    from jaxarc.registration import make
    from jaxarc.wrappers import (
        ClipboardObservationWrapper,
        InputGridObservationWrapper,
        ContextualObservationWrapper,
    )

    # Create base environment
    env, env_params = make("Mini")

    # Stack observation wrappers
    env = ClipboardObservationWrapper(env)
    env = InputGridObservationWrapper(env)
    env = ContextualObservationWrapper(env, num_context_pairs=3)

    # The observation will now have channels for:
    # 1. Working grid (base)
    # 2. Clipboard
    # 3. Input grid
    # 4. Context pair 1 (input)
    # 5. Context pair 1 (output)
    # ... and so on.
    state, timestep = env.reset(key, env_params=env_params)
    print(timestep.observation.shape)
    ```
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from stoa.core_wrappers.wrapper import Wrapper

from jaxarc.envs.spaces import BoundedArraySpace
from jaxarc.state import State
from jaxarc.types import EnvParams, TimeStep


class BaseObservationWrapper(Wrapper):
    """A base class for observation wrappers to handle boilerplate."""

    def _augment_observation(
        self, observation: jax.Array, state: State, env_params: EnvParams
    ) -> jax.Array:
        """Function to be implemented by subclasses to add channels."""
        raise NotImplementedError

    def _get_num_new_channels(self) -> int:
        """Function to be implemented by subclasses to declare new channels."""
        raise NotImplementedError

    def observation_space(
        self, env_params: EnvParams | None = None
    ) -> BoundedArraySpace:
        """Augment the observation space with the new channels."""
        p = self._env.params if env_params is None else env_params
        base_obs_space = self._env.observation_space(p)

        new_channels = self._get_num_new_channels()
        base_shape = base_obs_space.shape

        return BoundedArraySpace(
            minimum=base_obs_space.minimum,
            maximum=base_obs_space.maximum,
            shape=(base_shape[0], base_shape[1], base_shape[2] + new_channels),
            dtype=base_obs_space.dtype,
        )

    def _process_timestep(
        self, timestep: TimeStep, state: State, env_params: EnvParams
    ) -> TimeStep:
        """Generic function to process a timestep and augment its observation."""
        augmented_obs = self._augment_observation(
            timestep.observation, state, env_params
        )

        extras = timestep.extras
        if isinstance(extras, dict) and "next_obs" in extras:
            extras = dict(extras)  # Avoid mutation under JAX transforms
            extras["next_obs"] = self._augment_observation(
                extras["next_obs"], state, env_params
            )

        return TimeStep(
            step_type=timestep.step_type,
            reward=timestep.reward,
            discount=timestep.discount,
            observation=augmented_obs,
            extras=extras,
        )

    def step(
        self, state: State, action: Any, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        """Step the environment and augment the resulting observation."""
        p = self._env.params if env_params is None else env_params
        next_state, timestep = self._env.step(state, action, env_params=p)
        return next_state, self._process_timestep(timestep, next_state, p)

    def reset(
        self, rng_key: jax.Array, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        """Reset the environment and augment the resulting observation."""
        p = self._env.params if env_params is None else env_params
        state, timestep = self._env.reset(rng_key, env_params=p)
        return state, self._process_timestep(timestep, state, p)


class ClipboardObservationWrapper(BaseObservationWrapper):
    """Adds the agent's clipboard as a new observation channel."""

    def _get_num_new_channels(self) -> int:
        return 1

    def _augment_observation(
        self, observation: jax.Array, state: State, env_params: EnvParams
    ) -> jax.Array:
        clipboard_channel = jnp.expand_dims(state.clipboard, axis=-1)
        return jnp.concatenate([observation, clipboard_channel], axis=-1)


class InputGridObservationWrapper(BaseObservationWrapper):
    """Adds the original input grid of the current pair as a new observation channel."""

    def _get_num_new_channels(self) -> int:
        return 1

    def _augment_observation(
        self, observation: jax.Array, state: State, env_params: EnvParams
    ) -> jax.Array:
        input_grid_channel = jnp.expand_dims(state.input_grid, axis=-1)
        return jnp.concatenate([observation, input_grid_channel], axis=-1)


class AnswerObservationWrapper(BaseObservationWrapper):
    """Adds the task's target grid (solution) as a new observation channel."""

    def _get_num_new_channels(self) -> int:
        return 1

    def _augment_observation(
        self, observation: jax.Array, state: State, env_params: EnvParams
    ) -> jax.Array:
        # Note: The environment's state management already ensures that `state.target_grid`
        # is correctly masked (e.g., zeroed out) during test episodes to prevent cheating.
        # This wrapper can therefore simply append it.
        answer_channel = jnp.expand_dims(state.target_grid, axis=-1)
        return jnp.concatenate([observation, answer_channel], axis=-1)


class ContextualObservationWrapper(BaseObservationWrapper):
    """
    Adds N context demonstration pairs to the observation.

    This wrapper adds `2 * num_context_pairs` channels to the observation,
    representing the input and output grids of other demonstration pairs
    from the same task. This provides the agent with contextual examples.
    """

    def __init__(self, env, num_context_pairs: int = 5):
        super().__init__(env)
        if num_context_pairs < 1:
            raise ValueError("num_context_pairs must be at least 1.")
        self.num_context_pairs = num_context_pairs

    def _get_num_new_channels(self) -> int:
        return 2 * self.num_context_pairs

    def _augment_observation(
        self, observation: jax.Array, state: State, env_params: EnvParams
    ) -> jax.Array:
        """
        Create observation with contextual demonstration pairs.
        This implementation is inspired by the user-provided example.
        """
        current_pair_idx = state.pair_idx
        task_idx = state.task_idx

        # This logic assumes env_params.buffer is available and contains task data
        task_data = jax.tree_util.tree_map(lambda x: x[task_idx], env_params.buffer)

        # All demonstration pairs for the current task
        all_inputs = task_data.input_grids_examples.astype(jnp.float32)
        all_outputs = task_data.output_grids_examples.astype(jnp.float32)

        num_actual_examples = task_data.num_train_pairs
        grid_shape = observation.shape[:2]
        zero_grid = jnp.zeros(grid_shape, dtype=jnp.float32)

        # Indices for the N context slots we need to fill
        context_slot_indices = jnp.arange(self.num_context_pairs)

        # In training mode, we exclude the current pair from the context.
        # In test mode, the agent is solving a test pair, so all demo pairs can be context.
        is_training = jnp.asarray(env_params.episode_mode == 0)

        # Shift indices to skip the current pair if in training mode
        demo_indices = jnp.where(
            is_training & (context_slot_indices >= current_pair_idx),
            context_slot_indices + 1,
            context_slot_indices,
        )

        # Create a mask for valid context pairs (within bounds of available demos)
        valid_mask = demo_indices < num_actual_examples

        # Ensure indices are safe for gathering
        safe_demo_indices = jnp.clip(demo_indices, 0, all_inputs.shape[0] - 1)

        # Gather the context grids
        context_inputs = all_inputs[safe_demo_indices]
        context_outputs = all_outputs[safe_demo_indices]

        # Apply the validity mask to zero-out invalid pairs
        valid_mask_expanded = valid_mask[:, None, None]
        context_inputs = jnp.where(valid_mask_expanded, context_inputs, zero_grid)
        context_outputs = jnp.where(valid_mask_expanded, context_outputs, zero_grid)

        # Interleave inputs and outputs: [input0, output0, input1, output1, ...]
        context_channels_stacked = jnp.stack([context_inputs, context_outputs], axis=1)
        context_channels_flat = context_channels_stacked.reshape(
            2 * self.num_context_pairs, *grid_shape
        )

        # Add channel dimension to match observation
        context_channels = jnp.transpose(context_channels_flat, (1, 2, 0))

        return jnp.concatenate([observation, context_channels], axis=-1)


__all__ = [
    "AnswerObservationWrapper",
    "ClipboardObservationWrapper",
    "ContextualObservationWrapper",
    "InputGridObservationWrapper",
]
