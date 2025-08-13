"""ArcEnv class wrapper over functional JaxARC API.

Safe-by-default, thin convenience layer adding:
- Object-oriented interface (reset/step)
- Optional internal PRNG key management
- Batch mode handling via num_envs
- Auto-reset logic (optional)
- Warmup routine to trigger JIT compilation of major action paths

Functional primitives remain the single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxarc.configs import JaxArcConfig
from jaxarc.envs.functional import (
    _arc_step_unsafe,
    arc_reset,
    arc_step,
    batch_reset,
)
from jaxarc.types import JaxArcTask
from jaxarc.utils.jax_types import EPISODE_MODE_TRAIN, PRNGKey
from jaxarc.envs.actions import (
    create_bbox_action,
    create_mask_action,
    create_point_action,
    StructuredAction,
)
from jaxarc.state import ArcEnvState
from jaxarc.envs.functional import StepInfo  # re-exported structure
from jaxarc.utils.jax_types import ObservationArray, RewardValue, EpisodeDone


@dataclass
class _KeyManager:
    base_key: PRNGKey
    step_counter: int = 0

    def next_key(self) -> PRNGKey:
        k = jax.random.fold_in(self.base_key, self.step_counter)
        self.step_counter += 1
        return k

    def next_batch_keys(self, batch_size: int) -> jnp.ndarray:
        k = self.next_key()
        return jax.random.split(k, batch_size)


class ArcEnv:
    """Gym-like wrapper around functional JaxARC environment.

    Parameters
    ----------
    config : JaxArcConfig | DictConfig
        Environment configuration (typed preferred).
    num_envs : int, default 1
        >1 enables batched mode (vectorized reset/step semantics).
    task_data : JaxArcTask | None
        Optional task data (shared across envs in batch mode).
    auto_reset : bool | None
        Overrides config.environment.auto_reset if provided.
    use_unsafe_step : bool, default False
        If True, uses `_arc_step_unsafe` (donating, no validation fallback).
    manage_keys : bool, default True
        If True, internal key manager supplies keys when user passes None.
    seed : int | None
        Initial seed for internal key manager.
    """

    def __init__(
        self,
        config: JaxArcConfig,
        *,
        num_envs: int = 1,
        task_data: JaxArcTask | None = None,
        auto_reset: bool | None = None,
        use_unsafe_step: bool = False,
        manage_keys: bool = True,
        seed: int | None = None,
    ) -> None:
        if num_envs < 1:
            msg = "num_envs must be >= 1"
            raise ValueError(msg)
        self.config = config
        self.num_envs = num_envs
        self._task_data = task_data
        self._auto_reset = (
            auto_reset if auto_reset is not None else config.environment.auto_reset
        )
        self._manage_keys = manage_keys
        self._key_manager: _KeyManager | None = None
        if manage_keys:
            if seed is None:
                seed = 0
            self._key_manager = _KeyManager(jax.random.PRNGKey(seed))

        # Select step implementation
        self._use_unsafe = use_unsafe_step
        self._set_step_impl(use_unsafe_step)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _set_step_impl(self, unsafe: bool) -> None:
        self._use_unsafe = unsafe
        self._single_step_fn = _arc_step_unsafe if unsafe else arc_step
        # Vectorized version (outer vmap). Underlying step is already filter_jit'ed.
        if self.num_envs > 1:
            self._batched_step_fn = jax.vmap(
                self._single_step_fn, in_axes=(0, 0, None), out_axes=(0, 0, 0, 0, 0)
            )
        else:
            self._batched_step_fn = None

    def switch_step_impl(self, use_unsafe: bool) -> None:
        """Switch between safe and unsafe step implementation at runtime."""
        self._set_step_impl(use_unsafe)

    def set_seed(self, seed: int) -> None:
        if not self._manage_keys:
            msg = "Key management disabled; cannot set internal seed."
            raise ValueError(msg)
        self._key_manager = _KeyManager(jax.random.PRNGKey(seed))

    def _next_key(self) -> PRNGKey:
        if not self._manage_keys:
            msg = "Key management disabled; user must provide keys."
            raise ValueError(msg)
        assert self._key_manager is not None
        return self._key_manager.next_key()

    def _next_batch_keys(self) -> jnp.ndarray:
        if not self._manage_keys:
            msg = "Key management disabled; user must provide keys."
            raise ValueError(msg)
        assert self._key_manager is not None
        return self._key_manager.next_batch_keys(self.num_envs)

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------
    def reset(
        self,
        key: PRNGKey | None = None,
        *,
        task_data: JaxArcTask | None = None,
        episode_mode: int = EPISODE_MODE_TRAIN,
        initial_pair_idx: int | None = None,
    ) -> tuple[ArcEnvState, ObservationArray]:
        """Reset the environment(s)."""
        td = task_data if task_data is not None else self._task_data
        if td is None:
            msg = "task_data must be provided at least once (constructor or reset)."
            raise ValueError(msg)

        if self.num_envs == 1:
            use_key = key if key is not None else (self._next_key() if self._manage_keys else None)
            if use_key is None:
                msg = "Key required when key management disabled."
                raise ValueError(msg)
            state, obs = arc_reset(
                use_key, self.config, td, episode_mode=episode_mode, initial_pair_idx=initial_pair_idx
            )
            return state, obs
        # Batch mode
        if key is not None:
            keys = jax.random.split(key, self.num_envs)
        else:
            keys = self._next_batch_keys() if self._manage_keys else None
        if keys is None:
            msg = "Key required for batch reset when key management disabled."
            raise ValueError(msg)
        states, obs = batch_reset(keys, self.config, td)
        return states, obs

    def step(
        self,
        state: ArcEnvState,
        action: StructuredAction | dict,
    ) -> tuple[ArcEnvState, ObservationArray, RewardValue, EpisodeDone, StepInfo]:
        """Step environment(s) with action.

        Auto-reset behavior (if enabled) returns post-reset state while keeping
        `done=True` for the terminal transition (Gymnax style).
        """
        if self.num_envs == 1:
            new_state, obs, reward, done, info = self._single_step_fn(
                state, action, self.config
            )
            if self._auto_reset and bool(done):
                # Perform reset with fresh key (internal or error if not managed)
                reset_key = self._next_key() if self._manage_keys else None
                if reset_key is None:
                    # If user disabled management they must handle reset externally
                    return new_state, obs, reward, done, info
                reset_state, reset_obs = arc_reset(
                    reset_key, self.config, self._task_data, episode_mode=state.episode_mode
                )
                # Replace state & obs but preserve done for this step's signal
                return reset_state, reset_obs, reward, done, info
            return new_state, obs, reward, done, info
        # Batch mode
        if self._batched_step_fn is None:
            msg = "Batched step function not initialized."
            raise RuntimeError(msg)
        new_states, obs, rewards, dones, infos = self._batched_step_fn(state, action, self.config)
        if self._auto_reset and self._manage_keys:
            def do_resets(carry):
                ns, ob = carry
                keys = self._next_batch_keys()
                def full_reset(_st, k):  # _st unused; kept for vmap in_axes signature
                    # Use training mode constant to avoid traced Python branching inside arc_reset
                    rs, ro = arc_reset(k, self.config, self._task_data, episode_mode=EPISODE_MODE_TRAIN)
                    return rs, ro
                rs_states, rs_obs = jax.vmap(full_reset, in_axes=(0,0))(ns, keys)
                # Blend using mask broadcasting; handle arbitrary leaf shapes.
                def blend(a, b):
                    # If leaf has batch dimension first, broadcast dones.
                    if a.shape[0] == dones.shape[0]:
                        # Determine trailing broadcast shape
                        expand_shape = (dones.shape[0],) + (1,)* (a.ndim -1)
                        mask = dones.reshape(expand_shape)
                        return jnp.where(mask, b, a)
                    return a  # Non-batched leaf
                blended_states = jax.tree_util.tree_map(blend, ns, rs_states)
                # Observations assumed batched in first dim
                expand_shape_obs = (dones.shape[0],) + (1,)* (obs.ndim -1)
                obs_mask = dones.reshape(expand_shape_obs)
                blended_obs = jnp.where(obs_mask, rs_obs, ob)
                return blended_states, blended_obs

            new_states, obs = jax.lax.cond(
                jnp.any(dones),
                do_resets,
                lambda carry: carry,
                operand=(new_states, obs),
            )
        return new_states, obs, rewards, dones, infos

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    def warmup(self, state: ArcEnvState) -> ArcEnvState:
        """Run one step for each action archetype to trigger compilation.

        Returns final state after three no-op style actions.
        """
        h = state.working_grid.shape[-2]
        w = state.working_grid.shape[-1]
        point = create_point_action(0, 0, 0)
        bbox = create_bbox_action(0, 0, 0, min(1, h - 1), min(1, w - 1))
        mask = jnp.zeros((h, w), dtype=jnp.bool_)
        mask = mask.at[0, 0].set(True)
        mask_action = create_mask_action(0, mask)
        for act in (point, bbox, mask_action):
            state, *_ = self._single_step_fn(state, act, self.config)
        return state

    def get_num_envs(self) -> int:
        return self.num_envs

__all__ = ["ArcEnv"]
