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
from typing import Callable

import jax
import jax.numpy as jnp

from jaxarc.configs import JaxArcConfig
from jaxarc.envs.actions import (
    StructuredAction,
    create_bbox_action,
    create_mask_action,
    create_point_action,
)
from jaxarc.envs.functional import (
    StepInfo,  # re-exported structure
    reset as functional_reset,
    step as functional_step,
)
from jaxarc.types import EnvParams, TimeStep
from jaxarc.envs.observation import create_observation
from jaxarc.state import State
from jaxarc.types import JaxArcTask
from jaxarc.utils.jax_types import (
    EPISODE_MODE_TRAIN,
    EpisodeDone,
    ObservationArray,
    PRNGKey,
    RewardValue,
)


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
        # Track current episode parameters in the wrapper
        self._episode_mode = EPISODE_MODE_TRAIN
        self._pair_idx = 0

        # Select step implementation
        self._use_unsafe = use_unsafe_step
        self._set_step_impl(use_unsafe_step)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _set_step_impl(self, unsafe: bool) -> None:
        # New API uses functional_step; no internal switching needed.
        self._use_unsafe = False
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
    ) -> tuple[State, ObservationArray]:
        """Reset the environment(s)."""
        td = task_data if task_data is not None else self._task_data
        if td is None:
            msg = "task_data must be provided at least once (constructor or reset)."
            raise ValueError(msg)

        if self.num_envs == 1:
            use_key = (
                key
                if key is not None
                else (self._next_key() if self._manage_keys else None)
            )
            if use_key is None:
                msg = "Key required when key management disabled."
                raise ValueError(msg)
            self._episode_mode = episode_mode
            self._pair_idx = (initial_pair_idx or 0)
            params = EnvParams.from_config(
                self.config,
                td,
                episode_mode=self._episode_mode,
                pair_idx=self._pair_idx,
            )
            ts = functional_reset(params, use_key)
            return ts.state, ts.observation
        # Batch mode
        if key is not None:
            keys = jax.random.split(key, self.num_envs)
        else:
            keys = self._next_batch_keys() if self._manage_keys else None
        if keys is None:
            msg = "Key required for batch reset when key management disabled."
            raise ValueError(msg)
        self._episode_mode = episode_mode
        self._pair_idx = (initial_pair_idx or 0)
        params = EnvParams.from_config(
            self.config,
            td,
            episode_mode=self._episode_mode,
            pair_idx=self._pair_idx,
        )
        def _do_reset(k):
            ts = functional_reset(params, k)
            return ts.state, ts.observation
        states, obs = jax.vmap(_do_reset)(keys)
        return states, obs

    def step(
        self,
        state: State,
        action: StructuredAction | dict,
    ) -> tuple[State, ObservationArray, RewardValue, EpisodeDone, StepInfo]:
        """Step environment(s) with action.

        Auto-reset behavior (if enabled) returns post-reset state while keeping
        `done=True` for the terminal transition (Gymnax style).
        """
        if self.num_envs == 1:
            # Build params and a minimal TimeStep from current state to use new API
            if self._task_data is None:
                raise ValueError("task_data must be provided to use step()")
            params = EnvParams.from_config(
                self.config,
                self._task_data,
                episode_mode=self._episode_mode,
                pair_idx=self._pair_idx,
            )
            obs0 = create_observation(state, self.config)
            timestep = TimeStep(
                step_type=jnp.asarray(1, dtype=jnp.int32),
                reward=jnp.asarray(0.0, dtype=jnp.float32),
                discount=jnp.asarray(1.0, dtype=jnp.float32),
                observation=obs0,
                state=state,
            )
            ts2 = functional_step(params, timestep, action)
            done = ts2.step_type == jnp.asarray(2, dtype=jnp.int32)
            info = StepInfo(
                similarity=ts2.state.similarity_score,
                similarity_improvement=ts2.state.similarity_score - state.similarity_score,
                operation_type=getattr(action, "operation", jnp.asarray(-1, dtype=jnp.int32)),
                step_count=ts2.state.step_count,
                success=ts2.state.similarity_score >= 1.0,
            )
            new_state, obs, reward = ts2.state, ts2.observation, ts2.reward
            if self._auto_reset and bool(done):
                # Perform reset with fresh key (internal or error if not managed)
                reset_key = self._next_key() if self._manage_keys else None
                if reset_key is None:
                    # If user disabled management they must handle reset externally
                    return new_state, obs, reward, done, info
                params_reset = EnvParams.from_config(
                    self.config,
                    self._task_data,
                    episode_mode=self._episode_mode,
                    pair_idx=0,
                )
                ts_reset = functional_reset(params_reset, reset_key)
                reset_state, reset_obs = ts_reset.state, ts_reset.observation
                # Replace state & obs but preserve done for this step's signal
                return reset_state, reset_obs, reward, done, info
            return new_state, obs, reward, done, info
        # Batch mode
        def _do_step(st, act):
            if self._task_data is None:
                raise ValueError("task_data must be provided to use step()")
            params = EnvParams.from_config(
                self.config,
                self._task_data,
                episode_mode=self._episode_mode,
                pair_idx=self._pair_idx,
            )
            obs0 = create_observation(st, self.config)
            ts = TimeStep(
                step_type=jnp.asarray(1, dtype=jnp.int32),
                reward=jnp.asarray(0.0, dtype=jnp.float32),
                discount=jnp.asarray(1.0, dtype=jnp.float32),
                observation=obs0,
                state=st,
            )
            ts2 = functional_step(params, ts, act)
            done = ts2.step_type == jnp.asarray(2, dtype=jnp.int32)
            info = StepInfo(
                similarity=ts2.state.similarity_score,
                similarity_improvement=ts2.state.similarity_score - st.similarity_score,
                operation_type=getattr(act, "operation", jnp.asarray(-1, dtype=jnp.int32)),
                step_count=ts2.state.step_count,
                success=ts2.state.similarity_score >= 1.0,
            )
            return ts2.state, ts2.observation, ts2.reward, done, info

        new_states, obs, rewards, dones, infos = jax.vmap(_do_step)(state, action)
        if self._auto_reset and self._manage_keys:

            def do_resets(carry):
                ns, ob = carry
                keys = self._next_batch_keys()

                def full_reset(_st, k):  # _st unused; kept for vmap in_axes signature
                    # Use training mode constant to avoid traced Python branching inside arc_reset
                    params_rs = EnvParams.from_config(
                        self.config, self._task_data, episode_mode=EPISODE_MODE_TRAIN, pair_idx=0
                    )
                    ts_rs = functional_reset(params_rs, k)
                    return ts_rs.state, ts_rs.observation

                rs_states, rs_obs = jax.vmap(full_reset, in_axes=(0, 0))(ns, keys)

                # Blend using mask broadcasting; handle arbitrary leaf shapes.
                def blend(a, b):
                    # If leaf has batch dimension first, broadcast dones.
                    if a.shape[0] == dones.shape[0]:
                        # Determine trailing broadcast shape
                        expand_shape = (dones.shape[0],) + (1,) * (a.ndim - 1)
                        mask = dones.reshape(expand_shape)
                        return jnp.where(mask, b, a)
                    return a  # Non-batched leaf

                blended_states = jax.tree_util.tree_map(blend, ns, rs_states)
                # Observations assumed batched in first dim
                expand_shape_obs = (dones.shape[0],) + (1,) * (obs.ndim - 1)
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

    def get_rollout_fn(
        self,
        policy_fn: Callable[[State, PRNGKey, JaxArcConfig], StructuredAction],
        num_steps: int,
    ) -> Callable[[State, PRNGKey], State]:
        """
        Returns a pure, JIT-able function for running a complete episode rollout.

        This function is designed for high-performance benchmarking and batch
        processing using the efficient `vmap(scan(...))` pattern. It closes over
        the environment's static configuration.

        NOTE: This rollout function does NOT perform auto-reset for performance
        reasons inside a scan.

        Args:
            policy_fn: A pure function with signature `(state, key, config) -> action`
                       that determines the action to take at each step.
            num_steps: The number of steps to scan over (length of the episode).

        Returns:
            A JIT-able function with signature `(initial_state, key) -> final_state`.
            If `self.num_envs > 1`, this function will be vmapped.
        """
        config = self.config
        step_fn = self._single_step_fn

        def _rollout_body_fn(carry, _):
            state, key = carry
            key, policy_key = jax.random.split(key)
            action = policy_fn(state, policy_key, config)
            next_state, _, _, _, _ = step_fn(state, action, config)
            return (next_state, key), None

        def _single_env_rollout(
            initial_state: State, key: PRNGKey
        ) -> State:
            (final_state, _), _ = jax.lax.scan(
                _rollout_body_fn, (initial_state, key), None, length=num_steps
            )
            return final_state

        if self.num_envs > 1:
            return jax.vmap(_single_env_rollout, in_axes=(0, 0))
        return _single_env_rollout

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    def warmup(self, state: State) -> State:
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
        if self._task_data is None:
            raise ValueError("task_data must be provided to warmup()")
        params = EnvParams.from_config(
            self.config,
            self._task_data,
            episode_mode=EPISODE_MODE_TRAIN,
            pair_idx=0,
        )
        for act in (point, bbox, mask_action):
            obs0 = create_observation(state, self.config)
            timestep = TimeStep(
                step_type=jnp.asarray(1, dtype=jnp.int32),
                reward=jnp.asarray(0.0, dtype=jnp.float32),
                discount=jnp.asarray(1.0, dtype=jnp.float32),
                observation=obs0,
                state=state,
            )
            ts2 = functional_step(params, timestep, act)
            state = ts2.state
        return state

    def get_num_envs(self) -> int:
        return self.num_envs


__all__ = ["ArcEnv"]
