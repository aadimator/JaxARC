"""Extended metrics wrapper for collecting additional episode statistics."""

from __future__ import annotations

import jax.numpy as jnp
from chex import Numeric, PRNGKey
from stoa.core_wrappers.wrapper import Wrapper, WrapperState, wrapper_state_replace
from stoa.env_types import Action, EnvParams, TimeStep
from stoa.stoa_struct import dataclass


@dataclass(custom_replace_fn=wrapper_state_replace)
class ExtendedMetricsState(WrapperState):
    best_similarity: Numeric
    solved: Numeric
    steps_to_solve: Numeric
    final_similarity: Numeric
    was_truncated: Numeric


class ExtendedMetrics(Wrapper[ExtendedMetricsState]):
    def reset(self, rng_key, env_params=None):
        base_env_state, timestep = self._env.reset(rng_key, env_params)
        state = ExtendedMetricsState(
            base_env_state=base_env_state,
            best_similarity=jnp.array(0.0, dtype=jnp.float32),
            solved=jnp.array(False, dtype=bool),
            steps_to_solve=jnp.array(0, dtype=jnp.int32),
            final_similarity=jnp.array(0.0, dtype=jnp.float32),
            was_truncated=jnp.array(False, dtype=bool),
        )
        episode_metrics = timestep.extras.get("episode_metrics", {})
        episode_metrics = {
            **episode_metrics,
            "best_similarity": jnp.array(0.0, dtype=jnp.float32),
            "solved": jnp.array(False, dtype=bool),
            "steps_to_solve": jnp.array(0, dtype=jnp.int32),
            "final_similarity": jnp.array(0.0, dtype=jnp.float32),
            "was_truncated": jnp.array(False, dtype=bool),
        }
        new_extras = {**timestep.extras, "episode_metrics": episode_metrics}
        timestep = timestep.replace(extras=new_extras)
        return state, timestep

    def step(self, state, action, env_params=None):
        base_env_state, timestep = self._env.step(
            state.base_env_state, action, env_params
        )
        current_similarity = getattr(
            base_env_state, "similarity_score", jnp.array(0.0, dtype=jnp.float32)
        )
        done = timestep.done()
        was_truncated = done & (timestep.discount != 0.0)
        new_best_similarity = jnp.maximum(state.best_similarity, current_similarity)
        is_solved_now = current_similarity >= 1.0
        newly_solved = is_solved_now & ~state.solved
        current_step = getattr(
            base_env_state, "step_count", jnp.array(0, dtype=jnp.int32)
        )
        new_steps_to_solve = jnp.where(newly_solved, current_step, state.steps_to_solve)
        new_solved = state.solved | is_solved_now
        final_similarity = jnp.where(done, current_similarity, state.final_similarity)
        final_was_truncated = jnp.where(done, was_truncated, state.was_truncated)
        reset_best_similarity = jnp.where(
            done, jnp.array(0.0, dtype=jnp.float32), new_best_similarity
        )
        reset_solved = jnp.where(done, jnp.array(False, dtype=bool), new_solved)
        reset_steps_to_solve = jnp.where(
            done, jnp.array(0, dtype=jnp.int32), new_steps_to_solve
        )
        log_best_similarity = jnp.where(
            done, new_best_similarity, state.best_similarity
        )
        log_solved = jnp.where(done, new_solved, state.solved)
        log_steps_to_solve = jnp.where(done, new_steps_to_solve, state.steps_to_solve)
        episode_metrics = timestep.extras.get("episode_metrics", {})
        updated_episode_metrics = {
            **episode_metrics,
            "best_similarity": log_best_similarity,
            "solved": log_solved,
            "steps_to_solve": log_steps_to_solve,
            "final_similarity": final_similarity,
            "was_truncated": final_was_truncated,
        }
        new_extras = {**timestep.extras, "episode_metrics": updated_episode_metrics}
        timestep = timestep.replace(extras=new_extras)
        new_state = ExtendedMetricsState(
            base_env_state=base_env_state,
            best_similarity=reset_best_similarity,
            solved=reset_solved,
            steps_to_solve=reset_steps_to_solve,
            final_similarity=final_similarity,
            was_truncated=final_was_truncated,
        )
        return new_state, timestep
