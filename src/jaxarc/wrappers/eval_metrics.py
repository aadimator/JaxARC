"""Wrapper that reports how each episode scores against the true answer."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from stoa.core_wrappers.wrapper import Wrapper

from jaxarc.types import EnvParams, StepType, TimeStep


def score_grid(
    working_grid: jax.Array,
    working_mask: jax.Array,
    target_grid: jax.Array,
    target_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Score a grid against the answer: (exact match, cell accuracy).

    Cell accuracy is the fraction of the answer's cells that match, and 0 when
    the shape is wrong. Unlike the env's similarity score, a wrong shape gets no
    partial credit.
    """
    same_shape = jnp.array_equal(working_mask, target_mask)
    matches = jnp.sum((working_grid == target_grid) & target_mask)
    cells = jnp.maximum(jnp.sum(target_mask), 1)
    exact = same_shape & (matches == jnp.sum(target_mask))
    accuracy = jnp.where(same_shape, matches / cells, 0.0).astype(jnp.float32)
    return exact, accuracy


class EvalMetricsWrapper(Wrapper):
    """Adds ``extras["eval_metrics"]``: the episode's outcome against the true answer.

    The answer is the demo output in train mode and the hidden test output in
    test mode. ``exact`` and ``cell_accuracy`` count only a submitted grid, so
    an episode that runs out of steps scores 0. Put this on the eval env only.
    """

    def reset(
        self, rng_key: jax.Array, env_params: EnvParams | None = None
    ) -> tuple[Any, TimeStep]:
        p = self._env.params if env_params is None else env_params
        state, timestep = self._env.reset(rng_key, env_params=p)
        return state, self._add_outcome(timestep, state, p)

    def step(
        self, state: Any, action: Any, env_params: EnvParams | None = None
    ) -> tuple[Any, TimeStep]:
        p = self._env.params if env_params is None else env_params
        state, timestep = self._env.step(state, action, env_params=p)
        return state, self._add_outcome(timestep, state, p)

    def _add_outcome(self, timestep: TimeStep, state: Any, p: EnvParams) -> TimeStep:
        task, pair = state.task_idx, state.pair_idx
        test_mode = jnp.asarray(p.episode_mode) == 1
        target = jnp.where(
            test_mode,
            p.buffer.true_test_output_grids[task, pair],
            p.buffer.output_grids_examples[task, pair],
        )
        target_mask = jnp.where(
            test_mode,
            p.buffer.true_test_output_masks[task, pair],
            p.buffer.output_masks_examples[task, pair],
        )
        exact, accuracy = score_grid(
            state.working_grid, state.working_grid_mask, target, target_mask
        )
        # Only a submit ends an episode as TERMINATED.
        submitted = timestep.step_type == StepType.TERMINATED
        outcome = {
            "task_idx": jnp.asarray(task, jnp.int32),
            "pair_idx": jnp.asarray(pair, jnp.int32),
            "submitted": submitted,
            "exact": submitted & exact,
            "cell_accuracy": jnp.where(submitted, accuracy, 0.0),
        }
        return timestep.replace(extras={**timestep.extras, "eval_metrics": outcome})
