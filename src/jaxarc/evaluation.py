"""Held-out evaluation: run every pair of a split, then score per task.

``make_eval_reset_fn`` plugs into Stoix as ``env.eval_reset_fn``. With
``num_eval_episodes = k * n_pairs`` each pair of the eval split is run k times,
as independent attempts. The eval env from ``jaxarc.stoix_adapter`` reports
each episode's outcome in ``extras["eval_metrics"]`` (see ``EvalMetricsWrapper``),
and ``summarize`` turns those per-episode outcomes into per-task scores.
"""

from __future__ import annotations

from math import comb
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from jaxarc.wrappers.eval_metrics import EvalMetricsWrapper, score_grid

__all__ = [
    "EvalMetricsWrapper",
    "make_eval_reset_fn",
    "pass_at_k",
    "score_grid",
    "summarize",
]


def make_eval_reset_fn(config: Any, env: Any) -> Callable:
    """Stoix ``eval_reset_fn`` that resets episode i on pair ``i % n_pairs``.

    The pairs are those of the eval env's split and subset, in ``all_pairs()``
    order. ``config`` is unused; Stoix passes it.
    """
    del config
    params = env.unwrapped.params
    task_idx, pair_idx = params.all_pairs()
    n_pairs = int(task_idx.shape[0])

    def eval_reset_fn(key: jax.Array, num_environments: int):
        if num_environments % n_pairs != 0:
            msg = (
                f"{num_environments} eval episodes can't cover the {n_pairs} pairs "
                f"of the eval split evenly; use a multiple of {n_pairs}"
            )
            raise ValueError(msg)
        keys = jax.random.split(key, num_environments)
        i = jnp.arange(num_environments) % n_pairs
        return jax.vmap(lambda k, t, p: env.reset(k, params.with_pair(t, p)))(
            keys, task_idx[i], pair_idx[i]
        )

    return eval_reset_fn


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of k attempts drawn from n succeeds, given c did."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def summarize(metrics: dict, k: int = 2) -> dict:
    """Per-task scores from per-episode outcomes.

    ``metrics`` holds ``task_idx``, ``pair_idx``, ``exact``, ``cell_accuracy``
    and ``submitted`` arrays of any matching shape. Attempts at one pair are
    averaged, then pairs within a task, then tasks, so each task weighs the same.
    ``exact`` is the single-attempt success rate; ``pass@k`` uses the unbiased
    estimator over the attempts that were run.
    """
    flat = {
        key: np.asarray(metrics[key]).reshape(-1)
        for key in ("task_idx", "pair_idx", "exact", "cell_accuracy", "submitted")
    }
    pairs = np.stack([flat["task_idx"], flat["pair_idx"]], axis=1)
    unique_pairs, which, attempts = np.unique(
        pairs, axis=0, return_inverse=True, return_counts=True
    )
    which = which.reshape(-1)
    if attempts.min() < k:
        msg = (
            f"pass@{k} needs {k} attempts per pair, but some pairs had {attempts.min()}"
        )
        raise ValueError(msg)

    def per_pair(values):
        return np.bincount(which, weights=values.astype(np.float64)) / attempts

    solved = np.bincount(which, weights=flat["exact"].astype(np.float64)).astype(int)
    pair_scores = {
        "exact": per_pair(flat["exact"]),
        f"pass@{k}": np.array(
            [pass_at_k(int(n), int(c), k) for n, c in zip(attempts, solved)]
        ),
        "cell_accuracy": per_pair(flat["cell_accuracy"]),
        "submit_rate": per_pair(flat["submitted"]),
    }

    tasks, task_of_pair = np.unique(unique_pairs[:, 0], return_inverse=True)
    pairs_per_task = np.bincount(task_of_pair)
    per_task = {"task_idx": tasks}
    for name, scores in pair_scores.items():
        per_task[name] = np.bincount(task_of_pair, weights=scores) / pairs_per_task

    return {
        **{name: float(per_task[name].mean()) for name in pair_scores},
        "num_tasks": len(tasks),
        "num_pairs": len(unique_pairs),
        "min_attempts": int(attempts.min()),
        "per_task": per_task,
    }
