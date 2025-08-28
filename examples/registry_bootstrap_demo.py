#!/usr/bin/env python3
"""
registry_bootstrap_demo.py

Demonstrates the buffer-based, JIT/vmap-compatible reset by:
- Building stacked task buffers from parsers (MiniARC, ConceptARC, optional AGI1/AGI2)
- Creating EnvParams with the buffer
- Using registry.make(...) with provided params
- Running a minimal reset/step loop with the functional API

Run:
    pixi run python examples/registry_bootstrap_demo.py

Notes:
- This example loads tasks into memory (buffer) once per subset and reuses it.
- For large datasets (AGI), it limits the number of tasks by default; use env vars to override.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxarc.configs.main_config import JaxArcConfig
from jaxarc.envs import create_point_action, reset as env_reset, step as env_step
from jaxarc.parsers import ArcAgiParser, ConceptArcParser, MiniArcParser
from jaxarc.registration import make
from jaxarc.types import EnvParams, JaxArcTask
from jaxarc.utils.buffer import buffer_size, stack_task_list
from jaxarc.utils.config import get_config


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#


def _preview_ids(title: str, ids: Sequence[str], limit: int = 5) -> None:
    n = len(ids)
    print(f"[{title}] subset size: {n}")
    if n > 0:
        head = list(ids[: min(limit, n)])
        print(f"[{title}] first {min(limit, n)} ids: {head}")
    print("-" * 80)


def _stack_tasks_from_ids(
    parser, ids: Sequence[str], limit: int | None = None
) -> JaxArcTask:
    """Build a stacked JAX-native buffer (batched pytree) from task ids."""
    chosen = list(ids if limit is None else ids[:limit])
    if len(chosen) == 0:
        raise RuntimeError("No task ids available to build the buffer.")
    tasks: list[JaxArcTask] = [parser.get_task_by_id(tid) for tid in chosen]
    return stack_task_list(tasks)


def _params_from_buffer(cfg: JaxArcConfig, buf: JaxArcTask, episode_mode: int = 0) -> EnvParams:
    """Create EnvParams with the buffer and optional subset view."""
    return EnvParams.from_config(
        cfg,
        buffer=buf,
        subset_indices=None,  # use full buffer
        episode_mode=episode_mode,
    )


def _run_one_episode(params: EnvParams, submit_op: int = 34) -> None:
    """Reset once and take a single SUBMIT step."""
    key = jax.random.PRNGKey(0)

    # JIT-friendly reset and one step
    ts0 = env_reset(params, key)
    action = create_point_action(operation=jnp.int32(submit_op), row=jnp.int32(0), col=jnp.int32(0))
    ts1 = env_step(params, ts0, action)

    print("timestep0 step_type:", int(ts0.step_type))
    print("timestep1 step_type:", int(ts1.step_type), "(2 means last)")
    print("reward on submit:", float(ts1.reward))


# -----------------------------------------------------------------------------#
# Demos
# -----------------------------------------------------------------------------#


def demo_mini() -> None:
    print("=== MiniARC (buffer-based) ===")
    # Load Hydra config for MiniARC; convert to typed config
    hydra_cfg = get_config(overrides=["dataset=mini_arc"])
    cfg = JaxArcConfig.from_hydra(hydra_cfg)

    # Build parser and list of ids
    parser = MiniArcParser(cfg.dataset)
    ids = parser.get_available_task_ids()

    # Optional limit (env var) to build smaller buffers for quick demo
    limit = int(os.environ.get("MINIARC_LIMIT", "200"))
    buf = _stack_tasks_from_ids(parser, ids, limit=limit)
    _preview_ids("Mini", ids, limit=5)
    print("Mini buffer size:", buffer_size(buf))

    # Build EnvParams (train mode)
    params = _params_from_buffer(cfg, buf, episode_mode=0)

    # Construct env via registry (params must be provided)
    env, params = make("Mini", params=params)
    _run_one_episode(params)
    print("-" * 80)


def demo_concept() -> None:
    print("=== ConceptARC (buffer-based) ===")
    hydra_cfg = get_config(overrides=["dataset=concept_arc"])
    cfg = JaxArcConfig.from_hydra(hydra_cfg)

    parser = ConceptArcParser(cfg.dataset)

    # Pick a concept group (prefer AboveBelow if available, else the first)
    groups = parser.get_concept_groups()
    concept = "AboveBelow" if "AboveBelow" in groups else (groups[0] if groups else None)
    if concept is None:
        raise RuntimeError("No concept groups available in ConceptARC dataset.")

    ids = parser.get_tasks_in_concept(concept)
    _preview_ids(f"Concept-{concept}", ids, limit=5)

    # Optional limit via env var
    limit = int(os.environ.get("CONCEPT_LIMIT", "50"))
    buf = _stack_tasks_from_ids(parser, ids, limit=limit)
    print(f"Concept buffer size ({concept}):", buffer_size(buf))

    # Train mode params
    params = _params_from_buffer(cfg, buf, episode_mode=0)

    env, params = make("Concept", params=params)
    _run_one_episode(params)
    print("-" * 80)


def demo_agi(dataset_key: str = "AGI1") -> None:
    print(f"=== {dataset_key} (buffer-based, optional) ===")
    if os.environ.get("ENABLE_AGI_DEMO", "0") != "1":
        print("AGI demo disabled. Set ENABLE_AGI_DEMO=1 to enable.")
        print("-" * 80)
        return

    # Choose dataset and create parser
    override = "dataset=arc_agi_1" if dataset_key.upper().startswith("AGI1") else "dataset=arc_agi_2"
    hydra_cfg = get_config(overrides=[override])
    cfg = JaxArcConfig.from_hydra(hydra_cfg)
    parser = ArcAgiParser(cfg.dataset)

    # Use all ids but build a smaller buffer for demo via env var
    ids = parser.get_available_task_ids()
    limit_default = "128" if dataset_key.upper().startswith("AGI1") else "64"
    limit_env = "AGI1_LIMIT" if dataset_key.upper().startswith("AGI1") else "AGI2_LIMIT"
    limit = int(os.environ.get(limit_env, limit_default))

    _preview_ids(dataset_key, ids, limit=5)
    buf = _stack_tasks_from_ids(parser, ids, limit=limit)
    print(f"{dataset_key} buffer size:", buffer_size(buf))

    # Evaluation mode by default for AGI (episode_mode=1), change if desired
    params = _params_from_buffer(cfg, buf, episode_mode=1)

    env, params = make(dataset_key, params=params)
    _run_one_episode(params)
    print("-" * 80)


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#


def main() -> None:
    demo_mini()
    demo_concept()
    demo_agi("AGI1")
    demo_agi("AGI2")
    print("Demo complete.")


if __name__ == "__main__":
    main()
