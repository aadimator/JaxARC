#!/usr/bin/env python3
"""
registry_bootstrap_demo.py

Demonstrates the new registry-driven, buffer-backed API where make("<Dataset>-<Selector>")
automatically:
- resolves the appropriate parser,
- ensures the dataset is available (optionally downloading),
- resolves the selector (all, split, concept group, or specific task id),
- builds a stacked JAX buffer for the resolved tasks, and
- returns (env, params) with a fully JIT/vmap-compatible reset.

Examples demonstrated:
- Mini-all
- Mini-easy (named subset)
- Concept-AboveBelow
- AGI1-train (optional, large)
- AGI2-eval (optional, large)

Run:
    pixi run python examples/registry_bootstrap_demo.py

Environment variables:
- ENABLE_AGI_DEMO=1  # include AGI-1 and AGI-2 demos (downloads can be large)
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

from jaxarc.configs.main_config import JaxArcConfig
from jaxarc.envs import create_point_action
from jaxarc.envs import reset as env_reset
from jaxarc.envs import step as env_step
from jaxarc.registration import (
    available_named_subsets,
    make,
    register_subset,
    subset_task_ids,
)
from jaxarc.utils.buffer import buffer_size
from jaxarc.utils.config import get_config


def run_demo(id_str: str) -> None:
    """Create env+params via registry.make, run a single reset/step, and print summary."""
    print(f"=== Demo: {id_str} ===")

    # Config can be minimal; registry will normalize dataset and ensure availability.
    # If config.dataset.dataset_path is empty, registry uses ./data/<DatasetName> by default for downloads.
    cfg = get_config()
    cfg = JaxArcConfig.from_hydra(cfg)

    # auto_download=True allows the registry to fetch the dataset if missing.
    env, params = make(id_str, config=cfg, auto_download=True)

    # Show buffer size and mode
    try:
        bs = buffer_size(params.buffer)
    except Exception:
        bs = "unknown"
    print(f"Buffer size: {bs}")
    print(f"Episode mode: {int(params.episode_mode)}  (0=train, 1=eval)")

    # Minimal reset/step: submit immediately
    key = jax.random.PRNGKey(0)
    ts0 = env_reset(params, key)
    action = create_point_action(
        operation=jnp.int32(34), row=jnp.int32(0), col=jnp.int32(0)
    )  # SUBMIT
    ts1 = env_step(params, ts0, action)

    print("timestep0.step_type:", int(ts0.step_type))  # 0 = FIRST
    print("timestep1.step_type:", int(ts1.step_type), "(2 means LAST)")
    print("reward on submit:", float(ts1.reward))
    print("-" * 80)


def main() -> None:
    # Register and demonstrate a named subset for MiniARC
    register_subset(
        "Mini",
        "easy",
        [
            "Most_Common_color_l6ab0lf3xztbyxsu3p",
            "Simple_Color_Fill__l6af3wjj3htf3r242ir",
            "Simple_Unique_Box_l6adthlbktjkouruq0j",
            "Simple_Box_Moving_l6aapas5si5cuue2txa",
            "define_boundary_l6aeugn2pfna6pvwdt",
        ],
    )
    run_demo("Mini-easy")
    print("Available named subsets for MiniARC:", available_named_subsets("Mini"))
    print("Available task IDs:", subset_task_ids("Mini", "easy"))

    # MiniARC: all tasks
    # run_demo("Mini-all")
    run_demo("Mini-Most_Common_color_l6ab0lf3xztbyxsu3p")

    # ConceptARC: a specific concept group
    run_demo("Concept-AboveBelow")

    # Optional AGI demos (datasets are large)
    if os.environ.get("ENABLE_AGI_DEMO", "0") == "1":
        run_demo("AGI1-train")
        run_demo("AGI2-eval")

    print("Demo complete.")


if __name__ == "__main__":
    main()
