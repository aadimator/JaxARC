#!/usr/bin/env python3
"""
Demonstrates the new registry-driven, buffer-backed API where make("<Dataset>-<Selector>") automatically:
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

from jaxarc.configs.main_config import JaxArcConfig
from jaxarc.envs.actions import Action
from jaxarc.registration import (
    available_named_subsets,
    make,
    register_subset,
    subset_task_ids,
)
from jaxarc.utils.buffer import buffer_size
from jaxarc.utils.core import get_config


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
    
    # Demonstrate spaces API
    obs_space = env.observation_space(params)
    action_space = env.action_space(params)
    reward_space = env.reward_space(params)
    
    print(f"Observation space: {obs_space}")
    print(f"Action space: {action_space}")
    print(f"Reward space: {reward_space}")

    # Minimal reset/step: submit immediately
    # Reset and take one step using functional API
    key = jax.random.PRNGKey(0)
    state, ts0 = env.reset(key, env_params=params)

    # Sample a random action from action space
    key, action_key = jax.random.split(key)
    action_dict = action_space.sample(action_key)
    # Convert dict to Action object for unwrapped environment
    action = Action(operation=action_dict["operation"], selection=action_dict["selection"])
    state, ts1 = env.step(state, action, env_params=params)

    print("timestep0 - first():", bool(ts0.first()))  # True = FIRST
    print("timestep1 - last():", bool(ts1.last()))  # May be True if episode ends
    print("reward on action:", float(ts1.reward))
    print("-" * 80)


def main() -> None:
    # Test with Mini-all (simpler demo)
    run_demo("Mini-all")

    # Register and demonstrate a named subset for MiniARC
    # First get some available task IDs
    from jaxarc.registration import available_task_ids
    try:
        available_ids = available_task_ids("Mini", auto_download=True)
        if len(available_ids) >= 3:
            # Use first few available task IDs 
            subset_ids = available_ids[:3]
            register_subset("Mini", "easy", subset_ids)
            print("Available named subsets for MiniARC:", available_named_subsets("Mini"))
            print("Available task IDs for 'easy' subset:", subset_task_ids("Mini", "easy"))
            run_demo("Mini-easy")
    except Exception as e:
        print(f"Failed to create subset demo: {e}")

    # ConceptARC: a specific concept group
    try:
        run_demo("Concept-AboveBelow")
    except Exception as e:
        print(f"Failed to run Concept demo: {e}")

    # Optional AGI demos (datasets are large)
    if os.environ.get("ENABLE_AGI_DEMO", "0") == "1":
        try:
            run_demo("AGI1-train")
            run_demo("AGI2-eval") 
        except Exception as e:
            print(f"Failed to run AGI demos: {e}")

    print("Demo complete.")


if __name__ == "__main__":
    main()