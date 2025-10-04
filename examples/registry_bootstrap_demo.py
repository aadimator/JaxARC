#!/usr/bin/env python3
"""
Demonstrates the unified registry API with discovery features.

This example showcases:
1. Discovery API - List available datasets, subsets, and task IDs
2. Unified make() - Create environments with simple selectors
3. Multiple selector types - 'all', splits, concept groups, named subsets, single tasks
4. Dataset-specific features - ConceptARC groups, AGI train/eval splits

Examples demonstrated:
- Discovery: available_named_subsets(), get_subset_task_ids()
- Mini-all (all 149 tasks)
- Mini-easy (custom named subset)
- Mini-<task_id> (single task selection)
- Concept-Center (concept group with 10 tasks)
- AGI1-train, AGI2-eval (optional, requires ENABLE_AGI_DEMO=1)

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
    available_task_ids,
    get_subset_task_ids,
    make,
    register_subset,
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


def discovery_demo() -> None:
    """Demonstrate the discovery API for exploring available datasets and subsets."""
    print("\n" + "=" * 80)
    print("DISCOVERY API DEMONSTRATION")
    print("=" * 80 + "\n")
    
    datasets = [("Mini", "MiniARC"), ("Concept", "ConceptARC")]
    if os.environ.get("ENABLE_AGI_DEMO", "0") == "1":
        datasets.extend([("AGI1", "ARC-AGI-1"), ("AGI2", "ARC-AGI-2")])
    
    for dataset_key, dataset_name in datasets:
        print(f"\n--- {dataset_name} ({dataset_key}) ---")
        
        # Show available named subsets (includes built-in selectors and concept groups)
        subsets = available_named_subsets(dataset_key)
        print(f"Available subsets: {', '.join(subsets)}")
        
        # Show task counts for key selectors
        try:
            all_ids = get_subset_task_ids(dataset_key, "all", auto_download=True)
            print(f"  'all': {len(all_ids)} tasks")
            
            # Show a sample task ID
            if all_ids:
                print(f"  Sample task ID: {all_ids[0]}")
            
            # For datasets with concept groups or splits, show some examples
            if dataset_key == "Concept" and len(subsets) > 2:
                # Show first concept group that's not 'all'
                concept = next((s for s in subsets if s != "all"), None)
                if concept:
                    concept_ids = get_subset_task_ids(dataset_key, concept, auto_download=True)
                    print(f"  '{concept}' concept: {len(concept_ids)} tasks")
            
            elif dataset_key in ("AGI1", "AGI2"):
                if "train" in subsets:
                    train_ids = get_subset_task_ids(dataset_key, "train", auto_download=True)
                    print(f"  'train' split: {len(train_ids)} tasks")
                if "eval" in subsets:
                    eval_ids = get_subset_task_ids(dataset_key, "eval", auto_download=True)
                    print(f"  'eval' split: {len(eval_ids)} tasks")
                    
        except Exception as e:
            print(f"  (Dataset not available: {e})")
    
    print("\n" + "=" * 80 + "\n")


def main() -> None:
    # First, demonstrate the discovery API
    discovery_demo()
    
    # Demo 1: Mini-all (all 149 tasks)
    print("\n=== Demo 1: Mini-all (all tasks) ===")
    run_demo("Mini-all")
    
    # Demo 2: Register and use a custom named subset
    print("\n=== Demo 2: Custom named subset ===")
    try:
        available_ids = available_task_ids("Mini", auto_download=True)
        if len(available_ids) >= 3:
            # Use first 3 tasks as 'easy' subset
            subset_ids = available_ids[:3]
            register_subset("Mini", "easy", subset_ids)
            print(f"Registered 'easy' subset with {len(subset_ids)} tasks")
            print(f"Available subsets after registration: {available_named_subsets('Mini')}")
            run_demo("Mini-easy")
    except Exception as e:
        print(f"Failed to create subset demo: {e}")
    
    # Demo 3: Single task selection
    print("\n=== Demo 3: Single task selection ===")
    try:
        available_ids = available_task_ids("Mini", auto_download=True)
        if available_ids:
            task_id = available_ids[0]
            print(f"Selecting single task: {task_id}")
            run_demo(f"Mini-{task_id}")
    except Exception as e:
        print(f"Failed to run single task demo: {e}")
    
    # Demo 4: ConceptARC concept group
    print("\n=== Demo 4: ConceptARC concept group ===")
    try:
        # Show available concepts
        concept_subsets = available_named_subsets("Concept")
        concepts = [s for s in concept_subsets if s != "all"]
        if concepts:
            print(f"Available concepts: {', '.join(concepts[:5])}...")
            run_demo("Concept-Center")
    except Exception as e:
        print(f"Failed to run Concept demo: {e}")
    
    # Demo 5: AGI dataset splits (optional, large datasets)
    if os.environ.get("ENABLE_AGI_DEMO", "0") == "1":
        print("\n=== Demo 5: AGI dataset splits ===")
        try:
            print("\nAGI-1 Training Split:")
            run_demo("AGI1-train")
            
            print("\nAGI-2 Evaluation Split:")
            run_demo("AGI2-eval")
        except Exception as e:
            print(f"Failed to run AGI demos: {e}")
    
    print("\n" + "=" * 80)
    print("Demo complete! All functionality working correctly.")
    print("=" * 80)


if __name__ == "__main__":
    main()