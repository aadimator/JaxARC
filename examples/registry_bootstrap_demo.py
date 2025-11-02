#!/usr/bin/env python3
"""
Comprehensive JaxARC Registry and Discovery API Demo

This example showcases the complete registration system including:
1. Discovery API - Explore available datasets, subsets, and tasks
2. Unified make() - Create environments with flexible selectors
3. Cross-Split Loading - Load tasks from multiple splits (train/eval)
4. Custom Subsets - Register and use named subsets
5. Single Task Selection - Work with individual tasks
6. Curriculum Learning - Progressive difficulty across splits
7. Concept Groups - ConceptARC systematic evaluation
8. Benchmark Suites - Create evaluation sets with mixed tasks
9. Automatic Fallback - Cross-split task lookup

Examples demonstrated:
- Discovery: available_named_subsets(), get_subset_task_ids(), available_task_ids()
- Mini-all (all tasks), Mini-easy (custom subset), Mini-<task_id> (single task)
- Concept-Center (concept group)
- Cross-split loading for curriculum learning and mixed training
- Benchmark suites with selected tasks from multiple splits
- Automatic cross-split fallback for seamless task loading
- AGI1/AGI2 train/eval splits (optional, requires ENABLE_AGI_DEMO=1)

This replaces both cross_split_loading.py and discovery_demo.py with a unified demo.

Run:
    pixi run python examples/registry_bootstrap_demo.py

Environment variables:
- ENABLE_AGI_DEMO=1        # Include AGI-1 and AGI-2 demos (large downloads)
- ENABLE_CROSSSPLIT_DEMO=1 # Include cross-split loading demos (requires AGI datasets)
"""

from __future__ import annotations

import os

import jax
from loguru import logger

from jaxarc import JaxArcConfig
from jaxarc.registration import (
    available_named_subsets,
    available_task_ids,
    get_subset_task_ids,
    make,
    register_subset,
)
from jaxarc.utils.buffer import buffer_size


def run_simple_demo(id_str: str, config: JaxArcConfig | None = None) -> None:
    """Create env+params via registry.make, run a single reset/step, and print summary."""
    logger.info(f"=== Demo: {id_str} ===")

    if config is None:
        config = JaxArcConfig()

    # auto_download=True allows the registry to fetch the dataset if missing.
    env, params = make(id_str, config=config, auto_download=True)

    # Show buffer size and mode
    try:
        bs = buffer_size(params.buffer)
    except Exception:
        bs = "unknown"
    logger.info(f"Buffer size: {bs}")
    logger.info(f"Episode mode: {int(params.episode_mode)}  (0=train, 1=eval)")

    # Demonstrate spaces API
    obs_space = env.observation_space(params)
    action_space = env.action_space(params)
    reward_space = env.reward_space(params)

    logger.info(f"Observation space: {obs_space}")
    logger.info(f"Action space: {action_space}")
    logger.info(f"Reward space: {reward_space}")

    # Reset and take one step
    key = jax.random.PRNGKey(0)
    state, ts0 = env.reset(key, env_params=params)

    # Sample a random action from action space
    key, action_key = jax.random.split(key)
    action = action_space.sample(action_key)
    state, ts1 = env.step(state, action, env_params=params)

    logger.info(f"timestep0 - first(): {bool(ts0.first())}")  # True = FIRST
    logger.info(
        f"timestep1 - last(): {bool(ts1.last())}"
    )  # May be True if episode ends
    logger.info(f"reward on action: {float(ts1.reward)}")
    logger.info("-" * 80)


def demo_1_discovery_api() -> None:
    """Demonstrate the discovery API for exploring available datasets and subsets."""
    logger.info("=" * 80)
    logger.info("DEMO 1: DISCOVERY API")
    logger.info("=" * 80)

    datasets = [("Mini", "MiniARC"), ("Concept", "ConceptARC")]
    if os.environ.get("ENABLE_AGI_DEMO", "0") == "1":
        datasets.extend([("AGI1", "ARC-AGI-1"), ("AGI2", "ARC-AGI-2")])

    for dataset_key, dataset_name in datasets:
        logger.info(f"\n--- {dataset_name} ({dataset_key}) ---")

        # Show available named subsets (includes built-in selectors and concept groups)
        subsets = available_named_subsets(dataset_key)
        logger.info(f"Available subsets: {', '.join(subsets)}")

        # Show task counts for key selectors
        try:
            all_ids = get_subset_task_ids(dataset_key, "all", auto_download=True)
            logger.info(f"  'all': {len(all_ids)} tasks")

            # Show a sample task ID
            if all_ids:
                logger.info(f"  Sample task ID: {all_ids[0]}")

            # For datasets with concept groups or splits, show some examples
            if dataset_key == "Concept" and len(subsets) > 2:
                # Show first concept group that's not 'all'
                concept = next((s for s in subsets if s != "all"), None)
                if concept:
                    concept_ids = get_subset_task_ids(
                        dataset_key, concept, auto_download=True
                    )
                    logger.info(f"  '{concept}' concept: {len(concept_ids)} tasks")

            elif dataset_key in ("AGI1", "AGI2"):
                if "train" in subsets:
                    train_ids = get_subset_task_ids(
                        dataset_key, "train", auto_download=True
                    )
                    logger.info(f"  'train' split: {len(train_ids)} tasks")
                if "eval" in subsets:
                    eval_ids = get_subset_task_ids(
                        dataset_key, "eval", auto_download=True
                    )
                    logger.info(f"  'eval' split: {len(eval_ids)} tasks")

        except Exception as e:
            logger.info(f"  (Dataset not available: {e})")

    logger.info("\n" + "=" * 80 + "\n")


def demo_2_basic_usage() -> None:
    """Demonstrate basic environment creation with different selectors."""
    logger.info("=" * 80)
    logger.info("DEMO 2: BASIC USAGE")
    logger.info("=" * 80)

    config = JaxArcConfig()

    # Demo 2a: All tasks
    logger.info("\n--- 2a: Mini-all (all tasks) ---")
    run_simple_demo("Mini-all", config)

    # Demo 2b: Single task selection
    logger.info("\n--- 2b: Single task selection ---")
    try:
        available_ids = available_task_ids("Mini", config=config, auto_download=True)
        if available_ids:
            task_id = available_ids[0]
            logger.info(f"Selecting single task: {task_id}")
            run_simple_demo(f"Mini-{task_id}", config)
    except Exception as e:
        logger.error(f"Failed to run single task demo: {e}")

    # Demo 2c: ConceptARC concept group
    logger.info("\n--- 2c: ConceptARC concept group ---")
    try:
        concept_subsets = available_named_subsets("Concept")
        concepts = [s for s in concept_subsets if s != "all"]
        if concepts:
            logger.info(f"Available concepts: {', '.join(concepts[:5])}...")
            run_simple_demo("Concept-Center", config)
    except Exception as e:
        logger.error(f"Failed to run Concept demo: {e}")


def demo_3_custom_subsets() -> None:
    """Demonstrate creating and using custom named subsets."""
    logger.info("=" * 80)
    logger.info("DEMO 3: CUSTOM SUBSETS")
    logger.info("=" * 80)

    config = JaxArcConfig()

    try:
        # Get available tasks
        available_ids = available_task_ids("Mini", config=config, auto_download=True)
        if len(available_ids) < 10:
            logger.warning("Not enough tasks for custom subset demo")
            return

        # Create an 'easy' subset with first 3 tasks
        logger.info("\n--- 3a: Registering custom 'easy' subset ---")
        easy_ids = available_ids[:3]
        register_subset("Mini", "easy", easy_ids)
        logger.info(f"Registered 'easy' subset with {len(easy_ids)} tasks: {easy_ids}")

        # Show that it's now available
        subsets = available_named_subsets("Mini")
        logger.info(f"Available subsets after registration: {', '.join(subsets)}")

        # Use the custom subset
        logger.info("\n--- 3b: Using custom 'easy' subset ---")
        run_simple_demo("Mini-easy", config)

        # Create a 'medium' subset
        logger.info("\n--- 3c: Registering custom 'medium' subset ---")
        medium_ids = available_ids[3:8]
        register_subset("Mini", "medium", medium_ids)
        logger.info(f"Registered 'medium' subset with {len(medium_ids)} tasks")

        # Show distinction between built-in and custom subsets
        logger.info("\n--- 3d: Built-in vs Custom subsets ---")
        all_subsets = available_named_subsets("Mini", include_builtin=True)
        custom_only = available_named_subsets("Mini", include_builtin=False)
        logger.info(f"All subsets: {', '.join(all_subsets)}")
        logger.info(f"Custom only: {', '.join(custom_only)}")

    except Exception as e:
        logger.error(f"Failed custom subset demo: {e}")


def demo_4_cross_split_loading() -> None:
    """Demonstrate cross-split loading for AGI datasets (train + eval)."""
    if os.environ.get("ENABLE_CROSSSPLIT_DEMO", "0") != "1":
        logger.info("=" * 80)
        logger.info("DEMO 4: CROSS-SPLIT LOADING (SKIPPED)")
        logger.info("Set ENABLE_CROSSSPLIT_DEMO=1 to run this demo")
        logger.info("=" * 80)
        return

    logger.info("=" * 80)
    logger.info("DEMO 4: CROSS-SPLIT LOADING")
    logger.info("=" * 80)

    config = JaxArcConfig()

    try:
        # Demo 4a: Basic cross-split loading
        logger.info("\n--- 4a: Basic cross-split loading ---")
        train_ids = get_subset_task_ids(
            "AGI2", "train", config=config, auto_download=True
        )
        eval_ids = get_subset_task_ids(
            "AGI2", "eval", config=config, auto_download=True
        )

        logger.info(f"Training split: {len(train_ids)} tasks")
        logger.info(f"Evaluation split: {len(eval_ids)} tasks")

        # Create mixed subset
        mixed_ids = train_ids[:5] + eval_ids[:5]
        logger.info(f"\nCreating mixed subset with {len(mixed_ids)} tasks")
        logger.info(f"  - {len(train_ids[:5])} from training split")
        logger.info(f"  - {len(eval_ids[:5])} from evaluation split")

        register_subset("AGI2", "demo_mixed", mixed_ids)
        env, params = make("AGI2-demo_mixed", config=config, auto_download=True)
        logger.info("✓ Successfully created environment with mixed subset")

        # Test environment functionality
        key = jax.random.PRNGKey(0)
        state, timestep = env.reset(key, params)
        logger.info("✓ Environment reset successful")
        logger.info(f"  Observation shape: {timestep.observation.shape}")

        # Demo 4b: Curriculum learning
        logger.info("\n--- 4b: Curriculum learning ---")
        logger.info("Creating progressive difficulty stages:")

        # Stage 1: Easy (first 10 from training)
        stage_1 = train_ids[:10]
        register_subset("AGI2", "curriculum_stage1", stage_1)
        logger.info(f"  Stage 1 (Easy): {len(stage_1)} tasks from training")

        # Stage 2: Medium (next 10 from training + first 5 from eval)
        stage_2 = train_ids[10:20] + eval_ids[:5]
        register_subset("AGI2", "curriculum_stage2", stage_2)
        logger.info(f"  Stage 2 (Medium): {len(stage_2)} tasks (mixed splits)")

        # Stage 3: Hard (more eval tasks)
        stage_3 = train_ids[20:25] + eval_ids[5:15]
        register_subset("AGI2", "curriculum_stage3", stage_3)
        logger.info(f"  Stage 3 (Hard): {len(stage_3)} tasks (more eval tasks)")

        # Create stage 2 environment
        env, params = make("AGI2-curriculum_stage2", config=config, auto_download=True)
        logger.info("✓ Created curriculum stage 2 environment")

        # Test reset
        key = jax.random.PRNGKey(42)
        state, timestep = env.reset(key, params)
        logger.info("✓ Environment ready for training")

        # Demo 4c: Benchmark suite
        logger.info("\n--- 4c: Benchmark suite creation ---")
        benchmark_tasks = [
            train_ids[0],  # First training task
            train_ids[50]
            if len(train_ids) > 50
            else train_ids[-1],  # Mid training task
            train_ids[100]
            if len(train_ids) > 100
            else train_ids[-1],  # Another training task
            eval_ids[0],  # First eval task
            eval_ids[10] if len(eval_ids) > 10 else eval_ids[-1],  # Another eval task
        ]

        logger.info(
            f"Creating benchmark with {len(benchmark_tasks)} carefully selected tasks:"
        )
        logger.info(
            f"  - {sum(1 for t in benchmark_tasks if t in train_ids)} from training"
        )
        logger.info(
            f"  - {sum(1 for t in benchmark_tasks if t in eval_ids)} from evaluation"
        )

        register_subset("AGI2", "benchmark_suite", benchmark_tasks)
        env, params = make("AGI2-benchmark_suite", config=config, auto_download=True)
        logger.info("✓ Benchmark environment ready for evaluation")

        # Demo 4d: Automatic cross-split fallback
        logger.info("\n--- 4d: Automatic cross-split fallback ---")
        logger.info("System automatically finds tasks in both splits:")

        mixed_subset = [
            train_ids[0],  # From training
            train_ids[1],  # From training
            eval_ids[0],  # From eval - automatically found
            eval_ids[1],  # From eval - automatically found
        ]

        logger.info(f"  Task 1: {mixed_subset[0]} (from train)")
        logger.info(f"  Task 2: {mixed_subset[1]} (from train)")
        logger.info(f"  Task 3: {mixed_subset[2]} (from eval)")
        logger.info(f"  Task 4: {mixed_subset[3]} (from eval)")

        register_subset("AGI2", "auto_fallback_demo", mixed_subset)
        env, params = make("AGI2-auto_fallback_demo", config=config, auto_download=True)
        logger.info("✓ All tasks loaded with automatic cross-split lookup!")

    except Exception as e:
        logger.error(f"Cross-split demo failed: {e}")
        logger.info("(This requires AGI-2 dataset to be available)")


def demo_5_agi_datasets() -> None:
    """Demonstrate AGI dataset usage (optional, large datasets)."""
    if os.environ.get("ENABLE_AGI_DEMO", "0") != "1":
        logger.info("=" * 80)
        logger.info("DEMO 5: AGI DATASETS (SKIPPED)")
        logger.info("Set ENABLE_AGI_DEMO=1 to run this demo")
        logger.info("=" * 80)
        return

    logger.info("=" * 80)
    logger.info("DEMO 5: AGI DATASETS")
    logger.info("=" * 80)

    config = JaxArcConfig()

    try:
        logger.info("\n--- 5a: AGI-1 Training Split ---")
        run_simple_demo("AGI1-train", config)

        logger.info("\n--- 5b: AGI-2 Evaluation Split ---")
        run_simple_demo("AGI2-eval", config)

    except Exception as e:
        logger.error(f"AGI demos failed: {e}")


def demo_6_advanced_discovery() -> None:
    """Demonstrate advanced discovery patterns and comprehensive subset exploration."""
    logger.info("=" * 80)
    logger.info("DEMO 6: ADVANCED DISCOVERY & SUBSET EXPLORATION")
    logger.info("=" * 80)

    try:
        # Demo 6a: Comprehensive dataset exploration
        logger.info("\n--- 6a: Comprehensive dataset exploration ---")
        datasets = [("Mini", "MiniARC"), ("Concept", "ConceptARC")]
        if os.environ.get("ENABLE_AGI_DEMO", "0") == "1":
            datasets.extend([("AGI1", "ARC-AGI-1"), ("AGI2", "ARC-AGI-2")])

        for dataset_key, dataset_name in datasets:
            logger.info(f"\n{dataset_name} ({dataset_key}) Analysis:")

            # Show all available subsets
            all_subsets = available_named_subsets(dataset_key, include_builtin=True)
            custom_subsets = available_named_subsets(dataset_key, include_builtin=False)
            builtin_subsets = [s for s in all_subsets if s not in custom_subsets]

            logger.info(f"  Built-in subsets: {', '.join(builtin_subsets)}")
            logger.info(
                f"  Custom subsets: {', '.join(custom_subsets) if custom_subsets else '(none)'}"
            )

            # Show task counts for built-in subsets
            for subset in builtin_subsets[:5]:  # Limit to first 5 to avoid spam
                try:
                    task_ids = get_subset_task_ids(
                        dataset_key, subset, auto_download=True
                    )
                    logger.info(f"    {subset}: {len(task_ids)} tasks")
                    if subset == "all" and task_ids:
                        logger.info(f"      Sample task: {task_ids[0]}")
                except Exception as e:
                    logger.info(f"    {subset}: Error - {e}")

        # Demo 6b: ConceptARC concept group deep dive
        logger.info("\n--- 6b: ConceptARC concept groups deep dive ---")
        try:
            concept_subsets = available_named_subsets("Concept")
            concepts = [s for s in concept_subsets if s != "all"]

            logger.info(f"Available concept groups ({len(concepts)} total):")
            for i, concept in enumerate(concepts[:10], 1):  # Show first 10
                try:
                    concept_ids = get_subset_task_ids(
                        "Concept", concept, auto_download=True
                    )
                    logger.info(f"  {i:2d}. {concept:15s}: {len(concept_ids):2d} tasks")
                    if i <= 3 and concept_ids:  # Show sample task IDs for first 3
                        logger.info(f"      Sample: {concept_ids[0]}")
                except Exception:
                    logger.info(f"  {i:2d}. {concept:15s}: Error loading")

            if len(concepts) > 10:
                logger.info(f"  ... and {len(concepts) - 10} more concept groups")

            # Test creating environment with concept group
            if concepts:
                test_concept = concepts[0]
                logger.info(
                    f"\nTesting environment creation with '{test_concept}' concept:"
                )
                env, params = make(f"Concept-{test_concept}", auto_download=True)
                logger.info("  ✓ Environment created successfully")

        except Exception as e:
            logger.error(f"ConceptARC analysis failed: {e}")

        # Demo 6c: Single task selection across datasets
        logger.info("\n--- 6c: Single task selection validation ---")
        test_datasets = ["Mini", "Concept"]

        for dataset in test_datasets:
            try:
                # Get all tasks
                all_tasks = get_subset_task_ids(dataset, "all", auto_download=True)
                if not all_tasks:
                    logger.info(f"{dataset}: No tasks available")
                    continue

                # Test single task resolution
                task_id = all_tasks[0]
                logger.info(f"{dataset}: Testing single task '{task_id}'")

                # Verify resolution
                resolved_ids = get_subset_task_ids(dataset, task_id, auto_download=True)
                assert len(resolved_ids) == 1, (
                    f"Expected 1 task, got {len(resolved_ids)}"
                )
                assert resolved_ids[0] == task_id, (
                    f"Task ID mismatch: {resolved_ids[0]} != {task_id}"
                )

                # Create environment
                env, params = make(f"{dataset}-{task_id}", auto_download=True)
                logger.info("  ✓ Single task environment created successfully")

                # Test functionality
                key = jax.random.PRNGKey(0)
                state, timestep = env.reset(key, params)
                logger.info(
                    f"  ✓ Environment reset successful, obs shape: {timestep.observation.shape}"
                )

            except Exception as e:
                logger.error(f"{dataset}: Single task test failed - {e}")

        # Demo 6d: Task ID pattern analysis
        logger.info("\n--- 6d: Task ID pattern analysis ---")
        try:
            mini_ids = get_subset_task_ids("Mini", "all", auto_download=True)
            concept_ids = get_subset_task_ids("Concept", "all", auto_download=True)

            logger.info("Task ID patterns:")
            logger.info("  MiniARC task IDs (sample):")
            for task_id in mini_ids[:3]:
                logger.info(f"    - {task_id}")

            logger.info("  ConceptARC task IDs (sample):")
            for task_id in concept_ids[:3]:
                logger.info(f"    - {task_id}")

            # Analyze concept structure
            if concept_ids:
                concept_groups = set()
                for task_id in concept_ids:
                    if "/" in task_id:
                        concept_group = task_id.split("/")[0]
                        concept_groups.add(concept_group)

                logger.info(
                    f"  ConceptARC groups found in task IDs: {len(concept_groups)}"
                )
                logger.info(f"    Sample groups: {', '.join(list(concept_groups)[:5])}")

        except Exception as e:
            logger.error(f"Task ID analysis failed: {e}")

        # Demo 6e: Subset composition and validation
        logger.info("\n--- 6e: Subset composition validation ---")
        try:
            # Create and validate a custom subset
            all_mini = get_subset_task_ids("Mini", "all", auto_download=True)
            if len(all_mini) >= 5:
                # Create custom subset
                test_subset = all_mini[:5]
                register_subset("Mini", "validation_test", test_subset)

                # Validate it appears in listings
                subsets = available_named_subsets("Mini", include_builtin=False)
                assert "validation_test" in subsets, (
                    "Custom subset not found in listings"
                )

                # Validate task resolution
                resolved = get_subset_task_ids(
                    "Mini", "validation_test", auto_download=True
                )
                assert resolved == test_subset, "Custom subset task resolution failed"

                # Validate environment creation
                env, params = make("Mini-validation_test", auto_download=True)
                logger.info("  ✓ Custom subset validation passed")

                # Test against original tasks
                logger.info(
                    f"  Custom subset has {len(test_subset)} tasks from {len(all_mini)} total"
                )

        except Exception as e:
            logger.error(f"Subset validation failed: {e}")

    except Exception as e:
        logger.error(f"Advanced discovery demo failed: {e}")


def demo_7_builtin_vs_custom_analysis() -> None:
    """Demonstrate the distinction between built-in and custom subsets."""
    logger.info("=" * 80)
    logger.info("DEMO 7: BUILT-IN VS CUSTOM SUBSET ANALYSIS")
    logger.info("=" * 80)

    try:
        config = JaxArcConfig()

        # Demo 7a: Before custom subsets
        logger.info("\n--- 7a: Initial state (before custom subsets) ---")
        for dataset in ["Mini", "Concept"]:
            all_subsets = available_named_subsets(dataset, include_builtin=True)
            custom_subsets = available_named_subsets(dataset, include_builtin=False)
            builtin_subsets = [s for s in all_subsets if s not in custom_subsets]

            logger.info(f"{dataset}:")
            logger.info(f"  All subsets: {', '.join(all_subsets)}")
            logger.info(f"  Built-in: {', '.join(builtin_subsets)}")
            logger.info(
                f"  Custom: {', '.join(custom_subsets) if custom_subsets else '(none)'}"
            )

        # Demo 7b: Add custom subsets
        logger.info("\n--- 7b: Adding custom subsets ---")
        try:
            mini_tasks = get_subset_task_ids("Mini", "all", auto_download=True)
            if len(mini_tasks) >= 10:
                # Create multiple custom subsets
                register_subset("Mini", "first_five", mini_tasks[:5])
                register_subset("Mini", "next_five", mini_tasks[5:10])
                register_subset(
                    "Mini", "mixed_sample", mini_tasks[::2][:5]
                )  # Every other task

                logger.info("Created custom subsets:")
                logger.info(f"  'first_five': {mini_tasks[:5]}")
                logger.info(f"  'next_five': {mini_tasks[5:10]}")
                logger.info(f"  'mixed_sample': {mini_tasks[::2][:5]}")

        except Exception as e:
            logger.error(f"Failed to create custom subsets: {e}")

        # Demo 7c: After custom subsets
        logger.info("\n--- 7c: After adding custom subsets ---")
        for dataset in ["Mini", "Concept"]:
            all_subsets = available_named_subsets(dataset, include_builtin=True)
            custom_subsets = available_named_subsets(dataset, include_builtin=False)
            builtin_subsets = [s for s in all_subsets if s not in custom_subsets]

            logger.info(f"{dataset}:")
            logger.info(f"  All subsets: {', '.join(all_subsets)}")
            logger.info(f"  Built-in: {', '.join(builtin_subsets)}")
            logger.info(
                f"  Custom: {', '.join(custom_subsets) if custom_subsets else '(none)'}"
            )

        # Demo 7d: Test custom subset functionality
        logger.info("\n--- 7d: Testing custom subset environments ---")
        custom_subsets = available_named_subsets("Mini", include_builtin=False)
        for subset_name in custom_subsets[:3]:  # Test first 3 custom subsets
            try:
                env, params = make(
                    f"Mini-{subset_name}", config=config, auto_download=True
                )
                logger.info(
                    f"  ✓ 'Mini-{subset_name}' environment created successfully"
                )
            except Exception as e:
                logger.error(f"  ✗ 'Mini-{subset_name}' failed: {e}")

    except Exception as e:
        logger.error(f"Built-in vs custom analysis failed: {e}")


def main() -> None:
    """Run comprehensive registry and discovery demos."""
    logger.info("\n" + "=" * 80)
    logger.info("JaxARC Registry & Discovery API - Comprehensive Demo")
    logger.info("Replaces: cross_split_loading.py + discovery_demo.py")
    logger.info("=" * 80 + "\n")

    # Demo 1: Discovery API - Explore what's available
    demo_1_discovery_api()

    # Demo 2: Basic usage patterns - Simple environment creation
    demo_2_basic_usage()

    # Demo 3: Custom subsets - Create and use named subsets
    demo_3_custom_subsets()

    # Demo 4: Cross-split loading - Mix train/eval tasks (requires ENABLE_CROSSSPLIT_DEMO=1)
    demo_4_cross_split_loading()

    # Demo 5: AGI datasets - Large dataset handling (requires ENABLE_AGI_DEMO=1)
    demo_5_agi_datasets()

    # Demo 6: Advanced discovery - Deep exploration and validation
    demo_6_advanced_discovery()

    # Demo 7: Built-in vs custom - Subset management analysis
    demo_7_builtin_vs_custom_analysis()

    logger.info("\n" + "=" * 80)
    logger.info("🎉 COMPREHENSIVE DEMO COMPLETE! 🎉")
    logger.info("=" * 80)
    logger.info("\nFunctionality Consolidated:")
    logger.info(
        "  ✓ cross_split_loading.py - Cross-split task loading for curriculum learning"
    )
    logger.info("  ✓ discovery_demo.py - Dataset exploration and subset discovery")
    logger.info("  ✓ Additional comprehensive validation and analysis")
    logger.info("\n" + "=" * 40 + " KEY TAKEAWAYS " + "=" * 40)
    logger.info("Discovery & Exploration:")
    logger.info("  • available_named_subsets() - Find all available subsets")
    logger.info("  • get_subset_task_ids() - Get task lists for any subset")
    logger.info("  • available_task_ids() - Quick access to all tasks")
    logger.info("\nEnvironment Creation:")
    logger.info(
        "  • make('Dataset-subset') - Create environments with flexible selectors"
    )
    logger.info("  • make('Dataset-task_id') - Single task environments")
    logger.info("  • make('Concept-Center') - ConceptARC concept groups")
    logger.info("\nAdvanced Features:")
    logger.info("  • register_subset() - Create custom named subsets")
    logger.info(
        "  • Cross-split loading - Automatic train/eval mixing for AGI datasets"
    )
    logger.info("  • Curriculum learning - Progressive difficulty with mixed splits")
    logger.info("  • Benchmark suites - Curated evaluation sets")
    logger.info("  • Automatic fallback - Seamless cross-split task lookup")
    logger.info("\nBest Practices:")
    logger.info("  1. Use discovery API to explore before loading")
    logger.info("  2. Create custom subsets for reproducible experiments")
    logger.info("  3. Leverage cross-split loading for curriculum learning")
    logger.info("  4. Use single task selection for focused debugging")
    logger.info("  5. Build benchmark suites with carefully selected tasks")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
