#!/usr/bin/env python3
"""
Discovery API Demo - Exploring Available Subsets and Tasks

Demonstrates how to use the discovery API to explore datasets before loading environments.
This includes:
- Listing available subsets (built-in and custom)
- Getting task IDs for specific subsets
- Single task selection

Run:
    pixi run python examples/discovery_demo.py
"""

from __future__ import annotations

from jaxarc.registration import (
    available_named_subsets,
    get_subset_task_ids,
    make,
    register_subset,
)


def explore_dataset(dataset_key: str) -> None:
    """Explore what's available in a dataset."""
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_key}")
    print(f"{'='*70}")

    # 1. See available subsets (includes built-in like 'all', 'train', 'eval')
    subsets = available_named_subsets(dataset_key)
    print(f"\nAvailable subsets: {', '.join(subsets)}")

    # 2. Show task counts for key subsets
    for subset in ["all", "train", "eval"]:
        if subset in subsets:
            try:
                task_ids = get_subset_task_ids(dataset_key, subset, auto_download=True)
                print(f"\n'{subset}' subset: {len(task_ids)} tasks")
                if len(task_ids) > 0:
                    print(f"  First 3: {', '.join(task_ids[:3])}")
            except Exception as e:
                print(f"\n'{subset}' subset: Error - {e}")


def demonstrate_single_task() -> None:
    """Show how to select a single task."""
    print(f"\n{'='*70}")
    print("Single Task Selection")
    print(f"{'='*70}")

    # Get all task IDs
    all_tasks = get_subset_task_ids("Mini", "all", auto_download=True)
    print(f"\nTotal tasks in Mini: {len(all_tasks)}")

    # Select first task
    task_id = all_tasks[0]
    print(f"\nSelecting single task: {task_id}")

    # Query what will be loaded
    ids = get_subset_task_ids("Mini", task_id, auto_download=True)
    print(f"  Resolved to: {ids}")
    assert len(ids) == 1, "Single task should resolve to exactly 1 task ID"

    # Create environment with single task
    env, params = make(f"Mini-{task_id}", auto_download=True)
    print(f"  Environment created successfully!")


def demonstrate_custom_subset() -> None:
    """Show how custom subsets work."""
    print(f"\n{'='*70}")
    print("Custom Subset Demo")
    print(f"{'='*70}")

    # Get some tasks to create a custom subset
    all_tasks = get_subset_task_ids("Mini", "all", auto_download=True)

    # Register a custom subset with first 3 tasks
    custom_subset = all_tasks[:3]
    register_subset("Mini", "my-custom-subset", custom_subset)
    print(f"\nRegistered 'my-custom-subset' with {len(custom_subset)} tasks:")
    for task_id in custom_subset:
        print(f"  - {task_id}")

    # Now it appears in available subsets
    subsets = available_named_subsets("Mini")
    print(f"\nAvailable subsets now: {', '.join(subsets)}")
    assert "my-custom-subset" in subsets

    # Query tasks in custom subset
    ids = get_subset_task_ids("Mini", "my-custom-subset", auto_download=True)
    print(f"\nTasks in 'my-custom-subset': {len(ids)}")
    assert ids == custom_subset

    # Create environment with custom subset
    env, params = make("Mini-my-custom-subset", auto_download=True)
    print(f"  Environment created successfully!")


def demonstrate_builtin_vs_custom() -> None:
    """Show the difference between built-in and custom subsets."""
    print(f"\n{'='*70}")
    print("Built-in vs Custom Subsets")
    print(f"{'='*70}")

    # Show all subsets (built-in + custom)
    all_subsets = available_named_subsets("Mini", include_builtin=True)
    print(f"\nAll subsets (built-in + custom): {', '.join(all_subsets)}")

    # Show only custom subsets
    custom_only = available_named_subsets("Mini", include_builtin=False)
    print(f"\nCustom subsets only: {', '.join(custom_only) if custom_only else '(none)'}")


def explore_concept_groups() -> None:
    """Demonstrate ConceptARC concept group selection."""
    print(f"\n{'='*70}")
    print("ConceptARC Concept Groups")
    print(f"{'='*70}")

    try:
        # Get available subsets for ConceptARC
        subsets = available_named_subsets("Concept")
        print(f"\nAvailable subsets: {', '.join(subsets)}")

        # Get all tasks
        all_tasks = get_subset_task_ids("Concept", "all", auto_download=True)
        print(f"\nTotal tasks in ConceptARC: {len(all_tasks)}")

        # Try to find some concept groups by checking task IDs
        # ConceptARC tasks typically have concept group prefixes
        print("\nTesting concept group selection...")

        # Try a known concept group (these are common in ConceptARC)
        test_concepts = ["Center", "AboveBelow", "Horizontal", "Vertical"]
        for concept in test_concepts:
            try:
                concept_tasks = get_subset_task_ids("Concept", concept, auto_download=True)
                print(f"  '{concept}' concept: {len(concept_tasks)} tasks")
                if len(concept_tasks) > 0:
                    print(f"    First task: {concept_tasks[0]}")
                    break  # Found one that works
            except ValueError:
                continue  # This concept doesn't exist

    except Exception as e:
        print(f"\nConceptARC exploration failed: {e}")
        print("(This is expected if ConceptARC dataset is not available)")


def test_single_task_across_datasets() -> None:
    """Test single task selection across different datasets."""
    print(f"\n{'='*70}")
    print("Single Task Selection Across Datasets")
    print(f"{'='*70}")

    datasets = ["Mini", "Concept"]

    for dataset in datasets:
        try:
            # Get all tasks
            all_tasks = get_subset_task_ids(dataset, "all", auto_download=True)
            if not all_tasks:
                print(f"\n{dataset}: No tasks available")
                continue

            # Select first task
            task_id = all_tasks[0]
            print(f"\n{dataset}: Testing with '{task_id}'")

            # Verify single task resolution
            ids = get_subset_task_ids(dataset, task_id, auto_download=True)
            assert len(ids) == 1, f"Expected 1 task, got {len(ids)}"
            assert ids[0] == task_id, f"Task ID mismatch: {ids[0]} != {task_id}"

            # Create environment
            env, params = make(f"{dataset}-{task_id}", auto_download=True)
            print(f"  ✓ Environment created successfully")

        except Exception as e:
            print(f"\n{dataset}: Error - {e}")


def main() -> None:
    """Run all discovery demos."""
    print("\n" + "="*70)
    print("JaxARC Discovery API Demo")
    print("="*70)

    # Explore Mini dataset
    explore_dataset("Mini")

    # Single task selection
    demonstrate_single_task()

    # Custom subsets
    demonstrate_custom_subset()

    # Built-in vs custom
    demonstrate_builtin_vs_custom()

    # ConceptARC exploration
    explore_concept_groups()

    # Test single task across datasets
    test_single_task_across_datasets()

    print(f"\n{'='*70}")
    print("Demo complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
