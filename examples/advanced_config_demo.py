#!/usr/bin/env python3
"""
Advanced Config-Based API Demo for JaxARC

This example demonstrates advanced features including:
- Task sampling with real datasets
- Dataset-aware configurations
- Action restrictions for different environments
- Task filtering and customization
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import OmegaConf

from jaxarc.envs import (
    ArcEnvConfig,
    arc_reset,
    arc_step,
    attach_sampler_to_config,
    auto_create_sampler_for_config,
    # Task sampling
    create_arc_agi_sampler,
    create_complexity_filtered_sampler,
    create_dataset_config,
    # Factory functions
    create_raw_config,
    create_size_filtered_sampler,
    create_static_sampler,
)


def demo_dataset_aware_configs():
    """Demonstrate dataset-specific configurations."""
    logger.info("=== Dataset-Aware Configurations Demo ===")

    # Different datasets with different constraints
    datasets = ["arc-agi-1", "concept-arc", "mini-arc"]

    for dataset_name in datasets:
        logger.info(f"\nConfiguring for {dataset_name}:")

        try:
            config = create_dataset_config(
                dataset_name=dataset_name,
                task_split="train",
                max_episode_steps=50,
            )

            logger.info(
                f"  Grid size: {config.grid.max_grid_height}x{config.grid.max_grid_width}"
            )
            logger.info(f"  Max colors: {config.grid.max_colors}")
            logger.info(f"  Action format: {config.action.action_format}")
            logger.info(f"  Dataset config: {config.dataset.dataset_name}")

            # Show how dataset overrides work
            if config.dataset.dataset_max_grid_height:
                logger.info(
                    f"  Dataset override: max_height={config.dataset.dataset_max_grid_height}"
                )

        except Exception as e:
            logger.warning(f"  Could not create config for {dataset_name}: {e}")


def demo_action_restrictions():
    """Demonstrate action restrictions for different environments."""
    logger.info("\n=== Action Restrictions Demo ===")

    # Raw config: Only fill colors (0-9), resize (33), and submit (34)
    logger.info("1. Raw Config (restricted actions):")
    raw_config = create_raw_config(
        max_episode_steps=30,
        dataset_name="mini-arc",  # Use mini-arc for faster demo
    )

    logger.info(f"  Allowed operations: {raw_config.action.allowed_operations}")
    logger.info(f"  Total operations: {raw_config.action.num_operations}")

    # Test that invalid operations are handled
    key = jax.random.PRNGKey(42)
    state, obs = arc_reset(key, raw_config)

    # Try an action with a restricted operation (e.g., move up = 20)
    invalid_action = {
        "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
        "operation": jnp.array(20, dtype=jnp.int32),  # Move up - not allowed in raw
    }

    try:
        new_state, obs, reward, done, info = arc_step(state, invalid_action, raw_config)
        logger.info("  Invalid operation handled (clipped to valid range)")
    except Exception as e:
        logger.info(f"  Invalid operation rejected: {e}")

    # Valid action
    valid_action = {
        "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
        "operation": jnp.array(1, dtype=jnp.int32),  # Fill with color 1 - allowed
    }

    new_state, obs, reward, done, info = arc_step(state, valid_action, raw_config)
    logger.info(f"  Valid fill operation successful, reward: {reward:.3f}")


def demo_task_sampling_basic():
    """Demonstrate basic task sampling without real parser."""
    logger.info("\n=== Basic Task Sampling Demo ===")

    # Create a demo task for static sampling

    demo_task = _create_simple_demo_task()

    # Static sampler (always returns the same task)
    logger.info("1. Static Task Sampler:")
    static_sampler = create_static_sampler(demo_task)

    config = create_dataset_config("mini-arc")
    config_with_sampler = attach_sampler_to_config(config, static_sampler)

    key = jax.random.PRNGKey(123)
    state, obs = arc_reset(key, config_with_sampler)

    logger.info(f"  Task loaded successfully, grid shape: {obs.shape}")
    logger.info(f"  Initial similarity: {state.similarity_score:.3f}")


def demo_task_sampling_with_parser():
    """Demonstrate task sampling with real ARC parser (if available)."""
    logger.info("\n=== Task Sampling with Parser Demo ===")

    try:
        # Try to create a real ARC-AGI sampler
        sampler = create_arc_agi_sampler(lazy_init=True)

        # Create configuration for ARC-AGI dataset
        config = create_dataset_config(
            dataset_name="arc-agi-1",
            task_split="train",
            max_episode_steps=50,
        )

        # Attach sampler
        config_with_sampler = attach_sampler_to_config(config, sampler)

        # Alternative: use auto-create function
        auto_config = auto_create_sampler_for_config(config)

        logger.info("  ARC-AGI parser sampler created successfully")
        logger.info(f"  Dataset: {config.dataset.dataset_name}")
        logger.info(f"  Split: {config.dataset.task_split}")

        # Try sampling
        key = jax.random.PRNGKey(456)
        try:
            state, obs = arc_reset(key, config_with_sampler)
            logger.info("  Real task sampled successfully!")
            logger.info(f"  Grid shape: {obs.shape}")
            logger.info(f"  Similarity: {state.similarity_score:.3f}")
            logger.info(
                f"  Task has {state.task_data.num_train_pairs} training examples"
            )

        except Exception as e:
            logger.warning(f"  Task sampling failed (likely no data available): {e}")
            logger.info("  This is expected if ARC dataset is not downloaded")

    except Exception as e:
        logger.warning(f"  Could not create ARC-AGI sampler: {e}")
        logger.info("  This is expected if parsers are not available")


def demo_task_filtering():
    """Demonstrate task filtering capabilities."""
    logger.info("\n=== Task Filtering Demo ===")

    # Create base sampler (we'll use static for demo)
    demo_task = _create_simple_demo_task()
    base_sampler = create_static_sampler(demo_task)

    # Size filtering
    logger.info("1. Size-filtered sampler:")
    size_filtered = create_size_filtered_sampler(
        base_sampler,
        max_height=10,
        max_width=10,
    )

    config = create_dataset_config("mini-arc")
    config_with_filter = attach_sampler_to_config(config, size_filtered)

    key = jax.random.PRNGKey(789)
    try:
        state, obs = arc_reset(key, config_with_filter)
        logger.info(f"  Size filtering successful, grid shape: {obs.shape}")
    except Exception as e:
        logger.warning(f"  Size filtering failed: {e}")

    # Complexity filtering
    logger.info("2. Complexity-filtered sampler:")
    complexity_filtered = create_complexity_filtered_sampler(
        base_sampler,
        max_colors=5,
        min_examples=1,
        max_examples=3,
    )

    config_with_complexity = attach_sampler_to_config(config, complexity_filtered)

    try:
        state, obs = arc_reset(key, config_with_complexity)
        logger.info("  Complexity filtering successful")
        logger.info(f"  Training examples: {state.task_data.num_train_pairs}")
    except Exception as e:
        logger.warning(f"  Complexity filtering failed: {e}")


def demo_hydra_integration_advanced():
    """Demonstrate advanced Hydra integration with datasets and samplers."""
    logger.info("\n=== Advanced Hydra Integration Demo ===")

    # Create comprehensive Hydra config
    hydra_config = OmegaConf.create(
        {
            "max_episode_steps": 100,
            "log_operations": True,
            "reward": {
                "reward_on_submit_only": True,
                "success_bonus": 15.0,
            },
            "action": {
                "action_format": "point",
                "allowed_operations": [0, 1, 2, 3, 4, 34],  # Limited operations
            },
            "dataset": {
                "dataset_name": "mini-arc",
                "task_split": "train",
                "dataset_max_grid_height": 8,
                "dataset_max_grid_width": 8,
                "shuffle_tasks": True,
            },
        }
    )

    # Convert to typed config
    config = ArcEnvConfig.from_hydra(hydra_config)

    logger.info("  Hydra config converted successfully")
    logger.info(f"  Dataset: {config.dataset.dataset_name}")
    logger.info(
        f"  Grid size override: {config.grid.max_grid_height}x{config.grid.max_grid_width}"
    )
    logger.info(f"  Action restrictions: {config.action.allowed_operations}")

    # Auto-attach appropriate sampler
    config_with_sampler = auto_create_sampler_for_config(config)

    # Use the configuration
    key = jax.random.PRNGKey(999)
    try:
        state, obs = arc_reset(key, config_with_sampler)
        logger.info("  Environment initialized with Hydra config")
        logger.info(f"  Actual grid shape: {obs.shape}")
    except Exception as e:
        logger.warning(f"  Environment initialization failed: {e}")


def demo_curriculum_learning():
    """Demonstrate curriculum learning with progressive difficulty."""
    logger.info("\n=== Curriculum Learning Demo ===")

    curriculum_stages = [
        ("easy", {"dataset_name": "mini-arc", "max_colors": 3}),
        ("medium", {"dataset_name": "concept-arc", "max_colors": 6}),
        ("hard", {"dataset_name": "arc-agi-1", "max_colors": 10}),
    ]

    for stage_name, params in curriculum_stages:
        logger.info(f"\n{stage_name.capitalize()} Stage:")

        # Create base config for this stage
        config = create_dataset_config(
            dataset_name=params["dataset_name"],
            max_episode_steps=30,
        )

        # Add auto sampler
        config = auto_create_sampler_for_config(config)

        logger.info(f"  Dataset: {config.dataset.dataset_name}")
        logger.info(
            f"  Max grid size: {config.grid.max_grid_height}x{config.grid.max_grid_width}"
        )
        logger.info(f"  Action format: {config.action.action_format}")

        # Test the configuration
        key = jax.random.PRNGKey(100 + hash(stage_name) % 900)
        try:
            state, obs = arc_reset(key, config)
            logger.info("  ✓ Stage configured successfully")
        except Exception as e:
            logger.warning(f"  ✗ Stage configuration failed: {e}")


def demo_performance_comparison():
    """Compare performance between different configurations."""
    logger.info("\n=== Performance Comparison Demo ===")

    configs = {
        "raw": create_raw_config(max_episode_steps=10),
        "mini-arc": create_dataset_config("mini-arc", max_episode_steps=10),
        "concept-arc": create_dataset_config("concept-arc", max_episode_steps=10),
    }

    # Add samplers
    for name, config in configs.items():
        configs[name] = auto_create_sampler_for_config(config)

    # Test reset/step performance
    key = jax.random.PRNGKey(42)

    for name, config in configs.items():
        logger.info(f"\nTesting {name} config:")

        try:
            # Time reset
            import time

            start = time.time()

            state, obs = arc_reset(key, config)

            reset_time = time.time() - start

            # Time step
            action = {
                "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
                "operation": jnp.array(1, dtype=jnp.int32),
            }

            start = time.time()
            new_state, obs, reward, done, info = arc_step(state, action, config)
            step_time = time.time() - start

            logger.info(f"  Reset time: {reset_time * 1000:.2f}ms")
            logger.info(f"  Step time: {step_time * 1000:.2f}ms")
            logger.info(f"  Grid shape: {obs.shape}")
            logger.info(
                f"  Allowed ops: {len(config.action.allowed_operations) if config.action.allowed_operations else 'all'}"
            )

        except Exception as e:
            logger.warning(f"  Failed: {e}")


def _create_simple_demo_task():
    """Create a simple demo task for examples."""
    from jaxarc.types import JaxArcTask

    # Create 5x5 demo grids
    grid_shape = (5, 5)

    # Input: blue square in center
    input_grid = jnp.zeros(grid_shape, dtype=jnp.int32)
    input_grid = input_grid.at[2, 2].set(1)  # Blue center

    # Output: red square in center
    target_grid = jnp.zeros(grid_shape, dtype=jnp.int32)
    target_grid = target_grid.at[2, 2].set(2)  # Red center

    # Masks (all valid)
    mask = jnp.ones(grid_shape, dtype=jnp.bool_)

    # Pad to standard size (30x30)
    max_shape = (30, 30)
    padded_input = jnp.full(max_shape, -1, dtype=jnp.int32)
    padded_target = jnp.full(max_shape, -1, dtype=jnp.int32)
    padded_mask = jnp.zeros(max_shape, dtype=jnp.bool_)

    padded_input = padded_input.at[: grid_shape[0], : grid_shape[1]].set(input_grid)
    padded_target = padded_target.at[: grid_shape[0], : grid_shape[1]].set(target_grid)
    padded_mask = padded_mask.at[: grid_shape[0], : grid_shape[1]].set(mask)

    return JaxArcTask(
        input_grids_examples=jnp.expand_dims(padded_input, 0),
        output_grids_examples=jnp.expand_dims(padded_target, 0),
        input_masks_examples=jnp.expand_dims(padded_mask, 0),
        output_masks_examples=jnp.expand_dims(padded_mask, 0),
        num_train_pairs=1,
        test_input_grids=jnp.expand_dims(padded_input, 0),
        test_input_masks=jnp.expand_dims(padded_mask, 0),
        true_test_output_grids=jnp.expand_dims(padded_target, 0),
        true_test_output_masks=jnp.expand_dims(padded_mask, 0),
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )


def main():
    """Run all advanced demos."""
    logger.info("Starting JaxARC Advanced Config Demo")
    logger.info("=====================================")

    try:
        demo_dataset_aware_configs()
        demo_action_restrictions()
        demo_task_sampling_basic()
        demo_task_sampling_with_parser()
        demo_task_filtering()
        demo_hydra_integration_advanced()
        demo_curriculum_learning()
        demo_performance_comparison()

        logger.info("\n🎉 All advanced demos completed successfully!")
        logger.info("\nKey takeaways:")
        logger.info(
            "- Dataset configs automatically adjust grid sizes and action formats"
        )
        logger.info(
            "- Action restrictions enable curriculum learning (raw → standard → full)"
        )
        logger.info("- Task samplers integrate seamlessly with existing parsers")
        logger.info("- Filtering enables fine-grained control over task difficulty")
        logger.info("- Hydra integration supports complex multi-dataset configurations")

    except Exception as e:
        logger.error(f"Advanced demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
