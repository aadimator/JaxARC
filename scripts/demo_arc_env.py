#!/usr/bin/env python
"""
ARC Environment Demo Script - New Config-Based API

This script demonstrates the functionality of the new config-based ARC environment,
which implements grid-based operations with enhanced JAX compatibility.

Usage:
    python -m scripts.demo_arc_env
    python -m scripts.demo_arc_env environment=full
    python -m scripts.demo_arc_env environment=raw dataset=mini_arc
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here

from jaxarc.envs import (
    arc_reset,
    arc_step,
    create_config_from_hydra,
    create_standard_config,
    get_config_summary,
    validate_config,
)
from jaxarc.parsers import ArcAgiParser
from jaxarc.utils.visualization import log_grid_to_console


def visualize_grids(
    grid, input_grid, target_grid, title="ARC Environment State", save_path=None
):
    """Visualize the current state of the environment grids."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Convert from JAX arrays to numpy
    grid = np.array(grid)
    input_grid = np.array(input_grid)
    target_grid = np.array(target_grid)

    # Plot grids
    axes[0].imshow(input_grid, cmap="tab10", vmin=0, vmax=9)
    axes[0].set_title("Input Grid")
    axes[0].axis("off")

    axes[1].imshow(grid, cmap="tab10", vmin=0, vmax=9)
    axes[1].set_title("Working Grid")
    axes[1].axis("off")

    axes[2].imshow(target_grid, cmap="tab10", vmin=0, vmax=9)
    axes[2].set_title("Target Grid")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()


def create_selection_mask(h, w, region_type="square", center=None, size=3):
    """Create a demo selection mask for visualization."""
    mask = np.zeros((h, w), dtype=bool)

    if center is None:
        center = (h // 2, w // 2)

    cy, cx = center

    if region_type == "square":
        half_size = size // 2
        y_min = max(0, cy - half_size)
        y_max = min(h, cy + half_size + 1)
        x_min = max(0, cx - half_size)
        x_max = min(w, cx + half_size + 1)
        mask[y_min:y_max, x_min:x_max] = True

    elif region_type == "point":
        mask[cy, cx] = True

    elif region_type == "column":
        x = cx
        mask[:, x] = True

    elif region_type == "row":
        y = cy
        mask[y, :] = True

    return mask


def demo_operations(state, config, key):
    """Demonstrate different ARC operations using the new functional API."""
    h, w = state.working_grid.shape

    # List of operations to demonstrate
    demos = [
        # Fill operations
        {"op": 2, "mask_type": "square", "desc": "Fill square with color 2"},
        {"op": 5, "mask_type": "column", "desc": "Fill column with color 5"},
        # Flood fill (if available in config)
        {"op": 11, "mask_type": "point", "desc": "Flood fill with color 1"},
        # Object operations (if available in config)
        {"op": 20, "mask_type": "square", "desc": "Move object up"},
        {"op": 24, "mask_type": "square", "desc": "Rotate object 90°"},
        {"op": 26, "mask_type": "square", "desc": "Flip object horizontally"},
        # Clipboard operations (if available in config)
        {"op": 29, "mask_type": "square", "desc": "Copy to clipboard"},
        {
            "op": 30,
            "mask_type": "square",
            "center": (2, 2),
            "desc": "Paste from clipboard",
        },
        # Grid operations
        {"op": 31, "mask_type": "square", "desc": "Copy input to output"},
        {"op": 32, "mask_type": "square", "desc": "Reset grid"},
    ]

    output_dir = Path("outputs/arc_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter operations based on available operations in config
    available_ops = config.action.num_operations
    demos = [demo for demo in demos if demo["op"] < available_ops]

    # Run demonstrations
    for i, demo in enumerate(demos):
        # Create selection mask
        center = demo.get("center", (h // 2, w // 2))
        mask = create_selection_mask(
            h, w, demo.get("mask_type", "square"), center=center
        )

        # Create action using new API
        action = {
            "selection": jnp.array(mask, dtype=jnp.bool_),
            "operation": jnp.array(demo["op"], dtype=jnp.int32),
        }

        # Take action using functional API
        key, subkey = jax.random.split(key)
        state, obs, reward, done, info = arc_step(state, action, config)

        # Visualize result
        logger.info(f"Operation {i + 1}: {demo['desc']} (op_id={demo['op']})")
        logger.info(f"Reward: {reward:.4f}")
        logger.info(f"Similarity score: {info['similarity']:.4f}")
        logger.info(f"Step count: {info['step_count']}")

        # Save visualization
        save_path = output_dir / f"op_{i + 1}_{demo['op']}.png"
        visualize_grids(
            state.working_grid,
            state.task_data.input_grids_examples[state.current_example_idx],
            state.task_data.output_grids_examples[state.current_example_idx],
            title=f"Operation {i + 1}: {demo['desc']} (op_id={demo['op']})",
            save_path=save_path,
        )

        # Also log grid to console
        log_grid_to_console(
            state.working_grid,
            mask=state.working_grid_mask,
            title=f"Grid after {demo['desc']}",
            show_numbers=False,
        )

        if done:
            logger.info("Episode terminated")
            break

    return state


def demonstrate_jax_compatibility(config):
    """Demonstrate JAX transformations with the new config-based API."""
    logger.info("=== JAX Compatibility Demo ===")

    # JIT compilation with static config
    @jax.jit
    def jitted_reset(key):
        return arc_reset(key, config)

    @jax.jit
    def jitted_step(state, action):
        return arc_step(state, action, config)

    # Use JIT-compiled functions
    key = jax.random.PRNGKey(789)
    state, obs = jitted_reset(key)
    logger.info(f"JIT reset successful - similarity: {state.similarity_score:.3f}")

    action = {
        "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
        "operation": jnp.array(5, dtype=jnp.int32),
    }

    state, obs, reward, done, info = jitted_step(state, action)
    logger.info(f"JIT step successful - reward: {reward:.3f}")

    # Batch processing with vmap
    def single_episode(key):
        state, obs = arc_reset(key, config)
        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(1, dtype=jnp.int32),
        }
        state, obs, reward, done, info = arc_step(state, action, config)
        return reward

    # Process multiple episodes in parallel
    keys = jax.random.split(key, 5)
    batch_rewards = jax.vmap(single_episode)(keys)
    logger.info(f"Batch processing successful - rewards: {batch_rewards}")


@hydra.main(config_path=str(here("conf")), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the ARC environment demo with new config-based API."""
    logger.info("Starting ARC environment demo with config-based API")
    logger.info(f"JAX devices: {jax.devices()}")

    # Create configuration using new API
    try:
        # Option 1: Create from Hydra config
        config = create_config_from_hydra(cfg)
        logger.info("✅ Created config from Hydra configuration")

        # Validate configuration
        validate_config(config)
        logger.info("✅ Configuration validation passed")

        # Log configuration summary
        config_summary = get_config_summary(config)
        logger.info(f"Configuration summary:\n{config_summary}")

    except Exception as e:
        logger.error(f"❌ Failed to create configuration: {e}")
        # Fallback to standard configuration
        config = create_standard_config()
        logger.info("🔄 Using fallback standard configuration")

    # Create parser if using dataset
    parser = None
    if hasattr(cfg, "dataset") and cfg.dataset:
        try:
            parser = ArcAgiParser(cfg.dataset)
            logger.info(f"✅ Created parser with dataset: {cfg.dataset}")
        except Exception as e:
            logger.warning(f"⚠️ Could not create parser: {e}")
            logger.info("Using demo task instead")

    # Initialize random key
    key = jax.random.PRNGKey(int(time.time()))

    # Reset environment using new functional API
    logger.info("Resetting environment with new functional API")
    key, reset_key = jax.random.split(key)

    try:
        if parser:
            state, obs = arc_reset(reset_key, config, parser=parser)
        else:
            state, obs = arc_reset(reset_key, config)
        logger.info("✅ Environment reset successful")
    except Exception as e:
        logger.error(f"❌ Reset failed: {e}")
        return

    # Log initial state
    logger.info(f"Initial state: grid_shape={state.working_grid.shape}")
    logger.info(f"Action format: {config.action.action_format}")
    logger.info(f"Available operations: {config.action.num_operations}")

    log_grid_to_console(
        state.working_grid,
        mask=state.working_grid_mask,
        title="Initial Grid",
        show_numbers=False,
    )
    log_grid_to_console(
        state.task_data.output_grids_examples[state.current_example_idx],
        mask=state.task_data.output_masks_examples[state.current_example_idx],
        title="Target Grid",
        show_numbers=False,
    )

    # Create output directory
    output_dir = Path(cfg.get("output_dir", "outputs/arc_demo"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save initial visualization
    visualize_grids(
        state.working_grid,
        state.task_data.input_grids_examples[state.current_example_idx],
        state.task_data.output_grids_examples[state.current_example_idx],
        title="Initial State",
        save_path=output_dir / "initial_state.png",
    )

    # Demo operations
    logger.info("Demonstrating ARC operations with new API")
    try:
        state = demo_operations(state, config, key)
        logger.info("✅ Operations demo completed successfully")
    except Exception as e:
        logger.error(f"❌ Operations demo failed: {e}")
        return

    # Demonstrate JAX compatibility
    try:
        demonstrate_jax_compatibility(config)
        logger.info("✅ JAX compatibility demo completed successfully")
    except Exception as e:
        logger.error(f"❌ JAX compatibility demo failed: {e}")

    # Demonstrate submit operation
    logger.info("Demonstrating submit operation")
    try:
        action = {
            "selection": jnp.zeros_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(34, dtype=jnp.int32),  # Submit operation
        }

        key, subkey = jax.random.split(key)
        state, obs, reward, done, info = arc_step(state, action, config)

        logger.info(f"Submit reward: {reward:.4f}")
        logger.info(f"Final similarity score: {info['similarity']:.4f}")
        logger.info(f"Episode terminated: {done}")
        logger.info(f"Total steps: {info['step_count']}")

        # Save final visualization
        visualize_grids(
            state.working_grid,
            state.task_data.input_grids_examples[state.current_example_idx],
            state.task_data.output_grids_examples[state.current_example_idx],
            title=f"Final State (Similarity: {info['similarity']:.4f})",
            save_path=output_dir / "final_state.png",
        )

        logger.info(f"✅ Demo complete! Visualizations saved to {output_dir}")

    except Exception as e:
        logger.error(f"❌ Submit operation failed: {e}")


if __name__ == "__main__":
    main()
