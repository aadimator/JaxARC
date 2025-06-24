#!/usr/bin/env python
"""
ARCLE Environment Demo Script.

This script demonstrates the functionality of the ARCLE environment,
which implements the ARCLE approach with JAX optimizations.

Usage:
    python -m scripts.demo_arcle_env
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
from omegaconf import DictConfig, OmegaConf

from jaxarc.envs import ARCLEEnvironment
from jaxarc.utils.visualization import log_grid_to_console


def visualize_grids(
    grid, input_grid, target_grid, title="ARCLE Environment State", save_path=None
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


def demo_operations(env, state, key):
    """Demonstrate different ARCLE operations."""
    h, w = state.grid_dim

    # List of operations to demonstrate
    demos = [
        # Fill operations
        {"op": 2, "mask_type": "square", "desc": "Fill square with color 2"},
        {"op": 5, "mask_type": "column", "desc": "Fill column with color 5"},
        # Flood fill
        {"op": 11, "mask_type": "point", "desc": "Flood fill with color 1"},
        # Object operations
        {"op": 20, "mask_type": "square", "desc": "Move object up"},
        {"op": 24, "mask_type": "square", "desc": "Rotate object 90°"},
        {"op": 26, "mask_type": "square", "desc": "Flip object horizontally"},
        # Clipboard operations
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

    output_dir = Path("outputs/arcle_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run demonstrations
    for i, demo in enumerate(demos):
        # Create selection mask
        center = demo.get("center", (h // 2, w // 2))
        mask = create_selection_mask(
            h, w, demo.get("mask_type", "square"), center=center
        )

        # Create action
        action = {
            "agent_0": {
                "selection": jnp.array(mask.astype(np.float32)),
                "operation": jnp.array(demo["op"]),
            }
        }

        # Take action
        key, subkey = jax.random.split(key)
        (obs, new_state, rewards, dones, _) = env.step(subkey, state, action)
        state = new_state

        # Visualize result
        logger.info(f"Operation {i + 1}: {demo['desc']} (op_id={demo['op']})")
        logger.info(f"Reward: {rewards['agent_0']}")
        logger.info(f"Similarity score: {state.similarity_score}")

        # Save visualization
        save_path = output_dir / f"op_{i + 1}_{demo['op']}.png"
        visualize_grids(
            state.grid,
            state.input_grid,
            state.target_grid,
            title=f"Operation {i + 1}: {demo['desc']} (op_id={demo['op']})",
            save_path=save_path,
        )

        # Also log grid to console
        log_grid_to_console(state.grid, title=f"Grid after {demo['desc']}")

        if dones["agent_0"]:
            logger.info("Episode terminated")
            break

    return state


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the ARCLE environment demo."""
    logger.info("Starting ARCLE environment demo")
    logger.info(f"JAX devices: {jax.devices()}")

    # Create a basic ARCLE environment configuration if not available
    if hasattr(cfg, "environment") and hasattr(cfg.environment, "_target_"):
        env_config = OmegaConf.to_container(cfg.environment, resolve=True)
    else:
        # Use default ARCLE config
        env_config = {
            "max_grid_size": [10, 10],
            "max_episode_steps": 20,
            "reward_on_submit_only": True,
        }
    logger.info(f"Environment config: {env_config}")

    # Create environment
    env = ARCLEEnvironment(env_config)
    logger.info(f"Created environment: {env}")

    # Initialize random key
    key = jax.random.PRNGKey(int(time.time()))

    # Reset environment
    logger.info("Resetting environment")
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    # Log initial state
    logger.info(f"Initial state: grid_dim={state.grid_dim}")
    log_grid_to_console(state.grid, title="Initial Grid")
    log_grid_to_console(state.target_grid, title="Target Grid")

    # Create output directory
    output_dir = Path("outputs/arcle_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save initial visualization
    visualize_grids(
        state.grid,
        state.input_grid,
        state.target_grid,
        title="Initial State",
        save_path=output_dir / "initial_state.png",
    )

    # Demo operations
    logger.info("Demonstrating ARCLE operations")
    state = demo_operations(env, state, key)

    # Demonstrate submit operation
    logger.info("Demonstrating submit operation")
    action = {
        "agent_0": {
            "selection": jnp.zeros_like(state.grid, dtype=jnp.float32),
            "operation": jnp.array(34),  # Submit operation
        }
    }

    key, subkey = jax.random.split(key)
    (obs, state, rewards, dones, _) = env.step(subkey, state, action)

    logger.info(f"Submit reward: {rewards['agent_0']}")
    logger.info(f"Final similarity score: {state.similarity_score}")
    logger.info(f"Episode terminated: {dones['agent_0']}")

    # Save final visualization
    visualize_grids(
        state.grid,
        state.input_grid,
        state.target_grid,
        title=f"Final State (Similarity: {state.similarity_score:.4f})",
        save_path=output_dir / "final_state.png",
    )

    logger.info(f"Demo complete. Visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Override to use ARCLE environment
    import sys

    sys.argv.extend(["environment=arcle_env"])
    main()
