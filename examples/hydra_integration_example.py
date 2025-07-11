#!/usr/bin/env python3
"""
Example demonstrating Hydra integration with JaxARC functional API.

This example shows how to use the existing Hydra configuration system
with the ArcAgiParser and functional API, leveraging all the existing
infrastructure instead of reinventing it.

Run with:
    python examples/hydra_integration_example.py
    python examples/hydra_integration_example.py dataset=arc_agi_1
    python examples/hydra_integration_example.py environment=training
    python examples/hydra_integration_example.py environment=raw
    python examples/hydra_integration_example.py environment=full
    python examples/hydra_integration_example.py environment.max_episode_steps=50
"""

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
from loguru import logger

from jaxarc.envs.factory import create_complete_hydra_config
from jaxarc.envs.functional import arc_reset, arc_step
from jaxarc.visualization import log_grid_to_console


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function demonstrating Hydra integration.

    Args:
        cfg: Hydra configuration loaded from conf/ directory
    """
    logger.info("Starting JaxARC Hydra integration example")
    logger.info(f"Using dataset: {cfg.dataset.dataset_name}")
    logger.info(f"Dataset grid config: {cfg.dataset.grid}")
    logger.info(f"Environment config: {cfg.environment}")

    # Create environment configuration using existing Hydra infrastructure
    logger.info("Creating environment configuration from Hydra...")
    env_config = create_complete_hydra_config(cfg)

    # Initialize JAX random key
    key = jax.random.PRNGKey(cfg.seed)

    # Reset environment - this will use the ArcAgiParser if available
    logger.info("Resetting environment...")
    key, reset_key = jax.random.split(key)
    state, obs = arc_reset(reset_key, env_config)

    logger.info(f"Environment reset successfully!")
    logger.info(f"Task has {state.task_data.num_train_pairs} training pairs")
    logger.info(f"Task has {state.task_data.num_test_pairs} test pairs")
    logger.info(f"Initial similarity score: {state.similarity_score:.3f}")

    # Show initial grid
    logger.info("Initial working grid:")
    log_grid_to_console(state.working_grid, state.working_grid_mask)

    logger.info("Target grid:")
    log_grid_to_console(state.target_grid, state.working_grid_mask)

    # Take a few steps to demonstrate the functional API
    logger.info("Taking sample steps...")

    for step in range(3):
        # Create a simple action (fill a small area with a random color)
        key, action_key = jax.random.split(key)

        # Create a selection mask (select a 3x3 area)
        selection = jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
        selection = selection.at[1:4, 1:4].set(True)

        # Random operation (fill with random color)
        operation = jax.random.randint(action_key, (), 0, 10)

        action = {
            "selection": selection,
            "operation": operation,
        }

        # Take step
        key, step_key = jax.random.split(key)
        state, obs, reward, done, info = arc_step(state, action, env_config)

        logger.info(f"Step {step + 1}:")
        logger.info(f"  Reward: {reward:.3f}")
        logger.info(f"  Done: {done}")
        logger.info(f"  Similarity: {info['similarity']:.3f}")
        logger.info(f"  Step count: {state.step_count}")

        if done:
            logger.info("Episode finished!")
            break

    # Show final grid
    logger.info("Final working grid:")
    log_grid_to_console(state.working_grid, state.working_grid_mask)

    logger.info("Example completed successfully!")


def test_different_configurations():
    """
    Test different configuration scenarios without using Hydra decorator.
    This function demonstrates programmatic usage.
    """
    from omegaconf import OmegaConf
    from jaxarc.envs.factory import create_config_from_hydra

    logger.info("Testing different configuration scenarios...")

    # Scenario 1: Basic configuration with demo task
    basic_config = OmegaConf.create({
        "environment": {
            "max_episode_steps": 20,
            "log_operations": True,
            "reward": {"success_bonus": 5.0},
            "grid": {"max_grid_height": 10, "max_grid_width": 10},
        }
    })

    env_config = create_config_from_hydra(basic_config.environment)
    key = jax.random.PRNGKey(123)
    state, obs = arc_reset(key, env_config)

    logger.info(f"Basic config test - Initial similarity: {state.similarity_score:.3f}")

    # Scenario 2: Configuration with different action format
    point_config = OmegaConf.create({
        "environment": {
            "max_episode_steps": 15,
            "action": {"action_format": "point"},
            "reward": {"step_penalty": -0.05},
        }
    })

    env_config = create_config_from_hydra(point_config.environment)
    key = jax.random.PRNGKey(456)
    state, obs = arc_reset(key, env_config)

    logger.info(f"Point config test - Grid shape: {state.working_grid.shape}")

    # Test point action
    action = {
        "point": (2, 3),
        "operation": jnp.array(7, dtype=jnp.int32),
    }

    state, obs, reward, done, info = arc_step(state, action, env_config)
    logger.info(f"Point action test - Reward: {reward:.3f}")

    logger.info("Configuration tests completed!")


if __name__ == "__main__":
    # Run the main Hydra example
    main()

    # Optionally run additional tests
    # test_different_configurations()
