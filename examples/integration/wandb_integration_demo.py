#!/usr/bin/env python3
"""Demonstration of Weights & Biases integration for JaxARC.

This example shows how to use the wandb integration for experiment tracking,
including configuration, logging, and error handling.

Usage:
    pixi run python examples/wandb_integration_demo.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxarc.utils.visualization import (
    WandbConfig,
    WandbIntegration,
    create_development_wandb_config,
    create_research_wandb_config,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_wandb_config() -> None:
    """Demonstrate basic wandb configuration."""
    print("=== Basic Wandb Configuration Demo ===")

    # Create a basic config (disabled by default)
    basic_config = WandbConfig()
    print(f"Basic config enabled: {basic_config.enabled}")
    print(f"Project name: {basic_config.project_name}")
    print(f"Log frequency: {basic_config.log_frequency}")
    print()

    # Create an enabled config
    enabled_config = WandbConfig(
        enabled=True,
        project_name="jaxarc-demo",
        tags=["demo", "test"],
        log_frequency=5,
        offline_mode=True,  # Use offline mode for demo
    )
    print(f"Enabled config: {enabled_config.enabled}")
    print(f"Tags: {enabled_config.tags}")
    print(f"Offline mode: {enabled_config.offline_mode}")
    print()


def demo_config_factories() -> None:
    """Demonstrate configuration factory functions."""
    print("=== Configuration Factory Demo ===")

    # Research configuration
    research_config = create_research_wandb_config(
        project_name="arc-research", entity="research-team"
    )
    print(f"Research config - Log frequency: {research_config.log_frequency}")
    print(f"Research config - Image format: {research_config.image_format}")
    print(f"Research config - Tags: {research_config.tags}")
    print()

    # Development configuration
    dev_config = create_development_wandb_config(project_name="arc-dev")
    print(f"Dev config - Offline mode: {dev_config.offline_mode}")
    print(f"Dev config - Save code: {dev_config.save_code}")
    print(f"Dev config - Tags: {dev_config.tags}")
    print()


def demo_wandb_integration_disabled() -> None:
    """Demonstrate wandb integration when disabled."""
    print("=== Wandb Integration (Disabled) Demo ===")

    config = WandbConfig(enabled=False)
    integration = WandbIntegration(config)

    print(f"Wandb available: {integration.is_available}")
    print(f"Run initialized: {integration.is_initialized}")

    # Try to initialize run (should fail gracefully)
    result = integration.initialize_run({"demo": "config"})
    print(f"Run initialization result: {result}")

    # Try to log step (should fail gracefully)
    result = integration.log_step(1, {"reward": 0.5})
    print(f"Step logging result: {result}")
    print()


def demo_wandb_integration_offline() -> None:
    """Demonstrate wandb integration in offline mode."""
    print("=== Wandb Integration (Offline) Demo ===")

    config = WandbConfig(
        enabled=True,
        project_name="jaxarc-offline-demo",
        offline_mode=True,
        log_frequency=2,
    )

    integration = WandbIntegration(config)

    print(f"Wandb available: {integration.is_available}")

    if integration.is_available:
        # Initialize run
        experiment_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "environment": "arc",
            "algorithm": "ppo",
        }

        result = integration.initialize_run(experiment_config, run_name="offline-demo")
        print(f"Run initialization result: {result}")

        if result:
            print(f"Run ID: {integration.run_id}")
            print(f"Run name: {integration.run_name}")

            # Simulate training steps
            for step in range(10):
                metrics = {
                    "reward": float(
                        jnp.sin(step * 0.5)
                        + jax.random.normal(jax.random.PRNGKey(step))
                    ),
                    "loss": float(
                        1.0 / (step + 1)
                        + 0.1 * jax.random.normal(jax.random.PRNGKey(step + 100))
                    ),
                    "episode_length": step + 5,
                }

                # Log step (respects frequency)
                result = integration.log_step(step, metrics)
                if result and step % config.log_frequency == 0:
                    print(f"Logged step {step}: reward={metrics['reward']:.3f}")

                time.sleep(0.1)  # Simulate computation time

            # Log episode summary
            episode_summary = {
                "total_reward": 15.5,
                "episode_length": 25,
                "success": True,
                "final_similarity": 0.85,
            }

            result = integration.log_episode_summary(1, episode_summary)
            print(f"Episode summary logged: {result}")

            # Update config during run
            config_update = {"exploration_rate": 0.1}
            result = integration.log_config_update(config_update)
            print(f"Config update logged: {result}")

            # Finish run
            integration.finish_run()
            print("Run finished successfully")
    else:
        print("Wandb not available - install with 'pip install wandb' to enable")

    print()


def demo_error_handling() -> None:
    """Demonstrate error handling in wandb integration."""
    print("=== Error Handling Demo ===")

    # Test invalid configuration
    try:
        invalid_config = WandbConfig(image_format="invalid")
    except ValueError as e:
        print(f"Caught expected validation error: {e}")

    try:
        invalid_config = WandbConfig(log_frequency=-1)
    except ValueError as e:
        print(f"Caught expected validation error: {e}")

    # Test graceful fallback when wandb unavailable
    config = WandbConfig(enabled=True)
    integration = WandbIntegration(config)

    # These should all return False gracefully if wandb is not available
    print(
        f"Initialize run (no wandb): {integration.initialize_run({'test': 'config'})}"
    )
    print(f"Log step (no wandb): {integration.log_step(1, {'reward': 0.5})}")
    print(
        f"Log episode (no wandb): {integration.log_episode_summary(1, {'reward': 10})}"
    )
    print(
        f"Log config update (no wandb): {integration.log_config_update({'lr': 0.01})}"
    )

    # Finish run should not crash
    integration.finish_run()
    print("Error handling completed successfully")
    print()


def demo_image_processing() -> None:
    """Demonstrate image processing capabilities."""
    print("=== Image Processing Demo ===")

    config = WandbConfig(enabled=True, offline_mode=True, image_format="both")

    integration = WandbIntegration(config)

    if integration.is_available:
        # Create a simple test image (numpy array)
        test_image = jnp.ones((64, 64, 3)) * 0.5  # Gray image

        # Test image processing (this would normally be called internally)
        processed = integration._process_single_image(test_image, "test_grid")
        print(f"Image processing result: {processed is not None}")

        # Test with file path
        test_path = Path("test_image.png")
        processed_path = integration._process_single_image(test_path, "test_file")
        print(f"Path processing result: {processed_path is not None}")

    print()


def main() -> None:
    """Run all wandb integration demos."""
    print("JaxARC Weights & Biases Integration Demo")
    print("=" * 50)
    print()

    demo_basic_wandb_config()
    demo_config_factories()
    demo_wandb_integration_disabled()
    demo_wandb_integration_offline()
    demo_error_handling()
    demo_image_processing()

    print("Demo completed successfully!")
    print()
    print("To enable full wandb functionality:")
    print("1. Install wandb: pip install wandb")
    print("2. Login: wandb login")
    print("3. Set enabled=True and offline_mode=False in config")


if __name__ == "__main__":
    main()
