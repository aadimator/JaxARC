#!/usr/bin/env python3
"""
Enhanced Wandb Integration Demo

This example demonstrates the enhanced wandb logging methods implemented in task 4.2:
- Image optimization for wandb upload
- Automatic experiment tagging and organization
- Enhanced step and episode logging
"""

from __future__ import annotations

import numpy as np

from jaxarc.utils.visualization.wandb_integration import (
    WandbConfig,
    WandbIntegration,
    create_development_wandb_config,
    create_research_wandb_config,
)


def create_sample_image() -> np.ndarray:
    """Create a sample image for demonstration."""
    # Create a simple gradient image
    height, width = 200, 300
    image = np.zeros((height, width, 3))

    # Create a gradient
    for i in range(height):
        for j in range(width):
            image[i, j, 0] = i / height  # Red gradient
            image[i, j, 1] = j / width  # Green gradient
            image[i, j, 2] = 0.5  # Blue constant

    return image


def demo_image_optimization():
    """Demonstrate image optimization features."""
    print("=== Image Optimization Demo ===")

    # Create config with specific image optimization settings
    config = WandbConfig(
        enabled=True,
        project_name="jaxarc-image-optimization-demo",
        max_image_size=(400, 300),  # Smaller than our sample image
        image_format="png",
        offline_mode=True,  # For demo purposes
    )

    integration = WandbIntegration(config)

    if not integration.is_available:
        print("Wandb not available - skipping image optimization demo")
        return

    # Create sample experiment config
    experiment_config = {
        "dataset": {"name": "arc_agi_1", "split": "train"},
        "action": {"format": "mask"},
        "debug": {"level": "standard"},
    }

    # Initialize run (this will demonstrate automatic tagging)
    success = integration.initialize_run(
        experiment_config, run_name="image-optimization-test"
    )

    if success:
        print(f"✓ Wandb run initialized: {integration.run_name}")
        print(f"  Run ID: {integration.run_id}")
        print(f"  Run URL: {integration.run_url}")

        # Create sample image
        sample_image = create_sample_image()
        print(f"✓ Created sample image: {sample_image.shape}")

        # Log step with image (will be optimized)
        step_metrics = {"reward": 0.75, "similarity": 0.85, "step_time": 0.1}

        images = {
            "grid_visualization": sample_image,
            "action_mask": sample_image[:, :, 0],  # Grayscale version
        }

        success = integration.log_step(
            step_num=1, metrics=step_metrics, images=images, force_log=True
        )

        if success:
            print("✓ Step logged with optimized images")
        else:
            print("✗ Failed to log step")

        # Log episode summary
        episode_summary = {
            "total_reward": 15.5,
            "total_steps": 25,
            "final_similarity": 0.92,
            "success": True,
        }

        success = integration.log_episode_summary(
            episode_num=1, summary_data=episode_summary, summary_image=sample_image
        )

        if success:
            print("✓ Episode summary logged")
        else:
            print("✗ Failed to log episode summary")

        # Finish run
        integration.finish_run()
        print("✓ Wandb run finished")
    else:
        print("✗ Failed to initialize wandb run")


def demo_experiment_tagging():
    """Demonstrate automatic experiment tagging and organization."""
    print("\n=== Experiment Tagging Demo ===")

    # Create different experiment configurations to show tagging
    experiment_configs = [
        {
            "name": "Training Run",
            "config": {
                "dataset": {"name": "arc_agi_1", "split": "train"},
                "algorithm": {"name": "ppo", "lr": 0.001},
                "action": {"format": "mask"},
                "debug": {"level": "minimal"},
            },
        },
        {
            "name": "Debug Run",
            "config": {
                "dataset": {"name": "mini_arc", "split": "test"},
                "action": {"format": "point"},
                "debug": {"level": "verbose"},
                "visualization": {"debug_level": "full", "wandb": {"enabled": True}},
            },
        },
        {
            "name": "Evaluation Run",
            "config": {
                "dataset": {"name": "arc_agi_2", "split": "eval"},
                "algorithm": {"name": "dqn"},
                "evaluation": {"mode": "test"},
                "debug": {"level": "off"},
            },
        },
    ]

    config = WandbConfig(
        enabled=True, project_name="jaxarc-tagging-demo", offline_mode=True
    )

    for exp in experiment_configs:
        print(f"\n--- {exp['name']} ---")

        integration = WandbIntegration(config)

        if integration.is_available:
            # Show what tags and organization would be generated
            tags = integration._generate_experiment_tags(exp["config"])
            group = integration._generate_experiment_group(exp["config"])
            job_type = integration._generate_job_type(exp["config"])
            run_name = integration._generate_run_name(exp["config"], None)

            print(f"Generated tags: {tags}")
            print(f"Generated group: {group}")
            print(f"Generated job type: {job_type}")
            print(f"Generated run name: {run_name}")
        else:
            print("Wandb not available - showing what would be generated")


def demo_config_factories():
    """Demonstrate the different wandb config factory functions."""
    print("\n=== Config Factory Demo ===")

    # Research config
    research_config = create_research_wandb_config(
        project_name="jaxarc-research-project", entity="my-research-team"
    )
    print("Research Config:")
    print(f"  Enabled: {research_config.enabled}")
    print(f"  Log frequency: {research_config.log_frequency}")
    print(f"  Image format: {research_config.image_format}")
    print(f"  Tags: {research_config.tags}")
    print(f"  Save code: {research_config.save_code}")

    # Development config
    dev_config = create_development_wandb_config(project_name="jaxarc-dev-project")
    print("\nDevelopment Config:")
    print(f"  Enabled: {dev_config.enabled}")
    print(f"  Log frequency: {dev_config.log_frequency}")
    print(f"  Image format: {dev_config.image_format}")
    print(f"  Tags: {dev_config.tags}")
    print(f"  Offline mode: {dev_config.offline_mode}")
    print(f"  Save code: {dev_config.save_code}")


def main():
    """Run all demos."""
    print("Enhanced Wandb Integration Demo")
    print("=" * 40)

    demo_image_optimization()
    demo_experiment_tagging()
    demo_config_factories()

    print("\n" + "=" * 40)
    print("Demo completed!")
    print("\nKey features demonstrated:")
    print("✓ Image optimization with resizing and format conversion")
    print("✓ Automatic experiment tagging based on configuration")
    print("✓ Intelligent run organization with groups and job types")
    print("✓ Enhanced step and episode logging")
    print("✓ Research and development config presets")


if __name__ == "__main__":
    main()
