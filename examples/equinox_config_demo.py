#!/usr/bin/env python3
"""
Demo of the new Equinox-based configuration system.

This example demonstrates how to use the new unified configuration classes
with Equinox and JaxTyping for type safety and JAX compatibility.
"""

from omegaconf import OmegaConf

from jaxarc.envs.equinox_config import (
    DatasetConfig,
    EnvironmentConfig,
    LoggingConfig,
    StorageConfig,
    VisualizationConfig,
    WandbConfig,
)


def demo_environment_config():
    """Demonstrate EnvironmentConfig usage."""
    print("=== EnvironmentConfig Demo ===")
    print("Core environment behavior and runtime settings only")

    # Create with defaults
    config = EnvironmentConfig()
    print(
        f"Default config: max_episode_steps={config.max_episode_steps}, debug_level={config.debug_level}"
    )

    # Create with custom values
    custom_config = EnvironmentConfig(
        max_episode_steps=200, strict_validation=False, debug_level="verbose"
    )
    print(
        f"Custom config: max_episode_steps={custom_config.max_episode_steps}, debug_level={custom_config.debug_level}"
    )

    # Show computed properties
    print(f"Computed viz level: {custom_config.computed_visualization_level}")
    print(f"Computed storage policy: {custom_config.computed_storage_policy}")

    # Validate configuration
    errors = custom_config.validate()
    print(f"Validation errors: {errors}")

    # Create from Hydra config
    hydra_cfg = OmegaConf.create(
        {"max_episode_steps": 150, "debug_level": "research", "allow_invalid_actions": True}
    )
    hydra_config = EnvironmentConfig.from_hydra(hydra_cfg)
    print(
        f"From Hydra: max_episode_steps={hydra_config.max_episode_steps}, debug_level={hydra_config.debug_level}"
    )
    print()


def demo_dataset_config():
    """Demonstrate DatasetConfig usage."""
    print("=== DatasetConfig Demo ===")
    print("Dataset-specific settings and constraints")

    # Create with defaults
    config = DatasetConfig()
    print(
        f"Default: dataset={config.dataset_name}, grid_size={config.max_grid_height}x{config.max_grid_width}, colors={config.max_colors}"
    )

    # Create custom config
    custom_config = DatasetConfig(
        dataset_name="custom-arc",
        max_grid_height=50,
        max_grid_width=40,
        max_colors=15,
        task_split="eval",
        max_tasks=100,
    )
    print(
        f"Custom: dataset={custom_config.dataset_name}, grid_size={custom_config.max_grid_height}x{custom_config.max_grid_width}"
    )

    # Validate
    errors = custom_config.validate()
    print(f"Validation errors: {errors}")

    # Create from Hydra config
    hydra_cfg = OmegaConf.create({
        "dataset_name": "arc-agi-2",
        "max_grid_height": 35,
        "max_colors": 12,
        "task_split": "training",
    })
    hydra_config = DatasetConfig.from_hydra(hydra_cfg)
    print(
        f"From Hydra: dataset={hydra_config.dataset_name}, split={hydra_config.task_split}"
    )
    print()


def demo_visualization_config():
    """Demonstrate VisualizationConfig usage."""
    print("=== VisualizationConfig Demo ===")

    # Create with defaults
    config = VisualizationConfig()
    print(
        f"Default: enabled={config.enabled}, level={config.level}, formats={config.output_formats}"
    )

    # Create custom config
    custom_config = VisualizationConfig(
        level="full",
        output_formats=["svg", "png"],
        show_coordinates=True,
        max_memory_mb=1000,  # Standardized naming
    )
    print(
        f"Custom: level={custom_config.level}, formats={custom_config.output_formats}, memory_mb={custom_config.max_memory_mb}"
    )

    # Validate
    errors = custom_config.validate()
    print(f"Validation errors: {errors}")
    print()


def demo_storage_config():
    """Demonstrate StorageConfig usage."""
    print("=== StorageConfig Demo ===")

    # Create with different policies
    policies = ["none", "minimal", "standard", "research"]

    for policy in policies:
        config = StorageConfig(policy=policy)
        print(
            f"Policy '{policy}': max_episodes={config.max_episodes_per_run}, max_storage={config.max_storage_gb}GB"
        )

    # Create research config
    research_config = StorageConfig(
        policy="research",
        max_episodes_per_run=1000,
        max_storage_gb=50.0,
        cleanup_policy="manual",
        auto_cleanup=False,
    )
    print(
        f"Research: max_episodes={research_config.max_episodes_per_run}, cleanup={research_config.cleanup_policy}"
    )
    print()


def demo_logging_config():
    """Demonstrate LoggingConfig usage."""
    print("=== LoggingConfig Demo ===")

    # Create with defaults
    config = LoggingConfig()
    print(
        f"Default: format={config.log_format}, level={config.log_level}, queue_size={config.queue_size}"
    )

    # Create high-performance config with specific logging flags
    perf_config = LoggingConfig(
        log_format="json",
        log_level="DEBUG",
        log_operations=True,
        log_rewards=True,
        log_frequency=5,
        queue_size=2000,
        worker_threads=4,
        batch_size=20,
        flush_interval=2.0,
    )
    print(
        f"Performance: queue_size={perf_config.queue_size}, workers={perf_config.worker_threads}, log_ops={perf_config.log_operations}"
    )
    print()


def demo_wandb_config():
    """Demonstrate WandbConfig usage."""
    print("=== WandbConfig Demo ===")

    # Create with defaults
    config = WandbConfig()
    print(
        f"Default: enabled={config.enabled}, project={config.project_name}, tags={config.tags}"
    )

    # Create research config
    research_config = WandbConfig(
        enabled=True,
        project_name="jaxarc-research",
        tags=["research", "experiment"],
        log_frequency=5,
        image_format="both",
        max_image_size=(1024, 768),
        log_gradients=True,
        log_system_metrics=True,
    )
    print(
        f"Research: project={research_config.project_name}, tags={research_config.tags}"
    )
    print(
        f"Research: image_size={research_config.max_image_size}, log_gradients={research_config.log_gradients}"
    )

    # Validate
    errors = research_config.validate()
    print(f"Validation errors: {errors}")
    print()


def demo_type_safety():
    """Demonstrate type safety and JAX compatibility."""
    print("=== Type Safety Demo ===")

    # All configs are Equinox modules, making them JAX-compatible
    env_config = EnvironmentConfig(max_episode_steps=100)
    dataset_config = DatasetConfig(dataset_name="arc-agi-1")
    viz_config = VisualizationConfig(enabled=True)

    print(f"Environment config type: {type(env_config)}")
    print(f"Dataset config type: {type(dataset_config)}")
    print(f"Visualization config type: {type(viz_config)}")

    # Configs are immutable (frozen dataclasses)
    try:
        # This would raise an error if we tried to modify
        # env_config.max_episode_steps = 200  # Would fail
        print("Configs are immutable - use .replace() to modify")

        # Proper way to modify
        new_config = env_config.__class__(
            max_episode_steps=200,
            auto_reset=env_config.auto_reset,
            strict_validation=env_config.strict_validation,
            allow_invalid_actions=env_config.allow_invalid_actions,
            debug_level=env_config.debug_level,
        )
        print(f"Modified config: max_episode_steps={new_config.max_episode_steps}")

    except Exception as e:
        print(f"Error: {e}")

    print()


def main():
    """Run all configuration demos."""
    print("JaxARC Equinox Configuration System Demo")
    print("=" * 50)
    print()

    demo_environment_config()
    demo_dataset_config()
    demo_visualization_config()
    demo_storage_config()
    demo_logging_config()
    demo_wandb_config()
    demo_type_safety()

    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
