"""
Configuration Factory System Demo

This example demonstrates the new unified configuration factory system
that replaces the dual configuration pattern with a single, typed
configuration approach using JaxArcConfig.
"""

import sys
from pathlib import Path

# Add src to path for demo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jaxarc.envs import (
    ConfigFactory,
    ConfigPresets,
    JaxArcConfig,
    from_preset,
    get_available_presets,
)
from omegaconf import DictConfig


def demo_basic_factory_usage():
    """Demonstrate basic ConfigFactory usage."""
    print("=== Basic ConfigFactory Usage ===")
    
    # Create development configuration
    dev_config = ConfigFactory.create_development_config()
    print(f"Development config: {dev_config.environment.max_episode_steps} steps, "
          f"debug level: {dev_config.environment.debug_level}")
    
    # Create research configuration
    research_config = ConfigFactory.create_research_config()
    print(f"Research config: {research_config.environment.max_episode_steps} steps, "
          f"debug level: {research_config.environment.debug_level}")
    
    # Create production configuration
    prod_config = ConfigFactory.create_production_config()
    print(f"Production config: {prod_config.environment.max_episode_steps} steps, "
          f"debug level: {prod_config.environment.debug_level}")
    
    print()


def demo_factory_with_overrides():
    """Demonstrate ConfigFactory with custom overrides."""
    print("=== ConfigFactory with Overrides ===")
    
    # Create custom development config
    custom_dev = ConfigFactory.create_development_config(
        max_episode_steps=75,
        dataset_name="mini-arc",
        visualization_level="verbose",
        wandb_enabled=True,
        wandb_project="my-experiment"
    )
    
    print(f"Custom dev config:")
    print(f"  - Episode steps: {custom_dev.environment.max_episode_steps}")
    print(f"  - Dataset: {custom_dev.dataset.dataset_name}")
    print(f"  - Visualization: {custom_dev.visualization.level}")
    print(f"  - WandB enabled: {custom_dev.wandb.enabled}")
    print(f"  - WandB project: {custom_dev.wandb.project_name}")
    
    print()


def demo_hydra_conversion():
    """Demonstrate Hydra config conversion."""
    print("=== Hydra Config Conversion ===")
    
    # Simulate a Hydra configuration
    hydra_dict = {
        "environment": {
            "max_episode_steps": 120,
            "debug_level": "standard"
        },
        "dataset": {
            "dataset_name": "concept-arc",
            "task_split": "corpus"
        },
        "visualization": {
            "enabled": True,
            "level": "full"
        },
        "wandb": {
            "enabled": True,
            "project_name": "hydra-experiment"
        }
    }
    
    hydra_config = DictConfig(hydra_dict)
    
    # Convert to JaxArcConfig (eliminates dual config pattern)
    unified_config = ConfigFactory.from_hydra(hydra_config)
    
    print(f"Converted Hydra config:")
    print(f"  - Episode steps: {unified_config.environment.max_episode_steps}")
    print(f"  - Dataset: {unified_config.dataset.dataset_name}")
    print(f"  - Task split: {unified_config.dataset.task_split}")
    print(f"  - Visualization: {unified_config.visualization.level}")
    print(f"  - WandB project: {unified_config.wandb.project_name}")
    
    print()


def demo_preset_system():
    """Demonstrate the preset system."""
    print("=== Configuration Preset System ===")
    
    # Show available presets
    presets = get_available_presets()
    print(f"Available presets ({len(presets)}):")
    for preset in sorted(presets):
        print(f"  - {preset}")
    
    print()
    
    # Load specific presets
    presets_to_demo = ["development", "research", "testing", "mini_arc", "evaluation"]
    
    for preset_name in presets_to_demo:
        config = from_preset(preset_name)
        print(f"{preset_name.capitalize()} preset:")
        print(f"  - Episode steps: {config.environment.max_episode_steps}")
        print(f"  - Debug level: {config.environment.debug_level}")
        print(f"  - Dataset: {config.dataset.dataset_name}")
        print(f"  - Visualization: {config.visualization.level}")
        print()


def demo_preset_with_overrides():
    """Demonstrate preset system with overrides."""
    print("=== Preset System with Overrides ===")
    
    # Load mini_arc preset with custom settings
    custom_mini = ConfigPresets.from_preset(
        "mini_arc",
        max_episode_steps=60,  # Override default
        wandb_enabled=True,
        wandb_project="mini-arc-experiment"
    )
    
    print(f"Custom mini_arc preset:")
    print(f"  - Episode steps: {custom_mini.environment.max_episode_steps} (overridden)")
    print(f"  - Dataset: {custom_mini.dataset.dataset_name}")
    print(f"  - Grid size: {custom_mini.dataset.max_grid_height}x{custom_mini.dataset.max_grid_width}")
    print(f"  - Selection format: {custom_mini.action.selection_format}")
    print(f"  - WandB enabled: {custom_mini.wandb.enabled} (overridden)")
    print(f"  - WandB project: {custom_mini.wandb.project_name} (overridden)")
    
    print()


def demo_config_validation():
    """Demonstrate configuration validation."""
    print("=== Configuration Validation ===")
    
    # Create a config and validate it
    config = ConfigFactory.create_research_config(
        max_episode_steps=200,
        wandb_enabled=True
    )
    
    validation_errors = config.validate()
    
    if validation_errors:
        print(f"Validation errors found ({len(validation_errors)}):")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("✓ Configuration validation passed")
    
    # Demonstrate preset validation
    print("\nPreset validation results:")
    test_presets = ["development", "research", "production", "testing"]
    
    for preset_name in test_presets:
        errors = ConfigPresets.validate_preset(preset_name)
        if errors:
            print(f"  {preset_name}: {len(errors)} errors")
        else:
            print(f"  {preset_name}: ✓ valid")
    
    print()


def demo_yaml_serialization():
    """Demonstrate YAML serialization."""
    print("=== YAML Serialization ===")
    
    # Create a config and serialize to YAML
    config = ConfigFactory.create_development_config(
        max_episode_steps=50,
        dataset_name="mini-arc",
        visualization_level="standard"
    )
    
    yaml_str = config.to_yaml()
    
    print("Configuration as YAML:")
    print("```yaml")
    # Show first few lines of YAML
    yaml_lines = yaml_str.split('\n')
    for line in yaml_lines[:20]:  # Show first 20 lines
        print(line)
    if len(yaml_lines) > 20:
        print("... (truncated)")
    print("```")
    
    print()


def demo_single_config_pattern():
    """Demonstrate the single configuration pattern (eliminates dual config)."""
    print("=== Single Configuration Pattern ===")
    
    print("OLD DUAL PATTERN (deprecated):")
    print("  env = ArcEnvironment(arc_env_config, hydra_config)  # Confusing!")
    print()
    
    print("NEW SINGLE PATTERN:")
    print("  config = ConfigFactory.create_development_config()")
    print("  env = ArcEnvironment(config)  # Clear and simple!")
    print()
    
    print("OR with Hydra:")
    print("  config = ConfigFactory.from_hydra(hydra_cfg)")
    print("  env = ArcEnvironment(config)  # Single source of truth!")
    print()
    
    print("OR with presets:")
    print("  config = ConfigFactory.from_preset('research', wandb_enabled=True)")
    print("  env = ArcEnvironment(config)  # Preset with overrides!")
    print()


def main():
    """Run all configuration factory demos."""
    print("🚀 JaxARC Configuration Factory System Demo\n")
    
    demo_basic_factory_usage()
    demo_factory_with_overrides()
    demo_hydra_conversion()
    demo_preset_system()
    demo_preset_with_overrides()
    demo_config_validation()
    demo_yaml_serialization()
    demo_single_config_pattern()
    
    print("✨ Demo completed! The new configuration factory system provides:")
    print("  - Single, unified configuration object (JaxArcConfig)")
    print("  - Type-safe configuration with validation")
    print("  - Preset system for common patterns")
    print("  - Seamless Hydra integration")
    print("  - YAML serialization support")
    print("  - Eliminates dual configuration pattern")


if __name__ == "__main__":
    main()