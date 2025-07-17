#!/usr/bin/env python3
"""
Hydra Configuration Composition Demonstration for JaxARC.

This example showcases the enhanced Hydra configuration system with:
- Modular configuration composition
- Comprehensive validation with clear error messages
- Cross-field consistency checking
- Configuration presets and overrides
- Integration with new Equinox state management

Run with: pixi run python examples/hydra_composition_demo.py
"""

import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from jaxarc.envs.config import (
    ArcEnvConfig, RewardConfig, ActionConfig, GridConfig, DatasetConfig,
    ConfigValidationError
)
from jaxarc.envs import arc_reset, arc_step
from jaxarc.parsers import MiniArcParser


def demo_basic_hydra_config():
    """Demonstrate basic Hydra configuration creation and validation."""
    print("🔧 Basic Hydra Configuration Demo")
    print("=" * 50)
    
    # Create a basic Hydra config
    hydra_config = OmegaConf.create({
        "max_episode_steps": 100,
        "auto_reset": True,
        "log_operations": False,
        "reward": {
            "reward_on_submit_only": True,
            "step_penalty": -0.01,
            "success_bonus": 10.0,
            "similarity_weight": 1.0,
            "progress_bonus": 0.0,
            "invalid_action_penalty": -0.1
        },
        "grid": {
            "max_grid_height": 30,
            "max_grid_width": 30,
            "min_grid_height": 3,
            "min_grid_width": 3,
            "max_colors": 10,
            "background_color": 0
        },
        "action": {
            "selection_format": "mask",
            "selection_threshold": 0.5,
            "num_operations": 35,
            "validate_actions": True,
            "clip_invalid_actions": True
        },
        "dataset": {
            "dataset_name": "mini-arc",
            "task_split": "training",
            "shuffle_tasks": True
        }
    })
    
    print("Created Hydra configuration:")
    print(OmegaConf.to_yaml(hydra_config))
    
    # Convert to typed configuration with validation
    try:
        typed_config = ArcEnvConfig.from_hydra(hydra_config)
        print("✅ Configuration validation passed!")
        print(f"   Max episode steps: {typed_config.max_episode_steps}")
        print(f"   Reward success bonus: {typed_config.reward.success_bonus}")
        print(f"   Action selection format: {typed_config.action.selection_format}")
        print(f"   Grid max size: {typed_config.grid.max_grid_height}x{typed_config.grid.max_grid_width}")
        
    except ConfigValidationError as e:
        print(f"❌ Configuration validation failed: {e}")
    
    return typed_config


def demo_modular_composition():
    """Demonstrate modular configuration composition."""
    print("\n🧩 Modular Configuration Composition Demo")
    print("=" * 50)
    
    # Create separate configuration components
    reward_config = {
        "reward_on_submit_only": False,  # Dense rewards for training
        "step_penalty": -0.005,
        "success_bonus": 20.0,
        "similarity_weight": 2.0,
        "progress_bonus": 0.5,
        "invalid_action_penalty": -0.2
    }
    
    action_config = {
        "selection_format": "point",  # Point-based actions
        "num_operations": 20,         # Reduced operation set
        "allowed_operations": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 33, 34],  # Basic operations only
        "validate_actions": True,
        "clip_invalid_actions": True
    }
    
    grid_config = {
        "max_grid_height": 15,  # Smaller grids for faster training
        "max_grid_width": 15,
        "min_grid_height": 3,
        "min_grid_width": 3,
        "max_colors": 8,        # Reduced color palette
        "background_color": 0
    }
    
    dataset_config = {
        "dataset_name": "mini-arc",
        "task_split": "training",
        "max_tasks": 100,       # Limit for faster iteration
        "shuffle_tasks": True
    }
    
    # Compose into complete configuration
    training_config = OmegaConf.create({
        "max_episode_steps": 150,  # Longer episodes for training
        "auto_reset": True,
        "log_operations": True,    # Enable logging for debugging
        "reward": reward_config,
        "action": action_config,
        "grid": grid_config,
        "dataset": dataset_config
    })
    
    print("Composed training configuration:")
    print("Reward component:")
    print(f"  - Dense rewards: {not training_config.reward.reward_on_submit_only}")
    print(f"  - Success bonus: {training_config.reward.success_bonus}")
    print(f"  - Progress bonus: {training_config.reward.progress_bonus}")
    
    print("Action component:")
    print(f"  - Selection format: {training_config.action.selection_format}")
    print(f"  - Number of operations: {training_config.action.num_operations}")
    print(f"  - Allowed operations: {len(training_config.action.allowed_operations)} specified")
    
    print("Grid component:")
    print(f"  - Max grid size: {training_config.grid.max_grid_height}x{training_config.grid.max_grid_width}")
    print(f"  - Color palette: {training_config.grid.max_colors} colors")
    
    # Validate composed configuration
    try:
        typed_training_config = ArcEnvConfig.from_hydra(training_config)
        print("✅ Modular composition validation passed!")
        
    except ConfigValidationError as e:
        print(f"❌ Modular composition validation failed: {e}")
    
    return typed_training_config


def demo_configuration_presets():
    """Demonstrate different configuration presets for various use cases."""
    print("\n🎯 Configuration Presets Demo")
    print("=" * 50)
    
    presets = {}
    
    # Preset 1: Rapid Prototyping
    presets["rapid_prototyping"] = OmegaConf.create({
        "max_episode_steps": 30,
        "reward": {
            "step_penalty": -0.001,    # Very small penalty
            "success_bonus": 5.0,      # Quick feedback
            "similarity_weight": 0.5   # Less strict similarity
        },
        "grid": {
            "max_grid_height": 10,     # Small grids
            "max_grid_width": 10,
            "max_colors": 5            # Simple color palette
        },
        "action": {
            "selection_format": "point",
            "num_operations": 10,      # Minimal operations
            "allowed_operations": [0, 1, 2, 3, 4, 33, 34]  # Fill colors + submit
        },
        "dataset": {
            "dataset_name": "mini-arc",
            "max_tasks": 20           # Very small dataset
        }
    })
    
    # Preset 2: Standard Training
    presets["standard_training"] = OmegaConf.create({
        "max_episode_steps": 100,
        "reward": {
            "reward_on_submit_only": False,
            "step_penalty": -0.01,
            "success_bonus": 15.0,
            "similarity_weight": 1.5,
            "progress_bonus": 0.2
        },
        "grid": {
            "max_grid_height": 25,
            "max_grid_width": 25,
            "max_colors": 10
        },
        "action": {
            "selection_format": "mask",
            "num_operations": 25,      # Most operations
            "validate_actions": True
        },
        "dataset": {
            "dataset_name": "arc-agi-1",
            "task_split": "training"
        }
    })
    
    # Preset 3: Evaluation
    presets["evaluation"] = OmegaConf.create({
        "max_episode_steps": 50,
        "reward": {
            "reward_on_submit_only": True,  # Sparse rewards
            "step_penalty": -0.02,          # Higher penalty
            "success_bonus": 10.0,
            "similarity_weight": 1.0,
            "progress_bonus": 0.0           # No progress bonus
        },
        "grid": {
            "max_grid_height": 30,
            "max_grid_width": 30,
            "max_colors": 10
        },
        "action": {
            "selection_format": "mask",
            "num_operations": 35,           # All operations
            "validate_actions": True,
            "clip_invalid_actions": False   # Strict validation
        },
        "dataset": {
            "dataset_name": "arc-agi-1",
            "task_split": "evaluation"
        }
    })
    
    # Test each preset
    for preset_name, preset_config in presets.items():
        print(f"\n{preset_name.replace('_', ' ').title()} Preset:")
        try:
            typed_config = ArcEnvConfig.from_hydra(preset_config)
            print(f"  ✅ Valid configuration")
            print(f"     Episodes: {typed_config.max_episode_steps} steps")
            print(f"     Grid size: {typed_config.grid.max_grid_height}x{typed_config.grid.max_grid_width}")
            print(f"     Operations: {typed_config.action.num_operations}")
            print(f"     Success bonus: {typed_config.reward.success_bonus}")
            
        except ConfigValidationError as e:
            print(f"  ❌ Invalid configuration: {e}")
    
    return presets["standard_training"]


def demo_configuration_overrides():
    """Demonstrate configuration overrides and customization."""
    print("\n🔄 Configuration Overrides Demo")
    print("=" * 50)
    
    # Start with base configuration
    base_config = OmegaConf.create({
        "max_episode_steps": 100,
        "reward": {
            "success_bonus": 10.0,
            "step_penalty": -0.01
        },
        "action": {
            "selection_format": "mask",
            "num_operations": 35
        },
        "grid": {
            "max_grid_height": 30,
            "max_grid_width": 30
        }
    })
    
    print("Base configuration:")
    print(f"  Episode steps: {base_config.max_episode_steps}")
    print(f"  Success bonus: {base_config.reward.success_bonus}")
    print(f"  Selection format: {base_config.action.selection_format}")
    
    # Create overrides for different scenarios
    overrides = [
        {
            "name": "Quick Testing",
            "overrides": {
                "max_episode_steps": 20,
                "reward.success_bonus": 5.0,
                "grid.max_grid_height": 10,
                "grid.max_grid_width": 10
            }
        },
        {
            "name": "Point Actions",
            "overrides": {
                "action.selection_format": "point",
                "action.num_operations": 15
            }
        },
        {
            "name": "High Reward Training",
            "overrides": {
                "reward.success_bonus": 50.0,
                "reward.step_penalty": -0.001,
                "reward.progress_bonus": 1.0
            }
        }
    ]
    
    for override_set in overrides:
        print(f"\n{override_set['name']} Override:")
        
        # Apply overrides to base config
        modified_config = OmegaConf.merge(base_config, OmegaConf.create(override_set['overrides']))
        
        try:
            typed_config = ArcEnvConfig.from_hydra(modified_config)
            print("  ✅ Override applied successfully")
            
            # Show what changed
            for key, value in override_set['overrides'].items():
                print(f"     {key}: {value}")
                
        except ConfigValidationError as e:
            print(f"  ❌ Override validation failed: {e}")


def demo_validation_errors():
    """Demonstrate comprehensive validation error handling."""
    print("\n🚨 Validation Error Handling Demo")
    print("=" * 50)
    
    # Test various validation errors
    error_configs = [
        {
            "name": "Negative Episode Steps",
            "config": {"max_episode_steps": -10},
            "expected_error": "max_episode_steps must be positive"
        },
        {
            "name": "Invalid Selection Format",
            "config": {"action": {"selection_format": "invalid_format"}},
            "expected_error": "selection_format must be one of"
        },
        {
            "name": "Grid Size Inconsistency",
            "config": {
                "grid": {
                    "max_grid_height": 10,
                    "min_grid_height": 20  # min > max
                }
            },
            "expected_error": "min_grid_height"
        },
        {
            "name": "Invalid Color Configuration",
            "config": {
                "grid": {
                    "max_colors": 5,
                    "background_color": 5  # background_color >= max_colors
                }
            },
            "expected_error": "background_color"
        },
        {
            "name": "Invalid Operation List",
            "config": {
                "action": {
                    "allowed_operations": [0, 1, 2, 50]  # 50 is invalid
                }
            },
            "expected_error": "allowed_operations"
        },
        {
            "name": "Out of Range Values",
            "config": {
                "reward": {
                    "similarity_weight": 15.0  # > 10.0 max
                }
            },
            "expected_error": "similarity_weight must be in range"
        }
    ]
    
    for error_test in error_configs:
        print(f"\nTesting: {error_test['name']}")
        
        # Create config with error
        error_config = OmegaConf.create(error_test['config'])
        
        try:
            ArcEnvConfig.from_hydra(error_config)
            print(f"  ❌ Expected validation error but none occurred")
            
        except ConfigValidationError as e:
            error_message = str(e)
            if error_test['expected_error'].lower() in error_message.lower():
                print(f"  ✅ Caught expected error: {error_message}")
            else:
                print(f"  ⚠️  Caught unexpected error: {error_message}")
                print(f"     Expected: {error_test['expected_error']}")


def demo_environment_integration():
    """Demonstrate using validated configurations with the environment."""
    print("\n🌍 Environment Integration Demo")
    print("=" * 50)
    
    # Create a working configuration
    config = OmegaConf.create({
        "max_episode_steps": 50,
        "auto_reset": True,
        "log_operations": False,
        "reward": {
            "reward_on_submit_only": True,
            "step_penalty": -0.01,
            "success_bonus": 10.0,
            "similarity_weight": 1.0
        },
        "action": {
            "selection_format": "mask",
            "num_operations": 35,
            "validate_actions": True
        },
        "grid": {
            "max_grid_height": 5,  # Small for demo
            "max_grid_width": 5,
            "max_colors": 10
        },
        "dataset": {
            "dataset_name": "mini-arc",
            "task_split": "training"
        }
    })
    
    try:
        # Validate configuration
        typed_config = ArcEnvConfig.from_hydra(config)
        print("✅ Configuration validated successfully")
        
        # Create a simple demo task (since we don't have actual dataset files)
        print("\n🎮 Environment Usage Demo")
        print("Note: Using mock task data for demonstration")
        
        # Mock task data
        from jaxarc.types import JaxArcTask
        
        # Create simple 3x3 task
        input_grid = jnp.zeros((10, 30, 30), dtype=jnp.int32)  # Padded to max size
        output_grid = jnp.zeros((10, 30, 30), dtype=jnp.int32)
        
        # Set a simple pattern in the first task
        input_grid = input_grid.at[0, :3, :3].set(jnp.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]))
        
        output_grid = output_grid.at[0, :3, :3].set(jnp.array([
            [2, 2, 2],
            [2, 1, 2],
            [2, 2, 2]
        ]))
        
        task = JaxArcTask(
            input_grids_examples=input_grid,
            output_grids_examples=output_grid,
            num_train_pairs=1,
            test_input_grids=input_grid[:1],
            true_test_output_grids=output_grid[:1],
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32)
        )
        
        # Initialize environment
        key = jax.random.PRNGKey(42)
        state, observation = arc_reset(key, typed_config, task_data=task)
        
        print(f"Environment initialized successfully!")
        print(f"  Working grid shape: {state.working_grid.shape}")
        print(f"  Initial step count: {state.step_count}")
        print(f"  Episode done: {state.episode_done}")
        
        # Take a simple action
        action = {
            "selection": jnp.ones((3, 3), dtype=bool),  # Select all cells
            "operation": jnp.array(2, dtype=jnp.int32)  # Fill with color 2
        }
        
        new_state, obs, reward, done, info = arc_step(state, action, typed_config)
        
        print(f"\nAfter action:")
        print(f"  Step count: {new_state.step_count}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Done: {done}")
        print(f"  Similarity: {info.get('similarity', 'N/A')}")
        
        print("✅ Environment integration successful!")
        
    except ConfigValidationError as e:
        print(f"❌ Configuration validation failed: {e}")
    except Exception as e:
        print(f"❌ Environment integration failed: {e}")


def main():
    """Run all Hydra configuration demonstrations."""
    print("🚀 JaxARC Hydra Configuration Composition Demo")
    print("=" * 60)
    print("This demo showcases the enhanced Hydra configuration system:")
    print("- Modular configuration composition")
    print("- Comprehensive validation with clear error messages")
    print("- Configuration presets for different use cases")
    print("- Override system for customization")
    print("- Integration with Equinox state management")
    print("=" * 60)
    
    # Run demonstrations
    demo_basic_hydra_config()
    demo_modular_composition()
    demo_configuration_presets()
    demo_configuration_overrides()
    demo_validation_errors()
    demo_environment_integration()
    
    print("\n🎉 Demo Complete!")
    print("=" * 60)
    print("Key takeaways:")
    print("✅ Hydra configurations provide modular composition")
    print("✅ Comprehensive validation catches errors early with clear messages")
    print("✅ Configuration presets simplify common use cases")
    print("✅ Override system enables easy customization")
    print("✅ Cross-field validation ensures configuration consistency")
    print("✅ Seamless integration with new Equinox state management")
    print("\nFor more information, see:")
    print("- docs/configuration.md")
    print("- docs/migration_guide.md")
    print("- examples/config_api_demo.py")


if __name__ == "__main__":
    main()