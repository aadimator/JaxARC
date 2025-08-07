#!/usr/bin/env python3
"""
Batched Logging Configuration Examples

This example demonstrates different configuration patterns for batched logging
in various scenarios and use cases.

Requirements: 3.1, 3.2, 7.3, 7.4
"""

from jaxarc.envs import JaxArcConfig, LoggingConfig, EnvironmentConfig, WandbConfig
from jaxarc.utils.logging import ExperimentLogger
from omegaconf import DictConfig, OmegaConf
import tempfile
from pathlib import Path


def demonstrate_preset_configurations():
    """Show how to use preset configurations with batched logging."""
    print("=== Preset Configurations ===")
    
    # 1. Research configuration with batched logging
    print("\n1. Research Configuration (detailed logging):")
    research_logging = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=10,
        sampling_enabled=True,
        num_samples=3,
        sample_frequency=50,
        log_aggregated_rewards=True,
        log_aggregated_similarity=True,
        log_loss_metrics=True,
        log_gradient_norms=True,
        log_episode_lengths=True,
        log_success_rates=True,
    )
    research_config = JaxArcConfig(
        logging=research_logging,
        environment=EnvironmentConfig(debug_level="research")
    )
    
    print(f"   Batched logging enabled: {research_config.logging.batched_logging_enabled}")
    print(f"   Log frequency: {research_config.logging.log_frequency}")
    print(f"   Sampling enabled: {research_config.logging.sampling_enabled}")
    print(f"   Number of samples: {research_config.logging.num_samples}")
    print(f"   Sample frequency: {research_config.logging.sample_frequency}")
    print(f"   Aggregated rewards: {research_config.logging.log_aggregated_rewards}")
    print(f"   Loss metrics: {research_config.logging.log_loss_metrics}")
    
    # 2. Training configuration adapted for batched logging
    print("\n2. Training Configuration (performance focused):")
    training_logging = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=100,  # Less frequent for performance
        sampling_enabled=True,
        num_samples=2,  # Fewer samples
        sample_frequency=500,  # Much less frequent sampling
        log_aggregated_rewards=True,
        log_aggregated_similarity=False,  # Disable for performance
        log_loss_metrics=True,
        log_gradient_norms=True,
        log_episode_lengths=False,  # Disable for performance
        log_success_rates=True,
    )
    training_config = JaxArcConfig(logging=training_logging)
    
    print(f"   Batched logging enabled: {training_config.logging.batched_logging_enabled}")
    print(f"   Log frequency: {training_config.logging.log_frequency}")
    print(f"   Number of samples: {training_config.logging.num_samples}")
    print(f"   Sample frequency: {training_config.logging.sample_frequency}")
    
    # 3. Standard configuration with minimal batched logging
    print("\n3. Standard Configuration (minimal overhead):")
    standard_logging = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=50,
        sampling_enabled=False,  # Disable sampling for minimal overhead
        log_aggregated_rewards=True,
        log_aggregated_similarity=False,  # Disable non-essential metrics
        log_loss_metrics=True,
        log_gradient_norms=False,
        log_episode_lengths=False,
        log_success_rates=True,
    )
    standard_config = JaxArcConfig(logging=standard_logging)
    
    print(f"   Batched logging enabled: {standard_config.logging.batched_logging_enabled}")
    print(f"   Sampling enabled: {standard_config.logging.sampling_enabled}")
    print(f"   Aggregated rewards: {standard_config.logging.log_aggregated_rewards}")
    print(f"   Aggregated similarity: {standard_config.logging.log_aggregated_similarity}")
    print(f"   Gradient norms: {standard_config.logging.log_gradient_norms}")


def demonstrate_custom_configurations():
    """Show how to create custom configurations for specific needs."""
    print("\n=== Custom Configurations ===")
    
    # 1. High-frequency debugging configuration
    print("\n1. High-Frequency Debugging Configuration:")
    debug_logging = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=1,  # Log every update
        sampling_enabled=True,
        num_samples=8,  # Many samples for debugging
        sample_frequency=5,  # Frequent sampling
        # Enable all metrics for debugging
        log_aggregated_rewards=True,
        log_aggregated_similarity=True,
        log_loss_metrics=True,
        log_gradient_norms=True,
        log_episode_lengths=True,
        log_success_rates=True,
        # Enable detailed logging
        log_operations=True,
        log_grid_changes=True,
        include_full_states=True,
    )
    debug_config = JaxArcConfig(
        logging=debug_logging,
        environment=EnvironmentConfig(debug_level="verbose")
    )
    
    print(f"   Log frequency: {debug_config.logging.log_frequency}")
    print(f"   Sample frequency: {debug_config.logging.sample_frequency}")
    print(f"   Number of samples: {debug_config.logging.num_samples}")
    print(f"   All metrics enabled: True")
    
    # 2. Production training configuration
    print("\n2. Production Training Configuration:")
    production_logging = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=1000,  # Very infrequent
        sampling_enabled=True,
        num_samples=1,  # Minimal sampling
        sample_frequency=5000,  # Very infrequent sampling
        # Only essential metrics
        log_aggregated_rewards=True,
        log_aggregated_similarity=False,
        log_loss_metrics=True,
        log_gradient_norms=False,
        log_episode_lengths=False,
        log_success_rates=True,
        # Disable expensive features
        log_operations=False,
        log_grid_changes=False,
        include_full_states=False,
    )
    production_config = JaxArcConfig(
        logging=production_logging,
        environment=EnvironmentConfig(debug_level="minimal")
    )
    
    print(f"   Log frequency: {production_config.logging.log_frequency}")
    print(f"   Sample frequency: {production_config.logging.sample_frequency}")
    print(f"   Number of samples: {production_config.logging.num_samples}")
    print(f"   Minimal metrics enabled")
    
    # 3. Experiment comparison configuration
    print("\n3. Experiment Comparison Configuration:")
    comparison_logging = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=25,  # Regular logging
        sampling_enabled=True,
        num_samples=3,  # Moderate sampling
        sample_frequency=100,  # Regular sampling
        # Focus on comparison-relevant metrics
        log_aggregated_rewards=True,
        log_aggregated_similarity=True,
        log_loss_metrics=True,
        log_gradient_norms=True,
        log_episode_lengths=False,  # Less relevant for comparison
        log_success_rates=True,
    )
    comparison_config = JaxArcConfig(
        logging=comparison_logging,
        environment=EnvironmentConfig(debug_level="standard")
    )
    
    print(f"   Log frequency: {comparison_config.logging.log_frequency}")
    print(f"   Sample frequency: {comparison_config.logging.sample_frequency}")
    print(f"   Comparison-focused metrics enabled")


def demonstrate_wandb_integration_config():
    """Show how to configure batched logging with Weights & Biases."""
    print("\n=== Weights & Biases Integration ===")
    
    # Create configuration with wandb integration
    wandb_config_obj = WandbConfig(
        enabled=True,
        project_name="jaxarc-batched-training",
        tags=("batched", "research"),
        log_frequency=10,  # Sync with batched logging frequency
        offline_mode=False,
        save_code=True,
        save_config=True,
    )
    
    batched_logging = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=10,  # Align with wandb frequency
        sampling_enabled=True,
        num_samples=3,
        sample_frequency=50,
        log_aggregated_rewards=True,
        log_aggregated_similarity=True,
        log_loss_metrics=True,
        log_gradient_norms=True,
        log_episode_lengths=True,
        log_success_rates=True,
    )
    
    wandb_config = JaxArcConfig(
        logging=batched_logging,
        wandb=wandb_config_obj
    )
    
    print("Wandb configuration:")
    print(f"   Enabled: {wandb_config.wandb.enabled}")
    print(f"   Project: {wandb_config.wandb.project_name}")
    print(f"   Tags: {wandb_config.wandb.tags}")
    print(f"   Log frequency: {wandb_config.wandb.log_frequency}")
    print(f"   Batched log frequency: {wandb_config.logging.log_frequency}")
    print("   Note: Frequencies are aligned for optimal tracking")


def demonstrate_yaml_configuration():
    """Show how to create YAML configuration files for batched logging."""
    print("\n=== YAML Configuration Files ===")
    
    # Create a temporary directory for demonstration
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Create a custom batched logging configuration
        custom_batched_config = {
            "logging": {
                "structured_logging": True,
                "log_format": "json",
                "log_level": "INFO",
                "compression": True,
                "include_full_states": False,
                
                # Batched logging settings
                "batched_logging_enabled": True,
                "log_frequency": 20,
                
                # Sampling settings
                "sampling_enabled": True,
                "num_samples": 4,
                "sample_frequency": 80,
                
                # Metric selection
                "log_aggregated_rewards": True,
                "log_aggregated_similarity": True,
                "log_loss_metrics": True,
                "log_gradient_norms": True,
                "log_episode_lengths": True,
                "log_success_rates": True,
            }
        }
        
        # Save to YAML file
        config_file = temp_path / "custom_batched.yaml"
        with open(config_file, 'w') as f:
            OmegaConf.save(custom_batched_config, f)
        
        print(f"Created custom configuration file: {config_file}")
        
        # Load and display the configuration
        loaded_config = OmegaConf.load(config_file)
        print("Configuration contents:")
        print(OmegaConf.to_yaml(loaded_config))
        
        # 2. Create a performance-optimized configuration
        performance_config = {
            "logging": {
                "batched_logging_enabled": True,
                "log_frequency": 100,  # Less frequent for performance
                "sampling_enabled": True,
                "num_samples": 2,  # Fewer samples
                "sample_frequency": 500,  # Much less frequent
                
                # Disable expensive features
                "log_operations": False,
                "log_grid_changes": False,
                "include_full_states": False,
                
                # Essential metrics only
                "log_aggregated_rewards": True,
                "log_aggregated_similarity": False,
                "log_loss_metrics": True,
                "log_gradient_norms": False,
                "log_episode_lengths": False,
                "log_success_rates": True,
            }
        }
        
        perf_config_file = temp_path / "performance_batched.yaml"
        with open(perf_config_file, 'w') as f:
            OmegaConf.save(performance_config, f)
        
        print(f"\nCreated performance configuration file: {perf_config_file}")
        print("Performance configuration focuses on minimal overhead")


def demonstrate_backward_compatibility():
    """Show that old configurations still work without batched logging."""
    print("\n=== Backward Compatibility ===")
    
    # 1. Create old-style configuration (without batched logging fields)
    old_logging = LoggingConfig(
        structured_logging=True,
        log_format="json",
        log_level="INFO",
        compression=True,
        include_full_states=False,
        log_operations=False,
        log_grid_changes=False,
        log_rewards=False,
        log_episode_start=True,
        log_episode_end=True,
        log_key_moments=True,
        log_frequency=10,
        # Batched logging disabled (default behavior)
        batched_logging_enabled=False,
    )
    
    old_config = JaxArcConfig(logging=old_logging)
    
    print("Old-style configuration (batched logging disabled):")
    print(f"   Batched logging: {old_config.logging.batched_logging_enabled}")
    print(f"   Regular logging still works: {old_config.logging.structured_logging}")
    
    # Initialize logger with old configuration
    try:
        logger = ExperimentLogger(old_config)
        print(f"   Logger initialized successfully with {len(logger.handlers)} handlers")
        
        # Test regular logging methods still work
        step_data = {
            "step_num": 1,
            "reward": 0.5,
            "info": {"similarity": 0.8}
        }
        logger.log_step(step_data)
        print("   Regular step logging works")
        
        episode_data = {
            "episode_num": 1,
            "total_reward": 10.0,
            "success": True
        }
        logger.log_episode_summary(episode_data)
        print("   Episode summary logging works")
        
        # Test that batched logging is safely ignored
        batch_data = {
            "update_step": 1,
            "episode_returns": [1.0, 2.0, 3.0],
            "policy_loss": 0.5,
        }
        logger.log_batch_step(batch_data)  # Should be safely ignored
        print("   Batched logging safely ignored when disabled")
        
        logger.close()
        print("   Backward compatibility confirmed")
        
    except Exception as e:
        print(f"   Error: {e}")


def main():
    """Main demonstration of configuration patterns."""
    print("Batched Logging Configuration Examples")
    print("=" * 50)
    
    try:
        # Show preset configurations
        demonstrate_preset_configurations()
        
        # Show custom configurations
        demonstrate_custom_configurations()
        
        # Show wandb integration
        demonstrate_wandb_integration_config()
        
        # Show YAML configuration creation
        demonstrate_yaml_configuration()
        
        # Show backward compatibility
        demonstrate_backward_compatibility()
        
        print("\n" + "=" * 50)
        print("Configuration examples completed!")
        print("\nKey takeaways:")
        print("- Use preset configurations for common scenarios")
        print("- Customize frequencies based on performance needs")
        print("- Enable only necessary metrics for efficiency")
        print("- Old configurations continue to work unchanged")
        
    except Exception as e:
        print(f"Configuration demo failed: {e}")
        raise


if __name__ == "__main__":
    main()