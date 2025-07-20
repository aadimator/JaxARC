# Configuration Best Practices Guide

This guide provides recommendations for effectively using JaxARC's unified configuration system, including when to use different presets, how to extend configurations, and best practices for different use cases.

## Overview

JaxARC's unified configuration system is built around the `JaxArcConfig` class, which groups related parameters into logical sections. This design provides type safety, validation, and clear organization while maintaining flexibility for different use cases.

## Configuration Architecture

### Logical Parameter Grouping

Parameters are organized into six main groups:

```python
from jaxarc.envs.config import JaxArcConfig

config = JaxArcConfig(
    environment=EnvironmentConfig(),    # Core environment behavior
    debug=DebugConfig(),               # Debug levels and logging
    visualization=VisualizationConfig(), # Visualization settings
    storage=StorageConfig(),           # Storage policies and limits
    logging=LoggingConfig(),           # Logging configuration
    wandb=WandbConfig()               # Weights & Biases integration
)
```

**Design Principles:**
- **Single Responsibility**: Each group handles one aspect of the system
- **Clear Boundaries**: Minimal overlap between configuration groups
- **Type Safety**: All parameters have proper type annotations
- **Validation**: Comprehensive validation with clear error messages

## Choosing Configuration Presets

### Development Preset

**When to use:**
- Local development and debugging
- Interactive experimentation
- Learning and exploration
- Small-scale testing

```python
from jaxarc.envs.factory import ConfigFactory

config = ConfigFactory.create_development_config()

# Characteristics:
# - Standard debug level with step logging
# - Visualization enabled for immediate feedback
# - Moderate storage limits (5GB)
# - Console logging enabled
# - W&B disabled by default
```

**Best for:**
- First-time users learning JaxARC
- Developing new algorithms or approaches
- Debugging environment behavior
- Quick prototyping and iteration

### Research Preset

**When to use:**
- Academic research and publication
- Comprehensive experiments
- Long-running training sessions
- Detailed analysis and evaluation

```python
config = ConfigFactory.create_research_config()

# Characteristics:
# - Research debug level with full logging
# - Full visualization with detailed output
# - Large storage limits (20GB)
# - File logging enabled
# - W&B integration ready
```

**Best for:**
- Academic papers and research projects
- Comprehensive benchmarking
- Algorithm comparison studies
- Reproducible research workflows

### Production Preset

**When to use:**
- Deployed systems and services
- Large-scale training runs
- Performance-critical applications
- Resource-constrained environments

```python
config = ConfigFactory.create_production_config()

# Characteristics:
# - Minimal debug level (off by default)
# - Visualization disabled for performance
# - Minimal storage overhead
# - Error-level logging only
# - W&B for metrics tracking
```

**Best for:**
- Production training pipelines
- Automated evaluation systems
- Resource-constrained deployments
- Performance benchmarking

## Configuration Customization Patterns

### Preset with Overrides (Recommended)

Start with a preset and override specific parameters:

```python
# Development with custom episode length
config = ConfigFactory.from_preset("development", {
    "environment.max_episode_steps": 200,
    "debug.level": "verbose",
    "visualization.show_coordinates": True
})

# Research with custom storage
config = ConfigFactory.from_preset("research", {
    "storage.max_size_gb": 50.0,
    "storage.compression": True,
    "wandb.project": "my-research-project"
})
```

**Advantages:**
- Maintains preset consistency
- Clear intent and purpose
- Easy to understand and modify
- Inherits preset validation

### Component-Based Construction

Build configuration from individual components:

```python
from jaxarc.envs.config import (
    JaxArcConfig, EnvironmentConfig, DebugConfig,
    VisualizationConfig, StorageConfig, LoggingConfig, WandbConfig
)

# Custom configuration for specific use case
config = JaxArcConfig(
    environment=EnvironmentConfig(
        max_episode_steps=150,
        grid_size=(20, 20),
        reward_on_submit_only=True
    ),
    debug=DebugConfig(
        level="standard",
        log_steps=True,
        log_grids=False,
        save_episodes=True
    ),
    visualization=VisualizationConfig(
        enabled=True,
        level="standard",
        show_grids=True,
        show_actions=False,
        color_scheme="research"
    ),
    storage=StorageConfig(
        policy="standard",
        max_size_gb=10.0,
        cleanup_on_exit=False
    ),
    logging=LoggingConfig(
        level="INFO",
        console_logging=True,
        file_logging=True,
        log_file="experiment.log"
    ),
    wandb=WandbConfig(
        enabled=True,
        project="custom-experiment",
        entity="my-team"
    )
)
```

**When to use:**
- Highly specialized requirements
- Complex multi-component experiments
- Custom research setups
- Advanced users with specific needs

### YAML-Based Configuration

Define configurations in YAML files for reproducibility:

```yaml
# configs/my_experiment.yaml
environment:
  max_episode_steps: 200
  grid_size: [25, 25]
  reward_on_submit_only: false

debug:
  level: "verbose"
  log_steps: true
  log_grids: true
  save_episodes: true

visualization:
  enabled: true
  level: "full"
  show_coordinates: true
  color_scheme: "research"

storage:
  policy: "research"
  max_size_gb: 15.0
  compression: true

logging:
  level: "DEBUG"
  file_logging: true
  log_file: "my_experiment.log"

wandb:
  enabled: true
  project: "my-experiment"
  tags: ["baseline", "v1.0"]
```

```python
# Load from YAML
config = ConfigFactory.from_yaml("configs/my_experiment.yaml")
```

**Advantages:**
- Version control friendly
- Easy to share and reproduce
- Clear documentation of settings
- Supports complex nested structures

## Debug Level Guidelines

### Debug Level Hierarchy

Choose debug levels based on your needs:

```python
# Off - Production use
config = ConfigFactory.from_preset("production", {
    "debug.level": "off"
})
# - No debug output
# - Maximum performance
# - Minimal storage usage

# Minimal - Basic monitoring
config = ConfigFactory.from_preset("development", {
    "debug.level": "minimal"
})
# - Episode summaries only
# - Basic success/failure tracking
# - Lightweight logging

# Standard - Development default
config = ConfigFactory.from_preset("development", {
    "debug.level": "standard"
})
# - Step-by-step logging
# - Grid state tracking
# - Action validation
# - Moderate storage usage

# Verbose - Detailed debugging
config = ConfigFactory.from_preset("development", {
    "debug.level": "verbose"
})
# - Detailed step information
# - Grid transformation logging
# - Action processing details
# - Higher storage usage

# Research - Comprehensive logging
config = ConfigFactory.from_preset("research", {
    "debug.level": "research"
})
# - Complete episode recordings
# - Full grid history
# - Detailed metrics
# - Maximum storage usage
```

### Debug Level Selection Guide

| Use Case | Recommended Level | Rationale |
|----------|------------------|-----------|
| Learning JaxARC | `standard` | Good balance of information and performance |
| Algorithm development | `verbose` | Detailed feedback for debugging |
| Performance testing | `minimal` or `off` | Minimize overhead |
| Research experiments | `research` | Complete data for analysis |
| Production deployment | `off` | Maximum performance |
| Troubleshooting | `verbose` | Detailed diagnostic information |

## Storage Configuration Best Practices

### Storage Policy Selection

```python
# None - No persistent storage
config = ConfigFactory.from_preset("production", {
    "storage.policy": "none"
})
# Use for: Performance testing, temporary experiments

# Minimal - Essential data only
config = ConfigFactory.from_preset("development", {
    "storage.policy": "minimal"
})
# Use for: Development, quick experiments

# Standard - Balanced storage
config = ConfigFactory.from_preset("development", {
    "storage.policy": "standard"
})
# Use for: Regular development, moderate experiments

# Research - Comprehensive storage
config = ConfigFactory.from_preset("research", {
    "storage.policy": "research"
})
# Use for: Research, long-term experiments, reproducibility
```

### Storage Size Guidelines

```python
# Small experiments (< 1GB)
config = ConfigFactory.from_preset("development", {
    "storage.max_size_gb": 1.0,
    "storage.compression": True
})

# Medium experiments (1-10GB)
config = ConfigFactory.from_preset("development", {
    "storage.max_size_gb": 10.0,
    "storage.compression": True,
    "storage.cleanup_on_exit": False
})

# Large experiments (10-50GB)
config = ConfigFactory.from_preset("research", {
    "storage.max_size_gb": 50.0,
    "storage.compression": True,
    "storage.base_dir": "/path/to/large/storage"
})
```

## Visualization Configuration

### Visualization Levels

```python
# Off - No visualization (production)
config = ConfigFactory.from_preset("production", {
    "visualization.enabled": False
})

# Minimal - Basic grid display
config = ConfigFactory.from_preset("development", {
    "visualization.level": "minimal",
    "visualization.show_grids": True,
    "visualization.show_actions": False
})

# Standard - Balanced visualization
config = ConfigFactory.from_preset("development", {
    "visualization.level": "standard",
    "visualization.show_grids": True,
    "visualization.show_actions": True,
    "visualization.show_coordinates": False
})

# Full - Comprehensive visualization
config = ConfigFactory.from_preset("research", {
    "visualization.level": "full",
    "visualization.show_grids": True,
    "visualization.show_actions": True,
    "visualization.show_coordinates": True,
    "visualization.color_scheme": "research"
})
```

### Output Format Selection

```python
# Console only - Interactive development
config = ConfigFactory.from_preset("development", {
    "visualization.output_format": "console"
})

# SVG only - Publication quality
config = ConfigFactory.from_preset("research", {
    "visualization.output_format": "svg"
})

# Both - Maximum flexibility
config = ConfigFactory.from_preset("research", {
    "visualization.output_format": "both"
})
```

## Weights & Biases Integration

### Basic W&B Setup

```python
# Enable W&B for experiment tracking
config = ConfigFactory.from_preset("research", {
    "wandb.enabled": True,
    "wandb.project": "jaxarc-experiments",
    "wandb.entity": "my-team",
    "wandb.tags": ["baseline", "v1.0"]
})
```

### Advanced W&B Configuration

```python
# Comprehensive W&B setup
config = ConfigFactory.from_preset("research", {
    "wandb.enabled": True,
    "wandb.project": "arc-research-2024",
    "wandb.entity": "research-lab",
    "wandb.group": "baseline-experiments",
    "wandb.job_type": "training",
    "wandb.tags": ["conceptarc", "baseline", "equinox"],
    "wandb.notes": "Baseline experiments with ConceptARC dataset",
    "wandb.save_code": True,
    "wandb.log_frequency": 100
})
```

## Environment-Specific Configurations

### ConceptARC Configuration

```python
# Optimized for ConceptARC dataset
config = ConfigFactory.from_preset("research", {
    "environment.max_episode_steps": 150,
    "environment.grid_size": (30, 30),
    "debug.level": "verbose",
    "visualization.show_coordinates": True,
    "storage.policy": "research",
    "wandb.project": "conceptarc-experiments"
})
```

### MiniARC Configuration

```python
# Optimized for MiniARC (5x5 grids)
config = ConfigFactory.from_preset("development", {
    "environment.max_episode_steps": 50,
    "environment.grid_size": (5, 5),
    "debug.level": "standard",
    "storage.policy": "minimal",
    "visualization.level": "standard"
})
```

### Performance Testing Configuration

```python
# Minimal overhead for performance testing
config = ConfigFactory.from_preset("production", {
    "environment.max_episode_steps": 100,
    "debug.level": "off",
    "visualization.enabled": False,
    "storage.policy": "none",
    "logging.level": "ERROR"
})
```

## Configuration Validation and Testing

### Validation Best Practices

```python
# Always validate configurations
config = ConfigFactory.create_development_config()

validation_result = config.validate()
if validation_result.errors:
    print("Configuration errors:")
    for error in validation_result.errors:
        print(f"  - {error}")
    raise ValueError("Invalid configuration")

if validation_result.warnings:
    print("Configuration warnings:")
    for warning in validation_result.warnings:
        print(f"  - {warning}")
```

### Testing Configurations

```python
def test_configuration():
    """Test configuration creation and validation."""
    # Test preset creation
    config = ConfigFactory.create_development_config()
    assert isinstance(config, JaxArcConfig)
    
    # Test validation
    validation_result = config.validate()
    assert len(validation_result.errors) == 0
    
    # Test parameter access
    assert config.environment.max_episode_steps > 0
    assert config.debug.level in ["off", "minimal", "standard", "verbose", "research"]
    
    # Test environment integration
    from jaxarc.envs import ArcEnvironment
    env = ArcEnvironment(config)
    assert env.config == config
```

## Common Configuration Patterns

### Experiment Series Configuration

```python
# Base configuration for experiment series
base_config = {
    "environment.max_episode_steps": 100,
    "debug.level": "standard",
    "visualization.enabled": True,
    "storage.policy": "research",
    "wandb.enabled": True,
    "wandb.project": "experiment-series"
}

# Experiment variations
experiments = [
    {**base_config, "wandb.group": "baseline", "environment.reward_on_submit_only": False},
    {**base_config, "wandb.group": "submit-only", "environment.reward_on_submit_only": True},
    {**base_config, "wandb.group": "long-episodes", "environment.max_episode_steps": 200},
]

for i, exp_config in enumerate(experiments):
    config = ConfigFactory.from_preset("research", exp_config)
    # Run experiment with config
```

### A/B Testing Configuration

```python
def create_ab_test_configs():
    """Create configurations for A/B testing."""
    base_overrides = {
        "debug.level": "minimal",
        "storage.policy": "standard",
        "wandb.enabled": True,
        "wandb.project": "ab-test"
    }
    
    config_a = ConfigFactory.from_preset("development", {
        **base_overrides,
        "wandb.group": "variant-a",
        "environment.reward_on_submit_only": False
    })
    
    config_b = ConfigFactory.from_preset("development", {
        **base_overrides,
        "wandb.group": "variant-b", 
        "environment.reward_on_submit_only": True
    })
    
    return config_a, config_b
```

### Multi-Environment Configuration

```python
def create_multi_env_configs():
    """Create configurations for different environments."""
    configs = {}
    
    # ConceptARC configuration
    configs["conceptarc"] = ConfigFactory.from_preset("research", {
        "environment.max_episode_steps": 150,
        "environment.grid_size": (30, 30),
        "wandb.tags": ["conceptarc"]
    })
    
    # MiniARC configuration
    configs["miniarc"] = ConfigFactory.from_preset("development", {
        "environment.max_episode_steps": 50,
        "environment.grid_size": (5, 5),
        "wandb.tags": ["miniarc"]
    })
    
    # Standard ARC configuration
    configs["arc"] = ConfigFactory.from_preset("research", {
        "environment.max_episode_steps": 100,
        "environment.grid_size": (30, 30),
        "wandb.tags": ["arc-agi"]
    })
    
    return configs
```

## Extending the Configuration System

### Adding Custom Configuration Groups

```python
import equinox as eqx
from jaxtyping import Bool, Int, Float
from typing import Literal, Optional

class CustomConfig(eqx.Module):
    """Custom configuration group for specialized use cases."""
    
    # Custom parameters with type annotations
    custom_parameter: Float = 1.0
    custom_mode: Literal["mode_a", "mode_b", "mode_c"] = "mode_a"
    custom_enabled: Bool = True
    custom_threshold: Optional[Float] = None
    
    def validate(self) -> list[str]:
        """Validate custom configuration parameters."""
        errors = []
        
        if self.custom_parameter <= 0:
            errors.append("custom_parameter must be positive")
            
        if self.custom_threshold is not None and self.custom_threshold < 0:
            errors.append("custom_threshold must be non-negative")
            
        return errors

# Extend JaxArcConfig with custom group
class ExtendedJaxArcConfig(eqx.Module):
    """Extended configuration with custom group."""
    
    # Standard groups
    environment: EnvironmentConfig
    debug: DebugConfig
    visualization: VisualizationConfig
    storage: StorageConfig
    logging: LoggingConfig
    wandb: WandbConfig
    
    # Custom group
    custom: CustomConfig
    
    def validate(self):
        """Validate all configuration groups."""
        all_errors = []
        
        # Validate standard groups
        base_config = JaxArcConfig(
            environment=self.environment,
            debug=self.debug,
            visualization=self.visualization,
            storage=self.storage,
            logging=self.logging,
            wandb=self.wandb
        )
        base_validation = base_config.validate()
        all_errors.extend(base_validation.errors)
        
        # Validate custom group
        custom_errors = self.custom.validate()
        all_errors.extend([f"custom.{error}" for error in custom_errors])
        
        return ValidationResult(errors=all_errors, warnings=[])
```

### Custom Factory Functions

```python
class ExtendedConfigFactory:
    """Extended factory with custom configuration methods."""
    
    @staticmethod
    def create_custom_config(**overrides) -> ExtendedJaxArcConfig:
        """Create configuration with custom group."""
        base_config = ConfigFactory.create_development_config()
        
        custom_config = CustomConfig(
            custom_parameter=overrides.get("custom_parameter", 1.0),
            custom_mode=overrides.get("custom_mode", "mode_a"),
            custom_enabled=overrides.get("custom_enabled", True)
        )
        
        return ExtendedJaxArcConfig(
            environment=base_config.environment,
            debug=base_config.debug,
            visualization=base_config.visualization,
            storage=base_config.storage,
            logging=base_config.logging,
            wandb=base_config.wandb,
            custom=custom_config
        )
```

## Troubleshooting Configuration Issues

### Common Configuration Errors

```python
# Error: Missing configuration group
try:
    config = JaxArcConfig(
        environment=EnvironmentConfig(),
        debug=DebugConfig()
        # Missing other required groups
    )
except TypeError as e:
    print(f"Missing required configuration groups: {e}")

# Error: Invalid parameter values
try:
    config = JaxArcConfig(
        environment=EnvironmentConfig(max_episode_steps=-10),  # Invalid
        debug=DebugConfig(level="invalid"),  # Invalid
        # ... other groups
    )
except ConfigValidationError as e:
    print(f"Configuration validation failed: {e}")

# Error: Type mismatch
try:
    config = JaxArcConfig(
        environment=EnvironmentConfig(max_episode_steps="invalid"),  # Wrong type
        # ... other groups
    )
except TypeError as e:
    print(f"Type error: {e}")
```

### Debugging Configuration Issues

```python
def debug_configuration(config: JaxArcConfig):
    """Debug configuration issues."""
    
    # Check configuration structure
    print("Configuration structure:")
    for field_name in config.__dataclass_fields__:
        field_value = getattr(config, field_name)
        print(f"  {field_name}: {type(field_value).__name__}")
    
    # Validate configuration
    validation_result = config.validate()
    
    if validation_result.errors:
        print("\nConfiguration errors:")
        for error in validation_result.errors:
            print(f"  ❌ {error}")
    
    if validation_result.warnings:
        print("\nConfiguration warnings:")
        for warning in validation_result.warnings:
            print(f"  ⚠️  {warning}")
    
    if not validation_result.errors and not validation_result.warnings:
        print("\n✅ Configuration is valid")
    
    # Check parameter ranges
    print(f"\nParameter summary:")
    print(f"  Max episode steps: {config.environment.max_episode_steps}")
    print(f"  Debug level: {config.debug.level}")
    print(f"  Visualization enabled: {config.visualization.enabled}")
    print(f"  Storage policy: {config.storage.policy}")
```

## Performance Considerations

### Configuration Impact on Performance

```python
# High-performance configuration
high_perf_config = ConfigFactory.from_preset("production", {
    "debug.level": "off",           # No debug overhead
    "visualization.enabled": False,  # No visualization overhead
    "storage.policy": "none",       # No storage overhead
    "logging.level": "ERROR"        # Minimal logging
})

# Development configuration (moderate performance)
dev_config = ConfigFactory.from_preset("development", {
    "debug.level": "standard",      # Moderate debug overhead
    "visualization.enabled": True,   # Visualization overhead
    "storage.policy": "standard",   # Moderate storage overhead
    "logging.level": "INFO"         # Standard logging
})

# Research configuration (comprehensive but slower)
research_config = ConfigFactory.from_preset("research", {
    "debug.level": "research",      # Full debug overhead
    "visualization.level": "full",  # Full visualization overhead
    "storage.policy": "research",   # Full storage overhead
    "logging.level": "DEBUG"        # Verbose logging
})
```

### Memory Usage Guidelines

```python
# Memory-efficient configuration
memory_efficient = ConfigFactory.from_preset("production", {
    "environment.grid_size": (5, 5),    # Smaller grids
    "storage.compression": True,         # Compress stored data
    "storage.max_size_gb": 1.0,         # Limit storage
    "debug.save_episodes": False        # Don't save episodes
})

# Memory-intensive configuration
memory_intensive = ConfigFactory.from_preset("research", {
    "environment.grid_size": (30, 30),  # Large grids
    "storage.compression": False,        # No compression
    "storage.max_size_gb": 50.0,        # Large storage
    "debug.save_episodes": True         # Save all episodes
})
```

## Summary

### Key Recommendations

1. **Start with presets**: Use `ConfigFactory.create_*_config()` methods as starting points
2. **Use overrides**: Customize presets with specific parameter overrides
3. **Validate configurations**: Always call `config.validate()` before use
4. **Group related parameters**: Keep related settings in the same configuration group
5. **Document custom configurations**: Use YAML files for complex or reusable configurations
6. **Test configurations**: Verify configurations work with your specific use case
7. **Consider performance**: Choose appropriate debug and storage levels for your needs
8. **Use type hints**: Leverage the type system for better development experience

### Configuration Checklist

- [ ] Choose appropriate preset (development/research/production)
- [ ] Customize parameters using overrides or component construction
- [ ] Validate configuration with `config.validate()`
- [ ] Test configuration with actual environment
- [ ] Document configuration choices and rationale
- [ ] Consider performance implications of debug/storage settings
- [ ] Set up appropriate logging and monitoring
- [ ] Configure W&B integration if needed
- [ ] Plan for configuration versioning and reproducibility

By following these best practices, you'll be able to effectively use JaxARC's configuration system for a wide range of use cases while maintaining clarity, performance, and reproducibility.