# Config-Based API for JaxARC

This document describes the new config-based architecture for JaxARC environments, which provides better JAX compatibility, type safety, and Hydra integration.

## Overview

The config-based API replaces the traditional class-based environment approach with a functional API that uses typed configuration objects. This design is more JAX-friendly and provides better composability and testability.

### Key Benefits

- **JAX Compatibility**: Pure functional API that works seamlessly with `jax.jit`, `jax.vmap`, and other transformations
- **Type Safety**: Typed configuration dataclasses with validation
- **Hydra Integration**: Direct support for Hydra configuration management
- **Immutability**: Frozen dataclasses prevent accidental state mutations
- **Composability**: Modular configuration components that can be mixed and matched
- **Better Testing**: Easier to test with isolated, pure functions

## Quick Start

```python
import jax
from jaxarc.envs import arc_reset, arc_step, create_standard_config

# Create configuration
config = create_standard_config(max_episode_steps=100, success_bonus=10.0)

# Initialize environment
key = jax.random.PRNGKey(42)
state, observation = arc_reset(key, config)

# Take action
action = {
    "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
    "operation": jnp.array(1, dtype=jnp.int32),  # Fill with color 1
}
new_state, obs, reward, done, info = arc_step(state, action, config)
```

## Configuration Classes

### ArcEnvConfig
Main configuration class containing all environment settings:

```python
from jaxarc.envs import ArcEnvConfig, RewardConfig, GridConfig, ActionConfig

config = ArcEnvConfig(
    max_episode_steps=100,
    auto_reset=True,
    log_operations=True,
    reward=RewardConfig(success_bonus=15.0),
    grid=GridConfig(max_grid_height=25),
    action=ActionConfig(action_format="point"),
)
```

### RewardConfig
Controls reward calculation:

```python
reward_config = RewardConfig(
    reward_on_submit_only=True,  # Only give rewards on submit action
    step_penalty=-0.01,          # Penalty per step
    success_bonus=10.0,          # Bonus for solving task
    similarity_weight=1.0,       # Weight for similarity improvement
    progress_bonus=0.1,          # Bonus for making progress
)
```

### GridConfig
Manages grid dimensions and constraints:

```python
grid_config = GridConfig(
    max_grid_height=30,
    max_grid_width=30,
    min_grid_height=3,
    min_grid_width=3,
    max_colors=10,
    background_color=0,
)
```

### ActionConfig
Defines action space and validation:

```python
action_config = ActionConfig(
    action_format="selection_operation",  # "selection_operation", "point", "bbox"
    selection_threshold=0.5,              # Threshold for continuous->discrete
    num_operations=35,                    # Number of available operations
    validate_actions=True,                # Enable action validation
    clip_invalid_actions=True,            # Clip invalid ops to valid range
)
```

## Factory Functions

### Standard Presets

```python
from jaxarc.envs import (
    create_raw_config,        # Minimal settings
    create_standard_config,   # Balanced for training
    create_full_config,       # All features enabled
    create_point_config,      # Point-based actions
    create_bbox_config,       # Bounding box actions
    create_restricted_config, # Limited action space
)

# Quick access to presets
from jaxarc.envs import get_preset_config
config = get_preset_config("standard", max_episode_steps=150)
```

### Training Configurations

```python
from jaxarc.envs import create_training_config

# Curriculum learning configs
basic_config = create_training_config("basic")      # Simple operations only
standard_config = create_training_config("standard") # Full operations
advanced_config = create_training_config("advanced") # All features
expert_config = create_training_config("expert")     # High performance
```

### Evaluation Configuration

```python
from jaxarc.envs import create_evaluation_config

eval_config = create_evaluation_config(
    strict_mode=True  # Enable strict validation for evaluation
)
```

## Hydra Integration

### Direct Hydra Support

```python
from omegaconf import OmegaConf
from jaxarc.envs import arc_reset, ArcEnvConfig

# Create Hydra config
hydra_config = OmegaConf.create({
    "max_episode_steps": 100,
    "reward": {"success_bonus": 15.0},
    "action": {"action_format": "point"},
})

# Use directly with functional API
state, obs = arc_reset(key, hydra_config)

# Or convert to typed config
typed_config = ArcEnvConfig.from_hydra(hydra_config)
```

### Configuration Files

Create `config.yaml`:
```yaml
env:
  max_episode_steps: 100
  reward:
    reward_on_submit_only: true
    success_bonus: 10.0
    step_penalty: -0.01
  grid:
    max_grid_height: 30
    max_grid_width: 30
  action:
    action_format: "selection_operation"
    num_operations: 35
```

Use with Hydra:
```python
import hydra
from omegaconf import DictConfig
from jaxarc.envs import arc_reset, arc_step

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    key = jax.random.PRNGKey(42)
    state, obs = arc_reset(key, cfg.env)
    # ... rest of your code
```

## Action Formats

### Selection-Operation (Default)

```python
action = {
    "selection": jnp.array([[True, False], [False, True]], dtype=jnp.bool_),
    "operation": jnp.array(1, dtype=jnp.int32),  # Fill with color 1
}
```

### Point-Based Actions

```python
config = create_point_config()
action = {
    "point": (row, col),  # Single point coordinates
    "operation": jnp.array(2, dtype=jnp.int32),
}
```

### Bounding Box Actions

```python
config = create_bbox_config()
action = {
    "bbox": (row1, col1, row2, col2),  # Rectangle coordinates
    "operation": jnp.array(3, dtype=jnp.int32),
}
```

## JAX Compatibility

### JIT Compilation

```python
# Mark config as static for JIT
@jax.jit
def jitted_step(state, action, config):
    return arc_step(state, action, config)

# Or use static_argnums
jitted_step = jax.jit(arc_step, static_argnums=(2,))
```

### Batch Processing

```python
def single_episode(key):
    state, obs = arc_reset(key, config)
    # ... episode logic
    return final_reward

# Process multiple episodes in parallel
keys = jax.random.split(key, batch_size)
rewards = jax.vmap(single_episode)(keys)
```

### Training Loop Integration

```python
@jax.jit
def training_step(state, action, config):
    new_state, obs, reward, done, info = arc_step(state, action, config)
    return new_state, reward

# This can be used in JAX-based training loops
```

## Migration from Class-Based API

### Old Way (Class-Based)

```python
from jaxarc.envs import ArcEnvironment

env = ArcEnvironment(env_config, dataset_config)
state, obs = env.reset(key)
new_state, obs, reward, done, info = env.step(state, action)
```

### New Way (Config-Based)

```python
from jaxarc.envs import arc_reset, arc_step, create_standard_config

config = create_standard_config()
state, obs = arc_reset(key, config)
new_state, obs, reward, done, info = arc_step(state, action, config)
```

### Gradual Migration

Both APIs coexist, so you can migrate gradually:

1. Start using factory functions for configuration
2. Replace `env.reset()` with `arc_reset()`
3. Replace `env.step()` with `arc_step()`
4. Add JAX transformations as needed

## Configuration Validation

```python
from jaxarc.envs.config import validate_config, get_config_summary

# Validate configuration consistency
validate_config(config)

# Get human-readable summary
summary = get_config_summary(config)
print(summary)
```

## Advanced Usage

### Custom Configuration

```python
custom_config = ArcEnvConfig(
    max_episode_steps=200,
    reward=RewardConfig(
        reward_on_submit_only=False,
        step_penalty=-0.005,
        success_bonus=25.0,
        similarity_weight=2.0,
    ),
    grid=GridConfig(
        max_grid_height=20,
        max_colors=8,
    ),
    action=ActionConfig(
        action_format="bbox",
        selection_threshold=0.7,
    ),
)
```

### Configuration Merging

```python
from jaxarc.envs.config import merge_configs

base_config = create_standard_config()
override_config = OmegaConf.create({
    "max_episode_steps": 150,
    "reward": {"success_bonus": 20.0},
})

merged_config = merge_configs(base_config, override_config)
```

### Configuration Serialization

```python
# Convert to dictionary
config_dict = config.to_dict()

# Create from dictionary
from jaxarc.envs.config import config_from_dict
restored_config = config_from_dict(config_dict)
```

## Examples

See `examples/config_api_demo.py` for comprehensive examples including:

- Basic usage patterns
- Hydra integration
- Different action formats
- JAX compatibility demonstrations
- Preset configurations
- Manual configuration creation

## Best Practices

1. **Use factory functions** for common configurations instead of manual creation
2. **Mark configs as static** when using JAX transformations
3. **Validate configurations** before use in production
4. **Use typed configs** for better IDE support and error catching
5. **Leverage Hydra** for configuration management in larger projects
6. **Test with frozen configs** to ensure immutability
7. **Use appropriate action formats** for your specific use case

## Troubleshooting

### Common Issues

1. **"Non-hashable static arguments"**: Make sure configs are frozen dataclasses
2. **"Selection shape mismatch"**: Ensure selection shape matches grid shape
3. **"Invalid operation"**: Check operation is in valid range [0, num_operations)
4. **"Config field assignment"**: Configs are frozen - create new instances instead

### Debug Tips

- Enable logging with `log_operations=True` in config
- Use `get_config_summary()` to inspect configuration
- Check `info` dict returned by `arc_step()` for detailed metrics
- Use `validate_config()` to catch configuration issues early