# Configuration Guide

JaxARC provides a comprehensive configuration system that supports both programmatic and file-based configuration management. This guide covers all aspects of configuring environments, actions, rewards, and datasets.

## Quick Start

```python
import jax
from jaxarc.envs import arc_reset, arc_step, create_standard_config

# Create configuration with sensible defaults
config = create_standard_config(max_episode_steps=100, success_bonus=10.0)

# Use with functional API
key = jax.random.PRNGKey(42)
state, observation = arc_reset(key, config)

# Take action
action = {
    "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
    "operation": jnp.array(1, dtype=jnp.int32),  # Fill with color 1
}
new_state, obs, reward, done, info = arc_step(state, action, config)
```

## Configuration Architecture

### Core Configuration Classes

JaxARC uses typed configuration dataclasses for type safety and validation:

```python
from jaxarc.envs import ArcEnvConfig, RewardConfig, GridConfig, ActionConfig

# Main configuration containing all settings
config = ArcEnvConfig(
    max_episode_steps=100,
    auto_reset=True,
    log_operations=True,
    reward=RewardConfig(success_bonus=15.0),
    grid=GridConfig(max_grid_height=25),
    action=ActionConfig(selection_format="point"),
)
```

### Configuration Components

#### RewardConfig
Controls reward calculation and learning signals:

```python
reward_config = RewardConfig(
    reward_on_submit_only=True,  # Only give rewards on submit action
    step_penalty=-0.01,          # Penalty per step
    success_bonus=10.0,          # Bonus for solving task
    similarity_weight=1.0,       # Weight for similarity improvement
    progress_bonus=0.1,          # Bonus for making progress
    invalid_action_penalty=-0.5, # Penalty for invalid actions
)
```

#### GridConfig
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

#### ActionConfig
Defines action space and validation:

```python
action_config = ActionConfig(
    selection_format="mask",        # "mask", "point", "bbox"
    selection_threshold=0.5,        # Threshold for continuous->discrete
    num_operations=35,              # Number of available operations
    validate_actions=True,          # Enable action validation
    clip_invalid_actions=True,      # Clip invalid ops to valid range
    allowed_operations=[0,1,2,3,4,5,6,7,8,9,33,34],  # Restrict operations
)
```

## Factory Functions

### Standard Presets

Factory functions provide quick access to common configurations:

```python
from jaxarc.envs import (
    create_raw_config,        # Minimal settings (operations 0-9, 33-34)
    create_standard_config,   # Balanced for training (no object ops)
    create_full_config,       # All features enabled (all 35 operations)
    create_point_config,      # Point-based actions
    create_bbox_config,       # Bounding box actions
    create_restricted_config, # Limited action space
)

# Quick setup with customization
config = create_standard_config(
    max_episode_steps=150,
    success_bonus=20.0,
    step_penalty=-0.005
)
```

### Training Configurations

Specialized configurations for different training phases:

```python
from jaxarc.envs import create_training_config

# Curriculum learning configurations
basic_config = create_training_config("basic")      # Simple operations only
standard_config = create_training_config("standard") # Full operations
advanced_config = create_training_config("advanced") # All features
expert_config = create_training_config("expert")     # High performance
```

### Dataset-Specific Configurations

Optimized configurations for different datasets:

```python
from jaxarc.envs import create_conceptarc_config, create_miniarc_config

# ConceptARC configuration for concept-based evaluation
conceptarc_config = create_conceptarc_config(
    max_episode_steps=150,
    task_split="corpus",
    success_bonus=20.0,
    step_penalty=-0.01
)

# MiniARC configuration for rapid prototyping
miniarc_config = create_miniarc_config(
    max_episode_steps=50,    # Shorter episodes for rapid iteration
    success_bonus=5.0,       # Quick feedback
    step_penalty=-0.001,     # Lower penalty for experimentation
)
```

## Action Formats

### Selection-Operation (Default)

The default action format uses a selection mask and operation ID:

```python
config = create_standard_config()  # Uses mask selection by default

action = {
    "selection": jnp.array([[True, False], [False, True]], dtype=jnp.bool_),
    "operation": jnp.array(1, dtype=jnp.int32),  # Fill with color 1
}
```

### Point-Based Actions

Point-based actions specify a single coordinate:

```python
config = create_point_config()

action = {
    "point": (row, col),  # Single point coordinates
    "operation": jnp.array(2, dtype=jnp.int32),
}
```

### Bounding Box Actions

Bounding box actions specify a rectangular region:

```python
config = create_bbox_config()

action = {
    "bbox": (row1, col1, row2, col2),  # Rectangle coordinates
    "operation": jnp.array(3, dtype=jnp.int32),
}
```

## Environment Types

### Raw Environment
Minimal action set with only basic operations:

```python
config = create_raw_config()
# Available operations: Fill colors 0-9, resize grid, submit solution
```

### Standard Environment
Standard action set excluding object-based operations:

```python
config = create_standard_config()
# Available operations: Fill colors, flood fill, clipboard ops, grid ops, submit
```

### Full Environment
Complete action set including all object-based operations:

```python
config = create_full_config()
# Available operations: All standard actions plus movement, rotation, flipping
```

## Hydra Integration

### Direct Hydra Support

JaxARC integrates seamlessly with Hydra configuration management:

```python
from omegaconf import OmegaConf
from jaxarc.envs import arc_reset, ArcEnvConfig

# Create Hydra config
hydra_config = OmegaConf.create({
    "max_episode_steps": 100,
    "reward": {"success_bonus": 15.0},
    "action": {"selection_format": "point"},
})

# Use directly with functional API
state, obs = arc_reset(key, hydra_config)

# Or convert to typed config
typed_config = ArcEnvConfig.from_hydra(hydra_config)
```

### Configuration Files

Create hierarchical configuration files:

```yaml
# conf/config.yaml
defaults:
  - environment: arc_env
  - reward: standard
  - action: standard
  - dataset: arc_agi_1

seed: 42

environment:
  max_episode_steps: 100
  auto_reset: true
  log_operations: false

reward:
  reward_on_submit_only: true
  success_bonus: 10.0
  step_penalty: -0.01

action:
  selection_format: "mask"
  num_operations: 35
  validate_actions: true
```

### Using with Hydra

```python
import hydra
from omegaconf import DictConfig
from jaxarc.envs.factory import create_complete_hydra_config
from jaxarc.envs.functional import arc_reset, arc_step

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Create complete configuration from Hydra
    env_config = create_complete_hydra_config(cfg)
    
    key = jax.random.PRNGKey(cfg.seed)
    state, obs = arc_reset(key, env_config)
    # ... rest of your code

if __name__ == "__main__":
    main()
```

### Configuration Overrides

Use Hydra's override system for experimentation:

```bash
# Override specific settings
python script.py environment.max_episode_steps=50
python script.py environment.reward.success_bonus=20.0
python script.py environment.action.selection_format=point

# Use different presets
python script.py environment=training
python script.py environment=evaluation
python script.py action=point
python script.py reward=training
```

## JAX Compatibility

### JIT Compilation

Configurations work seamlessly with JAX transformations:

```python
# Mark config as static for JIT
@jax.jit
def jitted_step(state, action, config):
    return arc_step(state, action, config)

# Or use static_argnums
jitted_step = jax.jit(arc_step, static_argnums=(2,))

# Use in training loops
@jax.jit
def training_step(state, action, config):
    new_state, obs, reward, done, info = arc_step(state, action, config)
    return new_state, reward
```

### Batch Processing

Process multiple environments in parallel:

```python
def single_episode(key, config):
    state, obs = arc_reset(key, config)
    # ... episode logic
    return final_reward

# Process multiple episodes in parallel
keys = jax.random.split(key, batch_size)
configs = [config] * batch_size  # Same config for all
rewards = jax.vmap(single_episode)(keys, configs)
```

## Advanced Configuration

### Custom Configuration

Create fully custom configurations:

```python
custom_config = ArcEnvConfig(
    max_episode_steps=200,
    auto_reset=False,
    log_operations=True,
    reward=RewardConfig(
        reward_on_submit_only=False,
        step_penalty=-0.005,
        success_bonus=25.0,
        similarity_weight=2.0,
        progress_bonus=0.5,
        invalid_action_penalty=-0.1,
    ),
    grid=GridConfig(
        max_grid_height=20,
        max_grid_width=20,
        max_colors=8,
    ),
    action=ActionConfig(
        selection_format="bbox",
        selection_threshold=0.7,
        num_operations=20,
        allowed_operations=list(range(20)),
    ),
)
```

### Configuration Merging

Combine configurations programmatically:

```python
from jaxarc.envs.config import merge_configs

base_config = create_standard_config()
override_config = OmegaConf.create({
    "max_episode_steps": 150,
    "reward": {"success_bonus": 20.0},
})

merged_config = merge_configs(base_config, override_config)
```

### Configuration Validation

Validate configurations before use:

```python
from jaxarc.envs.config import validate_config, get_config_summary

# Validate configuration consistency
validate_config(config)

# Get human-readable summary
summary = get_config_summary(config)
print(summary)
```

## Practical Examples

### Training Configuration

```python
# Dense rewards for better learning signals
training_config = ArcEnvConfig(
    max_episode_steps=200,
    reward=RewardConfig(
        reward_on_submit_only=False,  # Dense rewards
        step_penalty=-0.005,          # Smaller penalty
        success_bonus=20.0,           # Higher bonus
        similarity_weight=2.0,        # Higher similarity weight
        progress_bonus=0.5,           # Larger progress bonus
        invalid_action_penalty=-0.1,  # Smaller penalty for exploration
    ),
    action=ActionConfig(
        selection_format="mask",
        validate_actions=True,
        clip_invalid_actions=True,
    ),
)
```

### Evaluation Configuration

```python
# Sparse rewards for realistic evaluation
eval_config = ArcEnvConfig(
    max_episode_steps=100,
    reward=RewardConfig(
        reward_on_submit_only=True,   # Only final reward
        step_penalty=-0.01,           # Standard penalty
        success_bonus=10.0,           # Standard bonus
        similarity_weight=1.0,        # Standard weight
        progress_bonus=0.0,           # No progress bonus
        invalid_action_penalty=-1.0,  # Higher penalty for mistakes
    ),
    action=ActionConfig(
        selection_format="mask",
        validate_actions=True,
        clip_invalid_actions=False,   # Strict validation
    ),
)
```

### Rapid Prototyping Configuration

```python
# Quick iteration with MiniARC
prototype_config = create_miniarc_config(
    max_episode_steps=30,
    success_bonus=5.0,
    step_penalty=-0.001,
    log_operations=True,  # Enable debugging
)
```

## Common Issues and Solutions

### Configuration Errors

**"Non-hashable static arguments"**
```python
# Problem: Config not frozen for JIT
config = ArcEnvConfig(...)  # Not frozen

# Solution: Ensure configs are frozen dataclasses
config = create_standard_config()  # Already frozen
```

**"Selection shape mismatch"**
```python
# Problem: Selection doesn't match grid shape
action = {"selection": wrong_shape_mask, "operation": 1}

# Solution: Ensure selection matches working grid
selection = jnp.ones_like(state.working_grid, dtype=jnp.bool_)
action = {"selection": selection, "operation": 1}
```

**"Invalid operation"**
```python
# Problem: Operation outside valid range
action = {"selection": mask, "operation": 50}  # Invalid

# Solution: Check allowed operations
config = create_standard_config()
valid_ops = config.action.allowed_operations
action = {"selection": mask, "operation": valid_ops[0]}
```

### Dataset Configuration Issues

**"Legacy Kaggle format detected"**
```python
# Problem: Using old Kaggle format
config = DictConfig({
    "training": {
        "challenges": "path/to/challenges.json",  # Old format
        "solutions": "path/to/solutions.json"
    }
})

# Solution: Update to GitHub format
config = DictConfig({
    "training": {"path": "data/raw/ARC-AGI-1/data/training"},
    "evaluation": {"path": "data/raw/ARC-AGI-1/data/evaluation"}
})
```

**"Dataset not found"**
```bash
# Problem: Dataset not downloaded
# Solution: Download required dataset
python scripts/download_dataset.py arc-agi-1
python scripts/download_dataset.py arc-agi-2
python scripts/download_dataset.py mini-arc
```

### Performance Issues

**"Slow JIT compilation"**
```python
# Problem: Config not marked as static
@jax.jit
def slow_step(state, action, config):
    return arc_step(state, action, config)

# Solution: Mark config as static
@jax.jit
def fast_step(state, action, config):
    return arc_step(state, action, config)

fast_step = jax.jit(arc_step, static_argnums=(2,))
```

**"Memory issues with large grids"**
```python
# Problem: Grid too large for available memory
config = ArcEnvConfig(grid=GridConfig(max_grid_height=100))

# Solution: Use appropriate grid sizes
config = create_standard_config()  # Uses reasonable defaults
# Or customize: GridConfig(max_grid_height=30, max_grid_width=30)
```

## Best Practices

1. **Use factory functions** for common configurations instead of manual creation
2. **Mark configs as static** when using JAX transformations with `static_argnums`
3. **Validate configurations** before use in production with `validate_config()`
4. **Use typed configs** for better IDE support and error catching
5. **Leverage Hydra** for configuration management in larger projects
6. **Test with frozen configs** to ensure immutability
7. **Use appropriate action formats** for your specific use case
8. **Start with presets** and customize only what you need
9. **Use dataset-specific configs** for optimal performance
10. **Enable logging** during development, disable for production

## Migration Guide

### From Class-Based API

```python
# Old way (Class-Based)
from jaxarc.envs import ArcEnvironment

env = ArcEnvironment(env_config, dataset_config)
state, obs = env.reset(key)
new_state, obs, reward, done, info = env.step(state, action)

# New way (Config-Based)
from jaxarc.envs import arc_reset, arc_step, create_standard_config

config = create_standard_config()
state, obs = arc_reset(key, config)
new_state, obs, reward, done, info = arc_step(state, action, config)
```

### Gradual Migration Strategy

1. Start using factory functions for configuration
2. Replace `env.reset()` with `arc_reset()`
3. Replace `env.step()` with `arc_step()`
4. Add JAX transformations as needed
5. Migrate to Hydra configuration files

Both APIs coexist, allowing gradual migration without breaking existing code.

## Examples

Complete examples are available in the `examples/` directory:

- `config_api_demo.py` - Comprehensive configuration examples
- `hydra_integration_example.py` - Hydra configuration management
- `advanced_config_demo.py` - Advanced configuration patterns

Run examples with:
```bash
pixi run python examples/config_api_demo.py
pixi run python examples/hydra_integration_example.py
```