# Basic Usage Examples

This document contains core functionality examples for JaxARC, covering environment setup, task loading, and basic action patterns.

## Environment Setup and Configuration

### Creating Standard Configuration

```python
import jax
from jaxarc.envs import create_standard_config, arc_reset, arc_step

# Create a balanced configuration for training
config = create_standard_config(
    max_episode_steps=100,
    success_bonus=10.0,
    step_penalty=-0.01,
    log_operations=True
)

print(f"Max steps: {config.max_episode_steps}")
print(f"Available operations: {len(config.action_handler.operations)}")
```

### Configuration Presets

```python
from jaxarc.envs import (
    create_raw_config,      # Minimal operations (fill colors, resize, submit)
    create_standard_config, # Balanced for training (+ flood fill, clipboard)
    create_full_config,     # All 35 operations
    create_point_config,    # Point-based actions
    create_bbox_config,     # Bounding box actions
)

# Compare different configurations
configs = {
    "raw": create_raw_config(),
    "standard": create_standard_config(),
    "full": create_full_config(),
}

for name, config in configs.items():
    print(f"{name}: {len(config.action_handler.operations)} operations")
```

## Task Loading and Parsing

### Loading MiniARC Tasks

```python
import jax
from jaxarc.parsers import MiniArcParser
from omegaconf import DictConfig

# Create parser configuration
parser_config = DictConfig({
    "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
    "grid": {"max_grid_height": 5, "max_grid_width": 5},
    "max_train_pairs": 3,
    "max_test_pairs": 1,
})

# Initialize parser and load task
parser = MiniArcParser(parser_config)
key = jax.random.PRNGKey(42)
task = parser.get_random_task(key)

print(f"Task has {task.num_train_pairs} training pairs")
print(f"Input grid shape: {task.train_input_grids.shape}")
print(f"Output grid shape: {task.train_output_grids.shape}")
```

### Loading ARC-AGI Tasks

```python
from jaxarc.parsers import ArcAgiParser

# Create ARC-AGI configuration
parser_config = DictConfig({
    "challenges": {"path": "data/raw/ARC-AGI-2/data/training"},
    "solutions": {"path": "data/raw/ARC-AGI-2/data/training"},
    "grid": {"max_grid_height": 30, "max_grid_width": 30},
    "max_train_pairs": 4,
    "max_test_pairs": 1,
})

parser = ArcAgiParser(parser_config)
task = parser.get_random_task(key)
print(f"Full ARC task loaded with {task.num_train_pairs} training pairs")
```

## Basic Action Patterns

### Selection-Based Actions (Default)

```python
import jax.numpy as jnp
from jaxarc.utils.visualization import log_grid_to_console

# Initialize environment with task
config = create_standard_config(max_episode_steps=50)
key = jax.random.PRNGKey(42)
state, observation = arc_reset(key, config, task)

print("Initial working grid:")
log_grid_to_console(state.working_grid)

# Create selection mask and apply operation
action = {
    "selection": jnp.array([
        [True, True, False, False, False],
        [True, True, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False]
    ], dtype=jnp.bool_),
    "operation": jnp.array(1, dtype=jnp.int32),  # Fill with color 1
}

# Step environment
state, obs, reward, done, info = arc_step(state, action, config)
print(f"Reward: {reward:.3f}, Done: {done}")
print(f"Similarity to target: {info['similarity']:.3f}")

print("Updated working grid:")
log_grid_to_console(state.working_grid)
```

### Point-Based Actions

```python
# Create point-based configuration
point_config = create_point_config(max_episode_steps=50)
state, obs = arc_reset(key, point_config, task)

# Point-based action
point_action = {
    "point": (1, 1),  # Row 1, Column 1
    "operation": jnp.array(2, dtype=jnp.int32),  # Fill with color 2
}

state, obs, reward, done, info = arc_step(state, point_action, point_config)
print(f"Point action reward: {reward:.3f}")
```

### Bounding Box Actions

```python
# Create bounding box configuration
bbox_config = create_bbox_config(max_episode_steps=50)
state, obs = arc_reset(key, bbox_config, task)

# Bounding box action
bbox_action = {
    "bbox": (0, 0, 2, 2),  # Top-left (0,0) to bottom-right (2,2)
    "operation": jnp.array(3, dtype=jnp.int32),  # Fill with color 3
}

state, obs, reward, done, info = arc_step(state, bbox_action, bbox_config)
print(f"Bounding box action reward: {reward:.3f}")
```

## JAX Transformations and JIT Compilation

### JIT Compilation for Performance

```python
import time

# JIT compile the step function for massive speedup
@jax.jit
def jitted_step(state, action, config):
    return arc_step(state, action, config)

# Compare performance
action = {
    "selection": jnp.ones((2, 2), dtype=jnp.bool_),
    "operation": jnp.array(1, dtype=jnp.int32),
}

# Regular step (first call includes compilation time)
start = time.time()
state, obs, reward, done, info = jitted_step(state, action, config)
compile_time = time.time() - start

# JIT compiled step (subsequent calls are fast)
start = time.time()
state, obs, reward, done, info = jitted_step(state, action, config)
jit_time = time.time() - start

print(f"First call (with compilation): {compile_time*1000:.2f}ms")
print(f"JIT compiled call: {jit_time*1000:.2f}ms")
print(f"Speedup: {compile_time/jit_time:.0f}x")
```

### Batch Processing with vmap

```python
# Process multiple environments in parallel
def single_episode(episode_key):
    """Run a single episode and return total reward."""
    state, obs = arc_reset(episode_key, config, task)
    total_reward = 0.0
    
    for step in range(10):  # Short episode
        action = {
            "selection": jnp.ones((1, 1), dtype=jnp.bool_),
            "operation": jnp.array(1, dtype=jnp.int32),
        }
        state, obs, reward, done, info = arc_step(state, action, config)
        total_reward += reward
        if done:
            break
    
    return total_reward

# Create batch of random keys
batch_size = 100
keys = jax.random.split(key, batch_size)

# Process all episodes in parallel
batch_rewards = jax.vmap(single_episode)(keys)
print(f"Processed {batch_size} episodes")
print(f"Average reward: {jnp.mean(batch_rewards):.3f}")
print(f"Reward std: {jnp.std(batch_rewards):.3f}")
```

## Error Handling and Validation

### Validating Actions

```python
# Invalid action example
try:
    invalid_action = {
        "selection": jnp.ones((10, 10), dtype=jnp.bool_),  # Too large
        "operation": jnp.array(1, dtype=jnp.int32),
    }
    state, obs, reward, done, info = arc_step(state, invalid_action, config)
except Exception as e:
    print(f"Action validation error: {e}")

# Valid action with proper bounds checking
grid_height, grid_width = state.working_grid.shape
selection = jnp.zeros((grid_height, grid_width), dtype=jnp.bool_)
selection = selection.at[0:2, 0:2].set(True)  # Safe indexing

valid_action = {
    "selection": selection,
    "operation": jnp.array(1, dtype=jnp.int32),
}
state, obs, reward, done, info = arc_step(state, valid_action, config)
print("Valid action executed successfully")
```

### Configuration Validation

```python
from jaxarc.envs.config import ArcEnvConfig
import chex

# Create configuration with validation
try:
    config = ArcEnvConfig(
        max_episode_steps=-10,  # Invalid negative value
        success_bonus=10.0,
    )
except Exception as e:
    print(f"Configuration validation error: {e}")

# Valid configuration
config = ArcEnvConfig(
    max_episode_steps=100,
    success_bonus=10.0,
    step_penalty=-0.01,
)
print("Configuration created successfully")
```

## Next Steps

- **[ConceptARC Examples](conceptarc-examples.md)**: Learn ConceptARC-specific patterns
- **[MiniARC Examples](miniarc-examples.md)**: Explore rapid prototyping workflows  
- **[Advanced Patterns](advanced-patterns.md)**: Master JAX transformations and batch processing
- **[Configuration Guide](../configuration.md)**: Complete configuration system documentation