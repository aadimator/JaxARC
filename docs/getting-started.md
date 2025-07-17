# Getting Started with JaxARC

Welcome to JaxARC! This guide will get you up and running with the JAX-based ARC
environment in just a few minutes.

## Installation

### Quick Install

```bash
pip install jaxarc
```

### Development Install

For development or to access the latest features:

```bash
git clone https://github.com/aadimator/JaxARC.git
cd JaxARC
pixi shell  # Activate project environment
pixi run -e dev pre-commit install  # Set up pre-commit hooks
```

## Download Your First Dataset

JaxARC supports multiple ARC datasets. Let's start with MiniARC for quick
experimentation:

```bash
# Download MiniARC (400+ tasks, 5x5 grids - perfect for getting started)
python scripts/download_dataset.py miniarc

# Or download the full ARC-AGI dataset
python scripts/download_dataset.py arc-agi-2
```

The datasets will be downloaded to `data/raw/` with proper organization.

## Your First Example

Here's a complete working example to get you started:

```python
import jax
import jax.numpy as jnp
from jaxarc.envs import arc_reset, arc_step, create_standard_config
from jaxarc.parsers import MiniArcParser
from jaxarc.utils.visualization import log_grid_to_console
from omegaconf import DictConfig

# 1. Create parser configuration
parser_config = DictConfig(
    {
        "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
        "grid": {"max_grid_height": 5, "max_grid_width": 5},
        "max_train_pairs": 3,
        "max_test_pairs": 1,
    }
)

# 2. Load a random task
parser = MiniArcParser(parser_config)
key = jax.random.PRNGKey(42)
task = parser.get_random_task(key)

print(f"Loaded task with {task.num_train_pairs} training pairs")

# 3. Create environment configuration
env_config = create_standard_config(
    max_episode_steps=50, success_bonus=10.0, log_operations=True
)

# 4. Initialize environment with the task
key, reset_key = jax.random.split(key)
state, observation = arc_reset(reset_key, env_config, task)

print("Initial working grid:")
log_grid_to_console(state.working_grid)

# 5. Take an action - fill selection with color 1
action = {
    "selection": jnp.ones((2, 2), dtype=jnp.bool_),  # Select 2x2 area
    "operation": jnp.array(1, dtype=jnp.int32),  # Fill with color 1
}

# 6. Step the environment
state, observation, reward, done, info = arc_step(state, action, env_config)

print(f"Reward: {reward:.3f}")
print(f"Done: {done}")
print(f"Similarity to target: {info['similarity']:.3f}")

print("Updated working grid:")
log_grid_to_console(state.working_grid)
```

## Common Usage Patterns

### Working with Different Datasets

```python
# MiniARC - Fast experimentation (5x5 grids)
from jaxarc.parsers import MiniArcParser

parser = MiniArcParser(miniarc_config)

# ConceptARC - Systematic evaluation (16 concept groups)
from jaxarc.parsers import ConceptArcParser

parser = ConceptArcParser(conceptarc_config)
task = parser.get_random_task_from_concept("Center", key)

# ARC-AGI - Full challenge dataset
from jaxarc.parsers import ArcAgiParser

parser = ArcAgiParser(arc_agi_config)
```

### JAX Transformations for Speed

```python
# JIT compile for 100x+ speedup
@jax.jit
def fast_step(state, action, config):
    return arc_step(state, action, config)


# Batch processing multiple environments
def run_episode(key):
    state, obs = arc_reset(key, config, task)
    # ... episode logic
    return total_reward


keys = jax.random.split(key, 100)  # 100 parallel episodes
rewards = jax.vmap(run_episode)(keys)
```

### Configuration Presets

```python
from jaxarc.envs import (
    create_raw_config,  # Minimal operations
    create_standard_config,  # Balanced for training
    create_full_config,  # All 35 operations
    create_point_config,  # Point-based actions
    create_bbox_config,  # Bounding box actions
)

# Quick setup for different use cases
config = create_standard_config(max_episode_steps=100)
```

## Next Steps

Now that you have JaxARC running, explore these resources:

- **[Datasets Guide](datasets.md)** - Learn about all supported datasets and
  their use cases
- **[Configuration Guide](configuration.md)** - Master the configuration system
  and action formats
- **[API Reference](api_reference.md)** - Complete API documentation
- **[Examples](examples/)** - Working examples for different scenarios

### Example Scripts

Try these example scripts to see JaxARC in action:

```bash
# Basic usage demonstration
pixi run python examples/config_api_demo.py

# ConceptARC exploration
pixi run python examples/conceptarc_usage_example.py --concept Center

# MiniARC rapid prototyping
pixi run python examples/miniarc_usage_example.py --performance-comparison

# Visualization utilities
pixi run python examples/visualization_demo.py
```

## Troubleshooting

**Dataset not found?**

- Run the download script: `python scripts/download_dataset.py <dataset-name>`
- Check that files exist in `data/raw/<dataset>/`

**Import errors?**

- Make sure you're in the pixi environment: `pixi shell`
- For development installs, ensure you're in the project directory

**Performance issues?**

- Use JIT compilation: `@jax.jit` decorator on your functions
- Start with MiniARC for faster iteration
- Use batch processing with `jax.vmap` for multiple environments

**Need help?** Check the [Configuration Guide](configuration.md) for detailed
troubleshooting or open an issue on GitHub.
