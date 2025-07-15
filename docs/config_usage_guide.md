# JaxARC Configuration Usage Guide

This guide explains how to use the new structured configuration system in
JaxARC, which provides a clean and flexible way to configure environments,
rewards, actions, and datasets.

## Overview

The new configuration system is organized hierarchically:

```
conf/
├── config.yaml              # Main config with defaults
├── environment/             # Environment configurations
│   ├── arc_env.yaml        # Standard environment
│   ├── raw.yaml            # Minimal action set
│   ├── training.yaml       # Training-optimized
│   ├── evaluation.yaml     # Evaluation-optimized
│   └── full.yaml           # Complete action set
├── reward/                 # Reward configurations
│   ├── standard.yaml       # Standard rewards
│   ├── training.yaml       # Training-optimized
│   └── evaluation.yaml     # Evaluation-optimized
├── action/                 # Action space configurations
│   ├── raw.yaml            # Minimal actions (fill colors + submit)
│   ├── standard.yaml       # Standard actions (no object ops)
│   ├── full.yaml           # All actions including object ops
│   ├── point.yaml          # Point-based actions
│   └── bbox.yaml           # Bounding box actions
└── dataset/               # Dataset configurations (with grid configs)
    ├── arc_agi_1.yaml     # ARC-AGI-1 dataset
    ├── arc_agi_2.yaml     # ARC-AGI-2 dataset
    └── mini_arc.yaml      # Mini ARC for testing
```

## Environment Types

### Raw Environment

Minimal action set with only basic fill colors, resize, and submit operations:

```bash
python script.py environment=raw
```

**Actions available:**

- Fill colors 0-9 (operations 0-9)
- Resize grid (operation 33)
- Submit solution (operation 34)

### Standard Environment

Standard action set excluding object-based operations:

```bash
python script.py environment=arc_env  # or just use default
```

**Actions available:**

- Fill colors 0-9 (operations 0-9)
- Flood fill colors 0-9 (operations 10-19)
- Clipboard operations: copy, paste, cut (operations 28-30)
- Grid operations: clear, copy_input, resize (operations 31-33)
- Submit solution (operation 34)

### Full Environment

Complete action set including all object-based operations:

```bash
python script.py environment=full
```

**Actions available:**

- All standard actions plus:
- Movement: up, down, left, right (operations 20-23)
- Rotation: clockwise, counter-clockwise (operations 24-25)
- Flipping: horizontal, vertical (operations 26-27)

### Training Environment

Optimized for training with dense rewards and exploration-friendly settings:

```bash
python script.py environment=training
```

**Features:**

- Dense rewards (reward at each step)
- Smaller penalties for invalid actions
- Longer episodes for exploration
- Detailed logging enabled

### Evaluation Environment

Optimized for evaluation with strict validation:

```bash
python script.py environment=evaluation
```

**Features:**

- Sparse rewards (only on submit)
- Strict validation
- No exploration allowances
- Minimal logging for clean evaluation

## Reward Configurations

### Standard Rewards

Balanced reward structure for general use:

```yaml
reward_on_submit_only: true
step_penalty: -0.01
success_bonus: 10.0
similarity_weight: 1.0
progress_bonus: 0.1
invalid_action_penalty: -0.5
```

### Training Rewards

Dense rewards for better learning signals:

```yaml
reward_on_submit_only: false # Dense rewards
step_penalty: -0.005 # Smaller penalty
success_bonus: 20.0 # Higher bonus
similarity_weight: 2.0 # Higher similarity weight
progress_bonus: 0.5 # Larger progress bonus
invalid_action_penalty: -0.1 # Smaller penalty for exploration
```

### Evaluation Rewards

Sparse rewards for realistic evaluation:

```yaml
reward_on_submit_only: true # Only final reward
step_penalty: -0.01 # Standard penalty
success_bonus: 10.0 # Standard bonus
similarity_weight: 1.0 # Standard weight
progress_bonus: 0.0 # No progress bonus
invalid_action_penalty: -1.0 # Higher penalty for mistakes
```

## Action Configurations

### Raw Actions (Operation IDs 0-9, 33-34)

```yaml
allowed_operations: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 33, 34]
```

### Standard Actions (Operation IDs 0-19, 28-34)

```yaml
allowed_operations:
  [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
  ]
```

### Full Actions (All Operation IDs 0-34)

```yaml
allowed_operations:
  [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
  ]
```

### Point-Based Actions

```yaml
selection_format: "point"
clip_invalid_actions: true
```

### Bounding Box Actions

```yaml
selection_format: "bbox"
clip_invalid_actions: true
```

## Dataset Configurations

Datasets now include their own grid configurations:

```yaml
# arc_agi_2.yaml
dataset_name: "ARC-AGI-2"
grid:
  max_grid_height: 30
  max_grid_width: 30
  min_grid_height: 1
  min_grid_width: 1
  max_colors: 10
  background_color: 0
```

```yaml
# mini_arc.yaml
dataset_name: "Mini-ARC"
grid:
  max_grid_height: 10 # Smaller for testing
  max_grid_width: 10
  min_grid_height: 1
  min_grid_width: 1
  max_colors: 10
  background_color: 0
```

## Using the Functional API

### Basic Usage with Hydra

```python
import hydra
from omegaconf import DictConfig
from jaxarc.envs.factory import create_complete_hydra_config
from jaxarc.envs.functional import arc_reset, arc_step


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Create config with parser integration
    env_config = create_complete_hydra_config(cfg)

    # Use functional API
    key = jax.random.PRNGKey(42)
    state, obs = arc_reset(key, env_config)

    # Take action
    action = {"selection": selection_mask, "operation": operation_id}
    state, obs, reward, done, info = arc_step(state, action, env_config)
```

### Programmatic Configuration

```python
from jaxarc.envs.factory import create_config_from_hydra
from omegaconf import OmegaConf

# Create config programmatically
config = OmegaConf.create(
    {
        "environment": {
            "max_episode_steps": 100,
            "reward": {
                "reward_on_submit_only": False,
                "step_penalty": -0.01,
                "success_bonus": 10.0,
            },
            "action": {
                "selection_format": "mask",
                "allowed_operations": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 33, 34],
            },
        }
    }
)

env_config = create_config_from_hydra(config.environment)
```

## Configuration Combinations

### Common Combinations

```bash
# Training with raw actions
python script.py environment=training environment.action=raw

# Evaluation with full actions
python script.py environment=evaluation environment.action=full

# Standard environment with training rewards
python script.py environment=arc_env environment.reward=training

# Mini dataset with raw environment
python script.py dataset=mini_arc environment=raw

# Point-based actions with training rewards
python script.py environment.action=point environment.reward=training
```

### Environment Overrides

```bash
# Override specific settings
python script.py environment.max_episode_steps=50
python script.py environment.reward.success_bonus=20.0
python script.py environment.action.validate_actions=false
```

## Migration from Old System

### Before (Task Sampler)

```python
# Old approach
from jaxarc.envs.task_sampling import create_arc_agi_sampler
from jaxarc.envs.factory import create_config_with_task_sampler

sampler = create_arc_agi_sampler()
config = create_config_with_task_sampler(base_config, sampler)
```

### After (Parser Integration)

```python
# New approach - automatic with Hydra
env_config = create_complete_hydra_config(hydra_cfg)

# Or manual parser attachment
from jaxarc.envs.factory import create_config_with_parser
from jaxarc.parsers import ArcAgiParser

parser = ArcAgiParser(dataset_cfg)
config = create_config_with_parser(base_config, parser)
```

## Best Practices

1. **Use appropriate environment types**: raw for simple experiments, standard
   for most use cases, full for complex tasks
2. **Match rewards to use case**: training rewards for learning, evaluation
   rewards for assessment
3. **Leverage dataset grid configs**: let datasets define their own grid
   constraints
4. **Use Hydra overrides**: modify specific settings without creating new config
   files
5. **Test with mini_arc**: use smaller dataset for rapid iteration

## Example Scripts

See `examples/hydra_integration_example.py` for a complete example of using the
new config system with different environment types and datasets.

## Configuration Reference

For complete configuration options, see:

- `src/jaxarc/envs/config.py` - Configuration dataclasses
- `src/jaxarc/types.py` - ARCLEOperationType for operation IDs
- `conf/` directory - All configuration files

The new system provides maximum flexibility while maintaining clean separation
of concerns and leveraging existing Hydra infrastructure.
