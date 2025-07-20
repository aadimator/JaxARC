# Configuration System Redesign

## Overview

The JaxARC configuration system has been redesigned to eliminate redundancy and confusion by establishing clear separation of concerns. Each configuration class now has a single, well-defined responsibility.

## Design Principles

### 1. Single Responsibility
Each configuration class handles exactly one aspect of the system:
- **EnvironmentConfig**: Core environment behavior and runtime settings
- **DatasetConfig**: Dataset-specific settings and constraints
- **VisualizationConfig**: All visualization and rendering settings
- **StorageConfig**: All storage, output, and file management
- **LoggingConfig**: All logging behavior and formats
- **WandbConfig**: Weights & Biases integration only

### 2. No Redundancy
- Each parameter exists in exactly one configuration class
- No duplicate or overlapping settings across classes
- Clear ownership of each functionality

### 3. Clear Boundaries
- Dataset constraints (grid size, colors) are only in `DatasetConfig`
- Logging parameters are only in `LoggingConfig`
- Storage/output paths are only in `StorageConfig`
- Visualization settings are only in `VisualizationConfig`
- Environment runtime behavior is only in `EnvironmentConfig`

## Configuration Classes

### EnvironmentConfig
**Purpose**: Core environment behavior and runtime settings

**Contains**:
- Episode settings (`max_episode_steps`, `auto_reset`)
- Environment behavior (`strict_validation`, `allow_invalid_actions`)
- Debug level (`debug_level`) with computed properties for other configs

**Does NOT contain**: Dataset constraints, logging, visualization, or storage settings

### DatasetConfig
**Purpose**: Dataset-specific settings and constraints

**Contains**:
- Dataset identification (`dataset_name`, `dataset_path`)
- Grid constraints (`max_grid_height`, `max_grid_width`, etc.)
- Color constraints (`max_colors`, `background_color`)
- Task sampling (`task_split`, `max_tasks`, `shuffle_tasks`)

**Does NOT contain**: Environment behavior, logging, or visualization settings

### VisualizationConfig
**Purpose**: All visualization and rendering settings

**Contains**:
- Core settings (`enabled`, `level`)
- Output formats and quality (`output_formats`, `image_quality`)
- Display settings (`show_coordinates`, `color_scheme`, etc.)
- Episode visualization flags
- Memory and performance settings

**Does NOT contain**: Logging settings or storage paths

### StorageConfig
**Purpose**: All storage, output, and file management

**Contains**:
- Storage policy and base directory
- Specific output directories for different content types
- Storage limits and cleanup settings
- File organization settings
- Safety settings

**Manages all output paths**:
- `base_output_dir`: Root output directory
- `episodes_dir`: Episode data
- `debug_dir`: Debug output
- `visualization_dir`: Visualization files
- `logs_dir`: Log files

### LoggingConfig
**Purpose**: All logging behavior and formats

**Contains**:
- Core logging settings (`log_format`, `log_level`, etc.)
- Specific content flags (`log_operations`, `log_grid_changes`, `log_rewards`)
- Logging frequency and timing
- Async logging performance settings

**Centralized logging control**: All "what to log" decisions are here

### WandbConfig
**Purpose**: Weights & Biases integration only

**Contains**:
- W&B connection settings
- W&B-specific logging preferences
- Error handling for W&B operations

**Does NOT contain**: Local logging or storage settings

## Migration from Old System

### Before (Problematic)
```python
# Dataset constraints scattered in environment config
env_config.max_grid_height = 30      # ❌ Wrong place (dataset-specific)
env_config.max_colors = 10            # ❌ Wrong place (dataset-specific)
env_config.log_operations = True      # ❌ Wrong place (logging-specific)

# Debug config with only one parameter
debug_config = DebugConfig(level="verbose")  # ❌ Unnecessary overhead

# Logging scattered across multiple configs
env_config.log_operations = True      # ❌ Confusing
viz_config.log_frequency = 10         # ❌ Wrong place
```

### After (Clean)
```python
# Environment only contains runtime behavior
env_config = EnvironmentConfig(
    max_episode_steps=100,
    strict_validation=True,
    debug_level="verbose",  # Debug level integrated here
)

# Dataset constraints in their proper place
dataset_config = DatasetConfig(
    dataset_name="arc-agi-2",
    max_grid_height=30,
    max_colors=10,
    task_split="training",
)

# All logging in one place
logging_config = LoggingConfig(
    log_operations=True,
    log_rewards=True,
    log_frequency=10,
)

# All storage/output paths managed centrally
storage_config = StorageConfig(
    base_output_dir="outputs",
    debug_dir="debug",
    visualization_dir="viz",
    logs_dir="logs",
)
```

## Benefits

### 1. Eliminates Confusion
- No more "which parameter should I use?"
- Clear ownership of each setting
- Single source of truth for each functionality

### 2. Reduces Errors
- No conflicting settings across configs
- No duplicate parameters with different values
- Validation is simpler and more reliable

### 3. Improves Maintainability
- Changes to logging behavior only affect `LoggingConfig`
- Storage changes only affect `StorageConfig`
- Clear boundaries make code easier to understand

### 4. Better User Experience
- Intuitive organization matches mental models
- Less cognitive load when configuring the system
- Easier to find the right setting

## Usage Examples

### Basic Usage
```python
from jaxarc.envs.equinox_config import (
    EnvironmentConfig, DatasetConfig, LoggingConfig, 
    StorageConfig, VisualizationConfig, WandbConfig
)

# Environment runtime behavior only
env_config = EnvironmentConfig(
    max_episode_steps=200,
    strict_validation=False,
    debug_level="verbose"  # Debug level integrated here
)

# Dataset-specific constraints
dataset_config = DatasetConfig(
    dataset_name="arc-agi-2",
    max_grid_height=50,
    max_colors=15,
    task_split="training"
)

# All logging decisions in one place
logging_config = LoggingConfig(
    log_operations=True,
    log_rewards=True,
    log_frequency=5,
    log_level="DEBUG"
)

# All storage/output management
storage_config = StorageConfig(
    base_output_dir="experiments",
    debug_dir="debug_output",
    max_storage_gb=10.0
)
```

### Hydra Integration
```python
# Each config can be created from Hydra independently
env_config = EnvironmentConfig.from_hydra(cfg.environment)
dataset_config = DatasetConfig.from_hydra(cfg.dataset)
logging_config = LoggingConfig.from_hydra(cfg.logging)
storage_config = StorageConfig.from_hydra(cfg.storage)
viz_config = VisualizationConfig.from_hydra(cfg.visualization)
wandb_config = WandbConfig.from_hydra(cfg.wandb)
```

## Validation

Each configuration class validates only its own parameters:

```python
# Each config validates independently
env_errors = env_config.validate()
dataset_errors = dataset_config.validate()
logging_errors = logging_config.validate()
storage_errors = storage_config.validate()
viz_errors = viz_config.validate()
wandb_errors = wandb_config.validate()

# All errors are clearly attributed to their source
all_errors = {
    "environment": env_errors,
    "dataset": dataset_errors,
    "logging": logging_errors,
    "storage": storage_errors,
    "visualization": viz_errors,
    "wandb": wandb_errors,
}
```

## Conclusion

The redesigned configuration system eliminates the confusion and redundancy of the previous system by establishing clear boundaries and single responsibility for each configuration class. This makes the system easier to use, maintain, and extend while reducing the likelihood of configuration errors.