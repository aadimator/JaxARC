# New Usage Patterns

This document describes the updated usage patterns in JaxARC following the codebase consistency cleanup.

## Parser Configuration Updates

### New Typed Configuration Pattern (Recommended)

All parsers now accept typed `DatasetConfig` objects for better type safety and validation:

```python
from jaxarc.parsers import MiniArcParser, ArcAgiParser, ConceptArcParser
from jaxarc.envs.config import DatasetConfig

# Create typed configuration
dataset_config = DatasetConfig(
    dataset_path="data/raw/MiniARC",
    max_grid_height=5,
    max_grid_width=5,
    max_colors=10,
    background_color=0,
    task_split="train"
)

# Initialize parser with typed config
parser = MiniArcParser(dataset_config)
```

### Backward Compatibility with Hydra

For backward compatibility, all parsers provide `from_hydra()` class methods:

```python
from omegaconf import DictConfig

# Hydra configuration
hydra_config = DictConfig({
    "dataset_path": "data/raw/MiniARC",
    "max_grid_height": 5,
    "max_grid_width": 5,
    # ... other fields
})

# Option 1: Convert to typed config first (recommended)
dataset_config = DatasetConfig.from_hydra(hydra_config)
parser = MiniArcParser(dataset_config)

# Option 2: Use from_hydra class method
parser = MiniArcParser.from_hydra(hydra_config)
```

## Enhanced Visualization System

### Task Visualization at Episode Start

The visualization system now supports task visualization at episode start:

```python
from jaxarc.utils.visualization.visualizer import Visualizer, VisualizationConfig

# Create visualizer
config = VisualizationConfig(debug_level="standard")
visualizer = Visualizer(config)

# Start episode with task visualization
visualizer.start_episode_with_task(
    episode_num=1,
    task_data=task,
    task_id="task_001",
    current_pair_index=0,
    episode_mode="train"
)
```

### Enhanced Step Visualization with Task Context

Step visualizations now include task context information:

```python
from jaxarc.utils.visualization.visualizer import StepVisualizationData

# Create step data with task context
step_data = StepVisualizationData(
    step_num=5,
    before_grid=before_grid,
    after_grid=after_grid,
    action=action,
    reward=reward,
    info=info,
    # New task context fields
    task_id="task_001",
    task_pair_index=0,
    total_task_pairs=3
)

# Visualize step
visualizer.visualize_step(step_data)
```

## Functional API Improvements

### Decomposed Functions

The main `arc_reset` and `arc_step` functions have been decomposed into focused helper functions for better maintainability:

#### arc_reset Helper Functions:
- `_get_or_create_task_data()`: Task data acquisition or demo creation
- `_select_initial_pair()`: Initial pair selection based on mode
- `_initialize_grids()`: Grid initialization with target masking
- `_create_initial_state()`: Complete state creation

#### arc_step Helper Functions:
- `_process_action()`: Action processing and validation
- `_update_state()`: State updates with history tracking
- `_calculate_reward_and_done()`: Reward calculation and termination

### Enhanced Configuration Support

Both functions now support automatic conversion from Hydra configs:

```python
from jaxarc.envs.functional import arc_reset, arc_step
from omegaconf import DictConfig

# Works with both typed configs and Hydra configs
hydra_config = DictConfig({...})
state, obs = arc_reset(key, hydra_config, task_data)  # Auto-converts to typed config

# Preferred: Use typed config directly
from jaxarc.envs.config import JaxArcConfig
typed_config = JaxArcConfig.from_hydra(hydra_config)
state, obs = arc_reset(key, typed_config, task_data)
```

## Migration Guide

### From Old Parser Usage

**Old Pattern:**
```python
# Old: Raw Hydra config
parser = MiniArcParser(hydra_config.dataset)
```

**New Pattern:**
```python
# New: Typed config (recommended)
dataset_config = DatasetConfig.from_hydra(hydra_config.dataset)
parser = MiniArcParser(dataset_config)

# Or: Use from_hydra method
parser = MiniArcParser.from_hydra(hydra_config.dataset)
```

### From Basic Visualization

**Old Pattern:**
```python
# Old: Basic episode start
visualizer.start_episode(episode_num, task_id)
```

**New Pattern:**
```python
# New: Episode start with task visualization
visualizer.start_episode_with_task(
    episode_num, task_data, task_id, current_pair_index, episode_mode
)
```

### Enhanced Step Data

**Old Pattern:**
```python
# Old: Basic step data
step_data = StepVisualizationData(
    step_num=step_num,
    before_grid=before_grid,
    after_grid=after_grid,
    action=action,
    reward=reward,
    info=info
)
```

**New Pattern:**
```python
# New: Step data with task context
step_data = StepVisualizationData(
    step_num=step_num,
    before_grid=before_grid,
    after_grid=after_grid,
    action=action,
    reward=reward,
    info=info,
    # Enhanced task context
    task_id=task_id,
    task_pair_index=current_pair_index,
    total_task_pairs=total_pairs
)
```

## Benefits of New Patterns

1. **Type Safety**: Typed configurations provide better IDE support and catch errors early
2. **Maintainability**: Decomposed functions are easier to understand and modify
3. **Enhanced Visualization**: Task context provides better research insights
4. **Backward Compatibility**: Existing Hydra-based code continues to work
5. **Consistency**: Single, clear API for each functionality

## Best Practices

1. **Use Typed Configs**: Prefer `DatasetConfig.from_hydra()` over raw Hydra configs
2. **Include Task Context**: Always provide task context in visualizations
3. **Leverage Helper Functions**: The decomposed functions are available for custom implementations
4. **Follow Examples**: Use the provided examples as templates for new code