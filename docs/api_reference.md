# API Reference

Complete reference for JaxARC parsers, configuration, and core functionality with practical examples.

## Parser Classes

### ArcAgiParser

Parser for ARC-AGI datasets (2024/2025) from GitHub repositories.

```python
from jaxarc.parsers import ArcAgiParser
from omegaconf import DictConfig
import jax

# Basic usage
config = DictConfig({
    "training": {"path": "data/raw/ARC-AGI-1/data/training"},
    "evaluation": {"path": "data/raw/ARC-AGI-1/data/evaluation"},
    "grid": {"max_grid_height": 30, "max_grid_width": 30},
    "max_train_pairs": 10, "max_test_pairs": 3
})

parser = ArcAgiParser(config)

# Get tasks
task_ids = parser.get_available_task_ids()  # List all task IDs
task = parser.get_task_by_id("007bbfb7")    # Get specific task
random_task = parser.get_random_task(jax.random.PRNGKey(42))  # Random task
```

**Key Methods:**
- `get_available_task_ids() -> list[str]`: List all available task IDs
- `get_task_by_id(task_id: str) -> JaxArcTask`: Get specific task
- `get_random_task(key: chex.PRNGKey) -> JaxArcTask`: Get random task

**Supported Datasets:**
- **ARC-AGI-1**: 400 train + 400 eval tasks (`fchollet/ARC-AGI`)
- **ARC-AGI-2**: 1000 train + 120 eval tasks (`arcprize/ARC-AGI-2`)

### ConceptArcParser

Parser for ConceptARC dataset with concept-based task organization.

```python
from jaxarc.parsers import ConceptArcParser
from omegaconf import DictConfig
import jax

# Basic usage
config = DictConfig({
    "corpus": {"path": "data/raw/ConceptARC/corpus"},
    "grid": {"max_grid_height": 30, "max_grid_width": 30},
    "max_train_pairs": 4, "max_test_pairs": 3
})

parser = ConceptArcParser(config)

# Explore concepts
concepts = parser.get_concept_groups()  # List all concept groups
tasks_in_center = parser.get_tasks_in_concept("Center")  # Tasks in concept
task = parser.get_random_task_from_concept("Center", jax.random.PRNGKey(42))
```

**Key Methods:**
- `get_concept_groups() -> list[str]`: List available concept groups
- `get_random_task_from_concept(concept, key) -> JaxArcTask`: Random task from concept
- `get_tasks_in_concept(concept: str) -> list[str]`: All tasks in concept group

**Concept Groups:** Spatial (Center, AboveBelow), Pattern (Copy, Order), Object (ExtractObjects), Property (Count, CleanUp)

### MiniArcParser

Parser for MiniARC dataset optimized for 5x5 grids and rapid prototyping.

```python
from jaxarc.parsers import MiniArcParser
from omegaconf import DictConfig
import jax

# Basic usage
config = DictConfig({
    "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
    "grid": {"max_grid_height": 5, "max_grid_width": 5},
    "max_train_pairs": 3, "max_test_pairs": 1
})

parser = MiniArcParser(config)

# Get tasks (5x5 optimized)
task_ids = parser.get_available_task_ids()  # List all task IDs
task = parser.get_random_task(jax.random.PRNGKey(42))  # Random 5x5 task
stats = parser.get_dataset_statistics()  # Performance metrics
```

**Key Methods:**
- `get_available_task_ids() -> list[str]`: List all available task IDs
- `get_random_task(key: chex.PRNGKey) -> JaxArcTask`: Get random 5x5 task
- `get_dataset_statistics() -> dict`: Performance and optimization metrics

**Performance Benefits:** 36x less memory, 10-50x faster processing, larger batch sizes

## Configuration Factory Functions

Quick configuration creation for different datasets and use cases.

```python
from jaxarc.envs.factory import create_conceptarc_config, create_miniarc_config

# Environment configurations
conceptarc_env = create_conceptarc_config(max_episode_steps=150, success_bonus=20.0)
miniarc_env = create_miniarc_config(max_episode_steps=50, success_bonus=5.0)

# Parser configurations  
from jaxarc.utils.config import create_conceptarc_config, create_miniarc_config

conceptarc_parser = create_conceptarc_config(max_episode_steps=100)
miniarc_parser = create_miniarc_config(max_episode_steps=80)
```

## Core Data Types

### JaxArcTask

Main data structure for ARC tasks with static shapes for JAX compatibility.

```python
from jaxarc.types import JaxArcTask
import chex

@chex.dataclass
class JaxArcTask:
    input_grids_examples: chex.Array    # (max_train_pairs, H, W)
    output_grids_examples: chex.Array   # (max_train_pairs, H, W)
    num_train_pairs: int
    
    test_input_grids: chex.Array        # (max_test_pairs, H, W)
    true_test_output_grids: chex.Array  # (max_test_pairs, H, W)
    num_test_pairs: int
    
    task_index: chex.Array              # Unique identifier
```

### Grid

```python
from jaxarc.types import Grid

Grid = chex.Array  # Shape: (height, width), dtype: int32
```

## Environment Integration

### Functional API

```python
from jaxarc.envs import arc_reset, arc_step

# Reset environment
state, obs = arc_reset(key, config, task_data=task)

# Step environment  
state, obs, reward, done, info = arc_step(state, action, config)
```

### Class-Based API

```python
from jaxarc.envs import ArcEnvironment

env = ArcEnvironment(config)
state, obs = env.reset(key, task_data=task)
state, obs, reward, info = env.step(action)
```

## Usage Examples

### ConceptARC Example

```python
import jax
from jaxarc.parsers import ConceptArcParser
from jaxarc.envs import create_conceptarc_config, ArcEnvironment
from omegaconf import DictConfig

# Create parser configuration
parser_config = DictConfig(
    {
        "corpus": {"path": "data/raw/ConceptARC/corpus"},
        "grid": {"max_grid_height": 30, "max_grid_width": 30},
        "max_train_pairs": 4,
        "max_test_pairs": 3,
    }
)

# Initialize parser
parser = ConceptArcParser(parser_config)

# Explore concept groups
concepts = parser.get_concept_groups()
print(f"Available concepts: {concepts}")

# Get task from specific concept
key = jax.random.PRNGKey(42)
task = parser.get_random_task_from_concept("Center", key)

# Create environment configuration
env_config = create_conceptarc_config(max_episode_steps=150, success_bonus=20.0)

# Run environment
env = ArcEnvironment(env_config)
state, obs = env.reset(key, task_data=task)
```

### MiniARC Example

```python
import jax
from jaxarc.parsers import MiniArcParser
from jaxarc.envs import create_miniarc_config, ArcEnvironment
from omegaconf import DictConfig

# Create parser configuration
parser_config = DictConfig(
    {
        "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
        "grid": {"max_grid_height": 5, "max_grid_width": 5},
        "max_train_pairs": 3,
        "max_test_pairs": 1,
    }
)

# Initialize parser
parser = MiniArcParser(parser_config)

# Get random task (5x5 optimized)
key = jax.random.PRNGKey(42)
task = parser.get_random_task(key)

# Create optimized environment configuration
env_config = create_miniarc_config(
    max_episode_steps=50, success_bonus=5.0  # Shorter for rapid iteration
)

# Run environment (10-50x faster than standard ARC)
env = ArcEnvironment(env_config)
state, obs = env.reset(key, task_data=task)
```

## Performance Considerations

### ConceptARC

- **Grid Size**: Up to 30×30 (same as standard ARC)
- **Memory Usage**: Standard ARC memory requirements
- **Use Case**: Systematic evaluation and concept-based analysis
- **Optimization**: Concept-based task sampling and organization

### MiniARC

- **Grid Size**: Maximum 5×5 (36x fewer cells than standard ARC)
- **Memory Usage**: 36x less memory per grid
- **Processing Speed**: 10-50x faster than standard ARC
- **Batch Size**: Support for much larger batch sizes
- **Use Case**: Rapid prototyping, algorithm development, quick experiments

### JAX Compatibility

All parsers and configurations are fully compatible with JAX transformations:

```python
# JIT compilation
@jax.jit
def process_task(task, config):
    return some_processing(task, config)


# Vectorization
batch_process = jax.vmap(process_task, in_axes=(0, None))

# Parallel processing
parallel_process = jax.pmap(process_task, in_axes=(0, None))
```

## Error Handling

### Common Exceptions

- **`FileNotFoundError`**: Dataset files not found
- **`ValueError`**: Invalid configuration or task parameters
- **`KeyError`**: Missing concept group or task ID
- **`ValidationError`**: Configuration validation failures
- **`RuntimeError`**: Legacy Kaggle format detected or GitHub download issues

### Best Practices

1. **Validate configurations** before use
2. **Handle missing datasets** gracefully
3. **Use appropriate grid sizes** for each dataset type
4. **Check task availability** before accessing specific tasks
5. **Monitor memory usage** with large batch sizes

## Testing

### Basic Testing Patterns

```python
import pytest
import jax
from jaxarc.parsers import ArcAgiParser
from omegaconf import DictConfig

def test_parser_functionality():
    """Test basic parser operations."""
    config = DictConfig({
        "training": {"path": "data/raw/ARC-AGI-1/data/training"},
        "grid": {"max_grid_height": 30, "max_grid_width": 30},
        "max_train_pairs": 10, "max_test_pairs": 3
    })
    
    parser = ArcAgiParser(config)
    task_ids = parser.get_available_task_ids()
    assert len(task_ids) > 0
    
    # Test task loading
    task = parser.get_task_by_id(task_ids[0])
    assert task.num_train_pairs > 0
    
    # Test random task
    key = jax.random.PRNGKey(42)
    random_task = parser.get_random_task(key)
    assert random_task.task_index is not None

def test_jax_compatibility():
    """Test JAX transformations work correctly."""
    @jax.jit
    def process_task(task):
        return task.input_grids_examples.sum()
    
    # Test with actual task data
    task = parser.get_random_task(jax.random.PRNGKey(0))
    result = process_task(task)  # Should not raise errors
    assert result is not None
```

### Common Issues

- **FileNotFoundError**: Download datasets with `python scripts/download_dataset.py <dataset-name>`
- **Legacy format error**: Update config to use `path` instead of `challenges`/`solutions`
- **Memory issues**: Use MiniARC for development or clear parser cache periodically
