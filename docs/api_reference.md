# API Reference

Complete reference for JaxARC parsers, configuration, and core functionality
with practical examples.

## Parser Classes

### ArcAgiParser

Parser for ARC-AGI datasets (2024/2025) from GitHub repositories.

```python
from jaxarc.parsers import ArcAgiParser
from omegaconf import DictConfig
import jax

# Basic usage
config = DictConfig(
    {
        "training": {"path": "data/raw/ARC-AGI-1/data/training"},
        "evaluation": {"path": "data/raw/ARC-AGI-1/data/evaluation"},
        "grid": {"max_grid_height": 30, "max_grid_width": 30},
        "max_train_pairs": 10,
        "max_test_pairs": 3,
    }
)

parser = ArcAgiParser(config)

# Get tasks
task_ids = parser.get_available_task_ids()  # List all task IDs
task = parser.get_task_by_id("007bbfb7")  # Get specific task
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
config = DictConfig(
    {
        "corpus": {"path": "data/raw/ConceptARC/corpus"},
        "grid": {"max_grid_height": 30, "max_grid_width": 30},
        "max_train_pairs": 4,
        "max_test_pairs": 3,
    }
)

parser = ConceptArcParser(config)

# Explore concepts
concepts = parser.get_concept_groups()  # List all concept groups
tasks_in_center = parser.get_tasks_in_concept("Center")  # Tasks in concept
task = parser.get_random_task_from_concept("Center", jax.random.PRNGKey(42))
```

**Key Methods:**

- `get_concept_groups() -> list[str]`: List available concept groups
- `get_random_task_from_concept(concept, key) -> JaxArcTask`: Random task from
  concept
- `get_tasks_in_concept(concept: str) -> list[str]`: All tasks in concept group

**Concept Groups:** Spatial (Center, AboveBelow), Pattern (Copy, Order), Object
(ExtractObjects), Property (Count, CleanUp)

### MiniArcParser

Parser for MiniARC dataset optimized for 5x5 grids and rapid prototyping.

```python
from jaxarc.parsers import MiniArcParser
from omegaconf import DictConfig
import jax

# Basic usage
config = DictConfig(
    {
        "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
        "grid": {"max_grid_height": 5, "max_grid_width": 5},
        "max_train_pairs": 3,
        "max_test_pairs": 1,
    }
)

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

**Performance Benefits:** 36x less memory, 10-50x faster processing, larger
batch sizes

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

### ArcEnvState (Equinox Module)

The centralized environment state using Equinox for better JAX integration:

```python
from jaxarc.state import ArcEnvState
from jaxarc.utils.jax_types import GridArray, MaskArray, SimilarityScore
import equinox as eqx

class ArcEnvState(eqx.Module):
    """ARC environment state with Equinox Module for better JAX integration."""
    
    # Core ARC state with JAXTyping annotations
    task_data: JaxArcTask
    working_grid: GridArray          # Int[Array, "height width"]
    working_grid_mask: MaskArray     # Bool[Array, "height width"]
    target_grid: GridArray
    
    # Episode management
    step_count: StepCount            # Int[Array, ""]
    episode_done: EpisodeDone        # Bool[Array, ""]
    current_example_idx: EpisodeIndex
    
    # Grid operations
    selected: SelectionArray         # Bool[Array, "height width"]
    clipboard: GridArray
    similarity_score: SimilarityScore # Float[Array, ""]

# Usage examples
import jax.numpy as jnp

# Create state
state = ArcEnvState(
    task_data=task,
    working_grid=jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
    working_grid_mask=jnp.ones((2, 2), dtype=bool),
    target_grid=jnp.array([[4, 3], [2, 1]], dtype=jnp.int32),
    step_count=jnp.array(0, dtype=jnp.int32),
    episode_done=jnp.array(False, dtype=bool),
    current_example_idx=jnp.array(0, dtype=jnp.int32),
    selected=jnp.zeros((2, 2), dtype=bool),
    clipboard=jnp.zeros((2, 2), dtype=jnp.int32),
    similarity_score=jnp.array(0.0, dtype=jnp.float32)
)

# Update state using Equinox patterns
import equinox as eqx

# Single field update
new_state = eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)

# Multiple field update
new_state = eqx.tree_at(
    lambda s: (s.step_count, s.episode_done),
    state,
    (state.step_count + 1, jnp.array(True))
)

# Convenience replace method
new_state = state.replace(
    step_count=state.step_count + 1,
    episode_done=True
)
```

### JAXTyping Type Definitions

Precise array type annotations for better type safety:

```python
from jaxarc.utils.jax_types import (
    # Grid types (support both single and batched operations)
    GridArray,        # Int[Array, "*batch height width"]
    MaskArray,        # Bool[Array, "*batch height width"]
    SelectionArray,   # Bool[Array, "*batch height width"]
    
    # Action types
    PointCoords,      # Int[Array, "2"] - [row, col]
    BboxCoords,       # Int[Array, "4"] - [r1, c1, r2, c2]
    OperationId,      # Int[Array, ""] - scalar operation ID
    
    # Scoring and state types
    SimilarityScore,  # Float[Array, "*batch"]
    StepCount,        # Int[Array, ""]
    EpisodeIndex,     # Int[Array, ""]
    EpisodeDone,      # Bool[Array, ""]
)

# Usage with precise type annotations
def compute_similarity(grid1: GridArray, grid2: GridArray) -> SimilarityScore:
    """Compute similarity with automatic shape validation."""
    diff = jnp.abs(grid1 - grid2)
    return 1.0 - jnp.mean(diff) / 9.0

def process_point_action(
    point: PointCoords,
    operation: OperationId,
    grid_shape: tuple[int, int]
) -> SelectionArray:
    """Convert point action to selection mask."""
    row, col = point
    selection = jnp.zeros(grid_shape, dtype=bool)
    return selection.at[row, col].set(True)

# Batch operations work with the same types
batch_grids: GridArray = jnp.stack([grid1, grid2])  # Shape: (2, height, width)
batch_similarities: SimilarityScore = jax.vmap(compute_similarity)(batch_grids, batch_grids)
```

### JaxArcTask

Main data structure for ARC tasks with static shapes for JAX compatibility:

```python
from jaxarc.types import JaxArcTask
import chex

@chex.dataclass
class JaxArcTask:
    input_grids_examples: chex.Array  # (max_train_pairs, H, W)
    output_grids_examples: chex.Array  # (max_train_pairs, H, W)
    num_train_pairs: int

    test_input_grids: chex.Array  # (max_test_pairs, H, W)
    true_test_output_grids: chex.Array  # (max_test_pairs, H, W)
    num_test_pairs: int

    task_index: chex.Array  # Unique identifier
```

## Equinox Utilities

Enhanced utilities for working with Equinox modules and state debugging:

```python
from jaxarc.utils.equinox_utils import (
    tree_map_with_path, validate_state_shapes, create_state_diff,
    print_state_summary, module_memory_usage
)

# State validation
if validate_state_shapes(state):
    print("✅ State validation passed")
else:
    print("❌ State validation failed")

# State debugging
print_state_summary(state, "Current State")

# Tree traversal with path information
def debug_arrays(path: str, value: Any) -> Any:
    if hasattr(value, 'shape'):
        print(f"{path}: shape={value.shape}, dtype={value.dtype}")
    return value

tree_map_with_path(debug_arrays, state)

# State comparison
diff = create_state_diff(old_state, new_state)
for path, change_info in diff.items():
    print(f"Changed: {path}")
    print(f"  Type: {change_info['type']}")
    if 'old' in change_info:
        print(f"  Old: {change_info['old']}")
    if 'new' in change_info:
        print(f"  New: {change_info['new']}")

# Memory analysis
memory_info = module_memory_usage(state)
print(f"Total memory: {memory_info['total_bytes']:,} bytes")
print(f"Total elements: {memory_info['total_elements']:,}")
```

**Key Functions:**

- `validate_state_shapes(state) -> bool`: Validate Equinox module structure
- `print_state_summary(state, name)`: Print comprehensive state summary
- `tree_map_with_path(fn, tree)`: Map function over tree with path information
- `create_state_diff(old, new) -> dict`: Create diff between two states
- `module_memory_usage(module) -> dict`: Analyze memory usage of Equinox module

## Enhanced Configuration System

Comprehensive configuration with validation and error handling:

```python
from jaxarc.envs.config import (
    ArcEnvConfig, RewardConfig, ActionConfig, GridConfig,
    ConfigValidationError
)

# Create configuration with validation
try:
    config = ArcEnvConfig(
        max_episode_steps=100,
        reward=RewardConfig(
            success_bonus=10.0,
            step_penalty=-0.01,
            similarity_weight=1.5
        ),
        action=ActionConfig(
            selection_format="point",
            num_operations=35,
            allowed_operations=[0, 1, 2, 3, 4, 5]
        ),
        grid=GridConfig(
            max_grid_height=30,
            max_grid_width=30,
            max_colors=10
        )
    )
    print("✅ Configuration validation passed")
    
except ConfigValidationError as e:
    print(f"❌ Configuration validation failed: {e}")

# Create from Hydra config
from omegaconf import OmegaConf

hydra_config = OmegaConf.create({
    "max_episode_steps": 100,
    "reward": {"success_bonus": 15.0},
    "action": {"selection_format": "mask"}
})

typed_config = ArcEnvConfig.from_hydra(hydra_config)
```

**Configuration Classes:**

- `ArcEnvConfig`: Main environment configuration
- `RewardConfig`: Reward calculation settings
- `ActionConfig`: Action space and validation
- `GridConfig`: Grid dimensions and constraints
- `DatasetConfig`: Dataset-specific settings
- `DebugConfig`: Debug and logging options

**Validation Features:**

- Field-level validation with clear error messages
- Cross-field consistency checking
- Range validation for numeric values
- Choice validation for string values
- Helpful warnings for potentially problematic configurations

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
    config = DictConfig(
        {
            "training": {"path": "data/raw/ARC-AGI-1/data/training"},
            "grid": {"max_grid_height": 30, "max_grid_width": 30},
            "max_train_pairs": 10,
            "max_test_pairs": 3,
        }
    )

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

- **FileNotFoundError**: Download datasets with
  `python scripts/download_dataset.py <dataset-name>`
- **Legacy format error**: Update config to use `path` instead of
  `challenges`/`solutions`
- **Memory issues**: Use MiniARC for development or clear parser cache
  periodically
