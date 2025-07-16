# API Reference

This document provides a comprehensive reference for the JaxARC API, including all parser classes, configuration utilities, and core functionality.

## Parser Classes

### ArcAgiParser

General parser for ARC-AGI datasets from Kaggle competitions.

```python
from jaxarc.parsers import ArcAgiParser

class ArcAgiParser(ArcDataParserBase):
    """Parser for ARC-AGI datasets (2024 and 2025 competitions)."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize parser with configuration."""
        
    def parse_task_file(self, file_path: str, task_id: str) -> JaxArcTask:
        """Parse a specific task from a JSON file."""
        
    def parse_all_tasks_from_file(self, file_path: str) -> dict[str, JaxArcTask]:
        """Parse all tasks from a JSON file."""
        
    def get_random_task(self, key: chex.PRNGKey) -> JaxArcTask:
        """Get a random task from the dataset."""
```

**Supported Datasets:**
- ARC-AGI-1 (2024 competition)
- ARC-AGI-2 (2025 competition)

**Configuration:**
```yaml
dataset_name: "ARC-AGI-2"
data_root: "data/raw/arc-prize-2025"
training:
  challenges_path: "${dataset.data_root}/arc-agi_training_challenges.json"
  solutions_path: "${dataset.data_root}/arc-agi_training_solutions.json"
evaluation:
  challenges_path: "${dataset.data_root}/arc-agi_evaluation_challenges.json"
  solutions_path: "${dataset.data_root}/arc-agi_evaluation_solutions.json"
```

### ConceptArcParser

Specialized parser for ConceptARC dataset with concept group organization.

```python
from jaxarc.parsers import ConceptArcParser

class ConceptArcParser(ArcDataParserBase):
    """Parser for ConceptARC dataset with concept group organization."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize parser with ConceptARC configuration."""
        
    def get_concept_groups(self) -> list[str]:
        """Get list of available concept groups."""
        
    def get_random_task_from_concept(self, concept: str, key: chex.PRNGKey) -> JaxArcTask:
        """Get random task from specific concept group."""
        
    def get_tasks_in_concept(self, concept: str) -> list[str]:
        """Get all task IDs in a specific concept group."""
        
    def get_task_by_id(self, task_id: str) -> JaxArcTask:
        """Get specific task by ID (format: 'ConceptGroup/TaskName')."""
        
    def get_task_metadata(self, task_id: str) -> dict:
        """Get metadata for a specific task."""
        
    def get_dataset_statistics(self) -> dict:
        """Get comprehensive dataset statistics."""
```

**Concept Groups:**
- **Spatial**: AboveBelow, Center, InsideOutside, TopBottom2D, TopBottom3D
- **Pattern**: Copy, CompleteShape, SameDifferent, Order
- **Object**: ExtractObjects, MoveToBoundary, ExtendToBoundary
- **Property**: FilledNotFilled, Count, CleanUp, HorizontalVertical

**Configuration:**
```yaml
dataset_name: "ConceptARC"
data_root: "data/raw/ConceptARC"
corpus:
  path: "${dataset.data_root}/corpus"
  concept_groups: [
    "AboveBelow", "Center", "CleanUp", "CompleteShape",
    "Copy", "Count", "ExtendToBoundary", "ExtractObjects",
    "FilledNotFilled", "HorizontalVertical", "InsideOutside",
    "MoveToBoundary", "Order", "SameDifferent", "TopBottom2D", "TopBottom3D"
  ]
max_train_pairs: 4
max_test_pairs: 3
```

### MiniArcParser

Optimized parser for MiniARC dataset with 5x5 grids and rapid prototyping capabilities.

```python
from jaxarc.parsers import MiniArcParser

class MiniArcParser(ArcDataParserBase):
    """Parser for MiniARC dataset optimized for 5x5 grids."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize parser with MiniARC configuration."""
        
    def load_task_file(self, task_file_path: str) -> Any:
        """Load raw task data from a JSON file."""
        
    def preprocess_task_data(self, raw_task_data: Any, key: chex.PRNGKey) -> JaxArcTask:
        """Convert raw task data into JaxArcTask structure."""
        
    def get_available_task_ids(self) -> list[str]:
        """Get list of all available task IDs."""
        
    def get_task_by_id(self, task_id: str) -> JaxArcTask:
        """Get specific task by ID."""
        
    def get_dataset_statistics(self) -> dict:
        """Get dataset statistics including performance metrics."""
        
    def get_random_task(self, key: chex.PRNGKey) -> JaxArcTask:
        """Get random task optimized for 5x5 processing."""
```

**Performance Benefits:**
- **Memory**: 36x less memory per grid (25 vs 900 cells)
- **Speed**: 10-50x faster processing and training
- **Batch Size**: Support for larger batch sizes
- **Development**: Seconds to minutes vs hours for iteration cycles

**Configuration:**
```yaml
dataset_name: "MiniARC"
data_root: "data/raw/MiniARC"
tasks:
  path: "${dataset.data_root}/data/MiniARC"
grid:
  max_grid_height: 5
  max_grid_width: 5
max_train_pairs: 3
max_test_pairs: 1
optimization:
  enable_5x5_optimizations: true
  fast_processing: true
  reduced_memory_usage: true
```

## Configuration Factory Functions

### Dataset-Specific Configurations

```python
from jaxarc.envs.factory import (
    create_conceptarc_config,
    create_miniarc_config
)

def create_conceptarc_config(
    max_episode_steps: int = 150,
    task_split: str = "corpus",
    reward_on_submit_only: bool = True,
    success_bonus: float = 20.0,
    step_penalty: float = -0.01,
    **kwargs
) -> ArcEnvConfig:
    """Create ConceptARC-optimized environment configuration."""

def create_miniarc_config(
    max_episode_steps: int = 50,
    task_split: str = "tasks",
    reward_on_submit_only: bool = False,
    success_bonus: float = 5.0,
    step_penalty: float = -0.001,
    **kwargs
) -> ArcEnvConfig:
    """Create MiniARC-optimized environment configuration."""
```

### Configuration Utilities

```python
from jaxarc.utils.config import (
    create_conceptarc_config,
    create_miniarc_config
)

def create_conceptarc_config(
    max_episode_steps: int = 100,
    task_split: str = "corpus",
    success_bonus: float = 15.0,
    **kwargs
) -> DictConfig:
    """Create ConceptARC configuration for parser usage."""

def create_miniarc_config(
    max_episode_steps: int = 80,
    task_split: str = "training",
    success_bonus: float = 5.0,
    **kwargs
) -> DictConfig:
    """Create MiniARC configuration for parser usage."""
```

## Core Data Types

### JaxArcTask

Main data structure for ARC tasks, compatible with all parsers.

```python
from jaxarc.types import JaxArcTask

@chex.dataclass
class JaxArcTask:
    """JAX-compatible ARC task representation."""
    
    input_grids_examples: chex.Array  # Shape: (max_train_pairs, H, W)
    input_masks_examples: chex.Array  # Shape: (max_train_pairs, H, W)
    output_grids_examples: chex.Array  # Shape: (max_train_pairs, H, W)
    output_masks_examples: chex.Array  # Shape: (max_train_pairs, H, W)
    num_train_pairs: int
    
    test_input_grids: chex.Array  # Shape: (max_test_pairs, H, W)
    test_input_masks: chex.Array  # Shape: (max_test_pairs, H, W)
    true_test_output_grids: chex.Array  # Shape: (max_test_pairs, H, W)
    true_test_output_masks: chex.Array  # Shape: (max_test_pairs, H, W)
    num_test_pairs: int
    
    task_index: chex.Array  # Unique task identifier
```

### Grid

2D color grid representation.

```python
from jaxarc.types import Grid

Grid = chex.Array  # Shape: (height, width), dtype: int32
```

## Environment Integration

### Functional API

```python
from jaxarc.envs import arc_reset, arc_step

def arc_reset(
    key: chex.PRNGKey,
    config: ArcEnvConfig,
    task_data: JaxArcTask | None = None
) -> tuple[ArcEnvState, chex.Array]:
    """Reset environment with optional task data."""

def arc_step(
    state: ArcEnvState,
    action: dict[str, chex.Array],
    config: ArcEnvConfig
) -> tuple[ArcEnvState, chex.Array, float, bool, dict]:
    """Step environment with action."""
```

### Class-Based API

```python
from jaxarc.envs import ArcEnvironment

class ArcEnvironment:
    """JAX-compatible ARC environment."""
    
    def __init__(self, config: ArcEnvConfig):
        """Initialize environment with configuration."""
        
    def reset(
        self,
        key: chex.PRNGKey,
        task_data: JaxArcTask | None = None
    ) -> tuple[ArcEnvState, chex.Array]:
        """Reset environment."""
        
    def step(
        self,
        action: dict[str, chex.Array]
    ) -> tuple[ArcEnvState, chex.Array, float, dict]:
        """Step environment."""
```

## Usage Examples

### ConceptARC Example

```python
import jax
from jaxarc.parsers import ConceptArcParser
from jaxarc.envs import create_conceptarc_config, ArcEnvironment
from omegaconf import DictConfig

# Create parser configuration
parser_config = DictConfig({
    "corpus": {"path": "data/raw/ConceptARC/corpus"},
    "grid": {"max_grid_height": 30, "max_grid_width": 30},
    "max_train_pairs": 4,
    "max_test_pairs": 3,
})

# Initialize parser
parser = ConceptArcParser(parser_config)

# Explore concept groups
concepts = parser.get_concept_groups()
print(f"Available concepts: {concepts}")

# Get task from specific concept
key = jax.random.PRNGKey(42)
task = parser.get_random_task_from_concept("Center", key)

# Create environment configuration
env_config = create_conceptarc_config(
    max_episode_steps=150,
    success_bonus=20.0
)

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
parser_config = DictConfig({
    "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
    "grid": {"max_grid_height": 5, "max_grid_width": 5},
    "max_train_pairs": 3,
    "max_test_pairs": 1,
})

# Initialize parser
parser = MiniArcParser(parser_config)

# Get random task (5x5 optimized)
key = jax.random.PRNGKey(42)
task = parser.get_random_task(key)

# Create optimized environment configuration
env_config = create_miniarc_config(
    max_episode_steps=50,  # Shorter for rapid iteration
    success_bonus=5.0
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

### Best Practices

1. **Validate configurations** before use
2. **Handle missing datasets** gracefully
3. **Use appropriate grid sizes** for each dataset type
4. **Check task availability** before accessing specific tasks
5. **Monitor memory usage** with large batch sizes

## Migration Guide

### From ArcAgi1Parser to ArcAgiParser

```python
# Old way
from jaxarc.parsers import ArcAgi1Parser
parser = ArcAgi1Parser()

# New way
from jaxarc.parsers import ArcAgiParser
parser = ArcAgiParser(config)
```

### Adding ConceptARC Support

```python
# Add ConceptARC to existing code
from jaxarc.parsers import ConceptArcParser
from jaxarc.envs import create_conceptarc_config

# Replace standard parser with ConceptARC
parser = ConceptArcParser(conceptarc_config)
env_config = create_conceptarc_config()
```

### Adding MiniARC for Rapid Prototyping

```python
# Add MiniARC for faster development
from jaxarc.parsers import MiniArcParser
from jaxarc.envs import create_miniarc_config

# Use MiniARC for rapid iteration
parser = MiniArcParser(miniarc_config)
env_config = create_miniarc_config(max_episode_steps=50)
```