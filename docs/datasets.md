# Datasets Guide

JaxARC supports multiple ARC dataset variants with dedicated parsers optimized for different use cases. All datasets are downloaded directly from GitHub repositories with no authentication required, providing fast and reliable access to the latest data.

## Supported Datasets

| Dataset | Tasks | Grid Size | Use Case | Parser |
|---------|-------|-----------|----------|---------|
| **ARC-AGI-2** | 1000 train + 120 eval | Up to 30×30 | Full challenge dataset | `ArcAgiParser` |
| **ConceptARC** | 160 (16 concepts × 10) | Up to 30×30 | Systematic evaluation | `ConceptArcParser` |
| **MiniARC** | 400+ | 5×5 | Rapid prototyping | `MiniArcParser` |
| **ARC-AGI-1** | 400 train + 400 eval | Up to 30×30 | Original 2024 dataset | `ArcAgiParser` |

## Quick Start

### 1. Download Datasets

Download datasets directly from GitHub repositories using the streamlined download script:

```bash
# Download your first dataset (recommended for beginners)
python scripts/download_dataset.py miniarc      # Fast experimentation

# Download full challenge datasets
python scripts/download_dataset.py arc-agi-2   # Latest challenge dataset
python scripts/download_dataset.py arc-agi-1   # Original 2024 dataset

# Download specialized datasets
python scripts/download_dataset.py conceptarc  # Systematic evaluation

# Download all datasets at once
python scripts/download_dataset.py all
```

### 2. Basic Usage

```python
import jax
from jaxarc.parsers import MiniArcParser
from omegaconf import DictConfig

# Quick setup with MiniARC (recommended for first-time users)
config = DictConfig({
    "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
    "grid": {
        "max_grid_height": 5, "max_grid_width": 5,
        "min_grid_height": 1, "min_grid_width": 1,
        "max_colors": 10, "background_color": 0
    },
    "max_train_pairs": 3, "max_test_pairs": 1
})

parser = MiniArcParser(config)
key = jax.random.PRNGKey(42)
task = parser.get_random_task(key)

print(f"Task loaded with {task.num_train_pairs} training pairs")
```

## Dataset Details

### ARC-AGI Datasets (GitHub Format)

The ARC-AGI datasets are downloaded directly from GitHub repositories, eliminating the need for external dependencies like Kaggle CLI.

**Key Benefits:**
- No authentication or API credentials required
- Faster, more reliable downloads
- Individual task files enable selective loading
- Better memory usage and performance

**Repository Sources:**
- **ARC-AGI-1**: [fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI)
- **ARC-AGI-2**: [arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2)

**Directory Structure:**
```
data/raw/ARC-AGI-1/  # or ARC-AGI-2/
└── data/
    ├── training/
    │   ├── 007bbfb7.json
    │   ├── 00d62c1b.json
    │   └── ... (400 or 1000 training tasks)
    └── evaluation/
        ├── 00576224.json
        ├── 009d5c81.json
        └── ... (400 or 120 evaluation tasks)
```

#### ARC-AGI Parser Usage

```python
from jaxarc.parsers import ArcAgiParser
from omegaconf import DictConfig

# Configuration for ARC-AGI datasets
config = DictConfig({
    "training": {"path": "data/raw/ARC-AGI-2/data/training"},
    "evaluation": {"path": "data/raw/ARC-AGI-2/data/evaluation"},
    "grid": {
        "max_grid_height": 30, "max_grid_width": 30,
        "min_grid_height": 1, "min_grid_width": 1,
        "max_colors": 10, "background_color": 0
    },
    "max_train_pairs": 10, "max_test_pairs": 3
})

# Create parser instance
parser = ArcAgiParser(config)

# Get random task
key = jax.random.PRNGKey(42)
task = parser.get_random_task(key)

# Get specific task by ID
task = parser.get_task_by_id("007bbfb7")

# Get available task IDs
task_ids = parser.get_available_task_ids()
print(f"Available tasks: {len(task_ids)}")
```

### ConceptARC Dataset

ConceptARC organizes tasks into 16 systematic concept groups for structured evaluation and analysis.

**Repository**: [victorvikram/ConceptARC](https://github.com/victorvikram/ConceptARC)

**Directory Structure:**
```
data/raw/ConceptARC/corpus/
├── AboveBelow/
├── Center/
├── CleanUp/
├── CompleteShape/
├── Copy/
├── Count/
├── ExtendToBoundary/
├── ExtractObjects/
├── FilledNotFilled/
├── HorizontalVertical/
├── InsideOutside/
├── MoveToBoundary/
├── Order/
├── SameDifferent/
├── TopBottom2D/
└── TopBottom3D/
```

#### Concept Groups

| Concept Group | Description | Tasks |
|---------------|-------------|-------|
| **AboveBelow** | Spatial relationships (above/below) | 10 |
| **Center** | Centering and central positioning | 10 |
| **CleanUp** | Removing noise or unwanted elements | 10 |
| **CompleteShape** | Shape completion tasks | 10 |
| **Copy** | Copying patterns or objects | 10 |
| **Count** | Counting-based transformations | 10 |
| **ExtendToBoundary** | Extending patterns to boundaries | 10 |
| **ExtractObjects** | Object extraction and isolation | 10 |
| **FilledNotFilled** | Distinguishing filled vs empty | 10 |
| **HorizontalVertical** | Horizontal/vertical relationships | 10 |
| **InsideOutside** | Inside/outside spatial concepts | 10 |
| **MoveToBoundary** | Moving objects to boundaries | 10 |
| **Order** | Ordering and sequencing | 10 |
| **SameDifferent** | Similarity and difference detection | 10 |
| **TopBottom2D** | 2D top/bottom relationships | 10 |
| **TopBottom3D** | 3D perspective top/bottom | 10 |

#### ConceptARC Parser Usage

```python
import jax
from jaxarc.parsers import ConceptArcParser
from omegaconf import DictConfig

# Configuration for ConceptARC
config = DictConfig({
    "corpus": {
        "path": "data/raw/ConceptARC/corpus",
        "concept_groups": [
            "AboveBelow", "Center", "CleanUp", "CompleteShape",
            "Copy", "Count", "ExtendToBoundary", "ExtractObjects",
            "FilledNotFilled", "HorizontalVertical", "InsideOutside",
            "MoveToBoundary", "Order", "SameDifferent", "TopBottom2D", "TopBottom3D"
        ]
    },
    "grid": {
        "max_grid_height": 30, "max_grid_width": 30,
        "min_grid_height": 1, "min_grid_width": 1,
        "max_colors": 10, "background_color": 0
    },
    "max_train_pairs": 4, "max_test_pairs": 3
})

parser = ConceptArcParser(config)
key = jax.random.PRNGKey(42)

# Get random task from any concept group
task = parser.get_random_task(key)

# Get random task from specific concept group
task = parser.get_random_task_from_concept("Center", key)

# Get all available concept groups
concept_groups = parser.get_concept_groups()
print(f"Available concepts: {concept_groups}")

# Get tasks in a specific concept
center_tasks = parser.get_tasks_in_concept("Center")
print(f"Center concept has {len(center_tasks)} tasks")

# Get specific task by ID
task = parser.get_task_by_id("Center/task_001")

# Get dataset statistics
stats = parser.get_dataset_statistics()
print(f"Total tasks: {stats['total_tasks']}")
print(f"Total concept groups: {stats['total_concept_groups']}")
```

### MiniARC Dataset

MiniARC provides a compact 5×5 grid version optimized for rapid prototyping and testing.

**Repository**: [KSB21ST/MINI-ARC](https://github.com/KSB21ST/MINI-ARC)

**Key Features:**
- **Optimized Performance**: 36x less memory usage than full ARC
- **Rapid Prototyping**: Quick iteration and testing
- **5×5 Grid Constraint**: All tasks fit within 5×5 grids
- **400+ Tasks**: Comprehensive coverage in compact format

**Directory Structure:**
```
data/raw/MiniARC/data/MiniARC/
├── task_001.json
├── task_002.json
├── task_003.json
└── ... (400+ tasks)
```

#### MiniARC Parser Usage

```python
import jax
from jaxarc.parsers import MiniArcParser
from omegaconf import DictConfig

# Configuration optimized for 5×5 grids
config = DictConfig({
    "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
    "grid": {
        "max_grid_height": 5,
        "max_grid_width": 5,
        "min_grid_height": 1,
        "min_grid_width": 1,
        "max_colors": 10,
        "background_color": 0
    },
    "max_train_pairs": 3, "max_test_pairs": 1
})

# Create parser instance (automatically loads and caches all tasks)
parser = MiniArcParser(config)
key = jax.random.PRNGKey(42)

# Get random task (optimized for 5×5 grids)
task = parser.get_random_task(key)

# Get available task IDs
task_ids = parser.get_available_task_ids()
print(f"Available tasks: {len(task_ids)}")

# Get specific task by ID
task = parser.get_task_by_id("task_001")

# Get dataset statistics
stats = parser.get_dataset_statistics()
print(f"Total tasks: {stats['total_tasks']}")
print(f"5×5 optimized: {stats['is_5x5_optimized']}")
print(f"Max dimensions: {stats['max_configured_dimensions']}")
```

## Configuration with Hydra

All parsers can be used with Hydra configuration for easy dataset switching:

```bash
# Run with different datasets
pixi run python scripts/demo_parser.py dataset=arc_agi_2
pixi run python scripts/demo_parser.py dataset=concept_arc
pixi run python scripts/demo_parser.py dataset=mini_arc
```

**Dataset Configuration Files:**
- `conf/dataset/arc_agi_1.yaml` - ARC-AGI-1 (2024 dataset)
- `conf/dataset/arc_agi_2.yaml` - ARC-AGI-2 (2025 dataset)
- `conf/dataset/concept_arc.yaml` - ConceptARC with concept groups
- `conf/dataset/mini_arc.yaml` - MiniARC with 5×5 grids

## Data Format

All datasets share the same basic JSON task structure but with different organizational patterns.

### Common JSON Structure

Each task contains:
- `"train"`: demonstration input/output pairs (list of pairs)
- `"test"`: test input(s) - your model should predict the output(s)

A "pair" contains:
- `"input"`: the input grid (list of lists of integers 0-9)
- `"output"`: the output grid (may be null for test pairs)

### Example Task Structure

```json
{
  "train": [
    {
      "input": [[0, 1, 0], [1, 2, 1], [0, 1, 0]],
      "output": [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
    }
  ],
  "test": [
    {
      "input": [[0, 5, 0], [5, 6, 5], [0, 5, 0]],
      "output": null  // To be predicted
    }
  ]
}
```

### JAX Data Types

JaxARC converts JSON data into JAX-compatible structures:

```python
@chex.dataclass
class JaxArcTask:
    # Training data
    input_grids_examples: jnp.ndarray    # Shape: (max_train_pairs, H, W)
    output_grids_examples: jnp.ndarray   # Shape: (max_train_pairs, H, W)
    num_train_pairs: int

    # Test data  
    test_input_grids: jnp.ndarray        # Shape: (max_test_pairs, H, W)
    true_test_output_grids: jnp.ndarray  # Shape: (max_test_pairs, H, W)
    num_test_pairs: int

    # Metadata
    task_index: jnp.ndarray              # Unique task identifier
```

**Key Features:**
- **Static Shapes**: All arrays padded to maximum dimensions for JIT compilation
- **Masks**: Boolean masks indicate valid data regions
- **Batch Compatible**: Designed for efficient batch processing with `jax.vmap`
- **Type Safety**: Uses `chex.dataclass` for immutable, type-safe structures

## Performance Comparison

| Dataset | Memory Usage | Load Time | Grid Size | Best For |
|---------|--------------|-----------|-----------|----------|
| **MiniARC** | 50 MB | 0.5s | 5×5 | Development, testing |
| **ConceptARC** | 200 MB | 2s | 30×30 | Systematic evaluation |
| **ARC-AGI-1** | 800 MB | 5s | 30×30 | Original challenge |
| **ARC-AGI-2** | 1.2 GB | 8s | 30×30 | Latest challenge |

## Common Issues and Solutions

### Dataset Download Issues

**"Repository not found" or "Permission denied":**
```bash
# Test internet connectivity
ping github.com

# Check repository access
curl -I https://github.com/fchollet/ARC-AGI

# Try with different protocol if behind firewall
git config --global url."https://".insteadOf git://
```

**"git: command not found":**
```bash
# macOS
brew install git

# Ubuntu/Debian
sudo apt update && sudo apt install git

# Verify installation
git --version
```

**Network timeouts:**
```bash
# Increase git timeout settings
git config --global http.lowSpeedTime 300
git config --global http.postBuffer 524288000

# Retry with verbose output
python scripts/download_dataset.py arc-agi-1 --verbose
```

### Parser Configuration Issues

**Legacy Kaggle format detected:**
```python
# OLD (no longer supported)
config = {
    "training": {
        "challenges": "data/raw/arc-prize-2024/arc-agi_training_challenges.json",
        "solutions": "data/raw/arc-prize-2024/arc-agi_training_solutions.json"
    }
}

# NEW (GitHub format)
config = {
    "training": {"path": "data/raw/ARC-AGI-1/data/training"},
    "evaluation": {"path": "data/raw/ARC-AGI-1/data/evaluation"},
    "grid": {"max_grid_height": 30, "max_grid_width": 30}
}
```

**No JSON files found:**
```bash
# Check directory contents
ls -la data/raw/ARC-AGI-1/data/training/

# Should see files like: 007bbfb7.json, 00d62c1b.json, etc.
# If empty, re-download
python scripts/download_dataset.py arc-agi-1 --force
```

### Performance Issues

**Slow task loading:**
```python
# Use MiniARC for development (36x less memory)
from jaxarc.parsers import MiniArcParser

config = DictConfig({
    "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
    "grid": {"max_grid_height": 5, "max_grid_width": 5}
})
parser = MiniArcParser(config)  # Much faster and lighter
```

**Memory issues with large datasets:**
```python
# Limit dataset size
config["max_tasks"] = 100  # Only load first 100 tasks

# Or use lazy loading - only load tasks when needed
task = parser.get_random_task(key)  # Loads single task
```

## Examples and Usage Patterns

### Basic Dataset Exploration

```python
# Quick dataset statistics
stats = parser.get_dataset_statistics()
print(f"Total tasks: {stats['total_tasks']}")

# For ConceptARC, explore concept groups
if hasattr(parser, 'get_concept_groups'):
    for concept in parser.get_concept_groups():
        tasks = parser.get_tasks_in_concept(concept)
        print(f"{concept}: {len(tasks)} tasks")
```

### Batch Processing

```python
# Load multiple tasks efficiently
task_ids = parser.get_available_task_ids()[:10]
tasks = [parser.get_task_by_id(task_id) for task_id in task_ids]

# Use with JAX vmap for batch processing
batched_tasks = jax.tree_map(lambda *x: jnp.stack(x), *tasks)
```

### Integration with Environment

```python
from jaxarc.envs import arc_reset, arc_step, create_standard_config

# Load task and create environment
task = parser.get_random_task(key)
env_config = create_standard_config(max_episode_steps=50)

# Run environment episode
state, obs = arc_reset(key, env_config, task_data=task)
# ... continue with environment steps
```

## Next Steps

- **Getting Started**: See [Getting Started Guide](getting-started.md) for complete setup
- **Configuration**: See [Configuration Guide](configuration.md) for advanced options
- **Examples**: Explore [Examples Directory](examples/) for practical usage patterns
- **API Reference**: See [API Reference](api_reference.md) for complete documentation

---

**Need Help?** Check the troubleshooting section above or visit our [GitHub Issues](https://github.com/aadimator/JaxARC/issues) for support.