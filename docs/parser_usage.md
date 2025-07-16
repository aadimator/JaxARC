# Parser Usage Guide

The JaxARC project includes multiple parsers for different ARC dataset variants:

- **`ArcAgiParser`**: General parser for ARC-AGI-1 and ARC-AGI-2 datasets from
  GitHub
- **`ConceptArcParser`**: Specialized parser for ConceptARC dataset with concept
  group organization and systematic evaluation
- **`MiniArcParser`**: Optimized parser for MiniARC dataset with 5x5 grids and
  rapid prototyping capabilities

## Quick Start

### 1. Download Data from GitHub

Download the datasets directly from GitHub repositories using the streamlined
download script:

```bash
# Download ARC-AGI datasets from GitHub (no Kaggle CLI required)
python scripts/download_dataset.py arc-agi-1
python scripts/download_dataset.py arc-agi-2

# Download other datasets
python scripts/download_dataset.py conceptarc
python scripts/download_dataset.py miniarc

# Download all datasets at once
python scripts/download_dataset.py all
```

Expected file structure:

```text
data/raw/
├── ARC-AGI-1/
│   └── data/
│       ├── training/
│       │   ├── 007bbfb7.json
│       │   ├── 00d62c1b.json
│       │   └── ... (400 training tasks)
│       └── evaluation/
│           ├── 00576224.json
│           ├── 009d5c81.json
│           └── ... (400 evaluation tasks)
└── ARC-AGI-2/
    └── data/
        ├── training/
        │   ├── 007bbfb7.json
        │   ├── 00d62c1b.json
        │   └── ... (1000 training tasks)
        └── evaluation/
            ├── 00576224.json
            ├── 009d5c81.json
            └── ... (120 evaluation tasks)
```

### 2. Download ConceptARC Dataset

For ConceptARC dataset, clone the repository:

```bash
# Clone ConceptARC repository
git clone https://github.com/victorvikram/ConceptARC.git data/raw/ConceptARC

# Expected structure:
# data/raw/ConceptARC/corpus/{ConceptGroup}/{TaskName}.json
```

### 3. Download MiniARC Dataset

For MiniARC dataset, clone the repository:

```bash
# Clone MiniARC repository
git clone https://github.com/KSB21ST/MINI-ARC.git data/raw/MiniARC

# Expected structure:
# data/raw/MiniARC/data/MiniARC/{TaskName}.json
```

### 4. Automated Dataset Download

Use the enhanced download script for automatic dataset downloading:

```bash
# Download ConceptARC dataset
python scripts/download_dataset.py conceptarc

# Download MiniARC dataset
python scripts/download_dataset.py miniarc

# Download ARC-AGI datasets from GitHub
python scripts/download_dataset.py arc-agi-1
python scripts/download_dataset.py arc-agi-2

# Download all datasets at once
python scripts/download_dataset.py all
```

### 3. Using the Parsers

#### ARC-AGI Parser (Basic Usage)

```python
from jaxarc.parsers import ArcAgiParser

# Create parser instance
parser = ArcAgiParser()

# Parse a specific task from a file
task = parser.parse_task_file("path/to/file.json", "task_id")

# Parse all tasks from a file
tasks = parser.parse_all_tasks_from_file("path/to/file.json")
```

#### ConceptARC Parser (Concept-Based Usage)

```python
import jax
from jaxarc.parsers import ConceptArcParser
from omegaconf import DictConfig

# Create configuration for ConceptARC
config = DictConfig(
    {
        "corpus": {
            "path": "data/raw/ConceptARC/corpus",
            "concept_groups": [
                "AboveBelow",
                "Center",
                "CleanUp",
                "CompleteShape",
                "Copy",
                "Count",
                "ExtendToBoundary",
                "ExtractObjects",
                "FilledNotFilled",
                "HorizontalVertical",
                "InsideOutside",
                "MoveToBoundary",
                "Order",
                "SameDifferent",
                "TopBottom2D",
                "TopBottom3D",
            ],
        },
        "grid": {
            "max_grid_height": 30,
            "max_grid_width": 30,
            "max_colors": 10,
            "background_color": 0,
        },
        "max_train_pairs": 4,
        "max_test_pairs": 3,
    }
)

# Create parser instance
parser = ConceptArcParser(config)

# Get random task from any concept group
key = jax.random.PRNGKey(42)
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

#### MiniARC Parser (Rapid Prototyping Usage)

```python
import jax
from jaxarc.parsers import MiniArcParser
from omegaconf import DictConfig

# Create configuration for MiniARC (optimized for 5x5 grids)
config = DictConfig(
    {
        "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
        "grid": {
            "max_grid_height": 5,
            "max_grid_width": 5,
            "min_grid_height": 1,
            "min_grid_width": 1,
            "max_colors": 10,
            "background_color": 0,
        },
        "max_train_pairs": 3,
        "max_test_pairs": 1,
    }
)

# Create parser instance (automatically loads and caches all tasks)
parser = MiniArcParser(config)

# Get random task (optimized for 5x5 grids)
key = jax.random.PRNGKey(42)
task = parser.get_random_task(key)

# Get available task IDs
task_ids = parser.get_available_task_ids()
print(f"Available tasks: {len(task_ids)}")

# Get specific task by ID (uses filename without extension)
task = parser.get_task_by_id("copy_pattern_5x5")

# Load task directly from file
raw_task_data = parser.load_task_file("data/raw/MiniARC/data/MiniARC/task_001.json")
processed_task = parser.preprocess_task_data(raw_task_data, key)

# Get dataset statistics
stats = parser.get_dataset_statistics()
print(f"Total tasks: {stats['total_tasks']}")
print(f"Optimization: {stats['optimization']}")
print(f"Max dimensions: {stats['max_configured_dimensions']}")
print(f"5x5 optimized: {stats['is_5x5_optimized']}")

# Grid dimension statistics
grid_dims = stats["grid_dimensions"]
print(f"Max grid size: {grid_dims['max_height']}x{grid_dims['max_width']}")
print(f"Avg grid size: {grid_dims['avg_height']:.1f}x{grid_dims['avg_width']:.1f}")

# Training/test pair statistics
print(
    f"Training pairs: {stats['train_pairs']['min']}-{stats['train_pairs']['max']} (avg: {stats['train_pairs']['avg']:.1f})"
)
print(
    f"Test pairs: {stats['test_pairs']['min']}-{stats['test_pairs']['max']} (avg: {stats['test_pairs']['avg']:.1f})"
)
```

#### MiniARC Validation and Error Handling

The MiniARC parser includes comprehensive validation to ensure optimal
performance:

```python
# Configuration validation - warns about suboptimal settings
suboptimal_config = DictConfig(
    {
        "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
        "grid": {
            "max_grid_height": 30,  # Suboptimal for MiniARC
            "max_grid_width": 30,  # Will log warning
        },
        "max_train_pairs": 3,
        "max_test_pairs": 1,
    }
)

# Parser will log: "MiniARC is optimized for 5x5 grids, but configured for 30x30"
parser = MiniArcParser(suboptimal_config)

# Grid constraint validation - automatically rejects oversized tasks
# Tasks with grids exceeding 5x5 are filtered out during loading
# Error logged: "Grid 6x6 exceeds MiniARC 5x5 constraint in task_oversized"

# Handle missing datasets gracefully
missing_config = DictConfig(
    {
        "tasks": {"path": "/nonexistent/path"},
        "grid": {"max_grid_height": 5, "max_grid_width": 5},
        "max_train_pairs": 3,
        "max_test_pairs": 1,
    }
)

parser = MiniArcParser(missing_config)  # Logs warning but doesn't crash
print(f"Tasks loaded: {len(parser.get_available_task_ids())}")  # Returns 0

# Error handling for invalid task access
try:
    task = parser.get_task_by_id("nonexistent_task")
except ValueError as e:
    print(f"Error: {e}")  # "Task ID 'nonexistent_task' not found in MiniARC dataset"

try:
    key = jax.random.PRNGKey(42)
    task = parser.get_random_task(key)  # Raises RuntimeError if no tasks
except RuntimeError as e:
    print(f"Error: {e}")  # "No tasks available in MiniARC dataset"
```

#### Using with Hydra Configuration

All parsers can be used with Hydra configuration to easily switch between
datasets:

```bash
# Run with ARC-AGI-1 (default)
pixi run python scripts/demo_parser.py dataset=arc_agi_1

# Run with ARC-AGI-2
pixi run python scripts/demo_parser.py dataset=arc_agi_2

# Run with ConceptARC
pixi run python scripts/demo_parser.py dataset=concept_arc

# Run with MiniARC
pixi run python scripts/demo_parser.py dataset=mini_arc
```

### 4. Configuration Files

Dataset configurations are provided in `conf/dataset/`:

- `conf/dataset/arc_agi_1.yaml` - For ARC-AGI-1 (2024 dataset)
- `conf/dataset/arc_agi_2.yaml` - For ARC-AGI-2 (2025 dataset)
- `conf/dataset/concept_arc.yaml` - For ConceptARC dataset with concept groups
- `conf/dataset/mini_arc.yaml` - For MiniARC dataset with 5x5 grids

You can easily switch between them by changing the dataset parameter.

### 5. Data Types

All parsers convert JSON data into JAX-compatible data structures:

- `JaxArcTask`: Complete task structure with padded arrays and masks
- `Grid`: 2D color grids as JAX arrays
- Task metadata with concept group information (ConceptARC only)

### 6. Example Scripts

#### ConceptARC Examples

See `examples/conceptarc_usage_example.py` for comprehensive ConceptARC usage:

```bash
# Basic ConceptARC demo
python examples/conceptarc_usage_example.py

# Demonstrate specific concept group
python examples/conceptarc_usage_example.py --concept Center --visualize

# Interactive exploration mode
python examples/conceptarc_usage_example.py --interactive

# Run complete environment episode
python examples/conceptarc_usage_example.py --run-episode --concept Copy
```

#### MiniARC Examples

See `examples/miniarc_usage_example.py` for comprehensive MiniARC usage:

```bash
# Basic MiniARC demo with performance benefits
python examples/miniarc_usage_example.py

# Performance comparison with standard ARC
python examples/miniarc_usage_example.py --performance-comparison

# Rapid prototyping workflow demonstration
python examples/miniarc_usage_example.py --rapid-prototyping --visualize

# Batch processing efficiency demo
python examples/miniarc_usage_example.py --batch-processing --verbose
```

#### General Parser Examples

See `scripts/demo_parser.py` for general parser usage:

```bash
# Run with different datasets
python scripts/demo_parser.py dataset=arc_agi_1
python scripts/demo_parser.py dataset=concept_arc
```

### 7. Advanced ConceptARC Features

#### Concept-Based Analysis

```python
# Analyze concept group characteristics
parser = ConceptArcParser(config)
stats = parser.get_dataset_statistics()

for concept, data in stats["concept_groups"].items():
    print(f"{concept}: {data['num_tasks']} tasks")
    print(f"  Avg demonstrations: {data['avg_demonstrations']:.1f}")
    print(f"  Avg test inputs: {data['avg_test_inputs']:.1f}")
```

#### Systematic Evaluation

```python
# Evaluate model performance by concept group
results = {}
for concept in parser.get_concept_groups():
    concept_tasks = parser.get_tasks_in_concept(concept)
    concept_results = []

    for task_id in concept_tasks:
        task = parser.get_task_by_id(task_id)
        # Run your model evaluation here
        # result = evaluate_model(task)
        # concept_results.append(result)

    results[concept] = concept_results
```

## Backward Compatibility

The original `ArcAgi1Parser` class is still available for backward
compatibility, but the new `ArcAgiParser` is recommended for all new code as it
supports both ARC-AGI-1 and ARC-AGI-2 datasets with the same interface.
