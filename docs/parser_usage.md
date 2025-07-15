# Parser Usage Guide

The JaxARC project includes multiple parsers for different ARC dataset variants:

- **`ArcAgiParser`**: General parser for ARC-AGI-1 and ARC-AGI-2 datasets from Kaggle
- **`ConceptArcParser`**: Specialized parser for ConceptARC dataset with concept group organization
- **`MiniArcParser`**: Optimized parser for MiniARC dataset with 5x5 grids

## Quick Start

### 1. Download Data from Kaggle

Download the datasets from Kaggle and place them in your data directory:

```bash
# For ARC-AGI-1 (2024)
# Place files in: data/raw/arc-prize-2024/

# For ARC-AGI-2 (2025)
# Place files in: data/raw/arc-prize-2025/
```

Expected file structure:

```text
data/raw/
├── arc-prize-2024/
│   ├── arc-agi_training_challenges.json
│   ├── arc-agi_training_solutions.json
│   ├── arc-agi_evaluation_challenges.json
│   ├── arc-agi_evaluation_solutions.json
│   ├── arc-agi_test_challenges.json
│   └── sample_submission.json
└── arc-prize-2025/
    ├── arc-agi_training_challenges.json
    ├── arc-agi_training_solutions.json
    ├── arc-agi_evaluation_challenges.json
    ├── arc-agi_evaluation_solutions.json
    ├── arc-agi_test_challenges.json
    └── sample_submission.json
```

### 2. Download ConceptARC Dataset

For ConceptARC dataset, clone the repository:

```bash
# Clone ConceptARC repository
git clone https://github.com/victorvikram/ConceptARC.git data/raw/ConceptARC

# Expected structure:
# data/raw/ConceptARC/corpus/{ConceptGroup}/{TaskName}.json
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
        "max_grid_height": 30,
        "max_grid_width": 30,
        "max_colors": 10,
        "background_color": 0
    },
    "max_train_pairs": 4,
    "max_test_pairs": 3
})

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

See `examples/concept_arc_demo.py` for comprehensive ConceptARC usage:

```bash
# Basic ConceptARC demo
python examples/concept_arc_demo.py

# Demonstrate specific concept group
python examples/concept_arc_demo.py --concept Center

# Show dataset statistics
python examples/concept_arc_demo.py --stats
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

for concept, data in stats['concept_groups'].items():
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
