# ARC-AGI Parser Usage Guide

The JaxARC project includes a general `ArcAgiParser` that can work with both
ARC-AGI-1 and ARC-AGI-2 datasets downloaded from Kaggle.

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

### 2. Using the Parser

#### Basic Usage

```python
from jaxarc.parsers import ArcAgiParser

# Create parser instance
parser = ArcAgiParser()

# Parse a specific task from a file
task = parser.parse_task_file("path/to/file.json", "task_id")

# Parse all tasks from a file
tasks = parser.parse_all_tasks_from_file("path/to/file.json")
```

#### Using with Configuration

The parser can be used with Hydra configuration to easily switch between
datasets:

```bash
# Run with ARC-AGI-1 (default)
pixi run python scripts/demo_parser.py

# Run with ARC-AGI-2
pixi run python scripts/demo_parser.py environment=arc_agi_2
```

### 3. Configuration Files

Two environment configurations are provided:

- `conf/environment/arc_agi_1.yaml` - For ARC-AGI-1 (2024 dataset)
- `conf/environment/arc_agi_2.yaml` - For ARC-AGI-2 (2025 dataset)

You can easily switch between them by changing the environment parameter.

### 4. Data Types

The parser converts JSON data into JAX-compatible data structures:

- `Grid`: Represents 2D color grids as JAX arrays
- `TaskPair`: Input-output grid pairs

### 5. Example Script

See `scripts/demo_parser.py` for a complete example of how to:

- Load configuration for different datasets
- Parse tasks from JSON files
- Access task data and metadata

## Backward Compatibility

The original `ArcAgi1Parser` class is still available for backward
compatibility, but the new `ArcAgiParser` is recommended for all new code as it
supports both ARC-AGI-1 and ARC-AGI-2 datasets with the same interface.
