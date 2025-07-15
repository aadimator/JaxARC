# Data Format Documentation

JaxARC supports multiple ARC dataset variants with different formats and characteristics. This document describes the data formats for each supported dataset type.

## Overview

When looking at a task, a "test-taker" has access to inputs and outputs of the
demonstration pairs (train pairs), plus the input(s) of the test pair(s). The
goal is to construct the output grid(s) corresponding to the test input grid(s).
"Constructing the output grid" involves picking the height and width of the output grid, 
then filling each cell in the grid with a symbol (integer between 0 and 9, which are 
visualized as colors). Only **exact** solutions (all cells match the expected answer) 
can be said to be correct.

## Supported Datasets

JaxARC supports four main dataset variants:

1. **ARC-AGI-1 (2024)**: Original Kaggle competition dataset
2. **ARC-AGI-2 (2025)**: Updated Kaggle competition dataset  
3. **ConceptARC**: Concept-organized dataset for systematic evaluation
4. **MiniARC**: Compact 5x5 grid version for rapid prototyping

For additional information about the ARC challenge, visit [ARCPrize.org](http://arcprize.org/play).

## ARC-AGI Dataset Format (Kaggle Competition)

### File Structure

The ARC-AGI datasets (both 2024 and 2025) store information in separate files:

- **arc-agi_training-challenges.json**: Training task demonstrations
- **arc-agi_training-solutions.json**: Training task solutions (ground truth)
- **arc-agi_evaluation-challenges.json**: Evaluation task demonstrations  
- **arc-agi_evaluation-solutions.json**: Evaluation task solutions (ground truth)
- **arc-agi_test-challenges.json**: Test tasks for leaderboard evaluation
- **sample_submission.json**: Submission format example

### Directory Structure
```
data/raw/arc-prize-2024/  # or arc-prize-2025/
├── arc-agi_training_challenges.json
├── arc-agi_training_solutions.json
├── arc-agi_evaluation_challenges.json
├── arc-agi_evaluation_solutions.json
├── arc-agi_test_challenges.json
└── sample_submission.json
```

## ConceptARC Dataset Format

### File Structure

ConceptARC organizes tasks into 16 concept groups with hierarchical directory structure:

```
data/raw/ConceptARC/corpus/
├── AboveBelow/
│   ├── task_001.json
│   ├── task_002.json
│   └── ... (10 tasks per concept)
├── Center/
│   ├── task_001.json
│   └── ...
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

### Concept Groups

ConceptARC includes 16 systematic concept groups:

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

### Task Characteristics

- **Demonstrations**: 1-4 training pairs per task
- **Test Inputs**: 3 test inputs per task  
- **Grid Sizes**: Standard ARC dimensions (up to 30x30)
- **Format**: Same JSON structure as ARC-AGI

## MiniARC Dataset Format

### File Structure

MiniARC uses a flat directory structure with individual task files:

```
data/raw/MiniARC/data/MiniARC/
├── task_001.json
├── task_002.json
├── task_003.json
└── ... (400+ tasks)
```

### Characteristics

- **Grid Size**: Optimized for 5x5 grids
- **Tasks**: 400+ individual tasks
- **Format**: Standard ARC JSON format
- **Purpose**: Rapid prototyping and testing

## Common JSON Structure

All datasets share the same basic JSON task structure:

### Task Format

Each task contains a dictionary with two fields:

- `"train"`: demonstration input/output pairs (list of pairs)
- `"test"`: test input(s) - your model should predict the output(s)

### Pair Format

A "pair" is a dictionary with two fields:

- `"input"`: the input "grid" for the pair
- `"output"`: the output "grid" for the pair (may be null for test pairs)

### Grid Format

A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive):

- **Smallest grid**: 1x1
- **Largest grid**: 30x30 (5x5 for MiniARC)
- **Colors**: Integers 0-9 representing different colors
- **Background**: Typically 0 (black)

### Example JSON Structure

```json
{
  "task_id": {
    "train": [
      {
        "input": [[0, 1, 0], [1, 2, 1], [0, 1, 0]],
        "output": [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
      },
      {
        "input": [[0, 3, 0], [3, 4, 3], [0, 3, 0]],
        "output": [[4, 4, 4], [4, 4, 4], [4, 4, 4]]
      }
    ],
    "test": [
      {
        "input": [[0, 5, 0], [5, 6, 5], [0, 5, 0]],
        "output": null  // To be predicted
      }
    ]
  }
}
```

## JAX Data Types

JaxARC converts JSON data into JAX-compatible structures:

### JaxArcTask

The main data structure used throughout JaxARC:

```python
@chex.dataclass
class JaxArcTask:
    # Training data
    input_grids_examples: jnp.ndarray      # Shape: (max_train_pairs, H, W)
    input_masks_examples: jnp.ndarray      # Shape: (max_train_pairs, H, W)
    output_grids_examples: jnp.ndarray     # Shape: (max_train_pairs, H, W)
    output_masks_examples: jnp.ndarray     # Shape: (max_train_pairs, H, W)
    num_train_pairs: int
    
    # Test data
    test_input_grids: jnp.ndarray          # Shape: (max_test_pairs, H, W)
    test_input_masks: jnp.ndarray          # Shape: (max_test_pairs, H, W)
    true_test_output_grids: jnp.ndarray    # Shape: (max_test_pairs, H, W)
    true_test_output_masks: jnp.ndarray    # Shape: (max_test_pairs, H, W)
    num_test_pairs: int
    
    # Metadata
    task_index: jnp.ndarray                # Unique task identifier
```

### Key Features

- **Static Shapes**: All arrays are padded to maximum dimensions for JIT compilation
- **Masks**: Boolean masks indicate valid data regions
- **Batch Compatible**: Designed for efficient batch processing with `jax.vmap`
- **Type Safety**: Uses `chex.dataclass` for immutable, type-safe structures

### Dataset-Specific Characteristics

| Dataset | Max Train Pairs | Max Test Pairs | Max Grid Size | Special Features |
|---------|----------------|----------------|---------------|------------------|
| **ARC-AGI-1/2** | 10 | 3 | 30x30 | Kaggle competition format |
| **ConceptARC** | 4 | 3 | 30x30 | Concept group organization |
| **MiniARC** | 3 | 1 | 5x5 | Optimized for speed |

## Usage Notes

- **Exact Solutions**: Only exact matches (all cells correct) are considered valid
- **Color Range**: All datasets use colors 0-9 (visualized as different colors)
- **Padding**: Smaller grids are padded with -1 and masked appropriately
- **JAX Compatibility**: All data structures support JIT compilation and vectorization
