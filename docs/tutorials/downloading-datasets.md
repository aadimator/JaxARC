# Downloading Datasets

This tutorial explains how to download and access various Abstraction and
Reasoning Corpus (ARC) datasets using JaxARC. It covers automatic downloading,
querying available tasks, working with subsets, and loading specific tasks.

## Supported ARC Datasets

JaxARC supports multiple ARC dataset variants, each designed for different use
cases:

### ARC-AGI-1

The original Abstraction and Reasoning Corpus challenge dataset.

- **Source**: [GitHub - fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI)
- **Size**: 400 training tasks, 400 evaluation tasks
- **Grid Size**: Variable (typically 1x1 to 30x30)
- **Use Case**: Original ARC challenge benchmark

### ARC-AGI-2

Updated version with additional tasks and refinements.

- **Source**:
  [GitHub - arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2)
- **Size**: Expanded task set
- **Grid Size**: Variable
- **Use Case**: Latest ARC challenge iteration

### ConceptARC

Organized by concept groups for systematic evaluation.

- **Source**:
  [GitHub - victorvikram/ConceptARC](https://github.com/victorvikram/ConceptARC)
- **Size**: 16 concept groups
- **Concepts**: Rotation, scaling, color, patterns, etc.
- **Use Case**: Systematic testing of specific reasoning capabilities

### MiniARC

Compact 5x5 grid version for rapid prototyping.

- **Source**: Subset of ARC-AGI
- **Size**: Smaller task set
- **Grid Size**: Fixed 5x5
- **Use Case**: Fast experimentation and debugging

## Automatic Download with `make()` (Recommended)

The easiest way to get started is to let `make()` download the dataset
automatically:

```python
import jax
import jaxarc

# Create environment - downloads dataset automatically if missing
env, env_params = jaxarc.make("Mini", auto_download=True)

# Dataset is now downloaded and ready to use!
key = jax.random.PRNGKey(0)
state, timestep = env.reset(key, env_params=env_params)
print(f"Dataset downloaded and environment ready")
print(f"  Observation shape: {timestep.observation.shape}")
```

**Available dataset keys:**

- `"Mini"` - MiniARC (5x5 grids, compact)
- `"AGI1"` - ARC-AGI-1 (original challenge)
- `"AGI2"` - ARC-AGI-2 (updated version)
- `"Concept"` - ConceptARC (organized by concepts)

**Why use `make()` with `auto_download=True`?**

- Simplest approach - one line gets you started
- Handles dataset paths automatically
- Validates downloaded data
- Ready to use immediately

## Pre-Download Using CLI Script (Optional)

If you prefer to download datasets before running your code, use the CLI script:

```bash
# Download MiniARC
python scripts/download_dataset.py miniarc

# Download ARC-AGI-1
python scripts/download_dataset.py arc_agi_1

# Download ARC-AGI-2
python scripts/download_dataset.py arc_agi_2

# Download ConceptARC
python scripts/download_dataset.py conceptarc

# Download all datasets
python scripts/download_dataset.py all

# Download to custom directory
python scripts/download_dataset.py miniarc --output ./my_data

# Force re-download
python scripts/download_dataset.py miniarc --force
```

## Query Available Tasks

```python
from jaxarc.registration import available_task_ids

# Query available tasks (downloads if needed)
task_ids = available_task_ids("Mini", auto_download=True)
print(f"Available MiniARC tasks: {len(task_ids)}")
print(f"First 5 tasks: {task_ids[:5]}")

# Query without auto-download (requires pre-downloaded dataset)
try:
    agi1_tasks = available_task_ids("AGI1", auto_download=False)
    print(f"ARC-AGI-1 tasks: {len(agi1_tasks)}")
except Exception as e:
    print(f"Dataset not downloaded: {e}")
```

## Work with Named Subsets

Query and use subsets of tasks:

```python
from jaxarc.registration import available_named_subsets, get_subset_task_ids

# See available subsets for a dataset
subsets = available_named_subsets("AGI1")
print(f"AGI1 subsets: {subsets}")
# Output: ('all', 'eval', 'train')

# Get task IDs for a specific subset
train_tasks = get_subset_task_ids("AGI1", "train", auto_download=True)
eval_tasks = get_subset_task_ids("AGI1", "eval", auto_download=True)
print(f"Training tasks: {len(train_tasks)}")
print(f"Evaluation tasks: {len(eval_tasks)}")

# ConceptARC has concept-based subsets
concept_subsets = available_named_subsets("Concept")
print(f"Concept subsets: {concept_subsets[:5]}")
# Output: ('AboveBelow', 'Center', 'CleanUp', 'CompleteShape', 'Copy')

# Get tasks for a specific concept
center_tasks = get_subset_task_ids("Concept", "Center", auto_download=True)
print(f"'Center' concept tasks: {len(center_tasks)}")
```

## Load Specific Tasks

Create environments for specific tasks or subsets:

```python
import jaxarc

# Load a specific task by ID
env, env_params = jaxarc.make(
    "Mini-Most_Common_color_l6ab0lf3xztbyxsu3p", auto_download=True
)
print("Loaded specific task")

# Load train split for AGI1
env, env_params = jaxarc.make("AGI1-train", auto_download=True)
print("Loaded AGI1 training split")

# Load eval split for AGI1
env, env_params = jaxarc.make("AGI1-eval", auto_download=True)
print("Loaded AGI1 evaluation split")

# Load specific concept from ConceptARC
env, env_params = jaxarc.make("Concept-Center", auto_download=True)
print("Loaded Center concept tasks")
```

## Complete Example

Here's a complete script that explores datasets:

```python
#!/usr/bin/env python3
"""
Explore ARC datasets with JaxARC.
"""

import jax
import jaxarc
from jaxarc.registration import (
    available_task_ids,
    available_named_subsets,
    get_subset_task_ids,
)


def explore_dataset(dataset_key="Mini"):
    """Explore an ARC dataset."""

    print(f"=== Exploring {dataset_key} Dataset ===\n")

    # Step 1: Query available subsets
    print("Available subsets:")
    subsets = available_named_subsets(dataset_key)
    print(f"  {subsets}\n")

    # Step 2: Get all available tasks
    print("Querying tasks...")
    task_ids = available_task_ids(dataset_key, auto_download=True)
    print(f"  Total tasks: {len(task_ids)}")
    print(f"  First 5: {task_ids[:5]}\n")

    # Step 3: Create environment for specific task
    print("Loading first task...")
    env, env_params = jaxarc.make(f"{dataset_key}-{task_ids[0]}", auto_download=True)
    print(f"  Environment created\n")

    # Step 4: Reset and explore
    print("Testing environment:")
    key = jax.random.PRNGKey(42)
    state, timestep = env.reset(key, env_params=env_params)

    print(f"  Observation shape: {timestep.observation.shape}")
    print(f"  Step type: {timestep.step_type}")
    print(f"  Initial reward: {timestep.reward}")

    # Step 5: Take a random action
    action_space = env.action_space(env_params)
    action = action_space.sample(key)
    next_state, next_timestep = env.step(state, action, env_params=env_params)

    print(f"  After step - reward: {next_timestep.reward}")
    print(f"  Environment working correctly\n")

    return env, env_params, task_ids


if __name__ == "__main__":
    # Try different datasets
    for dataset_key in ["Mini", "AGI1"]:
        print("=" * 60)
        try:
            explore_dataset(dataset_key)
        except Exception as e:
            print(f"Error exploring {dataset_key}: {e}")
        print()
```

## Custom Subsets

Register your own subset of tasks for curriculum learning or benchmarking:

```python
from jaxarc.registration import register_subset, get_subset_task_ids
import jaxarc

# Get available tasks
all_tasks = get_subset_task_ids("Mini", "all", auto_download=True)

# Create custom subset (e.g., first 10 tasks for quick testing)
quick_test_tasks = all_tasks[:10]
register_subset("Mini", "quick", quick_test_tasks)

# Use your custom subset
env, env_params = jaxarc.make("Mini-quick", auto_download=True)
print(f"Created environment with {len(quick_test_tasks)} tasks")

# Verify it worked
loaded_tasks = get_subset_task_ids("Mini", "quick")
print(f"Quick subset has {len(loaded_tasks)} tasks")
```

## Common Issues

### Issue: "Dataset not found"

**Cause**: Dataset not downloaded and `auto_download=False`.

**Solution**: Enable auto-download:

```python
# Enable auto-download
env, env_params = jaxarc.make("Mini", auto_download=True)

# Or pre-download with CLI
# python scripts/download_dataset.py miniarc
```

### Issue: "Task ID not found"

**Cause**: Typo in task ID or task doesn't exist in that dataset.

**Solution**: Query available tasks first:

```python
from jaxarc.registration import available_task_ids

# List all available tasks
tasks = available_task_ids("Mini", auto_download=True)
print(f"Available tasks: {tasks}")

# Use exact task ID
task_id = tasks[0]  # Use an actual task ID
env, env_params = jaxarc.make(f"Mini-{task_id}", auto_download=True)
```
