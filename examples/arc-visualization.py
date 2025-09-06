# %% [markdown]
# # ARC Data Visualization with Hydra Config
#
# This notebook demonstrates comprehensive ARC data loading and visualization using:
# - Hydra configuration management
# - ArcAgiParser for data loading
# - SVG-based visualization utilities
# - Task, task pair, and grid visualization examples

# %% [markdown]
# ## Setup and Imports

# %%
from __future__ import annotations

import jax
import numpy as np
from IPython.display import display
from loguru import logger

# JaxARC imports
from jaxarc.parsers.arc_agi import ArcAgiParser
from jaxarc.utils.core import get_config
from jaxarc.utils.visualization import (
    draw_grid_svg,
    draw_parsed_task_data_svg,
    draw_task_pair_svg,
    visualize_parsed_task_data_rich,
    visualize_task_pair_rich,
)

# Set up logging
logger.info("Starting ARC Visualization Demo")

# %% [markdown]
# ## Configuration Setup with Hydra
#
# Load configuration using Hydra's hierarchical config system.

# %%
# Load default configuration
cfg = get_config()

# Display configuration details
dataset_config = cfg.dataset
logger.info(f"Dataset: {dataset_config.dataset_name} ({dataset_config.dataset_repo})")
logger.info(f"Data path: {dataset_config.dataset_path}")
logger.info(f"Default split: {dataset_config.task_split}")
logger.info(f"Max grid height: {dataset_config.max_grid_height}")
logger.info(f"Max grid width: {dataset_config.max_grid_width}")

print("\n=== Configuration Overview ===")
print(f"Dataset: {dataset_config.dataset_name} ({dataset_config.dataset_repo})")
print(f"Split: {dataset_config.task_split}")
print(
    f"Max Grid Size: {dataset_config.max_grid_height}x{dataset_config.max_grid_width}"
)

# %% [markdown]
# ## Initialize Parser and Load Data
#
# Create the ArcAgiParser with our configuration and load available tasks.

# %%
# Initialize parser with configuration
from jaxarc.configs import DatasetConfig

typed_dataset_config = DatasetConfig.from_hydra(dataset_config)
parser = ArcAgiParser(typed_dataset_config)

# Get available task information
available_tasks = parser.get_available_task_ids()
logger.info(f"Available tasks: {len(available_tasks)}")

# Display first few task IDs
print(f"\nFirst 10 task IDs: {available_tasks[:10]}")
print(f"Total available tasks: {len(available_tasks)}")

# %% [markdown]
# ## Load and Parse Sample Tasks
#
# Load several tasks to demonstrate different visualization capabilities.

# %%
# Set random seed for reproducible results
key = jax.random.PRNGKey(42)

# Load multiple random tasks for demonstration
sample_tasks = []
num_samples = 3

for i in range(num_samples):
    key, subkey = jax.random.split(key)
    try:
        task = parser.get_random_task(subkey)
        sample_tasks.append(task)
        logger.info(
            f"Loaded task {i + 1}: index={task.task_index}, train_pairs={task.num_train_pairs}, test_pairs={task.num_test_pairs}"
        )
    except Exception as e:
        logger.error(f"Error loading task {i + 1}: {e}")

print(f"\nSuccessfully loaded {len(sample_tasks)} sample tasks")

# Display task statistics
for i, task in enumerate(sample_tasks):
    print(f"\nTask {i + 1} Statistics:")
    print(f"  Task Index: {task.task_index}")
    print(f"  Training Pairs: {task.num_train_pairs}")
    print(f"  Test Pairs: {task.num_test_pairs}")
    # print(f"  Input Grids Shape: {task.input_grids_examples.shape}")
    # print(f"  Output Grids Shape: {task.output_grids_examples.shape}")
    # print(f"  Test Input Shape: {task.test_input_grids.shape}")

# %% [markdown]
# ## 1. Individual Grid Visualization
#
# Demonstrate basic grid visualization using SVG rendering.

# %%
if sample_tasks:
    # Get the first task for demonstration
    demo_task = sample_tasks[0]

    print(f"=== Grid Visualization Demo (Task {demo_task.task_index}) ===")

    # Extract first training example input and output grids
    task_pair = demo_task.get_train_pair(0)
    input_grid = task_pair.input_grid
    output_grid = task_pair.output_grid

    print(f"Input grid shape: {input_grid.shape}")
    print(f"Output grid shape: {output_grid.shape}")

    # Create SVG visualizations for individual grids
    print("\n--- Input Grid ---")
    input_svg = draw_grid_svg(input_grid, label="Training Input Grid")
    display(input_svg)

    print("\n--- Output Grid ---")
    output_svg = draw_grid_svg(output_grid, label="Training Output Grid")
    display(output_svg)

    # Show grid values for small grids
    if input_grid.shape[0] <= 10 and input_grid.shape[1] <= 10:
        print("\n--- Grid Values ---")
        print("Input Grid Values:")
        print(np.array(input_grid))
        print("\nOutput Grid Values:")
        print(np.array(output_grid))
else:
    print("No tasks loaded for grid visualization demo")

# %% [markdown]
# ## 2. Task Pair Visualization
#
# Visualize input-output pairs side by side to show the transformation pattern.

# %%
if sample_tasks:
    demo_task = sample_tasks[0]

    print(f"=== Task Pair Visualization Demo (Task {demo_task.task_index}) ===")

    # Visualize multiple training pairs
    num_pairs_to_show = min(3, demo_task.num_train_pairs)

    for pair_idx in range(num_pairs_to_show):
        print(f"\n--- Training Pair {pair_idx + 1} ---")

        task_pair = demo_task.get_train_pair(pair_idx)
        input_grid = task_pair.input_grid
        output_grid = task_pair.output_grid

        # Create side-by-side visualization
        pair_svg = draw_task_pair_svg(
            input_grid=input_grid,
            output_grid=output_grid,
            label=f"Training Pair {pair_idx + 1}",
        )
        display(pair_svg)

        # Show rich terminal visualization as well
        print("Rich Terminal Visualization:")
        visualize_task_pair_rich(
            input_grid=input_grid,
            output_grid=output_grid,
        )

    # Show test input if available
    if demo_task.num_test_pairs > 0:
        print("\n--- Test Input ---")
        test_input = demo_task.get_test_input_grid(0)
        test_svg = draw_grid_svg(test_input, label="Test Input Grid")
        display(test_svg)
else:
    print("No tasks loaded for task pair visualization demo")

# %% [markdown]
# ## 3. Complete Task Visualization
#
# Show the entire task with all training and test examples in a comprehensive layout.

# %%
if sample_tasks:
    demo_task = sample_tasks[0]

    print(f"=== Complete Task Visualization (Task {demo_task.task_index}) ===")

    # Create comprehensive task visualization
    task_svg = draw_parsed_task_data_svg(
        task_data=demo_task,
        include_test=True,  # Don't show test outputs (they're unknown)
    )
    display(task_svg)

    # Also show rich terminal visualization
    print("\n--- Rich Terminal Visualization ---")
    visualize_parsed_task_data_rich(demo_task)
else:
    print("No tasks loaded for complete task visualization demo")

# %% [markdown]
# ## 4. Multiple Tasks Comparison
#
# Compare multiple tasks side by side to see different patterns and complexities.

# %%
if len(sample_tasks) >= 2:
    print("=== Multiple Tasks Comparison ===")

    for i, task in enumerate(sample_tasks[:2]):  # Show first 2 tasks
        print(f"\n--- Task {i + 1} (Index: {task.task_index}) ---")
        print(
            f"Training pairs: {task.num_train_pairs}, Test pairs: {task.num_test_pairs}"
        )

        # Show first training pair for each task
        if task.num_train_pairs > 0:
            input_grid = task.get_train_input_grid(0)
            output_grid = task.get_train_output_grid(0)

            pair_svg = draw_task_pair_svg(
                input_grid=input_grid,
                output_grid=output_grid,
                label=f"Task {task.task_index} - First Training Pair",
            )
            display(pair_svg)
else:
    print("Need at least 2 tasks for comparison demo")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated comprehensive ARC data visualization capabilities:
#
# 1. **Configuration Management**: Using Hydra for flexible configuration
# 2. **Data Loading**: ArcAgiParser for efficient task loading
# 3. **Individual Grid Visualization**: SVG rendering of single grids
# 4. **Task Pair Visualization**: Side-by-side input-output comparisons
# 5. **Complete Task Visualization**: Comprehensive task layouts
# 6. **Multi-Task Comparison**: Comparing different tasks
#
# The visualization utilities provide both SVG (for notebooks) and rich terminal output (for console use), making them suitable for various development and analysis workflows.

# %%
