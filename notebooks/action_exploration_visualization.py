# %% [markdown]
# # JaxARC Action Exploration and Visualization
#
# This notebook provides a comprehensive exploration of JaxARC's data loading capabilities
# and step-by-step visualization of different actions on ARC-AGI tasks.
#
# ## Key Features:
# - Load and parse ARC-AGI datasets
# - Visualize task structure and examples
# - Initialize environment with random tasks
# - Execute and visualize individual actions
# - Test all 35 available operations
# - Validate implementation correctness
#
# ## Operations Available (0-34):
# - **0-9**: Fill colors (fill selection with color 0-9)
# - **10-19**: Flood fill colors (flood fill from selection with color 0-9)
# - **20-23**: Move object (up, down, left, right)
# - **24-25**: Rotate object (90° clockwise, 90° counterclockwise)
# - **26-27**: Flip object (horizontal, vertical)
# - **28-30**: Clipboard operations (copy, paste, cut)
# - **31-33**: Grid operations (clear, copy input, resize)
# - **34**: Submit (mark as terminated)

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# JaxARC imports
from jaxarc.envs import ArcEnvironment
from jaxarc.parsers import ArcAgiParser
from jaxarc.types import ARCLEAction
from jaxarc.utils.config import get_config
from jaxarc.utils.visualization import (
    log_grid_to_console,
    visualize_parsed_task_data_rich,
)

# Initialize console for rich output
console = Console()

# %% [markdown]
# ## 1. Configuration and Setup
#
# Let's start by loading the configuration and setting up our environment.


# %%
def setup_jaxarc_environment():
    """Set up JaxARC environment with configuration."""
    try:
        config = get_config()
        console.print(
            Panel(
                f"[bold green]Configuration Loaded Successfully[/bold green]\n\n"
                f"Dataset: {config.dataset.dataset_name}\n"
                f"Environment: {config.environment._target_}\n"
                f"Data Root: {config.dataset.data_root}\n"
                f"Max Grid Size: {config.dataset.max_grid_height}x{config.dataset.max_grid_width}",
                title="JaxARC Configuration",
                border_style="green",
            )
        )
        return config
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        console.print("[yellow]Using default configuration...[/yellow]")

        # Create a minimal default configuration
        from omegaconf import OmegaConf

        default_config = OmegaConf.create(
            {
                "dataset": {
                    "dataset_name": "ARC-AGI-1",
                    "dataset_year": 2024,
                    "description": "ARC-AGI-1 dataset (2024) for abstract reasoning tasks",
                    "default_split": "training",
                    "data_root": "data/raw/arc-prize-2024",
                    "training": {
                        "challenges": "data/raw/arc-prize-2024/arc-agi_training_challenges.json",
                        "solutions": "data/raw/arc-prize-2024/arc-agi_training_solutions.json",
                    },
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "max_train_pairs": 10,
                    "max_test_pairs": 3,
                    "parser": {"_target_": "jaxarc.parsers.ArcAgiParser"},
                },
                "environment": {
                    "_target_": "jaxarc.envs.ArcEnvironment",
                    "max_episode_steps": 100,
                    "reward_on_submit_only": True,
                    "step_penalty": -0.01,
                    "success_bonus": 10.0,
                },
            }
        )

        return default_config


# Load configuration
config = setup_jaxarc_environment()

# %% [markdown]
# ## 2. Dataset Loading and Exploration
#
# Now let's load the dataset and explore available tasks.


# %%
def load_and_explore_dataset(config):
    """Load dataset and explore available tasks."""
    try:
        # Initialize the ARC-AGI parser
        parser = ArcAgiParser(config.dataset)

        # Get available task IDs
        available_task_ids = parser.get_available_task_ids()

        console.print(
            Panel(
                f"[bold blue]Dataset Loaded Successfully[/bold blue]\n\n"
                f"Name: {config.dataset.dataset_name}\n"
                f"Year: {config.dataset.dataset_year}\n"
                f"Description: {config.dataset.description}\n"
                f"Available Tasks: {len(available_task_ids)}\n"
                f"Sample Task IDs: {available_task_ids[:5] if available_task_ids else 'None'}",
                title="Dataset Information",
                border_style="blue",
            )
        )

        return parser, available_task_ids

    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        console.print(
            "[yellow]This is expected if you haven't downloaded the ARC-AGI dataset yet.[/yellow]"
        )
        console.print(
            "[yellow]You can still explore the notebook structure and operation definitions.[/yellow]"
        )
        return None, []


# Load dataset
parser, available_task_ids = load_and_explore_dataset(config)

# %% [markdown]
# ## 3. Task Selection and Visualization
#
# Let's select a random task and explore its structure in detail.


# %%
def select_and_visualize_task(parser, key):
    """Select a random task and visualize its structure."""
    if parser is None:
        console.print("[yellow]No parser available - skipping task selection[/yellow]")
        return None, None

    try:
        # Get a random task
        key, task_key = jr.split(key)
        parsed_task = parser.get_random_task(task_key)

        # Extract task ID for display
        from jaxarc.utils.task_manager import extract_task_id_from_index

        task_id = extract_task_id_from_index(parsed_task.task_index)

        console.print(f"[green]Selected task: {task_id}[/green]")

        # Display task structure
        console.print(
            Panel(
                f"[bold cyan]Task Structure[/bold cyan]\n\n"
                f"Task ID: {task_id}\n"
                f"Train Pairs: {parsed_task.num_train_pairs}\n"
                f"Test Pairs: {parsed_task.num_test_pairs}\n"
                f"Input Grids Shape: {parsed_task.input_grids_examples.shape}\n"
                f"Output Grids Shape: {parsed_task.output_grids_examples.shape}\n"
                f"Test Input Shape: {parsed_task.test_input_grids.shape}\n"
                f"Test Output Shape: {parsed_task.true_test_output_grids.shape}",
                title="Task Details",
                border_style="cyan",
            )
        )

        # Visualize the task using rich formatting
        visualize_parsed_task_data_rich(parsed_task, console)

        return parsed_task, task_id

    except Exception as e:
        console.print(f"[red]Error selecting task: {e}[/red]")
        return None, None


# Initialize random key
key = jr.PRNGKey(8)

# Select and visualize a task
parsed_task, task_id = select_and_visualize_task(parser, key)

# %% [markdown]
# ## 4. Environment Initialization
#
# Now let's initialize the ARC environment with our selected task.


# %%
def initialize_environment(config, parsed_task):
    """Initialize ARC environment with a task."""
    if parsed_task is None:
        console.print("[yellow]No task available - creating dummy environment[/yellow]")
        return None

    try:
        # Initialize environment
        env = ArcEnvironment(config.environment, config.dataset)

        console.print(
            Panel(
                f"[bold green]Environment Initialized[/bold green]\n\n"
                f"Max Episode Steps: {env.max_episode_steps}\n"
                f"Max Grid Size: {env.max_grid_size}\n"
                f"Reward on Submit Only: {env.reward_on_submit_only}\n"
                f"Step Penalty: {env.step_penalty}\n"
                f"Success Bonus: {env.success_bonus}",
                title="Environment Configuration",
                border_style="green",
            )
        )

        return env

    except Exception as e:
        console.print(f"[red]Error initializing environment: {e}[/red]")
        return None


# Initialize environment
env = initialize_environment(config, parsed_task)

# %% [markdown]
# ## 5. Action and Operation Definitions
#
# Let's explore all available operations and their definitions.


# %%
def display_operation_definitions():
    """Display comprehensive operation definitions."""

    operations = [
        # Fill Operations (0-9)
        ("Fill Color 0", 0, "Fill selected region with color 0 (black)"),
        ("Fill Color 1", 1, "Fill selected region with color 1 (blue)"),
        ("Fill Color 2", 2, "Fill selected region with color 2 (red)"),
        ("Fill Color 3", 3, "Fill selected region with color 3 (green)"),
        ("Fill Color 4", 4, "Fill selected region with color 4 (yellow)"),
        ("Fill Color 5", 5, "Fill selected region with color 5 (gray)"),
        ("Fill Color 6", 6, "Fill selected region with color 6 (magenta)"),
        ("Fill Color 7", 7, "Fill selected region with color 7 (orange)"),
        ("Fill Color 8", 8, "Fill selected region with color 8 (cyan)"),
        ("Fill Color 9", 9, "Fill selected region with color 9 (brown)"),
        # Flood Fill Operations (10-19)
        ("Flood Fill 0", 10, "Flood fill from selection with color 0"),
        ("Flood Fill 1", 11, "Flood fill from selection with color 1"),
        ("Flood Fill 2", 12, "Flood fill from selection with color 2"),
        ("Flood Fill 3", 13, "Flood fill from selection with color 3"),
        ("Flood Fill 4", 14, "Flood fill from selection with color 4"),
        ("Flood Fill 5", 15, "Flood fill from selection with color 5"),
        ("Flood Fill 6", 16, "Flood fill from selection with color 6"),
        ("Flood Fill 7", 17, "Flood fill from selection with color 7"),
        ("Flood Fill 8", 18, "Flood fill from selection with color 8"),
        ("Flood Fill 9", 19, "Flood fill from selection with color 9"),
        # Movement Operations (20-23)
        ("Move Up", 20, "Move selected object up with wrapping"),
        ("Move Down", 21, "Move selected object down with wrapping"),
        ("Move Left", 22, "Move selected object left with wrapping"),
        ("Move Right", 23, "Move selected object right with wrapping"),
        # Rotation Operations (24-25)
        ("Rotate 90° CW", 24, "Rotate selected region 90° clockwise"),
        ("Rotate 90° CCW", 25, "Rotate selected region 90° counterclockwise"),
        # Flip Operations (26-27)
        ("Flip Horizontal", 26, "Flip selected region horizontally"),
        ("Flip Vertical", 27, "Flip selected region vertically"),
        # Clipboard Operations (28-30)
        ("Copy", 28, "Copy selected region to clipboard"),
        ("Paste", 29, "Paste clipboard content to selected region"),
        ("Cut", 30, "Cut selected region to clipboard (copy + clear)"),
        # Grid Operations (31-33)
        ("Clear", 31, "Clear entire grid or selected region"),
        ("Copy Input", 32, "Copy input grid to working grid"),
        ("Resize", 33, "Resize grid active area based on selection"),
        # Submit Operation (34)
        ("Submit", 34, "Submit current grid as solution"),
    ]

    # Create a table
    table = Table(
        title="Available Operations (0-34)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("ID", style="cyan", justify="center")
    table.add_column("Operation", style="green")
    table.add_column("Description", style="yellow")

    for name, op_id, description in operations:
        table.add_row(str(op_id), name, description)

    console.print(table)


# Display operation definitions
display_operation_definitions()

# %% [markdown]
# ## 6. Action Creation and Execution Utilities
#
# Let's create utilities for creating and executing actions.


# %%
def get_working_grid_bounds(working_grid_mask):
    """Extract the bounds of the working grid from the mask."""
    # Find the bounding box of True values in the mask
    rows_with_data = jnp.any(working_grid_mask, axis=1)
    cols_with_data = jnp.any(working_grid_mask, axis=0)

    min_row = jnp.argmax(rows_with_data)
    max_row = len(rows_with_data) - jnp.argmax(rows_with_data[::-1]) - 1
    min_col = jnp.argmax(cols_with_data)
    max_col = len(cols_with_data) - jnp.argmax(cols_with_data[::-1]) - 1

    return int(min_row), int(max_row), int(min_col), int(max_col)


def extract_working_grid(grid, working_grid_mask):
    """Extract only the working grid area based on the mask."""
    min_row, max_row, min_col, max_col = get_working_grid_bounds(working_grid_mask)
    return grid[min_row : max_row + 1, min_col : max_col + 1]


def create_selection_mask(
    grid_shape,
    working_grid_mask=None,
    selection_type="single_pixel",
    x=0,
    y=0,
    width=1,
    height=1,
):
    """Create different types of selection masks for testing, respecting working grid boundaries."""
    mask = jnp.zeros(grid_shape, dtype=jnp.bool_)

    if working_grid_mask is not None:
        # Get working grid bounds
        min_row, max_row, min_col, max_col = get_working_grid_bounds(working_grid_mask)
        working_height = max_row - min_row + 1
        working_width = max_col - min_col + 1

        # Adjust coordinates to be within working grid
        x = min(x, working_width - 1)
        y = min(y, working_height - 1)
        width = min(width, working_width - x)
        height = min(height, working_height - y)

        # Offset by working grid position
        x += min_col
        y += min_row

        # Constrain to working area
        max_x = min_col + working_width
        max_y = min_row + working_height
    else:
        # Use full grid if no working mask provided
        max_x = grid_shape[1]
        max_y = grid_shape[0]

    if selection_type == "single_pixel":
        if x < max_x and y < max_y:
            mask = mask.at[y, x].set(True)
    elif selection_type == "rectangle":
        end_y = min(y + height, max_y)
        end_x = min(x + width, max_x)
        if y < max_y and x < max_x:
            mask = mask.at[y:end_y, x:end_x].set(True)
    elif selection_type == "entire_grid":
        if working_grid_mask is not None:
            mask = working_grid_mask
        else:
            mask = jnp.ones(grid_shape, dtype=jnp.bool_)
    elif selection_type == "diagonal":
        if working_grid_mask is not None:
            min_row, max_row, min_col, max_col = get_working_grid_bounds(
                working_grid_mask
            )
            diag_size = min(max_row - min_row + 1, max_col - min_col + 1)
            for i in range(diag_size):
                mask = mask.at[min_row + i, min_col + i].set(True)
        else:
            for i in range(min(grid_shape[0], grid_shape[1])):
                mask = mask.at[i, i].set(True)
    elif selection_type == "border":
        if working_grid_mask is not None:
            min_row, max_row, min_col, max_col = get_working_grid_bounds(
                working_grid_mask
            )
            mask = mask.at[min_row, min_col : max_col + 1].set(True)  # Top
            mask = mask.at[max_row, min_col : max_col + 1].set(True)  # Bottom
            mask = mask.at[min_row : max_row + 1, min_col].set(True)  # Left
            mask = mask.at[min_row : max_row + 1, max_col].set(True)  # Right
        else:
            mask = mask.at[0, :].set(True)  # Top
            mask = mask.at[-1, :].set(True)  # Bottom
            mask = mask.at[:, 0].set(True)  # Left
            mask = mask.at[:, -1].set(True)  # Right
    elif selection_type == "none":
        pass  # Keep all False

    # Always constrain to working grid if provided
    if working_grid_mask is not None:
        mask = mask & working_grid_mask

    return mask


def create_action(operation_id, selection_mask, agent_id=0, timestamp=0):
    """Create an ARCLE action with proper formatting."""
    return ARCLEAction(
        selection=selection_mask.astype(jnp.float32),
        operation=jnp.array(operation_id, dtype=jnp.int32),
        agent_id=agent_id,
        timestamp=timestamp,
    )


def visualize_state_comparison(state_before, state_after, operation_name):
    """Visualize state before and after an operation."""
    console.print(f"\n[bold cyan]Operation: {operation_name}[/bold cyan]")

    # Create side-by-side comparison
    try:
        console.print("[bold]Before:[/bold]")
        # Extract only the working grid area for visualization
        before_working_grid = extract_working_grid(
            state_before.working_grid, state_before.working_grid_mask
        )
        log_grid_to_console(
            before_working_grid, title="Before (Working Area)", show_numbers=False
        )
    except Exception as e:
        console.print(f"[red]Error visualizing before grid: {e}[/red]")

    try:
        console.print("[bold]After:[/bold]")
        # Extract only the working grid area for visualization
        after_working_grid = extract_working_grid(
            state_after.working_grid, state_after.working_grid_mask
        )
        log_grid_to_console(
            after_working_grid, title="After (Working Area)", show_numbers=False
        )
    except Exception as e:
        console.print(f"[red]Error visualizing after grid: {e}[/red]")

    # Show similarity score
    console.print(
        f"[bold]Similarity Score: {float(state_after.similarity_score):.4f}[/bold]"
    )

    # Show selection mask if any
    if jnp.sum(state_before.selected) > 0:
        try:
            console.print("[bold]Selection Mask:[/bold]")
            # Extract selection within working area
            selection_working_area = extract_working_grid(
                state_before.selected.astype(jnp.int32), state_before.working_grid_mask
            )
            log_grid_to_console(
                selection_working_area,
                title="Selection (Working Area)",
                show_numbers=False,
            )
        except Exception as e:
            console.print(f"[red]Error visualizing selection: {e}[/red]")


# %% [markdown]
# ## 7. Step-by-Step Action Testing
#
# Now let's test different operations step by step with visualization.


# %%
def test_fill_operations(initial_state, grid_shape):
    """Test fill operations with different colors."""
    console.print(
        Panel("[bold]Testing Fill Operations (0-9)[/bold]", border_style="blue")
    )

    # Create a small rectangular selection within working grid bounds
    selection = create_selection_mask(
        grid_shape,
        initial_state.working_grid_mask,
        "rectangle",
        x=1,
        y=1,
        width=2,
        height=2,
    )

    # Test fill operations
    for color in range(3):  # Test first 3 colors
        action = create_action(color, selection)

        # Execute operation
        from jaxarc.envs.grid_operations import execute_grid_operation

        # Set selection in state before executing operation
        state_with_selection = initial_state.replace(selected=selection)
        new_state = execute_grid_operation(state_with_selection, action.operation)

        visualize_state_comparison(
            state_with_selection, new_state, f"Fill Color {color}"
        )

        # Use this state as starting point for next test
        initial_state = new_state.replace(selected=jnp.zeros_like(selection))


def test_flood_fill_operations(initial_state, grid_shape):
    """Test flood fill operations."""
    console.print(
        Panel(
            "[bold]Testing Flood Fill Operations (10-19)[/bold]", border_style="green"
        )
    )

    # Create a single pixel selection for flood fill within working grid
    selection = create_selection_mask(
        grid_shape, initial_state.working_grid_mask, "single_pixel", x=2, y=2
    )

    # Test flood fill operation
    action = create_action(12, selection)  # Flood fill with color 2 (red)

    from jaxarc.envs.grid_operations import execute_grid_operation

    # Set selection in state before executing operation
    state_with_selection = initial_state.replace(selected=selection)
    new_state = execute_grid_operation(state_with_selection, action.operation)

    visualize_state_comparison(state_with_selection, new_state, "Flood Fill Color 2")

    return new_state


def test_movement_operations(initial_state, grid_shape):
    """Test movement operations."""
    console.print(
        Panel("[bold]Testing Movement Operations (20-23)[/bold]", border_style="yellow")
    )

    # Create selection around an object within working grid
    selection = create_selection_mask(
        grid_shape,
        initial_state.working_grid_mask,
        "rectangle",
        x=1,
        y=1,
        width=2,
        height=2,
    )

    # Test move operations
    for direction, name in [(20, "Up"), (21, "Down"), (22, "Left"), (23, "Right")]:
        action = create_action(direction, selection)

        from jaxarc.envs.grid_operations import execute_grid_operation

        # Set selection in state before executing operation
        state_with_selection = initial_state.replace(selected=selection)
        new_state = execute_grid_operation(state_with_selection, action.operation)

        visualize_state_comparison(state_with_selection, new_state, f"Move {name}")

        # Reset for next test
        initial_state = new_state.replace(selected=jnp.zeros_like(selection))


def test_rotation_operations(initial_state, grid_shape):
    """Test rotation operations."""
    console.print(
        Panel(
            "[bold]Testing Rotation Operations (24-25)[/bold]", border_style="magenta"
        )
    )

    # Create an L-shaped selection within working grid bounds
    min_row, max_row, min_col, max_col = get_working_grid_bounds(
        initial_state.working_grid_mask
    )

    selection = jnp.zeros(grid_shape, dtype=jnp.bool_)
    # Create L-shape within working area
    if max_row - min_row >= 2 and max_col - min_col >= 2:
        selection = selection.at[min_row : min_row + 3, min_col].set(
            True
        )  # Vertical line
        selection = selection.at[min_row + 2, min_col : min_col + 3].set(
            True
        )  # Horizontal line
        # Constrain to working grid
        selection = selection & initial_state.working_grid_mask
    else:
        # Fallback to simple rectangle if working area is too small
        selection = create_selection_mask(
            grid_shape,
            initial_state.working_grid_mask,
            "rectangle",
            x=0,
            y=0,
            width=2,
            height=2,
        )

    # Test rotation operations
    for rotation, name in [(24, "90° Clockwise"), (25, "90° Counterclockwise")]:
        action = create_action(rotation, selection)

        from jaxarc.envs.grid_operations import execute_grid_operation

        # Set selection in state before executing operation
        state_with_selection = initial_state.replace(selected=selection)
        new_state = execute_grid_operation(state_with_selection, action.operation)

        visualize_state_comparison(state_with_selection, new_state, f"{name}")


def test_clipboard_operations(initial_state, grid_shape):
    """Test clipboard operations."""
    console.print(
        Panel("[bold]Testing Clipboard Operations (28-30)[/bold]", border_style="cyan")
    )

    # Create selection for copying within working grid
    copy_selection = create_selection_mask(
        grid_shape,
        initial_state.working_grid_mask,
        "rectangle",
        x=0,
        y=0,
        width=2,
        height=2,
    )

    # Test copy operation
    action = create_action(28, copy_selection)  # Copy

    from jaxarc.envs.grid_operations import execute_grid_operation

    # Set selection in state before executing operation
    state_with_selection = initial_state.replace(selected=copy_selection)
    state_after_copy = execute_grid_operation(state_with_selection, action.operation)

    visualize_state_comparison(
        state_with_selection, state_after_copy, "Copy to Clipboard"
    )

    # Test paste operation - try to paste in a different area within working grid
    min_row, max_row, min_col, max_col = get_working_grid_bounds(
        initial_state.working_grid_mask
    )
    working_width = max_col - min_col + 1
    working_height = max_row - min_row + 1

    # Position paste area away from copy area if there's space
    paste_x = min(2, working_width - 2) if working_width > 4 else 0
    paste_y = min(2, working_height - 2) if working_height > 4 else 0

    paste_selection = create_selection_mask(
        grid_shape,
        initial_state.working_grid_mask,
        "rectangle",
        x=paste_x,
        y=paste_y,
        width=2,
        height=2,
    )
    action = create_action(29, paste_selection)  # Paste

    # Set selection in state before executing operation
    state_with_paste_selection = state_after_copy.replace(selected=paste_selection)
    state_after_paste = execute_grid_operation(
        state_with_paste_selection, action.operation
    )

    visualize_state_comparison(
        state_with_paste_selection, state_after_paste, "Paste from Clipboard"
    )


# %% [markdown]
# ## 8. Create Test Environment State
#
# Let's create a test environment state to run our operations on.


# %%
def create_test_environment_state(parsed_task=None):
    """Create a test environment state for operation testing."""
    if parsed_task is None:
        # Create a dummy task for testing
        grid_shape = (10, 10)

        # Create dummy task data
        from jaxarc.types import JaxArcTask

        parsed_task = JaxArcTask(
            input_grids_examples=jnp.zeros((1, *grid_shape), dtype=jnp.int32),
            input_masks_examples=jnp.ones((1, *grid_shape), dtype=jnp.bool_),
            output_grids_examples=jnp.zeros((1, *grid_shape), dtype=jnp.int32),
            output_masks_examples=jnp.ones((1, *grid_shape), dtype=jnp.bool_),
            num_train_pairs=1,
            test_input_grids=jnp.zeros((1, *grid_shape), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, *grid_shape), dtype=jnp.bool_),
            true_test_output_grids=jnp.zeros((1, *grid_shape), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, *grid_shape), dtype=jnp.bool_),
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Create a meaningful test pattern with smaller working area
        working_size = 8  # Use 8x8 working area instead of full 10x10
        test_grid = jnp.full(grid_shape, -1, dtype=jnp.int32)  # Fill with -1 (padding)
        # Create actual working grid content
        test_grid = test_grid.at[0:working_size, 0:working_size].set(
            0
        )  # Set working area to background
        test_grid = test_grid.at[1:4, 1:4].set(1)  # Blue square
        test_grid = test_grid.at[2, 2].set(2)  # Red center
        test_grid = test_grid.at[0, 0:working_size].set(3)  # Green top border
        test_grid = test_grid.at[0:working_size, 0].set(4)  # Yellow left border
        test_grid = test_grid.at[5:7, 5:7].set(5)  # Gray square in bottom right

        # Create working grid mask for the 8x8 area
        working_grid_mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
        working_grid_mask = working_grid_mask.at[0:working_size, 0:working_size].set(
            True
        )

        target_grid = jnp.full(
            grid_shape, -1, dtype=jnp.int32
        )  # Fill with -1 (padding)
        target_grid = target_grid.at[0:working_size, 0:working_size].set(
            0
        )  # Set working area to background
        target_grid = target_grid.at[2:5, 2:5].set(6)  # Magenta target square
    else:
        # Use real task data but ensure we have a copy for modifications
        grid_shape = parsed_task.input_grids_examples.shape[1:]
        test_grid = jnp.array(parsed_task.input_grids_examples[0])  # Make a copy
        test_mask = jnp.array(parsed_task.input_masks_examples[0])  # Make a copy
        target_grid = jnp.array(parsed_task.output_grids_examples[0])  # Make a copy

    # Create environment state
    from jaxarc.state import ArcEnvState

    # Set up working grid mask
    if parsed_task is None:
        # Use the mask we created above for dummy data
        final_working_mask = working_grid_mask
    else:
        # For real task data, use the input mask
        final_working_mask = parsed_task.input_masks_examples[0]

    state = ArcEnvState(
        task_data=parsed_task,
        working_grid=test_grid,
        working_grid_mask=final_working_mask,
        target_grid=target_grid,
        step_count=0,
        episode_done=False,
        current_example_idx=0,
        selected=jnp.zeros(grid_shape, dtype=jnp.bool_),
        clipboard=jnp.zeros(grid_shape, dtype=jnp.int32),
        similarity_score=jnp.array(0.0, dtype=jnp.float32),
    )

    return state, grid_shape


# Create test state
test_state, grid_shape = create_test_environment_state(parsed_task)

console.print(
    Panel("[bold]Test Environment State Created[/bold]", border_style="green")
)
console.print("[bold]Initial Working Grid:[/bold]")
try:
    # Extract only the working grid area for visualization
    initial_working_grid = extract_working_grid(
        test_state.working_grid, test_state.working_grid_mask
    )
    log_grid_to_console(
        initial_working_grid, title="Initial Working Grid", show_numbers=False
    )
except Exception as e:
    console.print(f"[red]Error displaying initial grid: {e}[/red]")

# %% [markdown]
# ## 8.5. Minimal Test to Isolate Grid Dimension Issue


# %%
def minimal_grid_test():
    """Minimal test to isolate grid dimension issues."""
    console.print(
        Panel("[bold]Running Minimal Grid Test[/bold]", border_style="yellow")
    )

    # Create a simple 2D grid
    simple_grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
    console.print(f"Simple grid shape: {simple_grid.shape}")
    console.print(f"Simple grid ndim: {simple_grid.ndim}")

    # Test visualization directly
    try:
        console.print("[bold]Testing log_grid_to_console directly:[/bold]")
        log_grid_to_console(simple_grid, title="Simple Grid Test", show_numbers=False)
        console.print("[green]✓ Direct visualization works[/green]")
    except Exception as e:
        console.print(f"[red]✗ Direct visualization failed: {e}[/red]")
        return False

    # Test action creation
    try:
        console.print("[bold]Testing action creation:[/bold]")
        selection = jnp.array(
            [[True, False, False], [False, True, False], [False, False, True]],
            dtype=jnp.bool_,
        )
        action = create_action(1, selection)
        console.print(f"Action operation: {action.operation}")
        console.print(f"Action operation shape: {action.operation.shape}")
        console.print("[green]✓ Action creation works[/green]")
    except Exception as e:
        console.print(f"[red]✗ Action creation failed: {e}[/red]")
        return False

    # Test grid operation execution
    try:
        console.print("[bold]Testing grid operation execution:[/bold]")
        from jaxarc.envs.grid_operations import execute_grid_operation
        from jaxarc.state import ArcEnvState

        # Create minimal state
        from jaxarc.types import JaxArcTask

        dummy_task = JaxArcTask(
            input_grids_examples=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            input_masks_examples=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            output_grids_examples=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            output_masks_examples=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            num_train_pairs=1,
            test_input_grids=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            true_test_output_grids=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        minimal_state = ArcEnvState(
            task_data=dummy_task,
            working_grid=simple_grid,
            working_grid_mask=jnp.ones((3, 3), dtype=jnp.bool_),
            target_grid=jnp.zeros((3, 3), dtype=jnp.int32),
            step_count=0,
            episode_done=False,
            current_example_idx=0,
            selected=jnp.zeros((3, 3), dtype=jnp.bool_),
            clipboard=jnp.zeros((3, 3), dtype=jnp.int32),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
        )

        console.print(
            f"Minimal state working_grid shape: {minimal_state.working_grid.shape}"
        )

        # Execute operation
        state_with_selection = minimal_state.replace(selected=selection)
        new_state = execute_grid_operation(state_with_selection, action.operation)
        console.print(f"New state working_grid shape: {new_state.working_grid.shape}")
        console.print("[green]✓ Grid operation execution works[/green]")

        # Test visualization of result - show only working area
        result_working_grid = extract_working_grid(
            new_state.working_grid, new_state.working_grid_mask
        )
        log_grid_to_console(
            result_working_grid,
            title="After Operation (Working Area)",
            show_numbers=False,
        )
        console.print("[green]✓ Result visualization works[/green]")

    except Exception as e:
        console.print(f"[red]✗ Grid operation execution failed: {e}[/red]")
        import traceback

        console.print(f"[red]Full traceback: {traceback.format_exc()}[/red]")
        return False

    return True


# Run minimal test
if minimal_grid_test():
    console.print("[green]✓ Minimal test passed - proceeding with full tests[/green]")
else:
    console.print("[red]✗ Minimal test failed - stopping execution[/red]")

# %% [markdown]
# ## 9. Execute Operation Tests
#
# Now let's run our comprehensive operation tests.

# %%
# Test fill operations
test_fill_operations(test_state, grid_shape)

# %%
# Test flood fill operations
test_state_after_flood = test_flood_fill_operations(test_state, grid_shape)

# %%
# Test movement operations
test_movement_operations(test_state_after_flood, grid_shape)

# %%
# Test rotation operations
test_rotation_operations(test_state_after_flood, grid_shape)

# %%
# Test clipboard operations
test_clipboard_operations(test_state_after_flood, grid_shape)

# %% [markdown]
# ## 10. Grid Operations Testing
#
# Let's test the remaining grid operations.


def test_grid_operations(initial_state, grid_shape):
    """Test grid operations (31-33)."""
    console.print(
        Panel("[bold]Testing Grid Operations (31-33)[/bold]", border_style="red")
    )

    # Test clear operation - create selection within working grid bounds
    selection = create_selection_mask(
        grid_shape,
        initial_state.working_grid_mask,
        "rectangle",
        x=1,
        y=1,
        width=3,
        height=3,
    )
    action = create_action(31, selection)  # Clear

    from jaxarc.envs.grid_operations import execute_grid_operation

    # Set selection in state before executing operation
    state_with_selection = initial_state.replace(selected=selection)
    state_after_clear = execute_grid_operation(state_with_selection, action.operation)

    visualize_state_comparison(
        state_with_selection, state_after_clear, "Clear Selected Region"
    )

    # Test copy input operation
    action = create_action(32, jnp.zeros(grid_shape, dtype=jnp.bool_))  # Copy input

    # Set selection in state before executing operation (empty selection for copy input)
    state_with_empty_selection = state_after_clear.replace(
        selected=jnp.zeros(grid_shape, dtype=jnp.bool_)
    )
    state_after_copy_input = execute_grid_operation(
        state_with_empty_selection, action.operation
    )

    visualize_state_comparison(
        state_after_clear, state_after_copy_input, "Copy Input Grid"
    )

    # Test resize operation - create selection for new working area
    min_row, max_row, min_col, max_col = get_working_grid_bounds(
        initial_state.working_grid_mask
    )
    working_width = max_col - min_col + 1
    working_height = max_row - min_row + 1

    # Create a slightly smaller resize selection
    resize_width = max(2, working_width - 2)
    resize_height = max(2, working_height - 2)
    resize_selection = create_selection_mask(
        grid_shape,
        initial_state.working_grid_mask,
        "rectangle",
        x=1,
        y=1,
        width=resize_width,
        height=resize_height,
    )
    action = create_action(33, resize_selection)  # Resize

    # Set selection in state before executing operation
    state_with_resize_selection = state_after_copy_input.replace(
        selected=resize_selection
    )
    state_after_resize = execute_grid_operation(
        state_with_resize_selection, action.operation
    )

    visualize_state_comparison(
        state_with_resize_selection, state_after_resize, "Resize Working Grid"
    )


# %%

# Test grid operations
test_grid_operations(test_state, grid_shape)

# %% [markdown]
# ## 11. Submit Operation Testing
#
# Let's test the submit operation which marks the episode as done.


# %%
def test_submit_operation(initial_state, grid_shape):
    """Test submit operation (34)."""
    console.print(
        Panel("[bold]Testing Submit Operation (34)[/bold]", border_style="green")
    )

    # Test submit operation
    action = create_action(34, jnp.zeros(grid_shape, dtype=jnp.bool_))  # Submit

    from jaxarc.envs.grid_operations import execute_grid_operation

    state_after_submit = execute_grid_operation(initial_state, action.operation)

    console.print(
        f"[bold]Before Submit - Episode Done: {initial_state.episode_done}[/bold]"
    )
    console.print(
        f"[bold]After Submit - Episode Done: {state_after_submit.episode_done}[/bold]"
    )

    if state_after_submit.episode_done:
        console.print("[green]✓ Submit operation working correctly![/green]")
    else:
        console.print("[red]✗ Submit operation not working![/red]")


# Test submit operation
test_submit_operation(test_state, grid_shape)

# %% [markdown]
# ## 12. Random Action Testing
#
# Let's test random action generation and execution.


# %%
def test_random_actions(initial_state, grid_shape, num_actions=5):
    """Test random action generation and execution."""
    console.print(
        Panel(f"[bold]Testing {num_actions} Random Actions[/bold]", border_style="blue")
    )

    key = jr.PRNGKey(123)
    current_state = initial_state

    for i in range(num_actions):
        key, action_key, op_key, sel_key = jr.split(key, 4)

        # Generate random operation (0-33, excluding submit)
        operation_id = jr.randint(op_key, (), 0, 34)

        # Generate random selection
        selection_type = jr.choice(
            sel_key, jnp.array(["single_pixel", "rectangle", "none"])
        )

        if selection_type == 0:  # single_pixel
            x = jr.randint(action_key, (), 0, grid_shape[1])
            y = jr.randint(action_key, (), 0, grid_shape[0])
            selection = create_selection_mask(
                grid_shape,
                current_state.working_grid_mask,
                "single_pixel",
                x=int(x),
                y=int(y),
            )
        elif selection_type == 1:  # rectangle
            x = jr.randint(action_key, (), 0, grid_shape[1] - 2)
            y = jr.randint(action_key, (), 0, grid_shape[0] - 2)
            width = jr.randint(action_key, (), 1, 4)
            height = jr.randint(action_key, (), 1, 4)
            selection = create_selection_mask(
                grid_shape,
                current_state.working_grid_mask,
                "rectangle",
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
            )
        else:  # none
            selection = create_selection_mask(
                grid_shape, current_state.working_grid_mask, "none"
            )

        # Create and execute action
        action = create_action(int(operation_id), selection)

        from jaxarc.envs.grid_operations import execute_grid_operation

        # Set selection in state before executing operation
        state_with_selection = current_state.replace(selected=selection)
        new_state = execute_grid_operation(state_with_selection, action.operation)

        # Get operation name
        operation_names = {
            0: "Fill Color 0",
            1: "Fill Color 1",
            2: "Fill Color 2",
            3: "Fill Color 3",
            4: "Fill Color 4",
            5: "Fill Color 5",
            6: "Fill Color 6",
            7: "Fill Color 7",
            8: "Fill Color 8",
            9: "Fill Color 9",
            10: "Flood Fill 0",
            11: "Flood Fill 1",
            12: "Flood Fill 2",
            13: "Flood Fill 3",
            14: "Flood Fill 4",
            15: "Flood Fill 5",
            16: "Flood Fill 6",
            17: "Flood Fill 7",
            18: "Flood Fill 8",
            19: "Flood Fill 9",
            20: "Move Up",
            21: "Move Down",
            22: "Move Left",
            23: "Move Right",
            24: "Rotate 90° CW",
            25: "Rotate 90° CCW",
            26: "Flip Horizontal",
            27: "Flip Vertical",
            28: "Copy",
            29: "Paste",
            30: "Cut",
            31: "Clear",
            32: "Copy Input",
            33: "Resize",
        }

        op_name = operation_names.get(int(operation_id), f"Unknown Op {operation_id}")

        console.print(f"\n[bold blue]Random Action {i + 1}: {op_name}[/bold blue]")
        visualize_state_comparison(
            current_state.replace(selected=selection), new_state, op_name
        )

        current_state = new_state.replace(selected=jnp.zeros_like(selection))

        # Break if episode is done
        if new_state.episode_done:
            console.print("[yellow]Episode terminated by submit operation[/yellow]")
            break


# Test random actions
test_random_actions(test_state, grid_shape, num_actions=5)

# %% [markdown]
# ## 13. Implementation Validation
#
# Let's validate that our implementation is working correctly with comprehensive tests.


# %%
def validate_implementation():
    """Validate implementation correctness."""
    console.print(Panel("[bold]Implementation Validation[/bold]", border_style="green"))

    validation_results = []

    # Create test state once for all tests
    test_state, grid_shape = create_test_environment_state()

    # Test 1: Action creation
    try:
        selection = create_selection_mask(
            grid_shape, test_state.working_grid_mask, "single_pixel", x=2, y=2
        )
        action = create_action(1, selection)
        validation_results.append(
            ("Action Creation", "✓ PASS", "Actions created successfully")
        )
    except Exception as e:
        validation_results.append(("Action Creation", "✗ FAIL", f"Error: {e}"))

    # Test 2: Grid operations
    try:
        selection = create_selection_mask(
            grid_shape,
            test_state.working_grid_mask,
            "rectangle",
            x=1,
            y=1,
            width=2,
            height=2,
        )
        action = create_action(2, selection)  # Fill with color 2

        from jaxarc.envs.grid_operations import execute_grid_operation

        # Set selection in state before executing operation
        state_with_selection = test_state.replace(selected=selection)
        new_state = execute_grid_operation(state_with_selection, action.operation)
        validation_results.append(
            ("Grid Operations", "✓ PASS", "Grid operations execute without errors")
        )
    except Exception as e:
        validation_results.append(("Grid Operations", "✗ FAIL", f"Error: {e}"))

    # Test 3: State consistency
    try:
        # Check that state maintains proper structure
        assert hasattr(new_state, "working_grid")
        assert hasattr(new_state, "similarity_score")
        assert hasattr(new_state, "episode_done")
        validation_results.append(
            ("State Consistency", "✓ PASS", "State structure maintained")
        )
    except Exception as e:
        validation_results.append(("State Consistency", "✗ FAIL", f"Error: {e}"))

    # Test 4: JAX compatibility
    try:
        # Test that functions can be JIT compiled
        @jax.jit
        def test_jit(state, operation):
            return execute_grid_operation(state, operation)

        jit_result = test_jit(test_state, jnp.array(1, dtype=jnp.int32))
        validation_results.append(
            ("JAX Compatibility", "✓ PASS", "Functions are JIT compatible")
        )
    except Exception as e:
        validation_results.append(("JAX Compatibility", "✗ FAIL", f"Error: {e}"))

    # Display results
    table = Table(title="Implementation Validation Results", show_header=True)
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Details", style="yellow")

    for test_name, status, details in validation_results:
        if "✓" in status:
            table.add_row(test_name, f"[green]{status}[/green]", details)
        else:
            table.add_row(test_name, f"[red]{status}[/red]", details)

    console.print(table)

    # Summary
    passed = sum(1 for _, status, _ in validation_results if "✓" in status)
    total = len(validation_results)
    console.print(f"\n[bold]Validation Summary: {passed}/{total} tests passed[/bold]")

    if passed == total:
        console.print(
            "[green]✓ All validation tests passed! Implementation is working correctly.[/green]"
        )
    else:
        console.print(
            "[red]✗ Some validation tests failed. Please review implementation.[/red]"
        )


# Run validation
validate_implementation()
