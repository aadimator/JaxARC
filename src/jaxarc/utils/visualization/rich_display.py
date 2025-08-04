"""Rich terminal display functions for JaxARC visualization.

This module provides functions for displaying ARC grids and tasks in the terminal
using the Rich library for enhanced formatting and styling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .constants import ARC_COLOR_PALETTE
from .utils import _extract_grid_data, _extract_valid_region

if TYPE_CHECKING:
    from jaxarc.types import Grid, JaxArcTask
    from jaxarc.utils.task_manager import extract_task_id_from_index


def _get_panel_border_style(border_style: str) -> str:
    """Get panel border style based on border type."""
    if border_style == "input":
        return "blue"
    if border_style == "output":
        return "green"
    return "blue"


def _get_title_style(border_style: str) -> str:
    """Get title style based on border type."""
    if border_style == "input":
        return "bold blue"
    if border_style == "output":
        return "bold green"
    return "bold"


def visualize_grid_rich(
    grid_input: jnp.ndarray | np.ndarray | Grid,
    mask: jnp.ndarray | np.ndarray | None = None,
    title: str = "Grid",
    show_coordinates: bool = False,
    show_numbers: bool = False,
    double_width: bool = True,
    border_style: str = "default",
) -> Table | Panel:
    """Create a Rich Table visualization of a single grid.

    Args:
        grid_input: Grid data (JAX array, numpy array, or Grid object)
        mask: Optional boolean mask indicating valid cells
        title: Title for the table
        show_coordinates: Whether to show row/column coordinates
        show_numbers: If True, show colored numbers; if False, show colored blocks
        double_width: If True and show_numbers=False, use double-width blocks for square appearance
        border_style: Border style - 'input' for blue borders, 'output' for green borders, 'default' for normal

    Returns:
        Rich Table object for display
    """
    grid, grid_mask = _extract_grid_data(grid_input)

    if mask is None:
        mask = grid_mask

    from ..serialization_utils import serialize_jax_array
    
    if mask is not None:
        mask = serialize_jax_array(mask)

    if grid.size == 0:
        table = Table(show_header=False, show_edge=False, show_lines=False, box=None)
        table.add_column("Empty")
        table.add_row("[grey23]Empty grid[/]")

        panel_style = _get_panel_border_style(border_style)
        title_style = _get_title_style(border_style)

        return Panel(
            table,
            title=Text(f"{title} (Empty)", style=title_style),
            border_style=panel_style,
            box=box.ROUNDED if border_style == "input" else box.HEAVY,
            padding=(0, 0),
        )

    # Extract valid region
    valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
        grid, mask
    )

    if height == 0 or width == 0:
        table = Table(show_header=False, show_edge=False, show_lines=False, box=None)
        table.add_column("Empty")
        table.add_row("[grey23]No valid data[/]")

        panel_style = _get_panel_border_style(border_style)
        title_style = _get_title_style(border_style)

        return Panel(
            table,
            title=Text(f"{title} (No valid data)", style=title_style),
            border_style=panel_style,
            box=box.ROUNDED if border_style == "input" else box.HEAVY,
            padding=(0, 0),
        )

    # Create table without borders (will be wrapped in panel)
    table = Table(
        show_header=show_coordinates,
        show_edge=False,
        show_lines=False,
        box=None,
        padding=0,
        pad_edge=False,
    )

    # Add columns
    if show_coordinates:
        table.add_column("", justify="center", width=3)  # Row numbers

    for j in range(width):
        col_header = str(start_col + j) if show_coordinates else ""
        # Adjust column width based on display mode
        col_width = 2  # Single blocks
        table.add_column(col_header, justify="center", width=col_width, no_wrap=True)

    # Add rows
    for i in range(height):
        row_items = []

        if show_coordinates:
            row_items.append(str(start_row + i))

        for j in range(width):
            color_val = int(valid_grid[i, j])

            # Check if this cell is valid (if mask is provided)
            is_valid = True
            if mask is not None:
                actual_row = start_row + i
                actual_col = start_col + j
                if actual_row < mask.shape[0] and actual_col < mask.shape[1]:
                    is_valid = mask[actual_row, actual_col]

            if not is_valid:
                if show_numbers:
                    row_items.append("[grey23]·[/]")
                else:
                    placeholder = "·" if not double_width else "··"
                    row_items.append(f"[grey23]{placeholder}[/]")
            elif show_numbers:
                # Show colored numbers
                rich_color = ARC_COLOR_PALETTE.get(color_val, "white")
                row_items.append(f"[{rich_color}]{color_val}[/]")
            elif double_width:
                # Use double-width blocks for more square appearance
                rich_color = ARC_COLOR_PALETTE.get(color_val, "white")
                row_items.append(f"[{rich_color}]██[/]")
            else:
                # Use single block character
                rich_color = ARC_COLOR_PALETTE.get(color_val, "white")
                row_items.append(f"[{rich_color}]█[/]")

        table.add_row(*row_items)

    # Wrap table in panel with appropriate border style
    panel_style = _get_panel_border_style(border_style)
    title_style = _get_title_style(border_style)

    return Panel(
        table,
        title=Text(f"{title} ({height}x{width})", style=title_style),
        border_style=panel_style,
        box=box.ROUNDED if border_style == "input" else box.HEAVY,
        padding=(0, 0),
    )


def log_grid_to_console(
    grid_input: jnp.ndarray | np.ndarray | Grid,
    mask: jnp.ndarray | np.ndarray | None = None,
    title: str = "Grid",
    show_coordinates: bool = False,
    show_numbers: bool = False,
    double_width: bool = True,
) -> None:
    """Log a grid visualization to the console using Rich.

    This function is designed to be used with jax.debug.callback for logging
    during JAX transformations.

    Args:
        grid_input: Grid data (JAX array, numpy array, or Grid object)
        mask: Optional boolean mask indicating valid cells
        title: Title for the grid display
        show_coordinates: Whether to show row/column coordinates
        show_numbers: If True, show colored numbers; if False, show colored blocks
        double_width: If True and show_numbers=False, use double-width blocks for square appearance
    """
    console = Console()
    table = visualize_grid_rich(
        grid_input, mask, title, show_coordinates, show_numbers, double_width
    )
    console.print(table)


def visualize_task_pair_rich(
    input_grid: jnp.ndarray | np.ndarray | Grid,
    output_grid: jnp.ndarray | np.ndarray | Grid | None = None,
    input_mask: jnp.ndarray | np.ndarray | None = None,
    output_mask: jnp.ndarray | np.ndarray | None = None,
    title: str = "Task Pair",
    show_numbers: bool = False,
    double_width: bool = True,
    console: Console | None = None,
) -> None:
    """Visualize an input-output pair using Rich tables with responsive layout.

    Args:
        input_grid: Input grid data
        output_grid: Output grid data (optional)
        input_mask: Optional mask for input grid
        output_mask: Optional mask for output grid
        title: Title for the visualization
        show_numbers: If True, show colored numbers; if False, show colored blocks
        double_width: If True and show_numbers=False, use double-width blocks for square appearance
        console: Optional Rich console (creates one if None)
    """
    if console is None:
        console = Console()

    # Create input table with blue border
    input_table = visualize_grid_rich(
        input_grid,
        input_mask,
        f"{title} - Input",
        show_numbers=show_numbers,
        double_width=double_width,
        border_style="input",
    )

    # Create output table or placeholder
    if output_grid is not None:
        output_table = visualize_grid_rich(
            output_grid,
            output_mask,
            f"{title} - Output",
            show_numbers=show_numbers,
            double_width=double_width,
            border_style="output",
        )
    else:
        # Create placeholder for unknown output
        output_table = Table(
            show_header=False,
            show_edge=False,
            show_lines=False,
            box=None,
        )
        output_table.add_column("Unknown", justify="center")
        question_text = Text("?", style="bold yellow")
        output_table.add_row(question_text)

        output_table = Panel(
            output_table,
            title=Text(f"{title} - Output", style="bold green"),
            border_style="green",
            box=box.HEAVY,
            padding=(0, 0),
        )

    # Responsive layout based on terminal width
    terminal_width = console.size.width

    # If terminal is wide enough, show side-by-side
    if terminal_width >= 120:
        columns = Columns([input_table, output_table], equal=True, expand=True)
        console.print(columns)
    else:
        # Stack vertically with clear separation
        console.print(input_table)
        arrow_text = Text("↓", justify="center", style="bold")
        console.print(arrow_text)
        console.print(output_table)


def visualize_parsed_task_data_rich(
    task_data: JaxArcTask,
    show_test: bool = True,
    show_coordinates: bool = False,
    show_numbers: bool = False,
    double_width: bool = True,
) -> None:
    """Visualize a JaxArcTask object using Rich console output with enhanced layout and grouping.

    Args:
        task_data: The parsed task data to visualize
        show_test: Whether to show test pairs
        show_coordinates: Whether to show grid coordinates
        show_numbers: If True, show colored numbers; if False, show colored blocks
        double_width: If True and show_numbers=False, use double-width blocks for square appearance
    """
    from jaxarc.utils.task_manager import extract_task_id_from_index

    console = Console()
    terminal_width = console.size.width

    # Enhanced task header with Panel
    task_id = extract_task_id_from_index(task_data.task_index)
    task_title = f"Task: {task_id}"

    # Create properly styled text for task info
    task_info = Text(justify="center")
    task_info.append("Training Examples: ", style="bold")
    task_info.append(str(task_data.num_train_pairs))
    task_info.append("  ")
    task_info.append("Test Examples: ", style="bold")
    task_info.append(str(task_data.num_test_pairs))

    header_panel = Panel(
        task_info,
        title=task_title,
        title_align="left",
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(header_panel)
    console.print()

    # Training examples with visual grouping
    if task_data.num_train_pairs > 0:
        training_content = []

        for i in range(task_data.num_train_pairs):
            # Create input table with input border style
            input_table = visualize_grid_rich(
                task_data.input_grids_examples[i],
                task_data.input_masks_examples[i],
                f"Input {i + 1}",
                show_coordinates,
                show_numbers,
                double_width,
                border_style="input",
            )

            # Create output table with output border style
            output_table = visualize_grid_rich(
                task_data.output_grids_examples[i],
                task_data.output_masks_examples[i],
                f"Output {i + 1}",
                show_coordinates,
                show_numbers,
                double_width,
                border_style="output",
            )

            # Responsive layout for each pair
            if terminal_width >= 120:
                # Side-by-side layout for wide terminals
                pair_layout = Columns(
                    [input_table, output_table], equal=True, expand=True
                )
                training_content.append(pair_layout)
            else:
                # Vertical layout for narrow terminals
                training_content.append(input_table)
                arrow_text = Text("↓", justify="center", style="bold")
                training_content.append(Padding(arrow_text, (0, 0, 1, 0)))
                training_content.append(output_table)

            # Add separator between examples
            if i < task_data.num_train_pairs - 1:
                training_content.append(Rule(style="dim"))

        # Wrap training examples in a blue panel
        training_group = Group(*training_content)
        training_panel = Panel(
            training_group,
            title=f"Training Examples ({task_data.num_train_pairs})",
            title_align="left",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 1),
        )
        console.print(training_panel)

    # Test examples with visual grouping
    if show_test and task_data.num_test_pairs > 0:
        console.print()  # Space between groups
        test_content = []

        for i in range(task_data.num_test_pairs):
            # Create test input table
            test_input_table = visualize_grid_rich(
                task_data.test_input_grids[i],
                task_data.test_input_masks[i],
                f"Test Input {i + 1}",
                show_coordinates,
                show_numbers,
                double_width,
                border_style="input",
            )

            # Create test output table or placeholder
            if (
                i < len(task_data.true_test_output_grids)
                and task_data.true_test_output_grids[i] is not None
            ):
                test_output_table = visualize_grid_rich(
                    task_data.true_test_output_grids[i],
                    task_data.true_test_output_masks[i],
                    f"Test Output {i + 1}",
                    show_coordinates,
                    show_numbers,
                    double_width,
                    border_style="output",
                )
            else:
                # Create placeholder for unknown test output
                test_output_table = Table(
                    show_header=False,
                    show_edge=False,
                    show_lines=False,
                    box=None,
                )
                test_output_table.add_column("Unknown", justify="center")
                question_text = Text("?", style="bold yellow")
                test_output_table.add_row(question_text)

                test_output_table = Panel(
                    test_output_table,
                    title=Text(f"Test Output {i + 1}", style="bold green"),
                    border_style="green",
                    box=box.HEAVY,
                    padding=(0, 0),
                )

            # Responsive layout for each test pair
            if terminal_width >= 120:
                # Side-by-side layout for wide terminals
                pair_layout = Columns(
                    [test_input_table, test_output_table], equal=True, expand=True
                )
                test_content.append(pair_layout)
            else:
                # Vertical layout for narrow terminals
                test_content.append(test_input_table)
                arrow_text = Text("↓", justify="center", style="bold")
                test_content.append(Padding(arrow_text, (0, 0, 1, 0)))
                test_content.append(test_output_table)

            # Add separator between examples
            if i < task_data.num_test_pairs - 1:
                test_content.append(Rule(style="dim"))

        # Wrap test examples in a red panel
        test_group = Group(*test_content)
        test_panel = Panel(
            test_group,
            title=f"Test Examples ({task_data.num_test_pairs})",
            title_align="left",
            border_style="red",
            box=box.ROUNDED,
            padding=(1, 1),
        )
        console.print(test_panel)