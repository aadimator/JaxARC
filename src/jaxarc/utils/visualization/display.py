"""
Display functionality for tasks and episodes.

Handles both terminal (Rich) and file-based (SVG) visualization.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, List, Optional

import drawsvg  # type: ignore[import-untyped]
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

from jaxarc.utils.serialization_utils import serialize_jax_array
from jaxarc.utils.task_manager import extract_task_id_from_index

from .core import (
    ARC_COLOR_PALETTE,
    _draw_dotted_squircle,
    _extract_grid_data,
    _extract_valid_region,
    draw_grid_svg,
)

if TYPE_CHECKING:
    from jaxarc.types import Grid, GridArray, JaxArcTask


# ============================================================================
# SECTION: Rich Terminal Display (from rich_display.py)
# ============================================================================


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


# ============================================================================
# SECTION: Task Visualization (from task_visualization.py)
# ============================================================================


def draw_task_pair_svg(
    input_grid: jnp.ndarray | np.ndarray | Grid,
    output_grid: jnp.ndarray | np.ndarray | Grid | None = None,
    input_mask: jnp.ndarray | np.ndarray | None = None,
    output_mask: jnp.ndarray | np.ndarray | None = None,
    width: float = 15.0,
    height: float = 8.0,
    label: str = "",
    show_unknown_output: bool = True,
) -> drawsvg.Drawing:
    """Draw an input-output task pair as SVG with strict height and flexible width.

    Args:
        input_grid: Input grid data
        output_grid: Output grid data (optional)
        input_mask: Optional mask for input grid
        output_mask: Optional mask for output grid
        width: Maximum width for the drawing (actual width may be less)
        height: Strict height constraint - all content must fit within this height
        label: Label for the pair
        show_unknown_output: Whether to show "?" for missing output

    Returns:
        SVG Drawing object
    """
    padding = 0.5
    extra_bottom_padding = 0.25
    io_gap = 0.4

    # Calculate available space for grids - height is STRICT
    ymax = (height - padding - extra_bottom_padding - io_gap) / 2

    # Calculate aspect ratios to determine width requirements
    input_grid_data, input_mask_data = _extract_grid_data(input_grid)
    if input_mask is not None:
        input_mask_data = np.asarray(input_mask)

    _, _, (input_h, input_w) = _extract_valid_region(input_grid_data, input_mask_data)

    input_ratio = input_w / input_h if input_h > 0 else 1.0
    max_ratio = input_ratio

    if output_grid is not None:
        output_grid_data, output_mask_data = _extract_grid_data(output_grid)
        if output_mask is not None:
            output_mask_data = np.asarray(output_mask)
        _, _, (output_h, output_w) = _extract_valid_region(
            output_grid_data, output_mask_data
        )

        output_ratio = output_w / output_h if output_h > 0 else 1.0
        max_ratio = max(input_ratio, output_ratio)

    # Calculate required width based on height constraint and aspect ratio
    required_width = ymax * max_ratio + padding * 2
    final_width = max(required_width, padding * 2 + 1.0)  # Minimum width

    # Don't exceed specified width constraint
    final_width = min(final_width, width)

    max_grid_width = final_width - padding * 2

    # Draw elements following two-pass approach
    drawlist = []
    x_ptr = 0.0
    y_ptr = 0.0

    # First pass: Draw input grid and determine dimensions
    input_result = draw_grid_svg(
        input_grid,
        input_mask,
        max_width=max_grid_width,
        max_height=ymax,
        label=f"{label} Input" if label else "Input",
        padding=padding,
        extra_bottom_padding=extra_bottom_padding,
        as_group=True,
    )

    if isinstance(input_result, tuple):
        input_group, input_origin, input_size = input_result
    else:
        msg = "Expected tuple result when as_group=True"
        raise ValueError(msg)

    # Calculate output dimensions for spacing
    actual_output_width = 0.0
    output_y_total_height = 0.0
    output_g = None
    output_origin_out = (-padding / 2, -padding / 2)

    if output_grid is not None:
        output_result = draw_grid_svg(
            output_grid,
            output_mask,
            max_width=max_grid_width,
            max_height=ymax,
            label=f"{label} Output" if label else "Output",
            padding=padding,
            extra_bottom_padding=extra_bottom_padding,
            as_group=True,
        )

        if isinstance(output_result, tuple):
            output_g, output_origin_out, output_size = output_result
            actual_output_width = output_size[0]
            output_y_total_height = output_size[1]
        else:
            msg = "Expected tuple result when as_group=True"
            raise ValueError(msg)
    else:
        # Approximate height for '?' slot
        output_y_total_height = ymax + padding + extra_bottom_padding

    # Position input grid
    drawlist.append(
        drawsvg.Use(
            input_group,
            x=(max_grid_width + padding - input_size[0]) / 2 - input_origin[0],
            y=-input_origin[1],
        )
    )

    x_ptr += max(input_size[0], actual_output_width)
    y_ptr = max(y_ptr, input_size[1])

    # Second pass: Draw arrow and output
    arrow_x_center = input_size[0] / 2
    arrow_top_y = y_ptr + padding - 0.6
    arrow_bottom_y = y_ptr + padding + io_gap - 0.6

    drawlist.append(
        drawsvg.Line(
            arrow_x_center,
            arrow_top_y,
            arrow_x_center,
            arrow_bottom_y,
            stroke_width=0.05,
            stroke="#888888",
        )
    )
    drawlist.append(
        drawsvg.Line(
            arrow_x_center - 0.15,
            arrow_bottom_y - 0.2,
            arrow_x_center,
            arrow_bottom_y,
            stroke_width=0.05,
            stroke="#888888",
        )
    )
    drawlist.append(
        drawsvg.Line(
            arrow_x_center + 0.15,
            arrow_bottom_y - 0.2,
            arrow_x_center,
            arrow_bottom_y,
            stroke_width=0.05,
            stroke="#888888",
        )
    )

    # Position output
    y_content_top_output_area = y_ptr + io_gap

    if output_g is not None:
        drawlist.append(
            drawsvg.Use(
                output_g,
                x=(max_grid_width + padding - actual_output_width) / 2
                - output_origin_out[0],
                y=y_ptr - output_origin_out[1] + io_gap,
            )
        )
    elif show_unknown_output:
        # Draw question mark for unknown output
        q_text_y_center = (
            y_content_top_output_area + (ymax / 2) + extra_bottom_padding / 2
        )
        drawlist.append(
            drawsvg.Text(
                "?",
                x=(max_grid_width + padding) / 2,
                y=q_text_y_center,
                font_size=1.0,
                font_family="Anuphan",
                font_weight="700",
                fill="#333333",
                text_anchor="middle",
                alignment_baseline="middle",
            )
        )

    y_ptr2 = y_ptr + io_gap + output_y_total_height

    # Calculate final drawing dimensions
    final_drawing_width = max(x_ptr, final_width)
    final_drawing_height = max(y_ptr2, height)  # Height is strict

    # Create final drawing
    drawing = drawsvg.Drawing(
        final_drawing_width, final_drawing_height + 0.3, origin=(0, 0)
    )
    drawing.append(drawsvg.Rectangle(0, 0, "100%", "100%", fill="#eeeff6"))

    # Add all draw elements
    for item in drawlist:
        drawing.append(item)

    # Embed font and set scale
    drawing.embed_google_font(
        "Anuphan:wght@400;600;700",
        text=set(
            "Input Output 0123456789x Test Task ABCDEFGHIJ? abcdefghjklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ),
    )
    drawing.set_pixel_scale(40)

    return drawing


def draw_parsed_task_data_svg(
    task_data: JaxArcTask,
    width: float = 30.0,
    height: float = 20.0,
    include_test: bool | str = False,
    border_colors: list[str] | None = None,
) -> drawsvg.Drawing:
    """Draw a complete JaxArcTask as an SVG with strict height and flexible width.

    Args:
        task_data: The parsed task data to visualize
        width: Maximum width for the drawing (actual width may be less)
        height: Strict height constraint - all content must fit within this height
        include_test: Whether to include test examples. If 'all', show test outputs too.
        border_colors: Custom border colors [input_color, output_color]

    Returns:
        SVG Drawing object
    """
    from jaxarc.utils.task_manager import extract_task_id_from_index

    if border_colors is None:
        border_colors = ["#111111ff", "#111111ff"]

    padding = 0.5
    extra_bottom_padding = 0.25
    io_gap = 0.4

    # Calculate available space for grids - height is STRICT
    ymax = (height - padding - extra_bottom_padding - io_gap) / 2

    # Prepare examples list
    examples = []

    # Add training examples
    for i in range(task_data.num_train_pairs):
        examples.append(
            (
                task_data.input_grids_examples[i],
                task_data.output_grids_examples[i],
                task_data.input_masks_examples[i],
                task_data.output_masks_examples[i],
                f"{i + 1}",
                False,  # is_test
            )
        )

    # Add test examples
    if include_test:
        for i in range(task_data.num_test_pairs):
            show_test_output = include_test == "all"
            output_grid = (
                task_data.true_test_output_grids[i] if show_test_output else None
            )
            output_mask = (
                task_data.true_test_output_masks[i] if show_test_output else None
            )

            examples.append(
                (
                    task_data.test_input_grids[i],
                    output_grid,
                    task_data.test_input_masks[i],
                    output_mask,
                    f"{i + 1}",
                    True,  # is_test
                )
            )

    if not examples:
        # Empty task
        drawing = drawsvg.Drawing(width, height, origin=(0, 0))
        drawing.append(drawsvg.Rectangle(0, 0, "100%", "100%", fill="#eeeff6"))
        drawing.append(
            drawsvg.Text(
                f"Task {extract_task_id_from_index(task_data.task_index)} (No examples)",
                x=width / 2,
                y=height / 2,
                font_size=0.5,
                text_anchor="middle",
                fill="black",
            )
        )
        drawing.set_pixel_scale(40)
        return drawing

    # Prepare training examples
    train_examples = []
    for i in range(task_data.num_train_pairs):
        train_examples.append(
            (
                task_data.input_grids_examples[i],
                task_data.output_grids_examples[i],
                task_data.input_masks_examples[i],
                task_data.output_masks_examples[i],
                f"{i + 1}",
                False,  # is_test
            )
        )

    # Prepare test examples
    test_examples = []
    if include_test:
        for i in range(task_data.num_test_pairs):
            show_test_output = include_test == "all"
            output_grid = (
                task_data.true_test_output_grids[i] if show_test_output else None
            )
            output_mask = (
                task_data.true_test_output_masks[i] if show_test_output else None
            )

            test_examples.append(
                (
                    task_data.test_input_grids[i],
                    output_grid,
                    task_data.test_input_masks[i],
                    output_mask,
                    f"{i + 1}",
                    True,  # is_test
                )
            )

    # Combine all examples
    examples = train_examples + test_examples

    # Calculate ideal width for each example based on aspect ratio and height constraint
    max_widths = np.zeros(len(examples))

    for i, (
        input_grid,
        output_grid,
        input_mask,
        output_mask,
        _label,
        _is_test,
    ) in enumerate(examples):
        input_grid_data, _ = _extract_grid_data(input_grid)
        input_mask_data = np.asarray(input_mask) if input_mask is not None else None
        _, _, (input_h, input_w) = _extract_valid_region(
            input_grid_data, input_mask_data
        )

        input_ratio = input_w / input_h if input_h > 0 else 1.0
        max_ratio = input_ratio

        if output_grid is not None:
            output_grid_data, _ = _extract_grid_data(output_grid)
            output_mask_data = (
                np.asarray(output_mask) if output_mask is not None else None
            )
            _, _, (output_h, output_w) = _extract_valid_region(
                output_grid_data, output_mask_data
            )

            output_ratio = output_w / output_h if output_h > 0 else 1.0
            max_ratio = max(input_ratio, output_ratio)

        # Calculate ideal width based on height constraint and aspect ratio
        xmax_for_pair = ymax * max_ratio
        max_widths[i] = xmax_for_pair

    # Add extra spacing between training and test groups
    group_spacing = 0.5 if len(train_examples) > 0 and len(test_examples) > 0 else 0.0

    # Proportional allocation algorithm - distribute width based on needs
    paddingless_width = width - padding * len(examples) - group_spacing
    allocation = np.zeros_like(max_widths)
    increment = 0.01

    if paddingless_width > 0 and len(examples) > 0:
        if np.any(max_widths > 0):
            for _ in range(int(paddingless_width // increment)):
                incr_mask = (allocation + increment) <= max_widths
                if incr_mask.sum() > 0:
                    allocation[incr_mask] += increment / incr_mask.sum()
                else:
                    break

        # Fallback: equal distribution if no progress made
        if np.sum(allocation) == 0:
            allocation[:] = paddingless_width / len(examples)

    # Two-pass rendering following reference implementation pattern
    drawlist = []

    # Account for squircle margins in positioning if we have grouping
    squircle_margin = 0.15
    has_grouping = len(train_examples) > 0 and len(test_examples) > 0
    x_offset = squircle_margin if has_grouping else 0.0
    y_offset = squircle_margin if has_grouping else 0.0

    # Calculate group boundaries
    train_width = (
        sum(allocation[: len(train_examples)]) + padding * len(train_examples)
        if train_examples
        else 0
    )
    test_start_x = x_offset + train_width + (group_spacing if has_grouping else 0)

    x_ptr = x_offset
    y_ptr = y_offset

    # First pass: Draw input grids and calculate input row height
    for i, (
        input_grid,
        output_grid,
        input_mask,
        output_mask,
        label,
        is_test,
    ) in enumerate(examples):
        input_result = draw_grid_svg(
            input_grid,
            input_mask,
            max_width=allocation[i],
            max_height=ymax,
            label=f"In #{label}",
            border_color=border_colors[0],
            padding=padding,
            extra_bottom_padding=extra_bottom_padding,
            as_group=True,
        )

        if isinstance(input_result, tuple):
            input_group, input_origin, input_size = input_result
        else:
            msg = "Expected tuple result when as_group=True"
            raise ValueError(msg)

        # Calculate actual output width for spacing
        actual_output_width = 0.0
        if output_grid is not None:
            output_result_for_spacing = draw_grid_svg(
                output_grid,
                output_mask,
                max_width=allocation[i],
                max_height=ymax,
                label=f"Out #{label}",
                border_color=border_colors[1],
                padding=padding,
                extra_bottom_padding=extra_bottom_padding,
                as_group=True,
            )
            if isinstance(output_result_for_spacing, tuple):
                _, _, (actual_output_width, _) = output_result_for_spacing

        # Determine x position based on whether this is a test example
        if is_test and has_grouping:
            # For test examples, position relative to test start
            test_index = i - len(train_examples)
            test_x_offset = (
                sum(allocation[len(train_examples) : len(train_examples) + test_index])
                + padding * test_index
            )
            current_x_ptr = test_start_x + test_x_offset
        else:
            # For training examples, use current x_ptr
            current_x_ptr = x_ptr

        # Position input grid
        drawlist.append(
            drawsvg.Use(
                input_group,
                x=current_x_ptr
                + (allocation[i] + padding - input_size[0]) / 2
                - input_origin[0],
                y=y_offset - input_origin[1],
            )
        )

        # Only advance x_ptr for training examples or when not grouping
        if not is_test or not has_grouping:
            x_ptr += max(input_size[0], actual_output_width)

        y_ptr = max(y_ptr, input_size[1])

    # Second pass: Draw arrows and outputs
    y_ptr2 = y_offset

    for i, (
        input_grid,
        output_grid,
        input_mask,
        output_mask,
        label,
        is_test,
    ) in enumerate(examples):
        # Recalculate input for positioning
        input_result = draw_grid_svg(
            input_grid,
            input_mask,
            max_width=allocation[i],
            max_height=ymax,
            label=f"In #{label}",
            border_color=border_colors[0],
            padding=padding,
            extra_bottom_padding=extra_bottom_padding,
            as_group=True,
        )

        if isinstance(input_result, tuple):
            input_group, input_origin, input_size = input_result
        else:
            msg = "Expected tuple result when as_group=True"
            raise ValueError(msg)

        output_g = None
        output_x_recalc = 0.0
        output_y_total_height = 0.0
        output_origin_recalc = (-padding / 2, -padding / 2)

        show_output = (not is_test) or (include_test == "all")

        if show_output and output_grid is not None:
            output_result = draw_grid_svg(
                output_grid,
                output_mask,
                max_width=allocation[i],
                max_height=ymax,
                label=f"Out #{label}",
                border_color=border_colors[1],
                padding=padding,
                extra_bottom_padding=extra_bottom_padding,
                as_group=True,
            )

            if isinstance(output_result, tuple):
                output_g, output_origin_recalc, output_size = output_result
                output_x_recalc = output_size[0]
                output_y_total_height = output_size[1]
            else:
                msg = "Expected tuple result when as_group=True"
                raise ValueError(msg)
        else:
            # Approximate height for '?' slot
            output_y_total_height = ymax + padding + extra_bottom_padding

        # Determine x position based on whether this is a test example
        if is_test and has_grouping:
            # For test examples, position relative to test start
            test_index = i - len(train_examples)
            test_x_offset = (
                sum(allocation[len(train_examples) : len(train_examples) + test_index])
                + padding * test_index
            )
            current_x_ptr = test_start_x + test_x_offset
        else:
            # For training examples, calculate position from start
            train_x_offset = sum(allocation[:i]) + padding * i
            current_x_ptr = x_offset + train_x_offset

        # Draw arrow
        arrow_x_center = current_x_ptr + input_size[0] / 2
        arrow_top_y = y_ptr + padding - 0.6
        arrow_bottom_y = y_ptr + padding + io_gap - 0.6

        drawlist.append(
            drawsvg.Line(
                arrow_x_center,
                arrow_top_y,
                arrow_x_center,
                arrow_bottom_y,
                stroke_width=0.05,
                stroke="#888888",
            )
        )
        drawlist.append(
            drawsvg.Line(
                arrow_x_center - 0.15,
                arrow_bottom_y - 0.2,
                arrow_x_center,
                arrow_bottom_y,
                stroke_width=0.05,
                stroke="#888888",
            )
        )
        drawlist.append(
            drawsvg.Line(
                arrow_x_center + 0.15,
                arrow_bottom_y - 0.2,
                arrow_x_center,
                arrow_bottom_y,
                stroke_width=0.05,
                stroke="#888888",
            )
        )

        # Position output
        y_content_top_output_area = y_ptr + io_gap

        if show_output and output_g is not None:
            drawlist.append(
                drawsvg.Use(
                    output_g,
                    x=current_x_ptr
                    + (allocation[i] + padding - output_x_recalc) / 2
                    - output_origin_recalc[0],
                    y=y_ptr - output_origin_recalc[1] + io_gap,
                )
            )
        else:
            # Draw question mark
            q_text_y_center = (
                y_content_top_output_area + (ymax / 2) + extra_bottom_padding / 2
            )
            drawlist.append(
                drawsvg.Text(
                    "?",
                    x=current_x_ptr + (allocation[i] + padding) / 2,
                    y=q_text_y_center,
                    font_size=1.0,
                    font_family="Anuphan",
                    font_weight="700",
                    fill="#333333",
                    text_anchor="middle",
                    alignment_baseline="middle",
                )
            )

        y_ptr2 = max(y_ptr2, y_ptr + io_gap + output_y_total_height)

    # Calculate final drawing dimensions accounting for squircle margins
    if has_grouping:
        test_width = (
            sum(allocation[len(train_examples) :]) + padding * len(test_examples)
            if test_examples
            else 0
        )
        final_drawing_width = round(
            x_offset + train_width + group_spacing + test_width + squircle_margin, 1
        )
    else:
        final_drawing_width = round(x_ptr, 1)
    final_drawing_height = round(y_ptr2 + (squircle_margin if has_grouping else 0), 1)

    # Ensure dimensions are not negative or too small
    final_drawing_width = max(final_drawing_width, 1.0)
    final_drawing_height = max(final_drawing_height, height)  # Height is strict

    # Create final drawing with calculated dimensions
    drawing = drawsvg.Drawing(
        final_drawing_width, final_drawing_height + 0.3, origin=(0, 0)
    )
    drawing.append(drawsvg.Rectangle(0, 0, "100%", "100%", fill="#eeeff6"))

    # Add all draw elements
    for item in drawlist:
        drawing.append(item)

    # Add grouping squircles if we have both training and test examples
    if len(train_examples) > 0 and len(test_examples) > 0:
        # Calculate training group bounds
        train_width = sum(allocation[: len(train_examples)]) + padding * len(
            train_examples
        )

        # Training group squircle
        train_squircle_elements = _draw_dotted_squircle(
            x=0,
            y=0,
            width=train_width + squircle_margin * 2,
            height=y_ptr2 - y_offset + squircle_margin,
            label="Train",
            stroke_color="#4A90E2",
        )
        for element in train_squircle_elements:
            drawing.append(element)

        # Test group squircle
        test_start_x = train_width + group_spacing + squircle_margin
        test_width = sum(allocation[len(train_examples) :]) + padding * len(
            test_examples
        )
        test_squircle_elements = _draw_dotted_squircle(
            x=test_start_x,
            y=0,
            width=test_width + squircle_margin,
            height=y_ptr2 - y_offset + squircle_margin,
            label="Test",
            stroke_color="#E94B3C",
        )
        for element in test_squircle_elements:
            drawing.append(element)

    # Add title
    font_size = 0.3
    title_text = f"Task: {extract_task_id_from_index(task_data.task_index)}"
    drawing.append(
        drawsvg.Text(
            title_text,
            x=final_drawing_width - 0.1,
            y=final_drawing_height + 0.2,
            font_size=font_size,
            font_family="Anuphan",
            font_weight="600",
            fill="#666666",
            text_anchor="end",
            alignment_baseline="bottom",
        )
    )

    # Embed font and set scale
    drawing.embed_google_font(
        "Anuphan:wght@400;600;700",
        text=set(
            "Input Output 0123456789x Test Task ABCDEFGHIJ? abcdefghjklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ),
    )
    drawing.set_pixel_scale(40)

    return drawing


# ============================================================================
# SECTION: Episode Visualization (from episode_visualization.py)
# ============================================================================


def draw_enhanced_episode_summary_svg(
    summary_data: Any,
    step_data: List[Any],
    config: Optional[Any] = None,
    width: float = 1400.0,
    height: float = 1000.0,
) -> str:
    """Generate enhanced SVG visualization of episode summary with comprehensive metrics.

    This enhanced version includes:
    - Reward progression chart with key moments highlighted
    - Similarity progression chart
    - Grid state thumbnails at key moments
    - Performance metrics panel
    - Success/failure analysis

    Args:
        summary_data: Episode summary data
        step_data: List of step visualization data
        config: Optional visualization configuration
        width: Width of the visualization
        height: Height of the visualization

    Returns:
        SVG string containing the enhanced episode summary
    """
    import drawsvg as draw

    # Create main drawing
    drawing = draw.Drawing(width, height)
    drawing.append(draw.Rectangle(0, 0, width, height, fill="#f8f9fa"))

    # Layout parameters
    padding = 40
    title_height = 100
    metrics_height = 80
    chart_height = 250
    thumbnails_height = 200
    remaining_height = (
        height
        - title_height
        - metrics_height
        - 2 * chart_height
        - thumbnails_height
        - 6 * padding
    )

    # Add enhanced title section
    title_bg_height = title_height - 20
    drawing.append(
        draw.Rectangle(
            padding,
            padding,
            width - 2 * padding,
            title_bg_height,
            fill="#ffffff",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Main title
    title_text = f"Episode {summary_data.episode_num} Summary"
    drawing.append(
        draw.Text(
            title_text,
            font_size=32,
            x=width / 2,
            y=padding + 40,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Success indicator
    success_color = "#27ae60" if summary_data.success else "#e74c3c"
    success_text = "SUCCESS" if summary_data.success else "FAILED"
    drawing.append(
        draw.Text(
            success_text,
            font_size=18,
            x=width - padding - 20,
            y=padding + 30,
            text_anchor="end",
            font_family="Anuphan",
            font_weight="700",
            fill=success_color,
        )
    )

    # Task ID
    drawing.append(
        draw.Text(
            f"Task: {summary_data.task_id}",
            font_size=16,
            x=padding + 20,
            y=padding + 70,
            text_anchor="start",
            font_family="Anuphan",
            font_weight="400",
            fill="#6c757d",
        )
    )

    # Metrics panel
    metrics_y = title_height + 2 * padding
    drawing.append(
        draw.Rectangle(
            padding,
            metrics_y,
            width - 2 * padding,
            metrics_height,
            fill="#ffffff",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Metrics grid
    metrics = [
        ("Total Steps", summary_data.total_steps, ""),
        ("Total Reward", summary_data.total_reward, ".3f"),
        ("Final Similarity", summary_data.final_similarity, ".3f"),
        (
            "Avg Reward/Step",
            summary_data.total_reward / max(summary_data.total_steps, 1),
            ".3f",
        ),
    ]

    metric_width = (width - 2 * padding - 60) / len(metrics)
    for i, (name, value, fmt) in enumerate(metrics):
        x_pos = padding + 20 + i * metric_width

        # Metric name
        drawing.append(
            draw.Text(
                name,
                font_size=14,
                x=x_pos,
                y=metrics_y + 25,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="600",
                fill="#495057",
            )
        )

        # Metric value
        value_text = f"{value:{fmt}}" if fmt else str(value)
        drawing.append(
            draw.Text(
                value_text,
                font_size=18,
                x=x_pos,
                y=metrics_y + 50,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="500",
                fill="#2c3e50",
            )
        )

    # Reward progression chart
    chart1_y = metrics_y + metrics_height + padding
    chart_width = (width - 3 * padding) / 2

    drawing.append(
        draw.Rectangle(
            padding,
            chart1_y,
            chart_width,
            chart_height,
            fill="white",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Chart title
    drawing.append(
        draw.Text(
            "Reward Progression",
            font_size=18,
            x=padding + 20,
            y=chart1_y + 30,
            text_anchor="start",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Draw reward progression line
    if summary_data.reward_progression and len(summary_data.reward_progression) > 1:
        rewards = summary_data.reward_progression
        chart_inner_width = chart_width - 40
        chart_inner_height = chart_height - 80

        max_reward = max(rewards) if max(rewards) > 0 else 1
        min_reward = min(rewards) if min(rewards) < 0 else 0
        reward_range = max_reward - min_reward if max_reward != min_reward else 1

        # Draw grid lines
        for i in range(5):
            y_grid = chart1_y + 50 + i * (chart_inner_height / 4)
            drawing.append(
                draw.Line(
                    padding + 20,
                    y_grid,
                    padding + 20 + chart_inner_width,
                    y_grid,
                    stroke="#e9ecef",
                    stroke_width=1,
                )
            )

        # Draw reward line
        points = []
        for i, reward in enumerate(rewards):
            x = padding + 20 + (i / (len(rewards) - 1)) * chart_inner_width
            y = (
                chart1_y
                + 50
                + chart_inner_height
                - ((reward - min_reward) / reward_range) * chart_inner_height
            )
            points.append((x, y))

        if len(points) > 1:
            path_data = f"M {points[0][0]} {points[0][1]}"
            for x, y in points[1:]:
                path_data += f" L {x} {y}"

            drawing.append(
                draw.Path(
                    d=path_data,
                    stroke="#3498db",
                    stroke_width=3,
                    fill="none",
                )
            )

            # Add points
            for i, (x, y) in enumerate(points):
                # Highlight key moments
                if i in summary_data.key_moments:
                    drawing.append(
                        draw.Circle(
                            x,
                            y,
                            6,
                            fill="#e74c3c",
                            stroke="white",
                            stroke_width=2,
                        )
                    )
                else:
                    drawing.append(
                        draw.Circle(
                            x,
                            y,
                            4,
                            fill="#3498db",
                            stroke="white",
                            stroke_width=1,
                        )
                    )

    # Similarity progression chart
    chart2_x = padding + chart_width + padding

    drawing.append(
        draw.Rectangle(
            chart2_x,
            chart1_y,
            chart_width,
            chart_height,
            fill="white",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Chart title
    drawing.append(
        draw.Text(
            "Similarity Progression",
            font_size=18,
            x=chart2_x + 20,
            y=chart1_y + 30,
            text_anchor="start",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Draw similarity progression line
    if (
        summary_data.similarity_progression
        and len(summary_data.similarity_progression) > 1
    ):
        similarities = summary_data.similarity_progression
        chart_inner_width = chart_width - 40
        chart_inner_height = chart_height - 80

        # Draw grid lines
        for i in range(5):
            y_grid = chart1_y + 50 + i * (chart_inner_height / 4)
            drawing.append(
                draw.Line(
                    chart2_x + 20,
                    y_grid,
                    chart2_x + 20 + chart_inner_width,
                    y_grid,
                    stroke="#e9ecef",
                    stroke_width=1,
                )
            )

        # Draw similarity line
        points = []
        for i, similarity in enumerate(similarities):
            x = chart2_x + 20 + (i / (len(similarities) - 1)) * chart_inner_width
            y = chart1_y + 50 + chart_inner_height - (similarity * chart_inner_height)
            points.append((x, y))

        if len(points) > 1:
            path_data = f"M {points[0][0]} {points[0][1]}"
            for x, y in points[1:]:
                path_data += f" L {x} {y}"

            drawing.append(
                draw.Path(
                    d=path_data,
                    stroke="#27ae60",
                    stroke_width=3,
                    fill="none",
                )
            )

            # Add points
            for i, (x, y) in enumerate(points):
                # Highlight key moments
                if i in summary_data.key_moments:
                    drawing.append(
                        draw.Circle(
                            x,
                            y,
                            6,
                            fill="#e74c3c",
                            stroke="white",
                            stroke_width=2,
                        )
                    )
                else:
                    drawing.append(
                        draw.Circle(
                            x,
                            y,
                            4,
                            fill="#27ae60",
                            stroke="white",
                            stroke_width=1,
                        )
                    )

    # Key moments thumbnails section
    thumbnails_y = chart1_y + chart_height + padding

    if summary_data.key_moments and step_data:
        drawing.append(
            draw.Rectangle(
                padding,
                thumbnails_y,
                width - 2 * padding,
                thumbnails_height,
                fill="white",
                stroke="#dee2e6",
                stroke_width=1,
                rx=8,
            )
        )

        # Section title
        drawing.append(
            draw.Text(
                "Key Moments",
                font_size=18,
                x=padding + 20,
                y=thumbnails_y + 30,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="600",
                fill="#2c3e50",
            )
        )

        # Draw thumbnails for key moments
        thumbnail_size = 120
        thumbnail_spacing = 20
        thumbnails_per_row = min(len(summary_data.key_moments), 8)

        for i, step_idx in enumerate(summary_data.key_moments[:thumbnails_per_row]):
            if step_idx < len(step_data):
                step = step_data[step_idx]

                thumb_x = padding + 20 + i * (thumbnail_size + thumbnail_spacing)
                thumb_y = thumbnails_y + 50

                # Draw thumbnail background
                drawing.append(
                    draw.Rectangle(
                        thumb_x,
                        thumb_y,
                        thumbnail_size,
                        thumbnail_size,
                        fill="#f8f9fa",
                        stroke="#dee2e6",
                        stroke_width=1,
                        rx=4,
                    )
                )

                # Draw simplified grid representation
                if hasattr(step, "after_grid"):
                    grid_data = np.asarray(step.after_grid.data)
                    grid_size = min(thumbnail_size - 20, 80)
                    cell_size = grid_size / max(grid_data.shape)

                    for row in range(min(grid_data.shape[0], 8)):
                        for col in range(min(grid_data.shape[1], 8)):
                            color_val = int(grid_data[row, col])
                            if config and hasattr(config, "get_color_palette"):
                                color_palette = config.get_color_palette()
                            else:
                                color_palette = ARC_COLOR_PALETTE

                            fill_color = color_palette.get(color_val, "#CCCCCC")

                            drawing.append(
                                draw.Rectangle(
                                    thumb_x + 10 + col * cell_size,
                                    thumb_y + 10 + row * cell_size,
                                    cell_size,
                                    cell_size,
                                    fill=fill_color,
                                    stroke="#6c757d",
                                    stroke_width=0.5,
                                )
                            )

                # Add step label
                drawing.append(
                    draw.Text(
                        f"Step {step_idx}",
                        font_size=12,
                        x=thumb_x + thumbnail_size / 2,
                        y=thumb_y + thumbnail_size + 15,
                        text_anchor="middle",
                        font_family="Anuphan",
                        font_weight="500",
                        fill="#495057",
                    )
                )

    # Add footer with timing information
    footer_y = height - 40
    if hasattr(summary_data, "start_time") and hasattr(summary_data, "end_time"):
        duration = summary_data.end_time - summary_data.start_time
        footer_text = f"Duration: {duration:.1f}s | Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        footer_text = f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"

    drawing.append(
        draw.Text(
            footer_text,
            font_size=12,
            x=width / 2,
            y=footer_y,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="300",
            fill="#adb5bd",
        )
    )

    return drawing.as_svg()


def draw_episode_summary_svg(
    summary_data: Any,
    step_data: List[Any],
    config: Optional[Any] = None,
    width: float = 1400.0,
    height: float = 1000.0,
) -> str:
    """Generate episode summary visualization (enhanced version)."""
    return draw_enhanced_episode_summary_svg(
        summary_data=summary_data,
        step_data=step_data,
        config=config,
        width=width,
        height=height,
    )


def create_episode_comparison_visualization(
    episodes_data: List[Any],
    comparison_type: str = "reward_progression",
    width: float = 1200.0,
    height: float = 600.0,
) -> str:
    """Create comparison visualization across multiple episodes.

    Args:
        episodes_data: List of episode summary data
        comparison_type: Type of comparison ("reward_progression", "similarity", "performance")
        width: Width of the visualization
        height: Height of the visualization

    Returns:
        SVG string containing the comparison visualization
    """
    import drawsvg as draw

    # Create main drawing
    drawing = draw.Drawing(width, height)
    drawing.append(draw.Rectangle(0, 0, width, height, fill="#f8f9fa"))

    # Layout parameters
    padding = 40
    title_height = 80
    chart_height = height - title_height - 2 * padding - 60

    # Add title
    title_text = f"Episode Comparison - {comparison_type.replace('_', ' ').title()}"
    drawing.append(
        draw.Text(
            title_text,
            font_size=28,
            x=width / 2,
            y=50,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Chart area
    chart_y = title_height + padding
    chart_width = width - 2 * padding

    drawing.append(
        draw.Rectangle(
            padding,
            chart_y,
            chart_width,
            chart_height,
            fill="white",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Colors for different episodes
    episode_colors = ["#3498db", "#e74c3c", "#27ae60", "#f39c12", "#9b59b6", "#1abc9c"]

    if comparison_type == "reward_progression":
        # Draw reward progression for each episode
        chart_inner_width = chart_width - 60
        chart_inner_height = chart_height - 60

        # Find global min/max for scaling
        all_rewards = []
        for episode in episodes_data:
            if hasattr(episode, "reward_progression") and episode.reward_progression:
                all_rewards.extend(episode.reward_progression)

        if all_rewards:
            max_reward = max(all_rewards)
            min_reward = min(all_rewards)
            reward_range = max_reward - min_reward if max_reward != min_reward else 1

            # Draw grid lines
            for i in range(5):
                y_grid = chart_y + 30 + i * (chart_inner_height / 4)
                drawing.append(
                    draw.Line(
                        padding + 30,
                        y_grid,
                        padding + 30 + chart_inner_width,
                        y_grid,
                        stroke="#e9ecef",
                        stroke_width=1,
                    )
                )

            # Draw each episode's progression
            for ep_idx, episode in enumerate(episodes_data[: len(episode_colors)]):
                if (
                    hasattr(episode, "reward_progression")
                    and episode.reward_progression
                ):
                    rewards = episode.reward_progression
                    color = episode_colors[ep_idx]

                    points = []
                    for i, reward in enumerate(rewards):
                        x = padding + 30 + (i / (len(rewards) - 1)) * chart_inner_width
                        y = (
                            chart_y
                            + 30
                            + chart_inner_height
                            - ((reward - min_reward) / reward_range)
                            * chart_inner_height
                        )
                        points.append((x, y))

                    if len(points) > 1:
                        path_data = f"M {points[0][0]} {points[0][1]}"
                        for x, y in points[1:]:
                            path_data += f" L {x} {y}"

                        drawing.append(
                            draw.Path(
                                d=path_data,
                                stroke=color,
                                stroke_width=2,
                                fill="none",
                            )
                        )

                        # Add points
                        for x, y in points:
                            drawing.append(
                                draw.Circle(
                                    x,
                                    y,
                                    3,
                                    fill=color,
                                    stroke="white",
                                    stroke_width=1,
                                )
                            )

                    # Add legend entry
                    legend_y = chart_y + chart_height + 20 + ep_idx * 20
                    drawing.append(
                        draw.Line(
                            padding + 20,
                            legend_y,
                            padding + 40,
                            legend_y,
                            stroke=color,
                            stroke_width=3,
                        )
                    )
                    drawing.append(
                        draw.Text(
                            f"Episode {episode.episode_num}",
                            font_size=14,
                            x=padding + 50,
                            y=legend_y + 5,
                            text_anchor="start",
                            font_family="Anuphan",
                            font_weight="400",
                            fill="#495057",
                        )
                    )

    elif comparison_type == "performance":
        # Create bar chart comparing final performance metrics
        metrics = ["total_reward", "final_similarity", "total_steps"]
        metric_labels = ["Total Reward", "Final Similarity", "Steps"]

        chart_inner_width = chart_width - 60
        chart_inner_height = chart_height - 60

        bar_width = (chart_width - 100) / (
            len(episodes_data) * len(metrics) + len(metrics)
        )
        group_spacing = bar_width * 0.5

        for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            # Get values for this metric
            values = []
            for episode in episodes_data:
                if hasattr(episode, metric):
                    values.append(getattr(episode, metric))
                else:
                    values.append(0)

            if values:
                max_val = max(values) if max(values) > 0 else 1

                # Draw bars for this metric
                for ep_idx, (episode, value) in enumerate(zip(episodes_data, values)):
                    x = (
                        padding
                        + 30
                        + metric_idx * (len(episodes_data) * bar_width + group_spacing)
                        + ep_idx * bar_width
                    )
                    bar_height = (value / max_val) * (chart_inner_height - 40)
                    y = chart_y + chart_height - 30 - bar_height

                    color = episode_colors[ep_idx % len(episode_colors)]

                    drawing.append(
                        draw.Rectangle(
                            x,
                            y,
                            bar_width * 0.8,
                            bar_height,
                            fill=color,
                            stroke="white",
                            stroke_width=1,
                        )
                    )

                # Add metric label
                label_x = (
                    padding
                    + 30
                    + metric_idx * (len(episodes_data) * bar_width + group_spacing)
                    + (len(episodes_data) * bar_width) / 2
                )
                drawing.append(
                    draw.Text(
                        label,
                        font_size=12,
                        x=label_x,
                        y=chart_y + chart_height - 10,
                        text_anchor="middle",
                        font_family="Anuphan",
                        font_weight="500",
                        fill="#495057",
                    )
                )

    return drawing.as_svg()


def display_grid(
    grid: GridArray | np.ndarray | Grid, title: str = "Grid", show_mask: bool = True
) -> None:
    """Display a single grid using Rich."""
    console = Console()
    # Note: visualize_grid_rich currently uses the mask embedded in grid if present.
    # show_mask parameter is kept for API compatibility but currently ignored
    # until visualize_grid_rich supports forcing mask visibility.
    console.print(visualize_grid_rich(grid, title=title))


def render_ansi(grid_input: GridArray | np.ndarray | Grid) -> str:
    """Render grid as ANSI string.

    Args:
        grid_input: Grid data (JAX array, numpy array, or Grid object)

    Returns:
        A string containing the ANSI representation of the grid.
    """
    # Create a temporary console for capturing output
    temp_console = Console(force_terminal=True, color_system="truecolor", width=1000)
    table = visualize_grid_rich(grid_input)
    with temp_console.capture() as capture:
        temp_console.print(table)
    return capture.get()


def display_step(
    step_data: dict[str, Any],
    step_idx: int,
    console: Console | None = None,
    show_coordinates: bool = False,
    show_numbers: bool = False,
    double_width: bool = True,
) -> None:
    """Display a single step of the episode using Rich.

    Args:
        step_data: Step data dictionary
        step_idx: Index of the step
        console: Optional Rich console (creates one if None)
        show_coordinates: Whether to show grid coordinates
        show_numbers: If True, show colored numbers; if False, show colored blocks
        double_width: If True and show_numbers=False, use double-width blocks for square appearance
    """
    if console is None:
        console = Console()

    # Extract relevant data
    input_grid = step_data["input_grid"]
    output_grid = step_data["output_grid"]
    input_mask = step_data["input_mask"]
    output_mask = step_data["output_mask"]

    # Create title
    title = f"Step {step_idx + 1}"

    # Display input-output pair
    visualize_task_pair_rich(
        input_grid,
        output_grid,
        input_mask,
        output_mask,
        title=title,
        show_numbers=show_numbers,
        double_width=double_width,
        console=console,
    )
