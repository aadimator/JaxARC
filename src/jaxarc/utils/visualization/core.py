"""Grid visualization utilities for JaxARC.

This module provides functionality to visualize ARC grids and tasks in different formats:
- Rich-based terminal visualization for logging and debugging
- SVG-based image generation for documentation and analysis

The module works with the core JaxARC data structures including Grid, TaskPair, and JaxArcTask.
"""

from __future__ import annotations

import io
import time
from typing import TYPE_CHECKING, Any, cast

import drawsvg  # type: ignore[import-untyped]
import jax.numpy as jnp
import numpy as np
from loguru import logger
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Optional matplotlib imports for enhanced visualizations
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import patches
    from matplotlib.gridspec import GridSpec

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from jaxarc.types import Grid
from jaxarc.utils.jax_types import (
    GridArray,
)
from jaxarc.utils.task_manager import extract_task_id_from_index

if TYPE_CHECKING:
    from jaxarc.types import JaxArcTask

# ARC color palette - matches the provided color map
ARC_COLOR_PALETTE: dict[int, str] = {
    0: "#252525",  # 0: black
    1: "#0074D9",  # 1: blue
    2: "#FF4136",  # 2: red
    3: "#37D449",  # 3: green
    4: "#FFDC00",  # 4: yellow
    5: "#E6E6E6",  # 5: grey
    6: "#F012BE",  # 6: pink
    7: "#FF871E",  # 7: orange
    8: "#54D2EB",  # 8: light blue
    9: "#8D1D2C",  # 9: brown
    10: "#FFFFFF",  # 10: white (for padding/invalid)
}


def _extract_grid_data(
    grid_input: GridArray | np.ndarray | Grid,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract numpy array and mask from various grid input types.

    Args:
        grid_input: Grid data in various formats

    Returns:
        Tuple of (grid_data as numpy array, mask as numpy array or None)

    Raises:
        ValueError: If input type is not supported
    """
    if isinstance(grid_input, Grid):
        return np.asarray(grid_input.data), np.asarray(grid_input.mask)
    if isinstance(grid_input, (jnp.ndarray, np.ndarray)):
        return np.asarray(grid_input), None

    msg = f"Unsupported grid input type: {type(grid_input)}"
    raise ValueError(msg)


def _extract_valid_region(
    grid: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Extract the valid (non-padded) region from a grid.

    Args:
        grid: The grid array
        mask: Optional boolean mask indicating valid cells

    Returns:
        Tuple of (valid_grid, (start_row, start_col), (height, width))
    """
    if mask is None:
        # Assume all cells are valid if no mask provided
        return grid, (0, 0), (grid.shape[0], grid.shape[1])

    if not np.any(mask):
        # No valid cells
        return np.array([[]], dtype=grid.dtype), (0, 0), (0, 0)

    # Find bounding box of valid region
    valid_rows = np.where(np.any(mask, axis=1))[0]
    valid_cols = np.where(np.any(mask, axis=0))[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        return np.array([[]], dtype=grid.dtype), (0, 0), (0, 0)

    start_row, end_row = valid_rows[0], valid_rows[-1] + 1
    start_col, end_col = valid_cols[0], valid_cols[-1] + 1

    valid_grid = grid[start_row:end_row, start_col:end_col]

    return (
        valid_grid,
        (start_row, start_col),
        (end_row - start_row, end_col - start_col),
    )


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
        mask = np.asarray(mask)

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


def draw_grid_svg(
    grid_input: jnp.ndarray | np.ndarray | Grid,
    mask: jnp.ndarray | np.ndarray | None = None,
    max_width: float = 10.0,
    max_height: float = 10.0,
    padding: float = 0.5,
    extra_bottom_padding: float = 0.5,
    label: str = "",
    border_color: str = "#111111ff",
    show_size: bool = True,
    as_group: bool = False,
) -> drawsvg.Drawing | tuple[drawsvg.Group, tuple[float, float], tuple[float, float]]:
    """Draw a single grid as an SVG.

    Args:
        grid_input: Grid data (JAX array, numpy array, or Grid object)
        mask: Optional boolean mask indicating valid cells
        max_width: Maximum width for the drawing
        max_height: Maximum height for the drawing
        padding: Padding around the grid
        extra_bottom_padding: Extra padding at bottom for labels
        label: Label to display below the grid
        border_color: Color for the grid border
        show_size: Whether to show grid dimensions
        as_group: If True, return as a group for inclusion in larger drawings

    Returns:
        Either a Drawing object or tuple of (Group, origin, size) if as_group=True
    """
    grid, grid_mask = _extract_grid_data(grid_input)

    if mask is None:
        mask = grid_mask

    if mask is not None:
        mask = np.asarray(mask)

    # Handle empty grids
    if grid.size == 0:
        if as_group:
            return (
                drawsvg.Group(),
                (-0.5 * padding, -0.5 * padding),
                (padding, padding + extra_bottom_padding),
            )

        drawing = drawsvg.Drawing(
            padding,
            padding + extra_bottom_padding,
            origin=(-0.5 * padding, -0.5 * padding),
        )
        drawing.set_pixel_scale(40)
        return drawing

    # Extract valid region
    valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
        grid, mask
    )

    if height == 0 or width == 0:
        if as_group:
            return (
                drawsvg.Group(),
                (-0.5 * padding, -0.5 * padding),
                (padding, padding + extra_bottom_padding),
            )

        drawing = drawsvg.Drawing(
            padding,
            padding + extra_bottom_padding,
            origin=(-0.5 * padding, -0.5 * padding),
        )
        drawing.set_pixel_scale(40)
        return drawing

    # Calculate cell size
    cell_size_x = max_width / width if width > 0 else max_height
    cell_size_y = max_height / height if height > 0 else max_width
    cell_size = min(cell_size_x, cell_size_y) if width > 0 and height > 0 else 0

    actual_width = width * cell_size
    actual_height = height * cell_size

    # Drawing setup
    line_thickness = 0.01
    border_width = 0.08
    lt = line_thickness / 2

    if as_group:
        drawing = drawsvg.Group()
    else:
        drawing = drawsvg.Drawing(
            actual_width + padding,
            actual_height + padding + extra_bottom_padding,
            origin=(-0.5 * padding, -0.5 * padding),
        )
        drawing.set_pixel_scale(40)

    # Draw grid cells
    for i in range(height):
        for j in range(width):
            color_val = int(valid_grid[i, j])

            # Check if cell is valid
            is_valid = True
            if mask is not None:
                actual_row = start_row + i
                actual_col = start_col + j
                if actual_row < mask.shape[0] and actual_col < mask.shape[1]:
                    is_valid = mask[actual_row, actual_col]

            if is_valid and 0 <= color_val < len(ARC_COLOR_PALETTE.keys()):
                fill_color = ARC_COLOR_PALETTE.get(color_val, "white")
            else:
                fill_color = "#CCCCCC"  # Light gray for invalid/unknown colors

            drawing.append(
                drawsvg.Rectangle(
                    j * cell_size + lt,
                    i * cell_size + lt,
                    cell_size - lt,
                    cell_size - lt,
                    fill=fill_color,
                )
            )

    # Add border
    border_margin = border_width / 3
    drawing.append(
        drawsvg.Rectangle(
            -border_margin,
            -border_margin,
            actual_width + border_margin * 2,
            actual_height + border_margin * 2,
            fill="none",
            stroke=border_color,
            stroke_width=border_width,
        )
    )

    if not as_group:
        # Embed font
        cast(drawsvg.Drawing, drawing).embed_google_font(
            "Anuphan:wght@400;600;700",
            text=set(
                "Input Output 0123456789x Test Task ABCDEFGHIJ? abcdefghjklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ),
        )

    # Add size and label text
    font_size = (padding / 2 + extra_bottom_padding) / 2

    if show_size:
        drawing.append(
            drawsvg.Text(
                text=f"{width}x{height}",
                x=actual_width,
                y=actual_height + font_size * 1.25,
                font_size=font_size,
                fill="black",
                text_anchor="end",
                font_family="Anuphan",
            )
        )

    if label:
        drawing.append(
            drawsvg.Text(
                text=label,
                x=-0.1 * font_size,
                y=actual_height + font_size * 1.25,
                font_size=font_size,
                fill="black",
                text_anchor="start",
                font_family="Anuphan",
                font_weight="600",
            )
        )

    if as_group:
        return (
            cast(drawsvg.Group, drawing),
            (-0.5 * padding, -0.5 * padding),
            (actual_width + padding, actual_height + padding + extra_bottom_padding),
        )

    return cast(drawsvg.Drawing, drawing)


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


def _draw_dotted_squircle(
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    stroke_color: str = "#666666",
    stroke_width: float = 0.05,
    corner_radius: float = 0.3,
    dash_array: str = "0.1,0.1",
) -> list[drawsvg.DrawingElement]:
    """Draw a dotted squircle (rounded rectangle) with label.

    Args:
        x: Left edge of the squircle
        y: Top edge of the squircle
        width: Width of the squircle
        height: Height of the squircle
        label: Label text to display
        stroke_color: Color of the dotted border
        stroke_width: Width of the border
        corner_radius: Radius for rounded corners
        dash_array: SVG dash pattern for dotted line

    Returns:
        List of drawing elements (squircle and label)
    """
    elements = []

    # Draw dotted squircle
    squircle = drawsvg.Rectangle(
        x,
        y,
        width,
        height,
        rx=corner_radius,
        ry=corner_radius,
        fill="none",
        stroke=stroke_color,
        stroke_width=stroke_width,
        stroke_dasharray=dash_array,
        opacity=0.7,
    )
    elements.append(squircle)

    # Add label
    label_x = x + width - 0.1
    label_y = y + 0.3
    label_text = drawsvg.Text(
        text=label,
        x=label_x,
        y=label_y,
        font_size=0.25,
        font_family="Anuphan",
        font_weight="700",
        fill=stroke_color,
        text_anchor="end",
        opacity=0.8,
    )
    elements.append(label_text)

    return elements


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


def save_svg_drawing(
    drawing: drawsvg.Drawing,
    filename: str,
    context: Any | None = None,
) -> None:
    """Save an SVG drawing to file with support for multiple formats.

    Args:
        drawing: The SVG drawing to save
        filename: Output filename (extension determines format: .svg, .png, .pdf)
        context: Optional context for PDF conversion

    Raises:
        ValueError: If file extension is not supported
        ImportError: If required dependencies are missing for PNG/PDF output
    """
    if filename.endswith(".svg"):
        drawing.save_svg(filename)
        logger.info(f"Saved SVG to {filename}")
    elif filename.endswith(".png"):
        drawing.save_png(filename)
        logger.info(f"Saved PNG to {filename}")
    elif filename.endswith(".pdf"):
        buffer = io.StringIO()
        drawing.as_svg(output_file=buffer, context=context)

        try:
            import cairosvg  # type: ignore[import-untyped,import-not-found]

            cairosvg.svg2pdf(bytestring=buffer.getvalue(), write_to=filename)
            logger.info(f"Saved PDF to {filename}")
        except ImportError as e:
            error_msg = "cairosvg is required for PDF output. Please install it with: pip install cairosvg"
            logger.error(error_msg)
            raise ImportError(error_msg) from e
    else:
        error_msg = (
            f"Unknown file extension for {filename}. Supported: .svg, .png, .pdf"
        )
        raise ValueError(error_msg)


def draw_rl_step_svg(
    before_grid: Grid,
    after_grid: Grid,
    selection_mask: jnp.ndarray | np.ndarray,
    operation_id: int,
    step_number: int,
    max_width: float = 1200.0,
    max_height: float = 600.0,
    label: str = "",
    show_operation_name: bool = True,
) -> str:
    """Generate SVG visualization of a single RL step.

    Layout: [Before Grid with Selection Overlay] -> [Operation] -> [After Grid]

    Args:
        before_grid: Grid state before the action
        after_grid: Grid state after the action
        selection_mask: Boolean mask showing selected cells
        operation_id: Integer ID of the operation performed
        step_number: Step number in the episode
        max_width: Maximum width of the entire visualization
        max_height: Maximum height of the entire visualization
        label: Optional label for the visualization
        show_operation_name: Whether to show operation name (vs just ID)

    Returns:
        SVG string containing the visualization
    """
    import drawsvg as draw

    # Layout parameters
    top_padding = 80
    bottom_padding = 100
    side_padding = 40
    grid_spacing = 150
    grid_max_width = 250
    grid_max_height = 250

    # Calculate total dimensions (only 2 grids now)
    total_width = 2 * grid_max_width + grid_spacing + 2 * side_padding
    total_height = grid_max_height + top_padding + bottom_padding

    # Create main drawing
    drawing = draw.Drawing(total_width, total_height)

    # Add title
    title_text = f"{label} - Step {step_number}" if label else f"Step {step_number}"
    drawing.append(
        draw.Text(
            title_text,
            font_size=32,
            x=total_width / 2,
            y=50,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="600",
        )
    )

    # Grid positions (only 2 grids now)
    before_x = side_padding
    after_x = side_padding + grid_max_width + grid_spacing
    grids_y = top_padding

    # Helper function to draw a single grid directly
    def draw_grid_direct(
        grid: Grid, x: float, y: float, grid_label: str
    ) -> tuple[float, float]:
        """Draw a grid directly into the main SVG and return actual dimensions."""
        grid_data, grid_mask = _extract_grid_data(grid)

        if grid_mask is not None:
            grid_mask = np.asarray(grid_mask)

        # Extract valid region
        valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
            grid_data, grid_mask
        )

        if height == 0 or width == 0:
            return 0, 0

        # Calculate cell size to fit within max dimensions
        cell_size = min(grid_max_width / width, grid_max_height / height)
        actual_width = width * cell_size
        actual_height = height * cell_size

        # Center the grid within the allocated space
        grid_x = x + (grid_max_width - actual_width) / 2
        grid_y = y + (grid_max_height - actual_height) / 2

        # Draw grid cells
        for i in range(height):
            for j in range(width):
                color_val = int(valid_grid[i, j])

                # Check if cell is valid
                is_valid = True
                if grid_mask is not None:
                    actual_row = start_row + i
                    actual_col = start_col + j
                    if (
                        actual_row < grid_mask.shape[0]
                        and actual_col < grid_mask.shape[1]
                    ):
                        is_valid = grid_mask[actual_row, actual_col]

                if is_valid and 0 <= color_val < len(ARC_COLOR_PALETTE.keys()):
                    fill_color = ARC_COLOR_PALETTE.get(color_val, "white")
                else:
                    fill_color = "#CCCCCC"

                drawing.append(
                    draw.Rectangle(
                        grid_x + j * cell_size,
                        grid_y + i * cell_size,
                        cell_size,
                        cell_size,
                        fill=fill_color,
                        stroke="#111111",
                        stroke_width=0.5,
                    )
                )

        # Add grid border
        drawing.append(
            draw.Rectangle(
                grid_x - 2,
                grid_y - 2,
                actual_width + 4,
                actual_height + 4,
                fill="none",
                stroke="#111111",
                stroke_width=2,
            )
        )

        # Add grid label
        drawing.append(
            draw.Text(
                grid_label,
                font_size=18,
                x=grid_x,
                y=grid_y + actual_height + 25,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="600",
            )
        )

        return actual_width, actual_height

    # Draw before grid with selection overlay
    before_width, before_height = draw_grid_direct(
        before_grid, before_x, grids_y, "Before (with Selection)"
    )

    # Add selection mask overlay directly on before grid
    selection_mask_np = np.asarray(selection_mask)
    if selection_mask_np.any():
        # Get valid region info for coordinate mapping
        before_grid_np = np.asarray(before_grid.data)
        before_mask_np = (
            np.asarray(before_grid.mask) if before_grid.mask is not None else None
        )

        _, (start_row, start_col), (display_height, display_width) = (
            _extract_valid_region(before_grid_np, before_mask_np)
        )

        if display_width > 0 and display_height > 0:
            # Calculate cell size and position for overlay on before grid
            cell_size = min(
                grid_max_width / display_width, grid_max_height / display_height
            )
            overlay_x = before_x + (grid_max_width - before_width) / 2
            overlay_y = grids_y + (grid_max_height - before_height) / 2

            # Draw selection overlay on before grid
            # Use magenta (#ff00ff) as it's not in the ARC color palette
            selection_color = "#ff00ff"

            # First pass: draw filled rectangles without borders
            for display_row in range(display_height):
                for display_col in range(display_width):
                    orig_row = start_row + display_row
                    orig_col = start_col + display_col

                    if (
                        orig_row < selection_mask_np.shape[0]
                        and orig_col < selection_mask_np.shape[1]
                        and selection_mask_np[orig_row, orig_col]
                    ):
                        drawing.append(
                            draw.Rectangle(
                                overlay_x + display_col * cell_size,
                                overlay_y + display_row * cell_size,
                                cell_size,
                                cell_size,
                                fill=selection_color,
                                fill_opacity=0.3,
                                stroke="none",
                            )
                        )

            # Second pass: draw boundary lines only on outer edges
            def is_selected(row, col):
                """Check if a cell is selected, handling bounds."""
                if (
                    row < 0
                    or row >= selection_mask_np.shape[0]
                    or col < 0
                    or col >= selection_mask_np.shape[1]
                ):
                    return False
                return selection_mask_np[row, col]

            for display_row in range(display_height):
                for display_col in range(display_width):
                    orig_row = start_row + display_row
                    orig_col = start_col + display_col

                    if (
                        orig_row < selection_mask_np.shape[0]
                        and orig_col < selection_mask_np.shape[1]
                        and selection_mask_np[orig_row, orig_col]
                    ):
                        cell_x = overlay_x + display_col * cell_size
                        cell_y = overlay_y + display_row * cell_size

                        # Check each edge and draw border line if it's on the boundary
                        # Top edge
                        if not is_selected(orig_row - 1, orig_col):
                            drawing.append(
                                draw.Line(
                                    cell_x,
                                    cell_y,
                                    cell_x + cell_size,
                                    cell_y,
                                    stroke=selection_color,
                                    stroke_width=3,
                                    stroke_opacity=0.9,
                                )
                            )

                        # Bottom edge
                        if not is_selected(orig_row + 1, orig_col):
                            drawing.append(
                                draw.Line(
                                    cell_x,
                                    cell_y + cell_size,
                                    cell_x + cell_size,
                                    cell_y + cell_size,
                                    stroke=selection_color,
                                    stroke_width=3,
                                    stroke_opacity=0.9,
                                )
                            )

                        # Left edge
                        if not is_selected(orig_row, orig_col - 1):
                            drawing.append(
                                draw.Line(
                                    cell_x,
                                    cell_y,
                                    cell_x,
                                    cell_y + cell_size,
                                    stroke=selection_color,
                                    stroke_width=3,
                                    stroke_opacity=0.9,
                                )
                            )

                        # Right edge
                        if not is_selected(orig_row, orig_col + 1):
                            drawing.append(
                                draw.Line(
                                    cell_x + cell_size,
                                    cell_y,
                                    cell_x + cell_size,
                                    cell_y + cell_size,
                                    stroke=selection_color,
                                    stroke_width=3,
                                    stroke_opacity=0.9,
                                )
                            )

    # Draw after grid
    after_width, after_height = draw_grid_direct(after_grid, after_x, grids_y, "After")

    # Add operation information
    operation_text = f"Operation: {operation_id}"
    if show_operation_name:
        try:
            from jaxarc.envs.operations import get_operation_display_text

            operation_text = get_operation_display_text(operation_id)
        except (ValueError, ImportError):
            operation_text = f"Operation: {operation_id}"

    drawing.append(
        draw.Text(
            operation_text,
            font_size=24,
            x=total_width / 2,
            y=grids_y + grid_max_height + 50,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="400",
        )
    )

    # Add arrow between grids
    arrow_y = grids_y + grid_max_height / 2

    # Arrow from before to after
    arrow_start_x = before_x + grid_max_width + 20
    arrow_end_x = after_x - 20
    drawing.append(
        draw.Line(
            arrow_start_x,
            arrow_y,
            arrow_end_x,
            arrow_y,
            stroke="#666666",
            stroke_width=4,
        )
    )
    drawing.append(
        draw.Lines(
            arrow_end_x - 12,
            arrow_y - 8,
            arrow_end_x - 12,
            arrow_y + 8,
            arrow_end_x,
            arrow_y,
            close=True,
            fill="#666666",
        )
    )

    return drawing.as_svg()


def save_rl_step_visualization(
    state: ArcEnvState,
    action: dict,
    next_state: ArcEnvState,
    output_dir: str = "output/rl_steps",
) -> None:
    """JAX callback function to save RL step visualization.

    This function is designed to be used with jax.debug.callback.

    Args:
        state: Environment state before the action
        action: Action dictionary with 'selection' and 'operation' keys
        next_state: Environment state after the action
        output_dir: Directory to save visualization files
    """
    from pathlib import Path

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create Grid objects (convert JAX arrays to numpy)
    before_grid = Grid(
        data=np.asarray(state.working_grid),
        mask=np.asarray(state.working_grid_mask),
    )
    after_grid = Grid(
        data=np.asarray(next_state.working_grid),
        mask=np.asarray(next_state.working_grid_mask),
    )

    # Extract action components
    operation_id = int(action["operation"])
    step_number = int(state.step_count)

    # Create dummy reward and info for visualization
    reward = 0.0  # Placeholder since we don't have reward in this context
    info = {"step_count": step_number}  # Basic info

    # Generate visualization
    svg_content = draw_rl_step_svg(
        before_grid=before_grid,
        after_grid=after_grid,
        action=action,
        reward=reward,
        info=info,
        step_num=step_number,
    )

    # Save to file with zero-padded step number
    filename = f"step_{step_number:03d}.svg"
    filepath = Path(output_dir) / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg_content)

    # Log the save (will appear in console during execution)
    from loguru import logger

    logger.info(f"Saved RL step visualization: {filepath}")


def _clear_output_directory(output_dir: str) -> None:
    """Clear output directory for new episode."""
    import shutil
    from pathlib import Path

    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def setup_matplotlib_style() -> None:
    """Set up matplotlib and seaborn styling for high-quality visualizations.

    Raises:
        ImportError: If matplotlib or seaborn is not available
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib and seaborn are required for this function. Install with: pip install matplotlib seaborn"
        )

    # Configure matplotlib for high-quality output
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

    # Set up seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")


def draw_rl_step_svg_enhanced(
    before_grid: Grid,
    after_grid: Grid,
    action: Dict[str, Any],
    reward: float,
    info: Dict[str, Any],
    step_num: int,
    operation_name: str = "",
    changed_cells: Optional[jnp.ndarray] = None,
    config: Optional[Any] = None,
    max_width: float = 1400.0,
    max_height: float = 700.0,
) -> str:
    """Generate enhanced SVG visualization of a single RL step with more information.

    This enhanced version shows:
    - Before and after grids with improved styling
    - Action selection highlighting
    - Changed cell highlighting
    - Reward information and metrics
    - Operation name and details
    - Step metadata

    Args:
        before_grid: Grid state before the action
        after_grid: Grid state after the action
        action: Action dictionary containing selection and operation info
        reward: Reward received for this step
        info: Additional information dictionary
        step_num: Step number in the episode
        operation_name: Human-readable operation name
        changed_cells: Optional mask of cells that changed
        config: Optional visualization configuration
        max_width: Maximum width of the entire visualization
        max_height: Maximum height of the entire visualization

    Returns:
        SVG string containing the enhanced visualization
    """
    import drawsvg as draw

    # Get color palette from config or use default
    if config and hasattr(config, "get_color_palette"):
        color_palette = config.get_color_palette()
    else:
        color_palette = ARC_COLOR_PALETTE

    # Layout parameters
    top_padding = 100
    bottom_padding = 120
    side_padding = 50
    grid_spacing = 180
    grid_max_width = 280
    grid_max_height = 280
    info_panel_height = 80

    # Calculate total dimensions
    total_width = 2 * grid_max_width + grid_spacing + 2 * side_padding
    total_height = grid_max_height + top_padding + bottom_padding + info_panel_height

    # Create main drawing with background
    drawing = draw.Drawing(total_width, total_height)
    drawing.append(draw.Rectangle(0, 0, total_width, total_height, fill="#f8f9fa"))

    # Add enhanced title with step info
    title_text = f"Step {step_num}"
    if operation_name:
        title_text += f" - {operation_name}"

    drawing.append(
        draw.Text(
            title_text,
            font_size=28,
            x=total_width / 2,
            y=40,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Add reward information
    reward_color = "#27ae60" if reward > 0 else "#e74c3c" if reward < 0 else "#95a5a6"
    reward_text = f"Reward: {reward:.3f}"
    drawing.append(
        draw.Text(
            reward_text,
            font_size=20,
            x=total_width / 2,
            y=70,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="500",
            fill=reward_color,
        )
    )

    # Grid positions
    before_x = side_padding
    after_x = side_padding + grid_max_width + grid_spacing
    grids_y = top_padding

    # Helper function to draw enhanced grid
    def draw_enhanced_grid(
        grid: Grid,
        x: float,
        y: float,
        grid_label: str,
        selection_mask: Optional[np.ndarray] = None,
        highlight_changes: bool = False,
        changed_cells: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """Draw an enhanced grid with overlays and styling."""
        grid_data, grid_mask = _extract_grid_data(grid)

        if grid_mask is not None:
            grid_mask = np.asarray(grid_mask)

        # Extract valid region
        valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
            grid_data, grid_mask
        )

        if height == 0 or width == 0:
            return 0, 0

        # Calculate cell size to fit within max dimensions
        cell_size = min(grid_max_width / width, grid_max_height / height)
        actual_width = width * cell_size
        actual_height = height * cell_size

        # Center the grid within the allocated space
        grid_x = x + (grid_max_width - actual_width) / 2
        grid_y = y + (grid_max_height - actual_height) / 2

        # Draw grid background
        drawing.append(
            draw.Rectangle(
                grid_x - 5,
                grid_y - 5,
                actual_width + 10,
                actual_height + 10,
                fill="white",
                stroke="#dee2e6",
                stroke_width=1,
                rx=5,
            )
        )

        # Draw grid cells
        for i in range(height):
            for j in range(width):
                color_val = int(valid_grid[i, j])

                # Check if cell is valid
                is_valid = True
                if grid_mask is not None:
                    actual_row = start_row + i
                    actual_col = start_col + j
                    if (
                        actual_row < grid_mask.shape[0]
                        and actual_col < grid_mask.shape[1]
                    ):
                        is_valid = grid_mask[actual_row, actual_col]

                if is_valid and 0 <= color_val < len(color_palette.keys()):
                    fill_color = color_palette.get(color_val, "white")
                else:
                    fill_color = "#CCCCCC"

                cell_x = grid_x + j * cell_size
                cell_y = grid_y + i * cell_size

                # Draw cell
                drawing.append(
                    draw.Rectangle(
                        cell_x,
                        cell_y,
                        cell_size,
                        cell_size,
                        fill=fill_color,
                        stroke="#6c757d",
                        stroke_width=0.5,
                    )
                )

        # Add changed cell highlighting after all cells are drawn
        if highlight_changes and changed_cells is not None:
            changed_mask = np.asarray(changed_cells)
            change_color = "#FF0080"  # Bright magenta for changes

            # Draw change highlighting with proper boundaries
            for i in range(height):
                for j in range(width):
                    actual_row = start_row + i
                    actual_col = start_col + j
                    if (
                        actual_row < changed_mask.shape[0]
                        and actual_col < changed_mask.shape[1]
                        and changed_mask[actual_row, actual_col]
                    ):
                        cell_x = grid_x + j * cell_size
                        cell_y = grid_y + i * cell_size

                        # Add bright border for changed cells
                        drawing.append(
                            draw.Rectangle(
                                cell_x - 1,
                                cell_y - 1,
                                cell_size + 2,
                                cell_size + 2,
                                fill="none",
                                stroke=change_color,
                                stroke_width=3,
                                stroke_opacity=0.8,
                            )
                        )

                        # Add inner glow effect
                        drawing.append(
                            draw.Rectangle(
                                cell_x + 2,
                                cell_y + 2,
                                cell_size - 4,
                                cell_size - 4,
                                fill=change_color,
                                fill_opacity=0.15,
                                stroke="none",
                            )
                        )

        # Add selection overlay if provided
        if selection_mask is not None and selection_mask.any():
            # Use bright neon cyan for better visibility
            selection_color = "#00FFFF"  # Bright cyan - very visible

            # First pass: Add fill overlay
            for display_row in range(height):
                for display_col in range(width):
                    orig_row = start_row + display_row
                    orig_col = start_col + display_col

                    if (
                        orig_row < selection_mask.shape[0]
                        and orig_col < selection_mask.shape[1]
                        and selection_mask[orig_row, orig_col]
                    ):
                        cell_x = grid_x + display_col * cell_size
                        cell_y = grid_y + display_row * cell_size

                        # Add bright selection fill
                        drawing.append(
                            draw.Rectangle(
                                cell_x,
                                cell_y,
                                cell_size,
                                cell_size,
                                fill=selection_color,
                                fill_opacity=0.4,  # Slightly more opaque
                                stroke="none",
                            )
                        )

            # Second pass: Add boundary lines for better definition
            def is_selected_cell(row, col):
                """Check if a cell is selected, handling bounds."""
                if (
                    row < 0
                    or row >= selection_mask.shape[0]
                    or col < 0
                    or col >= selection_mask.shape[1]
                ):
                    return False
                return selection_mask[row, col]

            for display_row in range(height):
                for display_col in range(width):
                    orig_row = start_row + display_row
                    orig_col = start_col + display_col

                    if (
                        orig_row < selection_mask.shape[0]
                        and orig_col < selection_mask.shape[1]
                        and selection_mask[orig_row, orig_col]
                    ):
                        cell_x = grid_x + display_col * cell_size
                        cell_y = grid_y + display_row * cell_size

                        # Draw boundary lines only on outer edges for clean look
                        stroke_width = 3

                        # Top edge
                        if not is_selected_cell(orig_row - 1, orig_col):
                            drawing.append(
                                draw.Line(
                                    cell_x,
                                    cell_y,
                                    cell_x + cell_size,
                                    cell_y,
                                    stroke=selection_color,
                                    stroke_width=stroke_width,
                                    stroke_opacity=0.9,
                                )
                            )

                        # Bottom edge
                        if not is_selected_cell(orig_row + 1, orig_col):
                            drawing.append(
                                draw.Line(
                                    cell_x,
                                    cell_y + cell_size,
                                    cell_x + cell_size,
                                    cell_y + cell_size,
                                    stroke=selection_color,
                                    stroke_width=stroke_width,
                                    stroke_opacity=0.9,
                                )
                            )

                        # Left edge
                        if not is_selected_cell(orig_row, orig_col - 1):
                            drawing.append(
                                draw.Line(
                                    cell_x,
                                    cell_y,
                                    cell_x,
                                    cell_y + cell_size,
                                    stroke=selection_color,
                                    stroke_width=stroke_width,
                                    stroke_opacity=0.9,
                                )
                            )

                        # Right edge
                        if not is_selected_cell(orig_row, orig_col + 1):
                            drawing.append(
                                draw.Line(
                                    cell_x + cell_size,
                                    cell_y,
                                    cell_x + cell_size,
                                    cell_y + cell_size,
                                    stroke=selection_color,
                                    stroke_width=stroke_width,
                                    stroke_opacity=0.9,
                                )
                            )

        # Add enhanced grid border
        drawing.append(
            draw.Rectangle(
                grid_x - 3,
                grid_y - 3,
                actual_width + 6,
                actual_height + 6,
                fill="none",
                stroke="#495057",
                stroke_width=2,
                rx=3,
            )
        )

        # Add enhanced grid label with background
        label_bg_width = len(grid_label) * 12 + 20
        drawing.append(
            draw.Rectangle(
                grid_x - 5,
                grid_y + actual_height + 15,
                label_bg_width,
                25,
                fill="#e9ecef",
                stroke="#dee2e6",
                stroke_width=1,
                rx=3,
            )
        )

        drawing.append(
            draw.Text(
                grid_label,
                font_size=16,
                x=grid_x + 5,
                y=grid_y + actual_height + 32,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="600",
                fill="#495057",
            )
        )

        return actual_width, actual_height

    # Extract selection mask from action
    selection_mask = None
    if "selection" in action:
        selection_mask = np.asarray(action["selection"])
    elif "bbox" in action:
        # Convert bbox to selection mask for visualization
        bbox = np.asarray(action["bbox"])
        if len(bbox) >= 4:
            # Get grid dimensions from before_grid
            grid_data, grid_mask = _extract_grid_data(before_grid)
            if grid_mask is not None:
                grid_mask = np.asarray(grid_mask)
            
            # Extract valid region to get actual dimensions
            valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
                grid_data, grid_mask
            )
            
            if height > 0 and width > 0:
                # Extract and clip coordinates
                r1 = int(np.clip(bbox[0], 0, height - 1))
                c1 = int(np.clip(bbox[1], 0, width - 1))
                r2 = int(np.clip(bbox[2], 0, height - 1))
                c2 = int(np.clip(bbox[3], 0, width - 1))
                
                # Ensure proper ordering
                min_r, max_r = min(r1, r2), max(r1, r2)
                min_c, max_c = min(c1, c2), max(c1, c2)
                
                # Create selection mask for the valid region
                selection_mask = np.zeros((height, width), dtype=bool)
                selection_mask[min_r:max_r+1, min_c:max_c+1] = True

    # Draw before grid with selection overlay
    before_width, before_height = draw_enhanced_grid(
        before_grid, before_x, grids_y, "Before State", selection_mask=selection_mask
    )

    # Draw after grid with change highlighting
    after_width, after_height = draw_enhanced_grid(
        after_grid,
        after_x,
        grids_y,
        "After State",
        highlight_changes=True,
        changed_cells=changed_cells,
    )

    # Add enhanced arrow between grids
    arrow_y = grids_y + grid_max_height / 2
    arrow_start_x = before_x + grid_max_width + 30
    arrow_end_x = after_x - 30

    # Arrow shaft
    drawing.append(
        draw.Line(
            arrow_start_x,
            arrow_y,
            arrow_end_x,
            arrow_y,
            stroke="#6c757d",
            stroke_width=3,
        )
    )

    # Arrow head
    drawing.append(
        draw.Lines(
            arrow_end_x - 15,
            arrow_y - 10,
            arrow_end_x - 15,
            arrow_y + 10,
            arrow_end_x,
            arrow_y,
            close=True,
            fill="#6c757d",
        )
    )

    # Arrow label removed - operation name is already in the title

    # Add information panel at bottom
    info_y = grids_y + grid_max_height + 60

    # Info panel background
    drawing.append(
        draw.Rectangle(
            side_padding,
            info_y,
            total_width - 2 * side_padding,
            info_panel_height,
            fill="#ffffff",
            stroke="#dee2e6",
            stroke_width=1,
            rx=5,
        )
    )

    # Add metadata information
    info_items = []

    # Add step metadata
    if "similarity" in info:
        similarity_val = float(info['similarity']) if hasattr(info['similarity'], 'item') else info['similarity']
        info_items.append(f"Similarity: {similarity_val:.3f}")

    if "episode_reward" in info:
        reward_val = float(info['episode_reward']) if hasattr(info['episode_reward'], 'item') else info['episode_reward']
        info_items.append(f"Episode Reward: {reward_val:.3f}")

    if "step_count" in info:
        step_val = int(info['step_count']) if hasattr(info['step_count'], 'item') else info['step_count']
        info_items.append(f"Total Steps: {step_val}")

    # Add action details
    if "operation" in action:
        op_val = int(action['operation']) if hasattr(action['operation'], 'item') else action['operation']
        info_items.append(f"Operation ID: {op_val}")

    # Display info items
    info_text = " | ".join(info_items) if info_items else "No additional information"
    drawing.append(
        draw.Text(
            info_text,
            font_size=14,
            x=total_width / 2,
            y=info_y + 25,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="400",
            fill="#6c757d",
        )
    )

    # Add timestamp
    import time

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    drawing.append(
        draw.Text(
            f"Generated: {timestamp}",
            font_size=12,
            x=total_width - side_padding,
            y=info_y + 50,
            text_anchor="end",
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
    width: float = 1200.0,
    height: float = 800.0,
) -> str:
    """Generate SVG visualization of episode summary with key metrics.

    Args:
        summary_data: Episode summary data
        step_data: List of step visualization data
        config: Optional visualization configuration
        width: Width of the visualization
        height: Height of the visualization

    Returns:
        SVG string containing the episode summary
    """
    import drawsvg as draw

    # Create main drawing
    drawing = draw.Drawing(width, height)
    drawing.append(draw.Rectangle(0, 0, width, height, fill="#f8f9fa"))

    # Layout parameters
    padding = 40
    title_height = 80
    chart_height = 200
    grid_section_height = height - title_height - chart_height - 3 * padding

    # Add title
    title_text = f"Episode {summary_data.episode_num} Summary"
    drawing.append(
        draw.Text(
            title_text,
            font_size=32,
            x=width / 2,
            y=50,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Add episode metrics
    metrics_y = title_height + 20
    metrics = [
        f"Total Steps: {summary_data.total_steps}",
        f"Total Reward: {summary_data.total_reward:.3f}",
        f"Final Similarity: {summary_data.final_similarity:.3f}",
        f"Success: {'Yes' if summary_data.success else 'No'}",
    ]

    metrics_text = " | ".join(metrics)
    drawing.append(
        draw.Text(
            metrics_text,
            font_size=18,
            x=width / 2,
            y=metrics_y,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="500",
            fill="#495057",
        )
    )

    # Draw reward progression chart
    chart_y = metrics_y + 40
    chart_width = width - 2 * padding

    if summary_data.reward_progression:
        # Chart background
        drawing.append(
            draw.Rectangle(
                padding,
                chart_y,
                chart_width,
                chart_height,
                fill="white",
                stroke="#dee2e6",
                stroke_width=1,
                rx=5,
            )
        )

        # Chart title
        drawing.append(
            draw.Text(
                "Reward Progression",
                font_size=16,
                x=padding + 10,
                y=chart_y + 20,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="600",
                fill="#495057",
            )
        )

        # Draw reward line
        rewards = summary_data.reward_progression
        if len(rewards) > 1:
            max_reward = max(rewards) if max(rewards) > 0 else 1
            min_reward = min(rewards) if min(rewards) < 0 else 0
            reward_range = max_reward - min_reward if max_reward != min_reward else 1

            points = []
            for i, reward in enumerate(rewards):
                x = padding + 20 + (i / (len(rewards) - 1)) * (chart_width - 40)
                y = (
                    chart_y
                    + chart_height
                    - 40
                    - ((reward - min_reward) / reward_range) * (chart_height - 80)
                )
                points.append((x, y))

            # Draw line
            if len(points) > 1:
                path_data = f"M {points[0][0]} {points[0][1]}"
                for x, y in points[1:]:
                    path_data += f" L {x} {y}"

                drawing.append(
                    draw.Path(
                        d=path_data,
                        stroke="#3498db",
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
                            fill="#3498db",
                            stroke="white",
                            stroke_width=1,
                        )
                    )

    # Add key moments section if available
    if summary_data.key_moments:
        key_moments_y = chart_y + chart_height + 30
        drawing.append(
            draw.Text(
                f"Key Moments: Steps {', '.join(map(str, summary_data.key_moments))}",
                font_size=14,
                x=width / 2,
                y=key_moments_y,
                text_anchor="middle",
                font_family="Anuphan",
                font_weight="400",
                fill="#6c757d",
            )
        )

    # Add task information
    task_info_y = height - 60
    task_info = f"Task: {summary_data.task_id}"
    if hasattr(summary_data, "start_time") and hasattr(summary_data, "end_time"):
        duration = summary_data.end_time - summary_data.start_time
        task_info += f" | Duration: {duration:.1f}s"

    drawing.append(
        draw.Text(
            task_info,
            font_size=12,
            x=width / 2,
            y=task_info_y,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="300",
            fill="#adb5bd",
        )
    )

    return drawing.as_svg()


# Wrapper functions for backward compatibility
def draw_rl_step_svg(
    before_grid: Grid,
    after_grid: Grid,
    action: Dict[str, Any],
    reward: float,
    info: Dict[str, Any],
    step_num: int,
    operation_name: str = "",
    changed_cells: Optional[jnp.ndarray] = None,
    config: Optional[Any] = None,
    **kwargs,
) -> str:
    """Enhanced wrapper for draw_rl_step_svg_enhanced with backward compatibility."""
    return draw_rl_step_svg_enhanced(
        before_grid=before_grid,
        after_grid=after_grid,
        action=action,
        reward=reward,
        info=info,
        step_num=step_num,
        operation_name=operation_name,
        changed_cells=changed_cells,
        config=config,
        **kwargs,
    )


def detect_changed_cells(
    before_grid: Grid,
    after_grid: Grid,
) -> jnp.ndarray:
    """Detect which cells changed between before and after grids.

    Args:
        before_grid: Grid state before the action
        after_grid: Grid state after the action

    Returns:
        Boolean mask indicating which cells changed
    """
    before_data = np.asarray(before_grid.data)
    after_data = np.asarray(after_grid.data)

    # Handle different shapes by padding to match
    max_height = max(before_data.shape[0], after_data.shape[0])
    max_width = max(before_data.shape[1], after_data.shape[1])

    # Pad both grids to same size
    before_padded = np.zeros((max_height, max_width), dtype=before_data.dtype)
    after_padded = np.zeros((max_height, max_width), dtype=after_data.dtype)

    before_padded[: before_data.shape[0], : before_data.shape[1]] = before_data
    after_padded[: after_data.shape[0], : after_data.shape[1]] = after_data

    # Find changed cells
    changed = before_padded != after_padded

    return jnp.array(changed)


def get_operation_display_name(
    operation_id: int, action_data: Dict[str, Any] = None
) -> str:
    """Get human-readable operation name from operation ID with context.

    Args:
        operation_id: Integer operation ID
        action_data: Optional action data for additional context

    Returns:
        Human-readable operation name with context
    """
    # Map of operation IDs to display names (enhanced for visualization)
    operation_names = {
        # Fill operations (0-9) - Enhanced with color names for clarity
        0: "Fill Black (0)",
        1: "Fill Blue (1)",
        2: "Fill Red (2)",
        3: "Fill Green (3)",
        4: "Fill Yellow (4)",
        5: "Fill Grey (5)",
        6: "Fill Pink (6)",
        7: "Fill Orange (7)",
        8: "Fill Light Blue (8)",
        9: "Fill Brown (9)",
        # Flood fill operations (10-19) - Enhanced with color names for clarity
        10: "Flood Fill Black (0)",
        11: "Flood Fill Blue (1)",
        12: "Flood Fill Red (2)",
        13: "Flood Fill Green (3)",
        14: "Flood Fill Yellow (4)",
        15: "Flood Fill Grey (5)",
        16: "Flood Fill Pink (6)",
        17: "Flood Fill Orange (7)",
        18: "Flood Fill Light Blue (8)",
        19: "Flood Fill Brown (9)",
        # Movement operations (20-23)
        20: "Move Up",
        21: "Move Down",
        22: "Move Left",
        23: "Move Right",
        # Transformation operations (24-27)
        24: "Rotate CW",
        25: "Rotate CCW",
        26: "Flip H",
        27: "Flip V",
        # Editing operations (28-31)
        28: "Copy",
        29: "Paste",
        30: "Cut",
        31: "Clear",
        # Special operations (32-34)
        32: "Copy Input",
        33: "Resize",
        34: "Submit",
        # Enhanced control operations (35-41)
        35: "Next Demo Pair",
        36: "Prev Demo Pair",
        37: "Next Test Pair",
        38: "Prev Test Pair",
        39: "Reset Current Pair",
        40: "First Unsolved Demo",
        41: "First Unsolved Test",
    }

    base_name = operation_names.get(operation_id, f"Operation {operation_id}")

    # For fill operations (0-9), the color is already included in the name
    # For flood fill operations (10-19), the color is already included in the name
    # No need for additional context processing since the names are already descriptive

    return base_name


def get_color_name(color_id: int) -> str:
    """Get human-readable color name from color ID.

    Args:
        color_id: Integer color ID

    Returns:
        Human-readable color name
    """
    color_names = {
        0: "Black (0)",
        1: "Blue (1)",
        2: "Red (2)",
        3: "Green (3)",
        4: "Yellow (4)",
        5: "Grey (5)",
        6: "Pink (6)",
        7: "Orange (7)",
        8: "Light Blue (8)",
        9: "Brown (9)",
    }

    return color_names.get(color_id, f"Color {color_id}")


def infer_fill_color_from_grids(
    before_grid: Grid, after_grid: Grid, selection_mask: np.ndarray
) -> int:
    """Infer what color was used to fill selected cells by comparing grids.

    Args:
        before_grid: Grid state before the action
        after_grid: Grid state after the action
        selection_mask: Boolean mask of selected cells

    Returns:
        Color ID that was used for filling, or -1 if can't determine
    """
    try:
        before_data = np.asarray(before_grid.data)
        after_data = np.asarray(after_grid.data)

        # Find cells that were selected and changed
        for i in range(min(before_data.shape[0], after_data.shape[0])):
            for j in range(min(before_data.shape[1], after_data.shape[1])):
                if (
                    i < selection_mask.shape[0]
                    and j < selection_mask.shape[1]
                    and selection_mask[i, j]
                    and before_data[i, j] != after_data[i, j]
                ):
                    # This cell was selected and changed, return the new color
                    return int(after_data[i, j])

        return -1  # Couldn't determine
    except Exception:
        return -1


def add_selection_visualization_overlay(
    drawing: Any,
    selection_mask: np.ndarray,
    grid_x: float,
    grid_y: float,
    cell_size: float,
    start_row: int,
    start_col: int,
    display_height: int,
    display_width: int,
    selection_color: str = "#3498db",
    selection_opacity: float = 0.3,
    border_width: float = 2,
) -> None:
    """Add selection visualization overlay to a grid.

    Args:
        drawing: DrawSVG drawing object to add overlay to
        selection_mask: Boolean mask of selected cells
        grid_x: X position of grid
        grid_y: Y position of grid
        cell_size: Size of each cell
        start_row: Starting row of valid region
        start_col: Starting column of valid region
        display_height: Height of display region
        display_width: Width of display region
        selection_color: Color for selection highlight
        selection_opacity: Opacity of selection fill
        border_width: Width of selection border
    """
    import drawsvg as draw

    if not selection_mask.any():
        return

    # First pass: draw filled rectangles
    for display_row in range(display_height):
        for display_col in range(display_width):
            orig_row = start_row + display_row
            orig_col = start_col + display_col

            if (
                orig_row < selection_mask.shape[0]
                and orig_col < selection_mask.shape[1]
                and selection_mask[orig_row, orig_col]
            ):
                cell_x = grid_x + display_col * cell_size
                cell_y = grid_y + display_row * cell_size

                drawing.append(
                    draw.Rectangle(
                        cell_x,
                        cell_y,
                        cell_size,
                        cell_size,
                        fill=selection_color,
                        fill_opacity=selection_opacity,
                        stroke="none",
                    )
                )

    # Second pass: draw boundary lines only on outer edges
    def is_selected(row, col):
        """Check if a cell is selected, handling bounds."""
        if (
            row < 0
            or row >= selection_mask.shape[0]
            or col < 0
            or col >= selection_mask.shape[1]
        ):
            return False
        return selection_mask[row, col]

    for display_row in range(display_height):
        for display_col in range(display_width):
            orig_row = start_row + display_row
            orig_col = start_col + display_col

            if (
                orig_row < selection_mask.shape[0]
                and orig_col < selection_mask.shape[1]
                and selection_mask[orig_row, orig_col]
            ):
                cell_x = grid_x + display_col * cell_size
                cell_y = grid_y + display_row * cell_size

                # Check each edge and draw border line if it's on the boundary
                # Top edge
                if not is_selected(orig_row - 1, orig_col):
                    drawing.append(
                        draw.Line(
                            cell_x,
                            cell_y,
                            cell_x + cell_size,
                            cell_y,
                            stroke=selection_color,
                            stroke_width=border_width,
                            stroke_opacity=0.9,
                        )
                    )

                # Bottom edge
                if not is_selected(orig_row + 1, orig_col):
                    drawing.append(
                        draw.Line(
                            cell_x,
                            cell_y + cell_size,
                            cell_x + cell_size,
                            cell_y + cell_size,
                            stroke=selection_color,
                            stroke_width=border_width,
                            stroke_opacity=0.9,
                        )
                    )

                # Left edge
                if not is_selected(orig_row, orig_col - 1):
                    drawing.append(
                        draw.Line(
                            cell_x,
                            cell_y,
                            cell_x,
                            cell_y + cell_size,
                            stroke=selection_color,
                            stroke_width=border_width,
                            stroke_opacity=0.9,
                        )
                    )

                # Right edge
                if not is_selected(orig_row, orig_col + 1):
                    drawing.append(
                        draw.Line(
                            cell_x + cell_size,
                            cell_y,
                            cell_x + cell_size,
                            cell_y + cell_size,
                            stroke=selection_color,
                            stroke_width=border_width,
                            stroke_opacity=0.9,
                        )
                    )


def add_change_highlighting(
    drawing: Any,
    changed_cells: np.ndarray,
    grid_x: float,
    grid_y: float,
    cell_size: float,
    start_row: int,
    start_col: int,
    display_height: int,
    display_width: int,
    change_color: str = "#ff6b6b",
    border_width: float = 3,
) -> None:
    """Add change highlighting overlay to a grid.

    Args:
        drawing: DrawSVG drawing object to add overlay to
        changed_cells: Boolean mask of changed cells
        grid_x: X position of grid
        grid_y: Y position of grid
        cell_size: Size of each cell
        start_row: Starting row of valid region
        start_col: Starting column of valid region
        display_height: Height of display region
        display_width: Width of display region
        change_color: Color for change highlight
        border_width: Width of change border
    """
    import drawsvg as draw

    if not changed_cells.any():
        return

    # Add pulsing border for changed cells
    for display_row in range(display_height):
        for display_col in range(display_width):
            orig_row = start_row + display_row
            orig_col = start_col + display_col

            if (
                orig_row < changed_cells.shape[0]
                and orig_col < changed_cells.shape[1]
                and changed_cells[orig_row, orig_col]
            ):
                cell_x = grid_x + display_col * cell_size
                cell_y = grid_y + display_row * cell_size

                # Add animated border effect
                drawing.append(
                    draw.Rectangle(
                        cell_x - border_width / 2,
                        cell_y - border_width / 2,
                        cell_size + border_width,
                        cell_size + border_width,
                        fill="none",
                        stroke=change_color,
                        stroke_width=border_width,
                        stroke_opacity=0.8,
                    )
                )

                # Add inner glow effect
                drawing.append(
                    draw.Rectangle(
                        cell_x + 1,
                        cell_y + 1,
                        cell_size - 2,
                        cell_size - 2,
                        fill=change_color,
                        fill_opacity=0.1,
                        stroke="none",
                    )
                )


def create_action_summary_panel(
    action: Dict[str, Any],
    reward: float,
    info: Dict[str, Any],
    operation_name: str = "",
    width: float = 400,
    height: float = 100,
) -> str:
    """Create an action summary panel as SVG.

    Args:
        action: Action dictionary
        reward: Reward received
        info: Additional information
        operation_name: Human-readable operation name
        width: Panel width
        height: Panel height

    Returns:
        SVG string for the action summary panel
    """
    import drawsvg as draw

    drawing = draw.Drawing(width, height)

    # Panel background
    drawing.append(
        draw.Rectangle(
            0,
            0,
            width,
            height,
            fill="#ffffff",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Title
    drawing.append(
        draw.Text(
            "Action Summary",
            font_size=16,
            x=10,
            y=25,
            text_anchor="start",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Operation info
    if operation_name:
        drawing.append(
            draw.Text(
                f"Operation: {operation_name}",
                font_size=14,
                x=10,
                y=45,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="400",
                fill="#495057",
            )
        )

    # Reward info
    reward_color = "#27ae60" if reward > 0 else "#e74c3c" if reward < 0 else "#95a5a6"
    drawing.append(
        draw.Text(
            f"Reward: {reward:.3f}",
            font_size=14,
            x=10,
            y=65,
            text_anchor="start",
            font_family="Anuphan",
            font_weight="500",
            fill=reward_color,
        )
    )

    # Additional info
    if "similarity" in info:
        similarity_val = float(info['similarity']) if hasattr(info['similarity'], 'item') else info['similarity']
        drawing.append(
            draw.Text(
                f"Similarity: {similarity_val:.3f}",
                font_size=12,
                x=10,
                y=85,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="400",
                fill="#6c757d",
            )
        )

    return drawing.as_svg()


def create_metrics_visualization(
    metrics: Dict[str, float],
    width: float = 300,
    height: float = 200,
) -> str:
    """Create a metrics visualization panel.

    Args:
        metrics: Dictionary of metric names to values
        width: Panel width
        height: Panel height

    Returns:
        SVG string for the metrics panel
    """
    import drawsvg as draw

    drawing = draw.Drawing(width, height)

    # Panel background
    drawing.append(
        draw.Rectangle(
            0,
            0,
            width,
            height,
            fill="#ffffff",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Title
    drawing.append(
        draw.Text(
            "Step Metrics",
            font_size=16,
            x=10,
            y=25,
            text_anchor="start",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Display metrics
    y_pos = 50
    for name, value in metrics.items():
        # Metric name
        drawing.append(
            draw.Text(
                f"{name}:",
                font_size=12,
                x=10,
                y=y_pos,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="500",
                fill="#495057",
            )
        )

        # Metric value
        drawing.append(
            draw.Text(
                f"{value:.3f}",
                font_size=12,
                x=width - 10,
                y=y_pos,
                text_anchor="end",
                font_family="Anuphan",
                font_weight="400",
                fill="#6c757d",
            )
        )

        y_pos += 20

        if y_pos > height - 20:
            break

    return drawing.as_svg()


# Update the visualizer to use these new functions
def update_visualizer_step_creation():
    """Update the visualizer to use the new step visualization functions."""
    # This is a placeholder function to indicate where the Visualizer
    # would be updated to use the new helper functions above.
    # The actual integration would happen in the _create_step_svg method.


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


# Update the episode summary function to use the enhanced version
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
