"""Grid visualization utilities for JaxARC.

This module provides functionality to visualize ARC grids and tasks in different formats:
- Rich-based terminal visualization for logging and debugging
- SVG-based image generation for documentation and analysis

The module works with the core JaxARC data structures including Grid, TaskPair,
ArcTask, and ParsedTaskData.
"""

from __future__ import annotations

import io
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

if TYPE_CHECKING:
    from jaxarc.types import Grid, ParsedTaskData

# ARC color palette - matches the provided color map
ARC_COLOR_PALETTE: list[str] = [
    "#252525",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#37D449",  # 3: green
    "#FFDC00",  # 4: yellow
    "#E6E6E6",  # 5: grey
    "#F012BE",  # 6: pink
    "#FF871E",  # 7: orange
    "#54D2EB",  # 8: light blue
    "#8D1D2C",  # 9: brown
    "#FFFFFF",  # 10: white (for padding/invalid)
]

# Rich color mapping for terminal display
# Using Rich's color names that are closest to the ARC palette
RICH_COLOR_MAP: dict[int, str] = {
    0: "#252525",
    1: "#0074D9",
    2: "#FF4136",
    3: "#37D449",
    4: "#FFDC00",
    5: "#E6E6E6",  # grey -> white for better visibility
    6: "#F012BE",  # pink -> magenta
    7: "#FF871E",  # orange -> bright_yellow
    8: "#54D2EB",  # light blue -> cyan
    9: "#8D1D2C",  # brown -> red3
    -1: "#FFFFFF",  # padding/invalid cells
}
# RICH_COLOR_MAP: dict[int, str] = {
#     0: "black",
#     1: "blue",
#     2: "red",
#     3: "green",
#     4: "yellow",
#     5: "white",  # grey -> white for better visibility
#     6: "magenta",  # pink -> magenta
#     7: "bright_yellow",  # orange -> bright_yellow
#     8: "cyan",  # light blue -> cyan
#     9: "red3",  # brown -> red3
#     -1: "grey23",  # padding/invalid cells
# }

# RICH_COLOR_MAP = {
#     0: "black",
#     1: "blue",
#     2: "red",
#     3: "green",
#     4: "yellow",
#     5: "white",  # grey -> white for better visibility
#     6: "magenta",  # pink -> magenta
#     7: "bright_yellow",  # orange -> bright_yellow
#     8: "cyan",  # light blue -> cyan
#     9: "red3",  # brown -> red3
#     -1: "grey23",  # padding/invalid cells
# }


def _extract_grid_data(grid_input: jnp.ndarray | np.ndarray | Grid) -> np.ndarray:
    """Extract numpy array from various grid input types.

    Args:
        grid_input: Grid data in various formats (JAX array, numpy array, or Grid object)

    Returns:
        numpy array representation of the grid

    Raises:
        ValueError: If input format is not supported
    """
    if hasattr(grid_input, "array"):  # Grid object
        return np.asarray(grid_input.array)  # type: ignore[attr-defined]
    if isinstance(grid_input, (jnp.ndarray, np.ndarray)):
        return np.asarray(grid_input)

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
    grid = _extract_grid_data(grid_input)

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
                rich_color = RICH_COLOR_MAP.get(color_val, "white")
                row_items.append(f"[{rich_color}]{color_val}[/]")
            elif double_width:
                # Use double-width blocks for more square appearance
                rich_color = RICH_COLOR_MAP.get(color_val, "white")
                row_items.append(f"[{rich_color}]██[/]")
            else:
                # Use single block character
                rich_color = RICH_COLOR_MAP.get(color_val, "white")
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
    grid = _extract_grid_data(grid_input)

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

            if is_valid and 0 <= color_val < len(ARC_COLOR_PALETTE):
                fill_color = ARC_COLOR_PALETTE[color_val]
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
    input_grid_data = _extract_grid_data(input_grid)
    input_mask_data = np.asarray(input_mask) if input_mask is not None else None
    _, _, (input_h, input_w) = _extract_valid_region(input_grid_data, input_mask_data)

    input_ratio = input_w / input_h if input_h > 0 else 1.0
    max_ratio = input_ratio

    if output_grid is not None:
        output_grid_data = _extract_grid_data(output_grid)
        output_mask_data = np.asarray(output_mask) if output_mask is not None else None
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
        final_drawing_width, final_drawing_height + 0.2, origin=(0, 0)
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
    task_data: ParsedTaskData,
    show_test: bool = True,
    show_coordinates: bool = False,
    show_numbers: bool = False,
    double_width: bool = True,
) -> None:
    """Visualize a ParsedTaskData object using Rich console output with enhanced layout and grouping.

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
    task_title = f"Task: {task_data.task_id or 'Unknown'}"

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
    task_data: ParsedTaskData,
    width: float = 30.0,
    height: float = 20.0,
    include_test: bool | str = False,
    border_colors: list[str] | None = None,
) -> drawsvg.Drawing:
    """Draw a complete ParsedTaskData as an SVG with strict height and flexible width.

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
                f"Task {task_data.task_id or 'Unknown'} (No examples)",
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
        label,
        is_test,
    ) in enumerate(examples):
        input_grid_data = _extract_grid_data(input_grid)
        input_mask_data = np.asarray(input_mask) if input_mask is not None else None
        _, _, (input_h, input_w) = _extract_valid_region(
            input_grid_data, input_mask_data
        )

        input_ratio = input_w / input_h if input_h > 0 else 1.0
        max_ratio = input_ratio

        if output_grid is not None:
            output_grid_data = _extract_grid_data(output_grid)
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
        final_drawing_width, final_drawing_height + 0.2, origin=(0, 0)
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
    title_text = f"Task {task_data.task_id or 'Unknown'}"
    drawing.append(
        drawsvg.Text(
            title_text,
            x=final_drawing_width - 0.1,
            y=final_drawing_height + 0.1,
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
