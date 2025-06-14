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
from rich.console import Console
from rich.table import Table

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
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "white",  # grey -> white for better visibility
    6: "magenta",  # pink -> magenta
    7: "bright_yellow",  # orange -> bright_yellow
    8: "cyan",  # light blue -> cyan
    9: "red3",  # brown -> red3
    -1: "grey23",  # padding/invalid cells
}


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
    grid: np.ndarray, 
    mask: np.ndarray | None = None
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
    
    return valid_grid, (start_row, start_col), (end_row - start_row, end_col - start_col)


def visualize_grid_rich(
    grid_input: jnp.ndarray | np.ndarray | Grid,
    mask: jnp.ndarray | np.ndarray | None = None,
    title: str = "Grid",
    show_coordinates: bool = False,
    show_numbers: bool = False,
    double_width: bool = True,
) -> Table:
    """Create a Rich Table visualization of a single grid.
    
    Args:
        grid_input: Grid data (JAX array, numpy array, or Grid object)
        mask: Optional boolean mask indicating valid cells
        title: Title for the table
        show_coordinates: Whether to show row/column coordinates
        show_numbers: If True, show colored numbers; if False, show colored blocks
        double_width: If True and show_numbers=False, use double-width blocks for square appearance
        
    Returns:
        Rich Table object for display
    """
    grid = _extract_grid_data(grid_input)
    
    if mask is not None:
        mask = np.asarray(mask)
    
    if grid.size == 0:
        table = Table(title=f"{title} (Empty)", show_header=False, show_edge=True)
        table.add_column("Empty")
        table.add_row("[grey23]Empty grid[/]")
        return table
    
    # Extract valid region
    valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(grid, mask)
    
    if height == 0 or width == 0:
        table = Table(title=f"{title} (No valid data)", show_header=False, show_edge=True)
        table.add_column("Empty")
        table.add_row("[grey23]No valid data[/]")
        return table
    
    # Create table
    table = Table(
        title=f"{title} ({height}x{width})",
        show_header=show_coordinates,
        show_edge=True,
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
        table.add_column(col_header, justify="center", width=col_width)
    
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
                if (actual_row < mask.shape[0] and actual_col < mask.shape[1]):
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
    
    return table


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
            return drawsvg.Group(), (-0.5 * padding, -0.5 * padding), (padding, padding + extra_bottom_padding)
        
        drawing = drawsvg.Drawing(
            padding, 
            padding + extra_bottom_padding,
            origin=(-0.5 * padding, -0.5 * padding)
        )
        drawing.set_pixel_scale(40)
        return drawing
    
    # Extract valid region
    valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(grid, mask)
    
    if height == 0 or width == 0:
        if as_group:
            return drawsvg.Group(), (-0.5 * padding, -0.5 * padding), (padding, padding + extra_bottom_padding)
        
        drawing = drawsvg.Drawing(
            padding,
            padding + extra_bottom_padding, 
            origin=(-0.5 * padding, -0.5 * padding)
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
            origin=(-0.5 * padding, -0.5 * padding)
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
                if (actual_row < mask.shape[0] and actual_col < mask.shape[1]):
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
            text=set("Input Output 0123456789x Test Task ABCDEFGHIJ? abcdefghjklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ")
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
) -> tuple[Table, Table | None]:
    """Visualize an input-output pair using Rich tables.
    
    Args:
        input_grid: Input grid data
        output_grid: Output grid data (optional)
        input_mask: Optional mask for input grid
        output_mask: Optional mask for output grid
        title: Title for the visualization
        show_numbers: If True, show colored numbers; if False, show colored blocks
        double_width: If True and show_numbers=False, use double-width blocks for square appearance
        
    Returns:
        Tuple of (input_table, output_table). output_table is None if no output_grid provided.
    """
    input_table = visualize_grid_rich(
        input_grid, 
        input_mask, 
        f"{title} - Input",
        show_numbers=show_numbers,
        double_width=double_width,
    )
    
    output_table = None
    if output_grid is not None:
        output_table = visualize_grid_rich(
            output_grid,
            output_mask,
            f"{title} - Output",
            show_numbers=show_numbers,
            double_width=double_width,
        )
    
    return input_table, output_table


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
    """Draw an input-output task pair as SVG.
    
    Args:
        input_grid: Input grid data
        output_grid: Output grid data (optional)
        input_mask: Optional mask for input grid
        output_mask: Optional mask for output grid
        width: Total width of the drawing
        height: Total height of the drawing
        label: Label for the pair
        show_unknown_output: Whether to show "?" for missing output
        
    Returns:
        SVG Drawing object
    """
    padding = 0.5
    io_gap = 0.8
    max_grid_height = (height - padding - io_gap) / 2
    max_grid_width = width / 2 - padding
    
    # Draw input grid
    input_result = draw_grid_svg(
        input_grid,
        input_mask,
        max_width=max_grid_width,
        max_height=max_grid_height,
        label=f"{label} Input" if label else "Input",
        as_group=True,
    )
    
    if isinstance(input_result, tuple):
        input_group, input_origin, input_size = input_result
    else:
        # This shouldn't happen when as_group=True, but handle gracefully
        msg = "Expected tuple result when as_group=True"
        raise ValueError(msg)
    
    # Calculate positions
    input_x = padding / 2
    input_y = 0
    
    # Create main drawing
    drawing = drawsvg.Drawing(width, height, origin=(0, 0))
    drawing.append(drawsvg.Rectangle(0, 0, "100%", "100%", fill="#eeeff6"))
    
    # Add input grid
    drawing.append(
        drawsvg.Use(
            input_group,
            x=input_x - input_origin[0],
            y=input_y - input_origin[1],
        )
    )
    
    # Draw arrow
    arrow_x = input_x + input_size[0] / 2
    arrow_top_y = input_size[1] - 0.3
    arrow_bottom_y = input_size[1] + io_gap - 0.3
    
    drawing.append(drawsvg.Line(
        arrow_x, arrow_top_y, arrow_x, arrow_bottom_y,
        stroke_width=0.05, stroke="#888888"
    ))
    drawing.append(drawsvg.Line(
        arrow_x - 0.15, arrow_bottom_y - 0.2,
        arrow_x, arrow_bottom_y,
        stroke_width=0.05, stroke="#888888"
    ))
    drawing.append(drawsvg.Line(
        arrow_x + 0.15, arrow_bottom_y - 0.2,
        arrow_x, arrow_bottom_y,
        stroke_width=0.05, stroke="#888888"
    ))
    
    # Draw output
    output_y = input_size[1] + io_gap
    
    if output_grid is not None:
        output_result = draw_grid_svg(
            output_grid,
            output_mask,
            max_width=max_grid_width,
            max_height=max_grid_height,
            label=f"{label} Output" if label else "Output",
            as_group=True,
        )
        
        if isinstance(output_result, tuple):
            output_group, output_origin, _output_size = output_result
        else:
            msg = "Expected tuple result when as_group=True"
            raise ValueError(msg)
        
        drawing.append(
            drawsvg.Use(
                output_group,
                x=input_x - output_origin[0],
                y=output_y - output_origin[1],
            )
        )
    elif show_unknown_output:
        # Draw question mark
        drawing.append(
            drawsvg.Text(
                "?",
                x=input_x + max_grid_width / 2,
                y=output_y + max_grid_height / 2,
                font_size=1.0,
                font_family="Anuphan",
                font_weight="700",
                fill="#333333",
                text_anchor="middle",
                alignment_baseline="middle",
            )
        )
    
    # Embed font and set scale
    drawing.embed_google_font(
        "Anuphan:wght@400;600;700",
        text=set("Input Output 0123456789x Test Task ABCDEFGHIJ? abcdefghjklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ")
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
    """Visualize a ParsedTaskData object using Rich console output.
    
    Args:
        task_data: The parsed task data to visualize
        show_test: Whether to show test pairs
        show_coordinates: Whether to show grid coordinates
        show_numbers: If True, show colored numbers; if False, show colored blocks
        double_width: If True and show_numbers=False, use double-width blocks for square appearance
    """
    console = Console()
    
    # Show task info
    console.print(f"\n[bold blue]Task: {task_data.task_id or 'Unknown'}[/bold blue]")
    console.print(f"Training pairs: {task_data.num_train_pairs}")
    console.print(f"Test pairs: {task_data.num_test_pairs}")
    
    # Show training examples
    for i in range(task_data.num_train_pairs):
        console.print(f"\n[bold]Training Example {i + 1}[/bold]")
        
        # Input
        input_table = visualize_grid_rich(
            task_data.input_grids_examples[i],
            task_data.input_masks_examples[i],
            f"Input {i + 1}",
            show_coordinates,
            show_numbers,
            double_width,
        )
        console.print(input_table)
        
        # Output
        output_table = visualize_grid_rich(
            task_data.output_grids_examples[i],
            task_data.output_masks_examples[i],
            f"Output {i + 1}",
            show_coordinates,
            show_numbers,
            double_width,
        )
        console.print(output_table)
    
    # Show test examples
    if show_test:
        for i in range(task_data.num_test_pairs):
            console.print(f"\n[bold]Test Example {i + 1}[/bold]")
            
            # Test input
            test_input_table = visualize_grid_rich(
                task_data.test_input_grids[i],
                task_data.test_input_masks[i],
                f"Test Input {i + 1}",
                show_coordinates,
                show_numbers,
                double_width,
            )
            console.print(test_input_table)
            
            # Test output (ground truth)
            test_output_table = visualize_grid_rich(
                task_data.true_test_output_grids[i],
                task_data.true_test_output_masks[i],
                f"Test Output {i + 1} (Ground Truth)",
                show_coordinates,
                show_numbers,
                double_width,
            )
            console.print(test_output_table)


def draw_parsed_task_data_svg(
    task_data: ParsedTaskData,
    width: float = 30.0,
    height: float = 20.0,
    include_test: bool | str = False,
    border_colors: list[str] | None = None,
) -> drawsvg.Drawing:
    """Draw a complete ParsedTaskData as an SVG.
    
    Args:
        task_data: The parsed task data to visualize
        width: Desired width of the drawing
        height: Desired height of the drawing
        include_test: Whether to include test examples. If 'all', show test outputs too.
        border_colors: Custom border colors [input_color, output_color]
        
    Returns:
        SVG Drawing object
    """
    if border_colors is None:
        border_colors = ["#111111ff", "#111111ff"]
    
    padding = 0.5
    io_gap = 0.6
    pair_gap = 0.3
    
    # Calculate available space per grid
    max_grid_height = (height - padding * 2 - io_gap) / 2
    
    # Determine how many examples to show
    num_examples = task_data.num_train_pairs
    if include_test:
        num_examples += task_data.num_test_pairs
    
    if num_examples == 0:
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
    
    # Calculate width allocation for each example
    available_width = width - padding * (num_examples + 1)
    width_per_example = available_width / num_examples
    
    # Create main drawing
    drawing = drawsvg.Drawing(width, height, origin=(0, 0))
    drawing.append(drawsvg.Rectangle(0, 0, "100%", "100%", fill="#eeeff6"))
    
    x_pos = padding
    max_input_height = 0
    
    example_data = []
    
    # Prepare training examples
    for i in range(task_data.num_train_pairs):
        example_data.append({
            'input_grid': task_data.input_grids_examples[i],
            'input_mask': task_data.input_masks_examples[i],
            'output_grid': task_data.output_grids_examples[i],
            'output_mask': task_data.output_masks_examples[i],
            'label': f"Example {i + 1}",
            'is_test': False,
        })
    
    # Prepare test examples
    if include_test:
        for i in range(task_data.num_test_pairs):
            show_test_output = include_test == "all"
            example_data.append({
                'input_grid': task_data.test_input_grids[i],
                'input_mask': task_data.test_input_masks[i],
                'output_grid': task_data.true_test_output_grids[i] if show_test_output else None,
                'output_mask': task_data.true_test_output_masks[i] if show_test_output else None,
                'label': f"Test {i + 1}",
                'is_test': True,
            })
    
    # Draw each example
    for example in example_data:
        # Draw input
        input_result = draw_grid_svg(
            example['input_grid'],
            example['input_mask'],
            max_width=width_per_example - pair_gap,
            max_height=max_grid_height,
            label=f"{example['label']} Input",
            border_color=border_colors[0],
            as_group=True,
        )
        
        if isinstance(input_result, tuple):
            input_group, input_origin, input_size = input_result
        else:
            msg = "Expected tuple result when as_group=True"
            raise ValueError(msg)
        
        drawing.append(
            drawsvg.Use(
                input_group,
                x=x_pos - input_origin[0],
                y=padding - input_origin[1],
            )
        )
        
        max_input_height = max(max_input_height, input_size[1])
        
        # Draw arrow
        arrow_x = x_pos + input_size[0] / 2
        arrow_top_y = padding + input_size[1] - 0.3
        arrow_bottom_y = padding + max_input_height + io_gap - 0.3
        
        drawing.append(drawsvg.Line(
            arrow_x, arrow_top_y, arrow_x, arrow_bottom_y,
            stroke_width=0.05, stroke="#888888"
        ))
        drawing.append(drawsvg.Line(
            arrow_x - 0.15, arrow_bottom_y - 0.2,
            arrow_x, arrow_bottom_y,
            stroke_width=0.05, stroke="#888888"
        ))
        drawing.append(drawsvg.Line(
            arrow_x + 0.15, arrow_bottom_y - 0.2,
            arrow_x, arrow_bottom_y,
            stroke_width=0.05, stroke="#888888"
        ))
        
        # Draw output
        output_y = padding + max_input_height + io_gap
        
        if example['output_grid'] is not None:
            output_result = draw_grid_svg(
                example['output_grid'],
                example['output_mask'],
                max_width=width_per_example - pair_gap,
                max_height=max_grid_height,
                label=f"{example['label']} Output",
                border_color=border_colors[1],
                as_group=True,
            )
            
            if isinstance(output_result, tuple):
                output_group, output_origin, _output_size = output_result
            else:
                msg = "Expected tuple result when as_group=True"
                raise ValueError(msg)
            
            drawing.append(
                drawsvg.Use(
                    output_group,
                    x=x_pos - output_origin[0],
                    y=output_y - output_origin[1],
                )
            )
        else:
            # Draw question mark for unknown output
            drawing.append(
                drawsvg.Text(
                    "?",
                    x=x_pos + width_per_example / 2,
                    y=output_y + max_grid_height / 2,
                    font_size=1.0,
                    font_family="Anuphan",
                    font_weight="700",
                    fill="#333333",
                    text_anchor="middle",
                    alignment_baseline="middle",
                )
            )
        
        x_pos += width_per_example + pair_gap
    
    # Add title
    font_size = 0.3
    title_text = f"Task {task_data.task_id or 'Unknown'}"
    drawing.append(
        drawsvg.Text(
            title_text,
            x=width - 0.1,
            y=height - 0.1,
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
        text=set("Input Output 0123456789x Test Task ABCDEFGHIJ? abcdefghjklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ")
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
        error_msg = f"Unknown file extension for {filename}. Supported: .svg, .png, .pdf"
        raise ValueError(error_msg)


# Fix the typo in RICH_COLOR_MAP
RICH_COLOR_MAP = {
    0: "black",
    1: "blue", 
    2: "red",
    3: "green",
    4: "yellow",
    5: "white",  # grey -> white for better visibility
    6: "magenta",  # pink -> magenta
    7: "bright_yellow",  # orange -> bright_yellow
    8: "cyan",  # light blue -> cyan
    9: "red3",  # brown -> red3
    -1: "grey23",  # padding/invalid cells
}
