"""
Rendering utilities for JaxARC environment.

This module provides rendering functions for different modes:
- RGB Array: Returns a numpy array of RGB values.
- ANSI: Returns a string with ANSI color codes for terminal display.
- SVG: Returns an SVG string or object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import drawsvg
import numpy as np
from loguru import logger

from jaxarc.utils.visualization.core import ARC_COLOR_PALETTE, draw_grid_svg

if TYPE_CHECKING:
    from jaxarc.state import State

try:
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Define dummy Console for type checking or fallback
    Console = None  # type: ignore


# ============================================================================
# RGB Rendering
# ============================================================================


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 8:  # Handle RGBA (ignore alpha for now)
        hex_color = hex_color[:6]
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


def render_rgb(state: State, cell_size: int = 20) -> np.ndarray:
    """
    Render the current state as an RGB array.

    Args:
        state: The current environment state.
        cell_size: The size of each grid cell in pixels.

    Returns:
        A numpy array of shape (height * cell_size, width * cell_size, 3) representing the RGB image.
    """
    # Extract grid data from state (convert to numpy for rendering)
    grid = np.array(state.working_grid)
    height, width = grid.shape

    # Create empty RGB array
    img_height = height * cell_size
    img_width = width * cell_size
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Precompute color map
    color_map = {
        k: hex_to_rgb(v) for k, v in ARC_COLOR_PALETTE.items()
    }
    # Add default for unknown colors (gray)
    default_color = (128, 128, 128)

    for r in range(height):
        for c in range(width):
            color_id = int(grid[r, c])
            rgb = color_map.get(color_id, default_color)
            
            # Fill the cell
            r_start = r * cell_size
            r_end = (r + 1) * cell_size
            c_start = c * cell_size
            c_end = (c + 1) * cell_size
            
            img[r_start:r_end, c_start:c_end] = rgb

            # Draw grid lines (optional, maybe simple 1px border)
            # For now, let's keep it simple without borders or add a thin border
            # Adding a 1px border for better visibility
            if cell_size > 2:
                border_color = (50, 50, 50)
                img[r_start:r_end, c_start] = border_color
                img[r_start, c_start:c_end] = border_color

    return img


# ============================================================================
# ANSI Rendering
# ============================================================================


def render_ansi(state: State) -> str:
    """
    Render the current state as an ANSI string.

    Args:
        state: The current environment state.

    Returns:
        A string containing ANSI color codes representing the grid.
    """
    if not RICH_AVAILABLE:
        logger.warning("rich library not found. Falling back to simple string representation.")
        return str(np.array(state.working_grid))

    grid = np.array(state.working_grid)
    height, width = grid.shape

    # ARC colors to Rich colors mapping (approximate)
    # 0: black, 1: blue, 2: red, 3: green, 4: yellow, 5: grey, 6: pink, 7: orange, 8: light blue, 9: brown
    rich_colors = {
        0: "black",
        1: "blue",
        2: "red",
        3: "green",
        4: "yellow",
        5: "bright_black", # grey
        6: "magenta",
        7: "rgb(255,135,30)", # orange
        8: "cyan",
        9: "rgb(141,29,44)", # brown
    }

    output = []
    output.append("Working Grid:")
    
    for r in range(height):
        row_str = ""
        for c in range(width):
            val = int(grid[r, c])
            color = rich_colors.get(val, "white")
            # Use two spaces for a roughly square aspect ratio in terminal
            row_str += f"[{color}]██[/{color}]" 
        output.append(row_str)
    
    markup_str = "\n".join(output)
    
    # Render to ANSI string using Rich
    # We use a separate console to capture the output as a string with ANSI codes
    if TYPE_CHECKING:
        assert Console is not None

    console = Console(force_terminal=True, color_system="truecolor", width=1000)
    with console.capture() as capture:
        console.print(markup_str)
    
    return capture.get()


# ============================================================================
# SVG Rendering
# ============================================================================


def render_svg(state: State) -> str:
    """
    Render the current state as an SVG string.

    Args:
        state: The current environment state.

    Returns:
        A string containing the SVG XML.
    """
    # Use the existing draw_grid_svg function from core.py
    # We need to pass the grid and mask
    
    # We can render the working grid
    drawing = draw_grid_svg(
        grid_input=state.working_grid,
        mask=state.working_grid_mask,
        label=f"Step {int(state.step_count)}",
        show_size=True
    )
    
    # Ensure we have a Drawing object
    if isinstance(drawing, tuple):
        # This case should not happen with default arguments, but for type safety:
        drawing = cast(drawsvg.Drawing, drawing[0])
    else:
        drawing = cast(drawsvg.Drawing, drawing)
    
    svg_str = drawing.as_svg()
    return svg_str if svg_str is not None else ""

