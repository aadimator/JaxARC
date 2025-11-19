"""
Visualization core utilities for JaxARC.

Contains constants, utilities, and SVG rendering engine.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import drawsvg  # type: ignore[import-untyped]
import jax.numpy as jnp
import numpy as np
from loguru import logger

from jaxarc.utils.serialization_utils import serialize_jax_array

if TYPE_CHECKING:
    from jaxarc.types import Grid, GridArray


# ============================================================================
# SECTION: Constants and Color Definitions (from constants.py)
# ============================================================================

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


# ============================================================================
# SECTION: Utility Functions (from utils.py)
# ============================================================================


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

    # Check for Grid type by duck typing (more robust than isinstance)
    if hasattr(grid_input, "data") and hasattr(grid_input, "mask"):
        return serialize_jax_array(grid_input.data), serialize_jax_array(
            grid_input.mask
        )
    if isinstance(grid_input, (jnp.ndarray, np.ndarray)):
        return serialize_jax_array(grid_input), None

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
    before_data = serialize_jax_array(before_grid.data)
    after_data = serialize_jax_array(after_grid.data)

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
        before_data = serialize_jax_array(before_grid.data)
        after_data = serialize_jax_array(after_grid.data)

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


def get_info_metric(info: dict, key: str, default=None):
    """Extract metric value from info dict, supporting both old and new structures.

    This function handles the transition from storing metrics directly in the info
    dictionary to storing them nested under info['metrics']. It prioritizes the
    nested structure when both are present.

    Args:
        info: Info dictionary from environment step
        key: Metric key to extract
        default: Default value if metric not found

    Returns:
        Metric value, converted to appropriate type
    """
    # First check if it's in info.metrics (higher priority - new format)
    if "metrics" in info and key in info["metrics"]:
        val = info["metrics"][key]
        return float(val) if hasattr(val, "item") else val

    # Then check if it's directly in info (lower priority - old format)
    if key in info:
        val = info[key]
        return float(val) if hasattr(val, "item") else val

    return default


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 8:  # Handle RGBA (ignore alpha for now)
        hex_color = hex_color[:6]
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


def render_rgb(
    grid_input: GridArray | np.ndarray | Grid, cell_size: int = 20
) -> np.ndarray:
    """Render grid as an RGB array.

    Args:
        grid_input: Grid data (JAX array, numpy array, or Grid object)
        cell_size: The size of each grid cell in pixels.

    Returns:
        A numpy array of shape (height * cell_size, width * cell_size, 3) representing the RGB image.
    """
    grid, _ = _extract_grid_data(grid_input)
    height, width = grid.shape

    # Create empty RGB array
    img_height = height * cell_size
    img_width = width * cell_size
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Precompute color map
    color_map = {k: hex_to_rgb(v) for k, v in ARC_COLOR_PALETTE.items()}
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


def _clear_output_directory(output_dir: str) -> None:
    """Clear output directory for new episode."""
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SECTION: SVG Core (from svg_core.py)
# ============================================================================


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
