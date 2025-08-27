"""RL-specific visualization functions for JaxARC.

This module provides functions for visualizing reinforcement learning steps,
actions, and related RL-specific data structures.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
from loguru import logger

from jaxarc.envs.actions import BboxAction, MaskAction, PointAction
from jaxarc.envs.grid_operations import get_operation_display_text

from .constants import ARC_COLOR_PALETTE
from .svg_core import add_change_highlighting, add_selection_visualization_overlay
from .utils import (
    _extract_grid_data,
    _extract_valid_region,
    get_info_metric,
)

if TYPE_CHECKING:
    from jaxarc.types import Grid


def get_operation_display_name(
    operation_id: int, action_data: Dict[str, Any] = None
) -> str:
    """Get human-readable operation name from operation ID with context."""
    return get_operation_display_text(operation_id)


def draw_rl_step_svg_enhanced(
    before_grid: Grid,
    after_grid: Grid,
    action: Any,  # Can be PointAction, BboxAction, MaskAction, or dict
    reward: float,
    info: Dict[str, Any],
    step_num: int,
    operation_name: str = "",
    changed_cells: Optional[jnp.ndarray] = None,
    config: Optional[Any] = None,
    max_width: float = 1400.0,
    max_height: float = 700.0,
    task_id: str = "",
    task_pair_index: int = 0,
    total_task_pairs: int = 1,
) -> str:
    """Generate enhanced SVG visualization of a single RL step with more information.

    This enhanced version shows:
    - Before and after grids with improved styling
    - Action selection highlighting
    - Changed cell highlighting
    - Reward information and metrics
    - Operation name and details
    - Step metadata
    - Task context information

    Args:
        before_grid: Grid state before the action
        after_grid: Grid state after the action
        action: Action object or dictionary
        reward: Reward received for this step
        info: Additional information dictionary or StepInfo object
        step_num: Step number in the episode
        operation_name: Human-readable operation name
        changed_cells: Optional mask of cells that changed
        config: Optional visualization configuration
        max_width: Maximum width of the entire visualization
        max_height: Maximum height of the entire visualization
        task_id: Task identifier for context
        task_pair_index: Current task pair index
        total_task_pairs: Total number of task pairs

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

    # Add task context information
    task_context_text = ""
    if task_id:
        task_context_text = f"Task: {task_id}"
    if total_task_pairs > 1:
        if task_context_text:
            task_context_text += f" | Pair {task_pair_index + 1}/{total_task_pairs}"
        else:
            task_context_text = f"Pair {task_pair_index + 1}/{total_task_pairs}"

    if task_context_text:
        drawing.append(
            draw.Text(
                task_context_text,
                font_size=16,
                x=total_width / 2,
                y=65,
                text_anchor="middle",
                font_family="Anuphan",
                font_weight="400",
                fill="#6c757d",
            )
        )

    # Add reward information (adjusted position for task context)
    reward_color = "#27ae60" if reward > 0 else "#e74c3c" if reward < 0 else "#95a5a6"
    reward_text = f"Reward: {reward:.3f}"
    reward_y = 85 if task_context_text else 70
    drawing.append(
        draw.Text(
            reward_text,
            font_size=20,
            x=total_width / 2,
            y=reward_y,
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
            add_change_highlighting(
                drawing,
                changed_cells,
                grid_x,
                grid_y,
                cell_size,
                start_row,
                start_col,
                height,
                width,
            )

        # Add selection overlay if provided
        if selection_mask is not None and selection_mask.any():
            add_selection_visualization_overlay(
                drawing,
                selection_mask,
                grid_x,
                grid_y,
                cell_size,
                start_row,
                start_col,
                height,
                width,
                selection_color="#00FFFF",  # Bright cyan - very visible
                selection_opacity=0.4,
                border_width=3,
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
    if isinstance(action, MaskAction):
        selection_mask = np.asarray(action.selection)
    elif isinstance(action, (PointAction, BboxAction)):
        # For Point and Bbox actions, we need to generate the mask
        grid_data, _ = _extract_grid_data(before_grid)
        grid_shape = grid_data.shape
        selection_mask = np.asarray(action.to_selection_mask(grid_shape))
    elif isinstance(action, dict):  # Fallback for old dictionary format
        if "selection" in action:
            selection_mask = np.asarray(action["selection"])

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

    # Add step metadata with support for both old and new info structure
    similarity_val = get_info_metric(info, "similarity")
    if similarity_val is not None:
        info_items.append(f"Similarity: {similarity_val:.3f}")

    # Check for episode_reward or total_reward
    episode_reward_val = get_info_metric(info, "episode_reward") or get_info_metric(
        info, "total_reward"
    )
    if episode_reward_val is not None:
        info_items.append(f"Episode Reward: {episode_reward_val:.3f}")

    step_count_val = get_info_metric(info, "step_count")
    if step_count_val is not None:
        info_items.append(f"Total Steps: {int(step_count_val)}")

    # Add action details
    # Handle both structured actions and legacy dictionary format for visualization
    if hasattr(action, "operation"):
        op_val = (
            int(action.operation)
            if hasattr(action.operation, "item")
            else action.operation
        )
        info_items.append(f"Operation ID: {op_val}")
    elif "operation" in action:
        op_val = (
            int(action["operation"])
            if hasattr(action["operation"], "item")
            else action["operation"]
        )
        info_items.append(
            f"Operation ID: {op_val}"
        )  # Legacy format for visualization only

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


def save_rl_step_visualization(
    state: Any,  # ArcEnvState
    action: dict,
    next_state: Any,  # ArcEnvState
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

    from jaxarc.types import Grid

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
    # Note: This handles both structured actions and legacy dictionary format for visualization
    if hasattr(action, "operation"):
        operation_id = int(action.operation)
    else:
        operation_id = int(action["operation"])  # Legacy format for visualization only
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
    logger.info(f"Saved RL step visualization: {filepath}")


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

    # Additional info - check both direct and nested metrics
    similarity_val = get_info_metric(info, "similarity")

    if similarity_val is not None:
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
