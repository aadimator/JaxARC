"""Enhanced visualization and logging system for JaxARC.

This module provides comprehensive visualization capabilities for ARC grids, tasks,
and RL training episodes with support for multiple output formats and performance optimization.

Public API:
    Core visualization functions:
        - log_grid_to_console: Console logging with Rich formatting
        - draw_grid_svg: SVG generation for single grids
        - visualize_grid_rich: Rich table visualization for grids
        - visualize_task_pair_rich: Rich visualization for input-output pairs
        - draw_task_pair_svg: SVG generation for task pairs
        - visualize_parsed_task_data_rich: Complete task visualization
        - draw_parsed_task_data_svg: SVG generation for complete tasks

    RL-specific functions:
        - draw_rl_step_svg: Visualization of RL step transitions
        - save_rl_step_visualization: Save step visualizations to disk

    Utility functions:
        - save_svg_drawing: Save SVG drawings to files

    Constants:
        - ARC_COLOR_PALETTE: Standard ARC color mapping
"""

from __future__ import annotations

# Import from new consolidated modules
from .core import (
    ARC_COLOR_PALETTE,
    _clear_output_directory,
    _extract_grid_data,
    draw_grid_svg,
    save_svg_drawing,
)
from .display import (
    create_episode_comparison_visualization,
    draw_episode_summary_svg,
    draw_parsed_task_data_svg,
    draw_task_pair_svg,
    log_grid_to_console,
    visualize_grid_rich,
    visualize_parsed_task_data_rich,
    visualize_task_pair_rich,
)
from .rl_display import (
    EpisodeConfig,
    EpisodeManager,
    draw_rl_step_svg,
    save_rl_step_visualization,
)

# Re-export all public functions for backward compatibility
__all__ = [
    "ARC_COLOR_PALETTE",
    "EpisodeConfig",
    "EpisodeManager",
    "_clear_output_directory",
    "_extract_grid_data",
    "create_episode_comparison_visualization",
    "draw_episode_summary_svg",
    "draw_grid_svg",
    "draw_parsed_task_data_svg",
    "draw_rl_step_svg",
    "draw_task_pair_svg",
    "log_grid_to_console",
    "save_rl_step_visualization",
    "save_svg_drawing",
    "visualize_grid_rich",
    "visualize_parsed_task_data_rich",
    "visualize_task_pair_rich",
]
