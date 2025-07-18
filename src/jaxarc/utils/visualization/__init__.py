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
        - setup_matplotlib_style: Configure matplotlib styling
        
    Constants:
        - ARC_COLOR_PALETTE: Standard ARC color mapping
"""

from __future__ import annotations

# Import core visualization functions from the core module
from .core import (
    # Core grid visualization
    log_grid_to_console,
    draw_grid_svg,
    visualize_grid_rich,
    
    # Task pair visualization
    visualize_task_pair_rich,
    draw_task_pair_svg,
    
    # Complete task visualization
    visualize_parsed_task_data_rich,
    draw_parsed_task_data_svg,
    
    # RL-specific visualization
    draw_rl_step_svg,
    save_rl_step_visualization,
    
    # Utility functions
    save_svg_drawing,
    setup_matplotlib_style,
    
    # Constants
    ARC_COLOR_PALETTE,
    
    # Internal functions that are used by other modules
    _clear_output_directory,
    _extract_grid_data,
)

# Import episode management functionality
from .episode_manager import (
    EpisodeConfig,
    EpisodeManager,
)

# Re-export all public functions for backward compatibility
__all__ = [
    # Core visualization functions
    "log_grid_to_console",
    "draw_grid_svg", 
    "visualize_grid_rich",
    
    # Task visualization functions
    "visualize_task_pair_rich",
    "draw_task_pair_svg",
    "visualize_parsed_task_data_rich", 
    "draw_parsed_task_data_svg",
    
    # RL visualization functions
    "draw_rl_step_svg",
    "save_rl_step_visualization",
    
    # Utility functions
    "save_svg_drawing",
    "setup_matplotlib_style",
    
    # Constants
    "ARC_COLOR_PALETTE",
    
    # Episode management
    "EpisodeConfig",
    "EpisodeManager",
    
    # Internal functions (for backward compatibility)
    "_clear_output_directory",
    "_extract_grid_data",
]