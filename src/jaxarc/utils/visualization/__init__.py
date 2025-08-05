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

from .analysis_tools import (
    AnalysisConfig,
    EpisodeAnalysisTools,
    FailureModeAnalysis,
    PerformanceMetrics,
)

# Async logging functionality removed - use synchronous logging instead

# Import constants
from .constants import (
    ARC_COLOR_PALETTE,
)

# Import utility functions
from .utils import (
    _clear_output_directory,
    _extract_grid_data,
)

# Import Rich display functions
from .rich_display import (
    log_grid_to_console,
    visualize_grid_rich,
    visualize_parsed_task_data_rich,
    visualize_task_pair_rich,
)

# Import SVG core functions
from .svg_core import (
    draw_grid_svg,
    save_svg_drawing,
)

# Import task visualization functions
from .task_visualization import (
    draw_parsed_task_data_svg,
    draw_task_pair_svg,
)

# Import RL visualization functions
from .rl_visualization import (
    draw_rl_step_svg,
    save_rl_step_visualization,
)

# Import episode visualization functions
from .episode_visualization import (
    draw_episode_summary_svg,
    create_episode_comparison_visualization,
)

# Import matplotlib utilities
from .matplotlib_utils import (
    setup_matplotlib_style,
)

# Import episode management functionality
from .episode_manager import (
    EpisodeConfig,
    EpisodeManager,
)

# Import wandb integration functionality from integrations
from .integrations.wandb import (
    WandbConfig,
    WandbIntegration,
    create_development_wandb_config,
    create_research_wandb_config,
    create_wandb_config,
)

# Import JAX integration (simplified)
from .jax_callbacks import (
    create_grid_from_arrays,
    jax_debug_callback,
    jax_log_grid,
    log_grid_callback,
    safe_callback_wrapper,
)
# Memory management functionality removed - use standard Python memory management

# Import replay and analysis functionality
from .replay_system import (
    EpisodeReplaySystem,
    ReplayConfig,
    ReplayValidationResult,
)

# Complex visualizer removed - use ExperimentLogger with focused handlers instead

# Custom wandb sync removed - use official wandb sync command instead

# Re-export all public functions for backward compatibility
__all__ = [
    # Constants
    "ARC_COLOR_PALETTE",
    "AnalysisConfig",
    # Configuration management
    "EpisodeAnalysisTools",
    # Episode management
    "EpisodeConfig",
    "EpisodeManager",
    "EpisodeReplaySystem",
    "FailureModeAnalysis",
    "PerformanceMetrics",
    # Replay and analysis functionality
    "ReplayConfig",
    "ReplayValidationResult",
    # Wandb integration
    "WandbConfig",
    "WandbIntegration",
    # Internal functions (for backward compatibility)
    "_clear_output_directory",
    "_extract_grid_data",
    "create_development_wandb_config",
    "create_research_wandb_config",
    "create_wandb_config",
    "draw_grid_svg",
    "draw_parsed_task_data_svg",
    # RL visualization functions
    "draw_rl_step_svg",
    "draw_task_pair_svg",
    # JAX integration
    "jax_debug_callback",
    "jax_log_grid",
    # Core visualization functions
    "log_grid_to_console",
    "save_rl_step_visualization",
    # Utility functions
    "save_svg_drawing",
    "setup_matplotlib_style",
    "visualize_grid_rich",
    "visualize_parsed_task_data_rich",
    # Task visualization functions
    "visualize_task_pair_rich",
]
