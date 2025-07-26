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

# Import async logging functionality
from .async_logger import (
    AsyncLogger,
    AsyncLoggerConfig,
    AsyncLoggerContext,
    LogEntry,
)
from .config_composition import (
    ConfigComposer,
    create_config_composer,
    get_config_help,
    quick_compose,
)
from .config_migration import (
    ConfigMigrator,
    check_config_compatibility,
    create_config_documentation,
    migrate_legacy_config,
)

# Import configuration management utilities
from .config_validation import (
    ConfigValidator,
    ValidationError,
    format_validation_errors,
    validate_and_raise,
    validate_config,
)

# Import core visualization functions from the core module
from .core import (
    # Constants
    ARC_COLOR_PALETTE,
    # Internal functions that are used by other modules
    _clear_output_directory,
    _extract_grid_data,
    draw_grid_svg,
    draw_parsed_task_data_svg,
    # RL-specific visualization
    draw_rl_step_svg,
    draw_task_pair_svg,
    # Core grid visualization
    log_grid_to_console,
    save_rl_step_visualization,
    # Utility functions
    save_svg_drawing,
    setup_matplotlib_style,
    visualize_grid_rich,
    # Complete task visualization
    visualize_parsed_task_data_rich,
    # Task pair visualization
    visualize_task_pair_rich,
)

# Import visualization system
from .visualizer import (
    EpisodeSummaryData,
    StepVisualizationData,
    VisualizationConfig,
    Visualizer,
)

# Import episode management functionality
from .episode_manager import (
    EpisodeConfig,
    EpisodeManager,
)

# Import JAX integration and performance optimization
from .jax_callbacks import (
    CallbackPerformanceMonitor,
    JAXCallbackError,
    get_callback_performance_stats,
    jax_debug_callback,
    jax_log_episode_summary,
    jax_log_grid,
    jax_save_step_visualization,
    print_callback_performance_report,
    reset_callback_performance_stats,
    serialize_action,
    serialize_arc_state,
    serialize_jax_array,
)
from .memory_manager import (
    CompressedStorage,
    GarbageCollectionOptimizer,
    LazyLoader,
    MemoryManager,
    MemoryUsageMonitor,
    VisualizationCache,
    create_lazy_visualization_loader,
    get_memory_manager,
    optimize_array_memory,
)

# Import replay and analysis functionality
from .replay_system import (
    EpisodeReplaySystem,
    ReplayConfig,
    ReplayValidationResult,
)

# Import wandb integration functionality
from .wandb_integration import (
    WandbConfig,
    WandbIntegration,
    create_development_wandb_config,
    create_research_wandb_config,
    create_wandb_config,
)

# Import wandb sync utilities
from .wandb_sync import (
    WandbSyncManager,
    check_wandb_status,
    create_sync_manager,
    sync_offline_wandb_data,
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
    # Async logging
    "AsyncLogger",
    "AsyncLoggerConfig",
    "LogEntry",
    "AsyncLoggerContext",
    # Wandb integration
    "WandbConfig",
    "WandbIntegration",
    "create_wandb_config",
    "create_research_wandb_config",
    "create_development_wandb_config",
    # Wandb sync utilities
    "WandbSyncManager",
    "create_sync_manager",
    "sync_offline_wandb_data",
    "check_wandb_status",
    # Visualization system
    "VisualizationConfig",
    "Visualizer",
    "StepVisualizationData",
    "EpisodeSummaryData",
    # Configuration management
    "ConfigValidator",
    "ValidationError",
    "validate_config",
    "format_validation_errors",
    "validate_and_raise",
    "ConfigComposer",
    "create_config_composer",
    "quick_compose",
    "get_config_help",
    "ConfigMigrator",
    "migrate_legacy_config",
    "check_config_compatibility",
    "create_config_documentation",
    # Replay and analysis functionality
    "ReplayConfig",
    "ReplayValidationResult",
    "EpisodeReplaySystem",
    "AnalysisConfig",
    "FailureModeAnalysis",
    "PerformanceMetrics",
    "EpisodeAnalysisTools",
    # JAX integration and performance optimization
    "jax_debug_callback",
    "jax_log_grid",
    "jax_save_step_visualization",
    "jax_log_episode_summary",
    "serialize_jax_array",
    "serialize_arc_state",
    "serialize_action",
    "get_callback_performance_stats",
    "reset_callback_performance_stats",
    "print_callback_performance_report",
    "CallbackPerformanceMonitor",
    "JAXCallbackError",
    "MemoryManager",
    "MemoryUsageMonitor",
    "LazyLoader",
    "CompressedStorage",
    "VisualizationCache",
    "GarbageCollectionOptimizer",
    "get_memory_manager",
    "create_lazy_visualization_loader",
    "optimize_array_memory",
    # Internal functions (for backward compatibility)
    "_clear_output_directory",
    "_extract_grid_data",
]
