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

# Import async logging functionality
from .async_logger import (
    AsyncLogger,
    AsyncLoggerConfig,
    LogEntry,
    AsyncLoggerContext,
)

# Import wandb integration functionality
from .wandb_integration import (
    WandbConfig,
    WandbIntegration,
    create_wandb_config,
    create_research_wandb_config,
    create_development_wandb_config,
)

# Import wandb sync utilities
from .wandb_sync import (
    WandbSyncManager,
    create_sync_manager,
    sync_offline_wandb_data,
    check_wandb_status,
)

# Import enhanced visualization system
from .enhanced_visualizer import (
    VisualizationConfig,
    EnhancedVisualizer,
    StepVisualizationData,
    EpisodeSummaryData,
)

# Import configuration management utilities
from .config_validation import (
    ConfigValidator,
    ValidationError,
    validate_config,
    format_validation_errors,
    validate_and_raise,
)

from .config_composition import (
    ConfigComposer,
    create_config_composer,
    quick_compose,
    get_config_help,
)

from .config_migration import (
    ConfigMigrator,
    migrate_legacy_config,
    check_config_compatibility,
    create_config_documentation,
)

# Import replay and analysis functionality
from .replay_system import (
    ReplayConfig,
    ReplayValidationResult,
    EpisodeReplaySystem,
)

from .analysis_tools import (
    AnalysisConfig,
    FailureModeAnalysis,
    PerformanceMetrics,
    EpisodeAnalysisTools,
)

# Import JAX integration and performance optimization
from .jax_callbacks import (
    jax_debug_callback,
    jax_log_grid,
    jax_save_step_visualization,
    jax_log_episode_summary,
    serialize_jax_array,
    serialize_arc_state,
    serialize_action,
    get_callback_performance_stats,
    reset_callback_performance_stats,
    print_callback_performance_report,
    CallbackPerformanceMonitor,
    JAXCallbackError,
)

from .memory_manager import (
    MemoryManager,
    MemoryUsageMonitor,
    LazyLoader,
    CompressedStorage,
    VisualizationCache,
    GarbageCollectionOptimizer,
    get_memory_manager,
    create_lazy_visualization_loader,
    optimize_array_memory,
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
    
    # Enhanced visualization system
    "VisualizationConfig",
    "EnhancedVisualizer",
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