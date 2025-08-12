# Requirements Document

## Introduction

This feature simplifies the JaxARC logging and visualization system by removing
overengineered components while preserving the valuable SVG debugging
capabilities. The current system has grown complex with async logging, custom
wandb sync, performance monitors, and memory managers that add unnecessary
complexity for a research project. The goal is to create a simple, maintainable
logging architecture centered around a single ExperimentLogger class with
focused handlers.

## Requirements

### Requirement 1: Remove Overengineered Components

**User Story:** As a developer, I want to remove unnecessary complexity from the
logging system, so that the codebase is easier to understand and maintain for
research purposes.

#### Acceptance Criteria

1. WHEN removing async logging THEN the system SHALL delete async_logger.py and
   switch to synchronous file writing
2. WHEN removing custom wandb sync THEN the system SHALL delete wandb_sync.py
   and rely on official wandb offline capabilities
3. WHEN removing performance abstractions THEN the system SHALL delete
   performance_monitor.py and memory_manager.py
4. WHEN simplifying wandb integration THEN the system SHALL remove custom retry
   logic and offline caching in favor of native wandb features
5. IF performance becomes an issue THEN standard profiling tools SHALL be used
   instead of custom monitoring

### Requirement 2: Create Centralized ExperimentLogger Architecture

**User Story:** As a developer, I want a single entry point for all logging
operations, so that the logging system is easy to understand and use.

#### Acceptance Criteria

1. WHEN initializing logging THEN the system SHALL create a single
   ExperimentLogger instance in ArcEnvironment
2. WHEN logging data THEN the ExperimentLogger SHALL manage a set of simple,
   single-purpose handlers
3. WHEN configuring logging THEN the ExperimentLogger SHALL initialize handlers
   based on configuration settings
4. WHEN calling logging methods THEN the ExperimentLogger SHALL provide simple
   public methods: log_step(), log_episode_summary(), close()
5. IF a handler fails THEN the ExperimentLogger SHALL continue with other
   handlers without crashing

### Requirement 3: Implement Focused Handler Architecture

**User Story:** As a developer, I want separate handlers for different logging
concerns, so that each component has a single responsibility and is easy to
maintain.

#### Acceptance Criteria

1. WHEN handling file logging THEN the FileHandler SHALL write episode data to
   JSON/pickle files synchronously
2. WHEN handling SVG generation THEN the SVGHandler SHALL contain core SVG logic
   from rl_visualization.py and episode_visualization.py
3. WHEN handling console output THEN the RichHandler SHALL use existing
   rich_display.py logic for terminal printing
4. WHEN handling wandb integration THEN the WandbHandler SHALL be a thin wrapper
   around official wandb library
5. IF a handler is disabled in config THEN it SHALL not be initialized or called

### Requirement 4: Preserve SVG Debugging Capabilities

**User Story:** As a researcher, I want to maintain the powerful SVG
visualization features, so that I can continue to debug and analyze agent
behavior effectively.

#### Acceptance Criteria

1. WHEN generating step visualizations THEN the system SHALL preserve
   draw_rl_step_svg_enhanced functionality
2. WHEN creating episode summaries THEN the system SHALL preserve
   draw_enhanced_episode_summary_svg functionality
3. WHEN displaying operations THEN the system SHALL maintain
   get_operation_display_name and overlay functions
4. WHEN saving SVG files THEN the system SHALL use EpisodeManager for correct
   file paths
5. IF SVG generation fails THEN the system SHALL log the error and continue
   without crashing

### Requirement 5: Simplify Wandb Integration

**User Story:** As a researcher, I want simple wandb integration that relies on
official features, so that I don't have to maintain custom sync and retry logic.

#### Acceptance Criteria

1. WHEN using wandb offline THEN the system SHALL set WANDB_MODE=offline
   environment variable instead of custom caching
2. WHEN logging to wandb THEN the system SHALL use simple wandb.init() and
   wandb.log() calls
3. WHEN handling wandb errors THEN the system SHALL rely on wandb's built-in
   error handling
4. WHEN syncing offline data THEN users SHALL use the official wandb sync
   command
5. IF wandb is unavailable THEN the system SHALL gracefully skip wandb logging
   without custom fallback logic

### Requirement 6: Establish Info Dictionary Convention

**User Story:** As a developer, I want clear conventions for passing custom data
through the logging system, so that new metrics can be added without code
changes.

#### Acceptance Criteria

1. WHEN passing custom metrics THEN the system SHALL use info['metrics']
   dictionary for scalar time-series data
2. WHEN adding new metrics THEN they SHALL be automatically logged to wandb if
   placed in info['metrics']
3. WHEN handling unknown data THEN FileHandler SHALL serialize the entire info
   dictionary automatically
4. WHEN processing visualization data THEN SVGHandler and RichHandler SHALL
   ignore unknown keys gracefully
5. IF new visualization metrics are needed THEN developers SHALL consciously
   edit the relevant handler

### Requirement 7: Maintain JAX Compatibility

**User Story:** As a developer, I want the simplified logging system to remain
JAX-compatible, so that performance and transformations are not affected.

#### Acceptance Criteria

1. WHEN using JAX callbacks THEN the system SHALL continue using
   jax.debug.callback for logging during transformations
2. WHEN serializing data THEN the system SHALL handle JAX arrays properly in the
   simplified handlers
3. WHEN logging from JIT functions THEN the system SHALL not interfere with JAX
   transformations
4. WHEN processing batched data THEN the system SHALL work correctly with
   jax.vmap and jax.pmap
5. IF JAX compatibility issues arise THEN they SHALL be resolved without adding
   complexity back

### Requirement 8: Update Environment Integration

**User Story:** As a developer, I want the ArcEnvironment to use the new
simplified logging system, so that the integration is clean and straightforward.

#### Acceptance Criteria

1. WHEN initializing ArcEnvironment THEN it SHALL create ExperimentLogger
   instead of Visualizer
2. WHEN stepping through episodes THEN the environment SHALL call
   logger.log_step() through JAX callbacks
3. WHEN completing episodes THEN the environment SHALL call
   logger.log_episode_summary()
4. WHEN cleaning up THEN the environment SHALL call logger.close() to properly
   shut down handlers
5. IF logging is disabled THEN the environment SHALL skip logging calls without
   performance impact

### Requirement 9: Remove Obsolete Components

**User Story:** As a developer, I want obsolete files and components removed
from the codebase, so that there's no confusion about which components to use.

#### Acceptance Criteria

1. WHEN refactoring is complete THEN visualizer.py SHALL be deleted as its
   orchestration role is obsolete
2. WHEN async logging is removed THEN async_logger.py SHALL be deleted
3. WHEN performance monitoring is removed THEN performance_monitor.py SHALL be
   deleted
4. WHEN memory management is removed THEN memory_manager.py SHALL be deleted
5. WHEN custom wandb sync is removed THEN wandb_sync.py SHALL be deleted

### Requirement 10: Maintain Configuration Compatibility

**User Story:** As a user, I want existing Hydra configurations to continue
working, so that I don't need to update all my experiment configs.

#### Acceptance Criteria

1. WHEN using existing debug configs THEN they SHALL continue to work with the
   simplified system
2. WHEN setting visualization options THEN existing config structure SHALL be
   preserved where possible
3. WHEN disabling logging THEN existing debug.level="off" SHALL work as expected
4. WHEN configuring wandb THEN existing wandb config structure SHALL be
   maintained
5. IF config changes are needed THEN they SHALL be minimal and
   backward-compatible where possible
