# Implementation Plan

- [x] 1. Audit existing utilities and prepare for reuse

  - Search for existing serialization functionality in
    utils/serialization_utils.py, utils/pytree_utils.py
  - Identify reusable JAX array handling in utils/equinox_utils.py and
    utils/jax_types.py
  - Locate existing file path utilities in utils/ that can be leveraged
  - Document available utilities to avoid reimplementation
  - _Requirements: 1.1, 2.2, 7.2_

- [x] 2. Create ExperimentLogger central coordinator

  - Create src/jaxarc/utils/logging/experiment_logger.py with ExperimentLogger
    as regular Python class (NOT equinox.Module)
  - Implement handler initialization - all handlers operate outside JAX boundary
    as regular Python classes
  - Add log_step(), log_episode_summary(), and close() methods with error
    isolation
  - Use existing configuration utilities from utils/config.py for handler setup
  - Write unit tests for handler coordination and error isolation
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 3. Refactor StructuredLogger into FileHandler

  - Rename and simplify src/jaxarc/utils/logging/structured_logger.py to
    file_handler.py as regular Python class
  - Remove async logging dependencies and convert to synchronous file writing
    (can use standard file I/O)
  - Reuse existing serialization utilities from utils/serialization_utils.py for
    JAX arrays
  - Implement JSON and pickle saving using existing pytree utilities where
    applicable
  - Write unit tests for synchronous file operations and JAX array serialization
  - _Requirements: 1.1, 3.1, 7.2_

- [x] 4. Create SVGHandler for visualization generation

  - Create src/jaxarc/utils/logging/svg_handler.py with SVGHandler as regular
    Python class (can use string manipulation, file I/O freely)
  - Move core SVG functions from rl_visualization.py and
    episode_visualization.py into handler methods
  - Integrate with existing EpisodeManager for file path management
  - Preserve get_operation_display_name and overlay functions from existing code
  - Use existing grid utilities from utils/grid_utils.py for grid operations
  - Write unit tests for SVG generation and file saving
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 5. Create RichHandler for console output

  - Create src/jaxarc/utils/logging/rich_handler.py as regular Python class (can
    use Rich library, console I/O freely)
  - Move console output logic from visualization/rich_display.py into handler
    methods
  - Reuse existing Rich console setup and display functions from rich_display.py
  - Implement log_step() and log_episode_summary() methods using existing
    display functions
  - Write unit tests for console output formatting
  - _Requirements: 3.3, 6.4_

- [x] 6. Create simplified WandbHandler

  - Create src/jaxarc/utils/logging/wandb_handler.py as regular Python class
    (can use wandb library, network requests freely)
  - Remove custom retry logic, offline caching, and network connectivity checks
    from existing wandb.py
  - Implement simple wandb.init() and wandb.log() wrapper using official wandb
    features
  - Add info['metrics'] extraction logic for automatic metric logging
  - Write unit tests with mocked wandb API calls
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 7. Update ArcEnvironment integration

  - Modify src/jaxarc/envs/environment.py to initialize ExperimentLogger instead
    of Visualizer
  - Update step() method to call logger.log_step() through existing JAX callback
    mechanism
  - Add logger.log_episode_summary() calls for episode completion
  - Add logger.close() call in environment cleanup
  - Ensure JAX callback integration remains unchanged for compatibility
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 8. Update JAX callback integration

  - Modify src/jaxarc/utils/visualization/jax_callbacks.py to work with new
    ExperimentLogger
  - Ensure jax_save_step_visualization callback uses new logger interface
  - Verify JAX transformations (jit, vmap, pmap) continue working correctly
  - Use existing JAX utilities from utils/jax_types.py for type handling
  - Write integration tests for JAX compatibility
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 9. Establish info dictionary conventions

  - Update functional.py to structure info dictionary with info['metrics'] for
    scalar data
  - Modify handlers to extract metrics from info['metrics'] automatically
  - Ensure FileHandler serializes entire info dictionary using existing
    serialization utils
  - Make SVGHandler and RichHandler ignore unknown keys gracefully
  - Write tests for info dictionary handling and metric extraction
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 9. Remove obsolete components

  - Delete src/jaxarc/utils/visualization/visualizer.py (complex orchestrator)
  - Delete src/jaxarc/utils/visualization/async_logger.py (async complexity)
  - Delete src/jaxarc/utils/logging/performance_monitor.py (premature
    optimization)
  - Delete src/jaxarc/utils/visualization/memory_manager.py (overengineering)
  - Delete src/jaxarc/utils/visualization/wandb_sync.py (custom sync logic)
  - Update imports throughout codebase to remove references to deleted files
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 10. Update configuration integration

  - Verify existing Hydra debug configurations work with new ExperimentLogger
  - Test that debug.level="off" properly disables logging
  - Ensure wandb configuration structure remains compatible
  - Add any necessary configuration validation using existing config utilities
  - Write tests for configuration compatibility and edge cases
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 11. Create comprehensive integration tests

  - Write end-to-end test for complete logging pipeline with all handlers
  - Test handler error isolation (one handler failing doesn't crash others)
  - Verify JAX performance impact remains minimal with new architecture
  - Test configuration-driven handler selection and initialization
  - Create test for migration from old to new system
  - _Requirements: 2.5, 7.5, 8.5_

- [x] 12. Update imports and clean up codebase
  - Update **init**.py files to export new ExperimentLogger and handlers
  - Remove imports of deleted components throughout the codebase
  - Update any remaining references to old Visualizer class
  - Ensure all new components use existing utilities instead of reimplementing
    functionality
  - Run full test suite to verify no broken imports or missing dependencies
  - _Requirements: 9.5, 10.5_
