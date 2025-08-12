# Implementation Plan

- [x] 1. Update parser configuration patterns to use typed configs

  - Update MiniArcParser to accept DatasetConfig instead of raw DictConfig
  - Update ArcAgiParser to accept DatasetConfig instead of raw DictConfig
  - Update ConceptArcParser to accept DatasetConfig instead of raw DictConfig
  - Add from_hydra class methods to all parsers for backward compatibility
  - Update usage in notebooks/miniarc_rl_loop.py to use typed configs
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 2. Decompose long functions in functional.py for better maintainability

  - Extract \_get_or_create_task_data helper function from arc_reset (20-30
    lines)
  - Extract \_select_initial_pair helper function from arc_reset (20-30 lines)
  - Extract \_initialize_grids helper function from arc_reset (20-30 lines)
  - Extract \_create_initial_state helper function from arc_reset (20-30 lines)
  - Extract \_process_action helper function from arc_step (30-40 lines)
  - Extract \_update_state helper function from arc_step (30-40 lines)
  - Extract \_calculate_reward_and_done helper function from arc_step (20-30
    lines)
  - Ensure all extracted functions maintain JAX compliance and are
    JIT-compatible
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 3. Consolidate visualization system and remove "Enhanced" naming

  - Rename enhanced_visualizer.py to visualizer.py
  - Rename EnhancedVisualizer class to Visualizer class
  - Update all imports across codebase from EnhancedVisualizer to Visualizer
  - Remove any basic/unenhanced visualization classes if they exist
  - Update **init**.py files to export Visualizer instead of EnhancedVisualizer
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 4. Add task visualization at episode start

  - Create TaskVisualizationData dataclass for task information
  - Add start_episode_with_task method to Visualizer class
  - Implement \_create_task_visualization method using existing SVG functions
  - Add task visualization as first step in episode visualization flow
  - Test task visualization generation with sample tasks
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 5. Enhance step visualizations with task context

  - Add task_id field to StepVisualizationData dataclass
  - Add task_pair_index field to StepVisualizationData dataclass
  - Add total_task_pairs field to StepVisualizationData dataclass
  - Update step visualization generation to include task context in display
  - Update notebooks/miniarc_rl_loop.py to provide task context in step data
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 6. Reorganize visualization module structure for clarity

  - Move wandb_integration.py to integrations/wandb.py
  - Consolidate multiple visualization config modules into single config.py
  - Update import statements across codebase for new structure
  - Update **init**.py files to reflect new organization
  - Remove duplicate or unused visualization configuration classes
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 7. Update example scripts and notebooks to use new patterns

  - Update notebooks/miniarc_rl_loop.py to use typed parser configs
  - Update notebooks/miniarc_rl_loop.py to use Visualizer instead of
    EnhancedVisualizer
  - Update notebooks/miniarc_rl_loop.py to include task visualization at episode
    start
  - Update any other example scripts to follow new configuration patterns
  - Test all updated examples to ensure they work correctly
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 8. Add comprehensive tests for refactored components

  - Write tests for parser initialization with typed configs
  - Write tests for from_hydra methods on all parsers
  - Write tests for decomposed functions maintain same behavior as originals
  - Write tests for JAX compliance of all extracted helper functions
  - Write tests for task visualization generation
  - Write tests for step visualization with task context
  - Write tests for Visualizer class functionality
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 9. Update documentation and type hints

  - Update docstrings for all modified parser classes
  - Update docstrings for all decomposed functions
  - Update docstrings for Visualizer class and new methods
  - Add type hints for all new dataclasses and methods
  - Update README or documentation to reflect new usage patterns
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 10. Performance validation and cleanup
  - Run performance tests to ensure refactoring doesn't impact speed
  - Validate JAX JIT compilation works with all decomposed functions
  - Remove any unused imports or dead code introduced during refactoring
  - Run full test suite to ensure no regressions
  - Update any remaining references to old class/function names
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
