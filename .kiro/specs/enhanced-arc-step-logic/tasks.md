# Implementation Plan

- [x] 1. Extend Core State Management and Observation Space
  - [x] 1.1 Create new JAX-compatible type definitions for enhanced functionality
    - Define EpisodeMode (integer-based), AvailablePairs, CompletionStatus types in utils/jax_types.py
    - Define ActionHistory, HistoryLength, OperationMask types
    - Update NUM_OPERATIONS from 35 to 42 to include new control operations
    - Add proper JAXTyping annotations for all new types with JAX-compatible numeric types
    - Ensure all types use static shapes with appropriate padding
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 1.2 Add new fields to ArcEnvState for enhanced functionality
    - Add episode_mode field to track train/test mode
    - Add available_demo_pairs and available_test_pairs masks
    - Add completion status tracking for demonstration and test pairs
    - Add action_history and action_history_length fields
    - Add allowed_operations_mask for dynamic action space control
    - Update __check_init__ method to validate new fields
    - _Requirements: 1.1, 1.2, 3.1, 4.1, 6.1_

  - [x] 1.3 Create structured ArcObservation class for focused agent view
    - Define ArcObservation dataclass with core grids and essential context
    - Add episode_mode, current_pair_idx, step_count, and progress tracking
    - Add allowed_operations_mask for action space awareness
    - Add optional target_grid field (masked in test mode)
    - Add optional recent_actions field for configurable action history
    - Implement ObservationConfig for research flexibility
    - _Requirements: 1.1, 1.2, 1.3, 6.1_

  - [x] 1.4 Implement enhanced non-parametric action operations
    - Update ARCLEOperationType with new control operations (expanding from 35 to 42 total)
    - Implement non-parametric pair switching operations (next/prev/first_unsolved)
    - Add proper validation for updated operation range (0-41)
    - Ensure backward compatibility with existing action format
    - _Requirements: 4.1, 4.2, 6.1_

  - [x] 1.5 Update utility methods for enhanced state management
    - Add utility methods for accessing demonstration and test pair data
    - Add methods for checking pair availability and completion status
    - Ensure all methods maintain JAX compatibility
    - _Requirements: 1.3, 6.1, 6.4_

- [x] 2. Implement Episode Management System

  - [x] 2.1 Create ArcEpisodeManager class for pair selection and lifecycle management
    - Implement select_initial_pair method with configurable strategies
    - Implement should_continue_episode method with flexible termination criteria
    - Implement execute_pair_control_operation method for non-parametric pair switching
    - Add support for both sequential and random pair selection
    - Add context-aware pair switching (next/prev/first_unsolved)
    - Note: This is separate from the existing visualization EpisodeManager
    - _Requirements: 1.1, 1.2, 1.4, 1.5, 5.1, 5.2_

  - [x] 2.2 Create ArcEpisodeConfig dataclass for episode behavior configuration
    - Define configuration options for demonstration and test pair handling
    - Add termination criteria and pair switching configuration
    - Add reward frequency configuration for different modes
    - Implement comprehensive validation for episode configuration
    - Note: This is separate from the existing visualization EpisodeConfig
    - _Requirements: 5.3, 5.4, 7.1, 7.5_

  - [x] 2.3 Implement mode-specific episode initialization logic
    - Add logic for training mode initialization with demonstration pairs
    - Add logic for evaluation mode initialization with test pairs
    - Implement proper target grid masking for evaluation mode
    - Add support for explicit pair index specification
    - _Requirements: 2.1, 2.2, 2.3, 5.1_

- [x] 3. Implement Action History Tracking

  - [x] 3.1 Create ActionHistoryTracker class for managing action sequences
    - Implement add_action method with fixed-size array management
    - Implement get_action_sequence method for history retrieval
    - Implement clear_history method for episode resets
    - Add proper indexing and overflow handling for circular buffer
    - _Requirements: 3.1, 3.2, 3.4, 6.2_

  - [x] 3.2 Define ActionRecord structure for individual action storage
    - Create comprehensive action record with selection data and metadata
    - Add timestamp and pair index tracking for each action
    - Implement proper padding and masking for JAX compatibility
    - Add validation for action record integrity
    - _Requirements: 3.1, 3.2, 6.1, 6.2_

  - [x] 3.3 Create HistoryConfig dataclass for memory-efficient history tracking
    - Add configuration options for history length and storage format
    - Add options for storing selection data and intermediate grids (memory-intensive)
    - Implement validation for history configuration parameters
    - Add memory usage estimation methods for different configurations
    - _Requirements: 3.3, 6.4, 6.5, 7.2, 7.5_

- [x] 4. Implement Basic Action Space Control

  - [x] 4.1 Create ActionSpaceController class for context-aware operation management
    - Implement get_allowed_operations method with context-aware mask generation
    - Implement validate_operation method with context validation (mode, available pairs)
    - Implement filter_invalid_operation method for policy enforcement
    - Add support for context-dependent operation availability (demo/test switching)
    - _Requirements: 4.1, 4.2, 4.4_

  - [x] 4.2 Extend ActionConfig with basic dynamic control options
    - Add dynamic_action_filtering flag for runtime filtering
    - Add context_dependent_operations for basic context awareness
    - Implement validation for enhanced action configuration
    - Keep configuration simple and focused on core functionality
    - _Requirements: 4.1, 4.2, 7.3_

  - [x] 4.3 Implement operation validation and filtering logic
    - Add JAX-compatible operation mask application
    - Implement proper error handling for invalid operations
    - Add support for basic validation policies (reject, clip, penalize)
    - Ensure all validation logic is JIT-compilable
    - _Requirements: 4.2, 4.4, 6.3, 6.4_

- [x] 5. Update Core Environment Functions

  - [x] 5.1 Enhance arc_reset function with multi-demonstration support
    - Add episode_mode parameter for train/test mode selection
    - Add initial_pair_idx parameter for explicit pair selection
    - Implement proper pair selection based on configuration strategy
    - Add action history initialization and allowed operations setup
    - Return tuple of (state, observation)
    - Implement create_observation function for state-to-observation conversion
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 3.5, 5.1_

  - [x] 5.2 Enhance arc_step function with comprehensive functionality
    - Add support for enhanced non-parametric control operations
    - Add action history tracking for each step with memory optimization
    - Implement dynamic action space validation and filtering
    - Add mode-specific reward calculation logic
    - Implement non-parametric pair switching logic (next/prev/first_unsolved)
    - Return tuple of (new_state, agent_observation, reward, done, info)
    - Use create_observation function to generate focused agent view
    - _Requirements: 1.2, 1.3, 1.4, 2.4, 3.1, 4.2, 5.2_

  - [x] 5.3 Implement enhanced reward calculation with mode awareness
    - Add training mode reward calculation with configurable frequency
    - Add evaluation mode reward calculation with target masking
    - Implement proper similarity scoring for different pair types
    - Add support for different reward structures based on configuration
    - Ensure reward calculation is JIT-compilable and efficient
    - _Requirements: 2.2, 2.4, 5.4, 6.3, 7.4_

- [x] 6. Update Configuration System

  - [x] 6.1 Extend JaxArcConfig with new configuration sections
    - Add ArcEpisodeConfig integration to main configuration
    - Add HistoryConfig integration for action tracking
    - Add enhanced ActionConfig with dynamic control options
    - Ensure all new configurations validate properly
    - _Requirements: 7.1, 7.2, 7.3, 7.5_

  - [x] 6.28 Update Hydra configuration files with new options
    - Add episode management configuration options
    - Add action history configuration options
    - Add enhanced action space configuration options
    - Create example configurations for different use cases
    - _Requirements: 7.1, 7.2, 7.3, 7.5_

- [x] 7. Implement Comprehensive Testing

  - [x] 7.1 Create unit tests for enhanced state management and observations
    - Test new ArcEnvState fields and validation
    - Test ArcObservation construction and context information
    - Test create_observation function with different configurations
    - Test ObservationConfig options and observation formats
    - Test utility methods for pair access and status tracking
    - Test JAX compatibility of all new state and observation operations
    - Test state transitions with new fields
    - _Requirements: 1.1, 1.2, 1.3, 6.1, 6.2, 6.4_

  - [x] 7.2 Create unit tests for episode management system
    - Test ArcEpisodeManager pair selection strategies
    - Test episode termination criteria and continuation logic
    - Test mode switching between training and evaluation
    - Test configuration validation and error handling
    - Test pair switching control operations
    - _Requirements: 2.1, 2.2, 2.3, 5.1, 5.2, 5.3, 5.4_

  - [x] 7.3 Create unit tests for action history tracking with memory optimization
    - Test ActionHistoryTracker with various history lengths
    - Test action sequence storage and retrieval
    - Test circular buffer behavior and overflow handling
    - Test memory-efficient configuration options
    - Test JAX compatibility of history operations
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 6.2, 6.4_

  - [x] 7.4 Create unit tests for basic action space control
    - Test ActionSpaceController context-aware operation filtering
    - Test non-parametric control operations (pair switching)
    - Test operation validation and filtering policies
    - Test invalid operation handling and error cases
    - Test JAX compatibility of action space operations
    - _Requirements: 1.4, 4.1, 4.2, 4.4, 4.5_

  - [x] 7.5 Create integration tests for complete system
    - Test end-to-end multi-demonstration training workflows
    - Test evaluation mode with proper target masking
    - Test JAX transformations (jit, vmap, pmap) with all enhancements
    - Test performance impact and memory usage of new features
    - Test ArcObservation with RL agents and different observation configurations
    - _Requirements: 1.4, 1.6, 2.5, 6.4, 6.5_

  - [x] 7.6 Create performance and memory benchmarks
    - Benchmark memory usage with different HistoryConfig settings
    - Benchmark step latency with enhanced functionality
    - Test scalability with large batch sizes (1000+ environments)
    - Validate JIT compilation of all new functions
    - Compare performance against baseline implementation
    - _Requirements: 6.4, 6.5_

- [x] 8. Create Documentation and Examples

  - [x] 8.1 Update API documentation with new functionality
    - Document all new state fields and their purposes
    - Document episode management configuration options
    - Document action history tracking capabilities
    - Document enhanced action space control features
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 8.2 Create usage examples for new features
    - Create multi-demonstration training example
    - Create test pair evaluation example
    - Create action history analysis example
    - Create restricted action space example
    - _Requirements: 1.4, 2.5, 3.4, 4.4_

  - [x] 8.3 Create migration guide for existing users
    - Document backward compatibility guarantees
    - Provide examples of upgrading existing configurations
    - Document performance considerations for new features
    - Provide troubleshooting guide for common issues
    - _Requirements: 7.1, 7.2, 7.3, 7.5_