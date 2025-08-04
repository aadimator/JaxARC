# Requirements Document

## Introduction

This specification addresses critical JAX compatibility issues in the JaxARC codebase that prevent optimal performance and scalability. The analysis revealed that while the codebase has excellent architectural foundations with Equinox modules and proper type annotations, several fundamental issues prevent JIT compilation, vectorization, and efficient memory usage.

The primary goal is to transform JaxARC from a codebase with 0% JAX function compatibility to one with full JIT compilation, batch processing capabilities, optimized memory usage, and advanced Equinox features - achieving 200x+ performance improvements and 90% memory reduction.

This specification also incorporates advanced Equinox features including filtered transformations, PyTree manipulation utilities, serialization capabilities, runtime error handling, and performance optimizations.

## Requirements

### Requirement 1: Configuration System JAX Compatibility

**User Story:** As a researcher using JaxARC, I want the configuration system to be JAX-compatible so that core functions can be JIT compiled for optimal performance.

#### Acceptance Criteria

1. WHEN configuration objects are created THEN they SHALL be hashable and compatible with jax.jit static_argnames
2. WHEN configuration contains list or complex types THEN they SHALL be replaced with hashable alternatives (tuples, frozensets, or primitives)
3. WHEN jax.jit(static_argnames=['config']) is applied to arc_reset THEN it SHALL compile successfully without errors
4. WHEN jax.jit(static_argnames=['config']) is applied to arc_step THEN it SHALL compile successfully without errors
5. WHEN configuration objects are hashed THEN they SHALL not raise TypeError: unhashable type exceptions
6. WHEN configuration validation is performed THEN all existing validation logic SHALL continue to work correctly

### Requirement 2: Action System Vectorization Compatibility

**User Story:** As a researcher training RL agents, I want actions to be JAX-compatible data structures so that I can process multiple environments in parallel using jax.vmap.

#### Acceptance Criteria

1. WHEN actions are created THEN they SHALL use Equinox modules instead of Python dictionaries
2. WHEN actions contain selection data THEN they SHALL use JAX arrays with static shapes
3. WHEN actions contain operation IDs THEN they SHALL use JAX scalar arrays (jnp.int32)
4. WHEN jax.vmap is applied to action processing functions THEN they SHALL execute successfully
5. WHEN batch processing 32+ environments THEN the system SHALL handle it without errors
6. WHEN structured actions are used THEN they SHALL maintain backward compatibility with existing operation logic
7. WHEN action validation is performed THEN it SHALL work with the new structured format

### Requirement 3: Memory-Efficient Action History Storage

**User Story:** As a researcher working with limited computational resources, I want action history storage to be memory-efficient so that I can run more environments simultaneously.

#### Acceptance Criteria

1. WHEN using point-based actions THEN action history SHALL use only 6 fields per record (2 coordinates + 4 metadata)
2. WHEN using bbox-based actions THEN action history SHALL use only 8 fields per record (4 coordinates + 4 metadata)  
3. WHEN using mask-based actions THEN action history SHALL use format-specific field count (900 + 4 metadata)
4. WHEN action history is allocated THEN it SHALL be sized based on the configured selection format
5. WHEN switching between action formats THEN memory usage SHALL reflect the appropriate format requirements
6. WHEN point actions are used THEN memory usage SHALL be reduced by 99.3% compared to current implementation
7. WHEN bbox actions are used THEN memory usage SHALL be reduced by 99.1% compared to current implementation

### Requirement 4: JIT Compilation for Core Functions

**User Story:** As a researcher running large-scale experiments, I want all core JaxARC functions to be JIT compiled so that I can achieve maximum performance.

#### Acceptance Criteria

1. WHEN arc_reset is called THEN it SHALL be JIT compilable with static_argnames=['config']
2. WHEN arc_step is called THEN it SHALL be JIT compilable with static_argnames=['config']
3. WHEN execute_grid_operation is called THEN it SHALL remain JIT compilable (already working)
4. WHEN compute_grid_similarity is called THEN it SHALL remain JIT compilable (already working)
5. WHEN action handlers (point, bbox, mask) are called THEN they SHALL remain JIT compilable (already working)
6. WHEN JIT compiled functions are executed THEN they SHALL produce identical results to non-JIT versions
7. WHEN JIT compilation occurs THEN it SHALL complete without abstract array interpretation errors

### Requirement 5: Batch Processing with jax.vmap

**User Story:** As a researcher training agents on multiple environments, I want to process batches of environments in parallel so that I can achieve high throughput training.

#### Acceptance Criteria

1. WHEN jax.vmap is applied to arc_reset THEN it SHALL process multiple environments simultaneously
2. WHEN jax.vmap is applied to arc_step THEN it SHALL process multiple environment steps simultaneously
3. WHEN batch processing 32 environments THEN it SHALL complete successfully
4. WHEN batch processing 1000+ environments THEN it SHALL handle the load without memory errors
5. WHEN batch operations are performed THEN they SHALL maintain deterministic behavior with proper PRNG key splitting
6. WHEN batch processing is used THEN performance SHALL scale linearly with batch size
7. WHEN batch results are produced THEN they SHALL be equivalent to sequential processing of the same operations

### Requirement 6: Performance Optimization and Validation

**User Story:** As a researcher concerned with computational efficiency, I want to validate that JAX compatibility fixes deliver the expected performance improvements.

#### Acceptance Criteria

1. WHEN JIT compilation is enabled THEN step execution time SHALL be reduced by at least 100x compared to non-JIT
2. WHEN memory-efficient action history is used THEN memory usage per environment SHALL be reduced by at least 85%
3. WHEN batch processing is enabled THEN throughput SHALL achieve at least 10,000 steps/second
4. WHEN performance benchmarks are run THEN they SHALL demonstrate measurable improvements in all key metrics
5. WHEN memory profiling is performed THEN action history SHALL no longer dominate memory usage for point/bbox actions
6. WHEN batch processing scaling is tested THEN it SHALL maintain consistent per-environment performance up to 1000+ environments
7. WHEN performance regression tests are run THEN they SHALL pass with improved metrics

### Requirement 7: Advanced Equinox Features Integration

**User Story:** As a researcher using JaxARC, I want to leverage advanced Equinox features for better development experience, debugging, and performance optimization.

#### Acceptance Criteria

1. WHEN using filtered transformations THEN the system SHALL use equinox.filter_jit instead of jax.jit for automatic static/dynamic argument handling
2. WHEN performing PyTree manipulation THEN the system SHALL use equinox.tree_at for efficient out-of-place updates
3. WHEN runtime errors occur THEN the system SHALL use equinox.error_if for JAX-compatible error handling
4. WHEN model serialization is needed THEN the system SHALL use equinox.tree_serialise_leaves and equinox.tree_deserialise_leaves
5. WHEN debugging is required THEN the system SHALL support EQX_ON_ERROR environment variable for breakpoint debugging
6. WHEN filtering PyTrees THEN the system SHALL use equinox.filter and equinox.partition for efficient data manipulation
7. WHEN combining PyTrees THEN the system SHALL use equinox.combine for efficient reconstruction

### Requirement 8: Serialization and State Management

**User Story:** As a researcher running long experiments, I want efficient serialization and state management so that I can save and resume training seamlessly.

#### Acceptance Criteria

1. WHEN saving environment state THEN it SHALL use Equinox serialization with custom filter specs for arrays vs non-arrays
2. WHEN loading environment state THEN it SHALL use equinox.tree_deserialise_leaves with proper like templates
3. WHEN serializing configurations THEN it SHALL handle hashable config objects efficiently
4. WHEN saving action history THEN it SHALL use format-specific serialization to minimize file size
5. WHEN resuming from checkpoint THEN it SHALL use equinox.filter_eval_shape to avoid memory allocation during loading
6. WHEN managing hyperparameters THEN it SHALL combine JSON hyperparameters with binary model weights in single files
7. WHEN handling large states THEN it SHALL support streaming serialization for memory efficiency
8. WHEN serializing state THEN it SHALL exclude large static task_data and reconstruct it from task_index during loading
9. WHEN saving multiple checkpoints THEN it SHALL avoid redundant serialization of static task data

### Requirement 9: Runtime Error Handling and Debugging

**User Story:** As a researcher debugging complex RL experiments, I want robust error handling and debugging capabilities that work within JAX transformations.

#### Acceptance Criteria

1. WHEN invalid actions are detected THEN the system SHALL use equinox.error_if to raise runtime errors within JIT
2. WHEN grid operations fail THEN the system SHALL use equinox.branched_error_if for specific error messages
3. WHEN debugging is needed THEN the system SHALL support EQX_ON_ERROR=breakpoint for interactive debugging
4. WHEN NaN values are detected THEN the system SHALL support EQX_ON_ERROR=nan for graceful degradation
5. WHEN error conditions are checked THEN they SHALL not be eliminated by dead code optimization
6. WHEN batch processing errors occur THEN the system SHALL provide clear error messages with batch indices
7. WHEN runtime validation is performed THEN it SHALL work correctly under all JAX transformations

### Requirement 10: Testing and Validation Framework

**User Story:** As a developer maintaining JaxARC, I want comprehensive tests to ensure JAX compatibility works correctly and performance improvements are maintained.

#### Acceptance Criteria

1. WHEN JAX compatibility tests are run THEN they SHALL verify JIT compilation of all core functions
2. WHEN batch processing tests are run THEN they SHALL validate vmap functionality with various batch sizes
3. WHEN memory usage tests are run THEN they SHALL confirm memory efficiency improvements
4. WHEN performance benchmarks are executed THEN they SHALL measure and validate speed improvements
5. WHEN regression tests are run THEN they SHALL ensure backward compatibility is maintained
6. WHEN configuration validation tests are run THEN they SHALL verify hashability of all config objects
7. WHEN integration tests are run THEN they SHALL validate end-to-end functionality with JAX optimizations enabled