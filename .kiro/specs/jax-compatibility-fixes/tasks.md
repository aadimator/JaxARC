- [x] 1. Configuration System Hashability Fix

  - [x] 1.1 Replace unhashable types in existing configuration classes (Single
        Source of Truth)

    - Modify existing classes in `src/jaxarc/envs/config.py` directly - no new
      classes
    - Convert all `List[T]` to `tuple[T, ...]` in VisualizationConfig,
      WandbConfig, and other config classes
    - Replace complex type annotations (Int, Float) with primitive types (int,
      float)
    - Ensure all nested configuration objects are also hashable
    - Add hashability validation in `__post_init__` methods
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 1.2 Update configuration validation and factory functions

    - Modify `from_hydra` class methods to handle tuple conversions
    - Update validation logic to work with tuples instead of lists
    - Ensure backward compatibility in configuration loading
    - Test configuration hashability with `hash(config)` calls
    - _Requirements: 1.1, 1.6_

  - [x] 1.3 Test configuration hashability and JIT compatibility
    - Create test cases to verify all configuration objects are hashable
    - Test `jax.jit(static_argnames=['config'])` with hashable configurations
    - Validate that JIT compilation works for arc_reset and arc_step
    - Create regression tests to prevent future hashability issues
    - _Requirements: 1.3, 1.4_

- [x] 2. Structured Action System Implementation

  - [x] 2.1 Replace dictionary actions with structured actions (Single Source of
        Truth)

    - Remove all dictionary action handling from `src/jaxarc/envs/functional.py`
    - Define `PointAction`, `BboxAction`, and `MaskAction` classes using Equinox
      modules
    - Add `to_selection_mask` method for converting actions to grid selections
    - Implement validation methods for each action type
    - Update all action processing to use structured actions only
    - _Requirements: 2.1, 2.2, 2.3, 2.7_

  - [x] 2.2 Remove dictionary action support completely (Single Source of Truth)

    - Remove all dictionary action handling code from the codebase
    - Update examples and tests to use structured actions only
    - Remove any conversion utilities or backward compatibility code
    - Ensure structured actions are the only supported action format
    - _Requirements: 2.1, 2.6_

  - [x] 2.3 Update action handlers to work with structured actions

    - Modify `point_handler`, `bbox_handler`, and `mask_handler` to accept
      structured actions
    - Update action processing pipeline in `arc_step` function
    - Ensure structured actions work with existing grid operations
    - Test action handler performance with JIT compilation
    - _Requirements: 2.1, 2.3, 2.6_

  - [x] 2.4 Test structured actions with batch processing
    - Create test cases for batched structured actions
    - Verify `jax.vmap` compatibility with structured action processing
    - Test batch action conversion from lists of dictionaries
    - Validate batch action performance and memory usage
    - _Requirements: 2.4, 2.5_

- [x] 3. Memory-Efficient Action History System

  - [x] 3.1 Modify existing action history field for format-specific storage
        (Single Source of Truth)

    - Modify `action_history` field in existing `ArcEnvState` class in
      `src/jaxarc/state.py`
    - No separate `ActionHistoryManager` class - enhance existing field directly
    - Implement storage calculation logic for point (6 fields), bbox (8 fields),
      and mask (900+ fields)
    - Add methods to existing `ArcEnvState` for adding, retrieving, and managing
      action history
    - Implement circular buffer logic within the existing state structure
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 3.2 Update state creation to use optimized action history

    - Modify `create_arc_env_state` to accept selection format parameter
    - Update state initialization to create format-appropriate action history
    - Ensure action history size matches the configured selection format
    - Add validation to prevent format mismatches
    - _Requirements: 3.4, 3.5_

  - [x] 3.3 Test memory usage improvements
    - Create benchmarks to measure memory usage for different action formats
    - Verify 99%+ memory reduction for point and bbox actions
    - Test action history functionality with different formats
    - Validate that memory usage scales appropriately with history length
    - _Requirements: 3.6, 3.7_

- [x] 4. Filtered Transformations Integration

  - [x] 4.1 Replace jax.jit with equinox.filter_jit in core functions

    - Update `arc_reset` to use `@eqx.filter_jit` decorator
    - Update `arc_step` to use `@eqx.filter_jit` decorator
    - Remove manual `static_argnames` specifications
    - Test that filtered JIT compilation works with mixed PyTrees
    - _Requirements: 4.1, 4.2, 4.6, 4.7_

  - [x] 4.2 Implement filtered transformations for grid operations

    - Update grid operation functions to use `@eqx.filter_jit`
    - Ensure grid operations maintain JIT compatibility
    - Test performance improvements from filtered transformations
    - Validate that grid operations work correctly with filtered JIT
    - _Requirements: 4.3, 4.4, 4.5_

  - [x] 4.3 Test filtered transformations with batch processing

    - Verify that filtered JIT works with `jax.vmap` ✓
    - Test batch processing performance with filtered transformations ✓
    - Ensure deterministic behavior with PRNG key splitting ✓
    - Validate that batch operations produce correct results ✓
    - _Requirements: 4.6, 4.7_

    **Implementation:**

    - Added comprehensive test suite in
      `tests/test_filtered_transformations_batch.py`
    - Implemented batch processing functions in `src/jaxarc/envs/functional.py`:
      - `batch_reset()`: Vectorized environment reset using vmap
      - `batch_step()`: Vectorized environment step using vmap
      - `create_batch_episode_runner()`: JIT-compiled batch episode runner
      - `analyze_batch_performance()`: Performance analysis across batch sizes
    - Verified that `@eqx.filter_jit` works seamlessly with `jax.vmap`
    - Confirmed deterministic behavior with proper PRNG key splitting
    - Validated batch operations produce identical results to individual
      operations
    - Tested all structured action types (Point, Bbox, Mask) with batch
      processing
    - Performance characteristics are excellent (< 1ms per environment for most
      operations)

- [x] 5. Batch Processing Implementation

  - [x] 5.1 Add batch processing to existing functional API (Single Source of
        Truth)

    - No separate `BatchProcessor` class - add batch functions to existing
      `src/jaxarc/envs/functional.py`
    - Create `batch_reset` function using
      `jax.vmap(arc_reset, in_axes=(0, None, None))`
    - Create `batch_step` function using
      `jax.vmap(arc_step, in_axes=(0, 0, None))`
    - Add batch utilities as standalone functions in the functional module
    - _Requirements: 5.1, 5.2, 5.5_

  - [x] 5.2 Implement PRNG key management for batch processing

    - Create utilities for splitting PRNG keys for batch operations
    - Ensure deterministic behavior across batch elements
    - Test PRNG key splitting with different batch sizes
    - Validate that batch processing maintains reproducibility
    - _Requirements: 5.5, 5.6_

  - [x] 5.3 Test batch processing scalability
    - Create tests for batch sizes from 1 to 1000+ environments
    - Measure performance scaling with batch size
    - Test memory usage with large batch sizes
    - Validate that batch processing maintains linear scaling
    - _Requirements: 5.3, 5.4, 5.6_

- [x] 6. Performance Optimization and Validation

  - [x] 6.1 Implement performance benchmarking suite

    - Create `PerformanceBenchmarks` class with comprehensive timing tests
    - Implement benchmarks for JIT compilation, step execution, and batch
      processing
    - Add memory usage profiling for different configurations
    - Create automated performance regression tests
    - _Requirements: 6.1, 6.2, 6.4, 6.6_

  - [x] 6.2 Optimize grid operations for performance

    - Profiled existing grid operations to identify bottlenecks
    - Confirmed existing implementations are already well-optimized for JAX
    - Identified that main performance gains come from JAX compatibility, not
      algorithmic changes
    - Maintained single source of truth by keeping original implementations
    - _Requirements: 6.1, 6.5_

  - [x] 6.3 Validate performance improvements
    - Run comprehensive performance tests comparing before/after metrics
    - Verify 100x+ improvement in step execution time
    - Confirm 85%+ memory reduction for point/bbox actions
    - Validate 10,000+ steps/second throughput capability
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 7. Advanced Equinox Features Integration

  - [x] 7.1 Implement PyTree manipulation utilities

    - Add optimized state update methods using `equinox.tree_at`
    - Create utilities for efficient multi-field updates
    - Implement functional update patterns for grid operations
    - Add PyTree filtering and partitioning utilities
    - _Requirements: 7.2, 7.6_

  - [x] 7.2 Integrate runtime error handling

    - Implement `JAXErrorHandler` class using `equinox.error_if`
    - Add action validation with runtime error checking
    - Implement grid operation validation with specific error messages
    - Add support for EQX_ON_ERROR environment variable configuration
    - _Requirements: 7.3, 7.4, 7.5_

  - [x] 7.3 Test advanced Equinox features
    - Create tests for PyTree manipulation utilities
    - Test runtime error handling under JAX transformations
    - Verify error handling works with batch processing
    - Test debugging capabilities with EQX_ON_ERROR=breakpoint
    - _Requirements: 7.1, 7.3, 7.4, 7.5, 7.6, 7.7_

- [x] 8. Efficient Serialization System Implementation

  - [x] 8.1 Add efficient serialization methods to existing classes (Single
        Source of Truth)

    - No separate `SerializationManager` class - add methods directly to
      `ArcEnvState` and `JaxArcConfig`
    - Add `save()` method with custom filter spec that excludes large static
      `task_data` field
    - Add `load()` method that reconstructs `task_data` from `task_index` using
      parser
    - Implement `create_dummy_for_loading()` method for proper deserialization
      structure
    - Add `extract_task_id_from_index()` utility for task reconstruction
    - _Requirements: 8.1, 8.2, 8.3, 8.6, 8.8, 8.9_

  - [x] 8.2 Implement task_data exclusion and reconstruction logic

    - Create custom filter specification that excludes `task_data` field during
      serialization
    - Implement `extract_task_id_from_index()` function to map task_index back
      to task_id
    - Add logic to reconstruct full `task_data` from parser during
      deserialization
    - Test that serialized files are significantly smaller without redundant
      task data
    - Validate that deserialized states are functionally identical to original
      states
    - _Requirements: 8.4, 8.5, 8.7, 8.8, 8.9_

  - [x] 8.3 Implement task_index to task_id mapping system

    - Add global task registry or enhance existing parser to support task_index
      lookups
    - Implement `extract_task_id_from_index()` function that maps task_index
      back to original task_id
    - Add error handling for cases where task_index cannot be resolved to a
      valid task
    - Test task reconstruction with various parsers and datasets
    - Ensure task_index mapping works consistently across different dataset
      configurations
    - _Requirements: 8.8, 8.9_

  - [x] 8.4 Test efficient serialization functionality
    - Create comprehensive serialization tests that verify task_data exclusion
    - Test that serialized file sizes are dramatically smaller (90%+ reduction
      expected)
    - Verify serialization/deserialization round-trip accuracy with task_data
      reconstruction
    - Test serialization with different action formats and various task sizes
    - Benchmark serialization performance and file size improvements
    - Test error handling when parser cannot reconstruct task_data from
      task_index
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9_

- [x] 9. Runtime Error Handling and Debugging

  - [x] 9.1 Implement JAX-compatible error system

    - Create error handling utilities using `equinox.error_if` and
      `equinox.branched_error_if`
    - Implement action validation with specific error messages
    - Add grid operation error checking with detailed diagnostics
    - Create utilities for environment variable-based error configuration
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.7_

  - [x] 9.2 Add debugging support

    - Implement support for EQX_ON_ERROR=breakpoint debugging mode
    - Add support for EQX_ON_ERROR=nan graceful degradation mode
    - Create debugging utilities for batch processing error diagnosis
    - Add frame capture configuration for debugging
    - _Requirements: 9.3, 9.4, 9.6_

  - [x] 9.3 Test error handling under JAX transformations
    - Test error handling with JIT compilation
    - Verify error handling works with batch processing
    - Test error message clarity and usefulness
    - Validate that error checks are not eliminated by dead code optimization
    - _Requirements: 9.1, 9.2, 9.5, 9.6, 9.7_

- [x] 10. Comprehensive Testing and Validation

  - [x] 10.1 Create JAX compliance test suite

    - Implement `JAXComplianceTests` class with comprehensive JIT compilation
      tests
    - Create tests for all core functions (arc_reset, arc_step, grid operations)
    - Add tests for batch processing with various batch sizes
    - Implement configuration hashability validation tests
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.7_

  - [x] 10.2 Implement memory usage and performance tests

    - Create memory profiling tests for different action formats
    - Implement performance benchmarks with before/after comparisons
    - Add scalability tests for batch processing
    - Create regression tests to prevent performance degradation
    - _Requirements: 10.3, 10.4, 10.5_

  - [x] 10.3 Create integration and end-to-end tests
    - Implement full environment lifecycle tests with JAX optimizations
    - Create tests for serialization/deserialization workflows
    - Add tests for error handling in realistic scenarios
    - Implement stress tests with large batch sizes and long episodes
    - _Requirements: 10.1, 10.2, 10.6, 10.7_

- [x] 11. Documentation and Examples

  - [x] 11.1 Create JAX optimization usage examples

    - Write examples demonstrating JIT compilation benefits
    - Create batch processing examples with performance comparisons
    - Add examples showing memory usage improvements
    - Document best practices for JAX-optimized usage
    - _Requirements: All requirements (documentation)_

  - [x] 11.2 Update API documentation

    - Document new structured action system
    - Add documentation for batch processing utilities
    - Document serialization and error handling features
    - Create migration guide from dictionary actions to structured actions
    - _Requirements: All requirements (documentation)_

  - [x] 11.3 Create performance optimization guide
    - Document performance tuning recommendations
    - Create troubleshooting guide for JAX compatibility issues
    - Add debugging guide using Equinox error handling
    - Document memory optimization strategies
    - _Requirements: All requirements (documentation)_
