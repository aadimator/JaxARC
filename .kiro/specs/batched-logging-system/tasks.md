# Implementation Plan

- [x] 1. Clean up deprecated async logging settings

  - Remove async logging fields from YAML configuration files (basic.yaml,
    full.yaml)
  - Update LoggingConfig dataclass to remove async logging fields
  - Update configuration validation to remove async logging parameter checks
  - Remove any unused async logging code references
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 2. Create batched logging configuration system

  - [x] 2.1 Create batched.yaml configuration file

    - Write conf/logging/batched.yaml with batched logging settings
    - Include aggregation frequency, sampling parameters, and metric selection
      options
    - _Requirements: 3.1, 3.2_

  - [x] 2.2 Extend LoggingConfig dataclass for batched logging
    - Add batched_logging_enabled, sampling_enabled, num_samples,
      sample_frequency fields
    - Add aggregated metrics selection fields (log_aggregated_rewards,
      log_aggregated_similarity, etc.)
    - Implement validation for new batched logging parameters
    - _Requirements: 3.3, 3.4_

- [x] 3. Implement core batched logging functionality in ExperimentLogger

  - [x] 3.1 Add log_batch_step method

    - Implement log_batch_step method that handles batched training data
    - Add frequency-based control for aggregation and sampling
    - Include error isolation for handler failures
    - _Requirements: 1.1, 1.2, 1.4_

  - [x] 3.2 Implement batch metrics aggregation

    - Create \_aggregate_batch_metrics method using JAX operations
    - Calculate mean, std, min, max for rewards, similarity scores, episode
      lengths
    - Handle scalar training metrics (policy_loss, value_loss, gradient_norm)
    - Include success rate calculations
    - _Requirements: 1.1, 1.3, 6.2_

  - [x] 3.3 Implement episode sampling for detailed logging
    - Create \_sample_episodes_from_batch method for representative sampling
    - Use deterministic sampling based on update step for reproducibility
    - Reconstruct episode summary data for existing log_episode_summary method
    - Handle cases where num_samples exceeds batch_size
    - _Requirements: 2.1, 2.2, 2.4, 2.5_

- [x] 4. Extend logging handlers to support aggregated metrics

  - [x] 4.1 Add log_aggregated_metrics to WandbHandler

    - Implement log_aggregated_metrics method for wandb integration
    - Add batch/ prefix to distinguish from individual step metrics
    - Include proper step information for time-series plots
    - Handle wandb logging failures gracefully
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 4.2 Add log_aggregated_metrics to FileHandler

    - Implement log_aggregated_metrics method for file output
    - Create batch_metrics.jsonl file for aggregated metrics
    - Include timestamp and step information in log entries
    - Handle file writing errors gracefully
    - _Requirements: 3.4_

  - [x] 4.3 Add log_aggregated_metrics to RichHandler

    - Implement log_aggregated_metrics method for console display
    - Create formatted table display for batch metrics
    - Group metrics by category (rewards, similarity, training)
    - Handle console display errors gracefully
    - _Requirements: 3.4_

  - [x] 4.4 Update SVGHandler for episode-level sampling
    - Modify SVGHandler to handle episode summary data from sampling
    - Support initial vs final state visualization when state data is available
    - Implement graceful degradation when state data is not provided
    - Ensure no step-by-step visualization attempts in batched mode
    - _Requirements: 2.3, 3.4_

- [x] 5. Implement comprehensive testing for batched logging

  - [x] 5.1 Create unit tests for configuration system

    - Test batched logging configuration loading and validation
    - Test async logging field removal from configs
    - Test configuration error handling and validation messages
    - _Requirements: 3.3, 5.4_

  - [x] 5.2 Create unit tests for ExperimentLogger extensions

    - Test log_batch_step method with mock batch data
    - Test \_aggregate_batch_metrics with various data scenarios
    - Test \_sample_episodes_from_batch with different batch sizes
    - Test error isolation when handlers fail
    - _Requirements: 1.1, 1.3, 2.1, 2.4, 2.5_

  - [x] 5.3 Create unit tests for handler extensions

    - Test log_aggregated_metrics for each handler type
    - Test integration with existing handler methods
    - Test error handling and graceful degradation
    - Test handler-specific functionality (wandb prefixes, file formats, console
      display)
    - _Requirements: 4.1, 4.2, 4.3, 3.4_

  - [x] 5.4 Create integration tests for end-to-end batched logging
    - Test complete batched logging pipeline with mock training data
    - Verify file outputs, wandb integration, and console display
    - Test configuration-driven behavior changes
    - Test performance with various batch sizes
    - _Requirements: 6.1, 6.3, 6.4, 7.1, 7.2_

- [x] 6. Performance optimization and validation

  - [x] 6.1 Optimize JAX operations for aggregation

    - Ensure all aggregation operations are JAX-compatible and JIT-compilable
    - Benchmark aggregation performance with large batch sizes
    - Verify memory usage remains reasonable during aggregation
    - _Requirements: 6.2, 6.5_

  - [x] 6.2 Validate sampling efficiency
    - Test sampling performance with various batch sizes (100, 1000, 10000)
    - Ensure sampling doesn't create memory bottlenecks
    - Verify deterministic sampling produces consistent results
    - _Requirements: 6.3, 6.5_

- [x] 7. Integration and documentation

  - [x] 7.1 Create example usage patterns

    - Write example code showing how to use batched logging in training loops
    - Create configuration examples for different use cases
    - Document integration with existing PureJaxRL patterns
    - _Requirements: 7.3, 7.4_

  - [x] 7.2 Update existing documentation

    - Update logging documentation to include batched logging capabilities
    - Document configuration options and their effects
    - Add troubleshooting guide for common batched logging issues
    - _Requirements: 7.4_

  - [x] 7.3 Validate backward compatibility
    - Ensure existing single-environment logging continues to work unchanged
    - Test that old configuration files load without batched logging settings
    - Verify no performance regression in single-environment scenarios
    - _Requirements: 5.4, 6.1_
