# Requirements Document

## Introduction

This feature implements a comprehensive, configurable logging solution for batched training scenarios in JaxARC. Currently, the logging system works well for single-environment evaluation and debugging, but lacks proper support for high-performance batched training where hundreds or thousands of environments run simultaneously. This feature will extend the existing logging infrastructure to handle batched data efficiently while maintaining the same configurability and handler-based architecture.

The implementation will also clean up deprecated async logging settings that are no longer used, ensuring the codebase remains maintainable and focused.

## Requirements

### Requirement 1

**User Story:** As a researcher training RL agents on batched environments, I want to log aggregated metrics across all environments in the batch, so that I can monitor training progress without performance overhead.

#### Acceptance Criteria

1. WHEN batched training data is provided to the logging system THEN the system SHALL calculate and log aggregated statistics (mean, std, min, max) for key metrics like rewards, episode lengths, and similarity scores
2. WHEN the logging frequency is configured THEN the system SHALL only log aggregated metrics at the specified intervals to minimize performance impact
3. WHEN aggregated metrics are calculated THEN the system SHALL use JAX operations for efficient computation across the batch dimension
4. WHEN batched logging is disabled in configuration THEN the system SHALL skip all batched logging operations without errors

### Requirement 2

**User Story:** As a researcher debugging training issues, I want to see detailed step-by-step information for a sample of environments from the batch, so that I can understand what's happening without overwhelming the logging system.

#### Acceptance Criteria

1. WHEN sampling is enabled in configuration THEN the system SHALL randomly select a configurable number of environments from the batch for detailed logging
2. WHEN detailed samples are logged THEN the system SHALL use the existing log_step method to maintain consistency with single-environment logging
3. WHEN sampling frequency is configured THEN the system SHALL only log detailed samples at the specified intervals
4. WHEN the number of requested samples exceeds the batch size THEN the system SHALL log all environments in the batch
5. WHEN sampling uses randomization THEN the system SHALL use deterministic keys based on the update step for reproducible sampling

### Requirement 3

**User Story:** As a researcher, I want to configure all aspects of batched logging through YAML configuration files, so that I can easily adjust logging behavior for different experiments without code changes.

#### Acceptance Criteria

1. WHEN a batched logging configuration is created THEN it SHALL include settings for enabling/disabling batched logging, aggregation frequency, sampling parameters, and metric selection
2. WHEN batched logging configuration is loaded THEN it SHALL integrate seamlessly with the existing Hydra configuration system
3. WHEN configuration validation occurs THEN the system SHALL validate all batched logging parameters and provide clear error messages for invalid settings
4. WHEN batched logging is configured THEN it SHALL support the same handler ecosystem (file, SVG, rich, wandb) as single-environment logging

### Requirement 4

**User Story:** As a researcher using Weights & Biases for experiment tracking, I want aggregated batch metrics to be logged to wandb automatically, so that I can create smooth training curves and compare experiments.

#### Acceptance Criteria

1. WHEN the wandb handler is enabled and batched logging is active THEN aggregated metrics SHALL be automatically logged to wandb at the configured frequency
2. WHEN aggregated metrics are logged to wandb THEN they SHALL include proper step information for creating time-series plots
3. WHEN wandb logging fails THEN the system SHALL continue with other handlers and log appropriate warnings
4. WHEN detailed samples are logged THEN they SHALL use the existing wandb integration without modification

### Requirement 5

**User Story:** As a maintainer of the JaxARC codebase, I want to remove deprecated async logging settings, so that the configuration system remains clean and focused on actively used features.

#### Acceptance Criteria

1. WHEN async logging settings are removed THEN they SHALL be eliminated from all YAML configuration files (basic.yaml, full.yaml)
2. WHEN the LoggingConfig dataclass is updated THEN it SHALL no longer contain async logging fields (queue_size, worker_threads, batch_size, flush_interval, enable_compression)
3. WHEN configuration validation occurs THEN it SHALL not reference any async logging parameters
4. WHEN existing configurations are loaded THEN they SHALL continue to work without async logging settings
5. WHEN the cleanup is complete THEN no references to async logging SHALL remain in the logging system

### Requirement 6

**User Story:** As a researcher, I want the batched logging system to maintain the same performance characteristics as the current single-environment logging, so that logging doesn't become a bottleneck during training.

#### Acceptance Criteria

1. WHEN batched logging is active THEN the performance overhead SHALL be minimal compared to single-environment logging
2. WHEN aggregation is performed THEN it SHALL use efficient JAX operations that can be JIT-compiled
3. WHEN sampling is used THEN it SHALL minimize the amount of data processed for detailed logging
4. WHEN handlers fail THEN the system SHALL continue with error isolation to prevent training interruption
5. WHEN memory usage is considered THEN batched logging SHALL not significantly increase memory consumption beyond the batch data itself

### Requirement 7

**User Story:** As a researcher, I want the batched logging system to be extensible, so that I can add new metrics or handlers in the future without major refactoring.

#### Acceptance Criteria

1. WHEN new aggregated metrics are needed THEN they SHALL be easily added to the aggregation function
2. WHEN new handlers are created THEN they SHALL optionally implement batched logging support through a standard interface
3. WHEN the logging system evolves THEN the batched logging extension SHALL maintain compatibility with the existing handler architecture
4. WHEN configuration changes are made THEN they SHALL follow the same patterns as existing logging configuration