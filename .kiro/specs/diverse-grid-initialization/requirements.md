# Requirements Document

## Introduction

This feature enhances the ARC environment's initialization system to support
diverse working grid initialization strategies. Currently, the working grid is
always initialized with one of the demo input grids, which limits training
diversity when using batched environments. This enhancement will add multiple
initialization modes including demo grids, permutations, empty grids, and random
grids to maximize training variability and help agents learn more generalizable
transformation principles.

## Requirements

### Requirement 1

**User Story:** As a researcher training RL agents on ARC tasks, I want multiple
working grid initialization strategies so that I can create diverse training
scenarios from a single task and improve generalization.

#### Acceptance Criteria

1. WHEN the environment is configured with diverse initialization THEN the
   system SHALL support at least four initialization modes: demo, permutation,
   empty, and random
2. WHEN using demo mode THEN the system SHALL initialize the working grid with
   one of the available demo input grids (current behavior)
3. WHEN using permutation mode THEN the system SHALL create variations of demo
   input grids through valid transformations
4. WHEN using empty mode THEN the system SHALL initialize the working grid as
   completely empty (all zeros)
5. WHEN using random mode THEN the system SHALL initialize the working grid with
   random valid colors and patterns

### Requirement 2

**User Story:** As a developer using batched environments, I want configurable
initialization distribution so that I can control the mix of different
initialization strategies across the batch.

#### Acceptance Criteria

1. WHEN configuring initialization THEN the system SHALL allow specifying
   probability weights for each initialization mode
2. WHEN processing a batch THEN the system SHALL distribute initialization modes
   according to the specified probabilities
3. WHEN no probabilities are specified THEN the system SHALL default to equal
   distribution across all enabled modes
4. WHEN a single mode is specified THEN the system SHALL use only that mode for
   all batch elements

### Requirement 3

**User Story:** As a researcher, I want permutation-based initialization to
create meaningful variations so that agents can learn robust transformation
patterns.

#### Acceptance Criteria

1. WHEN using permutation mode THEN the system SHALL apply valid grid
   transformations such as rotation, reflection, and color remapping
2. WHEN applying permutations THEN the system SHALL ensure the resulting grid
   maintains valid ARC constraints (colors 0-9, proper dimensions)
3. WHEN creating permutations THEN the system SHALL generate variations that
   preserve the underlying structure while changing surface features
4. WHEN no valid permutations exist THEN the system SHALL fall back to demo mode
   for that batch element

### Requirement 4

**User Story:** As a developer, I want random initialization to create diverse
starting conditions so that agents learn to work with various grid
configurations.

#### Acceptance Criteria

1. WHEN using random mode THEN the system SHALL generate grids with random
   colors from the valid ARC color palette (0-9)
2. WHEN generating random grids THEN the system SHALL respect the task's grid
   dimensions and constraints
3. WHEN creating random patterns THEN the system SHALL include both sparse and
   dense configurations
4. WHEN using random initialization THEN the system SHALL ensure reproducibility
   through proper PRNG key management

### Requirement 5

**User Story:** As a researcher, I want backward compatibility and single source
of truth so that existing code continues to work without modification and
there's only one way to accomplish each task.

#### Acceptance Criteria

1. WHEN no initialization mode is specified THEN the system SHALL default to
   demo mode (current behavior)
2. WHEN using existing configuration THEN the system SHALL maintain identical
   behavior to the current implementation
3. WHEN migrating to new initialization THEN the system SHALL provide clear
   configuration options without breaking changes
4. WHEN enhancing existing functions THEN the system SHALL modify the original
   functions (e.g., arc_reset, arc_step) rather than creating new parallel
   functions
5. WHEN implementing new features THEN the system SHALL maintain a single source
   of truth by replacing old implementations completely rather than keeping both
   versions

### Requirement 6

**User Story:** As a developer, I want efficient batch processing so that
diverse initialization doesn't significantly impact performance.

#### Acceptance Criteria

1. WHEN processing batched initialization THEN the system SHALL maintain JAX
   compatibility for JIT compilation
2. WHEN generating diverse grids THEN the system SHALL use vectorized operations
   where possible
3. WHEN creating permutations THEN the system SHALL pre-compute transformations
   to avoid runtime overhead
4. WHEN using random initialization THEN the system SHALL efficiently manage
   PRNG keys across batch dimensions
