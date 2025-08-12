# Requirements Document

## Introduction

The current ARC environment step logic is rudimentary and needs significant
enhancement to properly support the multi-demonstration nature of ARC tasks.
Currently, the environment only uses the first training pair and lacks proper
support for parallel training on multiple demonstration pairs, test pair
handling, action history tracking, and flexible action space control. This
feature will transform the environment into a more sophisticated system that can
properly leverage all available ARC task data for training and evaluation.

## Requirements

### Requirement 1: Multi-Demonstration Training Support

**User Story:** As an RL researcher, I want the environment to support training
on all available demonstration pairs, either sequentially or in parallel, so
that my agent can learn from the complete set of examples provided in each ARC
task.

#### Acceptance Criteria

1. WHEN the environment is configured for multi-demonstration mode THEN the
   system SHALL support initialization with any available training demonstration
   pair
2. WHEN stepping through the environment THEN the system SHALL work on the
   selected demonstration pair (input/output grid combination)
3. WHEN accessing demonstration pairs THEN the system SHALL provide access to
   the correct target grid for reward calculation and validation
4. WHEN parallel processing is enabled THEN the system SHALL support efficient
   JAX vectorization across multiple demonstration pairs
5. WHEN a demonstration pair is completed THEN the system SHALL allow switching
   to other available demonstration pairs or episode termination

### Requirement 2: Test Pair Handling and Evaluation Mode

**User Story:** As an RL researcher, I want the environment to support
evaluation on test pairs without access to target grids during solving, so that
I can properly evaluate my agent's generalization capabilities.

#### Acceptance Criteria

1. WHEN switching to evaluation mode THEN the system SHALL initialize
   environments with test input grids
2. WHEN in evaluation mode THEN the system SHALL NOT provide access to true test
   output grids during stepping
3. WHEN evaluating test pairs THEN the system SHALL support 1-4 test pairs per
   task as specified in ARC format
4. WHEN calculating rewards in evaluation mode THEN the system SHALL use
   alternative reward signals (e.g., step penalties only)
5. WHEN submitting solutions for test pairs THEN the system SHALL allow
   comparison against true outputs for final evaluation

### Requirement 3: Action History Tracking

**User Story:** As an RL researcher, I want the environment to track the
complete history of actions taken during an episode, so that I can analyze agent
behavior, implement replay mechanisms, and support advanced RL algorithms that
require action sequences.

#### Acceptance Criteria

1. WHEN an action is executed THEN the system SHALL store the action in a
   JAX-compatible history structure
2. WHEN accessing action history THEN the system SHALL provide chronologically
   ordered action sequences
3. WHEN storing action history THEN the system SHALL maintain fixed-size arrays
   with proper padding for JAX compatibility
4. WHEN the episode ends THEN the system SHALL provide complete action history
   for analysis
5. WHEN resetting the environment THEN the system SHALL clear previous action
   history appropriately

### Requirement 4: Enhanced Action Space Control

**User Story:** As an RL researcher, I want basic control over which actions are
allowed in different contexts, so that I can experiment with different action
subsets and restrict operations when needed.

#### Acceptance Criteria

1. WHEN configuring the environment THEN the system SHALL support specification
   of allowed operation subsets
2. WHEN an invalid operation is attempted THEN the system SHALL handle it
   according to configured policy (reject, clip, or penalize)
3. WHEN in different contexts THEN the system SHALL support basic dynamic action
   space modification
4. WHEN validating actions THEN the system SHALL provide clear feedback about
   action validity
5. WHEN using restricted action sets THEN the system SHALL maintain performance
   and JAX compatibility

### Requirement 5: Flexible Episode Management

**User Story:** As an RL researcher, I want flexible control over episode
termination and continuation across demonstration pairs, so that I can implement
different training strategies and evaluation protocols.

#### Acceptance Criteria

1. WHEN working with multiple demonstration pairs THEN the system SHALL support
   independent episode management for each pair
2. WHEN a demonstration pair is solved THEN the system SHALL allow continuation
   to other unsolved pairs or episode termination
3. WHEN configuring episode behavior THEN the system SHALL support different
   termination criteria (solve all, solve any, time limits)
4. WHEN switching between training and evaluation THEN the system SHALL adapt
   episode management accordingly
5. WHEN tracking progress THEN the system SHALL provide detailed metrics for
   each demonstration/test pair

### Requirement 6: JAX Compatibility and Performance

**User Story:** As an RL researcher, I want all enhancements to maintain full
JAX compatibility and high performance, so that I can leverage JAX
transformations for efficient training at scale.

#### Acceptance Criteria

1. WHEN implementing new features THEN all data structures SHALL use static
   shapes with appropriate padding
2. WHEN storing variable-length data THEN the system SHALL use fixed-size arrays
   with masking
3. WHEN performing JAX transformations THEN all functions SHALL remain pure and
   JIT-compilable
4. WHEN batching operations THEN the system SHALL support efficient
   vectorization across demonstration pairs
5. WHEN using JAX transformations THEN the system SHALL maintain compatibility
   with vmap, pmap, and jit

### Requirement 7: Configuration-Driven Behavior

**User Story:** As an RL researcher, I want all new functionality to be
configurable through the existing configuration system, so that I can easily
experiment with different settings without code changes.

#### Acceptance Criteria

1. WHEN configuring multi-demonstration support THEN the system SHALL provide
   options for parallel vs sequential processing
2. WHEN configuring action history THEN the system SHALL allow specification of
   history length and storage format
3. WHEN configuring action space control THEN the system SHALL support operation
   filtering and validation policies
4. WHEN configuring episode management THEN the system SHALL provide options for
   termination criteria and continuation policies
5. WHEN using different configurations THEN the system SHALL validate
   configuration compatibility and provide clear error messages
