# Requirements Document

## Introduction

This specification addresses the need to simplify and standardize the JaxARC codebase by eliminating backwards compatibility concerns, establishing consistent coding patterns, improving code organization, and enhancing the visualization system to include task information. The goal is to create a single, coherent way to accomplish each task while maintaining JAX compliance and improving developer experience.

## Requirements

### Requirement 1: Eliminate Backwards Compatibility Code

**User Story:** As a developer, I want a single, clear API for each functionality so that I don't have to choose between multiple ways to accomplish the same task.

#### Acceptance Criteria

1. WHEN there are multiple functions that accomplish the same task (e.g., `arc_step` vs `arc_step_enhanced`) THEN the system SHALL keep only the most current/best implementation
2. WHEN old functions are removed THEN all tests, scripts, and examples SHALL be updated to use the new implementation
3. WHEN API changes are made THEN no deprecated functions SHALL remain in the codebase
4. WHEN functionality is consolidated THEN the remaining function SHALL have the most intuitive name

### Requirement 2: Standardize Configuration Pattern

**User Story:** As a developer, I want all configuration classes to follow the same initialization pattern so that I can work with them consistently.

#### Acceptance Criteria

1. WHEN a configuration class exists THEN it SHALL implement a `from_hydra(hydra_config)` class method
2. WHEN a configuration class has a `from_hydra` method THEN it SHALL also support direct Python initialization
3. WHEN other classes need configuration THEN they SHALL accept typed configuration objects rather than raw Hydra DictConfig objects
4. WHEN parsers or other components need dataset configuration THEN they SHALL accept `DatasetConfig` objects initialized via `DatasetConfig.from_hydra(hydra_config)`

### Requirement 3: Improve Code Organization and Modularity

**User Story:** As a developer, I want related functionality grouped together and long functions broken down so that the code is easier to understand and maintain.

#### Acceptance Criteria

1. WHEN similar functionality exists across multiple files THEN it SHALL be consolidated into appropriate modules
2. WHEN a function exceeds reasonable length (>50 lines) THEN it SHALL be broken down into smaller, focused functions
3. WHEN code organization is improved THEN the external package API SHALL remain consistent and intuitive
4. WHEN modules are reorganized THEN import patterns SHALL be simplified and documented

### Requirement 4: Enhance Task Visualization

**User Story:** As a researcher, I want to see the actual task I'm trying to solve at the beginning of each episode so that I can understand the context of the agent's actions.

#### Acceptance Criteria

1. WHEN an episode starts THEN the system SHALL log/visualize the actual task being solved
2. WHEN task visualization is created THEN it SHALL use existing SVG drawing functionality for parsed tasks
3. WHEN task information is displayed THEN it SHALL be clearly labeled and easy to understand
4. WHEN episode visualization is generated THEN it SHALL include the task visualization as the first item

### Requirement 5: Add Task Context to Step Visualizations

**User Story:** As a researcher, I want to see which specific task pair I'm working with in each step visualization so that I can track progress against the target.

#### Acceptance Criteria

1. WHEN step visualizations are generated THEN they SHALL include the task pair index information
2. WHEN task pair information is displayed THEN it SHALL appear alongside other metadata like Operation ID
3. WHEN task context is added THEN it SHALL not interfere with existing visualization layout
4. WHEN multiple task pairs exist THEN the current pair index SHALL be clearly indicated

### Requirement 6: Simplify and Improve Overall Code Quality

**User Story:** As a developer, I want the codebase to be easy to work with and understand so that I can focus on research rather than navigating complex code.

#### Acceptance Criteria

1. WHEN code patterns are inconsistent THEN they SHALL be standardized across the codebase
2. WHEN complex logic exists THEN it SHALL be simplified without losing functionality
3. WHEN external APIs are defined THEN they SHALL be intuitive and well-documented
4. WHEN refactoring is complete THEN the codebase SHALL maintain all existing functionality while being easier to understand
5. WHEN JAX compliance is required THEN all refactored code SHALL maintain compatibility with JAX transformations