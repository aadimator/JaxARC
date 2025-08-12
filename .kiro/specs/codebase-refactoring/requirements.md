# Requirements Document

## Introduction

This specification addresses critical architectural and code quality issues in
the JaxARC codebase that are hindering maintainability, readability, and
efficiency. The current codebase suffers from code duplication, circular
dependencies, overly complex configuration systems, and inconsistent patterns
that make it difficult to extend and maintain. This refactoring initiative will
modernize the codebase using best practices, eliminate redundancy, and introduce
Equinox for better JAX integration.

## Requirements

### Requirement 1: Eliminate Code Duplication and Circular Dependencies

**User Story:** As a developer, I want a clean, non-duplicated codebase
structure so that I can easily understand and modify the code without
encountering circular import issues.

#### Acceptance Criteria

1. WHEN the codebase is analyzed THEN there SHALL be no duplicate class
   definitions (e.g., ArcEnvState defined in multiple files)
2. WHEN importing modules THEN there SHALL be no circular dependencies between
   modules
3. WHEN core types are needed THEN they SHALL be imported from a single
   canonical location (types.py)
4. WHEN state management is implemented THEN ArcEnvState SHALL be defined only
   in src/jaxarc/types.py
5. WHEN other modules need state types THEN they SHALL import from the canonical
   types module

### Requirement 2: Simplify Action Handling Architecture

**User Story:** As a developer, I want a streamlined action handling system so
that the code is easier to understand and maintain without redundant validation
logic.

#### Acceptance Criteria

1. WHEN arc_step processes actions THEN it SHALL delegate all action handling to
   the actions.py handler system
2. WHEN actions are processed THEN there SHALL be no duplicate validation logic
   between arc_step and action handlers
3. WHEN action handlers are used THEN they SHALL return standardized selection
   masks consistently
4. WHEN different action formats are supported THEN the handler selection SHALL
   be based solely on configuration
5. WHEN grid operations are executed THEN they SHALL receive standardized action
   dictionaries with selection masks

### Requirement 3: Consolidate Parser Logic Using Inheritance

**User Story:** As a developer, I want DRY parser implementations so that common
functionality is not duplicated across multiple parser classes.

#### Acceptance Criteria

1. WHEN parser classes are implemented THEN common methods SHALL be defined only
   in ArcDataParserBase
2. WHEN specific parsers need common functionality THEN they SHALL call super()
   methods instead of reimplementing
3. WHEN \_process_training_pairs is needed THEN it SHALL be implemented once in
   the base class
4. WHEN \_pad_and_create_masks is needed THEN it SHALL be implemented once in
   the base class
5. WHEN \_validate_grid_colors is needed THEN it SHALL be implemented once in
   the base class

### Requirement 4: Streamline Configuration System with Hydra-First Approach

**User Story:** As a developer, I want a simplified configuration system that
leverages Hydra's capabilities so that configuration management is consistent
and not overly complex.

#### Acceptance Criteria

1. WHEN configurations are created THEN Hydra YAML files SHALL be the primary
   configuration method
2. WHEN factory functions exist THEN they SHALL be minimal and only for
   essential programmatic use cases
3. WHEN configuration objects are needed THEN they SHALL be created directly
   from Hydra configs when possible
4. WHEN multiple configuration methods exist THEN the codebase SHALL prefer
   Hydra-based configuration over factory functions
5. WHEN configuration complexity is evaluated THEN the system SHALL favor
   simplicity over multiple configuration pathways

### Requirement 5: Integrate Equinox and JAXTyping for Modern JAX Patterns

**User Story:** As a developer, I want to use Equinox and JAXTyping for better
JAX integration so that the codebase follows modern JAX best practices with
proper array typing and is more maintainable.

#### Acceptance Criteria

1. WHEN JAX modules are implemented THEN they SHALL use Equinox Module classes
   where appropriate
2. WHEN state management is needed THEN Equinox PyTree structures SHALL be
   preferred over chex dataclasses where beneficial
3. WHEN functional transformations are applied THEN Equinox patterns SHALL be
   used for better JAX compatibility
4. WHEN array types are defined THEN JAXTyping SHALL be used for precise array
   shape and dtype annotations
5. WHEN JAX arrays are used THEN they SHALL be properly typed with JAXTyping
   annotations (e.g., Float[Array, "height width"], Int[Array, "batch height
   width"])
6. WHEN the codebase is evaluated THEN Equinox and JAXTyping integration SHALL
   improve code clarity, type safety, and JAX performance
7. WHEN existing chex dataclasses are migrated THEN the migration SHALL maintain
   backward compatibility

### Requirement 6: Establish Consistent Code Organization Patterns

**User Story:** As a developer, I want consistent code organization patterns so
that I can easily navigate and understand the codebase structure.

#### Acceptance Criteria

1. WHEN modules are organized THEN each module SHALL have a single, clear
   responsibility
2. WHEN imports are structured THEN they SHALL follow a consistent pattern
   across all modules
3. WHEN utility functions are needed THEN they SHALL be grouped logically in
   appropriate utility modules
4. WHEN core functionality is implemented THEN it SHALL be separated from
   configuration and utility code
5. WHEN the codebase is reviewed THEN the organization SHALL be intuitive and
   follow Python best practices

### Requirement 7: Reduce Configuration Factory Complexity

**User Story:** As a developer, I want simplified configuration factories so
that creating configurations is straightforward and not verbose.

#### Acceptance Criteria

1. WHEN configuration factories are implemented THEN they SHALL use a modular
   base configuration approach
2. WHEN factory functions create configs THEN they SHALL minimize code
   duplication through shared base functions
3. WHEN specific configurations are needed THEN they SHALL compose smaller
   configuration components
4. WHEN factory complexity is evaluated THEN the system SHALL prefer composition
   over large monolithic functions
5. WHEN configurations are created THEN the process SHALL be clear and require
   minimal boilerplate code

### Requirement 8: Improve Type Safety and Validation

**User Story:** As a developer, I want robust type safety and validation so that
errors are caught early and the code is more reliable.

#### Acceptance Criteria

1. WHEN types are defined THEN they SHALL be comprehensive and cover all major
   data structures
2. WHEN validation is implemented THEN it SHALL be consistent across all modules
3. WHEN Equinox is integrated THEN type annotations SHALL be maintained and
   improved
4. WHEN runtime validation is needed THEN it SHALL be implemented efficiently
   without performance penalties
5. WHEN the type system is evaluated THEN it SHALL provide clear error messages
   and catch common mistakes
