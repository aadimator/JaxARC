# Requirements Document

## Introduction

The current testing setup for JaxARC has become outdated and convoluted due to
significant codebase evolution and API changes. The project has migrated to
Equinox-based modules, updated the type system with JAXTyping, and restructured
the architecture. Many existing tests are no longer relevant, test outdated
APIs, or are duplicated. This feature will completely overhaul the testing
infrastructure to align with the current codebase state, focusing on the
essential functionality while removing obsolete tests.

## Requirements

### Requirement 1

**User Story:** As a developer, I want a clean and focused test suite that only
tests the current API and functionality, so that I can confidently develop and
maintain the codebase without confusion from outdated tests.

#### Acceptance Criteria

1. WHEN analyzing existing tests THEN the system SHALL identify tests that are
   no longer relevant to the current codebase
2. WHEN removing obsolete tests THEN the system SHALL ensure no backwards
   compatibility concerns are considered
3. WHEN organizing tests THEN the system SHALL group tests by current module
   structure (types, envs, parsers, utils)
4. WHEN creating the new test structure THEN the system SHALL eliminate all
   duplicate test cases

### Requirement 2

**User Story:** As a developer, I want comprehensive test coverage for the
current Equinox-based architecture, so that I can ensure the JAX-compatible
modules work correctly with transformations.

#### Acceptance Criteria

1. WHEN testing core types THEN the system SHALL validate Grid, JaxArcTask, and
   ARCLEAction Equinox modules
2. WHEN testing JAX compatibility THEN the system SHALL verify jit, vmap, and
   pmap transformations work correctly
3. WHEN testing type validation THEN the system SHALL ensure JAXTyping
   annotations are properly validated
4. WHEN testing module initialization THEN the system SHALL verify
   **check_init** methods work correctly

### Requirement 3

**User Story:** As a developer, I want tests that focus on the current
environment API, so that I can validate the functional and object-oriented
interfaces work as expected.

#### Acceptance Criteria

1. WHEN testing environment functionality THEN the system SHALL test the current
   ArcEnvironment class
2. WHEN testing functional API THEN the system SHALL validate arc_reset and
   arc_step functions
3. WHEN testing configuration THEN the system SHALL verify the current config
   system with factory functions
4. WHEN testing actions THEN the system SHALL validate ARCLE action handling and
   grid operations

### Requirement 4

**User Story:** As a developer, I want tests for the current parser
implementations, so that I can ensure data loading works correctly with the
updated type system.

#### Acceptance Criteria

1. WHEN testing parsers THEN the system SHALL validate ArcAgiParser,
   ConceptArcParser, and MiniArcParser
2. WHEN testing data loading THEN the system SHALL ensure parsers create valid
   JaxArcTask objects
3. WHEN testing parser utilities THEN the system SHALL verify grid conversion
   and validation functions
4. WHEN testing parser integration THEN the system SHALL ensure parsers work
   with the current dataset structure

### Requirement 5

**User Story:** As a developer, I want tests for current utility modules, so
that I can ensure visualization, grid operations, and configuration utilities
work correctly.

#### Acceptance Criteria

1. WHEN testing visualization THEN the system SHALL validate current terminal
   and SVG rendering functions
2. WHEN testing grid utilities THEN the system SHALL verify grid manipulation
   and shape detection functions
3. WHEN testing configuration utilities THEN the system SHALL validate config
   loading and factory functions
4. WHEN testing JAX types THEN the system SHALL ensure JAXTyping definitions
   work correctly

### Requirement 6

**User Story:** As a developer, I want a streamlined test organization that
matches the current project structure, so that tests are easy to find and
maintain.

#### Acceptance Criteria

1. WHEN organizing test files THEN the system SHALL mirror the src/jaxarc
   directory structure
2. WHEN naming test files THEN the system SHALL use clear, descriptive names
   that match the modules being tested
3. WHEN structuring test directories THEN the system SHALL eliminate unnecessary
   subdirectories and complexity
4. WHEN removing files THEN the system SHALL delete all obsolete test files and
   their associated cache files
