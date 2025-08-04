# Requirements Document

## Introduction

This feature adds support for ConceptARC and MiniARC datasets to the JaxARC project. ConceptARC is a benchmark dataset organized around 16 concept groups with 10 tasks each, designed to systematically assess abstraction and generalization abilities. MiniARC is a 5x5 compact version of ARC with 400 training and 400 evaluation tasks, designed for faster experimentation and prototyping. Both datasets follow the same JSON format as the original ARC dataset but have different characteristics and download sources.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to use ConceptARC dataset for evaluating concept-based reasoning, so that I can test my models on systematically organized concept groups.

#### Acceptance Criteria

1. WHEN I configure the environment with ConceptARC dataset THEN the system SHALL load tasks from the ConceptARC corpus directory structure
2. WHEN I request a random task from ConceptARC THEN the system SHALL return a task from one of the 16 concept groups
3. WHEN I process ConceptARC tasks THEN the system SHALL handle 1-4 demonstration pairs and 3 test inputs per task
4. WHEN I use ConceptARC configuration THEN the system SHALL support the same grid dimensions as standard ARC (up to 30x30)

### Requirement 2

**User Story:** As a developer, I want to use MiniARC dataset for rapid prototyping and testing, so that I can iterate quickly with smaller grid sizes and faster processing.

#### Acceptance Criteria

1. WHEN I configure the environment with MiniARC dataset THEN the system SHALL load tasks from MiniARC training and evaluation directories
2. WHEN I process MiniARC tasks THEN the system SHALL handle grids with maximum size of 5x5
3. WHEN I request tasks from MiniARC THEN the system SHALL provide access to 400 training and 400 evaluation tasks
4. WHEN I use MiniARC configuration THEN the system SHALL optimize for smaller grid dimensions and faster processing

### Requirement 3

**User Story:** As a user, I want automatic dataset downloading capabilities, so that I can easily obtain the required datasets without manual setup.

#### Acceptance Criteria

1. WHEN I run a download command for ConceptARC THEN the system SHALL clone the GitHub repository and extract the corpus directory
2. WHEN I run a download command for MiniARC THEN the system SHALL download from the appropriate source and organize the data correctly
3. WHEN datasets are downloaded THEN the system SHALL place them in the configured data directory structure
4. WHEN download fails THEN the system SHALL provide clear error messages and fallback instructions

### Requirement 4

**User Story:** As a user, I want flexible dataset configuration options, so that I can point to custom dataset locations or use different data organization structures.

#### Acceptance Criteria

1. WHEN I specify custom data paths in configuration THEN the system SHALL use those paths instead of default locations
2. WHEN I have datasets in different directory structures THEN the system SHALL support configurable path mappings
3. WHEN I want to use local datasets THEN the system SHALL work without requiring downloads
4. WHEN configuration is invalid THEN the system SHALL provide clear validation errors

### Requirement 5

**User Story:** As a developer, I want dedicated parsers for each dataset type, so that I can handle dataset-specific characteristics and optimizations.

#### Acceptance Criteria

1. WHEN I use ConceptARC parser THEN the system SHALL handle concept group organization and metadata
2. WHEN I use MiniARC parser THEN the system SHALL optimize for 5x5 grid constraints and smaller data structures
3. WHEN parsers process tasks THEN they SHALL maintain compatibility with the existing JaxArcTask interface
4. WHEN parsers encounter errors THEN they SHALL provide dataset-specific error messages and validation

### Requirement 6

**User Story:** As a researcher, I want configuration presets for different dataset types, so that I can easily switch between datasets with appropriate settings.

#### Acceptance Criteria

1. WHEN I select ConceptARC configuration THEN the system SHALL use appropriate grid sizes, task limits, and parser settings
2. WHEN I select MiniARC configuration THEN the system SHALL use optimized settings for 5x5 grids and smaller datasets
3. WHEN I switch between dataset configurations THEN the system SHALL maintain consistent behavior and interfaces
4. WHEN configurations are loaded THEN they SHALL include dataset-specific metadata and descriptions