# Requirements Document

## Introduction

This feature migrates ARC-AGI dataset downloading from Kaggle to GitHub
repositories and consolidates all dataset downloading under a unified API. The
migration addresses the need to remove Kaggle dependencies and standardize
dataset access across all supported datasets (ARC-AGI-1, ARC-AGI-2, ConceptARC,
MiniARC). The GitHub repositories provide individual JSON task files instead of
combined JSON files, requiring parser updates to handle the different data
organization.

## Requirements

### Requirement 1

**User Story:** As a developer, I want to download ARC-AGI datasets from GitHub
instead of Kaggle, so that I can eliminate Kaggle CLI dependencies and use a
consistent download mechanism.

#### Acceptance Criteria

1. WHEN I run a download command for ARC-AGI-1 THEN the system SHALL clone the
   GitHub repository `https://github.com/fchollet/ARC-AGI`
2. WHEN I run a download command for ARC-AGI-2 THEN the system SHALL clone the
   GitHub repository `https://github.com/arcprize/ARC-AGI-2`
3. WHEN datasets are downloaded THEN the system SHALL organize them in the same
   directory structure as other GitHub datasets
4. WHEN download completes THEN the system SHALL validate the repository
   structure and provide clear success messages

### Requirement 2

**User Story:** As a user, I want the ARC-AGI parser to handle individual JSON
task files from GitHub, so that I can load tasks from the new data format
without breaking existing functionality.

#### Acceptance Criteria

1. WHEN the parser loads ARC-AGI-1 from GitHub THEN it SHALL read individual
   task files from `data/training/` and `data/evaluation/` directories
2. WHEN the parser loads ARC-AGI-2 from GitHub THEN it SHALL read individual
   task files from `data/training/` and `data/evaluation/` directories
3. WHEN processing individual task files THEN the parser SHALL maintain
   compatibility with existing `JaxArcTask` interface
4. WHEN task files are missing or malformed THEN the parser SHALL provide clear
   error messages with file-specific information

### Requirement 3

**User Story:** As a developer, I want all dataset downloading consolidated
under one unified API, so that I can use consistent commands and error handling
across all datasets.

#### Acceptance Criteria

1. WHEN I use the download script THEN it SHALL support all four datasets
   (ARC-AGI-1, ARC-AGI-2, ConceptARC, MiniARC) through GitHub
2. WHEN I run download commands THEN they SHALL use the same underlying
   `DatasetDownloader` class with consistent error handling
3. WHEN downloads fail THEN the system SHALL provide unified error messages and
   recovery suggestions
4. WHEN I download all datasets THEN the system SHALL use only GitHub
   repositories without Kaggle dependencies

### Requirement 7

**User Story:** As a user, I want a streamlined and straightforward CLI
interface for dataset downloading, so that I can easily download datasets
without complex command structures.

#### Acceptance Criteria

1. WHEN I use the download script THEN it SHALL provide simple, intuitive
   commands for each dataset
2. WHEN I run help commands THEN they SHALL show clear usage examples and
   descriptions
3. WHEN I specify download options THEN the CLI SHALL use consistent parameter
   names and behaviors
4. WHEN I make mistakes THEN the CLI SHALL provide helpful error messages and
   suggest corrections

### Requirement 4

**User Story:** As a user, I want updated configuration files that point to
GitHub data structures, so that I can use the new dataset locations without
manual configuration changes.

#### Acceptance Criteria

1. WHEN I use ARC-AGI-1 configuration THEN it SHALL point to GitHub repository
   data paths with individual task files
2. WHEN I use ARC-AGI-2 configuration THEN it SHALL point to GitHub repository
   data paths with individual task files
3. WHEN configurations are loaded THEN they SHALL maintain backward
   compatibility with existing environment setup
4. WHEN data paths are invalid THEN the system SHALL provide clear validation
   errors with suggested fixes

### Requirement 5

**User Story:** As a developer, I want to remove all Kaggle-related code and
dependencies, so that I can simplify the codebase and eliminate external CLI
tool requirements.

#### Acceptance Criteria

1. WHEN the migration is complete THEN the system SHALL not require Kaggle CLI
   installation
2. WHEN I review the codebase THEN it SHALL not contain Kaggle-specific download
   functions or imports
3. WHEN I check dependencies THEN the system SHALL not include `kaggle` package
   requirements
4. WHEN I run tests THEN they SHALL not depend on Kaggle API credentials or
   network access to Kaggle

### Requirement 6

**User Story:** As a user, I want clear migration documentation and examples, so
that I can understand the changes and update my workflows accordingly.

#### Acceptance Criteria

1. WHEN I read the documentation THEN it SHALL explain the differences between
   Kaggle and GitHub data formats
2. WHEN I follow migration guides THEN they SHALL provide step-by-step
   instructions for updating existing setups
3. WHEN I run examples THEN they SHALL demonstrate the new GitHub-based dataset
   loading
4. WHEN I encounter issues THEN the documentation SHALL provide troubleshooting
   guidance for common problems
