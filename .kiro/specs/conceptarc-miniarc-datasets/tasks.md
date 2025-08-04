# Implementation Plan

- [x] 1. Create ConceptARC parser implementation
  - Implement `ConceptArcParser` class extending `ArcDataParserBase`
  - Add concept group discovery and organization logic
  - Implement task loading from hierarchical directory structure
  - Add concept-based random sampling functionality
  - _Requirements: 1.1, 1.2, 1.3, 5.1, 5.3_

- [x] 2. Create MiniARC parser implementation
  - Implement `MiniArcParser` class extending `ArcDataParserBase`
  - Add 5x5 grid optimization and validation
  - Implement task loading from flat directory structure
  - Add performance optimizations for smaller grids
  - _Requirements: 2.1, 2.2, 2.3, 5.1, 5.3_

- [x] 3. Create ConceptARC configuration file
  - Write `conf/dataset/concept_arc.yaml` with concept group definitions
  - Configure parser target and grid dimensions for standard ARC
  - Set appropriate task limits for ConceptARC characteristics
  - Include dataset metadata and descriptions
  - _Requirements: 1.1, 6.1, 6.3_

- [x] 4. Create MiniARC configuration file
  - Write `conf/dataset/mini_arc.yaml` optimized for 5x5 grids
  - Configure parser target with MiniARC-specific settings
  - Set task limits appropriate for smaller dataset
  - Include optimization flags and metadata
  - _Requirements: 2.1, 2.2, 6.2, 6.3_

- [x] 5. Implement dataset download functionality
  - Create `DatasetDownloader` class with Git repository cloning
  - Add `download_conceptarc()` method for ConceptARC repository
  - Add `download_miniarc()` method for MiniARC repository
  - Implement error handling for network and filesystem issues
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 6. Enhance download script CLI interface
  - Add `download-conceptarc` command to download script
  - Add `download-miniarc` command to download script
  - Implement command-line argument parsing for dataset selection
  - Add help text and usage examples for new commands
  - _Requirements: 3.1, 3.2, 4.1, 4.3_

- [x] 7. Update parser module exports
  - Add `ConceptArcParser` to `src/jaxarc/parsers/__init__.py`
  - Add `MiniArcParser` to `src/jaxarc/parsers/__init__.py`
  - Update `__all__` list with new parser classes
  - Ensure proper import structure for new parsers
  - _Requirements: 5.1, 5.3_

- [x] 8. Create configuration factory functions
  - Implement `create_conceptarc_config()` factory function
  - Implement `create_miniarc_config()` factory function
  - Add configuration validation and error handling
  - Include factory functions in configuration utilities module
  - _Requirements: 4.1, 4.2, 6.1, 6.2_

- [x] 9. Write comprehensive unit tests for ConceptARC parser
  - Test concept group discovery and organization
  - Test task loading from hierarchical structure
  - Test concept-based random sampling
  - Test error handling for missing concept groups
  - _Requirements: 1.1, 1.2, 1.3, 5.2_

- [x] 10. Write comprehensive unit tests for MiniARC parser
  - Test 5x5 grid constraint validation
  - Test task loading from flat directory structure
  - Test performance optimizations
  - Test error handling for oversized grids
  - _Requirements: 2.1, 2.2, 2.3, 5.2_

- [x] 11. Write integration tests for download functionality
  - Test ConceptARC repository cloning (mocked)
  - Test MiniARC repository cloning (mocked)
  - Test error handling for network failures
  - Test directory structure validation after download
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 12. Write configuration validation tests
  - Test ConceptARC configuration loading and validation
  - Test MiniARC configuration loading and validation
  - Test factory function behavior
  - Test error handling for invalid configurations
  - _Requirements: 4.1, 4.2, 4.4, 6.4_

- [x] 13. Create usage examples for ConceptARC
  - Write example script demonstrating ConceptARC usage
  - Show concept group exploration and task sampling
  - Demonstrate integration with existing environment
  - Include visualization of concept-based tasks
  - _Requirements: 1.1, 1.2, 6.1_

- [x] 14. Create usage examples for MiniARC
  - Write example script demonstrating MiniARC usage
  - Show 5x5 grid optimization benefits
  - Demonstrate rapid prototyping workflow
  - Include performance comparison with standard ARC
  - _Requirements: 2.1, 2.2, 6.2_

- [x] 15. Update project documentation
  - Add ConceptARC and MiniARC sections to README
  - Update configuration documentation with new datasets
  - Add download instructions for new datasets
  - Update API documentation with new parser classes
  - _Requirements: 4.3, 6.3, 6.4_