# Implementation Plan

- [x] 1. Update ArcAgiParser to support GitHub format

  - Replace Kaggle format loading logic with individual JSON file loading
  - Update `_load_and_cache_tasks()` method to read from directory of JSON files
  - Modify task loading to use filename as task ID
  - Remove challenges/solutions merging logic
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 2. Add ARC-AGI dataset download methods to DatasetDownloader

  - Implement `download_arc_agi_1()` method for fchollet/ARC-AGI repository
  - Implement `download_arc_agi_2()` method for arcprize/ARC-AGI-2 repository
  - Add validation for ARC-AGI dataset structure in `_validate_download()`
  - Update repository cloning to handle both ARC-AGI repositories
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 3. Update ARC-AGI configuration files for GitHub format

  - Modify `conf/dataset/arc_agi_1.yaml` to use directory paths instead of file
    paths
  - Modify `conf/dataset/arc_agi_2.yaml` to use directory paths instead of file
    paths
  - Remove challenges/solutions file references
  - Update data_root paths to point to GitHub repository structure
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 4. Create streamlined download script CLI interface

  - Replace existing Kaggle commands with simple dataset-specific commands
  - Add `arc-agi-1`, `arc-agi-2`, `conceptarc`, `miniarc` commands
  - Add `all` command to download all datasets
  - Implement consistent parameter handling across all commands
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 5. Remove Kaggle dependencies and code

  - Remove `kaggle` package from requirements
  - Remove Kaggle CLI download functions from download script
  - Remove Kaggle-specific error handling and validation
  - Clean up imports and unused Kaggle-related code
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 6. Update parser module exports and imports

  - Ensure `ArcAgiParser` exports remain unchanged for compatibility
  - Update any internal imports that reference Kaggle-specific functionality
  - Verify parser interface remains consistent for existing users
  - Test parser instantiation with new configuration format
  - _Requirements: 2.1, 2.2, 4.3_

- [x] 7. Write unit tests for GitHub format parsing

  - Test individual JSON file loading functionality
  - Test task ID extraction from filenames
  - Test error handling for missing or malformed JSON files
  - Test directory structure validation
  - _Requirements: 2.2, 2.4_

- [x] 8. Write unit tests for new download functionality

  - Test ARC-AGI-1 repository cloning (mocked)
  - Test ARC-AGI-2 repository cloning (mocked)
  - Test dataset structure validation
  - Test error handling for network and filesystem issues
  - _Requirements: 1.1, 1.2, 1.4_

- [x] 9. Write integration tests for end-to-end workflow

  - Test complete download and parsing workflow for ARC-AGI-1
  - Test complete download and parsing workflow for ARC-AGI-2
  - Test CLI interface with new commands
  - Test configuration loading with new format
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 10. Create usage examples for GitHub-based datasets

  - Update existing examples to use GitHub format
  - Create example scripts demonstrating ARC-AGI-1 usage
  - Create example scripts demonstrating ARC-AGI-2 usage
  - Show performance comparisons and benefits of GitHub format
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 11. Update project documentation

  - Update README with new download instructions
  - Create migration guide from Kaggle to GitHub format
  - Update API documentation for parser changes
  - Add troubleshooting section for common GitHub download issues
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 12. Validate GitHub dataset integrity
  - Test that all expected tasks are loaded from GitHub repositories
  - Verify task data structure matches expected JaxArcTask format
  - Test parser performance with large datasets
  - Ensure no data corruption during JSON file loading
  - _Requirements: 2.3, 4.3_
