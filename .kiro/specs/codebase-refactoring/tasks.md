# Implementation Plan

## Phase 1: Foundation - Eliminate Code Duplication and Centralize Types

- [x] 1. Create centralized JAXTyping definitions

  - Create `src/jaxarc/utils/jax_types.py` with precise array type aliases
  - Define GridArray, MaskArray, SelectionArray, and other core types
  - Add batch types for vectorized operations
  - _Requirements: 5.4, 5.5, 8.1_

- [x] 2. Move ArcEnvState to centralized location

  - Remove duplicate ArcEnvState definition from `src/jaxarc/envs/arc_base.py`
  - Remove duplicate ArcEnvState definition from `src/jaxarc/envs/functional.py`
  - Create `src/jaxarc/state.py` with single ArcEnvState definition using
    JAXTyping
  - Update all imports to use centralized state definition
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 3. Update core types with JAXTyping annotations

  - Add JAXTyping annotations to Grid dataclass in `src/jaxarc/types.py`
  - Add JAXTyping annotations to JaxArcTask dataclass
  - Add JAXTyping annotations to ARCLEAction dataclass
  - Update validation logic to leverage JAXTyping
  - _Requirements: 5.5, 8.1, 8.2_

- [x] 4. Simplify arc_step function to use action handlers

  - Remove complex validation logic from `arc_step` in
    `src/jaxarc/envs/functional.py`
  - Implement clean delegation to action handlers from
    `src/jaxarc/envs/actions.py`
  - Remove redundant action transformation functions
  - Ensure standardized selection mask creation through handlers
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 5. Add comprehensive tests for foundation changes
  - Write tests for centralized state definition
  - Write tests for JAXTyping type validation
  - Write tests for simplified arc_step function
  - Ensure backward compatibility with existing code
  - _Requirements: 8.3, 8.4_

## Phase 2: Parser Consolidation - Eliminate Parser Duplication

- [x] 6. Move common parser methods to base class

  - Identify all duplicate methods across `ArcAgiParser`, `ConceptArcParser`,
    `MiniArcParser`
  - Move `_process_training_pairs` method to `ArcDataParserBase`
  - Move `_pad_and_create_masks` method to `ArcDataParserBase`
  - Move `_validate_grid_colors` method to `ArcDataParserBase`
  - _Requirements: 3.1, 3.2_

- [x] 7. Update specific parsers to use inheritance

  - Refactor `ArcAgiParser` to call `super()._process_training_pairs()`
  - Refactor `ConceptArcParser` to call `super()._pad_and_create_masks()`
  - Refactor `MiniArcParser` to call `super()._validate_grid_colors()`
  - Remove duplicate method implementations from specific parsers
  - _Requirements: 3.3, 3.4, 3.5_

- [x] 8. Add comprehensive parser tests
  - Write tests to ensure parser refactoring maintains functionality
  - Add tests for base class method implementations
  - Add tests for inheritance behavior in specific parsers
  - Verify no regression in data loading capabilities
  - _Requirements: 3.1, 3.2_

## Phase 3: Equinox Integration - Modernize State Management

- [x] 9. Create Equinox-based state management

  - Install and configure Equinox library in project dependencies
  - Create `src/jaxarc/utils/equinox_utils.py` with integration utilities
  - Convert ArcEnvState from chex dataclass to Equinox Module
  - Implement automatic PyTree registration and validation
  - _Requirements: 5.1, 5.2, 5.6_

- [x] 10. Add Equinox utility functions

  - Implement `tree_map_with_path` function for enhanced tree operations
  - Implement `validate_state_shapes` function for state validation
  - Implement `create_state_diff` function for debugging support
  - Add comprehensive documentation for Equinox utilities
  - _Requirements: 5.3, 5.6_

- [x] 11. Migrate state management patterns

  - Update state creation and modification patterns to use Equinox
  - Ensure JAX transformations (jit, vmap, pmap) work with Equinox modules
  - Update state replacement operations to use Equinox patterns
  - Verify performance is maintained or improved with Equinox
  - _Requirements: 5.1, 5.2, 5.7_

- [x] 12. Add Equinox integration tests
  - Write tests for Equinox module functionality
  - Write tests for JAX transformation compatibility
  - Write performance benchmarks comparing old vs new state management
  - Ensure backward compatibility during transition
  - _Requirements: 5.6, 5.7_

## Phase 4: Configuration Simplification - Leverage Hydra Fully

- [x] 13. Evaluate Hydra preset configurations (DECIDED AGAINST)

  - Explored creating preset configurations to replace factory functions
  - Determined that Hydra's native composition system is sufficient and more
    flexible
  - Removed preset system to avoid unnecessary complexity and maintain
    modularity
  - Kept existing modular configuration files (action, reward, environment,
    dataset)
  - Users can easily compose configurations using Hydra's override system
  - _Requirements: 4.1, 4.2, 4.4_

- [ ] 14. Refactor configuration factory functions (SIMPLIFIED APPROACH)

  - Keep existing factory functions for backward compatibility
  - Document migration path to Hydra's native composition system
  - Add examples showing how to replace factory functions with Hydra overrides
  - Encourage users to use modular Hydra configuration instead of factory
    functions
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 15. Enhance structured config validation

  - Add comprehensive validation to configuration dataclasses
  - Implement field validators for configuration parameters
  - Add clear error messages for invalid configuration combinations
  - Ensure runtime validation catches configuration errors early
  - _Requirements: 4.3, 4.5, 8.2, 8.5_

- [ ] 16. Add configuration system tests
  - Write tests for Hydra modular composition system
  - Write tests for refactored factory functions
  - Write tests for enhanced configuration validation
  - Ensure all configuration combinations work correctly
  - _Requirements: 4.1, 4.2, 4.5_

## Phase 5: Type Safety Enhancement - Add Runtime Validation

- [x] 17. Add comprehensive JAXTyping annotations

  - Add JAXTyping annotations to all grid operation functions
  - Add JAXTyping annotations to all environment functions
  - Add JAXTyping annotations to all parser functions
  - Add JAXTyping annotations to all utility functions
  - _Requirements: 5.5, 8.1, 8.3_

- [ ] 18. Implement runtime type checking

  - Install and configure beartype for runtime type validation
  - Add @jaxtyped decorators to critical functions
  - Configure runtime type checking for development and testing
  - Add comprehensive error handling for type validation failures
  - _Requirements: 8.2, 8.4, 8.5_

- [ ] 19. Add property-based testing

  - Install and configure hypothesis for property-based testing
  - Write property tests for grid operations shape preservation
  - Write property tests for state transitions consistency
  - Write property tests for configuration validation
  - _Requirements: 8.1, 8.3, 8.4_

- [ ] 20. Add comprehensive type safety tests
  - Write tests for JAXTyping annotation validation
  - Write tests for runtime type checking behavior
  - Write tests for property-based testing coverage
  - Ensure type safety improvements catch real errors
  - _Requirements: 8.1, 8.2, 8.5_

## Phase 6: Code Organization - Final Cleanup and Optimization

- [x] 21. Reorganize modules for clarity

  - Ensure each module has single, clear responsibility
  - Update import patterns to be consistent across all modules
  - Group utility functions logically in appropriate modules
  - Separate core functionality from configuration and utility code
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 22. Modernize factory functions (OPTIONAL)

  - Keep factory functions for backward compatibility but keep them minimal, and
    they should load mostly from Hydra Configs
  - Add deprecation warnings encouraging Hydra composition
  - Provide migration guide showing Hydra alternatives
  - Update examples to demonstrate both approaches
  - _Requirements: 4.4, 7.5_

- [x] 23. Add comprehensive documentation

  - Update API documentation with new patterns and types
  - Create migration guide for users upgrading from old patterns
  - Add examples demonstrating new Equinox and JAXTyping usage
  - Update configuration documentation with Hydra composition examples
  - _Requirements: 6.5, 8.5_

- [ ] 24. Final optimization and cleanup
  - Remove all duplicate code identified during refactoring
  - Optimize performance bottlenecks discovered during migration
  - Add final validation that all requirements are met
  - Ensure comprehensive test coverage for all refactored code
  - _Requirements: 1.1, 6.1, 7.4, 8.4_

## Phase 7: Validation and Documentation

- [ ] 25. Comprehensive integration testing

  - Run full test suite to ensure no regressions
  - Test all JAX transformations (jit, vmap, pmap) with new code
  - Benchmark performance to ensure improvements or no degradation
  - Test backward compatibility with existing user code
  - _Requirements: 5.6, 5.7, 8.3, 8.4_

- [ ] 26. Update examples and demos

  - Update all example scripts to use new patterns
  - Update demo notebooks to showcase new capabilities
  - Create examples demonstrating Equinox and JAXTyping benefits
  - Ensure all examples run successfully with refactored code
  - _Requirements: 6.5, 8.5_

- [ ] 27. Create migration documentation

  - Write step-by-step migration guide for existing users
  - Document breaking changes and how to address them
  - Provide before/after code examples for common patterns
  - Create troubleshooting guide for migration issues
  - _Requirements: 6.5, 8.5_

- [ ] 28. Final validation and release preparation
  - Verify all requirements have been implemented successfully
  - Run comprehensive test suite including new property-based tests
  - Validate that code quality metrics have improved as expected
  - Prepare release notes documenting all improvements and changes
  - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1_
