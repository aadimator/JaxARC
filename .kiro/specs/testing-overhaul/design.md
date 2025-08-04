# Design Document

## Overview

The testing overhaul will completely restructure the test suite to align with the current JaxARC codebase, which has evolved significantly from its original state. The design focuses on creating a clean, focused test suite that validates the current Equinox-based architecture, JAXTyping system, and functional API while eliminating outdated and duplicate tests.

## Architecture

### Current Codebase Analysis

Based on the source code analysis, the current JaxARC architecture consists of:

1. **Core Types** (`src/jaxarc/types.py`):
   - `Grid` (Equinox Module with JAXTyping)
   - `JaxArcTask` (Equinox Module with fixed-size arrays)
   - `ARCLEAction` (Equinox Module with continuous selection)
   - `TaskPair` (Equinox Module for input-output pairs)

2. **Environment System** (`src/jaxarc/envs/`):
   - `ArcEnvironment` (main environment class)
   - Functional API (`arc_reset`, `arc_step`)
   - Configuration system (legacy + unified Equinox-based)
   - Action handlers (point, bbox, mask)
   - Grid operations (35 ARCLE operations)

3. **State Management** (`src/jaxarc/state.py`):
   - `ArcEnvState` (centralized Equinox Module)

4. **Parsers** (`src/jaxarc/parsers/`):
   - `ArcAgiParser`, `ConceptArcParser`, `MiniArcParser`
   - Base parser class with common functionality

5. **Utilities** (`src/jaxarc/utils/`):
   - JAXTyping definitions
   - Grid utilities
   - Configuration utilities
   - Dataset management
   - Task management

### Test Organization Strategy

The new test structure will mirror the source structure exactly:

```
tests/
├── test_types.py                    # Core types (Grid, JaxArcTask, ARCLEAction)
├── test_state.py                    # ArcEnvState testing
├── envs/
│   ├── test_environment.py          # ArcEnvironment class
│   ├── test_functional.py           # arc_reset, arc_step functions
│   ├── test_config.py               # Legacy configuration system
│   ├── test_equinox_config.py       # Unified Equinox configuration
│   ├── test_factory.py              # Configuration factory functions
│   ├── test_actions.py              # Action handlers
│   ├── test_grid_operations.py      # Grid operations and ARCLE ops
│   └── test_spaces.py               # Action/observation spaces
├── parsers/
│   ├── test_base_parser.py          # Base parser functionality
│   ├── test_arc_agi.py              # ARC-AGI parser
│   ├── test_concept_arc.py          # ConceptARC parser
│   ├── test_mini_arc.py             # MiniARC parser
│   └── test_utils.py                # Parser utilities
└── utils/
    ├── test_jax_types.py            # JAXTyping definitions
    ├── test_grid_utils.py           # Grid manipulation utilities
    ├── test_config.py               # Configuration utilities
    ├── test_dataset_downloader.py   # Dataset management
    ├── test_task_manager.py         # Task management
    └── visualization/
        └── test_visualization.py    # Visualization utilities
```

## Components and Interfaces

### Test Categories

1. **Core Type Tests** (`test_types.py`, `test_state.py`):
   - Equinox Module validation
   - JAXTyping annotation compliance
   - JAX transformation compatibility (jit, vmap, pmap)
   - Shape and type validation
   - Initialization and validation methods

2. **Environment Tests** (`envs/`):
   - Environment lifecycle (reset, step, termination)
   - Functional API correctness
   - Configuration system validation
   - Action processing and validation
   - Grid operations and ARCLE compliance
   - Reward computation
   - State transitions

3. **Parser Tests** (`parsers/`):
   - Data loading and validation
   - JaxArcTask creation
   - Format compatibility
   - Error handling
   - Integration with different dataset formats

4. **Utility Tests** (`utils/`):
   - JAXTyping system validation
   - Grid manipulation functions
   - Configuration loading and validation
   - Dataset management
   - Task management system
   - Visualization functions

### JAX Compatibility Testing

Each test module will include JAX-specific validation:

```python
# Example test structure for JAX compatibility
def test_jax_transformations():
    """Test that modules work with JAX transformations."""
    # Test jit compilation
    jitted_fn = jax.jit(some_function)
    result = jitted_fn(inputs)
    
    # Test vmap batching
    vmapped_fn = jax.vmap(some_function)
    batch_result = vmapped_fn(batch_inputs)
    
    # Test pmap (if applicable)
    pmapped_fn = jax.pmap(some_function)
    parallel_result = pmapped_fn(parallel_inputs)
```

### Configuration Testing Strategy

The design addresses the dual configuration system:

1. **Legacy Configuration Tests**: Validate existing `ArcEnvConfig` and factory functions
2. **Unified Configuration Tests**: Validate new `JaxArcConfig` Equinox-based system
3. **Conversion Tests**: Validate conversion between legacy and unified configs
4. **Integration Tests**: Ensure both systems work with the environment

## Data Models

### Test Data Management

1. **Minimal Test Data**: Create small, focused test datasets for each parser
2. **Mock Objects**: Use JAX-compatible mock objects for unit tests
3. **Fixture System**: Pytest fixtures for common test objects
4. **Property-Based Testing**: Use Hypothesis for JAX array property testing

### Test Validation Patterns

```python
# Standard validation pattern for Equinox modules
def validate_equinox_module(module, expected_types, expected_shapes):
    """Validate Equinox module structure and JAX compatibility."""
    # Check module is PyTree
    assert eqx.is_array_like(module)
    
    # Validate field types and shapes
    for field_name, expected_type in expected_types.items():
        field_value = getattr(module, field_name)
        assert isinstance(field_value, expected_type)
    
    # Test JAX transformations
    jitted_module = jax.jit(lambda x: x)(module)
    assert eqx.tree_equal(module, jitted_module)
```

## Error Handling

### Test Error Categories

1. **Validation Errors**: Invalid inputs, shape mismatches, type errors
2. **JAX Transformation Errors**: JIT compilation failures, shape inference issues
3. **Configuration Errors**: Invalid config combinations, missing required fields
4. **Parser Errors**: Malformed data, missing files, format incompatibilities
5. **Environment Errors**: Invalid actions, state transition failures

### Error Testing Strategy

Each module will include comprehensive error testing:
- Invalid input validation
- Edge case handling
- JAX-specific error conditions
- Configuration validation errors
- Resource availability errors

## Testing Strategy

### Test Execution Approach

1. **Unit Tests**: Individual function and class testing
2. **Integration Tests**: Component interaction testing
3. **JAX Compatibility Tests**: Transformation and compilation testing
4. **Property-Based Tests**: Hypothesis-driven testing for array operations
5. **Performance Tests**: Basic performance regression testing

### Test Data Strategy

1. **Synthetic Data**: Generated test grids and tasks
2. **Minimal Real Data**: Small subsets of actual ARC data
3. **Mock Objects**: JAX-compatible mocks for external dependencies
4. **Parameterized Tests**: Multiple configurations and scenarios

### Coverage Goals

- **Core Types**: 100% coverage of public API
- **Environment**: 95% coverage including error paths
- **Parsers**: 90% coverage with focus on data validation
- **Utilities**: 85% coverage with focus on public functions

## Implementation Plan

### Phase 1: Cleanup and Analysis
1. Analyze existing tests for relevance
2. Identify obsolete and duplicate tests
3. Remove outdated test files and cache
4. Create test inventory and mapping

### Phase 2: Core Infrastructure
1. Create new test structure
2. Implement common test utilities and fixtures
3. Set up JAX compatibility testing framework
4. Create mock objects and test data

### Phase 3: Core Tests Implementation
1. Implement type system tests
2. Implement state management tests
3. Implement basic environment tests
4. Validate JAX transformation compatibility

### Phase 4: Component Tests
1. Implement parser tests
2. Implement utility tests
3. Implement configuration tests
4. Implement visualization tests

### Phase 5: Integration and Validation
1. Implement integration tests
2. Validate test coverage
3. Performance and regression testing
4. Documentation and cleanup

## Dependencies

### Testing Dependencies
- `pytest`: Test framework
- `pytest-cov`: Coverage reporting
- `hypothesis`: Property-based testing
- `chex`: JAX testing utilities
- `jax`: Core JAX functionality

### Mock and Fixture Dependencies
- Custom JAX-compatible mock objects
- Pytest fixtures for common objects
- Test data generators for grids and tasks
- Configuration builders for different scenarios