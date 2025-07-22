# JaxARC Testing Infrastructure

This directory contains the comprehensive testing infrastructure for the JaxARC
project, focusing on JAX-compatible testing patterns, Equinox module testing,
and property-based testing with Hypothesis. The test suite has been completely
overhauled to align with the current Equinox-based architecture and JAXTyping
system.

## Test Organization

The test structure mirrors the source code organization for easy navigation:

```
tests/
├── test_types.py                    # Core types (Grid, JaxArcTask, ARCLEAction)
├── test_state.py                    # ArcEnvState testing
├── envs/                           # Environment system tests
│   ├── test_actions.py             # Action handlers
│   ├── test_arc_base.py            # Base environment functionality
│   ├── test_config_validation.py   # Configuration validation
│   ├── test_equinox_config.py      # Unified Equinox configuration
│   ├── test_factory.py             # Configuration factory functions
│   ├── test_functional.py          # Pure functional API
│   ├── test_grid_operations.py     # Grid operations and ARCLE ops
│   ├── test_operations.py          # Core operations
│   └── test_spaces.py              # Action/observation spaces
├── parsers/                        # Parser system tests
│   ├── test_arc_agi_comprehensive.py      # ARC-AGI parser
│   ├── test_base_parser_comprehensive.py  # Base parser functionality
│   ├── test_concept_arc_comprehensive.py  # ConceptARC parser
│   ├── test_mini_arc_comprehensive.py     # MiniARC parser
│   ├── test_parser_utils_comprehensive.py # Parser utilities
│   └── test_utils.py               # Parser utility functions
└── utils/                          # Utility tests
    ├── test_config.py              # Configuration utilities
    ├── test_dataset_downloader.py  # Dataset management
    ├── test_dataset_validation.py  # Dataset validation
    ├── test_grid_utils.py          # Grid manipulation utilities
    ├── test_jax_types.py           # JAXTyping definitions
    ├── test_task_manager.py        # Task management
    └── visualization/
        └── test_visualization.py   # Visualization utilities
```

## Core Testing Utilities

### JAX Testing Framework

- **`jax_test_framework.py`**: Comprehensive framework for testing JAX
  transformations (jit, vmap, pmap)
- **`jax_testing_utils.py`**: Specialized utilities for JAX-compatible testing
- **`hypothesis_utils.py`**: Property-based testing utilities using Hypothesis
  for array operations
- **`test_utils.py`**: Common testing utilities and mock data generators
- **`equinox_test_utils.py`**: Utilities for testing Equinox modules and PyTree
  compatibility

### Fixtures and Configuration

- **`conftest.py`**: Pytest fixtures and configuration for JAX testing with
  consistent PRNG keys

## Key Testing Features

### JAX Transformation Testing

Test that functions work correctly with JAX transformations (jit, vmap, pmap):

```python
from tests.jax_test_framework import run_jax_transformation_tests


def test_my_function():
    def simple_func(x):
        return x * 2

    test_inputs = (jnp.array([1, 2, 3], dtype=jnp.float32),)

    # Tests jit, vmap, and pmap compatibility
    run_jax_transformation_tests(simple_func, test_inputs)
```

### Equinox Module Testing

Test Equinox modules for JAX compatibility and PyTree structure:

```python
from tests.equinox_test_utils import run_equinox_module_tests


def test_my_module():
    valid_args = {"in_size": 3, "out_size": 2}

    # Tests module creation, JAX transformations, and PyTree compatibility
    run_equinox_module_tests(MyModule, valid_args)
```

### Property-Based Testing with Hypothesis

Test properties of JAX functions using Hypothesis for comprehensive array
testing:

```python
from tests.hypothesis_utils import test_jax_function_properties, jax_arrays


def test_my_function_properties():
    def check_shape_property(func, input_data, result):
        assert input_data.shape == result.shape

    test_jax_function_properties(
        my_function, jax_arrays(dtype=jnp.float32), [check_shape_property]
    )
```

### Mock Data Generation

Generate JAX-compatible mock data for testing:

```python
from tests.test_utils import MockDataGenerator


def test_with_mock_data():
    # Create mock grid data with proper JAX arrays
    data, mask = MockDataGenerator.create_mock_grid(5, 5)

    # Create mock task data with static shapes
    task_data = MockDataGenerator.create_mock_task_data(
        num_train_pairs=2, num_test_pairs=1
    )
```

### JAXTyping Validation

Test JAXTyping annotations and runtime validation:

```python
import chex
from jaxarc.utils.jax_types import Grid2D, TaskArray


def test_jaxtyping_validation():
    # Test shape validation
    valid_grid = jnp.ones((10, 10), dtype=jnp.int32)
    chex.assert_type(valid_grid, Grid2D)

    # Test invalid shapes (should raise)
    with pytest.raises(TypeError):
        invalid_grid = jnp.ones((5,), dtype=jnp.int32)
        chex.assert_type(invalid_grid, Grid2D)
```

## Common Fixtures

Available in `conftest.py` for consistent testing across the suite:

- **`jax_key`**: Consistent JAX PRNG key (seed=42) for reproducible tests
- **`split_key`**: Function to split JAX keys consistently
- **`mock_grid`**, **`mock_task`**, **`mock_action`**: Pre-configured mock
  objects with proper JAX types
- **`small_grid_shape`**, **`medium_grid_shape`**, **`large_grid_shape`**:
  Standard grid shapes (5x5, 10x10, 30x30)
- **`mock_config`**: Standard configuration objects for testing
- **`mock_state`**: Pre-initialized environment states

## JAX-Specific Testing Guidelines

### 1. JAX Transformation Testing

Always test that functions work with JAX transformations:

```python
def test_function_jax_compatibility():
    # Test JIT compilation
    jitted_fn = jax.jit(my_function)
    result = jitted_fn(inputs)

    # Test vectorization
    vmapped_fn = jax.vmap(my_function)
    batch_result = vmapped_fn(batch_inputs)

    # Test parallel mapping (if applicable)
    pmapped_fn = jax.pmap(my_function)
    parallel_result = pmapped_fn(parallel_inputs)
```

### 2. Equinox Module Testing

Test Equinox modules for proper PyTree structure and JAX compatibility:

```python
def test_equinox_module():
    module = MyModule(param1=value1, param2=value2)

    # Test PyTree structure
    assert eqx.is_array_like(module)

    # Test JAX transformations
    jitted_module = jax.jit(lambda x: x)(module)
    assert eqx.tree_equal(module, jitted_module)

    # Test serialization
    leaves, treedef = jax.tree_util.tree_flatten(module)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert eqx.tree_equal(module, reconstructed)
```

### 3. Static Shape Requirements

Ensure all functions work with JAX's static shape requirements:

```python
def test_static_shapes():
    # Use fixed shapes for all arrays
    fixed_shape_array = jnp.ones((10, 10), dtype=jnp.int32)

    # Avoid dynamic shapes that break JIT
    # BAD: variable_shape = jnp.ones((n, n))  # where n is runtime variable
    # GOOD: padded_array = jnp.pad(array, pad_width, constant_values=0)
```

### 4. PRNG Key Management

Use explicit PRNG key management for reproducible tests:

```python
def test_with_prng_keys(jax_key):
    key1, key2 = jax.random.split(jax_key)

    # Use keys explicitly
    result1 = my_random_function(key1)
    result2 = my_random_function(key2)

    # Results should be different but reproducible
    assert not jnp.array_equal(result1, result2)
```

### 5. Property-Based Testing

Use Hypothesis for comprehensive testing of array operations:

```python
from hypothesis import given
from tests.hypothesis_utils import jax_arrays


@given(jax_arrays(shape=(10, 10), dtype=jnp.float32))
def test_function_properties(array):
    result = my_function(array)

    # Test invariants
    assert result.shape == array.shape
    assert result.dtype == array.dtype
    assert jnp.all(jnp.isfinite(result))
```

## Best Practices

1. **Use fixtures**: Leverage the provided fixtures for consistent testing
2. **Test JAX transformations**: Always test jit, vmap, and pmap compatibility
3. **Property-based testing**: Use Hypothesis for thorough testing of array
   operations
4. **Static shapes**: Ensure functions work with JAX's static shape requirements
5. **Explicit PRNG**: Use explicit PRNG key management for reproducible
   randomness
6. **Equinox validation**: Test PyTree structure and serialization for Equinox
   modules
7. **Type checking**: Use `chex.assert_type` for JAXTyping validation
8. **Performance testing**: Include basic performance regression tests for
   critical paths

## Running Tests

```bash
# Run all tests with coverage
pixi run -e test test --cov=src/jaxarc --cov-report=html

# Run specific test categories
pixi run -e test test tests/test_types.py          # Core types
pixi run -e test test tests/envs/                  # Environment tests
pixi run -e test test tests/parsers/               # Parser tests
pixi run -e test test tests/utils/                 # Utility tests

# Run with verbose output and show local variables on failure
pixi run -e test test -v --tb=long

# Run performance regression tests
pixi run -e test test tests/test_performance_regression.py

# Run integration tests
pixi run -e test test tests/test_integration_basic.py
```

## Test Coverage Goals

- **Core Types**: 100% coverage of public API
- **Environment System**: 95% coverage including error paths
- **Parsers**: 90% coverage with focus on data validation
- **Utilities**: 85% coverage with focus on public functions
- **Overall Project**: Target 90%+ coverage

## Debugging Failed Tests

When tests fail, use these debugging strategies:

1. **JAX debugging**: Use `jax.debug.print()` and `jax.debug.callback()` for
   debugging inside JIT
2. **Array inspection**: Use `chex.assert_shape()` and `chex.assert_type()` for
   validation
3. **Transformation debugging**: Test functions without transformations first,
   then add jit/vmap
4. **Mock data**: Use simpler mock data to isolate issues
5. **Hypothesis shrinking**: Let Hypothesis find minimal failing examples for
   property-based tests
