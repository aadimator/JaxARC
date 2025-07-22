# JaxARC Testing Infrastructure

This directory contains the testing infrastructure for the JaxARC project, focusing on JAX-compatible testing patterns, Equinox module testing, and property-based testing with Hypothesis.

## Core Testing Utilities

### JAX Testing Utilities

- **`jax_test_framework.py`**: Framework for testing JAX transformations (jit, vmap, pmap)
- **`jax_testing_utils.py`**: Specialized utilities for JAX-compatible testing
- **`hypothesis_utils.py`**: Property-based testing utilities using Hypothesis
- **`test_utils.py`**: Common testing utilities and helpers
- **`equinox_test_utils.py`**: Utilities for testing Equinox modules

### Fixtures and Configuration

- **`conftest.py`**: Pytest fixtures and configuration for JAX testing

## Key Features

### JAX Transformation Testing

Test that functions work correctly with JAX transformations:

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

Test Equinox modules for JAX compatibility:

```python
from tests.equinox_test_utils import run_equinox_module_tests

def test_my_module():
    valid_args = {
        "in_size": 3,
        "out_size": 2
    }
    
    run_equinox_module_tests(MyModule, valid_args)
```

### Property-Based Testing

Test properties of JAX functions using Hypothesis:

```python
from tests.hypothesis_utils import test_jax_function_properties, jax_arrays

def test_my_function_properties():
    def check_shape_property(func, input_data, result):
        assert input_data.shape == result.shape
    
    test_jax_function_properties(
        my_function,
        jax_arrays(dtype=jnp.float32),
        [check_shape_property]
    )
```

### Mock Data Generation

Generate mock data for testing:

```python
from tests.test_utils import MockDataGenerator

def test_with_mock_data():
    # Create mock grid data
    data, mask = MockDataGenerator.create_mock_grid(5, 5)
    
    # Create mock task data
    task_data = MockDataGenerator.create_mock_task_data(
        num_train_pairs=2, 
        num_test_pairs=1
    )
```

## Common Fixtures

- **`jax_key`**: Consistent JAX PRNG key for reproducible tests
- **`split_key`**: Function to split JAX keys consistently
- **`mock_grid`**, **`mock_task`**, **`mock_action`**: Pre-configured mock objects
- **`small_grid_shape`**, **`medium_grid_shape`**, **`large_grid_shape`**: Standard grid shapes

## Best Practices

1. **Use fixtures**: Leverage the provided fixtures for consistent testing
2. **Test JAX transformations**: Always test that functions work with JAX transformations
3. **Property-based testing**: Use Hypothesis for thorough testing of array operations
4. **Mock objects**: Use mock objects for testing without dependencies
5. **Static shapes**: Ensure functions work with JAX's static shape requirements

## Running Tests

```bash
# Run all tests
pixi run -e test test

# Run specific test file
pixi run -e test test tests/test_file.py

# Run with verbose output
pixi run -e test test -v
```