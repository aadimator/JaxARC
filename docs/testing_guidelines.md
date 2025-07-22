# JaxARC Testing Guidelines

This document provides comprehensive guidelines for testing in the JaxARC
project, focusing on JAX-compatible testing patterns, Equinox module testing,
and best practices for maintaining a robust test suite.

## Overview

The JaxARC testing infrastructure is designed around JAX's functional
programming paradigm and Equinox's PyTree-based modules. All tests must be
compatible with JAX transformations (jit, vmap, pmap) and follow functional
programming principles.

## Core Testing Principles

### 1. JAX-First Testing

All tests must consider JAX's requirements:

- **Static shapes**: Use fixed array shapes that work with JIT compilation
- **Pure functions**: Test functions without side effects
- **Explicit randomness**: Use JAX PRNG keys explicitly
- **Transformation compatibility**: Verify jit, vmap, and pmap work correctly

### 2. Equinox Module Testing

Equinox modules require special testing considerations:

- **PyTree structure**: Verify modules are proper PyTrees
- **Serialization**: Test tree flattening and unflattening
- **JAX compatibility**: Ensure modules work with JAX transformations
- **Initialization**: Test `__check_init__` methods work correctly

### 3. Property-Based Testing

Use Hypothesis for comprehensive testing:

- **Array properties**: Test invariants across different array inputs
- **Shape preservation**: Verify operations maintain expected shapes
- **Type consistency**: Ensure operations preserve or transform types correctly
- **Edge cases**: Let Hypothesis find edge cases automatically

## Testing Patterns

### JAX Transformation Testing

```python
import jax
import jax.numpy as jnp
from tests.jax_test_framework import run_jax_transformation_tests


def test_function_jax_compatibility():
    """Test that function works with all JAX transformations."""

    def my_function(x):
        return x * 2 + 1

    test_inputs = (jnp.array([1.0, 2.0, 3.0]),)

    # This tests jit, vmap, and pmap compatibility
    run_jax_transformation_tests(my_function, test_inputs)


def test_manual_jax_transformations():
    """Manual testing of JAX transformations."""

    def my_function(x):
        return jnp.sum(x**2)

    x = jnp.array([1.0, 2.0, 3.0])

    # Test JIT compilation
    jitted_fn = jax.jit(my_function)
    result_jit = jitted_fn(x)
    result_normal = my_function(x)
    assert jnp.allclose(result_jit, result_normal)

    # Test vectorization
    batch_x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    vmapped_fn = jax.vmap(my_function)
    batch_result = vmapped_fn(batch_x)
    assert batch_result.shape == (2,)
```

### Equinox Module Testing

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from tests.equinox_test_utils import run_equinox_module_tests


def test_equinox_module_comprehensive():
    """Comprehensive testing of Equinox module."""
    from jaxarc.types import Grid

    # Test module creation
    grid_data = jnp.ones((5, 5), dtype=jnp.int32)
    grid = Grid(data=grid_data)

    # Test PyTree structure
    assert eqx.is_array_like(grid)

    # Test serialization
    leaves, treedef = jax.tree_util.tree_flatten(grid)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert eqx.tree_equal(grid, reconstructed)

    # Test JAX transformations
    jitted_grid = jax.jit(lambda x: x)(grid)
    assert eqx.tree_equal(grid, jitted_grid)

    # Test with operations
    def grid_operation(g):
        return eqx.tree_at(lambda x: x.data, g, g.data * 2)

    modified_grid = grid_operation(grid)
    assert jnp.array_equal(modified_grid.data, grid_data * 2)


def test_equinox_module_with_framework():
    """Using the testing framework for Equinox modules."""
    from jaxarc.types import Grid

    valid_args = {"data": jnp.ones((5, 5), dtype=jnp.int32)}

    # This tests creation, JAX compatibility, and PyTree structure
    run_equinox_module_tests(Grid, valid_args)
```

### Property-Based Testing

```python
from hypothesis import given, strategies as st
import jax.numpy as jnp
from tests.hypothesis_utils import jax_arrays, test_jax_function_properties


@given(jax_arrays(shape=(10, 10), dtype=jnp.float32, min_value=-1.0, max_value=1.0))
def test_grid_operation_properties(grid_array):
    """Test properties of grid operations using Hypothesis."""
    from jaxarc.envs.grid_operations import normalize_grid

    result = normalize_grid(grid_array)

    # Test invariants
    assert result.shape == grid_array.shape
    assert result.dtype == grid_array.dtype
    assert jnp.all(result >= 0.0)
    assert jnp.all(result <= 1.0)


def test_function_properties_with_framework():
    """Using the testing framework for property-based testing."""

    def my_function(x):
        return jnp.clip(x, 0.0, 1.0)

    def check_clipping_property(func, input_data, result):
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)
        assert result.shape == input_data.shape

    test_jax_function_properties(
        my_function, jax_arrays(dtype=jnp.float32), [check_clipping_property]
    )
```

### Configuration Testing

```python
import pytest
from jaxarc.envs.config import ArcEnvConfig
from jaxarc.envs.factory import create_standard_config


def test_config_validation():
    """Test configuration validation and factory functions."""
    # Test valid configuration
    config = create_standard_config()
    assert isinstance(config, ArcEnvConfig)
    assert config.max_episode_steps > 0
    assert config.grid_size > 0

    # Test invalid configuration
    with pytest.raises(ValueError):
        ArcEnvConfig(max_episode_steps=-1, grid_size=30)  # Invalid


def test_config_serialization():
    """Test configuration serialization for JAX compatibility."""
    config = create_standard_config()

    # Test that config can be used in JAX functions
    def use_config(cfg):
        return cfg.max_episode_steps * 2

    jitted_fn = jax.jit(use_config)
    result = jitted_fn(config)
    assert result == config.max_episode_steps * 2
```

### Parser Testing

```python
import pytest
from pathlib import Path
from jaxarc.parsers import ArcAgiParser
from jaxarc.types import JaxArcTask


def test_parser_functionality():
    """Test parser data loading and validation."""
    parser = ArcAgiParser()

    # Test with mock data
    mock_task_data = {
        "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
        "test": [{"input": [[1, 1], [0, 0]], "output": [[0, 0], [1, 1]]}],
    }

    task = parser.parse_task(mock_task_data)
    assert isinstance(task, JaxArcTask)
    assert task.train_pairs.shape[0] == 1
    assert task.test_pairs.shape[0] == 1


def test_parser_error_handling():
    """Test parser error handling for invalid data."""
    parser = ArcAgiParser()

    # Test with invalid data
    invalid_data = {"invalid": "data"}

    with pytest.raises(ValueError):
        parser.parse_task(invalid_data)
```

### Integration Testing

```python
import jax
from jaxarc.envs import ArcEnvironment, create_standard_config
from jaxarc.envs.functional import arc_reset, arc_step
from jaxarc.types import ARCLEAction


def test_end_to_end_workflow():
    """Test complete workflow from configuration to environment execution."""
    # Create configuration
    config = create_standard_config()
    key = jax.random.PRNGKey(42)

    # Test functional API
    state = arc_reset(config, key)
    assert state is not None

    # Create action
    action = ARCLEAction(
        operation_id=0,  # Fill operation
        selection=jnp.array([0.5, 0.5, 0.7, 0.7]),  # Bounding box
        color=1,
    )

    # Test step
    key, step_key = jax.random.split(key)
    new_state, reward, done, info = arc_step(state, action, step_key)

    assert new_state is not None
    assert isinstance(reward, (int, float, jnp.ndarray))
    assert isinstance(done, (bool, jnp.ndarray))
    assert isinstance(info, dict)


def test_environment_class_integration():
    """Test environment class integration."""
    config = create_standard_config()
    env = ArcEnvironment(config)

    key = jax.random.PRNGKey(42)
    state = env.reset(key)

    # Test that environment works with JAX transformations
    jitted_reset = jax.jit(env.reset)
    jitted_state = jitted_reset(key)

    # States should be equivalent
    assert jnp.array_equal(state.current_grid.data, jitted_state.current_grid.data)
```

## Test Organization Guidelines

### File Naming

- **Test files**: `test_*.py` following pytest conventions
- **Utility files**: `*_utils.py` for testing utilities
- **Framework files**: `*_framework.py` for testing frameworks

### Directory Structure

Mirror the source code structure:

```
tests/
├── test_types.py              # Core types
├── test_state.py              # State management
├── envs/                      # Environment tests
├── parsers/                   # Parser tests
└── utils/                     # Utility tests
```

### Test Categories

1. **Unit tests**: Individual function and class testing
2. **Integration tests**: Component interaction testing
3. **Property-based tests**: Hypothesis-driven testing
4. **Performance tests**: Basic performance regression testing
5. **JAX compatibility tests**: Transformation testing

## Mock Data and Fixtures

### Creating Mock Data

```python
from tests.test_utils import MockDataGenerator


def test_with_mock_data():
    """Example of using mock data generators."""
    # Create mock grid
    grid_data, mask = MockDataGenerator.create_mock_grid(5, 5)
    assert grid_data.shape == (5, 5)
    assert mask.shape == (5, 5)

    # Create mock task
    task_data = MockDataGenerator.create_mock_task_data(
        num_train_pairs=2, num_test_pairs=1, grid_size=5
    )
    assert len(task_data["train"]) == 2
    assert len(task_data["test"]) == 1
```

### Using Fixtures

```python
def test_with_fixtures(jax_key, mock_grid, mock_config):
    """Example of using pytest fixtures."""
    # Use consistent PRNG key
    key1, key2 = jax.random.split(jax_key)

    # Use pre-configured mock objects
    assert mock_grid.data.shape == (5, 5)
    assert mock_config.max_episode_steps > 0

    # Test with fixtures
    result = some_function(mock_grid, mock_config, key1)
    assert result is not None
```

## Performance Testing

### Basic Performance Tests

```python
import time
import jax


def test_jit_compilation_performance():
    """Test that JIT compilation provides expected speedup."""

    def slow_function(x):
        return jnp.sum(x**2) + jnp.mean(x)

    x = jnp.ones((1000, 1000))

    # Time normal execution
    start = time.time()
    for _ in range(10):
        result_normal = slow_function(x)
    normal_time = time.time() - start

    # Time JIT execution
    jitted_fn = jax.jit(slow_function)
    # Warm up JIT
    _ = jitted_fn(x)

    start = time.time()
    for _ in range(10):
        result_jit = jitted_fn(x)
    jit_time = time.time() - start

    # JIT should be faster (allow some variance)
    assert jit_time < normal_time * 0.8
    assert jnp.allclose(result_normal, result_jit)
```

### Memory Usage Testing

```python
import psutil
import os


def test_memory_usage():
    """Test that functions don't have memory leaks."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Run function many times
    for _ in range(1000):
        result = some_function(test_input)

    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory

    # Memory increase should be reasonable (< 100MB)
    assert memory_increase < 100 * 1024 * 1024
```

## Debugging Test Failures

### JAX Debugging

```python
import jax


def debug_jax_function():
    """Example of debugging JAX functions."""

    def my_function(x):
        # Use jax.debug.print for debugging inside JIT
        jax.debug.print("Input shape: {}", x.shape)
        jax.debug.print("Input values: {}", x)

        result = x * 2

        jax.debug.print("Result: {}", result)
        return result

    # Debug without JIT first
    x = jnp.array([1.0, 2.0, 3.0])
    result = my_function(x)

    # Then test with JIT
    jitted_fn = jax.jit(my_function)
    jit_result = jitted_fn(x)
```

### Array Debugging

```python
import chex


def debug_array_operations():
    """Example of debugging array operations."""
    x = jnp.array([[1, 2], [3, 4]])

    # Validate shapes and types
    chex.assert_shape(x, (2, 2))
    chex.assert_type(x, jnp.ndarray)

    # Check for common issues
    assert not jnp.any(jnp.isnan(x)), "Array contains NaN values"
    assert not jnp.any(jnp.isinf(x)), "Array contains infinite values"

    # Validate ranges
    assert jnp.all(x >= 0), "Array contains negative values"
```

## Continuous Integration

### Test Commands

```bash
# Run all tests with coverage
pixi run -e test test --cov=src/jaxarc --cov-report=html --cov-report=term

# Run specific test categories
pixi run -e test test tests/test_types.py -v
pixi run -e test test tests/envs/ -v
pixi run -e test test tests/parsers/ -v
pixi run -e test test tests/utils/ -v

# Run performance tests
pixi run -e test test tests/test_performance_regression.py -v

# Run with strict warnings
pixi run -e test test -W error::UserWarning
```

### Coverage Goals

- **Core Types**: 100% coverage
- **Environment System**: 95% coverage
- **Parsers**: 90% coverage
- **Utilities**: 85% coverage
- **Overall**: 90%+ coverage

## Common Pitfalls and Solutions

### 1. Dynamic Shapes in JAX

**Problem**: Using dynamic shapes that break JIT compilation.

```python
# BAD: Dynamic shape based on runtime value
def bad_function(x, n):
    return jnp.ones((n, n))  # n is runtime value


# GOOD: Use static shapes with padding/masking
def good_function(x, n, max_size=30):
    result = jnp.ones((max_size, max_size))
    mask = jnp.arange(max_size) < n
    return result * mask[:, None] * mask[None, :]
```

### 2. Side Effects in Pure Functions

**Problem**: Functions with side effects that break JAX transformations.

```python
# BAD: Side effects
def bad_function(x):
    print(f"Processing {x}")  # Side effect
    return x * 2


# GOOD: Use JAX debug callbacks
def good_function(x):
    jax.debug.print("Processing {}", x)
    return x * 2
```

### 3. Incorrect PRNG Usage

**Problem**: Not using JAX PRNG keys correctly.

```python
# BAD: Using Python random
import random


def bad_random_function():
    return random.random()


# GOOD: Using JAX PRNG
def good_random_function(key):
    return jax.random.uniform(key)
```

### 4. Equinox Module Issues

**Problem**: Equinox modules not properly structured as PyTrees.

```python
# BAD: Mutable attributes
class BadModule(eqx.Module):
    data: list  # Lists are not JAX arrays

    def __init__(self):
        self.data = [1, 2, 3]  # Mutable


# GOOD: Immutable JAX arrays
class GoodModule(eqx.Module):
    data: jnp.ndarray

    def __init__(self):
        self.data = jnp.array([1, 2, 3])  # Immutable JAX array
```

## Future Testing Considerations

As the project evolves, consider these testing strategies:

1. **Multi-device testing**: Test pmap functionality across multiple devices
2. **Large-scale testing**: Test with larger datasets and longer episodes
3. **Fuzzing**: Use property-based testing for security and robustness
4. **Benchmark testing**: Establish performance baselines and regression
   detection
5. **Integration testing**: Test with actual RL training loops and agents

## Resources

- [JAX Testing Documentation](https://jax.readthedocs.io/en/latest/debugging/index.html)
- [Equinox Documentation](https://docs.kidger.site/equinox/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Chex Documentation](https://github.com/deepmind/chex)
- [Pytest Documentation](https://docs.pytest.org/)
