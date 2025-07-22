# JAX Testing Patterns and Utilities

This document provides comprehensive guidance on testing JAX-based code in the
JaxARC project, including specific patterns, utilities, and best practices for
ensuring JAX compatibility.

## Overview

JAX introduces unique testing challenges due to its functional programming
paradigm, JIT compilation, and transformation system. This guide covers the
essential patterns and utilities for effective JAX testing.

## Core JAX Testing Concepts

### 1. Pure Function Testing

JAX functions must be pure (no side effects) to work with transformations:

```python
import jax
import jax.numpy as jnp
import pytest


def test_pure_function():
    """Test a pure function for JAX compatibility."""

    def pure_add(x, y):
        return x + y

    # Test normal execution
    result = pure_add(1.0, 2.0)
    assert result == 3.0

    # Test JIT compilation
    jitted_add = jax.jit(pure_add)
    jit_result = jitted_add(1.0, 2.0)
    assert jnp.allclose(result, jit_result)


def test_impure_function_conversion():
    """Example of converting impure function to pure."""

    # BAD: Impure function with side effects
    def impure_function(x):
        print(f"Processing {x}")  # Side effect
        return x * 2

    # GOOD: Pure function with JAX debug callbacks
    def pure_function(x):
        jax.debug.print("Processing {}", x)
        return x * 2

    # Test the pure version
    x = jnp.array([1.0, 2.0, 3.0])
    result = pure_function(x)

    # Should work with JIT
    jitted_fn = jax.jit(pure_function)
    jit_result = jitted_fn(x)
    assert jnp.allclose(result, jit_result)
```

### 2. Static Shape Requirements

JAX requires static shapes for JIT compilation:

```python
def test_static_shapes():
    """Test functions with static shape requirements."""

    def process_fixed_grid(grid):
        """Function that works with static shapes."""
        assert grid.shape == (10, 10), "Grid must be 10x10"
        return jnp.sum(grid, axis=1)

    # Test with correct shape
    grid = jnp.ones((10, 10))
    result = process_fixed_grid(grid)
    assert result.shape == (10,)

    # Test JIT compilation
    jitted_fn = jax.jit(process_fixed_grid)
    jit_result = jitted_fn(grid)
    assert jnp.allclose(result, jit_result)


def test_dynamic_to_static_conversion():
    """Convert dynamic shape function to static shape."""

    def dynamic_function(x, size):
        """BAD: Dynamic shape based on runtime value."""
        return jnp.ones((size, size))  # Breaks JIT

    def static_function(x, size, max_size=30):
        """GOOD: Static shape with masking."""
        result = jnp.ones((max_size, max_size))
        mask = jnp.arange(max_size) < size
        return result * mask[:, None] * mask[None, :]

    # Test static version
    x = jnp.array([1.0])
    size = 5
    result = static_function(x, size)

    # Should work with JIT
    jitted_fn = jax.jit(static_function, static_argnums=(1,))
    jit_result = jitted_fn(x, size)
    assert jnp.allclose(result, jit_result)
```

### 3. PRNG Key Management

JAX uses explicit PRNG keys for reproducible randomness:

```python
def test_prng_key_usage():
    """Test proper PRNG key usage."""

    def random_function(key, shape):
        return jax.random.normal(key, shape)

    # Test with explicit key
    key = jax.random.PRNGKey(42)
    result1 = random_function(key, (5,))
    result2 = random_function(key, (5,))

    # Same key should produce same result
    assert jnp.allclose(result1, result2)

    # Different keys should produce different results
    key2 = jax.random.PRNGKey(43)
    result3 = random_function(key2, (5,))
    assert not jnp.allclose(result1, result3)


def test_key_splitting():
    """Test proper key splitting for multiple random operations."""

    def multi_random_function(key):
        key1, key2, key3 = jax.random.split(key, 3)

        a = jax.random.normal(key1, (3,))
        b = jax.random.uniform(key2, (3,))
        c = jax.random.randint(key3, (3,), 0, 10)

        return a, b, c

    key = jax.random.PRNGKey(42)
    a1, b1, c1 = multi_random_function(key)
    a2, b2, c2 = multi_random_function(key)

    # Same key should produce same results
    assert jnp.allclose(a1, a2)
    assert jnp.allclose(b1, b2)
    assert jnp.array_equal(c1, c2)

    # Test JIT compatibility
    jitted_fn = jax.jit(multi_random_function)
    a3, b3, c3 = jitted_fn(key)
    assert jnp.allclose(a1, a3)
    assert jnp.allclose(b1, b3)
    assert jnp.array_equal(c1, c3)
```

## JAX Transformation Testing

### 1. JIT Compilation Testing

```python
def test_jit_compilation():
    """Test JIT compilation compatibility."""

    def complex_function(x, y):
        z = x * y
        w = jnp.sin(z) + jnp.cos(z)
        return jnp.sum(w**2)

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])

    # Test normal execution
    result_normal = complex_function(x, y)

    # Test JIT compilation
    jitted_fn = jax.jit(complex_function)
    result_jit = jitted_fn(x, y)

    # Results should be identical
    assert jnp.allclose(result_normal, result_jit, rtol=1e-6)

    # Test compilation time (should be fast after first call)
    import time

    start = time.time()
    for _ in range(100):
        _ = jitted_fn(x, y)
    jit_time = time.time() - start

    start = time.time()
    for _ in range(100):
        _ = complex_function(x, y)
    normal_time = time.time() - start

    # JIT should be faster for repeated calls
    assert jit_time < normal_time


def test_jit_with_static_args():
    """Test JIT with static arguments."""

    def function_with_static_args(x, operation="add", factor=2):
        if operation == "add":
            return x + factor
        elif operation == "multiply":
            return x * factor
        else:
            return x

    x = jnp.array([1.0, 2.0, 3.0])

    # JIT with static arguments
    jitted_fn = jax.jit(
        function_with_static_args, static_argnames=["operation", "factor"]
    )

    result_add = jitted_fn(x, operation="add", factor=5)
    result_mult = jitted_fn(x, operation="multiply", factor=3)

    assert jnp.allclose(result_add, x + 5)
    assert jnp.allclose(result_mult, x * 3)
```

### 2. Vectorization (vmap) Testing

```python
def test_vmap_transformation():
    """Test vectorization with vmap."""

    def single_item_function(x):
        return jnp.sum(x**2)

    # Test single item
    single_x = jnp.array([1.0, 2.0, 3.0])
    single_result = single_item_function(single_x)
    assert single_result.shape == ()

    # Test batch processing with vmap
    batch_x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    vmapped_fn = jax.vmap(single_item_function)
    batch_result = vmapped_fn(batch_x)

    assert batch_result.shape == (3,)

    # Verify results match manual computation
    expected = jnp.array(
        [jnp.sum(batch_x[0] ** 2), jnp.sum(batch_x[1] ** 2), jnp.sum(batch_x[2] ** 2)]
    )
    assert jnp.allclose(batch_result, expected)


def test_vmap_with_multiple_args():
    """Test vmap with multiple arguments."""

    def two_arg_function(x, y):
        return jnp.dot(x, y)

    # Batch data
    batch_x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    batch_y = jnp.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]])

    # Vectorize over both arguments
    vmapped_fn = jax.vmap(two_arg_function)
    batch_result = vmapped_fn(batch_x, batch_y)

    assert batch_result.shape == (3,)

    # Verify results
    expected = jnp.array(
        [
            jnp.dot(batch_x[0], batch_y[0]),
            jnp.dot(batch_x[1], batch_y[1]),
            jnp.dot(batch_x[2], batch_y[2]),
        ]
    )
    assert jnp.allclose(batch_result, expected)
```

### 3. Parallel Mapping (pmap) Testing

```python
def test_pmap_transformation():
    """Test parallel mapping with pmap (if multiple devices available)."""

    def parallel_function(x):
        return jnp.sum(x**2, axis=1)

    # Check if multiple devices are available
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("Multiple devices not available for pmap testing")

    # Create data for parallel processing
    # Shape should be (num_devices, batch_per_device, features)
    num_devices = len(devices)
    data = jnp.ones((num_devices, 4, 10))  # 4 items per device, 10 features each

    # Test pmap
    pmapped_fn = jax.pmap(parallel_function)
    result = pmapped_fn(data)

    # Result shape should be (num_devices, batch_per_device)
    assert result.shape == (num_devices, 4)

    # Verify results (all should be 10.0 since sum of 10 ones squared)
    expected = jnp.full((num_devices, 4), 10.0)
    assert jnp.allclose(result, expected)


def test_pmap_with_axis_name():
    """Test pmap with axis names for collective operations."""

    def collective_function(x):
        # Sum across all devices
        return jax.lax.psum(x, axis_name="devices")

    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("Multiple devices not available for pmap testing")

    num_devices = len(devices)
    data = jnp.ones((num_devices, 5))  # 5 elements per device

    pmapped_fn = jax.pmap(collective_function, axis_name="devices")
    result = pmapped_fn(data)

    # Each device should have the sum across all devices
    expected_sum = num_devices * 5  # Total sum across all devices
    expected = jnp.full((num_devices, 5), expected_sum)
    assert jnp.allclose(result, expected)
```

## Testing Utilities and Frameworks

### 1. JAX Test Framework Usage

```python
from tests.jax_test_framework import run_jax_transformation_tests


def test_with_framework():
    """Using the JAX testing framework."""

    def my_function(x, y):
        return x * y + jnp.sin(x)

    # Test inputs
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    test_inputs = (x, y)

    # This automatically tests jit, vmap, and pmap (if available)
    run_jax_transformation_tests(my_function, test_inputs)


def test_framework_with_custom_checks():
    """Using framework with custom validation."""

    def my_function(x):
        return jnp.clip(x, 0.0, 1.0)

    def custom_validator(original_fn, transformed_fn, inputs):
        """Custom validation for transformation results."""
        original_result = original_fn(*inputs)
        transformed_result = transformed_fn(*inputs)

        # Check that clipping worked
        assert jnp.all(original_result >= 0.0)
        assert jnp.all(original_result <= 1.0)
        assert jnp.all(transformed_result >= 0.0)
        assert jnp.all(transformed_result <= 1.0)

        # Check results are close
        assert jnp.allclose(original_result, transformed_result)

    test_inputs = (jnp.array([-1.0, 0.5, 2.0]),)
    run_jax_transformation_tests(
        my_function, test_inputs, custom_validator=custom_validator
    )
```

### 2. Property-Based Testing with Hypothesis

```python
from hypothesis import given, strategies as st
from tests.hypothesis_utils import jax_arrays


@given(jax_arrays(shape=(5, 5), dtype=jnp.float32))
def test_matrix_properties(matrix):
    """Test matrix operation properties."""

    def matrix_operation(m):
        return jnp.transpose(jnp.transpose(m))  # Double transpose should be identity

    result = matrix_operation(matrix)

    # Property: double transpose is identity
    assert jnp.allclose(result, matrix)

    # Test JAX compatibility
    jitted_fn = jax.jit(matrix_operation)
    jit_result = jitted_fn(matrix)
    assert jnp.allclose(result, jit_result)


@given(
    jax_arrays(shape=(10,), dtype=jnp.float32, min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_scaling_properties(array, scale_factor):
    """Test scaling operation properties."""

    def scale_array(arr, factor):
        return arr * factor

    result = scale_array(array, scale_factor)

    # Properties
    assert result.shape == array.shape
    assert result.dtype == array.dtype
    assert jnp.all(result >= 0.0)  # Should remain non-negative

    # Test with JAX transformations
    jitted_fn = jax.jit(scale_array)
    jit_result = jitted_fn(array, scale_factor)
    assert jnp.allclose(result, jit_result)
```

### 3. Chex Utilities for Validation

```python
import chex


def test_with_chex_validation():
    """Using chex for JAX-specific validation."""

    def validated_function(x, y):
        # Validate inputs
        chex.assert_rank(x, 2)  # x should be 2D
        chex.assert_rank(y, 1)  # y should be 1D
        chex.assert_type(x, jnp.ndarray)
        chex.assert_type(y, jnp.ndarray)

        # Perform operation
        result = jnp.dot(x, y)

        # Validate output
        chex.assert_rank(result, 1)
        chex.assert_shape(result, (x.shape[0],))

        return result

    # Test with valid inputs
    x = jnp.ones((3, 4))
    y = jnp.ones((4,))

    result = validated_function(x, y)
    assert result.shape == (3,)

    # Test JAX compatibility
    jitted_fn = jax.jit(validated_function)
    jit_result = jitted_fn(x, y)
    assert jnp.allclose(result, jit_result)

    # Test with invalid inputs (should raise)
    with pytest.raises(AssertionError):
        invalid_y = jnp.ones((3,))  # Wrong shape
        validated_function(x, invalid_y)


def test_chex_dataclass_validation():
    """Test chex dataclass validation."""

    @chex.dataclass
    class TestConfig:
        learning_rate: float
        batch_size: int
        hidden_dims: jnp.ndarray

        def __check_init__(self):
            chex.assert_scalar_positive(self.learning_rate)
            chex.assert_scalar_positive(self.batch_size)
            chex.assert_rank(self.hidden_dims, 1)

    # Test valid config
    config = TestConfig(
        learning_rate=0.01, batch_size=32, hidden_dims=jnp.array([64, 32, 16])
    )

    # Test JAX compatibility
    def use_config(cfg):
        return cfg.learning_rate * cfg.batch_size

    jitted_fn = jax.jit(use_config)
    result = jitted_fn(config)
    assert result == 0.01 * 32

    # Test invalid config (should raise during initialization)
    with pytest.raises(AssertionError):
        TestConfig(
            learning_rate=-0.01,  # Invalid: negative
            batch_size=32,
            hidden_dims=jnp.array([64, 32, 16]),
        )
```

## Advanced JAX Testing Patterns

### 1. Testing with JAX Debugging

```python
def test_with_jax_debugging():
    """Test using JAX debugging utilities."""

    def debug_function(x):
        jax.debug.print("Input: {}", x)

        # Conditional debugging
        jax.debug.print("Sum: {}", jnp.sum(x), ordered=True)

        result = x**2
        jax.debug.print("Result: {}", result)

        return result

    x = jnp.array([1.0, 2.0, 3.0])

    # Test normal execution (debug prints will show)
    result = debug_function(x)
    assert jnp.allclose(result, x**2)

    # Test JIT execution (debug prints will show during execution)
    jitted_fn = jax.jit(debug_function)
    jit_result = jitted_fn(x)
    assert jnp.allclose(result, jit_result)


def test_with_debug_callbacks():
    """Test using debug callbacks for complex debugging."""

    def callback_function(x):
        def debug_callback(x_val):
            print(f"Callback: x shape = {x_val.shape}, mean = {jnp.mean(x_val)}")

        jax.debug.callback(debug_callback, x)
        return jnp.sum(x)

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    # Test with callback
    result = callback_function(x)
    assert result == 10.0

    # Test JIT compatibility
    jitted_fn = jax.jit(callback_function)
    jit_result = jitted_fn(x)
    assert jnp.allclose(result, jit_result)
```

### 2. Testing Gradient Computations

```python
def test_gradient_computation():
    """Test functions that use gradients."""

    def loss_function(params, x, y):
        pred = jnp.dot(x, params)
        return jnp.mean((pred - y) ** 2)

    # Test data
    params = jnp.array([1.0, 2.0])
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([5.0, 11.0])

    # Test loss computation
    loss = loss_function(params, x, y)
    assert loss >= 0.0

    # Test gradient computation
    grad_fn = jax.grad(loss_function)
    gradients = grad_fn(params, x, y)
    assert gradients.shape == params.shape

    # Test JIT compilation of gradient
    jitted_grad = jax.jit(grad_fn)
    jit_gradients = jitted_grad(params, x, y)
    assert jnp.allclose(gradients, jit_gradients)

    # Test vmap over batch dimension
    batch_grad_fn = jax.vmap(grad_fn, in_axes=(None, 0, 0))
    batch_x = jnp.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    batch_y = jnp.array([[5.0], [11.0]])

    batch_gradients = batch_grad_fn(params, batch_x, batch_y)
    assert batch_gradients.shape == (2, 2)  # (batch_size, param_size)


def test_higher_order_derivatives():
    """Test higher-order derivatives."""

    def quadratic_function(x):
        return x**4 + 2 * x**3 + x**2

    x = 2.0

    # First derivative
    first_deriv = jax.grad(quadratic_function)(x)
    expected_first = 4 * x**3 + 6 * x**2 + 2 * x
    assert jnp.allclose(first_deriv, expected_first)

    # Second derivative
    second_deriv = jax.grad(jax.grad(quadratic_function))(x)
    expected_second = 12 * x**2 + 12 * x + 2
    assert jnp.allclose(second_deriv, expected_second)

    # Test JIT compilation
    jitted_second_deriv = jax.jit(jax.grad(jax.grad(quadratic_function)))
    jit_second_deriv = jitted_second_deriv(x)
    assert jnp.allclose(second_deriv, jit_second_deriv)
```

### 3. Testing Custom JAX Primitives

```python
def test_custom_primitive():
    """Test custom JAX primitive (advanced)."""
    from jax import lax

    def custom_clip(x, min_val, max_val):
        """Custom clipping function using JAX primitives."""
        return lax.clamp(min_val, x, max_val)

    x = jnp.array([-2.0, 0.5, 3.0])
    min_val = 0.0
    max_val = 1.0

    # Test custom primitive
    result = custom_clip(x, min_val, max_val)
    expected = jnp.array([0.0, 0.5, 1.0])
    assert jnp.allclose(result, expected)

    # Test JAX transformations
    jitted_fn = jax.jit(custom_clip)
    jit_result = jitted_fn(x, min_val, max_val)
    assert jnp.allclose(result, jit_result)

    # Test gradient
    grad_fn = jax.grad(lambda x: jnp.sum(custom_clip(x, min_val, max_val)))
    gradients = grad_fn(x)

    # Gradient should be 0 for clipped values, 1 for unclipped
    expected_grad = jnp.array([0.0, 1.0, 0.0])
    assert jnp.allclose(gradients, expected_grad)
```

## Performance Testing Patterns

### 1. Compilation Time Testing

```python
import time


def test_compilation_time():
    """Test JIT compilation time."""

    def complex_function(x):
        for _ in range(10):
            x = jnp.sin(x) + jnp.cos(x)
        return x

    x = jnp.ones((1000,))

    # Measure compilation time
    start = time.time()
    jitted_fn = jax.jit(complex_function)
    _ = jitted_fn(x)  # First call triggers compilation
    compilation_time = time.time() - start

    # Compilation should be reasonable (< 5 seconds for most functions)
    assert compilation_time < 5.0

    # Subsequent calls should be fast
    start = time.time()
    for _ in range(100):
        _ = jitted_fn(x)
    execution_time = time.time() - start

    # Should be much faster than compilation
    assert execution_time < compilation_time / 10
```

### 2. Memory Usage Testing

```python
def test_memory_usage():
    """Test memory usage of JAX functions."""

    def memory_intensive_function(x):
        # Create intermediate arrays
        y = x * 2
        z = jnp.sin(y)
        w = jnp.cos(z)
        return jnp.sum(w)

    # Test with different sizes
    small_x = jnp.ones((100,))
    large_x = jnp.ones((10000,))

    # Both should work without memory issues
    small_result = memory_intensive_function(small_x)
    large_result = memory_intensive_function(large_x)

    assert jnp.isfinite(small_result)
    assert jnp.isfinite(large_result)

    # Test JIT compilation doesn't cause memory leaks
    jitted_fn = jax.jit(memory_intensive_function)

    for _ in range(100):
        _ = jitted_fn(small_x)

    # Should complete without memory errors
```

## Common JAX Testing Pitfalls

### 1. Array Precision Issues

```python
def test_precision_handling():
    """Test handling of floating-point precision."""

    def precision_sensitive_function(x):
        return jnp.sqrt(x**2)  # Should equal abs(x)

    # Test with different precisions
    x_float32 = jnp.array([-1.0, 2.0, -3.0], dtype=jnp.float32)
    x_float64 = jnp.array([-1.0, 2.0, -3.0], dtype=jnp.float64)

    result_32 = precision_sensitive_function(x_float32)
    result_64 = precision_sensitive_function(x_float64)

    expected = jnp.array([1.0, 2.0, 3.0])

    # Use appropriate tolerance for each precision
    assert jnp.allclose(result_32, expected, rtol=1e-6)  # float32 tolerance
    assert jnp.allclose(result_64, expected, rtol=1e-12)  # float64 tolerance


def test_nan_and_inf_handling():
    """Test handling of NaN and infinity values."""

    def robust_function(x):
        # Handle potential NaN/inf values
        safe_x = jnp.where(jnp.isfinite(x), x, 0.0)
        return jnp.sum(safe_x)

    # Test with problematic values
    x_with_nan = jnp.array([1.0, jnp.nan, 3.0])
    x_with_inf = jnp.array([1.0, jnp.inf, 3.0])

    result_nan = robust_function(x_with_nan)
    result_inf = robust_function(x_with_inf)

    # Should handle gracefully
    assert jnp.isfinite(result_nan)
    assert jnp.isfinite(result_inf)
    assert result_nan == 4.0  # 1.0 + 0.0 + 3.0
    assert result_inf == 4.0  # 1.0 + 0.0 + 3.0
```

### 2. Device Placement Issues

```python
def test_device_placement():
    """Test device placement and data transfer."""
    # Get available devices
    devices = jax.devices()

    def device_function(x):
        return x * 2

    x = jnp.array([1.0, 2.0, 3.0])

    # Test on default device
    result = device_function(x)
    assert jnp.allclose(result, x * 2)

    # Test explicit device placement
    if len(devices) > 1:
        # Place on specific device
        x_on_device = jax.device_put(x, devices[0])
        result_on_device = device_function(x_on_device)

        assert jnp.allclose(result, result_on_device)

        # Test cross-device operations
        y_on_other_device = jax.device_put(x, devices[1])
        # This should work (JAX handles device transfers)
        combined_result = x_on_device + y_on_other_device
        assert combined_result.shape == x.shape
```

## Integration with Testing Framework

### Using the JaxARC Testing Utilities

```python
# Example of comprehensive test using all utilities
from tests.jax_test_framework import run_jax_transformation_tests
from tests.equinox_test_utils import run_equinox_module_tests
from tests.hypothesis_utils import test_jax_function_properties, jax_arrays
from tests.test_utils import MockDataGenerator


def test_comprehensive_jax_function():
    """Comprehensive test using all JAX testing utilities."""

    def my_jax_function(grid, action_mask, key):
        """Example JAX function that processes grid with action mask."""
        # Apply mask
        masked_grid = grid * action_mask

        # Add some randomness
        noise = jax.random.normal(key, grid.shape) * 0.1
        noisy_grid = masked_grid + noise

        # Compute result
        return jnp.clip(noisy_grid, 0.0, 1.0)

    # Test with mock data
    grid_data, mask = MockDataGenerator.create_mock_grid(5, 5)
    key = jax.random.PRNGKey(42)

    # Test basic functionality
    result = my_jax_function(grid_data, mask, key)
    assert result.shape == (5, 5)
    assert jnp.all(result >= 0.0)
    assert jnp.all(result <= 1.0)

    # Test JAX transformations
    test_inputs = (grid_data, mask, key)
    run_jax_transformation_tests(my_jax_function, test_inputs)

    # Test properties with Hypothesis
    def check_clipping_property(func, inputs, result):
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)
        assert result.shape == inputs[0].shape

    test_jax_function_properties(
        lambda inputs: my_jax_function(*inputs),
        st.tuples(
            jax_arrays(shape=(5, 5), dtype=jnp.float32),
            jax_arrays(shape=(5, 5), dtype=jnp.float32, min_value=0.0, max_value=1.0),
            st.just(jax.random.PRNGKey(42)),
        ),
        [check_clipping_property],
    )
```

This comprehensive guide provides the foundation for effective JAX testing in
the JaxARC project. Use these patterns and utilities to ensure your JAX code is
robust, performant, and compatible with all JAX transformations.
