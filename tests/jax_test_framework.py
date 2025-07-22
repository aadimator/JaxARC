"""
JAX transformation testing framework for JaxARC.

This module provides comprehensive testing utilities for validating JAX transformations
(jit, vmap, pmap) and ensuring functions work correctly with JAX's functional programming model.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import chex
import jax
import jax.numpy as jnp
import pytest

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class JaxTransformationTester:
    """Comprehensive tester for JAX transformations."""
    
    def __init__(self, func: Callable, test_inputs: tuple, test_kwargs: dict | None = None):
        """
        Initialize the transformation tester.
        
        Args:
            func: Function to test
            test_inputs: Input arguments for testing
            test_kwargs: Keyword arguments for testing
        """
        self.func = func
        self.test_inputs = test_inputs
        self.test_kwargs = test_kwargs or {}
        self._baseline_result = None
    
    def get_baseline_result(self) -> Any:
        """Get baseline result from untransformed function."""
        if self._baseline_result is None:
            self._baseline_result = self.func(*self.test_inputs, **self.test_kwargs)
        return self._baseline_result
    
    def test_jit_compilation(self) -> None:
        """Test that function compiles and runs correctly with jit."""
        baseline = self.get_baseline_result()
        
        # Test jit compilation
        jitted_func = jax.jit(self.func)
        jitted_result = jitted_func(*self.test_inputs, **self.test_kwargs)
        
        # Compare results
        chex.assert_trees_all_close(baseline, jitted_result, rtol=1e-6)
        
        # Test that function can be called multiple times (compilation caching)
        jitted_result2 = jitted_func(*self.test_inputs, **self.test_kwargs)
        chex.assert_trees_all_close(jitted_result, jitted_result2, rtol=1e-6)
    
    def test_vmap_batching(self, batch_size: int = 3, in_axes: int | tuple = 0) -> None:
        """
        Test that function works correctly with vmap batching.
        
        Args:
            batch_size: Size of batch dimension to test
            in_axes: Axis specification for vmap
        """
        baseline = self.get_baseline_result()
        
        # Create batched inputs
        if isinstance(in_axes, int):
            in_axes_tuple = (in_axes,) * len(self.test_inputs)
        else:
            in_axes_tuple = in_axes
        
        batched_inputs = []
        for i, (arg, axis) in enumerate(zip(self.test_inputs, in_axes_tuple)):
            if axis is None:
                # Don't batch this argument
                batched_inputs.append(arg)
            else:
                # Batch this argument
                if isinstance(arg, jnp.ndarray):
                    # Stack multiple copies along the specified axis
                    batched_arg = jnp.stack([arg] * batch_size, axis=axis)
                    batched_inputs.append(batched_arg)
                else:
                    # For non-arrays, just repeat
                    batched_inputs.append(arg)
        
        # Test vmap
        vmapped_func = jax.vmap(self.func, in_axes=in_axes_tuple)
        vmapped_result = vmapped_func(*batched_inputs, **self.test_kwargs)
        
        # Check that result has correct batch dimension
        if isinstance(baseline, jnp.ndarray):
            expected_shape = (batch_size,) + baseline.shape
            assert vmapped_result.shape == expected_shape
            
            # Check that first element matches baseline
            first_result = vmapped_result[0]
            chex.assert_trees_all_close(baseline, first_result, rtol=1e-6)
        elif isinstance(vmapped_result, (tuple, list)):
            # Handle tuple/list results
            for i, (base_elem, vmap_elem) in enumerate(zip(baseline, vmapped_result)):
                if isinstance(base_elem, jnp.ndarray):
                    expected_shape = (batch_size,) + base_elem.shape
                    assert vmap_elem.shape == expected_shape
                    chex.assert_trees_all_close(base_elem, vmap_elem[0], rtol=1e-6)
    
    def test_pmap_parallel(self, num_devices: int | None = None) -> None:
        """
        Test that function works with pmap (if multiple devices available).
        
        Args:
            num_devices: Number of devices to use (None for all available)
        """
        available_devices = jax.device_count()
        if available_devices < 2:
            pytest.skip("pmap testing requires multiple devices")
        
        if num_devices is None:
            num_devices = min(available_devices, 2)  # Use at most 2 for testing
        
        baseline = self.get_baseline_result()
        
        # Create inputs for each device
        device_inputs = []
        for arg in self.test_inputs:
            if isinstance(arg, jnp.ndarray):
                # Replicate across devices
                device_arg = jnp.stack([arg] * num_devices)
                device_inputs.append(device_arg)
            else:
                device_inputs.append(arg)
        
        # Test pmap
        pmapped_func = jax.pmap(self.func)
        pmapped_result = pmapped_func(*device_inputs, **self.test_kwargs)
        
        # Check results from each device
        if isinstance(baseline, jnp.ndarray):
            assert pmapped_result.shape == (num_devices,) + baseline.shape
            
            # Each device should produce the same result
            for device_idx in range(num_devices):
                device_result = pmapped_result[device_idx]
                chex.assert_trees_all_close(baseline, device_result, rtol=1e-6)
    
    def test_grad_compatibility(self) -> None:
        """Test that function works with grad (if differentiable)."""
        try:
            # Try to compute gradient
            grad_func = jax.grad(self.func)
            
            # This will only work if function returns a scalar and inputs are differentiable
            if all(isinstance(arg, jnp.ndarray) and jnp.issubdtype(arg.dtype, jnp.floating) 
                   for arg in self.test_inputs):
                baseline = self.get_baseline_result()
                if isinstance(baseline, jnp.ndarray) and baseline.shape == ():
                    # Function returns scalar, can compute gradient
                    grad_result = grad_func(*self.test_inputs, **self.test_kwargs)
                    
                    # Gradient should have same structure as inputs (but grad_result is not a tuple)
                    if len(self.test_inputs) == 1:
                        chex.assert_trees_all_equal_shapes(self.test_inputs[0], grad_result)
                    else:
                        chex.assert_trees_all_equal_shapes(self.test_inputs, grad_result)
        except (TypeError, ValueError):
            # Function is not differentiable, skip gradient test
            pytest.skip("Function is not differentiable")
    
    def test_all_transformations(self, batch_size: int = 3) -> None:
        """Run all transformation tests."""
        self.test_jit_compilation()
        self.test_vmap_batching(batch_size=batch_size)
        
        # Only test pmap if multiple devices available
        if jax.device_count() > 1:
            self.test_pmap_parallel()
        
        # Test gradient compatibility if applicable
        self.test_grad_compatibility()


def run_jax_transformation_tests(
    func: Callable,
    test_inputs: tuple,
    test_kwargs: dict | None = None,
    batch_size: int = 3
) -> None:
    """
    Convenience function to test all JAX transformations.
    
    Args:
        func: Function to test
        test_inputs: Input arguments for testing
        test_kwargs: Keyword arguments for testing
        batch_size: Size of batch dimension for vmap testing
    """
    tester = JaxTransformationTester(func, test_inputs, test_kwargs)
    tester.test_all_transformations(batch_size=batch_size)


def jax_test_case(batch_size: int = 3):
    """
    Decorator to automatically test JAX transformations for a test function.
    
    Usage:
        @jax_test_case()
        def test_my_function():
            def my_func(x):
                return x * 2
            
            test_input = jnp.array([1, 2, 3])
            return my_func, (test_input,), {}
    """
    def decorator(test_func: Callable) -> Callable:
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            # Get function and test data from test function
            func, test_inputs, test_kwargs = test_func(*args, **kwargs)
            
            # Run JAX transformation tests
            test_jax_transformations(func, test_inputs, test_kwargs, batch_size)
            
        return wrapper
    return decorator


class JaxPropertyTester:
    """Property-based testing utilities for JAX functions."""
    
    @staticmethod
    def test_pure_function(func: Callable, *args, **kwargs) -> None:
        """Test that function is pure (no side effects, deterministic)."""
        result1 = func(*args, **kwargs)
        result2 = func(*args, **kwargs)
        
        chex.assert_trees_all_close(result1, result2, rtol=1e-6)
    
    @staticmethod
    def test_shape_preservation(func: Callable, input_array: jnp.ndarray, *args, **kwargs) -> None:
        """Test that function preserves input array shape."""
        result = func(input_array, *args, **kwargs)
        
        if isinstance(result, jnp.ndarray):
            assert result.shape == input_array.shape
        elif hasattr(result, 'shape'):
            assert result.shape == input_array.shape
    
    @staticmethod
    def test_dtype_preservation(func: Callable, input_array: jnp.ndarray, *args, **kwargs) -> None:
        """Test that function preserves input array dtype."""
        result = func(input_array, *args, **kwargs)
        
        if isinstance(result, jnp.ndarray):
            assert result.dtype == input_array.dtype
        elif hasattr(result, 'dtype'):
            assert result.dtype == input_array.dtype
    
    @staticmethod
    def test_commutativity(func: Callable, arg1: Any, arg2: Any, *args, **kwargs) -> None:
        """Test that binary function is commutative."""
        result1 = func(arg1, arg2, *args, **kwargs)
        result2 = func(arg2, arg1, *args, **kwargs)
        
        chex.assert_trees_all_close(result1, result2, rtol=1e-6)
    
    @staticmethod
    def test_associativity(func: Callable, arg1: Any, arg2: Any, arg3: Any, *args, **kwargs) -> None:
        """Test that binary function is associative."""
        result1 = func(func(arg1, arg2, *args, **kwargs), arg3, *args, **kwargs)
        result2 = func(arg1, func(arg2, arg3, *args, **kwargs), *args, **kwargs)
        
        chex.assert_trees_all_close(result1, result2, rtol=1e-6)
    
    @staticmethod
    def test_idempotency(func: Callable, *args, **kwargs) -> None:
        """Test that function is idempotent (f(f(x)) = f(x))."""
        result1 = func(*args, **kwargs)
        result2 = func(result1, *args[1:], **kwargs)  # Apply function to its own result
        
        chex.assert_trees_all_close(result1, result2, rtol=1e-6)


# Convenience functions
def assert_jax_pure(func: Callable, *args, **kwargs) -> None:
    """Assert that function is pure."""
    JaxPropertyTester.test_pure_function(func, *args, **kwargs)


def assert_shape_preserved(func: Callable, input_array: jnp.ndarray, *args, **kwargs) -> None:
    """Assert that function preserves shape."""
    JaxPropertyTester.test_shape_preservation(func, input_array, *args, **kwargs)


def assert_dtype_preserved(func: Callable, input_array: jnp.ndarray, *args, **kwargs) -> None:
    """Assert that function preserves dtype."""
    JaxPropertyTester.test_dtype_preservation(func, input_array, *args, **kwargs)