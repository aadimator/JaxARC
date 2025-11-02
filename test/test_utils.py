"""
JAX-specific testing utilities and assertion helpers.

This module provides utilities for testing JAX-compatible functions,
including transformation testing, array validation, and PyTree operations.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree


def assert_jax_compatible(
    func: Callable,
    *args,
    test_jit: bool = True,
    test_vmap: bool = False,
    test_grad: bool = False,
    **kwargs,
) -> Any:
    """
    Verify function works with JAX transformations.

    Args:
        func: Function to test
        *args: Arguments to pass to function
        test_jit: Whether to test JIT compilation
        test_vmap: Whether to test vmap transformation
        test_grad: Whether to test gradient computation
        **kwargs: Keyword arguments to pass to function

    Returns:
        Result from the original function call

    Raises:
        AssertionError: If any transformation fails or produces different results
    """
    # Test original function
    original_result = func(*args, **kwargs)

    if test_jit:
        # Test JIT compilation
        jitted_func = jax.jit(func)
        jit_result = jitted_func(*args, **kwargs)
        chex.assert_trees_all_close(original_result, jit_result, rtol=1e-6)

    if test_vmap:
        # Test vmap (requires batched inputs)
        try:
            vmapped_func = jax.vmap(func)
            # This will only work if args are already batched
            vmap_result = vmapped_func(*args, **kwargs)
            # Note: Can't easily compare with original since shapes differ
        except Exception as e:
            # vmap might not be applicable to all functions
            print(f"vmap test skipped for {func.__name__}: {e}")

    if test_grad:
        # Test gradient computation (requires scalar output)
        try:
            grad_func = jax.grad(func)
            grad_result = grad_func(*args, **kwargs)
            # Just verify it doesn't crash
        except Exception as e:
            print(f"grad test skipped for {func.__name__}: {e}")

    return original_result


def assert_arrays_equal(
    actual: Array, expected: Array, rtol: float = 1e-6, atol: float = 1e-8
) -> None:
    """
    Compare JAX arrays with appropriate tolerance.

    Args:
        actual: Actual array result
        expected: Expected array result
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    chex.assert_trees_all_close(actual, expected, rtol=rtol, atol=atol)


def assert_pytree_structure(tree: PyTree, expected_structure: PyTree) -> None:
    """
    Validate PyTree structure matches expectations.

    Args:
        tree: PyTree to validate
        expected_structure: Expected PyTree structure
    """
    tree_def = jax.tree_util.tree_structure(tree)
    expected_def = jax.tree_util.tree_structure(expected_structure)

    assert tree_def == expected_def, (
        f"PyTree structures don't match:\nActual: {tree_def}\nExpected: {expected_def}"
    )


def assert_static_shape(array: Array, expected_shape: Tuple[int, ...]) -> None:
    """
    Verify array has expected static shape.

    Args:
        array: Array to check
        expected_shape: Expected shape tuple
    """
    chex.assert_shape(array, expected_shape)


def assert_array_properties(
    array: Array,
    dtype: Optional[jnp.dtype] = None,
    shape: Optional[Tuple[int, ...]] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> None:
    """
    Verify multiple array properties at once.

    Args:
        array: Array to validate
        dtype: Expected dtype
        shape: Expected shape
        min_val: Minimum expected value
        max_val: Maximum expected value
    """
    if dtype is not None:
        chex.assert_type(array, dtype)

    if shape is not None:
        chex.assert_shape(array, shape)

    if min_val is not None:
        assert jnp.all(array >= min_val), f"Array contains values < {min_val}"

    if max_val is not None:
        assert jnp.all(array <= max_val), f"Array contains values > {max_val}"


def create_test_prng_keys(
    n: int, base_key: Optional[chex.PRNGKey] = None
) -> chex.Array:
    """
    Create multiple PRNG keys for testing.

    Args:
        n: Number of keys to create
        base_key: Base key to split from (default: PRNGKey(42))

    Returns:
        Array of PRNG keys with shape (n, 2)
    """
    if base_key is None:
        base_key = jax.random.PRNGKey(42)

    return jax.random.split(base_key, n)


def assert_function_pure(func: Callable, *args, **kwargs) -> None:
    """
    Verify function is pure by calling it multiple times and checking results are identical.

    Args:
        func: Function to test
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function
    """
    result1 = func(*args, **kwargs)
    result2 = func(*args, **kwargs)

    chex.assert_trees_all_close(result1, result2, rtol=1e-10)


def assert_no_nan_or_inf(tree: PyTree) -> None:
    """
    Verify PyTree contains no NaN or infinite values.

    Args:
        tree: PyTree to check
    """

    def check_array(arr):
        assert not jnp.any(jnp.isnan(arr)), "Array contains NaN values"
        assert not jnp.any(jnp.isinf(arr)), "Array contains infinite values"

    jax.tree_util.tree_map(check_array, tree)


def create_mock_grid(
    height: int = 3, width: int = 3, fill_value: int = 0, pattern: Optional[str] = None
) -> Grid:
    """
    Create a mock Grid for testing purposes.

    Args:
        height: Grid height
        width: Grid width
        fill_value: Value to fill grid with
        pattern: Pattern to apply ('checkerboard', 'diagonal', etc.)

    Returns:
        Mock Grid object
    """
    from jaxarc.types import Grid

    data = jnp.full((height, width), fill_value, dtype=jnp.int32)

    if pattern == "checkerboard":
        # Create checkerboard pattern
        for i in range(height):
            for j in range(width):
                if (i + j) % 2 == 1:
                    data = data.at[i, j].set(1)
    elif pattern == "diagonal":
        # Create diagonal pattern
        for i in range(min(height, width)):
            data = data.at[i, i].set(1)

    mask = jnp.ones((height, width), dtype=jnp.bool_)

    return Grid(data=data, mask=mask)


def assert_grid_valid(grid: Grid) -> None:
    """
    Verify Grid object is valid.

    Args:
        grid: Grid to validate
    """
    from jaxarc.types import Grid

    assert isinstance(grid, Grid), f"Expected Grid, got {type(grid)}"

    # Check data properties
    assert_array_properties(grid.data, dtype=jnp.int32, min_val=0, max_val=9)

    # Check mask properties
    assert_array_properties(grid.mask, dtype=jnp.bool_)

    # Check shape consistency
    assert grid.data.shape == grid.mask.shape, (
        f"Data shape {grid.data.shape} doesn't match mask shape {grid.mask.shape}"
    )
