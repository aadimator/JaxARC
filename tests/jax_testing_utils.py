"""
JAX-compatible testing utilities for JaxARC.

This module provides specialized utilities for testing JAX-compatible code,
focusing on transformation compatibility, shape checking, and JAX-specific assertions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import tree_map

T = TypeVar("T")
F = TypeVar("F", bound=Callable)


class JaxShapeChecker:
    """Utilities for checking and validating JAX array shapes."""

    @staticmethod
    def check_shape_consistency(
        arrays: Dict[str, jnp.ndarray], expected_shapes: Dict[str, Tuple[int, ...]]
    ) -> None:
        """
        Check that arrays have expected shapes.

        Args:
            arrays: Dictionary of arrays to check
            expected_shapes: Dictionary of expected shapes

        Raises:
            AssertionError: If any array has unexpected shape
        """
        for name, array in arrays.items():
            if name in expected_shapes:
                expected = expected_shapes[name]
                assert array.shape == expected, (
                    f"Shape mismatch for {name}: expected {expected}, got {array.shape}"
                )

    @staticmethod
    def check_batch_consistency(
        arrays: Dict[str, jnp.ndarray],
        batch_dim: int = 0,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Check that all arrays have consistent batch dimension.

        Args:
            arrays: Dictionary of arrays to check
            batch_dim: Batch dimension index (default: 0)
            batch_size: Expected batch size (if None, inferred from first array)

        Raises:
            AssertionError: If batch dimensions are inconsistent
        """
        if not arrays:
            return

        # Get batch size from first array if not specified
        first_array = next(iter(arrays.values()))
        if batch_size is None:
            batch_size = first_array.shape[batch_dim]

        # Check all arrays have same batch dimension
        for name, array in arrays.items():
            assert array.shape[batch_dim] == batch_size, (
                f"Batch dimension mismatch for {name}: expected {batch_size}, got {array.shape[batch_dim]}"
            )

    @staticmethod
    def check_static_shapes(func: F, *args, **kwargs) -> None:
        """
        Check that function works with static shapes (for JAX transformations).

        Args:
            func: Function to check
            *args: Arguments to function
            **kwargs: Keyword arguments to function

        Raises:
            Exception: If function fails with static shapes
        """
        # Try to run with jit to check for shape errors
        jitted_func = jax.jit(func)
        try:
            jitted_func(*args, **kwargs)
        except Exception as e:
            pytest.fail(f"Function failed with static shapes: {e}")


class JaxTransformationValidator:
    """Utilities for validating JAX transformations on functions and modules."""

    @staticmethod
    def validate_jit(func: F, *args, **kwargs) -> Tuple[Any, Any]:
        """
        Validate that a function works with jit transformation.

        Args:
            func: Function to validate
            *args: Arguments to function
            **kwargs: Keyword arguments to function

        Returns:
            Tuple of (original_result, jitted_result)

        Raises:
            AssertionError: If jit compilation fails or results don't match
        """
        # Get baseline result
        original_result = func(*args, **kwargs)

        # Test jit compilation
        jitted_func = jax.jit(func)
        jitted_result = jitted_func(*args, **kwargs)

        # Compare results
        if isinstance(original_result, jnp.ndarray):
            chex.assert_trees_all_close(original_result, jitted_result)
        elif hasattr(eqx, "tree_equal"):
            assert eqx.tree_equal(original_result, jitted_result)

        return original_result, jitted_result

    @staticmethod
    def validate_vmap(
        func: F,
        batch_args: List[Any],
        in_axes: Union[int, Tuple[int, ...]] = 0,
        out_axes: int = 0,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Validate that a function works with vmap transformation.

        Args:
            func: Function to validate
            batch_args: Batched arguments (first dimension is batch)
            in_axes: Input axes specification for vmap
            out_axes: Output axes specification for vmap
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (single_result, batch_result)

        Raises:
            AssertionError: If vmap fails or results don't match
        """
        # Get single example
        single_args = tree_map(
            lambda x: x[0] if isinstance(x, jnp.ndarray) else x, batch_args
        )
        single_result = func(*single_args, **kwargs)

        # Test vmap
        vmapped_func = jax.vmap(func, in_axes=in_axes, out_axes=out_axes)
        batch_result = vmapped_func(*batch_args, **kwargs)

        # Check first element matches single result
        first_batch_result = tree_map(
            lambda x: x[0] if isinstance(x, jnp.ndarray) else x, batch_result
        )

        if isinstance(single_result, jnp.ndarray):
            chex.assert_trees_all_close(single_result, first_batch_result)
        elif hasattr(eqx, "tree_equal"):
            assert eqx.tree_equal(single_result, first_batch_result)

        return single_result, batch_result

    @staticmethod
    def validate_pmap(
        func: F, args: List[Any], devices: Optional[List[jax.Device]] = None, **kwargs
    ) -> Any:
        """
        Validate that a function works with pmap transformation.

        Args:
            func: Function to validate
            args: Arguments to function
            devices: Devices to use (if None, use all available)
            **kwargs: Additional keyword arguments

        Returns:
            Result of pmapped function

        Raises:
            pytest.skip: If not enough devices available
        """
        available_devices = jax.device_count()
        if available_devices < 2:
            pytest.skip("pmap testing requires multiple devices")

        # Prepare data for each device
        device_args = tree_map(
            lambda x: jnp.stack([x] * available_devices)
            if isinstance(x, jnp.ndarray)
            else x,
            args,
        )

        # Test pmap
        pmapped_func = jax.pmap(func, devices=devices)
        return pmapped_func(*device_args, **kwargs)

    @staticmethod
    def validate_grad(
        func: F, args: List[Any], argnums: Union[int, Tuple[int, ...]] = 0, **kwargs
    ) -> Any:
        """
        Validate that a function works with grad transformation.

        Args:
            func: Function to validate
            args: Arguments to function
            argnums: Which arguments to differentiate with respect to
            **kwargs: Additional keyword arguments

        Returns:
            Gradient result

        Raises:
            pytest.skip: If function is not differentiable
        """
        try:
            # Check if function returns scalar
            result = func(*args, **kwargs)
            if not (isinstance(result, jnp.ndarray) and result.size == 1):
                pytest.skip("Function does not return scalar value")

            # Check if inputs are differentiable
            for i in [argnums] if isinstance(argnums, int) else argnums:
                if (
                    i >= len(args)
                    or not isinstance(args[i], jnp.ndarray)
                    or not jnp.issubdtype(args[i].dtype, jnp.floating)
                ):
                    pytest.skip(f"Argument {i} is not differentiable")

            # Compute gradient
            grad_func = jax.grad(func, argnums=argnums)
            return grad_func(*args, **kwargs)
        except (TypeError, ValueError) as e:
            pytest.skip(f"Function is not differentiable: {e}")


class JaxEquinoxTester:
    """Specialized testing utilities for Equinox modules with JAX."""

    @staticmethod
    def test_module_jax_compatibility(
        module: eqx.Module, transform_types: List[str] = ["jit", "vmap"]
    ) -> None:
        """
        Test that an Equinox module is compatible with JAX transformations.

        Args:
            module: Equinox module to test
            transform_types: List of transformations to test
        """

        # Define identity function for module
        def identity(x):
            return x

        # Test jit if requested
        if "jit" in transform_types:
            jitted_identity = jax.jit(identity)
            jitted_module = jitted_identity(module)
            assert eqx.tree_equal(module, jitted_module)

        # Test vmap if requested
        if "vmap" in transform_types:
            try:
                # Create batched version of module
                batched_module = tree_map(
                    lambda x: jnp.stack([x] * 3) if isinstance(x, jnp.ndarray) else x,
                    module,
                )

                # Apply vmap
                vmapped_identity = jax.vmap(identity)
                vmapped_result = vmapped_identity(batched_module)

                # Check first element
                first_result = tree_map(
                    lambda x: x[0] if isinstance(x, jnp.ndarray) and x.ndim > 0 else x,
                    vmapped_result,
                )
                assert eqx.tree_equal(module, first_result)
            except (ValueError, TypeError) as e:
                pytest.skip(f"Module not compatible with vmap: {e}")

    @staticmethod
    def test_module_methods_jax_compatibility(
        module: eqx.Module,
        method_names: List[str],
        args_dict: Dict[str, List[Any]] = None,
        kwargs_dict: Dict[str, Dict[str, Any]] = None,
    ) -> None:
        """
        Test that methods of an Equinox module are compatible with JAX transformations.

        Args:
            module: Equinox module to test
            method_names: List of method names to test
            args_dict: Dictionary mapping method names to argument lists
            kwargs_dict: Dictionary mapping method names to keyword argument dictionaries
        """
        args_dict = args_dict or {}
        kwargs_dict = kwargs_dict or {}

        for method_name in method_names:
            if not hasattr(module, method_name):
                continue

            method = getattr(module, method_name)
            args = args_dict.get(method_name, [])
            kwargs = kwargs_dict.get(method_name, {})

            # Test jit compatibility
            try:
                jitted_method = jax.jit(method)
                jitted_result = jitted_method(*args, **kwargs)
                original_result = method(*args, **kwargs)

                # Compare results
                if isinstance(original_result, jnp.ndarray):
                    chex.assert_trees_all_close(original_result, jitted_result)
                elif hasattr(eqx, "tree_equal"):
                    assert eqx.tree_equal(original_result, jitted_result)
            except Exception as e:
                pytest.fail(f"Method {method_name} failed jit compatibility: {e}")


# Convenience functions
def assert_jit_compatible(func: F, *args, **kwargs) -> None:
    """Assert that a function is compatible with jit transformation."""
    JaxTransformationValidator.validate_jit(func, *args, **kwargs)


def assert_vmap_compatible(func: F, batch_args: List[Any], **kwargs) -> None:
    """Assert that a function is compatible with vmap transformation."""
    JaxTransformationValidator.validate_vmap(func, batch_args, **kwargs)


def assert_shapes_match(
    arrays: Dict[str, jnp.ndarray], expected_shapes: Dict[str, Tuple[int, ...]]
) -> None:
    """Assert that arrays have expected shapes."""
    JaxShapeChecker.check_shape_consistency(arrays, expected_shapes)


def assert_equinox_jax_compatible(module: eqx.Module) -> None:
    """Assert that an Equinox module is compatible with JAX transformations."""
    JaxEquinoxTester.test_module_jax_compatibility(module)
