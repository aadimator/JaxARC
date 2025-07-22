"""
Core testing utilities for JaxARC.

This module provides common utilities, fixtures, and helper functions for testing
JAX-compatible code, Equinox modules, and JaxARC-specific functionality.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given
from hypothesis import strategies as st

T = TypeVar("T")


class JaxTestUtils:
    """Utilities for testing JAX-compatible functions and modules."""

    @staticmethod
    def assert_jax_compatible(func: Callable, *args, **kwargs) -> None:
        """
        Assert that a function is compatible with JAX transformations.

        Tests jit compilation, vmap batching, and basic functionality.
        """
        # Test basic functionality
        result = func(*args, **kwargs)

        # Test jit compilation
        jitted_func = jax.jit(func)
        jitted_result = jitted_func(*args, **kwargs)

        # Compare results (handle both arrays and pytrees)
        if isinstance(result, jnp.ndarray):
            chex.assert_trees_all_close(result, jitted_result)
        # For pytrees, use equinox tree_equal if available
        elif hasattr(eqx, "tree_equal"):
            assert eqx.tree_equal(result, jitted_result)
        else:
            chex.assert_trees_all_close(result, jitted_result)

    @staticmethod
    def assert_vmap_compatible(func: Callable, batch_args: tuple, **kwargs) -> None:
        """
        Assert that a function is compatible with vmap batching.

        Args:
            func: Function to test
            batch_args: Batched arguments (first dimension is batch)
            **kwargs: Additional keyword arguments
        """
        # Test single example
        single_args = jax.tree.map(lambda x: x[0], batch_args)
        single_result = func(*single_args, **kwargs)

        # Test vmapped version
        vmapped_func = jax.vmap(func, in_axes=(0,) * len(batch_args))
        batch_result = vmapped_func(*batch_args, **kwargs)

        # Check that batch dimension is preserved
        if isinstance(single_result, jnp.ndarray):
            expected_shape = (batch_args[0].shape[0],) + single_result.shape
            assert batch_result.shape == expected_shape

        # Check first element matches single result
        first_batch_result = jax.tree.map(lambda x: x[0], batch_result)
        if isinstance(single_result, jnp.ndarray):
            chex.assert_trees_all_close(single_result, first_batch_result)
        elif hasattr(eqx, "tree_equal"):
            assert eqx.tree_equal(single_result, first_batch_result)

    @staticmethod
    def assert_deterministic(func: Callable, *args, **kwargs) -> None:
        """Assert that a function produces deterministic results."""
        result1 = func(*args, **kwargs)
        result2 = func(*args, **kwargs)

        if isinstance(result1, jnp.ndarray):
            chex.assert_trees_all_close(result1, result2)
        elif hasattr(eqx, "tree_equal"):
            assert eqx.tree_equal(result1, result2)

    @staticmethod
    def assert_shape_preserved(
        func: Callable, input_array: jnp.ndarray, *args, **kwargs
    ) -> None:
        """Assert that a function preserves input array shape."""
        result = func(input_array, *args, **kwargs)
        if isinstance(result, jnp.ndarray) or hasattr(result, "shape"):
            assert result.shape == input_array.shape


class EquinoxTestUtils:
    """Utilities for testing Equinox modules."""

    @staticmethod
    def assert_equinox_module(module: Any) -> None:
        """Assert that an object is a valid Equinox module."""
        assert eqx.is_array_like(module), (
            f"Object {type(module)} is not an Equinox module"
        )

        # Test that it can be used in JAX transformations
        def identity(x):
            return x

        jitted_identity = jax.jit(identity)
        jitted_module = jitted_identity(module)

        # Should be able to compare with tree_equal
        assert eqx.tree_equal(module, jitted_module)

    @staticmethod
    def assert_module_validation(
        module_class: type, valid_args: dict, invalid_args: dict
    ) -> None:
        """
        Test that an Equinox module validates inputs correctly.

        Args:
            module_class: The Equinox module class to test
            valid_args: Dictionary of valid constructor arguments
            invalid_args: Dictionary of invalid constructor arguments that should raise errors
        """
        # Test valid construction
        valid_module = module_class(**valid_args)
        EquinoxTestUtils.assert_equinox_module(valid_module)

        # Test invalid construction
        for invalid_key, invalid_value in invalid_args.items():
            test_args = valid_args.copy()
            test_args[invalid_key] = invalid_value

            with pytest.raises((ValueError, TypeError)):
                module_class(**test_args)

    @staticmethod
    def assert_module_immutable(module: Any) -> None:
        """Assert that an Equinox module is immutable."""
        # Try to modify a field (should fail or create new instance)
        if hasattr(module, "__dataclass_fields__"):
            field_name = next(iter(module.__dataclass_fields__.keys()))
            original_value = getattr(module, field_name)

            # Using replace should create a new instance
            if hasattr(module, "replace"):
                new_module = module.replace(**{field_name: original_value})
                assert not eqx.tree_equal(module, new_module) or eqx.tree_equal(
                    module, new_module
                )


class MockDataGenerator:
    """Generate mock data for testing JaxARC components."""

    @staticmethod
    def create_mock_grid(
        height: int = 5, width: int = 5, key: jax.Array | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Create mock grid data and mask.

        Returns:
            Tuple of (grid_data, grid_mask)
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        # Create random grid data with ARC colors (0-9)
        grid_data = jax.random.randint(key, (height, width), 0, 10, dtype=jnp.int32)

        # Create a mask (for now, all True)
        grid_mask = jnp.ones((height, width), dtype=jnp.bool_)

        return grid_data, grid_mask

    @staticmethod
    def create_mock_task_data(
        num_train_pairs: int = 3,
        num_test_pairs: int = 1,
        max_height: int = 10,
        max_width: int = 10,
        key: jax.Array | None = None,
    ) -> dict[str, Any]:
        """
        Create mock task data for JaxArcTask.

        Returns:
            Dictionary with all required JaxArcTask fields
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        keys = jax.random.split(key, 8)

        # Create training data
        input_grids_examples = jax.random.randint(
            keys[0], (num_train_pairs, max_height, max_width), 0, 10, dtype=jnp.int32
        )
        input_masks_examples = jnp.ones(
            (num_train_pairs, max_height, max_width), dtype=jnp.bool_
        )
        output_grids_examples = jax.random.randint(
            keys[1], (num_train_pairs, max_height, max_width), 0, 10, dtype=jnp.int32
        )
        output_masks_examples = jnp.ones(
            (num_train_pairs, max_height, max_width), dtype=jnp.bool_
        )

        # Create test data
        test_input_grids = jax.random.randint(
            keys[2], (num_test_pairs, max_height, max_width), 0, 10, dtype=jnp.int32
        )
        test_input_masks = jnp.ones(
            (num_test_pairs, max_height, max_width), dtype=jnp.bool_
        )
        true_test_output_grids = jax.random.randint(
            keys[3], (num_test_pairs, max_height, max_width), 0, 10, dtype=jnp.int32
        )
        true_test_output_masks = jnp.ones(
            (num_test_pairs, max_height, max_width), dtype=jnp.bool_
        )

        return {
            "input_grids_examples": input_grids_examples,
            "input_masks_examples": input_masks_examples,
            "output_grids_examples": output_grids_examples,
            "output_masks_examples": output_masks_examples,
            "num_train_pairs": num_train_pairs,
            "test_input_grids": test_input_grids,
            "test_input_masks": test_input_masks,
            "true_test_output_grids": true_test_output_grids,
            "true_test_output_masks": true_test_output_masks,
            "num_test_pairs": num_test_pairs,
            "task_index": jnp.array(0, dtype=jnp.int32),
        }

    @staticmethod
    def create_mock_action(
        height: int = 10, width: int = 10, key: jax.Array | None = None
    ) -> dict[str, Any]:
        """
        Create mock ARCLE action data.

        Returns:
            Dictionary with all required ARCLEAction fields
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        keys = jax.random.split(key, 2)

        # Create continuous selection (0.0 to 1.0)
        selection = jax.random.uniform(keys[0], (height, width), dtype=jnp.float32)

        # Create valid operation ID (0-34)
        operation = jax.random.randint(keys[1], (), 0, 35, dtype=jnp.int32)

        return {
            "selection": selection,
            "operation": operation,
            "agent_id": 0,
            "timestamp": 0,
        }


class PropertyTestUtils:
    """Utilities for property-based testing with Hypothesis."""

    @staticmethod
    def test_function_properties(
        func: Callable,
        input_strategy: st.SearchStrategy,
        properties: list[Callable[[Any, Any], None]],
    ) -> None:
        """
        Test multiple properties of a function using Hypothesis.

        Args:
            func: Function to test
            input_strategy: Hypothesis strategy for generating inputs
            properties: List of property functions that take (input, output) and assert properties
        """

        @given(input_strategy)
        def test_properties(inputs):
            output = func(inputs)
            for prop in properties:
                prop(inputs, output)

        test_properties()

    @staticmethod
    def idempotent_property(inputs: Any, output: Any) -> None:
        """Property: function should be idempotent when applied twice."""
        # This is a template - specific implementations needed per function

    @staticmethod
    def shape_preservation_property(inputs: jnp.ndarray, output: jnp.ndarray) -> None:
        """Property: output should preserve input shape."""
        if isinstance(output, jnp.ndarray):
            assert output.shape == inputs.shape

    @staticmethod
    def type_preservation_property(inputs: jnp.ndarray, output: jnp.ndarray) -> None:
        """Property: output should preserve input dtype."""
        if isinstance(output, jnp.ndarray):
            assert output.dtype == inputs.dtype


# Convenience functions for common test patterns
def assert_jax_function(func: Callable, *args, **kwargs) -> None:
    """Convenience function to test JAX compatibility."""
    JaxTestUtils.assert_jax_compatible(func, *args, **kwargs)


def assert_equinox_module(module: Any) -> None:
    """Convenience function to test Equinox module."""
    EquinoxTestUtils.assert_equinox_module(module)


def create_test_grid(
    height: int = 5, width: int = 5
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convenience function to create test grid data."""
    return MockDataGenerator.create_mock_grid(height, width)


def create_test_task() -> dict[str, Any]:
    """Convenience function to create test task data."""
    return MockDataGenerator.create_mock_task_data()


def create_test_action() -> dict[str, Any]:
    """Convenience function to create test action data."""
    return MockDataGenerator.create_mock_action()
