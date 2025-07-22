"""
Property-based testing utilities using Hypothesis for JaxARC.

This module provides Hypothesis strategies and utilities specifically designed
for testing JAX arrays, Equinox modules, and JaxARC-specific data structures.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp


# Basic JAX array strategies
def jax_arrays(
    shape: Union[Tuple[int, ...], st.SearchStrategy[Tuple[int, ...]], None] = None,
    dtype: jnp.dtype = jnp.float32,
    elements: Optional[st.SearchStrategy] = None,
    min_value: Optional[Union[float, int]] = None,
    max_value: Optional[Union[float, int]] = None,
) -> st.SearchStrategy[jnp.ndarray]:
    """
    Generate JAX arrays with specified constraints.

    Args:
        shape: Array shape or strategy to generate shapes
        dtype: JAX dtype for the array
        elements: Strategy for array elements (overrides min/max_value)
        min_value: Minimum value for array elements
        max_value: Maximum value for array elements

    Returns:
        Strategy that generates JAX arrays
    """
    if shape is None:
        # Generate reasonable shapes for testing
        shape = st.tuples(
            st.integers(min_value=1, max_value=10),
            st.integers(min_value=1, max_value=10),
        )

    if elements is None:
        if dtype == jnp.int32:
            elements = st.integers(
                min_value=min_value or -100, max_value=max_value or 100
            )
        elif dtype == jnp.float32:
            elements = st.floats(
                min_value=min_value or -10.0,
                max_value=max_value or 10.0,
                allow_nan=False,
                allow_infinity=False,
            )
        elif dtype == jnp.bool_:
            elements = st.booleans()
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    return hnp.arrays(dtype=dtype, shape=shape, elements=elements).map(jnp.array)


# ARC-specific strategies
def arc_grid_arrays(
    max_height: int = 30, max_width: int = 30, min_height: int = 1, min_width: int = 1
) -> st.SearchStrategy[jnp.ndarray]:
    """
    Generate valid ARC grid arrays with colors 0-9.

    Args:
        max_height: Maximum grid height
        max_width: Maximum grid width
        min_height: Minimum grid height
        min_width: Minimum grid width

    Returns:
        Strategy that generates ARC grid arrays
    """
    shape_strategy = st.tuples(
        st.integers(min_value=min_height, max_value=max_height),
        st.integers(min_value=min_width, max_value=max_width),
    )

    # Ensure we only generate valid ARC colors (0-9)
    elements = st.integers(min_value=0, max_value=9)

    return hnp.arrays(dtype=np.int32, shape=shape_strategy, elements=elements).map(
        jnp.array
    )


def arc_mask_arrays(
    shape: Union[Tuple[int, ...], st.SearchStrategy[Tuple[int, ...]], None] = None,
) -> st.SearchStrategy[jnp.ndarray]:
    """
    Generate boolean mask arrays for ARC grids.

    Args:
        shape: Array shape or strategy to generate shapes

    Returns:
        Strategy that generates boolean mask arrays
    """
    if shape is None:
        shape = st.tuples(
            st.integers(min_value=1, max_value=30),
            st.integers(min_value=1, max_value=30),
        )

    return jax_arrays(shape=shape, dtype=jnp.bool_, elements=st.booleans())


def arc_selection_arrays(
    shape: Union[Tuple[int, ...], st.SearchStrategy[Tuple[int, ...]], None] = None,
) -> st.SearchStrategy[jnp.ndarray]:
    """
    Generate continuous selection arrays for ARCLE actions.

    Args:
        shape: Array shape or strategy to generate shapes

    Returns:
        Strategy that generates selection arrays with values in [0, 1]
    """
    if shape is None:
        shape = st.tuples(
            st.integers(min_value=1, max_value=30),
            st.integers(min_value=1, max_value=30),
        )

    # Ensure we only generate values in [0, 1]
    elements = st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    )

    return hnp.arrays(dtype=np.float32, shape=shape, elements=elements).map(jnp.array)


def arc_operation_ids() -> st.SearchStrategy[jnp.ndarray]:
    """
    Generate valid ARCLE operation IDs (0-34).

    Returns:
        Strategy that generates operation ID arrays
    """
    return st.integers(min_value=0, max_value=34).map(
        lambda x: jnp.array(x, dtype=jnp.int32)
    )


def arc_task_indices() -> st.SearchStrategy[jnp.ndarray]:
    """
    Generate task index arrays.

    Returns:
        Strategy that generates task index arrays
    """
    return st.integers(min_value=0, max_value=1000).map(
        lambda x: jnp.array(x, dtype=jnp.int32)
    )


# Composite strategies for JaxARC data structures
@st.composite
def arc_grids(draw) -> Dict[str, jnp.ndarray]:
    """
    Generate Grid data (data + mask with matching shapes).

    Returns:
        Dictionary with 'data' and 'mask' keys
    """
    # Generate shape first
    height = draw(st.integers(min_value=1, max_value=30))
    width = draw(st.integers(min_value=1, max_value=30))
    shape = (height, width)

    # Generate data with valid ARC colors (0-9)
    elements = st.integers(min_value=0, max_value=9)
    data = draw(hnp.arrays(dtype=np.int32, shape=shape, elements=elements)).astype(
        jnp.int32
    )

    # Generate mask
    mask = draw(hnp.arrays(dtype=np.bool_, shape=shape, elements=st.booleans())).astype(
        jnp.bool_
    )

    return {"data": jnp.array(data), "mask": jnp.array(mask)}


@st.composite
def arc_task_pairs(draw, max_pairs: int = 5) -> Dict[str, Any]:
    """
    Generate TaskPair data for JaxArcTask.

    Args:
        max_pairs: Maximum number of training/test pairs

    Returns:
        Dictionary with all JaxArcTask fields
    """
    # Generate dimensions
    num_train_pairs = draw(st.integers(min_value=1, max_value=max_pairs))
    num_test_pairs = draw(st.integers(min_value=1, max_value=max_pairs))
    max_height = draw(st.integers(min_value=5, max_value=20))
    max_width = draw(st.integers(min_value=5, max_value=20))

    # Generate training data
    input_grids_examples = draw(
        jax_arrays(
            shape=(num_train_pairs, max_height, max_width),
            dtype=jnp.int32,
            min_value=0,
            max_value=9,
        )
    )
    input_masks_examples = draw(
        jax_arrays(shape=(num_train_pairs, max_height, max_width), dtype=jnp.bool_)
    )
    output_grids_examples = draw(
        jax_arrays(
            shape=(num_train_pairs, max_height, max_width),
            dtype=jnp.int32,
            min_value=0,
            max_value=9,
        )
    )
    output_masks_examples = draw(
        jax_arrays(shape=(num_train_pairs, max_height, max_width), dtype=jnp.bool_)
    )

    # Generate test data
    test_input_grids = draw(
        jax_arrays(
            shape=(num_test_pairs, max_height, max_width),
            dtype=jnp.int32,
            min_value=0,
            max_value=9,
        )
    )
    test_input_masks = draw(
        jax_arrays(shape=(num_test_pairs, max_height, max_width), dtype=jnp.bool_)
    )
    true_test_output_grids = draw(
        jax_arrays(
            shape=(num_test_pairs, max_height, max_width),
            dtype=jnp.int32,
            min_value=0,
            max_value=9,
        )
    )
    true_test_output_masks = draw(
        jax_arrays(shape=(num_test_pairs, max_height, max_width), dtype=jnp.bool_)
    )

    # Generate task index
    task_index = draw(arc_task_indices())

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
        "task_index": task_index,
    }


@st.composite
def arcle_actions(draw) -> Dict[str, Any]:
    """
    Generate ARCLEAction data.

    Returns:
        Dictionary with all ARCLEAction fields
    """
    # Generate dimensions
    height = draw(st.integers(min_value=1, max_value=30))
    width = draw(st.integers(min_value=1, max_value=30))

    # Generate action components
    selection = draw(arc_selection_arrays(shape=(height, width)))
    operation = draw(arc_operation_ids())
    agent_id = draw(st.integers(min_value=0, max_value=10))
    timestamp = draw(st.integers(min_value=0, max_value=1000))

    return {
        "selection": selection,
        "operation": operation,
        "agent_id": agent_id,
        "timestamp": timestamp,
    }


# Batch strategies for testing vmap compatibility
def batched_arrays(
    base_strategy: st.SearchStrategy[jnp.ndarray],
    batch_size: Union[int, st.SearchStrategy[int], None] = None,
) -> st.SearchStrategy[jnp.ndarray]:
    """
    Create batched versions of arrays for vmap testing.

    Args:
        base_strategy: Strategy for generating base arrays
        batch_size: Batch size or strategy to generate batch sizes

    Returns:
        Strategy that generates batched arrays
    """
    if batch_size is None:
        batch_size = st.integers(min_value=1, max_value=5)

    @st.composite
    def _batched_arrays(draw):
        base_array = draw(base_strategy)
        if isinstance(batch_size, int):
            size = batch_size
        else:
            size = draw(batch_size)

        # Stack multiple copies to create batch
        return jnp.stack([base_array] * size, axis=0)

    return _batched_arrays()


# Property testing utilities for JAX functions
class JaxPropertyTester:
    """Property-based testing utilities for JAX functions."""

    @staticmethod
    def assert_shape_preserved(
        func: Callable, input_array: jnp.ndarray, *args, **kwargs
    ) -> None:
        """Assert that function preserves input array shape."""
        result = func(input_array, *args, **kwargs)
        if isinstance(result, jnp.ndarray):
            assert result.shape == input_array.shape, (
                f"Shape not preserved: input {input_array.shape}, output {result.shape}"
            )

    @staticmethod
    def assert_dtype_preserved(
        func: Callable, input_array: jnp.ndarray, *args, **kwargs
    ) -> None:
        """Assert that function preserves input array dtype."""
        result = func(input_array, *args, **kwargs)
        if isinstance(result, jnp.ndarray):
            assert result.dtype == input_array.dtype, (
                f"Dtype not preserved: input {input_array.dtype}, output {result.dtype}"
            )

    @staticmethod
    def assert_bounds_preserved(
        func: Callable,
        input_array: jnp.ndarray,
        lower_bound: Union[float, int],
        upper_bound: Union[float, int],
        *args,
        **kwargs,
    ) -> None:
        """Assert that function preserves value bounds."""
        result = func(input_array, *args, **kwargs)
        if isinstance(result, jnp.ndarray):
            assert jnp.all(result >= lower_bound) and jnp.all(result <= upper_bound), (
                f"Bounds not preserved: min {jnp.min(result)}, max {jnp.max(result)}"
            )

    @staticmethod
    def assert_deterministic(func: Callable, *args, **kwargs) -> None:
        """Assert that function is deterministic (same inputs -> same outputs)."""
        result1 = func(*args, **kwargs)
        result2 = func(*args, **kwargs)

        if isinstance(result1, jnp.ndarray):
            assert jnp.array_equal(result1, result2), "Function is not deterministic"
        else:
            assert result1 == result2, "Function is not deterministic"

    @staticmethod
    def assert_jit_compatible(func: Callable, *args, **kwargs) -> None:
        """Assert that function is compatible with JAX JIT compilation."""
        # Get baseline result
        baseline = func(*args, **kwargs)

        # Test jit compilation
        jitted_func = jax.jit(func)
        jitted_result = jitted_func(*args, **kwargs)

        # Compare results
        if isinstance(baseline, jnp.ndarray):
            assert jnp.allclose(baseline, jitted_result), (
                "JIT compilation changed function behavior"
            )
        else:
            assert baseline == jitted_result, (
                "JIT compilation changed function behavior"
            )


# Convenience functions for common property tests
def test_grid_property(property_func: Callable, num_examples: int = 100):
    """Test a property on randomly generated grids."""
    from hypothesis import given, settings

    @settings(max_examples=num_examples)
    @given(arc_grids())
    def test_property(grid_data):
        property_func(grid_data["data"], grid_data["mask"])

    # Run the test
    test_property()


def test_action_property(property_func: Callable, num_examples: int = 100):
    """Test a property on randomly generated actions."""
    from hypothesis import given, settings

    @settings(max_examples=num_examples)
    @given(arcle_actions())
    def test_property(action_data):
        property_func(action_data)

    # Run the test
    test_property()


def test_jax_function_properties(
    func: Callable,
    input_strategy: st.SearchStrategy,
    properties: List[Callable],
    num_examples: int = 100,
):
    """
    Test multiple properties of a JAX function using Hypothesis.

    Args:
        func: Function to test
        input_strategy: Strategy for generating inputs
        properties: List of property functions to test
        num_examples: Number of examples to test
    """
    from hypothesis import given, settings

    @settings(max_examples=num_examples)
    @given(input_strategy)
    def test_properties(input_data):
        result = func(input_data)
        for prop in properties:
            prop(func, input_data, result)

    # Run the test
    test_properties()
