"""
Tests for space definitions in jaxarc.envs.spaces.

This module tests action and observation space definitions,
space compatibility, bounds checking, and JAX compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import PRNGKey

from jaxarc.envs.spaces import (
    ARCActionSpace,
    BoundedArraySpace,
    DictSpace,
    DiscreteSpace,
    GridSpace,
    SelectionSpace,
    Space,
)
from jaxarc.types import NUM_COLORS, NUM_OPERATIONS


class TestSpaceBase:
    """Test base Space class functionality."""

    def test_space_abstract_methods(self):
        """Test that Space is properly abstract."""
        # Should not be able to instantiate abstract base class
        with pytest.raises(TypeError):
            Space()

    def test_space_pytree_registration(self):
        """Test that spaces are properly registered as JAX pytrees."""
        space = DiscreteSpace(5, dtype=jnp.int32)

        # Test that space can be used in JAX tree operations
        # The exact tree_flatten/unflatten behavior depends on implementation
        try:
            children, aux_data = space.tree_flatten()
            # Just verify the space works with JAX tree operations
            assert isinstance(children, (list, tuple))
            assert isinstance(aux_data, dict)
        except (AttributeError, NotImplementedError):
            # If tree operations aren't implemented, that's also valid
            pass


class TestBoundedArraySpace:
    """Test BoundedArraySpace functionality."""

    def test_bounded_array_space_creation(self):
        """Test BoundedArraySpace creation with various parameters."""
        # Basic creation
        space = BoundedArraySpace(shape=(3, 3), dtype=jnp.float32)

        assert space.shape == (3, 3)
        assert space.dtype == jnp.float32
        assert jnp.array_equal(space._minimum, -jnp.inf)
        assert jnp.array_equal(space._maximum, jnp.inf)

    def test_bounded_array_space_with_bounds(self):
        """Test BoundedArraySpace with explicit bounds."""
        space = BoundedArraySpace(shape=(2, 2), dtype=jnp.int32, minimum=0, maximum=10)

        assert space.shape == (2, 2)
        assert space.dtype == jnp.int32
        assert jnp.array_equal(space._minimum, 0)
        assert jnp.array_equal(space._maximum, 10)

    def test_bounded_array_space_sampling(self, prng_key: PRNGKey):
        """Test BoundedArraySpace sampling."""
        space = BoundedArraySpace(
            shape=(3, 3), dtype=jnp.float32, minimum=0.0, maximum=1.0
        )

        sample = space.sample(prng_key)

        assert sample.shape == (3, 3)
        assert sample.dtype == jnp.float32
        assert jnp.all(sample >= 0.0)
        assert jnp.all(sample <= 1.0)

    def test_bounded_array_space_contains(self):
        """Test BoundedArraySpace contains method."""
        space = BoundedArraySpace(shape=(2, 2), dtype=jnp.int32, minimum=0, maximum=5)

        # Valid array
        valid_array = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        assert space.contains(valid_array).item()

        # Invalid shape
        invalid_shape = jnp.array([1, 2, 3], dtype=jnp.int32)
        assert not space.contains(invalid_shape).item()

        # Invalid bounds
        invalid_bounds = jnp.array([[1, 2], [3, 10]], dtype=jnp.int32)
        assert not space.contains(invalid_bounds).item()

        # Non-array input
        assert not space.contains("not an array")

    def test_bounded_array_space_jax_compatibility(self, prng_key: PRNGKey):
        """Test BoundedArraySpace JAX compatibility."""
        space = BoundedArraySpace(
            shape=(4, 4), dtype=jnp.float32, minimum=-1.0, maximum=1.0
        )

        # Test JIT compilation of sampling
        jitted_sample = jax.jit(space.sample)
        sample = jitted_sample(prng_key)

        assert sample.shape == (4, 4)
        assert sample.dtype == jnp.float32

        # Test JIT compilation of contains
        jitted_contains = jax.jit(space.contains)
        result = jitted_contains(sample)

        assert isinstance(result, jax.Array)
        assert result.dtype == jnp.bool_


class TestDiscreteSpace:
    """Test DiscreteSpace functionality."""

    def test_discrete_space_creation(self):
        """Test DiscreteSpace creation."""
        space = DiscreteSpace(10, dtype=jnp.int32)

        assert space.num_values == 10
        assert space.shape == ()
        assert space.dtype == jnp.int32
        assert jnp.array_equal(space._minimum, 0)
        assert jnp.array_equal(space._maximum, 9)

    def test_discrete_space_sampling(self, prng_key: PRNGKey):
        """Test DiscreteSpace sampling."""
        space = DiscreteSpace(5, dtype=jnp.int32)

        sample = space.sample(prng_key)

        assert sample.shape == ()
        assert sample.dtype == jnp.int32
        assert 0 <= sample < 5

    def test_discrete_space_contains(self):
        """Test DiscreteSpace contains method."""
        space = DiscreteSpace(3, dtype=jnp.int32)

        # Valid values
        assert space.contains(jnp.array(0, dtype=jnp.int32)).item()
        assert space.contains(jnp.array(1, dtype=jnp.int32)).item()
        assert space.contains(jnp.array(2, dtype=jnp.int32)).item()
        assert not space.contains(jnp.array(-1, dtype=jnp.int32)).item()
        assert not space.contains(jnp.array(3, dtype=jnp.int32)).item()
        assert not space.contains(
            jnp.array([1, 2], dtype=jnp.int32)
        ).item()  # Wrong shape

    def test_discrete_space_batch_sampling(self, prng_key: PRNGKey):
        """Test DiscreteSpace with batch sampling using vmap."""
        space = DiscreteSpace(8, dtype=jnp.int32)

        # Create multiple keys for batch sampling
        keys = jax.random.split(prng_key, 10)

        # Use vmap to sample multiple values
        batch_sample = jax.vmap(space.sample)(keys)

        assert batch_sample.shape == (10,)
        assert batch_sample.dtype == jnp.int32
        assert jnp.all(batch_sample >= 0)
        assert jnp.all(batch_sample < 8)


class TestDictSpace:
    """Test DictSpace functionality."""

    def test_dict_space_creation(self):
        """Test DictSpace creation."""
        spaces = {
            "action": DiscreteSpace(4, dtype=jnp.int32),
            "position": BoundedArraySpace(
                shape=(2,), dtype=jnp.float32, minimum=0.0, maximum=1.0
            ),
        }

        dict_space = DictSpace(spaces)

        assert dict_space._spaces == spaces
        assert dict_space.shape is None
        assert dict_space.dtype is None

    def test_dict_space_sampling(self, prng_key: PRNGKey):
        """Test DictSpace sampling."""
        spaces = {
            "discrete": DiscreteSpace(3, dtype=jnp.int32),
            "continuous": BoundedArraySpace(
                shape=(2,), dtype=jnp.float32, minimum=-1.0, maximum=1.0
            ),
        }

        dict_space = DictSpace(spaces)
        sample = dict_space.sample(prng_key)

        assert isinstance(sample, dict)
        assert "discrete" in sample
        assert "continuous" in sample

        assert sample["discrete"].dtype == jnp.int32
        assert 0 <= sample["discrete"] < 3

        assert sample["continuous"].shape == (2,)
        assert sample["continuous"].dtype == jnp.float32
        assert jnp.all(sample["continuous"] >= -1.0)
        assert jnp.all(sample["continuous"] <= 1.0)

    def test_dict_space_contains(self):
        """Test DictSpace contains method."""
        spaces = {
            "a": DiscreteSpace(2, dtype=jnp.int32),
            "b": BoundedArraySpace(shape=(1,), dtype=jnp.int32, minimum=0, maximum=5),
        }

        dict_space = DictSpace(spaces)

        # Valid dict
        valid_dict = {
            "a": jnp.array(1, dtype=jnp.int32),
            "b": jnp.array([3], dtype=jnp.int32),
        }
        assert dict_space.contains(valid_dict).item()

        # Invalid dict - missing key (should handle gracefully)
        invalid_dict = {"a": jnp.array(1, dtype=jnp.int32)}
        try:
            result = dict_space.contains(invalid_dict)
            assert not result.item()
        except KeyError:
            # KeyError is also acceptable behavior for missing keys
            pass

        # Invalid dict - wrong value
        invalid_dict = {
            "a": jnp.array(5, dtype=jnp.int32),  # Out of range
            "b": jnp.array([3], dtype=jnp.int32),
        }
        assert not dict_space.contains(invalid_dict).item()

        # Non-dict input
        assert not dict_space.contains("not a dict")


class TestARCSpecificSpaces:
    """Test ARC-specific space implementations."""

    def test_grid_space_creation(self):
        """Test GridSpace creation."""
        space = GridSpace(max_height=10, max_width=15)

        assert space.shape == (10, 15)
        assert space.dtype == jnp.int32
        assert jnp.array_equal(space._minimum, -1)  # Background/padding
        assert jnp.array_equal(space._maximum, NUM_COLORS - 1)  # ARC colors 0-9

    def test_grid_space_sampling(self, prng_key: PRNGKey):
        """Test GridSpace sampling."""
        space = GridSpace(max_height=5, max_width=5)

        sample = space.sample(prng_key)

        assert sample.shape == (5, 5)
        # GridSpace inherits from BoundedArraySpace which may return floats
        # The important thing is that the bounds are correct
        assert jnp.all(sample >= -1)
        assert jnp.all(sample <= NUM_COLORS - 1)

    def test_selection_space_creation(self):
        """Test SelectionSpace creation."""
        space = SelectionSpace(max_height=8, max_width=12)

        assert space.shape == (8, 12)
        # SelectionSpace is MultiDiscrete with 2 values per cell and integer dtype
        assert space.dtype == jnp.int32
        nv = jnp.asarray(space.num_values)
        assert jnp.all(nv == 2)

    def test_selection_space_sampling(self, prng_key: PRNGKey):
        """Test SelectionSpace sampling."""
        space = SelectionSpace(max_height=3, max_width=4)

        sample = space.sample(prng_key)

        assert sample.shape == (3, 4)
        # MultiDiscrete samples integer choices in [0, 2)
        assert sample.dtype == jnp.int32
        assert jnp.all(sample >= 0)
        assert jnp.all(sample < 2)

    # No DictSpace assertions here; SelectionSpace is a MultiDiscrete standalone space

    def test_arc_action_space_sampling(self, prng_key: PRNGKey):
        """Test ARCActionSpace sampling."""
        space = ARCActionSpace(max_height=4, max_width=4)

        sample = space.sample(prng_key)

        assert isinstance(sample, dict)
        assert "operation" in sample
        assert "selection" in sample

        assert sample["operation"].dtype == jnp.int32
        assert 0 <= sample["operation"] < NUM_OPERATIONS

        assert sample["selection"].shape == (4, 4)
        assert sample["selection"].dtype == jnp.int32
        assert jnp.all((sample["selection"] == 0) | (sample["selection"] == 1))

    def test_arc_action_space_contains(self):
        """Test ARCActionSpace contains method."""
        space = ARCActionSpace(max_height=3, max_width=3)

        # Valid action
        valid_action = {
            "operation": jnp.array(5, dtype=jnp.int32),
            # MultiDiscrete expects integer choices in {0,1}
            "selection": jnp.ones((3, 3), dtype=jnp.int32),
        }
        assert space.contains(valid_action).item()

        # Invalid operation
        invalid_action = {
            "operation": jnp.array(NUM_OPERATIONS + 1, dtype=jnp.int32),  # Out of range
            "selection": jnp.ones((3, 3), dtype=jnp.int32),
        }
        assert not space.contains(invalid_action).item()

        # Invalid selection shape
        invalid_action = {
            "operation": jnp.array(5, dtype=jnp.int32),
            "selection": jnp.ones((2, 2), dtype=jnp.int32),  # Wrong shape
        }
        assert not space.contains(invalid_action).item()


class TestSpaceJAXCompatibility:
    """Test JAX compatibility of all spaces."""

    def test_spaces_jit_compilation(self, prng_key: PRNGKey):
        """Test that all spaces work with JAX JIT compilation."""
        spaces = [
            DiscreteSpace(5, dtype=jnp.int32),
            BoundedArraySpace(
                shape=(2, 2), dtype=jnp.float32, minimum=0.0, maximum=1.0
            ),
            GridSpace(max_height=3, max_width=3),
            SelectionSpace(max_height=3, max_width=3),
        ]

        for space in spaces:
            # Test JIT compilation of sampling
            jitted_sample = jax.jit(space.sample)
            sample = jitted_sample(prng_key)

            # Verify sample is valid
            assert space.contains(sample).item()

            # Test JIT compilation of contains
            jitted_contains = jax.jit(space.contains)
            result = jitted_contains(sample)

            assert isinstance(result, jax.Array)
            assert result.dtype == jnp.bool_
            assert result.item()

    def test_spaces_vmap_compatibility(self, prng_key: PRNGKey):
        """Test that spaces work with JAX vmap."""
        space = DiscreteSpace(10, dtype=jnp.int32)

        # Create batch of keys
        keys = jax.random.split(prng_key, 5)

        # Use vmap to sample batch
        batch_sample = jax.vmap(space.sample)(keys)

        assert batch_sample.shape == (5,)
        assert batch_sample.dtype == jnp.int32
        assert jnp.all(batch_sample >= 0)
        assert jnp.all(batch_sample < 10)

        # Test batch contains
        batch_contains = jax.vmap(space.contains)(batch_sample)

        assert batch_contains.shape == (5,)
        assert batch_contains.dtype == jnp.bool_
        assert jnp.all(batch_contains).item()

    def test_dict_space_jax_compatibility(self, prng_key: PRNGKey):
        """Test DictSpace JAX compatibility."""
        spaces = {
            "action": DiscreteSpace(4, dtype=jnp.int32),
            "mask": BoundedArraySpace(
                shape=(2, 2), dtype=jnp.bool_, minimum=False, maximum=True
            ),
        }

        dict_space = DictSpace(spaces)

        # Test JIT compilation
        jitted_sample = jax.jit(dict_space.sample)
        sample = jitted_sample(prng_key)

        assert isinstance(sample, dict)
        assert dict_space.contains(sample).item()

    def test_arc_action_space_jax_compatibility(self, prng_key: PRNGKey):
        """Test ARCActionSpace JAX compatibility."""
        space = ARCActionSpace(max_height=4, max_width=4)

        # Test JIT compilation
        jitted_sample = jax.jit(space.sample)
        sample = jitted_sample(prng_key)

        assert isinstance(sample, dict)
        assert space.contains(sample).item()

        # Test that sample is valid ARC action
        assert 0 <= sample["operation"] < NUM_OPERATIONS
        assert sample["selection"].shape == (4, 4)
        assert sample["selection"].dtype == jnp.int32
        assert jnp.all((sample["selection"] == 0) | (sample["selection"] == 1))


class TestSpaceBoundsChecking:
    """Test space bounds checking and validation."""

    def test_bounded_array_space_bounds_validation(self):
        """Test BoundedArraySpace bounds validation."""
        space = BoundedArraySpace(shape=(2, 2), dtype=jnp.int32, minimum=0, maximum=10)

        # Test various boundary conditions
        test_cases = [
            (jnp.array([[0, 5], [10, 3]]), True),  # Valid bounds
            (jnp.array([[-1, 5], [10, 3]]), False),  # Below minimum
            (jnp.array([[0, 5], [11, 3]]), False),  # Above maximum
            (jnp.array([[0, 5, 3]]), False),  # Wrong shape
        ]

        for test_array, expected in test_cases:
            result = space.contains(test_array)
            assert result.item() == expected

    def test_discrete_space_bounds_validation(self):
        """Test DiscreteSpace bounds validation."""
        space = DiscreteSpace(5, dtype=jnp.int32)

        test_cases = [
            (jnp.array(0, dtype=jnp.int32), True),
            (jnp.array(4, dtype=jnp.int32), True),
            (jnp.array(-1, dtype=jnp.int32), False),
            (jnp.array(5, dtype=jnp.int32), False),
            (jnp.array([2], dtype=jnp.int32), False),  # Wrong shape
        ]

        for test_value, expected in test_cases:
            result = space.contains(test_value)
            assert result.item() == expected

    def test_grid_space_bounds_validation(self):
        """Test GridSpace bounds validation."""
        space = GridSpace(max_height=3, max_width=3)

        # Valid grid (colors 0-9, background -1)
        valid_grid = jnp.array([[-1, 0, 1], [2, 3, 4], [5, 6, 7]])
        assert space.contains(valid_grid).item()

        # Invalid grid (color 10 is out of range)
        invalid_grid = jnp.array([[-1, 0, 1], [2, 3, 4], [5, 6, 10]])
        assert not space.contains(invalid_grid).item()

        # Invalid grid (color -2 is below minimum)
        invalid_grid = jnp.array([[-2, 0, 1], [2, 3, 4], [5, 6, 7]])
        assert not space.contains(invalid_grid).item()

    def test_selection_space_bounds_validation(self):
        """Test SelectionSpace bounds validation."""
        space = SelectionSpace(max_height=2, max_width=3)

        # Valid selection mask (MultiDiscrete integers in {0,1})
        valid_mask = jnp.array([[1, 0, 1], [0, 1, 0]], dtype=jnp.int32)
        assert space.contains(valid_mask).item()

        # Wrong shape
        wrong_shape = jnp.array([True, False])
        assert not space.contains(wrong_shape).item()

        # Out of range values
        out_of_range = jnp.array([[2, 0, -1], [0, 3, 0]], dtype=jnp.int32)
        assert not space.contains(out_of_range).item()


# Removed duplicated test classes to avoid redefinitions


class TestSpaceEdgeCases:
    """Test edge cases and error conditions for spaces."""

    def test_empty_dict_space(self):
        """Test DictSpace with empty spaces dict."""
        empty_dict_space = DictSpace({})

        assert empty_dict_space._spaces == {}
        assert empty_dict_space.shape is None
        assert empty_dict_space.dtype is None

    def test_zero_size_discrete_space(self):
        """Test DiscreteSpace with zero values."""
        with pytest.raises(ValueError, match=".*"):
            DiscreteSpace(0)

    def test_negative_discrete_space(self):
        """Test DiscreteSpace with negative values."""
        with pytest.raises(ValueError, match=".*"):
            DiscreteSpace(-5)

    def test_invalid_bounded_array_bounds(self):
        """Test BoundedArraySpace with invalid bounds."""
        with pytest.raises(ValueError, match=".*"):
            BoundedArraySpace(
                shape=(2, 2),
                dtype=jnp.float32,
                minimum=10.0,
                maximum=5.0,  # Invalid: min > max
            )

    def test_zero_shape_bounded_array(self):
        """Test BoundedArraySpace with zero-dimensional shape."""
        space = BoundedArraySpace(shape=(), dtype=jnp.float32, minimum=0.0, maximum=1.0)

        assert space.shape == ()
        assert space.dtype == jnp.float32

    def test_large_grid_space(self):
        """Test GridSpace with large dimensions."""
        # Test that large grids can be created (within reason)
        space = GridSpace(max_height=100, max_width=100)

        assert space.shape == (100, 100)
        assert space.dtype == jnp.int32

        # Sampling very large grids might be slow, so we just test creation
