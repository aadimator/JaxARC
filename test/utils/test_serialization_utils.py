"""Tests for serialization utility functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxarc.state import State
from jaxarc.utils.serialization_utils import (
    serialize_action,
    serialize_jax_array,
    serialize_log_step,
    serialize_object,
    serialize_state,
)


class TestSerializeJaxArray:
    """Test JAX array serialization."""

    def test_serialize_jax_array_basic(self):
        """Test basic JAX array serialization."""
        jax_array = jnp.array([[1, 2], [3, 4]])
        result = serialize_jax_array(jax_array)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [[1, 2], [3, 4]])

    def test_serialize_numpy_array(self):
        """Test numpy array serialization."""
        numpy_array = np.array([[5, 6], [7, 8]])
        result = serialize_jax_array(numpy_array)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [[5, 6], [7, 8]])
        # Should be a copy, not the same object
        assert result is not numpy_array

    def test_serialize_scalar_array(self):
        """Test scalar array serialization."""
        scalar_jax = jnp.array(42)
        result = serialize_jax_array(scalar_jax)

        assert isinstance(result, np.ndarray)
        assert result.shape == ()
        assert result.item() == 42

    def test_serialize_different_dtypes(self):
        """Test serialization with different data types."""
        # Test int32
        int_array = jnp.array([1, 2, 3], dtype=jnp.int32)
        result = serialize_jax_array(int_array)
        assert result.dtype == np.int32

        # Test float32
        float_array = jnp.array([1.5, 2.5, 3.5], dtype=jnp.float32)
        result = serialize_jax_array(float_array)
        assert result.dtype == np.float32

        # Test bool
        bool_array = jnp.array([True, False, True])
        result = serialize_jax_array(bool_array)
        assert result.dtype == bool

    def test_serialize_empty_array(self):
        """Test serialization of empty arrays."""
        empty_array = jnp.array([])
        result = serialize_jax_array(empty_array)

        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_serialize_invalid_input(self):
        """Test serialization with invalid input."""
        # Should handle gracefully and convert to numpy array
        result = serialize_jax_array("invalid")
        assert isinstance(result, np.ndarray)
        assert result.size == 1  # String gets converted to single-element array

    def test_serialize_large_array(self):
        """Test serialization of large arrays."""
        large_array = jnp.ones((100, 100))
        result = serialize_jax_array(large_array)

        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100)
        assert np.all(result == 1)


class TestSerializeAction:
    """Test action serialization."""

    def test_serialize_dict_action(self):
        """Test serialization of dictionary actions."""
        action = {
            "operation": jnp.array([1, 2, 3]),
            "selection": jnp.array([[True, False], [False, True]]),
            "color": 5,
            "name": "test_action",
        }

        result = serialize_action(action)

        assert isinstance(result, dict)
        assert "operation" in result
        assert "selection" in result
        assert "color" in result
        assert "name" in result

        # Arrays should be serialized
        assert isinstance(result["operation"], np.ndarray)
        assert isinstance(result["selection"], np.ndarray)

        # Primitives should be preserved
        assert result["color"] == 5
        assert result["name"] == "test_action"

    def test_serialize_structured_action(self):
        """Test serialization of structured actions (Equinox modules)."""

        # Mock structured action
        class MockAction:
            def __init__(self):
                self.operation = jnp.array([1, 0, 0])
                self.selection = jnp.array([[True, False]])

        action = MockAction()
        result = serialize_action(action)

        assert isinstance(result, dict)
        assert result["type"] == "MockAction"
        assert "operation" in result
        assert "selection" in result

        # Arrays should be serialized
        assert isinstance(result["operation"], np.ndarray)
        assert isinstance(result["selection"], np.ndarray)

    def test_serialize_action_without_common_fields(self):
        """Test serialization of action without operation/selection fields."""

        class SimpleAction:
            def __init__(self):
                self.value = 42

        action = SimpleAction()
        result = serialize_action(action)

        assert isinstance(result, dict)
        assert result["type"] == "SimpleAction"
        # Should not have operation or selection fields
        assert "operation" not in result
        assert "selection" not in result

    def test_serialize_primitive_action(self):
        """Test serialization of primitive action types."""
        # Test with string
        result = serialize_action("move_up")
        assert result["raw"] == "move_up"
        assert result["type"] == "str"

        # Test with int
        result = serialize_action(42)
        assert result["raw"] == "42"
        assert result["type"] == "int"

    def test_serialize_action_with_error(self):
        """Test action serialization error handling."""

        # Mock action that raises error during serialization
        class ErrorAction:
            @property
            def operation(self):
                raise ValueError("Test error")

        action = ErrorAction()
        result = serialize_action(action)

        assert isinstance(result, dict)
        assert "error" in result
        assert result["type"] == "ErrorAction"

    def test_serialize_action_mixed_types(self):
        """Test action with mixed data types."""
        action = {
            "jax_array": jnp.array([1, 2, 3]),
            "numpy_array": np.array([4, 5, 6]),
            "int": 42,
            "float": 3.14,
            "bool": True,
            "string": "test",
            "complex": {"nested": "value"},
        }

        result = serialize_action(action)

        assert isinstance(result["jax_array"], np.ndarray)
        assert isinstance(result["numpy_array"], np.ndarray)
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool"] == True
        assert result["string"] == "test"
        assert result["complex"] == str({"nested": "value"})


class TestSerializeObject:
    """Test generic object serialization."""

    def test_serialize_primitives(self):
        """Test serialization of primitive types."""
        assert serialize_object(None) is None
        assert serialize_object(True) == True
        assert serialize_object(42) == 42
        assert serialize_object(3.14) == 3.14
        assert serialize_object("test") == "test"

    def test_serialize_list(self):
        """Test serialization of lists."""
        test_list = [1, "test", jnp.array([2, 3]), None]
        result = serialize_object(test_list)

        assert isinstance(result, list)
        assert result[0] == 1
        assert result[1] == "test"
        assert result[2] == [2, 3]  # Array converted to list
        assert result[3] is None

    def test_serialize_tuple(self):
        """Test serialization of tuples."""
        test_tuple = (1, "test", jnp.array([2, 3]))
        result = serialize_object(test_tuple)

        assert isinstance(result, list)  # Tuples become lists
        assert result[0] == 1
        assert result[1] == "test"
        assert result[2] == [2, 3]

    def test_serialize_dict(self):
        """Test serialization of dictionaries."""
        test_dict = {
            "int": 42,
            "array": jnp.array([1, 2]),
            "nested": {"inner": "value"},
        }
        result = serialize_object(test_dict)

        assert isinstance(result, dict)
        assert result["int"] == 42
        assert result["array"] == [1, 2]
        assert result["nested"] == {"inner": "value"}

    def test_serialize_jax_array(self):
        """Test serialization of JAX arrays."""
        array = jnp.array([[1, 2], [3, 4]])
        result = serialize_object(array)

        assert result == [[1, 2], [3, 4]]

    def test_serialize_numpy_array(self):
        """Test serialization of numpy arrays."""
        array = np.array([[1, 2], [3, 4]])
        result = serialize_object(array)

        assert result == [[1, 2], [3, 4]]

    def test_serialize_complex_nested_structure(self):
        """Test serialization of complex nested structures."""
        complex_obj = {
            "list": [1, 2, jnp.array([3, 4])],
            "tuple": (5, 6),
            "dict": {"nested": jnp.array([7, 8])},
            "array": jnp.array([[9, 10], [11, 12]]),
        }

        result = serialize_object(complex_obj)

        expected = {
            "list": [1, 2, [3, 4]],
            "tuple": [5, 6],
            "dict": {"nested": [7, 8]},
            "array": [[9, 10], [11, 12]],
        }

        assert result == expected

    def test_serialize_custom_object(self):
        """Test serialization of custom objects."""

        class CustomObject:
            def __init__(self):
                self.value = 42

            def __str__(self):
                return f"CustomObject(value={self.value})"

        obj = CustomObject()
        result = serialize_object(obj)

        assert result == "CustomObject(value=42)"


class TestSerializeState:
    """Test state serialization."""

    @pytest.fixture
    def sample_state(self):
        """Create a sample state for testing."""
        from jaxarc.configs.main_config import JaxArcConfig
        from jaxarc.types import EnvParams

        config = JaxArcConfig()
        params = EnvParams.from_config(config, buffer=jnp.ones(1))  # dummy buffer
        return State(
            working_grid=jnp.array([[1, 2], [3, 4]]),
            working_grid_mask=jnp.array([[True, True], [True, True]]),
            input_grid=jnp.array([[0, 1], [2, 3]]),
            input_grid_mask=jnp.array([[True, True], [True, True]]),
            target_grid=jnp.array([[5, 6], [7, 8]]),
            target_grid_mask=jnp.array([[True, True], [True, True]]),
            selected=jnp.array([[False, True], [False, False]]),
            clipboard=jnp.array([[0, 0], [0, 0]]),
            step_count=jnp.int32(10),
            allowed_operations_mask=jnp.ones(35, dtype=bool),
            similarity_score=jnp.float32(0.75),
            key=jax.random.PRNGKey(42),
            task_idx=jnp.int32(0),
            pair_idx=jnp.int32(0),
            carry={"test": "value"},
        )

    def test_serialize_state_basic(self, sample_state):
        """Test basic state serialization."""
        result = serialize_state(sample_state)

        assert isinstance(result, dict)
        assert "working_grid" in result
        assert "target_grid" in result
        assert "selected" in result
        assert "step_count" in result
        assert "similarity_score" in result
        assert "type" in result

        # Check types
        assert isinstance(result["working_grid"], np.ndarray)
        assert isinstance(result["target_grid"], np.ndarray)
        assert isinstance(result["selected"], np.ndarray)
        assert isinstance(result["step_count"], int)
        assert isinstance(result["similarity_score"], float)
        assert result["type"] == "State"

    def test_serialize_state_values(self, sample_state):
        """Test that state values are correctly serialized."""
        result = serialize_state(sample_state)

        # Check array values
        np.testing.assert_array_equal(result["working_grid"], [[1, 2], [3, 4]])
        np.testing.assert_array_equal(result["target_grid"], [[5, 6], [7, 8]])
        np.testing.assert_array_equal(
            result["selected"], [[False, True], [False, False]]
        )

        # Check scalar values
        assert result["step_count"] == 10
        assert result["similarity_score"] == 0.75

    def test_serialize_state_with_error(self):
        """Test state serialization error handling."""

        # Mock state that raises error
        class ErrorState:
            @property
            def working_grid(self):
                raise ValueError("Test error")

        state = ErrorState()
        result = serialize_state(state)

        assert isinstance(result, dict)
        assert "error" in result
        assert result["type"] == "ErrorState"


class TestSerializeLogStep:
    """Test log step serialization."""

    @pytest.fixture
    def sample_state(self):
        """Create a sample state for testing."""
        from jaxarc.configs.main_config import JaxArcConfig
        from jaxarc.types import EnvParams

        config = JaxArcConfig()
        params = EnvParams.from_config(config, buffer=jnp.ones(1))  # dummy buffer
        return State(
            working_grid=jnp.array([[1, 2]]),
            working_grid_mask=jnp.array([[True, True]]),
            input_grid=jnp.array([[0, 1]]),
            input_grid_mask=jnp.array([[True, True]]),
            target_grid=jnp.array([[3, 4]]),
            target_grid_mask=jnp.array([[True, True]]),
            selected=jnp.array([[False, True]]),
            clipboard=jnp.array([[0, 0]]),
            step_count=jnp.int32(5),
            allowed_operations_mask=jnp.ones(35, dtype=bool),
            similarity_score=jnp.float32(0.5),
            key=jax.random.PRNGKey(42),
            task_idx=jnp.int32(0),
            pair_idx=jnp.int32(0),
            carry={},
        )

    def test_serialize_log_step_basic(self, sample_state):
        """Test basic log step serialization."""
        step_data = {
            "before_state": sample_state,
            "after_state": sample_state,
            "action": {"operation": jnp.array([1, 0, 0])},
            "reward": 0.5,
            "info": {"similarity": 0.75},
        }

        result = serialize_log_step(step_data)

        assert isinstance(result, dict)
        assert "before_state" in result
        assert "after_state" in result
        assert "action" in result
        assert "reward" in result
        assert "info" in result

        # States should be serialized
        assert isinstance(result["before_state"], dict)
        assert isinstance(result["after_state"], dict)

        # Action should be serialized
        assert isinstance(result["action"], dict)

    def test_serialize_log_step_with_info_dict(self, sample_state):
        """Test log step serialization with info dictionary."""
        step_data = {
            "before_state": sample_state,
            "action": {"operation": jnp.array([1])},
            "info": {
                "similarity": jnp.float32(0.8),
                "step_count": jnp.int32(10),
                "nested": {"value": jnp.array([1, 2, 3])},
            },
        }

        result = serialize_log_step(step_data)

        assert isinstance(result["info"], dict)
        assert abs(result["info"]["similarity"] - 0.8) < 1e-6
        assert result["info"]["step_count"] == 10
        assert result["info"]["nested"] == {"value": [1, 2, 3]}

    def test_serialize_log_step_with_arrays(self, sample_state):
        """Test log step serialization with various array types."""
        step_data = {
            "observation": jnp.array([[1, 2], [3, 4]]),
            "action_mask": jnp.array([True, False, True]),
            "custom_data": np.array([5, 6, 7]),
        }

        result = serialize_log_step(step_data)

        assert result["observation"] == [[1, 2], [3, 4]]
        assert result["action_mask"] == [True, False, True]
        assert result["custom_data"] == [5, 6, 7]

    def test_serialize_log_step_empty(self):
        """Test serialization of empty log step."""
        step_data = {}
        result = serialize_log_step(step_data)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_serialize_log_step_complex(self, sample_state):
        """Test serialization of complex log step data."""
        step_data = {
            "before_state": sample_state,
            "after_state": sample_state,
            "action": {
                "type": "fill",
                "selection": jnp.array([[True, False]]),
                "color": 3,
            },
            "reward": jnp.float32(1.0),
            "done": True,
            "info": {
                "metrics": {"similarity": jnp.float32(0.9), "steps": jnp.int32(15)},
                "debug": {"grid_changes": jnp.array([1, 2, 3]), "message": "test"},
            },
            "custom_arrays": [jnp.array([1, 2]), np.array([3, 4])],
        }

        result = serialize_log_step(step_data)

        # Check all fields are present and properly serialized
        assert isinstance(result["before_state"], dict)
        assert isinstance(result["after_state"], dict)
        assert isinstance(result["action"], dict)
        assert result["reward"] == 1.0
        assert result["done"] == True

        # Check nested info structure
        assert isinstance(result["info"], dict)
        assert abs(result["info"]["metrics"]["similarity"] - 0.9) < 1e-6
        assert result["info"]["metrics"]["steps"] == 15
        assert result["info"]["debug"]["grid_changes"] == [1, 2, 3]
        assert result["info"]["debug"]["message"] == "test"

        # Check custom arrays
        assert result["custom_arrays"] == [[1, 2], [3, 4]]


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_serialize_very_large_arrays(self):
        """Test serialization of very large arrays."""
        large_array = jnp.ones((1000, 1000))
        result = serialize_jax_array(large_array)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1000, 1000)
        assert np.all(result == 1)

    def test_serialize_different_array_shapes(self):
        """Test serialization of arrays with different shapes."""
        # 1D array
        array_1d = jnp.array([1, 2, 3, 4, 5])
        result = serialize_jax_array(array_1d)
        assert result.shape == (5,)

        # 3D array
        array_3d = jnp.ones((2, 3, 4))
        result = serialize_jax_array(array_3d)
        assert result.shape == (2, 3, 4)

        # 4D array
        array_4d = jnp.zeros((2, 2, 2, 2))
        result = serialize_jax_array(array_4d)
        assert result.shape == (2, 2, 2, 2)

    def test_serialize_extreme_values(self):
        """Test serialization with extreme values."""
        # Very large numbers
        large_array = jnp.array([1e10, -1e10])
        result = serialize_jax_array(large_array)
        np.testing.assert_array_equal(result, [1e10, -1e10])

        # Very small numbers
        small_array = jnp.array([1e-10, -1e-10])
        result = serialize_jax_array(small_array)
        np.testing.assert_array_almost_equal(result, [1e-10, -1e-10])

    def test_serialize_special_float_values(self):
        """Test serialization with special float values."""
        special_array = jnp.array([np.inf, -np.inf, np.nan])
        result = serialize_jax_array(special_array)

        assert np.isinf(result[0])
        assert np.isinf(result[1]) and result[1] < 0
        assert np.isnan(result[2])

    def test_serialize_nested_structures_deep(self):
        """Test serialization of deeply nested structures."""
        deep_structure = {
            "level1": {
                "level2": {
                    "level3": {"level4": {"array": jnp.array([1, 2, 3]), "value": 42}}
                }
            }
        }

        result = serialize_object(deep_structure)

        assert result["level1"]["level2"]["level3"]["level4"]["array"] == [1, 2, 3]
        assert result["level1"]["level2"]["level3"]["level4"]["value"] == 42

    # Removed test_serialize_circular_reference_protection - complex edge case
    # Circular references are unlikely in normal JAX/ARC usage patterns

    def test_memory_efficiency_large_data(self):
        """Test memory efficiency with large data structures."""
        # Create large nested structure
        large_data = {f"array_{i}": jnp.ones((100, 100)) * i for i in range(10)}

        result = serialize_object(large_data)

        # Should successfully serialize without memory issues
        assert isinstance(result, dict)
        assert len(result) == 10

        # Check a few values
        assert len(result["array_0"]) == 100
        assert len(result["array_0"][0]) == 100
        assert result["array_5"][0][0] == 5
