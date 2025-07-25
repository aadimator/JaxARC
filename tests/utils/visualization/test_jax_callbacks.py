"""Tests for JAX debug callback integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxarc.state import ArcEnvState
from jaxarc.utils.visualization.jax_callbacks import (
    CallbackPerformanceMonitor,
    JAXCallbackError,
    get_callback_performance_stats,
    jax_debug_callback,
    jax_log_episode_summary,
    jax_log_grid,
    jax_save_step_visualization,
    print_callback_performance_report,
    reset_callback_performance_stats,
    safe_callback_wrapper,
    serialize_action,
    serialize_arc_state,
    serialize_jax_array,
)


class TestArraySerialization:
    """Test JAX array serialization functions."""

    def test_serialize_jax_array_with_jax_array(self):
        """Test serializing JAX arrays."""
        jax_array = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)

        result = serialize_jax_array(jax_array)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, jax_array)

    def test_serialize_jax_array_with_numpy_array(self):
        """Test serializing numpy arrays."""
        numpy_array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

        result = serialize_jax_array(numpy_array)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, numpy_array)
        # Should be a copy, not the same object
        assert result is not numpy_array

    def test_serialize_jax_array_with_scalar(self):
        """Test serializing scalar values."""
        scalar = jnp.array(42)

        result = serialize_jax_array(scalar)

        assert isinstance(result, np.ndarray)
        assert result.item() == 42

    def test_serialize_jax_array_with_invalid_input(self):
        """Test serializing invalid input."""
        # Should convert to numpy array (even strings)
        result = serialize_jax_array("invalid")

        assert isinstance(result, np.ndarray)
        # String gets converted to numpy array
        assert result.dtype.kind == "U"  # Unicode string

    def test_serialize_jax_array_error_handling(self):
        """Test error handling in array serialization."""

        # Create an object that will cause serialization to fail
        class BadArray:
            def __array__(self):
                raise Exception("Test error")

        result = serialize_jax_array(BadArray())

        # Should return empty array on error
        assert isinstance(result, np.ndarray)
        assert result.size == 0


class TestStateAndActionSerialization:
    """Test state and action serialization functions."""

    def test_serialize_arc_state(self):
        """Test ArcEnvState serialization."""
        # Create mock state with all required fields
        mock_state = MagicMock(spec=ArcEnvState)
        
        # Core grid data
        mock_state.working_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        mock_state.working_grid_mask = jnp.array(
            [[True, True], [True, True]], dtype=jnp.bool_
        )
        mock_state.target_grid = jnp.array([[1, 1], [2, 2]], dtype=jnp.int32)
        mock_state.target_grid_mask = jnp.array(
            [[True, True], [True, True]], dtype=jnp.bool_
        )
        
        # Episode management
        mock_state.step_count = 5
        mock_state.current_example_idx = 1
        mock_state.episode_done = False
        
        # Grid operations
        mock_state.selected = jnp.array([[False, True], [True, False]], dtype=jnp.bool_)
        mock_state.clipboard = jnp.array([[0, 0], [0, 0]], dtype=jnp.int32)
        mock_state.similarity_score = 0.8
        
        # Enhanced functionality fields
        mock_state.episode_mode = 0
        mock_state.available_demo_pairs = jnp.array([True, True, False], dtype=jnp.bool_)
        mock_state.available_test_pairs = jnp.array([True, False], dtype=jnp.bool_)
        mock_state.demo_completion_status = jnp.array([True, False, False], dtype=jnp.bool_)
        mock_state.test_completion_status = jnp.array([False, False], dtype=jnp.bool_)
        mock_state.action_history = jnp.zeros((10, 20), dtype=jnp.float32)
        mock_state.action_history_length = 3
        mock_state.allowed_operations_mask = jnp.ones(42, dtype=jnp.bool_)

        result = serialize_arc_state(mock_state)

        assert isinstance(result, dict)
        
        # Check core fields
        assert "working_grid" in result
        assert "working_grid_mask" in result
        assert "target_grid" in result
        assert "target_grid_mask" in result
        assert "step_count" in result
        assert "current_example_idx" in result
        assert "episode_done" in result
        assert "similarity_score" in result
        
        # Check enhanced functionality fields
        assert "selected" in result
        assert "clipboard" in result
        assert "episode_mode" in result
        assert "available_demo_pairs" in result
        assert "available_test_pairs" in result
        assert "demo_completion_status" in result
        assert "test_completion_status" in result
        assert "action_history" in result
        assert "action_history_length" in result
        assert "allowed_operations_mask" in result
        
        # Check that backward compatibility aliases are NOT present (removed)
        assert "episode_index" not in result  # Old alias removed
        assert "done" not in result  # Old alias removed  
        assert "similarity" not in result  # Old alias removed

        # Check values
        assert result["step_count"] == 5
        assert result["current_example_idx"] == 1
        assert result["episode_done"] is False
        assert result["similarity_score"] == 0.8
        assert result["episode_mode"] == 0
        assert result["action_history_length"] == 3
        
        # Check that the correct field names are used
        assert result["current_example_idx"] == 1
        assert result["episode_done"] is False
        assert result["similarity_score"] == 0.8

        # Check that arrays were serialized
        assert isinstance(result["working_grid"], np.ndarray)
        assert isinstance(result["working_grid_mask"], np.ndarray)

    def test_serialize_arc_state_error_handling(self):
        """Test error handling in state serialization."""
        # Create mock state that will cause an exception during serialization
        mock_state = MagicMock()

        # Mock the serialize_jax_array to raise an exception
        with patch(
            "jaxarc.utils.visualization.jax_callbacks.serialize_jax_array",
            side_effect=Exception("Test error"),
        ):
            result = serialize_arc_state(mock_state)

            # Should return empty dict on error
            assert isinstance(result, dict)
            assert len(result) == 0

    def test_serialize_action_basic(self):
        """Test basic action serialization."""
        action = {
            "operation": 1,
            "selection": jnp.array([[True, False], [False, False]], dtype=jnp.bool_),
            "fill_color": 2,
            "metadata": "test_action",
        }

        result = serialize_action(action)

        assert isinstance(result, dict)
        assert result["operation"] == 1
        assert result["fill_color"] == 2
        assert result["metadata"] == "test_action"
        assert isinstance(result["selection"], np.ndarray)
        np.testing.assert_array_equal(result["selection"], action["selection"])

    def test_serialize_action_with_various_types(self):
        """Test action serialization with various data types."""
        action = {
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "str_val": "test",
            "array_val": jnp.array([1, 2, 3]),
            "numpy_array": np.array([4, 5, 6]),
            "other_val": {"nested": "dict"},
        }

        result = serialize_action(action)

        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
        assert result["bool_val"] is True
        assert result["str_val"] == "test"
        assert isinstance(result["array_val"], np.ndarray)
        assert isinstance(result["numpy_array"], np.ndarray)
        assert result["other_val"] == "{'nested': 'dict'}"  # Converted to string

    def test_serialize_action_error_handling(self):
        """Test error handling in action serialization."""
        # Mock the serialize_jax_array to raise an exception
        with patch(
            "jaxarc.utils.visualization.jax_callbacks.serialize_jax_array",
            side_effect=Exception("Test error"),
        ):
            action = {"array_key": jnp.array([1, 2, 3])}

            result = serialize_action(action)

            # Should return empty dict on error
            assert isinstance(result, dict)
            assert len(result) == 0


class TestCallbackPerformanceMonitor:
    """Test callback performance monitoring."""

    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = CallbackPerformanceMonitor()

        assert isinstance(monitor.callback_times, dict)
        assert isinstance(monitor.error_counts, dict)
        assert isinstance(monitor.total_calls, dict)
        assert len(monitor.callback_times) == 0

    def test_record_call_success(self):
        """Test recording successful callback calls."""
        monitor = CallbackPerformanceMonitor()

        monitor.record_call("test_callback", 0.01, False)
        monitor.record_call("test_callback", 0.02, False)

        assert "test_callback" in monitor.callback_times
        assert len(monitor.callback_times["test_callback"]) == 2
        assert monitor.total_calls["test_callback"] == 2
        assert monitor.error_counts["test_callback"] == 0

    def test_record_call_with_error(self):
        """Test recording callback calls with errors."""
        monitor = CallbackPerformanceMonitor()

        monitor.record_call("error_callback", 0.05, True)
        monitor.record_call("error_callback", 0.03, False)
        monitor.record_call("error_callback", 0.04, True)

        assert monitor.total_calls["error_callback"] == 3
        assert monitor.error_counts["error_callback"] == 2

    def test_get_stats(self):
        """Test getting performance statistics."""
        monitor = CallbackPerformanceMonitor()

        # Record some calls
        monitor.record_call("test_callback", 0.01, False)
        monitor.record_call("test_callback", 0.02, False)
        monitor.record_call("test_callback", 0.03, True)

        stats = monitor.get_stats("test_callback")

        assert stats["total_calls"] == 3
        assert stats["error_count"] == 1
        assert stats["avg_time_ms"] == 20.0  # (0.01 + 0.02 + 0.03) / 3 * 1000
        assert stats["max_time_ms"] == 30.0  # 0.03 * 1000
        assert stats["min_time_ms"] == 10.0  # 0.01 * 1000
        assert stats["total_time_ms"] == 60.0  # (0.01 + 0.02 + 0.03) * 1000

    def test_get_stats_nonexistent_callback(self):
        """Test getting stats for non-existent callback."""
        monitor = CallbackPerformanceMonitor()

        stats = monitor.get_stats("nonexistent")

        assert stats == {}

    def test_should_reduce_logging(self):
        """Test logging reduction decision."""
        monitor = CallbackPerformanceMonitor()

        # Fast callback - should not reduce logging
        monitor.record_call("fast_callback", 0.001, False)  # 1ms
        assert monitor.should_reduce_logging("fast_callback", 10.0) == False

        # Slow callback - should reduce logging
        monitor.record_call("slow_callback", 0.02, False)  # 20ms
        assert monitor.should_reduce_logging("slow_callback", 10.0) == True

        # Non-existent callback
        assert monitor.should_reduce_logging("nonexistent", 10.0) == False


class TestSafeCallbackWrapper:
    """Test safe callback wrapper functionality."""

    def test_safe_callback_wrapper_success(self):
        """Test wrapper with successful callback."""
        mock_callback = MagicMock()
        wrapped = safe_callback_wrapper(mock_callback, "test_callback")

        wrapped("arg1", "arg2", kwarg1="value1")

        mock_callback.assert_called_once_with("arg1", "arg2", kwarg1="value1")

    def test_safe_callback_wrapper_with_exception(self):
        """Test wrapper with callback that raises exception."""
        error_callback = MagicMock(side_effect=Exception("Test error"))
        wrapped = safe_callback_wrapper(error_callback, "error_callback")

        # Should not raise exception
        wrapped("test_arg")

        error_callback.assert_called_once_with("test_arg")

    def test_safe_callback_wrapper_performance_monitoring(self):
        """Test wrapper with performance monitoring."""
        mock_callback = MagicMock()

        # Mock the global performance monitor
        with patch(
            "jaxarc.utils.visualization.jax_callbacks._performance_monitor"
        ) as mock_monitor:
            wrapped = safe_callback_wrapper(mock_callback, "test_callback", True)
            wrapped("test")

            # Should record the call
            mock_monitor.record_call.assert_called_once()
            call_args = mock_monitor.record_call.call_args[0]
            assert call_args[0] == "test_callback"
            assert isinstance(call_args[1], float)  # Duration
            assert call_args[2] is False  # No error

    def test_safe_callback_wrapper_no_performance_monitoring(self):
        """Test wrapper without performance monitoring."""
        mock_callback = MagicMock()

        with patch(
            "jaxarc.utils.visualization.jax_callbacks._performance_monitor"
        ) as mock_monitor:
            wrapped = safe_callback_wrapper(mock_callback, "test_callback", False)
            wrapped("test")

            # Should not record the call
            mock_monitor.record_call.assert_not_called()


class TestJAXDebugCallback:
    """Test JAX debug callback functionality."""

    def test_jax_debug_callback_basic(self):
        """Test basic JAX debug callback."""
        mock_callback = MagicMock()

        # Mock jax.debug.callback
        with patch("jax.debug.callback") as mock_jax_callback:
            jax_debug_callback(mock_callback, "arg1", "arg2", callback_name="test")

            # Should call jax.debug.callback with wrapped function
            mock_jax_callback.assert_called_once()

    def test_jax_debug_callback_with_performance_monitoring(self):
        """Test JAX debug callback with performance monitoring."""
        mock_callback = MagicMock()

        with patch("jax.debug.callback") as mock_jax_callback:
            jax_debug_callback(
                mock_callback,
                "arg1",
                callback_name="test",
                enable_performance_monitoring=True,
            )

            mock_jax_callback.assert_called_once()

    def test_jax_debug_callback_without_performance_monitoring(self):
        """Test JAX debug callback without performance monitoring."""
        mock_callback = MagicMock()

        with patch("jax.debug.callback") as mock_jax_callback:
            jax_debug_callback(
                mock_callback,
                "arg1",
                callback_name="test",
                enable_performance_monitoring=False,
            )

            mock_jax_callback.assert_called_once()


class TestSpecificJAXCallbacks:
    """Test specific JAX callback functions."""

    def test_jax_log_grid(self):
        """Test JAX grid logging callback."""
        test_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        with patch(
            "jaxarc.utils.visualization.jax_callbacks.jax_debug_callback"
        ) as mock_debug:
            jax_log_grid(test_grid, title="Test Grid")

            # Should call jax_debug_callback with log_grid_to_console
            mock_debug.assert_called_once()
            call_args = mock_debug.call_args[0]
            # First argument should be the log function
            assert callable(call_args[0])

    def test_jax_log_grid_with_mask(self):
        """Test JAX grid logging with mask."""
        test_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        test_mask = jnp.array([[True, True], [False, True]], dtype=jnp.bool_)

        with patch(
            "jaxarc.utils.visualization.jax_callbacks.jax_debug_callback"
        ) as mock_debug:
            jax_log_grid(test_grid, mask=test_mask, title="Masked Grid")

            mock_debug.assert_called_once()

    def test_jax_save_step_visualization(self):
        """Test JAX step visualization saving."""
        # Create mock states
        mock_before_state = MagicMock()
        mock_after_state = MagicMock()
        action = {
            "operation": 1,
            "selection": jnp.array([[True, False], [False, False]]),
        }
        reward = 0.5
        info = {"step_count": 1, "success": False}

        with patch(
            "jaxarc.utils.visualization.jax_callbacks.adaptive_visualization_callback"
        ) as mock_callback:
            jax_save_step_visualization(
                before_state=mock_before_state,
                action=action,
                after_state=mock_after_state,
                reward=reward,
                info=info,
                output_dir="/test/output",
            )

            mock_callback.assert_called_once()

    def test_jax_log_episode_summary(self):
        """Test JAX episode summary logging."""
        with patch(
            "jaxarc.utils.visualization.jax_callbacks.jax_debug_callback"
        ) as mock_debug:
            jax_log_episode_summary(
                episode_num=1,
                total_steps=10,
                total_reward=5.0,
                final_similarity=0.8,
                success=True,
            )

            mock_debug.assert_called_once()


class TestGlobalPerformanceStats:
    """Test global performance statistics functions."""

    def test_get_callback_performance_stats(self):
        """Test getting global performance stats."""
        # Reset stats first
        reset_callback_performance_stats()

        # Mock the global monitor
        with patch(
            "jaxarc.utils.visualization.jax_callbacks._performance_monitor"
        ) as mock_monitor:
            mock_monitor.callback_times = {"test_callback": [0.01, 0.02]}
            mock_monitor.get_stats.return_value = {"total_calls": 5}

            stats = get_callback_performance_stats()

            assert isinstance(stats, dict)

    def test_reset_callback_performance_stats(self):
        """Test resetting global performance stats."""
        # This should not raise an exception
        reset_callback_performance_stats()

    def test_print_callback_performance_report(self):
        """Test printing performance report."""
        with patch(
            "jaxarc.utils.visualization.jax_callbacks._performance_monitor"
        ) as mock_monitor:
            mock_monitor.callback_times = {"test_callback": [0.01, 0.02]}
            mock_monitor.get_stats.return_value = {
                "total_calls": 2,
                "error_count": 0,
                "avg_time_ms": 15.0,
                "max_time_ms": 20.0,
                "min_time_ms": 10.0,
                "total_time_ms": 30.0,
            }

            # Should not raise an exception
            print_callback_performance_report()


class TestJAXCallbackError:
    """Test JAXCallbackError exception."""

    def test_jax_callback_error_creation(self):
        """Test creating JAXCallbackError."""
        error = JAXCallbackError("Test error message")

        assert isinstance(error, Exception)
        assert str(error) == "Test error message"

    def test_jax_callback_error_inheritance(self):
        """Test JAXCallbackError inheritance."""
        error = JAXCallbackError("Test")

        assert isinstance(error, Exception)
        assert isinstance(error, JAXCallbackError)


class TestIntegrationWithJAXTransformations:
    """Test integration with JAX transformations."""

    def test_callback_with_jit(self):
        """Test callback functionality with JIT compilation."""

        @jax.jit
        def test_function(x):
            # Use a callback inside JIT
            jax_log_grid(x, title="JIT Test")
            return x + 1

        test_input = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        # Should not raise an exception
        result = test_function(test_input)
        np.testing.assert_array_equal(result, test_input + 1)

    def test_callback_with_vmap(self):
        """Test callback functionality with vmap."""

        def test_function(x):
            jax_log_grid(x, title="vmap Test")
            return x.sum()

        # Create batch of inputs
        batch_input = jnp.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]], dtype=jnp.int32)

        vmapped_fn = jax.vmap(test_function)

        # Should not raise an exception
        result = vmapped_fn(batch_input)
        expected = jnp.array([6, 10])  # Sum of each grid
        np.testing.assert_array_equal(result, expected)

    def test_serialization_preserves_data_integrity(self):
        """Test that serialization preserves data integrity through transformations."""
        original_array = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)

        # Serialize and check integrity
        serialized = serialize_jax_array(original_array)

        assert isinstance(serialized, np.ndarray)
        np.testing.assert_array_equal(serialized, original_array)

        # Test with transformed array
        transformed = jax.jit(lambda x: x * 2)(original_array)
        serialized_transformed = serialize_jax_array(transformed)

        np.testing.assert_array_equal(serialized_transformed, original_array * 2)


if __name__ == "__main__":
    pytest.main([__file__])
