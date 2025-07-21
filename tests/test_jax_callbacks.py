"""Tests for JAX callback system."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxarc.state import ArcEnvState
from jaxarc.types import Grid
from jaxarc.utils.visualization.jax_callbacks import (
    CallbackPerformanceMonitor,
    adaptive_visualization_callback,
    create_grid_from_arrays,
    get_callback_performance_stats,
    jax_debug_callback,
    jax_log_episode_summary,
    jax_log_grid,
    log_episode_summary_callback,
    log_grid_callback,
    print_callback_performance_report,
    reset_callback_performance_stats,
    safe_callback_wrapper,
    save_step_visualization_callback,
    serialize_action,
    serialize_arc_state,
    serialize_jax_array,
)


class TestArraySerialization:
    """Test array serialization functions."""

    def test_serialize_jax_array(self):
        """Test JAX array serialization."""
        # Test JAX array
        jax_arr = jnp.array([[1, 2], [3, 4]])
        result = serialize_jax_array(jax_arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [[1, 2], [3, 4]])

        # Test numpy array
        np_arr = np.array([[5, 6], [7, 8]])
        result = serialize_jax_array(np_arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [[5, 6], [7, 8]])

    def test_serialize_arc_state(self):
        """Test ArcEnvState serialization."""
        # Create mock state
        state = Mock(spec=ArcEnvState)
        state.working_grid = jnp.array([[1, 2], [3, 4]])
        state.working_grid_mask = jnp.array([[True, True], [False, True]])
        state.target_grid = jnp.array([[4, 3], [2, 1]])
        state.target_grid_mask = jnp.array([[True, False], [True, True]])
        state.step_count = jnp.array(5)
        state.episode_index = jnp.array(2)
        state.task_index = jnp.array(10)
        state.done = jnp.array(False)
        state.similarity = jnp.array(0.75)

        result = serialize_arc_state(state)

        assert isinstance(result, dict)
        assert "working_grid" in result
        assert "working_grid_mask" in result
        assert "target_grid" in result
        assert "target_grid_mask" in result
        assert result["step_count"] == 5
        assert result["episode_index"] == 2
        assert result["task_index"] == 10
        assert result["done"] is False
        assert result["similarity"] == 0.75

    def test_serialize_action(self):
        """Test action dictionary serialization."""
        action = {
            "selection": jnp.array([[True, False], [False, True]]),
            "operation": jnp.array(5),
            "metadata": "test_string",
            "score": 0.8,
        }

        result = serialize_action(action)

        assert isinstance(result, dict)
        assert "selection" in result
        assert "operation" in result
        assert "metadata" in result
        assert "score" in result
        assert isinstance(result["selection"], np.ndarray)
        assert result["operation"] == 5
        assert result["metadata"] == "test_string"
        assert result["score"] == 0.8


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""

    def test_callback_performance_monitor(self):
        """Test CallbackPerformanceMonitor."""
        monitor = CallbackPerformanceMonitor()

        # Record some calls
        monitor.record_call("test_callback", 0.01, False)
        monitor.record_call("test_callback", 0.02, False)
        monitor.record_call("test_callback", 0.015, True)

        stats = monitor.get_stats("test_callback")
        assert stats["total_calls"] == 3
        assert stats["error_count"] == 1
        assert stats["avg_time_ms"] == pytest.approx(15.0, rel=1e-2)
        assert stats["max_time_ms"] == pytest.approx(20.0, rel=1e-2)
        assert stats["min_time_ms"] == pytest.approx(10.0, rel=1e-2)

        # Test should_reduce_logging
        assert not monitor.should_reduce_logging("test_callback", 20.0)
        assert monitor.should_reduce_logging("test_callback", 10.0)

    def test_safe_callback_wrapper(self):
        """Test safe callback wrapper."""
        # Test successful callback
        mock_func = Mock()
        wrapped = safe_callback_wrapper(mock_func, "test_callback")

        wrapped("arg1", "arg2", kwarg1="value1")
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")

        # Test callback with exception
        error_func = Mock(side_effect=Exception("Test error"))
        wrapped_error = safe_callback_wrapper(error_func, "error_callback")

        # Should not raise exception
        wrapped_error("test_arg")
        error_func.assert_called_once_with("test_arg")


class TestJAXCallbacks:
    """Test JAX callback functions."""

    def test_create_grid_from_arrays(self):
        """Test Grid creation from JAX arrays."""
        data = jnp.array([[1, 2], [3, 4]])
        mask = jnp.array([[True, False], [True, True]])

        grid = create_grid_from_arrays(data, mask)

        assert isinstance(grid, Grid)
        assert isinstance(grid.data, np.ndarray)
        assert isinstance(grid.mask, np.ndarray)
        np.testing.assert_array_equal(grid.data, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(grid.mask, [[True, False], [True, True]])

    @patch("jaxarc.utils.visualization.jax_callbacks.log_grid_to_console")
    def test_log_grid_callback(self, mock_log_grid):
        """Test log grid callback."""
        grid_data = jnp.array([[1, 2], [3, 4]])
        mask = jnp.array([[True, True], [False, True]])

        log_grid_callback(grid_data, mask, "Test Grid", True, False)

        mock_log_grid.assert_called_once()
        args, kwargs = mock_log_grid.call_args
        assert isinstance(args[0], Grid)
        assert kwargs["title"] == "Test Grid"
        assert kwargs["show_coordinates"] is True
        assert kwargs["show_numbers"] is False

    def test_save_step_visualization_callback(self):
        """Test save step visualization callback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock states
            before_state = Mock(spec=ArcEnvState)
            before_state.working_grid = jnp.array([[1, 2], [3, 4]])
            before_state.working_grid_mask = jnp.array([[True, True], [True, True]])
            before_state.step_count = jnp.array(1)

            after_state = Mock(spec=ArcEnvState)
            after_state.working_grid = jnp.array([[2, 1], [4, 3]])
            after_state.working_grid_mask = jnp.array([[True, True], [True, True]])

            action = {
                "selection": jnp.array([[True, False], [False, True]]),
                "operation": jnp.array(5),
            }

            # Mock the draw_rl_step_svg function
            with patch(
                "jaxarc.utils.visualization.jax_callbacks.draw_rl_step_svg"
            ) as mock_draw:
                mock_draw.return_value = "<svg>test</svg>"

                save_step_visualization_callback(
                    before_state, action, after_state, temp_dir, "Test Step"
                )

                # Check that file was created
                expected_file = Path(temp_dir) / "step_001.svg"
                assert expected_file.exists()

                # Check file content
                with open(expected_file) as f:
                    content = f.read()
                assert content == "<svg>test</svg>"

    def test_log_episode_summary_callback(self):
        """Test log episode summary callback."""
        with patch("jaxarc.utils.visualization.jax_callbacks.logger") as mock_logger:
            log_episode_summary_callback(5, 100, 15.5, 0.85, True)

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "Episode 5 SUCCESS" in call_args
            assert "steps=100" in call_args
            assert "reward=15.500" in call_args
            assert "similarity=0.850" in call_args

    @patch("jaxarc.utils.visualization.jax_callbacks.jax.debug.callback")
    def test_jax_debug_callback(self, mock_jax_callback):
        """Test JAX debug callback wrapper."""
        mock_func = Mock()

        jax_debug_callback(mock_func, "arg1", "arg2", callback_name="test_callback")

        mock_jax_callback.assert_called_once()
        # The first argument should be the wrapped function
        wrapped_func = mock_jax_callback.call_args[0][0]
        assert callable(wrapped_func)

    @patch("jaxarc.utils.visualization.jax_callbacks.jax_debug_callback")
    def test_convenience_functions(self, mock_jax_debug):
        """Test convenience functions."""
        # Test jax_log_grid
        grid_data = jnp.array([[1, 2], [3, 4]])
        mask = jnp.array([[True, True], [False, True]])

        jax_log_grid(grid_data, mask, "Test Grid")
        mock_jax_debug.assert_called()

        # Test jax_log_episode_summary
        mock_jax_debug.reset_mock()
        jax_log_episode_summary(1, 50, 10.0, 0.9, True)
        mock_jax_debug.assert_called()

    def test_adaptive_visualization_callback(self):
        """Test adaptive visualization callback."""
        mock_func = Mock()

        with patch(
            "jaxarc.utils.visualization.jax_callbacks._performance_monitor"
        ) as mock_monitor:
            # Test normal case (should call)
            mock_monitor.should_reduce_logging.return_value = False
            mock_monitor.get_stats.return_value = {"total_calls": 5}

            with patch(
                "jaxarc.utils.visualization.jax_callbacks.jax_debug_callback"
            ) as mock_jax_debug:
                adaptive_visualization_callback(mock_func, "arg1", "arg2")
                mock_jax_debug.assert_called_once()

            # Test reduced logging case (should skip most calls)
            mock_jax_debug.reset_mock()
            mock_monitor.should_reduce_logging.return_value = True
            mock_monitor.get_stats.return_value = {
                "total_calls": 15
            }  # Not divisible by 10

            adaptive_visualization_callback(mock_func, "arg1", "arg2")
            mock_jax_debug.assert_not_called()

            # Test reduced logging case (should call every 10th)
            mock_monitor.get_stats.return_value = {"total_calls": 20}  # Divisible by 10

            adaptive_visualization_callback(mock_func, "arg1", "arg2")
            mock_jax_debug.assert_called_once()


class TestPerformanceStatsAPI:
    """Test performance statistics API."""

    def test_get_callback_performance_stats(self):
        """Test getting performance stats."""
        # Reset stats first
        reset_callback_performance_stats()

        # Record some test data
        from jaxarc.utils.visualization.jax_callbacks import _performance_monitor

        _performance_monitor.record_call("test_callback", 0.01, False)
        _performance_monitor.record_call("another_callback", 0.02, True)

        stats = get_callback_performance_stats()

        assert isinstance(stats, dict)
        assert "test_callback" in stats
        assert "another_callback" in stats
        assert stats["test_callback"]["total_calls"] == 1
        assert stats["another_callback"]["error_count"] == 1

    def test_reset_callback_performance_stats(self):
        """Test resetting performance stats."""
        # Add some data
        from jaxarc.utils.visualization.jax_callbacks import _performance_monitor

        _performance_monitor.record_call("test_callback", 0.01, False)

        # Verify data exists
        stats_before = get_callback_performance_stats()
        assert len(stats_before) > 0

        # Reset and verify empty
        reset_callback_performance_stats()
        stats_after = get_callback_performance_stats()
        assert len(stats_after) == 0

    def test_print_callback_performance_report(self):
        """Test printing performance report."""
        reset_callback_performance_stats()

        with patch("jaxarc.utils.visualization.jax_callbacks.logger") as mock_logger:
            # Test with no data
            print_callback_performance_report()
            mock_logger.info.assert_called_with(
                "No callback performance data available"
            )

            # Add some data and test report
            from jaxarc.utils.visualization.jax_callbacks import _performance_monitor

            _performance_monitor.record_call("test_callback", 0.01, False)

            mock_logger.reset_mock()
            print_callback_performance_report()

            # Should have multiple info calls for the report
            assert mock_logger.info.call_count > 1
            call_args = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Callback Performance Report" in arg for arg in call_args)
            assert any("test_callback" in arg for arg in call_args)


class TestJAXIntegration:
    """Test actual JAX integration."""

    def test_jax_compilation_compatibility(self):
        """Test that callbacks work with JAX compilation."""

        @jax.jit
        def test_function(x):
            # This should work without breaking JIT compilation
            jax.debug.callback(lambda val: None, x)
            return x * 2

        result = test_function(jnp.array(5.0))
        assert result == 10.0

    def test_callback_in_jax_transformation(self):
        """Test callbacks work within JAX transformations."""
        call_count = 0

        def counting_callback(x):
            nonlocal call_count
            call_count += 1

        @jax.jit
        def test_function(x):
            jax.debug.callback(counting_callback, x)
            return x + 1

        # Test with vmap
        inputs = jnp.array([1.0, 2.0, 3.0])
        results = jax.vmap(test_function)(inputs)

        np.testing.assert_array_equal(results, [2.0, 3.0, 4.0])
        assert call_count == 3  # Should be called once per input

    def test_error_handling_in_jax_context(self):
        """Test that callback errors don't break JAX execution."""

        def error_callback(x):
            raise ValueError("Test error")

        @jax.jit
        def test_function(x):
            # Use safe wrapper to prevent JAX from breaking
            wrapped_callback = safe_callback_wrapper(error_callback, "error_test")
            jax.debug.callback(wrapped_callback, x)
            return x * 2

        # This should not raise an exception
        result = test_function(jnp.array(5.0))
        assert result == 10.0
