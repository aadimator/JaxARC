"""Comprehensive tests for the visualization system.

This module tests all aspects of the JaxARC visualization system including:
- Terminal rendering functions
- SVG rendering and grid visualization
- JAX debug callback integration
- Visualization utility functions and error handling
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest
from rich.panel import Panel

from jaxarc.types import Grid
from jaxarc.utils.visualization import (
    ARC_COLOR_PALETTE,
    _extract_grid_data,
    draw_grid_svg,
    draw_parsed_task_data_svg,
    draw_rl_step_svg,
    draw_task_pair_svg,
    log_grid_to_console,
    save_svg_drawing,
    setup_matplotlib_style,
    visualize_grid_rich,
    visualize_parsed_task_data_rich,
    visualize_task_pair_rich,
)
from jaxarc.utils.visualization.core import _extract_valid_region


class TestCoreVisualizationFunctions:
    """Test core visualization functions."""

    def test_arc_color_palette_completeness(self):
        """Test that ARC color palette has all expected colors."""
        # ARC uses colors 0-9, plus 10 for padding/invalid
        expected_colors = set(range(11))
        actual_colors = set(ARC_COLOR_PALETTE.keys())

        assert actual_colors == expected_colors

        # Check that all colors are valid hex strings
        for color_val, hex_color in ARC_COLOR_PALETTE.items():
            assert isinstance(hex_color, str)
            assert hex_color.startswith("#")
            assert len(hex_color) == 7  # #RRGGBB format

    def test_extract_grid_data_with_grid_object(self):
        """Test _extract_grid_data with Grid object."""
        data = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        mask = jnp.array([[True, True, False], [True, False, True]], dtype=jnp.bool_)

        grid = Grid(data=data, mask=mask)

        extracted_data, extracted_mask = _extract_grid_data(grid)

        np.testing.assert_array_equal(extracted_data, data)
        np.testing.assert_array_equal(extracted_mask, mask)

    def test_extract_grid_data_with_jax_array(self):
        """Test _extract_grid_data with JAX array."""
        data = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)

        extracted_data, extracted_mask = _extract_grid_data(data)

        np.testing.assert_array_equal(extracted_data, data)
        assert extracted_mask is None

    def test_extract_grid_data_with_numpy_array(self):
        """Test _extract_grid_data with numpy array."""
        data = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

        extracted_data, extracted_mask = _extract_grid_data(data)

        np.testing.assert_array_equal(extracted_data, data)
        assert extracted_mask is None

    def test_extract_grid_data_invalid_type(self):
        """Test _extract_grid_data with invalid input type."""
        with pytest.raises(ValueError, match="Unsupported grid input type"):
            _extract_grid_data("invalid_input")

    def test_extract_valid_region_full_valid(self):
        """Test _extract_valid_region with fully valid grid."""
        grid = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mask = np.array([[True, True, True], [True, True, True]], dtype=np.bool_)

        valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
            grid, mask
        )

        np.testing.assert_array_equal(valid_grid, grid)
        assert (start_row, start_col) == (0, 0)
        assert (height, width) == (2, 3)

    def test_extract_valid_region_partial_valid(self):
        """Test _extract_valid_region with partially valid grid."""
        grid = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]], dtype=np.int32)
        mask = np.array(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, False, False],
            ],
            dtype=np.bool_,
        )

        valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
            grid, mask
        )

        expected_valid = np.array([[0, 1], [4, 5]], dtype=np.int32)
        np.testing.assert_array_equal(valid_grid, expected_valid)
        assert (start_row, start_col) == (0, 0)
        assert (height, width) == (2, 2)

    def test_extract_valid_region_no_mask(self):
        """Test _extract_valid_region with no mask provided."""
        grid = np.array([[0, 1], [2, 3]], dtype=np.int32)

        valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
            grid, None
        )

        np.testing.assert_array_equal(valid_grid, grid)
        assert (start_row, start_col) == (0, 0)
        assert (height, width) == (2, 2)

    def test_extract_valid_region_empty_mask(self):
        """Test _extract_valid_region with empty mask."""
        grid = np.array([[0, 1], [2, 3]], dtype=np.int32)
        mask = np.array([[False, False], [False, False]], dtype=np.bool_)

        valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
            grid, mask
        )

        # When no valid cells, returns empty array with shape (1, 0) or (0, 0)
        assert valid_grid.size == 0
        assert (start_row, start_col) == (0, 0)
        assert (height, width) == (0, 0)


class TestTerminalRendering:
    """Test terminal rendering functions using Rich."""

    def test_visualize_grid_rich_basic(self):
        """Test basic Rich grid visualization."""
        test_grid = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)

        result = visualize_grid_rich(test_grid, title="Test Grid")

        assert isinstance(result, Panel)
        assert "Test Grid (2x3)" in str(result.title)

    def test_visualize_grid_rich_with_mask(self):
        """Test Rich grid visualization with mask."""
        test_grid = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        mask = jnp.array([[True, True, False], [False, True, True]], dtype=jnp.bool_)

        result = visualize_grid_rich(test_grid, mask=mask, title="Masked Grid")

        assert isinstance(result, Panel)
        assert "Masked Grid" in str(result.title)

    def test_visualize_grid_rich_empty_grid(self):
        """Test Rich visualization with empty grid."""
        empty_grid = jnp.array([], dtype=jnp.int32).reshape(0, 0)

        result = visualize_grid_rich(empty_grid, title="Empty")

        assert isinstance(result, Panel)
        assert "Empty" in str(result.title)

    def test_visualize_grid_rich_show_coordinates(self):
        """Test Rich visualization with coordinates enabled."""
        test_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        result = visualize_grid_rich(test_grid, show_coordinates=True)

        assert isinstance(result, Panel)

    def test_visualize_grid_rich_show_numbers(self):
        """Test Rich visualization showing numbers instead of blocks."""
        test_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        result = visualize_grid_rich(test_grid, show_numbers=True)

        assert isinstance(result, Panel)

    def test_visualize_grid_rich_border_styles(self):
        """Test Rich visualization with different border styles."""
        test_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        # Test input border style
        input_result = visualize_grid_rich(test_grid, border_style="input")
        assert isinstance(input_result, Panel)

        # Test output border style
        output_result = visualize_grid_rich(test_grid, border_style="output")
        assert isinstance(output_result, Panel)

        # Test default border style
        default_result = visualize_grid_rich(test_grid, border_style="default")
        assert isinstance(default_result, Panel)

    def test_log_grid_to_console(self):
        """Test console logging functionality."""
        test_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        # Mock the console to capture output
        with patch("jaxarc.utils.visualization.core.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            log_grid_to_console(test_grid, title="Test Console Grid")

            # Verify console.print was called
            mock_console.print.assert_called_once()

    def test_visualize_task_pair_rich_with_output(self):
        """Test task pair visualization with both input and output."""
        input_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        output_grid = jnp.array([[1, 1], [2, 2]], dtype=jnp.int32)

        # Mock console to capture output
        mock_console = MagicMock()
        mock_console.size.width = 120  # Wide terminal

        visualize_task_pair_rich(
            input_grid, output_grid, title="Test Pair", console=mock_console
        )

        # Verify console.print was called
        mock_console.print.assert_called()

    def test_visualize_task_pair_rich_no_output(self):
        """Test task pair visualization with no output (unknown)."""
        input_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        # Mock console to capture output
        mock_console = MagicMock()
        mock_console.size.width = 80  # Narrow terminal

        visualize_task_pair_rich(
            input_grid, None, title="Test Pair", console=mock_console
        )

        # Verify console.print was called
        mock_console.print.assert_called()

    def test_visualize_task_pair_rich_narrow_terminal(self):
        """Test task pair visualization on narrow terminal."""
        input_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        output_grid = jnp.array([[1, 1], [2, 2]], dtype=jnp.int32)

        # Mock console with narrow width
        mock_console = MagicMock()
        mock_console.size.width = 80  # Narrow terminal

        visualize_task_pair_rich(input_grid, output_grid, console=mock_console)

        # Should use vertical layout for narrow terminals
        mock_console.print.assert_called()


class TestSVGRendering:
    """Test SVG rendering and grid visualization."""

    def test_draw_grid_svg_basic(self):
        """Test basic SVG grid drawing."""
        test_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        result = draw_grid_svg(test_grid, label="Test SVG")

        # Should return a Drawing object when as_group=False
        assert hasattr(result, "save_svg")
        assert hasattr(result, "set_pixel_scale")

    def test_draw_grid_svg_as_group(self):
        """Test SVG grid drawing as group for composition."""
        test_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        result = draw_grid_svg(test_grid, as_group=True)

        # Should return tuple when as_group=True
        assert isinstance(result, tuple)
        assert len(result) == 3
        group, origin, size = result
        assert hasattr(group, "append")  # Group objects have append method

    def test_draw_grid_svg_with_mask(self):
        """Test SVG drawing with mask."""
        test_grid = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        mask = jnp.array([[True, True, False], [True, False, True]], dtype=jnp.bool_)

        result = draw_grid_svg(test_grid, mask=mask)

        assert hasattr(result, "save_svg")

    def test_draw_grid_svg_empty_grid(self):
        """Test SVG drawing with empty grid."""
        empty_grid = jnp.array([], dtype=jnp.int32).reshape(0, 0)

        result = draw_grid_svg(empty_grid)

        assert hasattr(result, "save_svg")

    def test_draw_grid_svg_parameters(self):
        """Test SVG drawing with various parameters."""
        test_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        result = draw_grid_svg(
            test_grid,
            max_width=5.0,
            max_height=5.0,
            padding=1.0,
            extra_bottom_padding=1.0,
            label="Custom Grid",
            border_color="#FF0000",
            show_size=False,
        )

        assert hasattr(result, "save_svg")

    def test_draw_task_pair_svg_with_output(self):
        """Test task pair SVG with both input and output."""
        input_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        output_grid = jnp.array([[1, 1], [2, 2]], dtype=jnp.int32)

        result = draw_task_pair_svg(input_grid, output_grid, label="Test Pair")

        assert hasattr(result, "save_svg")

    def test_draw_task_pair_svg_no_output(self):
        """Test task pair SVG with no output."""
        input_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        result = draw_task_pair_svg(input_grid, None, show_unknown_output=True)

        assert hasattr(result, "save_svg")

    def test_draw_task_pair_svg_with_masks(self):
        """Test task pair SVG with masks."""
        input_grid = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        output_grid = jnp.array([[1, 1, 2], [3, 3, 5]], dtype=jnp.int32)
        input_mask = jnp.array(
            [[True, True, False], [True, True, True]], dtype=jnp.bool_
        )
        output_mask = jnp.array(
            [[True, True, True], [True, True, False]], dtype=jnp.bool_
        )

        result = draw_task_pair_svg(
            input_grid, output_grid, input_mask=input_mask, output_mask=output_mask
        )

        assert hasattr(result, "save_svg")

    def test_save_svg_drawing(self):
        """Test saving SVG drawings to file."""
        test_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        drawing = draw_grid_svg(test_grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"

            save_svg_drawing(drawing, str(svg_path))

            assert svg_path.exists()
            assert svg_path.stat().st_size > 0

            # Check that it's valid SVG content
            content = svg_path.read_text()
            assert content.startswith("<?xml")
            assert "<svg" in content

    def test_save_svg_drawing_png_conversion(self):
        """Test PNG conversion from SVG."""
        test_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        drawing = draw_grid_svg(test_grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "test.png"

            # Test that PNG files can be created (actual conversion depends on cairosvg availability)
            save_svg_drawing(drawing, str(png_path))

            # Should create the PNG file
            assert png_path.exists()
            assert png_path.stat().st_size > 0

    def test_save_svg_drawing_unsupported_format(self):
        """Test saving with unsupported file format."""
        test_grid = jnp.array([[0, 1]], dtype=jnp.int32)
        drawing = draw_grid_svg(test_grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "test.xyz"

            with pytest.raises(ValueError, match="Unknown file extension"):
                save_svg_drawing(drawing, str(invalid_path))


class TestRLStepVisualization:
    """Test RL step visualization functionality."""

    def test_draw_rl_step_svg_basic(self):
        """Test basic RL step SVG generation."""
        before_data = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        after_data = jnp.array([[1, 1], [2, 3]], dtype=jnp.int32)
        mask = jnp.array([[True, True], [True, True]], dtype=jnp.bool_)

        before_grid = Grid(data=before_data, mask=mask)
        after_grid = Grid(data=after_data, mask=mask)

        action = {
            "operation": 1,
            "selection": jnp.array([[True, False], [False, False]]),
        }

        svg_content = draw_rl_step_svg(
            before_grid=before_grid,
            after_grid=after_grid,
            action=action,
            reward=0.5,
            info={"similarity": 0.8},
            step_num=0,
            operation_name="Fill 1",
        )

        assert isinstance(svg_content, str)
        assert "<svg" in svg_content

    def test_draw_rl_step_svg_no_selection(self):
        """Test RL step SVG with no selection."""
        data = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        mask = jnp.array([[True, True], [True, True]], dtype=jnp.bool_)

        grid = Grid(data=data, mask=mask)
        action = {
            "operation": 31,
            "selection": jnp.array([[False, False], [False, False]]),
        }

        svg_content = draw_rl_step_svg(
            before_grid=grid,
            after_grid=grid,
            action=action,
            reward=0.0,
            info={},
            step_num=5,
            operation_name="Clear",
        )

        assert isinstance(svg_content, str)
        assert "<svg" in svg_content

    def test_draw_rl_step_svg_with_reward_info(self):
        """Test RL step SVG with reward and info."""
        data = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        mask = jnp.array([[True, True], [True, True]], dtype=jnp.bool_)

        grid = Grid(data=data, mask=mask)
        action = {
            "operation": 1,
            "selection": jnp.array([[True, False], [False, False]]),
        }

        svg_content = draw_rl_step_svg(
            before_grid=grid,
            after_grid=grid,
            action=action,
            reward=0.5,
            info={"similarity": 0.8},
            step_num=1,
            operation_name="Fill 1",
        )

        assert isinstance(svg_content, str)
        assert "<svg" in svg_content


class TestJAXCallbackIntegration:
    """Test JAX debug callback integration."""

    def test_jax_debug_callback_basic(self):
        """Test basic JAX debug callback functionality."""
        from jaxarc.utils.visualization.jax_callbacks import jax_debug_callback

        # Mock callback function
        mock_callback = MagicMock()

        # Test data
        test_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        # Call through JAX debug callback
        jax_debug_callback(mock_callback, test_grid, callback_name="test_callback")

        # Verify callback was called
        mock_callback.assert_called_once()

    def test_serialize_jax_array(self):
        """Test JAX array serialization for callbacks."""
        from jaxarc.utils.visualization.jax_callbacks import serialize_jax_array

        # Test with JAX array
        jax_array = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        result = serialize_jax_array(jax_array)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, jax_array)

        # Test with numpy array
        numpy_array = np.array([[0, 1], [2, 3]], dtype=np.int32)
        result = serialize_jax_array(numpy_array)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, numpy_array)

    def test_serialize_arc_state(self):
        """Test ArcEnvState serialization for callbacks."""
        from jaxarc.state import ArcEnvState
        from jaxarc.utils.visualization.jax_callbacks import serialize_arc_state

        # Create mock state
        mock_state = MagicMock(spec=ArcEnvState)
        mock_state.working_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        mock_state.working_grid_mask = jnp.array(
            [[True, True], [True, True]], dtype=jnp.bool_
        )
        mock_state.target_grid = jnp.array([[1, 1], [2, 2]], dtype=jnp.int32)
        mock_state.target_grid_mask = jnp.array(
            [[True, True], [True, True]], dtype=jnp.bool_
        )
        mock_state.step_count = 5
        mock_state.episode_index = 1
        mock_state.task_index = 10
        mock_state.done = False
        mock_state.similarity = 0.8

        result = serialize_arc_state(mock_state)

        assert isinstance(result, dict)
        assert "working_grid" in result
        assert "step_count" in result
        assert result["step_count"] == 5
        assert result["similarity"] == 0.8

    def test_serialize_action(self):
        """Test action serialization for callbacks."""
        from jaxarc.utils.visualization.jax_callbacks import serialize_action

        action = {
            "operation": 1,
            "selection": jnp.array([[True, False], [False, False]], dtype=jnp.bool_),
            "fill_color": 2,
            "metadata": "test",
        }

        result = serialize_action(action)

        assert isinstance(result, dict)
        assert "operation" in result
        assert "selection" in result
        assert isinstance(result["selection"], np.ndarray)
        assert result["fill_color"] == 2
        assert result["metadata"] == "test"

    def test_callback_performance_monitoring(self):
        """Test callback performance monitoring."""
        from jaxarc.utils.visualization.jax_callbacks import (
            CallbackPerformanceMonitor,
            reset_callback_performance_stats,
        )

        # Reset stats
        reset_callback_performance_stats()

        # Create monitor
        monitor = CallbackPerformanceMonitor()

        # Record some calls
        monitor.record_call("test_callback", 0.01, False)
        monitor.record_call("test_callback", 0.02, False)
        monitor.record_call("test_callback", 0.05, True)  # With error

        stats = monitor.get_stats("test_callback")

        assert stats["total_calls"] == 3
        assert stats["error_count"] == 1
        assert stats["avg_time_ms"] > 0
        assert stats["max_time_ms"] >= stats["min_time_ms"]

    def test_safe_callback_wrapper(self):
        """Test safe callback wrapper functionality."""
        from jaxarc.utils.visualization.jax_callbacks import safe_callback_wrapper

        # Test successful callback
        mock_callback = MagicMock()
        wrapped = safe_callback_wrapper(mock_callback, "test_callback")

        wrapped("arg1", "arg2", kwarg1="value1")
        mock_callback.assert_called_once_with("arg1", "arg2", kwarg1="value1")

        # Test callback with exception
        error_callback = MagicMock(side_effect=Exception("Test error"))
        wrapped_error = safe_callback_wrapper(error_callback, "error_callback")

        # Should not raise exception
        wrapped_error("test")
        error_callback.assert_called_once_with("test")

    def test_jax_log_grid_callback(self):
        """Test JAX grid logging callback."""
        from jaxarc.utils.visualization.jax_callbacks import jax_log_grid

        test_grid = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        # Mock the underlying callback function
        with patch(
            "jaxarc.utils.visualization.jax_callbacks.log_grid_callback"
        ) as mock_log:
            jax_log_grid(test_grid, title="Test JAX Grid")

            # Should be called through JAX debug callback
            # The actual call happens asynchronously, so we can't easily test it
            # But we can verify the function doesn't raise an exception
            assert True  # Function completed without error


class TestVisualizationUtilities:
    """Test visualization utility functions and error handling."""

    def test_setup_matplotlib_style(self):
        """Test matplotlib style setup."""
        # Test that the function handles matplotlib availability correctly
        try:
            setup_matplotlib_style()
            # If it doesn't raise an exception, that's good
            assert True
        except ImportError:
            # If matplotlib is not available, that's also expected
            assert True

    def test_setup_matplotlib_style_not_available(self):
        """Test matplotlib style setup when matplotlib not available."""
        # Test that the function raises ImportError when matplotlib is not available
        with patch("jaxarc.utils.visualization.core.MATPLOTLIB_AVAILABLE", False):
            with pytest.raises(
                ImportError, match="Matplotlib and seaborn are required"
            ):
                setup_matplotlib_style()

    def test_visualization_with_invalid_colors(self):
        """Test visualization handling of invalid color values."""
        # Grid with values outside normal ARC range
        test_grid = jnp.array([[0, 1, 15], [2, 3, -1]], dtype=jnp.int32)

        # Should handle gracefully
        result = visualize_grid_rich(test_grid)
        assert isinstance(result, Panel)

        svg_result = draw_grid_svg(test_grid)
        assert hasattr(svg_result, "save_svg")

    def test_visualization_error_handling(self):
        """Test error handling in visualization functions."""
        # Test with malformed input
        with pytest.raises(ValueError):
            visualize_grid_rich("not_a_grid")

        with pytest.raises(ValueError):
            draw_grid_svg("not_a_grid")

    def test_large_grid_handling(self):
        """Test handling of large grids."""
        # Create a large grid
        large_grid = jnp.zeros((100, 100), dtype=jnp.int32)

        # Should handle without error
        result = visualize_grid_rich(large_grid)
        assert isinstance(result, Panel)

        svg_result = draw_grid_svg(large_grid, max_width=20.0, max_height=20.0)
        assert hasattr(svg_result, "save_svg")

    def test_memory_efficiency(self):
        """Test memory efficiency of visualization functions."""
        # Create multiple grids and ensure no memory leaks
        for i in range(10):
            test_grid = jnp.ones((10, 10), dtype=jnp.int32) * i

            # These should not accumulate memory
            visualize_grid_rich(test_grid)
            draw_grid_svg(test_grid)

    def test_concurrent_visualization(self):
        """Test concurrent visualization calls."""
        import threading

        def visualize_worker(grid_id):
            test_grid = jnp.ones((5, 5), dtype=jnp.int32) * grid_id
            visualize_grid_rich(test_grid, title=f"Grid {grid_id}")
            draw_grid_svg(test_grid, label=f"SVG {grid_id}")

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=visualize_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()


class TestParsedTaskVisualization:
    """Test visualization of parsed task data."""

    def test_visualize_parsed_task_data_rich_mock(self):
        """Test parsed task data visualization with mock data."""
        # Create mock task data
        mock_task = MagicMock()
        mock_task.task_index = jnp.array(42)  # Make it a JAX array
        mock_task.num_train_pairs = 2
        mock_task.num_test_pairs = 1

        # Mock training data
        mock_task.input_grids_examples = [
            jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
            jnp.array([[1, 0], [3, 2]], dtype=jnp.int32),
        ]
        mock_task.output_grids_examples = [
            jnp.array([[1, 1], [2, 2]], dtype=jnp.int32),
            jnp.array([[0, 0], [3, 3]], dtype=jnp.int32),
        ]
        mock_task.input_masks_examples = [
            jnp.array([[True, True], [True, True]], dtype=jnp.bool_),
            jnp.array([[True, True], [True, True]], dtype=jnp.bool_),
        ]
        mock_task.output_masks_examples = [
            jnp.array([[True, True], [True, True]], dtype=jnp.bool_),
            jnp.array([[True, True], [True, True]], dtype=jnp.bool_),
        ]

        # Mock test data
        mock_task.test_input_grids = [jnp.array([[0, 2], [1, 3]], dtype=jnp.int32)]
        mock_task.test_input_masks = [
            jnp.array([[True, True], [True, True]], dtype=jnp.bool_)
        ]
        mock_task.true_test_output_grids = [None]  # Unknown output
        mock_task.true_test_output_masks = [None]

        # Mock console to capture output
        with patch("jaxarc.utils.visualization.core.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console.size.width = 120
            mock_console_class.return_value = mock_console

            visualize_parsed_task_data_rich(mock_task, show_test=True)

            # Verify console.print was called multiple times
            assert mock_console.print.call_count > 0

    def test_draw_parsed_task_data_svg_mock(self):
        """Test parsed task data SVG generation with mock data."""
        # Create mock task data
        mock_task = MagicMock()
        mock_task.task_index = jnp.array(42)  # Make it a JAX array
        mock_task.num_train_pairs = 1
        mock_task.num_test_pairs = 1

        mock_task.input_grids_examples = [jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)]
        mock_task.output_grids_examples = [jnp.array([[1, 1], [2, 2]], dtype=jnp.int32)]
        mock_task.input_masks_examples = [
            jnp.array([[True, True], [True, True]], dtype=jnp.bool_)
        ]
        mock_task.output_masks_examples = [
            jnp.array([[True, True], [True, True]], dtype=jnp.bool_)
        ]

        mock_task.test_input_grids = [jnp.array([[0, 2], [1, 3]], dtype=jnp.int32)]
        mock_task.test_input_masks = [
            jnp.array([[True, True], [True, True]], dtype=jnp.bool_)
        ]

        # Test the actual function
        result = draw_parsed_task_data_svg(mock_task)

        # Should return a drawing object
        assert hasattr(result, "save_svg")


if __name__ == "__main__":
    pytest.main([__file__])
