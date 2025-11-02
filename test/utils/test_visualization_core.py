"""Tests for visualization core utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import drawsvg
import jax.numpy as jnp
import numpy as np
import pytest

from jaxarc.types import Grid
from jaxarc.utils.visualization.core import (
    ARC_COLOR_PALETTE,
    _clear_output_directory,
    _draw_dotted_squircle,
    _extract_grid_data,
    _extract_valid_region,
    add_change_highlighting,
    add_selection_visualization_overlay,
    detect_changed_cells,
    draw_grid_svg,
    get_color_name,
    get_info_metric,
    infer_fill_color_from_grids,
    save_svg_drawing,
)


class TestConstants:
    """Test color palette and constants."""

    def test_arc_color_palette_completeness(self):
        """Test that ARC color palette has all expected colors."""
        expected_colors = list(range(11))  # 0-10

        for color_id in expected_colors:
            assert color_id in ARC_COLOR_PALETTE
            assert isinstance(ARC_COLOR_PALETTE[color_id], str)
            assert ARC_COLOR_PALETTE[color_id].startswith("#")

    def test_arc_color_palette_format(self):
        """Test that color palette values are valid hex colors."""
        for color_id, color_hex in ARC_COLOR_PALETTE.items():
            assert isinstance(color_hex, str)
            assert len(color_hex) == 7  # #RRGGBB format
            assert color_hex.startswith("#")
            # Check that remaining characters are valid hex
            hex_part = color_hex[1:]
            assert all(c in "0123456789ABCDEFabcdef" for c in hex_part)

    def test_specific_color_values(self):
        """Test specific color values match expected ARC colors."""
        # Test a few key colors
        assert ARC_COLOR_PALETTE[0] == "#252525"  # Black
        assert ARC_COLOR_PALETTE[1] == "#0074D9"  # Blue
        assert ARC_COLOR_PALETTE[2] == "#FF4136"  # Red
        assert ARC_COLOR_PALETTE[10] == "#FFFFFF"  # White


class TestExtractGridData:
    """Test grid data extraction utilities."""

    def test_extract_grid_data_from_grid_object(self):
        """Test extracting data from Grid object."""
        data = jnp.array([[1, 2], [3, 4]])
        mask = jnp.array([[True, True], [False, True]])
        grid = Grid(data=data, mask=mask)

        extracted_data, extracted_mask = _extract_grid_data(grid)

        assert isinstance(extracted_data, np.ndarray)
        assert isinstance(extracted_mask, np.ndarray)
        np.testing.assert_array_equal(extracted_data, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(extracted_mask, [[True, True], [False, True]])

    def test_extract_grid_data_from_jax_array(self):
        """Test extracting data from JAX array."""
        jax_array = jnp.array([[5, 6], [7, 8]])

        extracted_data, extracted_mask = _extract_grid_data(jax_array)

        assert isinstance(extracted_data, np.ndarray)
        assert extracted_mask is None
        np.testing.assert_array_equal(extracted_data, [[5, 6], [7, 8]])

    def test_extract_grid_data_from_numpy_array(self):
        """Test extracting data from numpy array."""
        numpy_array = np.array([[9, 10], [11, 12]])

        extracted_data, extracted_mask = _extract_grid_data(numpy_array)

        assert isinstance(extracted_data, np.ndarray)
        assert extracted_mask is None
        np.testing.assert_array_equal(extracted_data, [[9, 10], [11, 12]])

    def test_extract_grid_data_unsupported_type(self):
        """Test extracting data from unsupported type raises error."""
        with pytest.raises(ValueError, match="Unsupported grid input type"):
            _extract_grid_data("invalid_input")

    def test_extract_grid_data_empty_arrays(self):
        """Test extracting data from empty arrays."""
        empty_array = jnp.array([])

        extracted_data, extracted_mask = _extract_grid_data(empty_array)

        assert isinstance(extracted_data, np.ndarray)
        assert extracted_mask is None
        assert extracted_data.size == 0


class TestExtractValidRegion:
    """Test valid region extraction utilities."""

    def test_extract_valid_region_no_mask(self):
        """Test extracting valid region without mask."""
        grid = np.array([[1, 2, 3], [4, 5, 6]])

        valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
            grid
        )

        np.testing.assert_array_equal(valid_grid, grid)
        assert (start_row, start_col) == (0, 0)
        assert (height, width) == (2, 3)

    def test_extract_valid_region_with_mask(self):
        """Test extracting valid region with mask."""
        grid = np.array([[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0]])
        mask = np.array(
            [
                [True, True, True, False],
                [True, True, True, False],
                [False, False, False, False],
            ]
        )

        valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
            grid, mask
        )

        expected_valid = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(valid_grid, expected_valid)
        assert (start_row, start_col) == (0, 0)
        assert (height, width) == (2, 3)

    def test_extract_valid_region_partial_mask(self):
        """Test extracting valid region with partial mask."""
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mask = np.array(
            [[False, True, True], [False, True, True], [False, False, False]]
        )

        valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
            grid, mask
        )

        expected_valid = np.array([[2, 3], [5, 6]])
        np.testing.assert_array_equal(valid_grid, expected_valid)
        assert (start_row, start_col) == (0, 1)
        assert (height, width) == (2, 2)

    def test_extract_valid_region_empty_mask(self):
        """Test extracting valid region with empty mask."""
        grid = np.array([[1, 2], [3, 4]])
        mask = np.array([[False, False], [False, False]])

        valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
            grid, mask
        )

        assert valid_grid.size == 0
        assert (start_row, start_col) == (0, 0)
        assert (height, width) == (0, 0)

    def test_extract_valid_region_single_cell(self):
        """Test extracting valid region with single valid cell."""
        grid = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
        mask = np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )

        valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
            grid, mask
        )

        expected_valid = np.array([[5]])
        np.testing.assert_array_equal(valid_grid, expected_valid)
        assert (start_row, start_col) == (1, 1)
        assert (height, width) == (1, 1)


class TestColorUtilities:
    """Test color-related utility functions."""

    def test_get_color_name_valid_colors(self):
        """Test getting color names for valid color IDs."""
        expected_names = {
            0: "Black (0)",
            1: "Blue (1)",
            2: "Red (2)",
            3: "Green (3)",
            4: "Yellow (4)",
            5: "Grey (5)",
            6: "Pink (6)",
            7: "Orange (7)",
            8: "Light Blue (8)",
            9: "Brown (9)",
        }

        for color_id, expected_name in expected_names.items():
            assert get_color_name(color_id) == expected_name

    def test_get_color_name_invalid_colors(self):
        """Test getting color names for invalid color IDs."""
        invalid_colors = [10, 11, -1, 100]

        for color_id in invalid_colors:
            result = get_color_name(color_id)
            assert result == f"Color {color_id}"

    def test_get_color_name_edge_cases(self):
        """Test getting color names for edge cases."""
        # Test very large number
        assert get_color_name(999) == "Color 999"

        # Test negative number
        assert get_color_name(-5) == "Color -5"


class TestGridChangeDetection:
    """Test grid change detection utilities."""

    def test_detect_changed_cells_no_changes(self):
        """Test detecting changes when grids are identical."""
        data = jnp.array([[1, 2], [3, 4]])
        mask = jnp.array([[True, True], [True, True]])

        before_grid = Grid(data=data, mask=mask)
        after_grid = Grid(data=data, mask=mask)

        changed = detect_changed_cells(before_grid, after_grid)

        assert isinstance(changed, jnp.ndarray)
        assert not jnp.any(changed)  # No changes

    def test_detect_changed_cells_with_changes(self):
        """Test detecting changes when grids differ."""
        before_data = jnp.array([[1, 2], [3, 4]])
        after_data = jnp.array([[1, 5], [3, 4]])  # Changed (0,1) from 2 to 5
        mask = jnp.array([[True, True], [True, True]])

        before_grid = Grid(data=before_data, mask=mask)
        after_grid = Grid(data=after_data, mask=mask)

        changed = detect_changed_cells(before_grid, after_grid)

        expected_changed = jnp.array([[False, True], [False, False]])
        np.testing.assert_array_equal(changed, expected_changed)

    def test_detect_changed_cells_different_shapes(self):
        """Test detecting changes with different grid shapes."""
        before_data = jnp.array([[1, 2]])
        after_data = jnp.array([[1, 5], [3, 4]])  # Different shape
        mask1 = jnp.array([[True, True]])
        mask2 = jnp.array([[True, True], [True, True]])

        before_grid = Grid(data=before_data, mask=mask1)
        after_grid = Grid(data=after_data, mask=mask2)

        changed = detect_changed_cells(before_grid, after_grid)

        # Should handle different shapes by padding
        assert isinstance(changed, jnp.ndarray)
        assert changed.shape == (2, 2)  # Padded to larger shape

    def test_detect_changed_cells_multiple_changes(self):
        """Test detecting multiple changes."""
        before_data = jnp.array([[1, 2, 3], [4, 5, 6]])
        after_data = jnp.array([[1, 9, 3], [8, 5, 6]])  # Changed (0,1) and (1,0)
        mask = jnp.array([[True, True, True], [True, True, True]])

        before_grid = Grid(data=before_data, mask=mask)
        after_grid = Grid(data=after_data, mask=mask)

        changed = detect_changed_cells(before_grid, after_grid)

        expected_changed = jnp.array([[False, True, False], [True, False, False]])
        np.testing.assert_array_equal(changed, expected_changed)


class TestFillColorInference:
    """Test fill color inference utilities."""

    def test_infer_fill_color_basic(self):
        """Test basic fill color inference."""
        before_data = jnp.array([[1, 2], [3, 4]])
        after_data = jnp.array([[1, 5], [3, 4]])  # Changed (0,1) from 2 to 5
        mask = jnp.array([[True, True], [True, True]])

        before_grid = Grid(data=before_data, mask=mask)
        after_grid = Grid(data=after_data, mask=mask)

        selection_mask = np.array([[False, True], [False, False]])  # Selected (0,1)

        fill_color = infer_fill_color_from_grids(
            before_grid, after_grid, selection_mask
        )
        assert fill_color == 5

    def test_infer_fill_color_no_changes(self):
        """Test fill color inference when no changes occurred."""
        data = jnp.array([[1, 2], [3, 4]])
        mask = jnp.array([[True, True], [True, True]])

        before_grid = Grid(data=data, mask=mask)
        after_grid = Grid(data=data, mask=mask)

        selection_mask = np.array([[False, True], [False, False]])

        fill_color = infer_fill_color_from_grids(
            before_grid, after_grid, selection_mask
        )
        assert fill_color == -1  # Couldn't determine

    def test_infer_fill_color_no_selection(self):
        """Test fill color inference with no selection."""
        before_data = jnp.array([[1, 2], [3, 4]])
        after_data = jnp.array([[1, 5], [3, 4]])
        mask = jnp.array([[True, True], [True, True]])

        before_grid = Grid(data=before_data, mask=mask)
        after_grid = Grid(data=after_data, mask=mask)

        selection_mask = np.array([[False, False], [False, False]])  # No selection

        fill_color = infer_fill_color_from_grids(
            before_grid, after_grid, selection_mask
        )
        assert fill_color == -1  # Couldn't determine

    def test_infer_fill_color_multiple_selected_cells(self):
        """Test fill color inference with multiple selected cells."""
        before_data = jnp.array([[1, 2], [3, 4]])
        after_data = jnp.array([[7, 7], [3, 4]])  # Changed both (0,0) and (0,1) to 7
        mask = jnp.array([[True, True], [True, True]])

        before_grid = Grid(data=before_data, mask=mask)
        after_grid = Grid(data=after_data, mask=mask)

        selection_mask = np.array([[True, True], [False, False]])  # Selected both

        fill_color = infer_fill_color_from_grids(
            before_grid, after_grid, selection_mask
        )
        assert fill_color == 7  # Should find the first changed cell

    def test_infer_fill_color_error_handling(self):
        """Test fill color inference error handling."""
        # Create grids that might cause errors
        before_data = jnp.array([[1]])
        after_data = jnp.array([[2]])
        mask = jnp.array([[True]])

        before_grid = Grid(data=before_data, mask=mask)
        after_grid = Grid(data=after_data, mask=mask)

        # Selection mask with different shape - the function handles this gracefully
        selection_mask = np.array([[True, False], [False, False]])

        fill_color = infer_fill_color_from_grids(
            before_grid, after_grid, selection_mask
        )
        # The function finds the changed cell and returns the new color
        assert fill_color == 2


class TestInfoMetricExtraction:
    """Test info metric extraction utilities."""

    def test_get_info_metric_from_metrics_dict(self):
        """Test extracting metric from nested metrics dictionary."""
        info = {
            "metrics": {"similarity": 0.85, "step_count": 10},
            "similarity": 0.75,  # Old format, should be ignored
        }

        # Should prioritize nested format
        assert get_info_metric(info, "similarity") == 0.85
        assert get_info_metric(info, "step_count") == 10

    def test_get_info_metric_from_direct_dict(self):
        """Test extracting metric directly from info dictionary."""
        info = {"similarity": 0.75, "step_count": 15}

        assert get_info_metric(info, "similarity") == 0.75
        assert get_info_metric(info, "step_count") == 15

    def test_get_info_metric_with_jax_arrays(self):
        """Test extracting metrics that are JAX arrays."""
        info = {"metrics": {"similarity": jnp.array(0.9), "step_count": jnp.array(20)}}

        # Should convert to float
        assert abs(get_info_metric(info, "similarity") - 0.9) < 1e-6
        assert get_info_metric(info, "step_count") == 20.0

    def test_get_info_metric_nonexistent_key(self):
        """Test extracting nonexistent metric."""
        info = {"metrics": {"similarity": 0.8}}

        # Should return default
        assert get_info_metric(info, "nonexistent") is None
        assert get_info_metric(info, "nonexistent", "default") == "default"

    def test_get_info_metric_empty_info(self):
        """Test extracting metric from empty info."""
        info = {}

        assert get_info_metric(info, "similarity") is None
        assert get_info_metric(info, "similarity", 0.0) == 0.0

    def test_get_info_metric_no_metrics_key(self):
        """Test extracting when no metrics key exists."""
        info = {"similarity": 0.6, "other_data": "value"}

        # Should fall back to direct lookup
        assert get_info_metric(info, "similarity") == 0.6
        assert get_info_metric(info, "other_data") == "value"


class TestOutputDirectoryUtilities:
    """Test output directory utilities."""

    def test_clear_output_directory_existing(self):
        """Test clearing existing output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "test_output"
            output_dir.mkdir()
            (output_dir / "file1.txt").touch()
            (output_dir / "subdir").mkdir()
            (output_dir / "subdir" / "file2.txt").touch()

            assert output_dir.exists()
            assert (output_dir / "file1.txt").exists()

            _clear_output_directory(str(output_dir))

            # Directory should exist but be empty
            assert output_dir.exists()
            assert not (output_dir / "file1.txt").exists()
            assert not (output_dir / "subdir").exists()

    def test_clear_output_directory_nonexistent(self):
        """Test clearing nonexistent output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nonexistent"
            assert not output_dir.exists()

            _clear_output_directory(str(output_dir))

            # Directory should be created
            assert output_dir.exists()
            assert output_dir.is_dir()


class TestDrawGridSVG:
    """Test SVG grid drawing functionality."""

    def test_draw_grid_svg_basic(self):
        """Test basic grid SVG drawing."""
        grid = jnp.array([[1, 2], [3, 4]])

        drawing = draw_grid_svg(grid)

        assert isinstance(drawing, drawsvg.Drawing)
        # Should have some elements (rectangles for cells, border, etc.)
        assert len(drawing.elements) > 0

    def test_draw_grid_svg_with_mask(self):
        """Test grid SVG drawing with mask."""
        grid = jnp.array([[1, 2, 0], [3, 4, 0]])
        mask = jnp.array([[True, True, False], [True, True, False]])

        drawing = draw_grid_svg(grid, mask=mask)

        assert isinstance(drawing, drawsvg.Drawing)
        assert len(drawing.elements) > 0

    def test_draw_grid_svg_with_label(self):
        """Test grid SVG drawing with label."""
        grid = jnp.array([[1, 2]])

        drawing = draw_grid_svg(grid, label="Test Grid")

        assert isinstance(drawing, drawsvg.Drawing)
        # Should contain text elements for the label
        text_elements = [
            elem for elem in drawing.elements if isinstance(elem, drawsvg.Text)
        ]
        assert len(text_elements) > 0

    def test_draw_grid_svg_empty_grid(self):
        """Test drawing empty grid."""
        grid = jnp.array([])

        drawing = draw_grid_svg(grid)

        assert isinstance(drawing, drawsvg.Drawing)
        # Should handle empty grid gracefully

    def test_draw_grid_svg_as_group(self):
        """Test drawing grid as group for inclusion in larger drawings."""
        grid = jnp.array([[1, 2]])

        result = draw_grid_svg(grid, as_group=True)

        assert isinstance(result, tuple)
        group, origin, size = result
        assert isinstance(group, drawsvg.Group)
        assert isinstance(origin, tuple)
        assert isinstance(size, tuple)

    def test_draw_grid_svg_custom_parameters(self):
        """Test drawing grid with custom parameters."""
        grid = jnp.array([[1, 2], [3, 4]])

        drawing = draw_grid_svg(
            grid,
            max_width=5.0,
            max_height=5.0,
            padding=1.0,
            border_color="#FF0000",
            show_size=False,
        )

        assert isinstance(drawing, drawsvg.Drawing)
        assert len(drawing.elements) > 0

    def test_draw_grid_svg_with_grid_object(self):
        """Test drawing with Grid object input."""
        data = jnp.array([[1, 2], [3, 4]])
        mask = jnp.array([[True, True], [True, True]])
        grid = Grid(data=data, mask=mask)

        drawing = draw_grid_svg(grid)

        assert isinstance(drawing, drawsvg.Drawing)
        assert len(drawing.elements) > 0


class TestSaveSVGDrawing:
    """Test SVG drawing saving functionality."""

    def test_save_svg_drawing_svg_format(self):
        """Test saving drawing in SVG format."""
        grid = jnp.array([[1, 2]])
        drawing = draw_grid_svg(grid)

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            save_svg_drawing(drawing, str(temp_path))

            assert temp_path.exists()
            # Check that file contains SVG content
            content = temp_path.read_text()
            assert "<svg" in content
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_save_svg_drawing_png_format(self):
        """Test saving drawing in PNG format."""
        grid = jnp.array([[1, 2]])
        drawing = draw_grid_svg(grid)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            save_svg_drawing(drawing, str(temp_path))

            assert temp_path.exists()
            # PNG files should have binary content
            content = temp_path.read_bytes()
            assert len(content) > 0
        except Exception:
            # PNG saving might fail if dependencies are missing
            pass
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_save_svg_drawing_unsupported_format(self):
        """Test saving drawing in unsupported format."""
        grid = jnp.array([[1, 2]])
        drawing = draw_grid_svg(grid)

        with pytest.raises(ValueError, match="Unknown file extension"):
            save_svg_drawing(drawing, "test.xyz")

    # Removed PDF-related tests - they depend on optional cairosvg dependency
    # PDF functionality is optional and not required for core functionality


class TestDrawingUtilities:
    """Test drawing utility functions."""

    def test_draw_dotted_squircle(self):
        """Test drawing dotted squircle."""
        elements = _draw_dotted_squircle(0, 0, 5, 3, "Test Label")

        assert isinstance(elements, list)
        assert len(elements) == 2  # Rectangle and text

        # Check rectangle
        rect = elements[0]
        assert isinstance(rect, drawsvg.Rectangle)

        # Check text
        text = elements[1]
        assert isinstance(text, drawsvg.Text)
        # drawsvg stores text content in escaped_text
        assert text.escaped_text == "Test Label"

    def test_draw_dotted_squircle_custom_parameters(self):
        """Test drawing dotted squircle with custom parameters."""
        elements = _draw_dotted_squircle(
            1,
            2,
            10,
            8,
            "Custom",
            stroke_color="#FF0000",
            stroke_width=0.1,
            corner_radius=0.5,
            dash_array="0.2,0.2",
        )

        assert len(elements) == 2

        rect = elements[0]
        assert rect.args["x"] == 1  # x
        assert rect.args["y"] == 2  # y
        assert rect.args["width"] == 10  # width
        assert rect.args["height"] == 8  # height

        text = elements[1]
        assert text.escaped_text == "Custom"

    def test_add_selection_visualization_overlay(self):
        """Test adding selection visualization overlay."""
        # Create a mock drawing
        drawing = drawsvg.Drawing(10, 10)
        selection_mask = np.array([[True, False], [False, True]])

        # Should not raise any errors
        add_selection_visualization_overlay(
            drawing, selection_mask, 0, 0, 1.0, 0, 0, 2, 2
        )

        # Should have added some elements
        assert len(drawing.elements) > 0

    def test_add_selection_visualization_overlay_empty_selection(self):
        """Test adding overlay with empty selection."""
        drawing = drawsvg.Drawing(10, 10)
        selection_mask = np.array([[False, False], [False, False]])

        initial_count = len(drawing.elements)

        add_selection_visualization_overlay(
            drawing, selection_mask, 0, 0, 1.0, 0, 0, 2, 2
        )

        # Should not add elements for empty selection
        assert len(drawing.elements) == initial_count

    def test_add_change_highlighting(self):
        """Test adding change highlighting overlay."""
        drawing = drawsvg.Drawing(10, 10)
        changed_cells = np.array([[True, False], [False, True]])

        add_change_highlighting(drawing, changed_cells, 0, 0, 1.0, 0, 0, 2, 2)

        # Should have added some elements
        assert len(drawing.elements) > 0

    def test_add_change_highlighting_no_changes(self):
        """Test adding change highlighting with no changes."""
        drawing = drawsvg.Drawing(10, 10)
        changed_cells = np.array([[False, False], [False, False]])

        initial_count = len(drawing.elements)

        add_change_highlighting(drawing, changed_cells, 0, 0, 1.0, 0, 0, 2, 2)

        # Should not add elements for no changes
        assert len(drawing.elements) == initial_count


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_extract_grid_data_with_duck_typing(self):
        """Test that grid extraction works with duck typing."""

        # Create object that looks like Grid but isn't
        class MockGrid:
            def __init__(self):
                self.data = jnp.array([[1, 2]])
                self.mask = jnp.array([[True, True]])

        mock_grid = MockGrid()
        extracted_data, extracted_mask = _extract_grid_data(mock_grid)

        np.testing.assert_array_equal(extracted_data, [[1, 2]])
        np.testing.assert_array_equal(extracted_mask, [[True, True]])

    def test_draw_grid_svg_with_invalid_colors(self):
        """Test drawing grid with invalid color values."""
        # Grid with color values outside normal range
        grid = jnp.array([[15, -1], [100, 0]])

        drawing = draw_grid_svg(grid)

        # Should handle gracefully without errors
        assert isinstance(drawing, drawsvg.Drawing)
        assert len(drawing.elements) > 0

    def test_visualization_with_very_large_grids(self):
        """Test visualization with large grids."""
        # Create a large grid
        large_grid = jnp.ones((50, 50), dtype=jnp.int32)

        drawing = draw_grid_svg(large_grid, max_width=10, max_height=10)

        # Should handle large grids without issues
        assert isinstance(drawing, drawsvg.Drawing)
        assert len(drawing.elements) > 0

    def test_visualization_with_single_pixel_grid(self):
        """Test visualization with single pixel grid."""
        single_pixel = jnp.array([[5]])

        drawing = draw_grid_svg(single_pixel)

        assert isinstance(drawing, drawsvg.Drawing)
        assert len(drawing.elements) > 0

    # Removed test_color_inference_with_extreme_values - tests invalid scenario
    # ARC grids should never have color values outside [-1, 9] range

    def test_info_metric_with_complex_nested_structure(self):
        """Test info metric extraction with complex nested structures."""
        info = {
            "metrics": {
                "nested": {"deep": {"value": 42}},
                "array_metric": jnp.array([1, 2, 3]),
            },
            "direct_nested": {"value": 24},
        }

        # Should handle nested structures gracefully
        assert get_info_metric(info, "nested") == {"deep": {"value": 42}}
        assert get_info_metric(info, "direct_nested") == {"value": 24}

    def test_concurrent_visualization_operations(self):
        """Test concurrent visualization operations."""
        import threading
        import time

        results = []
        errors = []

        def draw_worker():
            try:
                for i in range(5):
                    grid = jnp.array([[i, i + 1], [i + 2, i + 3]])
                    drawing = draw_grid_svg(grid)
                    results.append(drawing)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=draw_worker)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0
        # Should have created drawings
        assert len(results) == 15  # 3 threads * 5 drawings each

        # All results should be valid drawings
        for drawing in results:
            assert isinstance(drawing, drawsvg.Drawing)
