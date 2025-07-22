"""Tests for the visualization module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
from rich.panel import Panel

from jaxarc.envs.operations import (
    OPERATION_NAMES,
    get_operation_display_text,
    get_operation_name,
    is_valid_operation_id,
)
from jaxarc.types import Grid
from jaxarc.utils.visualization import (
    _extract_grid_data,
    draw_grid_svg,
    draw_rl_step_svg,
    log_grid_to_console,
    save_svg_drawing,
    visualize_grid_rich,
)


class TestGridVisualization:
    """Test grid visualization functions."""

    def test_visualize_grid_rich_basic(self):
        """Test basic Rich grid visualization."""
        test_grid = jnp.array(
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
            dtype=jnp.int32,
        )

        panel = visualize_grid_rich(test_grid, title="Test Grid")

        assert isinstance(panel, Panel)
        assert "Test Grid (2x3)" in str(panel.title)

    def test_visualize_grid_rich_with_mask(self):
        """Test Rich grid visualization with mask."""
        test_grid = jnp.array(
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
            dtype=jnp.int32,
        )

        mask = jnp.array(
            [
                [True, True, False],
                [False, True, True],
            ],
            dtype=jnp.bool_,
        )

        panel = visualize_grid_rich(test_grid, mask=mask, title="Masked Grid")

        assert isinstance(panel, Panel)
        assert "Masked Grid" in str(panel.title)

    def test_visualize_grid_rich_empty(self):
        """Test Rich grid visualization with empty grid."""
        empty_grid = jnp.array([], dtype=jnp.int32).reshape(0, 0)

        panel = visualize_grid_rich(empty_grid, title="Empty")

        assert isinstance(panel, Panel)
        assert "Empty" in str(panel.title)

    def test_log_grid_to_console(self):
        """Test console logging functionality."""
        test_grid = jnp.array(
            [
                [0, 1],
                [2, 3],
            ],
            dtype=jnp.int32,
        )

        # This function prints to console, so we can't easily test the output
        # but we can ensure it doesn't raise an exception
        log_grid_to_console(test_grid, title="Test Console Grid")

    def test_draw_grid_svg_basic(self):
        """Test basic SVG grid drawing."""
        test_grid = jnp.array(
            [
                [0, 1],
                [2, 3],
            ],
            dtype=jnp.int32,
        )

        result = draw_grid_svg(test_grid, label="Test SVG")

        # Should return a Drawing object (not a tuple) when as_group=False
        assert hasattr(result, "save_svg")  # Drawing objects have this method

    def test_draw_grid_svg_as_group(self):
        """Test SVG grid drawing as group."""
        test_grid = jnp.array(
            [
                [0, 1],
                [2, 3],
            ],
            dtype=jnp.int32,
        )

        result = draw_grid_svg(test_grid, as_group=True)

        # Should return a tuple when as_group=True
        assert isinstance(result, tuple)
        assert len(result) == 3
        group, origin, size = result
        assert hasattr(group, "append")  # Group objects have this method

    def test_draw_grid_svg_empty(self):
        """Test SVG drawing with empty grid."""
        empty_grid = jnp.array([], dtype=jnp.int32).reshape(0, 0)

        result = draw_grid_svg(empty_grid)

        # Should still return a Drawing object
        assert hasattr(result, "save_svg")

    def test_save_svg_drawing(self):
        """Test saving SVG drawings to file."""
        test_grid = jnp.array(
            [
                [0, 1],
                [2, 3],
            ],
            dtype=jnp.int32,
        )

        drawing = draw_grid_svg(test_grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"

            # This should be a Drawing object, not a tuple
            assert hasattr(drawing, "save_svg")
            save_svg_drawing(drawing, str(svg_path))  # type: ignore[arg-type]

            assert svg_path.exists()
            assert svg_path.stat().st_size > 0

    def test_save_svg_drawing_unsupported_format(self):
        """Test saving with unsupported file format."""
        test_grid = jnp.array([[0, 1]], dtype=jnp.int32)
        drawing = draw_grid_svg(test_grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "test.xyz"

            with pytest.raises(ValueError, match="Unknown file extension"):
                save_svg_drawing(drawing, str(invalid_path))  # type: ignore[arg-type]

    def test_grid_with_mask_partial_valid(self):
        """Test grid visualization with partially valid mask."""
        test_grid = jnp.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 0, 1],
            ],
            dtype=jnp.int32,
        )

        # Only the top-left 2x2 region is valid
        mask = jnp.array(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, False, False],
            ],
            dtype=jnp.bool_,
        )

        panel = visualize_grid_rich(test_grid, mask, title="Partial Grid")
        assert isinstance(panel, Panel)

        svg = draw_grid_svg(test_grid, mask)
        assert hasattr(svg, "save_svg")

    def test_invalid_grid_input(self):
        """Test handling of invalid grid input."""
        with pytest.raises(ValueError, match="Unsupported grid input type"):
            visualize_grid_rich("not a grid")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Unsupported grid input type"):
            draw_grid_svg(42)  # type: ignore[arg-type]


class TestGridTypeSupport:
    """Test Grid type support in visualization functions."""

    def test_extract_grid_data_with_grid_object(self):
        """Test _extract_grid_data with Grid object."""
        data = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        mask = jnp.array([[True, True], [False, True]], dtype=jnp.bool_)

        grid = Grid(data=data, mask=mask)

        extracted_data, extracted_mask = _extract_grid_data(grid)

        assert jnp.array_equal(extracted_data, data)
        assert jnp.array_equal(extracted_mask, mask)

    def test_extract_grid_data_with_jax_array(self):
        """Test _extract_grid_data with JAX array."""
        data = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        extracted_data, extracted_mask = _extract_grid_data(data)

        assert jnp.array_equal(extracted_data, data)
        assert extracted_mask is None

    def test_extract_grid_data_invalid_type(self):
        """Test _extract_grid_data with invalid type."""
        with pytest.raises(ValueError, match="Unsupported grid input type"):
            _extract_grid_data("invalid")  # type: ignore[arg-type]

    def test_visualize_grid_rich_with_grid_object(self):
        """Test visualize_grid_rich with Grid object."""
        data = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        mask = jnp.array([[True, True], [False, True]], dtype=jnp.bool_)

        grid = Grid(data=data, mask=mask)

        panel = visualize_grid_rich(grid, title="Grid Object Test")
        assert isinstance(panel, Panel)

    def test_draw_grid_svg_with_grid_object(self):
        """Test draw_grid_svg with Grid object."""
        data = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        mask = jnp.array([[True, True], [False, True]], dtype=jnp.bool_)

        grid = Grid(data=data, mask=mask)

        result = draw_grid_svg(grid, label="Grid Object SVG")
        assert hasattr(result, "save_svg")


class TestOperationNames:
    """Test operation names utility functions."""

    def test_get_operation_name_valid(self):
        """Test get_operation_name with valid IDs."""
        assert get_operation_name(0) == "Fill 0"
        assert get_operation_name(10) == "Flood Fill 0"
        assert get_operation_name(20) == "Move Up"
        assert get_operation_name(34) == "Submit"

    def test_get_operation_name_invalid(self):
        """Test get_operation_name with invalid ID."""
        with pytest.raises(ValueError, match="Unknown operation ID"):
            get_operation_name(999)

    def test_get_operation_display_text(self):
        """Test get_operation_display_text formatting."""
        assert get_operation_display_text(0) == "Op 0: Fill 0"
        assert get_operation_display_text(20) == "Op 20: Move Up"

    def test_is_valid_operation_id(self):
        """Test is_valid_operation_id function."""
        assert is_valid_operation_id(0) is True
        assert is_valid_operation_id(34) is True
        assert is_valid_operation_id(999) is False
        assert is_valid_operation_id(-1) is False

    def test_operation_names_coverage(self):
        """Test that all expected operation IDs are covered."""
        # Test some key operations exist
        expected_ops = [0, 9, 10, 19, 20, 23, 24, 27, 28, 31, 32, 34]
        for op_id in expected_ops:
            assert op_id in OPERATION_NAMES
            assert is_valid_operation_id(op_id)


class TestRLStepVisualization:
    """Test RL step visualization functionality."""

    def test_draw_rl_step_svg_basic(self):
        """Test basic RL step SVG generation."""
        # Create simple before and after grids
        before_data = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        after_data = jnp.array([[1, 1], [2, 3]], dtype=jnp.int32)
        mask = jnp.array([[True, True], [True, True]], dtype=jnp.bool_)

        before_grid = Grid(data=before_data, mask=mask)
        after_grid = Grid(data=after_data, mask=mask)

        # Create selection mask
        selection_mask = jnp.array([[True, False], [False, False]], dtype=jnp.bool_)

        # Generate SVG
        svg_content = draw_rl_step_svg(
            before_grid=before_grid,
            after_grid=after_grid,
            selection_mask=selection_mask,
            operation_id=1,
            step_number=0,
            label="Test Step",
        )

        # Check that SVG contains expected elements
        assert isinstance(svg_content, str)
        assert "<svg" in svg_content
        assert "Test Step - Step 0" in svg_content
        assert "Op 1: Fill 1" in svg_content

    def test_draw_rl_step_svg_no_selection(self):
        """Test RL step SVG with no selection."""
        # Create simple grids
        data = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        mask = jnp.array([[True, True], [True, True]], dtype=jnp.bool_)

        grid = Grid(data=data, mask=mask)

        # Create empty selection mask
        selection_mask = jnp.array([[False, False], [False, False]], dtype=jnp.bool_)

        # Generate SVG
        svg_content = draw_rl_step_svg(
            before_grid=grid,
            after_grid=grid,
            selection_mask=selection_mask,
            operation_id=31,
            step_number=5,
        )

        # Check basic structure
        assert isinstance(svg_content, str)
        assert "<svg" in svg_content
        assert "Step 5" in svg_content
        assert "Op 31: Clear" in svg_content
