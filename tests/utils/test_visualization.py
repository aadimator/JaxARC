"""Tests for the visualization module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
from rich.panel import Panel

from jaxarc.utils.visualization import (
    draw_grid_svg,
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

            try:
                save_svg_drawing(drawing, str(invalid_path))  # type: ignore[arg-type]
                raise AssertionError("Expected ValueError for unsupported format")
            except ValueError as e:
                assert "Unknown file extension" in str(e)

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
        try:
            visualize_grid_rich("not a grid")  # type: ignore[arg-type]
            raise AssertionError("Expected ValueError for invalid input")
        except ValueError as e:
            assert "Unsupported grid input type" in str(e)

        try:
            draw_grid_svg(42)  # type: ignore[arg-type]
            raise AssertionError("Expected ValueError for invalid input")
        except ValueError as e:
            assert "Unsupported grid input type" in str(e)
