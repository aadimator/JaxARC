"""Tests for parser utility functions."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxarc.parsers.utils import (
    convert_grid_to_jax,
    log_parsing_stats,
    pad_array_sequence,
    pad_grid_to_size,
    validate_arc_grid_data,
)


class TestParserUtils:
    """Test suite for parser utility functions."""

    def test_validate_arc_grid_data_valid(self):
        """Test validation of valid grid data."""
        # Should not raise for valid data
        validate_arc_grid_data([[1, 2], [3, 4]])
        validate_arc_grid_data([[0]])
        validate_arc_grid_data([[1, 2, 3]])

    def test_validate_arc_grid_data_invalid(self):
        """Test validation of invalid grid data."""
        # Empty grid
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_arc_grid_data([])

        # Non-list input
        with pytest.raises(ValueError, match="must be a list"):
            validate_arc_grid_data("not a list")  # type: ignore[arg-type]

        # Non-list rows
        with pytest.raises(ValueError, match="must be a list of lists"):
            validate_arc_grid_data([1, 2, 3])  # type: ignore[list-item]

        # Inconsistent row lengths
        with pytest.raises(ValueError, match="same length"):
            validate_arc_grid_data([[1, 2], [3]])

        # Non-integer cells
        with pytest.raises(ValueError, match="must be an integer"):
            validate_arc_grid_data([[1, 2.5]])  # type: ignore[list-item]

    def test_convert_grid_to_jax(self):
        """Test conversion of grid data to JAX arrays."""
        grid_data = [[1, 2], [3, 4]]
        jax_grid = convert_grid_to_jax(grid_data)

        assert isinstance(jax_grid, jnp.ndarray)
        assert jax_grid.dtype == jnp.int32
        assert jax_grid.shape == (2, 2)
        assert jnp.array_equal(jax_grid, jnp.array([[1, 2], [3, 4]]))

    def test_convert_invalid_grid_to_jax(self):
        """Test conversion of invalid grid data."""
        with pytest.raises(ValueError, match="cannot be empty"):
            convert_grid_to_jax([])

    def test_pad_grid_to_size(self):
        """Test padding grid to target size."""
        grid = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        padded_grid, mask = pad_grid_to_size(grid, 4, 4, fill_value=0)

        # Check shape
        assert padded_grid.shape == (4, 4)
        assert mask.shape == (4, 4)

        # Check original data is preserved
        assert jnp.array_equal(padded_grid[:2, :2], grid)

        # Check padding
        assert jnp.all(padded_grid[2:, :] == 0)
        assert jnp.all(padded_grid[:, 2:] == 0)

        # Check mask
        assert jnp.all(mask[:2, :2])  # Original region should be True
        assert not jnp.any(mask[2:, :])  # Padded rows should be False
        assert not jnp.any(mask[:, 2:])  # Padded columns should be False

    def test_pad_grid_exceeds_target(self):
        """Test padding when grid exceeds target size."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="exceed target dimensions"):
            pad_grid_to_size(grid, 2, 2)

    def test_pad_array_sequence(self):
        """Test padding a sequence of arrays."""
        arrays = [
            jnp.array([[1, 2]], dtype=jnp.int32),
            jnp.array([[3]], dtype=jnp.int32),
        ]

        padded_arrays, masks = pad_array_sequence(arrays, 4, 3, 3)

        # Check final shape
        assert padded_arrays.shape == (4, 3, 3)
        assert masks.shape == (4, 3, 3)

        # Check original data preservation
        assert jnp.array_equal(padded_arrays[0, :1, :2], jnp.array([[1, 2]]))
        assert jnp.array_equal(padded_arrays[1, :1, :1], jnp.array([[3]]))

        # Check empty slots are filled
        assert jnp.all(padded_arrays[2:, :, :] == 0)  # Empty slots should be zero

        # Check masks
        assert jnp.all(masks[0, :1, :2])  # First array's valid region
        assert jnp.all(masks[1, :1, :1])  # Second array's valid region
        assert not jnp.any(masks[2:, :, :])  # Empty slots should be False

    def test_pad_array_sequence_exceeds_length(self):
        """Test padding when sequence exceeds target length."""
        arrays = [
            jnp.array([[1]], dtype=jnp.int32),
            jnp.array([[2]], dtype=jnp.int32),
            jnp.array([[3]], dtype=jnp.int32),
        ]

        with pytest.raises(ValueError, match="exceeds target length"):
            pad_array_sequence(arrays, 2, 3, 3)

    def test_pad_array_sequence_exceeds_dimensions(self):
        """Test padding when array dimensions exceed target."""
        arrays = [
            jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),  # Too wide
        ]

        with pytest.raises(ValueError, match="Grid dimensions.*exceed target"):
            pad_array_sequence(arrays, 2, 2, 2)

    def test_log_parsing_stats(self):
        """Test logging of parsing statistics."""
        # For now, just test that the function runs without error
        # In a full implementation, we would mock the logger
        log_parsing_stats(3, 2, (10, 15), "test_task")
        # Test passes if no exception is raised

    def test_log_parsing_stats_no_task_id(self):
        """Test logging without task ID."""
        # For now, just test that the function runs without error
        log_parsing_stats(1, 1, (5, 5))
        # Test passes if no exception is raised

    def test_pad_grid_with_custom_fill_value(self):
        """Test padding with custom fill value."""
        grid = jnp.array([[1]], dtype=jnp.int32)
        padded_grid, mask = pad_grid_to_size(grid, 3, 3, fill_value=99)

        # Check that padding uses custom fill value
        assert padded_grid[0, 0] == 1  # Original value
        assert jnp.all(padded_grid[1:, :] == 99)  # Padded rows
        assert jnp.all(padded_grid[:, 1:] == 99)  # Padded columns

        # Mask should still be correct
        assert mask[0, 0]  # Original cell
        assert not jnp.any(mask[1:, :])  # Padded rows
        assert not jnp.any(mask[:, 1:])  # Padded columns

    def test_pad_grid_exact_size(self):
        """Test padding when grid is already exact target size."""
        grid = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        padded_grid, mask = pad_grid_to_size(grid, 2, 2)

        # Should be identical to original
        assert jnp.array_equal(padded_grid, grid)
        assert jnp.all(mask)  # All cells should be valid
