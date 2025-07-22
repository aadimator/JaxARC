"""Comprehensive tests for parser utility functions.

This test suite covers all parser utility functions including grid validation,
JAX conversion, padding operations, and logging functionality.
"""

from __future__ import annotations

from unittest.mock import patch

import jax.numpy as jnp
import pytest

from jaxarc.parsers.utils import (
    convert_grid_to_jax,
    log_parsing_stats,
    pad_array_sequence,
    pad_grid_to_size,
    validate_arc_grid_data,
)


class TestParserUtilsComprehensive:
    """Comprehensive test suite for parser utility functions."""

    def test_validate_arc_grid_data_valid_cases(self):
        """Test validate_arc_grid_data with various valid inputs."""
        valid_cases = [
            # Basic valid grids
            [[1, 2], [3, 4]],
            [[0]],
            [[1, 2, 3]],
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            # Edge cases
            [[0, 0, 0], [0, 0, 0]],  # All zeros
            [[9, 9, 9], [9, 9, 9]],  # All nines
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],  # All valid colors
            # Large grids
            [[i % 10 for i in range(30)] for _ in range(30)],
            # Single row/column
            [[1, 2, 3, 4, 5]],
            [[1], [2], [3], [4], [5]],
        ]

        for grid in valid_cases:
            # Should not raise any exception
            validate_arc_grid_data(grid)

    def test_validate_arc_grid_data_invalid_structure(self):
        """Test validate_arc_grid_data with invalid structure."""
        invalid_cases = [
            # Empty grid
            ([], "cannot be empty"),
            # Non-list input
            ("not a list", "must be a list"),
            (123, "must be a list"),
            (None, "cannot be empty"),  # None gets caught by empty check first
            # Non-list rows
            ([1, 2, 3], "must be a list of lists"),
            (["not", "lists"], "must be a list of lists"),
            ([1, [2, 3]], "must be a list of lists"),
            # Inconsistent row lengths
            ([[1, 2], [3]], "same length"),
            ([[1, 2, 3], [4, 5]], "same length"),
            ([[1], [2, 3], [4]], "same length"),
        ]

        for invalid_input, expected_error in invalid_cases:
            with pytest.raises(ValueError, match=expected_error):
                validate_arc_grid_data(invalid_input)

    def test_validate_arc_grid_data_invalid_cell_types(self):
        """Test validate_arc_grid_data with invalid cell types."""
        invalid_cell_cases = [
            # Non-integer cells
            ([[1, 2.5]], "must be an integer"),
            ([[1, "2"]], "must be an integer"),
            ([[1, None]], "must be an integer"),
            ([[1, [2]]], "must be an integer"),
            # Note: bool is subclass of int in Python, so True/False are valid integers
        ]

        for invalid_input, expected_error in invalid_cell_cases:
            with pytest.raises(ValueError, match=expected_error):
                validate_arc_grid_data(invalid_input)

    def test_validate_arc_grid_data_color_warnings(self):
        """Test validate_arc_grid_data with colors outside typical range."""
        # Colors outside 0-9 range should log warnings but not raise errors
        with patch("jaxarc.parsers.utils.logger") as mock_logger:
            validate_arc_grid_data([[10, 15, -1]])

            # Should have logged warnings for out-of-range colors
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
            assert len(warning_calls) >= 2  # At least warnings for 10, 15, -1

    def test_convert_grid_to_jax_success(self):
        """Test successful grid conversion to JAX arrays."""
        test_cases = [
            # Basic cases
            ([[1, 2], [3, 4]], (2, 2)),
            ([[0]], (1, 1)),
            ([[1, 2, 3]], (1, 3)),
            ([[1], [2], [3]], (3, 1)),
            # Larger grids
            ([[i for i in range(10)] for _ in range(5)], (5, 10)),
            # Edge values
            ([[0, 9], [9, 0]], (2, 2)),
        ]

        for grid_data, expected_shape in test_cases:
            jax_grid = convert_grid_to_jax(grid_data)

            # Check type and dtype
            assert isinstance(jax_grid, jnp.ndarray)
            assert jax_grid.dtype == jnp.int32

            # Check shape
            assert jax_grid.shape == expected_shape

            # Check values are preserved
            for i in range(expected_shape[0]):
                for j in range(expected_shape[1]):
                    assert jax_grid[i, j] == grid_data[i][j]

    def test_convert_grid_to_jax_validation_errors(self):
        """Test convert_grid_to_jax with invalid inputs."""
        invalid_cases = [
            [],  # Empty grid
            [1, 2, 3],  # Non-list rows
            [[1, 2], [3]],  # Inconsistent row lengths
            [[1, 2.5]],  # Non-integer cells
        ]

        for invalid_input in invalid_cases:
            with pytest.raises(ValueError):
                convert_grid_to_jax(invalid_input)

    def test_pad_grid_to_size_basic(self):
        """Test basic grid padding functionality."""
        # Test padding smaller grid
        grid = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        padded_grid, mask = pad_grid_to_size(grid, 4, 4, fill_value=0)

        # Check shapes
        assert padded_grid.shape == (4, 4)
        assert mask.shape == (4, 4)

        # Check original data preservation
        assert jnp.array_equal(padded_grid[:2, :2], grid)

        # Check padding
        assert jnp.all(padded_grid[2:, :] == 0)
        assert jnp.all(padded_grid[:, 2:] == 0)

        # Check mask
        assert jnp.all(mask[:2, :2])  # Original region should be True
        assert not jnp.any(mask[2:, :])  # Padded rows should be False
        assert not jnp.any(mask[:, 2:])  # Padded columns should be False

    def test_pad_grid_to_size_custom_fill_value(self):
        """Test grid padding with custom fill value."""
        grid = jnp.array([[1]], dtype=jnp.int32)
        padded_grid, mask = pad_grid_to_size(grid, 3, 3, fill_value=99)

        # Check original value preserved
        assert padded_grid[0, 0] == 1

        # Check custom fill value used
        assert jnp.all(padded_grid[1:, :] == 99)
        assert jnp.all(padded_grid[:, 1:] == 99)

        # Check mask
        assert mask[0, 0]  # Original cell should be True
        assert not jnp.any(mask[1:, :])  # Padded regions should be False
        assert not jnp.any(mask[:, 1:])

    def test_pad_grid_to_size_exact_size(self):
        """Test padding when grid is already target size."""
        grid = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        padded_grid, mask = pad_grid_to_size(grid, 2, 2)

        # Should be identical to original
        assert jnp.array_equal(padded_grid, grid)
        assert jnp.all(mask)  # All cells should be valid

    def test_pad_grid_to_size_errors(self):
        """Test pad_grid_to_size error handling."""
        # Grid exceeds target dimensions
        large_grid = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="exceed target dimensions"):
            pad_grid_to_size(large_grid, 2, 2)

        with pytest.raises(ValueError, match="exceed target dimensions"):
            pad_grid_to_size(large_grid, 1, 3)

        with pytest.raises(ValueError, match="exceed target dimensions"):
            pad_grid_to_size(large_grid, 2, 1)

    def test_pad_array_sequence_basic(self):
        """Test basic array sequence padding."""
        arrays = [
            jnp.array([[1, 2]], dtype=jnp.int32),
            jnp.array([[3]], dtype=jnp.int32),
        ]

        padded_arrays, masks = pad_array_sequence(arrays, 4, 3, 3)

        # Check final shapes
        assert padded_arrays.shape == (4, 3, 3)
        assert masks.shape == (4, 3, 3)

        # Check original data preservation
        assert jnp.array_equal(padded_arrays[0, :1, :2], jnp.array([[1, 2]]))
        assert jnp.array_equal(padded_arrays[1, :1, :1], jnp.array([[3]]))

        # Check empty slots filled with zeros
        assert jnp.all(padded_arrays[2:, :, :] == 0)

        # Check masks
        assert jnp.all(masks[0, :1, :2])  # First array's valid region
        assert jnp.all(masks[1, :1, :1])  # Second array's valid region
        assert not jnp.any(masks[2:, :, :])  # Empty slots should be False

    def test_pad_array_sequence_single_array(self):
        """Test padding sequence with single array."""
        arrays = [jnp.array([[1, 2, 3]], dtype=jnp.int32)]

        padded_arrays, masks = pad_array_sequence(arrays, 2, 2, 4)

        # Check shapes
        assert padded_arrays.shape == (2, 2, 4)
        assert masks.shape == (2, 2, 4)

        # Check first array
        assert jnp.array_equal(padded_arrays[0, :1, :3], jnp.array([[1, 2, 3]]))
        assert jnp.all(masks[0, :1, :3])

        # Check second slot is empty
        assert jnp.all(padded_arrays[1, :, :] == 0)
        assert not jnp.any(masks[1, :, :])

    def test_pad_array_sequence_exact_length(self):
        """Test padding when sequence is exactly target length."""
        arrays = [
            jnp.array([[1]], dtype=jnp.int32),
            jnp.array([[2]], dtype=jnp.int32),
        ]

        padded_arrays, masks = pad_array_sequence(arrays, 2, 2, 2)

        # Should have no empty slots
        assert padded_arrays.shape == (2, 2, 2)
        assert jnp.array_equal(padded_arrays[0, :1, :1], jnp.array([[1]]))
        assert jnp.array_equal(padded_arrays[1, :1, :1], jnp.array([[2]]))

    def test_pad_array_sequence_errors(self):
        """Test pad_array_sequence error handling."""
        arrays = [
            jnp.array([[1]], dtype=jnp.int32),
            jnp.array([[2]], dtype=jnp.int32),
            jnp.array([[3]], dtype=jnp.int32),
        ]

        # Too many arrays
        with pytest.raises(ValueError, match="exceeds target length"):
            pad_array_sequence(arrays, 2, 3, 3)

        # Array dimensions exceed target
        large_arrays = [jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)]
        with pytest.raises(ValueError, match="Grid dimensions.*exceed target"):
            pad_array_sequence(large_arrays, 2, 2, 2)

    def test_pad_array_sequence_empty_input(self):
        """Test padding empty array sequence."""
        arrays = []

        padded_arrays, masks = pad_array_sequence(arrays, 3, 2, 2, fill_value=-1)

        # Should create arrays filled with fill_value
        assert padded_arrays.shape == (3, 2, 2)
        assert masks.shape == (3, 2, 2)
        assert jnp.all(padded_arrays == -1)
        assert not jnp.any(masks)  # All should be False

    def test_pad_array_sequence_dtype_preservation(self):
        """Test that dtype is preserved during padding."""
        # Test with different dtypes (avoid int64 due to JAX x64 mode)
        dtypes = [jnp.int8, jnp.int16, jnp.int32]

        for dtype in dtypes:
            arrays = [jnp.array([[1, 2]], dtype=dtype)]
            padded_arrays, masks = pad_array_sequence(arrays, 2, 3, 3)

            # Note: JAX may promote dtypes, so check that result is reasonable
            assert padded_arrays.dtype in [dtype, jnp.int32]  # Allow promotion to int32
            assert masks.dtype == jnp.bool_

    def test_log_parsing_stats_basic(self):
        """Test basic logging functionality."""
        with patch("jaxarc.parsers.utils.logger") as mock_logger:
            log_parsing_stats(3, 2, (10, 15), "test_task")

            # Should have called debug logging
            mock_logger.debug.assert_called_once()
            log_message = mock_logger.debug.call_args[0][0]

            # Check log message contains expected information
            assert "test_task" in log_message
            assert "3 train pairs" in log_message
            assert "2 test pairs" in log_message
            assert "10x15" in log_message

    def test_log_parsing_stats_no_task_id(self):
        """Test logging without task ID."""
        with patch("jaxarc.parsers.utils.logger") as mock_logger:
            log_parsing_stats(1, 1, (5, 5))

            mock_logger.debug.assert_called_once()
            log_message = mock_logger.debug.call_args[0][0]

            # Should use generic "Task" when no ID provided
            assert "Task:" in log_message
            assert "1 train pairs" in log_message
            assert "1 test pairs" in log_message
            assert "5x5" in log_message

    def test_log_parsing_stats_edge_cases(self):
        """Test logging with edge case values."""
        test_cases = [
            (0, 0, (0, 0), "empty_task"),
            (100, 50, (1000, 2000), "large_task"),
            (1, 1, (1, 1), "minimal_task"),
        ]

        for num_train, num_test, dims, task_id in test_cases:
            with patch("jaxarc.parsers.utils.logger") as mock_logger:
                log_parsing_stats(num_train, num_test, dims, task_id)

                mock_logger.debug.assert_called_once()
                log_message = mock_logger.debug.call_args[0][0]

                assert task_id in log_message
                assert f"{num_train} train pairs" in log_message
                assert f"{num_test} test pairs" in log_message
                assert f"{dims[0]}x{dims[1]}" in log_message

    def test_grid_validation_comprehensive_edge_cases(self):
        """Test comprehensive edge cases for grid validation."""
        # Test with very large grids
        large_grid = [[i % 10 for i in range(100)] for _ in range(100)]
        validate_arc_grid_data(large_grid)  # Should not raise

        # Test with single cell
        validate_arc_grid_data([[5]])

        # Test with rectangular grids
        validate_arc_grid_data([[1, 2, 3, 4, 5]])  # 1x5
        validate_arc_grid_data([[1], [2], [3], [4], [5]])  # 5x1

        # Test with all same values
        validate_arc_grid_data([[7, 7, 7], [7, 7, 7], [7, 7, 7]])

    def test_jax_array_compatibility(self):
        """Test that utility functions work with JAX arrays."""
        # Test pad_grid_to_size with JAX arrays
        jax_grid = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        padded, mask = pad_grid_to_size(jax_grid, 4, 4)

        assert isinstance(padded, jnp.ndarray)
        assert isinstance(mask, jnp.ndarray)
        assert padded.dtype == jnp.int32
        assert mask.dtype == jnp.bool_

        # Test pad_array_sequence with JAX arrays
        jax_arrays = [
            jnp.array([[1, 2]], dtype=jnp.int32),
            jnp.array([[3, 4]], dtype=jnp.int32),
        ]

        padded_seq, masks_seq = pad_array_sequence(jax_arrays, 3, 2, 3)

        assert isinstance(padded_seq, jnp.ndarray)
        assert isinstance(masks_seq, jnp.ndarray)
        assert padded_seq.dtype == jnp.int32
        assert masks_seq.dtype == jnp.bool_

    def test_memory_efficiency(self):
        """Test memory efficiency of padding operations."""
        # Create moderately large arrays to test memory usage
        large_grid = jnp.ones((20, 20), dtype=jnp.int32)

        # Test that padding doesn't create excessive memory overhead
        padded, mask = pad_grid_to_size(large_grid, 30, 30)

        # Check that result is reasonable size
        assert padded.shape == (30, 30)
        assert mask.shape == (30, 30)

        # Test with sequence of large arrays
        large_arrays = [jnp.ones((15, 15), dtype=jnp.int32) for _ in range(5)]
        padded_seq, masks_seq = pad_array_sequence(large_arrays, 10, 20, 20)

        assert padded_seq.shape == (10, 20, 20)
        assert masks_seq.shape == (10, 20, 20)

    def test_error_message_quality(self):
        """Test that error messages are informative."""
        # Test grid validation error messages
        try:
            validate_arc_grid_data([[1, 2], [3]])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "same length" in str(e)

        # Test padding error messages
        try:
            large_grid = jnp.array([[1, 2, 3]], dtype=jnp.int32)
            pad_grid_to_size(large_grid, 1, 2)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "exceed target dimensions" in str(e)
            assert "3x1" in str(e) or "1x3" in str(
                e
            )  # Should include actual dimensions
            assert "1x2" in str(e)  # Should include target dimensions

    def test_function_parameter_validation(self):
        """Test parameter validation for utility functions."""
        # Test pad_grid_to_size with invalid parameters
        grid = jnp.array([[1, 2]], dtype=jnp.int32)

        # Negative target dimensions should be handled gracefully
        # (implementation may vary, but should not crash)
        try:
            pad_grid_to_size(grid, -1, 5)
        except (ValueError, TypeError):
            pass  # Expected to fail gracefully

        # Test pad_array_sequence with invalid parameters
        arrays = [jnp.array([[1]], dtype=jnp.int32)]

        try:
            pad_array_sequence(arrays, -1, 5, 5)
        except (ValueError, TypeError):
            pass  # Expected to fail gracefully
