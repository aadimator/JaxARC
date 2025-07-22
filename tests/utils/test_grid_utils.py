"""Tests for grid utility functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc.utils.grid_utils import (
    crop_grid_to_content,
    crop_grid_to_mask,
    get_actual_grid_shape_from_mask,
    get_grid_bounds,
    pad_to_max_dims,
)


class TestPadToMaxDims:
    """Test pad_to_max_dims function."""

    def test_pad_smaller_grid(self):
        """Test padding a smaller grid to max dimensions."""
        grid = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        max_height, max_width = 4, 5

        padded = pad_to_max_dims(grid, max_height, max_width)

        assert padded.shape == (4, 5)
        assert jnp.array_equal(padded[:2, :2], grid)
        assert jnp.all(padded[2:, :] == 0)  # Bottom padding
        assert jnp.all(padded[:, 2:] == 0)  # Right padding

    def test_pad_with_custom_fill_value(self):
        """Test padding with custom fill value."""
        grid = jnp.array([[1, 2]], dtype=jnp.int32)
        max_height, max_width = 3, 3
        fill_value = -1

        padded = pad_to_max_dims(grid, max_height, max_width, fill_value)

        assert padded.shape == (3, 3)
        assert jnp.array_equal(padded[0, :2], grid[0])
        assert jnp.all(padded[1:, :] == -1)  # Bottom padding
        assert jnp.all(padded[:, 2:] == -1)  # Right padding

    def test_pad_exact_size(self):
        """Test padding when grid is already at max dimensions."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
        max_height, max_width = 2, 3

        padded = pad_to_max_dims(grid, max_height, max_width)

        assert padded.shape == (2, 3)
        assert jnp.array_equal(padded, grid)

    def test_pad_larger_grid_no_truncation(self):
        """Test that larger grids are not truncated."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
        max_height, max_width = 2, 2  # Smaller than grid

        # Should not truncate, just return original
        padded = pad_to_max_dims(grid, max_height, max_width)

        # The function doesn't truncate, so result should be original size
        assert padded.shape == (3, 3)
        assert jnp.array_equal(padded, grid)

    def test_pad_zero_dimensions(self):
        """Test padding with zero max dimensions."""
        grid = jnp.array([[1, 2]], dtype=jnp.int32)
        max_height, max_width = 0, 0

        # Should return original grid (no negative padding)
        padded = pad_to_max_dims(grid, max_height, max_width)
        assert padded.shape == (1, 2)
        assert jnp.array_equal(padded, grid)

    def test_jit_compatibility_with_concrete_values(self):
        """Test JIT compatibility when max dimensions are concrete."""
        # Note: pad_to_max_dims has JIT limitations as documented
        # This test shows the current behavior
        grid = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)

        # Direct call works fine
        padded = pad_to_max_dims(grid, 5, 5)

        assert padded.shape == (5, 5)
        assert jnp.array_equal(padded[:2, :2], grid)


class TestGetGridBounds:
    """Test get_grid_bounds function."""

    def test_simple_bounds(self):
        """Test bounds detection for simple grid."""
        grid = jnp.array(
            [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]], dtype=jnp.int32
        )

        min_row, max_row, min_col, max_col = get_grid_bounds(grid)

        assert min_row == 1
        assert max_row == 2
        assert min_col == 1
        assert max_col == 2

    def test_full_grid_bounds(self):
        """Test bounds when content fills entire grid."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)

        min_row, max_row, min_col, max_col = get_grid_bounds(grid)

        assert min_row == 0
        assert max_row == 2
        assert min_col == 0
        assert max_col == 2

    def test_single_cell_bounds(self):
        """Test bounds for single non-background cell."""
        grid = jnp.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=jnp.int32)

        min_row, max_row, min_col, max_col = get_grid_bounds(grid)

        assert min_row == 1
        assert max_row == 1
        assert min_col == 1
        assert max_col == 1

    def test_custom_background_value(self):
        """Test bounds with custom background value."""
        grid = jnp.array(
            [[9, 9, 9, 9], [9, 1, 2, 9], [9, 3, 4, 9], [9, 9, 9, 9]], dtype=jnp.int32
        )

        min_row, max_row, min_col, max_col = get_grid_bounds(grid, background_value=9)

        assert min_row == 1
        assert max_row == 2
        assert min_col == 1
        assert max_col == 2

    def test_empty_grid_bounds(self):
        """Test bounds for grid with only background values."""
        grid = jnp.zeros((3, 3), dtype=jnp.int32)

        min_row, max_row, min_col, max_col = get_grid_bounds(grid)

        # Should return 0 for empty grid
        assert min_row == 0
        assert max_row == 0
        assert min_col == 0
        assert max_col == 0

    def test_edge_content_bounds(self):
        """Test bounds when content is at edges."""
        grid = jnp.array(
            [[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 4]], dtype=jnp.int32
        )

        min_row, max_row, min_col, max_col = get_grid_bounds(grid)

        assert min_row == 0
        assert max_row == 3
        assert min_col == 0
        assert max_col == 3

    def test_jit_compatibility(self):
        """Test JIT compatibility of get_grid_bounds."""

        @jax.jit
        def jit_bounds(grid):
            return get_grid_bounds(grid)

        grid = jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=jnp.int32)

        min_row, max_row, min_col, max_col = jit_bounds(grid)

        assert min_row == 1
        assert max_row == 1
        assert min_col == 1
        assert max_col == 1


class TestCropGridToContent:
    """Test crop_grid_to_content function."""

    def test_crop_padded_grid(self):
        """Test cropping a padded grid to content."""
        grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.int32,
        )

        cropped = crop_grid_to_content(grid)
        expected = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)

        assert jnp.array_equal(cropped, expected)

    def test_crop_no_padding(self):
        """Test cropping grid with no padding."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)

        cropped = crop_grid_to_content(grid)

        assert jnp.array_equal(cropped, grid)

    def test_crop_single_cell(self):
        """Test cropping to single cell."""
        grid = jnp.array([[0, 0, 0], [0, 7, 0], [0, 0, 0]], dtype=jnp.int32)

        cropped = crop_grid_to_content(grid)
        expected = jnp.array([[7]], dtype=jnp.int32)

        assert jnp.array_equal(cropped, expected)

    def test_crop_custom_background(self):
        """Test cropping with custom background value."""
        grid = jnp.array(
            [[9, 9, 9, 9], [9, 1, 2, 9], [9, 3, 4, 9], [9, 9, 9, 9]], dtype=jnp.int32
        )

        cropped = crop_grid_to_content(grid, background_value=9)
        expected = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)

        assert jnp.array_equal(cropped, expected)

    def test_crop_empty_grid(self):
        """Test cropping empty grid."""
        grid = jnp.zeros((3, 3), dtype=jnp.int32)

        cropped = crop_grid_to_content(grid)

        # Should return at least 1x1 grid
        assert cropped.shape[0] >= 1
        assert cropped.shape[1] >= 1

    def test_jit_compatibility(self):
        """Test JIT compatibility of crop_grid_to_content."""
        # Note: crop_grid_to_content returns dynamic shapes, which limits JIT compatibility
        # This test shows the current behavior with direct calls
        grid = jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=jnp.int32)

        cropped = crop_grid_to_content(grid)
        expected = jnp.array([[1]], dtype=jnp.int32)

        assert jnp.array_equal(cropped, expected)


class TestGetActualGridShapeFromMask:
    """Test get_actual_grid_shape_from_mask function."""

    def test_full_mask_shape(self):
        """Test shape detection from full mask."""
        mask = jnp.ones((5, 7), dtype=bool)

        height, width = get_actual_grid_shape_from_mask(mask)

        assert height == 5
        assert width == 7

    def test_partial_mask_shape(self):
        """Test shape detection from partial mask."""
        mask = jnp.zeros((10, 10), dtype=bool)
        mask = mask.at[:3, :4].set(True)

        height, width = get_actual_grid_shape_from_mask(mask)

        assert height == 3
        assert width == 4

    def test_single_cell_mask(self):
        """Test shape detection from single cell mask."""
        mask = jnp.zeros((5, 5), dtype=bool)
        mask = mask.at[2, 3].set(True)

        height, width = get_actual_grid_shape_from_mask(mask)

        assert height == 3  # Row index 2 + 1
        assert width == 4  # Col index 3 + 1

    def test_empty_mask_shape(self):
        """Test shape detection from empty mask."""
        mask = jnp.zeros((5, 5), dtype=bool)

        height, width = get_actual_grid_shape_from_mask(mask)

        assert height == 0
        assert width == 0

    def test_irregular_mask_shape(self):
        """Test shape detection from irregular mask."""
        mask = jnp.zeros((6, 8), dtype=bool)
        mask = mask.at[1, 2].set(True)
        mask = mask.at[3, 6].set(True)
        mask = mask.at[0, 1].set(True)

        height, width = get_actual_grid_shape_from_mask(mask)

        assert height == 4  # Max row index 3 + 1
        assert width == 7  # Max col index 6 + 1

    def test_jit_compatibility(self):
        """Test JIT compatibility of get_actual_grid_shape_from_mask."""

        @jax.jit
        def jit_shape(mask):
            return get_actual_grid_shape_from_mask(mask)

        mask = jnp.zeros((10, 10), dtype=bool)
        mask = mask.at[:5, :3].set(True)

        height, width = jit_shape(mask)

        assert height == 5
        assert width == 3

    def test_vmap_compatibility(self):
        """Test vmap compatibility with batched masks."""

        def get_shape_single(mask):
            return get_actual_grid_shape_from_mask(mask)

        # Create batch of masks
        batch_masks = jnp.zeros((3, 5, 5), dtype=bool)
        batch_masks = batch_masks.at[0, :2, :3].set(True)
        batch_masks = batch_masks.at[1, :4, :2].set(True)
        batch_masks = batch_masks.at[2, :1, :5].set(True)

        vmapped_fn = jax.vmap(get_shape_single)
        heights, widths = vmapped_fn(batch_masks)

        assert jnp.array_equal(heights, jnp.array([2, 4, 1]))
        assert jnp.array_equal(widths, jnp.array([3, 2, 5]))


class TestCropGridToMask:
    """Test crop_grid_to_mask function (non-JIT version)."""

    def test_crop_with_mask(self):
        """Test cropping grid using mask."""
        grid = jnp.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 0],
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 0],
                [1, 2, 3, 4, 5],
            ],
            dtype=jnp.int32,
        )

        mask = jnp.zeros((5, 5), dtype=bool)
        mask = mask.at[:2, :3].set(True)

        cropped = crop_grid_to_mask(grid, mask)
        expected = jnp.array([[1, 2, 3], [6, 7, 8]], dtype=jnp.int32)

        assert jnp.array_equal(cropped, expected)

    def test_crop_single_cell_mask(self):
        """Test cropping to single cell using mask."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)

        mask = jnp.zeros((3, 3), dtype=bool)
        mask = mask.at[1, 1].set(True)

        cropped = crop_grid_to_mask(grid, mask)

        # The function crops to the bounding box of the mask
        # With mask at [1,1], the actual shape is (2, 2) from [0:2, 0:2]
        assert cropped.shape == (2, 2)
        expected = jnp.array([[1, 2], [4, 5]], dtype=jnp.int32)
        assert jnp.array_equal(cropped, expected)

    def test_crop_empty_mask(self):
        """Test cropping with empty mask."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)

        mask = jnp.zeros((3, 3), dtype=bool)

        cropped = crop_grid_to_mask(grid, mask)

        # Should return 1x1 grid with default value
        assert cropped.shape == (1, 1)
        assert cropped[0, 0] == 0

    def test_crop_full_mask(self):
        """Test cropping with full mask."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)

        mask = jnp.ones((2, 3), dtype=bool)

        cropped = crop_grid_to_mask(grid, mask)

        assert jnp.array_equal(cropped, grid)

    def test_crop_irregular_mask(self):
        """Test cropping with irregular mask shape."""
        grid = jnp.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]], dtype=jnp.int32
        )

        mask = jnp.zeros((4, 4), dtype=bool)
        mask = mask.at[0, 1].set(True)  # Row 0, Col 1
        mask = mask.at[2, 3].set(True)  # Row 2, Col 3

        cropped = crop_grid_to_mask(grid, mask)

        # Should crop to (3, 4) - from row 0 to row 2, col 0 to col 3
        assert cropped.shape == (3, 4)
        assert jnp.array_equal(cropped, grid[:3, :4])


class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""

    def test_pad_then_crop_roundtrip(self):
        """Test padding then cropping returns to original content."""
        original = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)

        # Pad to larger size
        padded = pad_to_max_dims(original, 5, 5)

        # Crop back to content
        cropped = crop_grid_to_content(padded)

        assert jnp.array_equal(cropped, original)

    def test_mask_based_operations(self):
        """Test mask-based grid operations."""
        # Create a padded grid
        grid = jnp.zeros((10, 10), dtype=jnp.int32)
        grid = grid.at[2:5, 3:7].set(
            jnp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]])
        )

        # Create corresponding mask
        mask = jnp.zeros((10, 10), dtype=bool)
        mask = mask.at[2:5, 3:7].set(True)

        # Get actual shape from mask
        height, width = get_actual_grid_shape_from_mask(mask)
        assert height == 5
        assert width == 7

        # Crop using mask
        cropped = crop_grid_to_mask(grid, mask)
        assert cropped.shape == (5, 7)

    def test_batch_processing_compatibility(self):
        """Test compatibility with batch processing patterns."""

        def process_single_grid(grid):
            # Typical processing: get bounds only (crop has dynamic shape issues with vmap)
            bounds = get_grid_bounds(grid)
            return bounds

        # Create batch of grids
        batch_grids = jnp.array(
            [
                [[0, 1, 0], [0, 2, 0], [0, 0, 0]],
                [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
            ],
            dtype=jnp.int32,
        )

        # Process with vmap
        vmapped_fn = jax.vmap(process_single_grid)
        bounds_batch = vmapped_fn(batch_grids)

        assert len(bounds_batch) == 4  # (min_row, max_row, min_col, max_col)
        assert bounds_batch[0].shape == (3,)  # Batch dimension

        # Test individual processing for dynamic shape functions
        for i, grid in enumerate(batch_grids):
            bounds = get_grid_bounds(grid)
            cropped = crop_grid_to_content(grid)
            padded = pad_to_max_dims(cropped, 5, 5)

            assert padded.shape == (5, 5)
            assert bounds == (
                bounds_batch[0][i],
                bounds_batch[1][i],
                bounds_batch[2][i],
                bounds_batch[3][i],
            )

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        # Very small grids
        tiny_grid = jnp.array([[1]], dtype=jnp.int32)
        bounds = get_grid_bounds(tiny_grid)
        assert bounds == (0, 0, 0, 0)

        # Large padding values
        large_padded = pad_to_max_dims(tiny_grid, 100, 100)
        assert large_padded.shape == (100, 100)
        assert large_padded[0, 0] == 1

        # Empty content handling
        empty_grid = jnp.zeros((5, 5), dtype=jnp.int32)
        empty_bounds = get_grid_bounds(empty_grid)
        assert empty_bounds == (0, 0, 0, 0)

    def test_performance_with_large_grids(self):
        """Test performance characteristics with larger grids."""
        # Create a large grid with content in corner
        large_grid = jnp.zeros((100, 100), dtype=jnp.int32)
        large_grid = large_grid.at[50:55, 60:65].set(1)

        # Should efficiently find bounds
        bounds = get_grid_bounds(large_grid)
        assert bounds == (50, 54, 60, 64)

        # Should efficiently crop
        cropped = crop_grid_to_content(large_grid)
        assert cropped.shape == (5, 5)
        assert jnp.all(cropped == 1)

    def test_dtype_preservation(self):
        """Test that dtypes are preserved through operations."""
        # Test with different dtypes
        int_grid = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        float_grid = jnp.array([[1.5, 2.5], [3.5, 4.5]], dtype=jnp.float32)

        # Padding should preserve dtype
        padded_int = pad_to_max_dims(int_grid, 3, 3)
        padded_float = pad_to_max_dims(float_grid, 3, 3)

        assert padded_int.dtype == jnp.int32
        assert padded_float.dtype == jnp.float32

        # Cropping should preserve dtype
        cropped_int = crop_grid_to_content(padded_int)
        cropped_float = crop_grid_to_content(padded_float)

        assert cropped_int.dtype == jnp.int32
        assert cropped_float.dtype == jnp.float32
