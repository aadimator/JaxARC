"""
Tests for action handlers in JaxARC.

This module tests the specialized action handlers that convert different
action formats (point, bbox, mask) to standardized 30x30 boolean masks.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.envs.actions import (
    bbox_handler,
    create_test_action_data,
    get_action_handler,
    mask_handler,
    point_handler,
    validate_action_data,
)


class TestPointHandler:
    """Test the point action handler."""

    def test_basic_point_selection(self):
        """Test basic point selection functionality."""
        # Create test data
        action_data = jnp.array([5, 10])
        working_mask = jnp.ones((20, 20), dtype=jnp.bool_)

        # Apply handler
        result = point_handler(action_data, working_mask)

        # Verify result
        chex.assert_shape(result, (20, 20))
        chex.assert_type(result, jnp.bool_)
        assert result[5, 10] == True
        assert jnp.sum(result) == 1

    def test_coordinate_clipping(self):
        """Test that coordinates are clipped to valid range."""
        # Test coordinates outside valid range
        action_data = jnp.array([-5, 35])
        working_mask = jnp.ones((20, 15), dtype=jnp.bool_)

        result = point_handler(action_data, working_mask)

        # Should be clipped to (0, 14) for 20x15 grid
        assert result[0, 14] == True
        assert jnp.sum(result) == 1

    def test_working_grid_mask_constraint(self):
        """Test that selection is constrained by working grid mask."""
        # Create restricted working mask
        working_mask = jnp.zeros((20, 20), dtype=jnp.bool_)
        working_mask = working_mask.at[0:10, 0:10].set(True)

        # Point inside working area
        action_data = jnp.array([5, 5])
        result = point_handler(action_data, working_mask)
        assert result[5, 5] == True
        assert jnp.sum(result) == 1

        # Point outside working area
        action_data = jnp.array([15, 15])
        result = point_handler(action_data, working_mask)
        assert jnp.sum(result) == 0  # No selection due to mask constraint

    def test_float_coordinates(self):
        """Test handling of float coordinates."""
        action_data = jnp.array([5.7, 10.3])
        working_mask = jnp.ones((20, 20), dtype=jnp.bool_)

        result = point_handler(action_data, working_mask)

        # Should be converted to int (truncated)
        assert result[5, 10] == True
        assert jnp.sum(result) == 1

    def test_jit_compilation(self):
        """Test that handler works with JIT compilation."""

        @jax.jit
        def jitted_point_handler(action_data, working_mask):
            return point_handler(action_data, working_mask)

        action_data = jnp.array([3, 7])
        working_mask = jnp.ones((15, 15), dtype=jnp.bool_)

        result = jitted_point_handler(action_data, working_mask)

        assert result[3, 7] == True
        assert jnp.sum(result) == 1


class TestBboxHandler:
    """Test the bbox action handler."""

    def test_basic_bbox_selection(self):
        """Test basic bbox selection functionality."""
        # Create test data: bbox from (2,3) to (4,5)
        action_data = jnp.array([2, 3, 4, 5])
        working_mask = jnp.ones((20, 20), dtype=jnp.bool_)

        result = bbox_handler(action_data, working_mask)

        # Verify result
        chex.assert_shape(result, (20, 20))
        chex.assert_type(result, jnp.bool_)

        # Check that the bbox region is selected
        expected_count = (4 - 2 + 1) * (5 - 3 + 1)  # 3x3 = 9 cells
        assert jnp.sum(result) == expected_count

        # Check specific cells
        assert result[2, 3] == True  # Top-left
        assert result[4, 5] == True  # Bottom-right
        assert result[3, 4] == True  # Middle
        assert result[1, 2] == False  # Outside

    def test_coordinate_ordering(self):
        """Test that bbox coordinates are properly ordered."""
        # Reversed coordinates should work the same
        action_data = jnp.array([4, 5, 2, 3])  # Same bbox as above, reversed
        working_mask = jnp.ones((20, 20), dtype=jnp.bool_)

        result = bbox_handler(action_data, working_mask)

        expected_count = (4 - 2 + 1) * (5 - 3 + 1)  # 3x3 = 9 cells
        assert jnp.sum(result) == expected_count
        assert result[2, 3] == True
        assert result[4, 5] == True

    def test_coordinate_clipping(self):
        """Test that coordinates are clipped to valid range."""
        # Coordinates outside valid range
        action_data = jnp.array([-5, -10, 35, 40])
        working_mask = jnp.ones((15, 12), dtype=jnp.bool_)

        result = bbox_handler(action_data, working_mask)

        # Should be clipped to (0,0) to (14,11)
        expected_count = 15 * 12  # Full grid
        assert jnp.sum(result) == expected_count

    def test_single_cell_bbox(self):
        """Test bbox with same start and end coordinates."""
        action_data = jnp.array([5, 5, 5, 5])
        working_mask = jnp.ones((20, 20), dtype=jnp.bool_)

        result = bbox_handler(action_data, working_mask)

        # Should select single cell
        assert jnp.sum(result) == 1
        assert result[5, 5] == True

    def test_working_grid_mask_constraint(self):
        """Test that selection is constrained by working grid mask."""
        # Create restricted working mask
        working_mask = jnp.zeros((20, 20), dtype=jnp.bool_)
        working_mask = working_mask.at[0:10, 0:10].set(True)

        # Bbox partially outside working area
        action_data = jnp.array([5, 5, 15, 15])
        result = bbox_handler(action_data, working_mask)

        # Only cells within working area should be selected
        expected_count = (10 - 5) * (10 - 5)  # 5x5 = 25 cells
        assert jnp.sum(result) == expected_count

    def test_jit_compilation(self):
        """Test that handler works with JIT compilation."""

        @jax.jit
        def jitted_bbox_handler(action_data, working_mask):
            return bbox_handler(action_data, working_mask)

        action_data = jnp.array([1, 2, 3, 4])
        working_mask = jnp.ones((15, 15), dtype=jnp.bool_)

        result = jitted_bbox_handler(action_data, working_mask)

        expected_count = (3 - 1 + 1) * (4 - 2 + 1)  # 3x3 = 9 cells
        assert jnp.sum(result) == expected_count


class TestMaskHandler:
    """Test the mask action handler."""

    def test_basic_mask_passthrough(self):
        """Test basic mask passthrough functionality."""
        # Create test mask
        mask = jnp.zeros((15, 12), dtype=jnp.bool_)
        mask = mask.at[5:8, 8:11].set(True)
        action_data = mask.flatten()
        working_mask = jnp.ones((15, 12), dtype=jnp.bool_)

        result = mask_handler(action_data, working_mask)

        # Verify result
        chex.assert_shape(result, (15, 12))
        chex.assert_type(result, jnp.bool_)

        # Should be identical to input mask
        assert jnp.array_equal(result, mask)

    def test_working_grid_mask_constraint(self):
        """Test that selection is constrained by working grid mask."""
        # Create test mask
        mask = jnp.ones((20, 20), dtype=jnp.bool_)
        action_data = mask.flatten()

        # Create restricted working mask
        working_mask = jnp.zeros((20, 20), dtype=jnp.bool_)
        working_mask = working_mask.at[0:10, 0:10].set(True)

        result = mask_handler(action_data, working_mask)

        # Only cells within working area should be selected
        expected_count = 10 * 10  # 100 cells
        assert jnp.sum(result) == expected_count

    def test_oversized_action_data(self):
        """Test handling of action data larger than grid size."""
        # Create action data with more elements than grid size
        grid_size = 15 * 12  # 180 elements
        action_data = jnp.ones(300, dtype=jnp.float32)
        working_mask = jnp.ones((15, 12), dtype=jnp.bool_)

        result = mask_handler(action_data, working_mask)

        # Should use only first grid_size elements
        chex.assert_shape(result, (15, 12))
        assert jnp.sum(result) == grid_size  # All cells selected

    def test_type_conversion(self):
        """Test conversion of different numeric types to boolean."""
        # Test with float data
        grid_size = 12 * 8  # 96 elements
        action_data = jnp.array(
            [0.0, 1.0, 0.5, -1.0] * (grid_size // 4)
        )  # grid_size elements
        working_mask = jnp.ones((12, 8), dtype=jnp.bool_)

        result = mask_handler(action_data, working_mask)

        # Should convert to boolean (0.0 -> False, everything else -> True)
        chex.assert_type(result, jnp.bool_)
        assert jnp.sum(result) == grid_size * 3 // 4  # 3/4 of cells are truthy

    def test_jit_compilation(self):
        """Test that handler works with JIT compilation."""

        @jax.jit
        def jitted_mask_handler(action_data, working_mask):
            return mask_handler(action_data, working_mask)

        mask = jnp.zeros((10, 10), dtype=jnp.bool_)
        mask = mask.at[0:5, 0:5].set(True)
        action_data = mask.flatten()
        working_mask = jnp.ones((10, 10), dtype=jnp.bool_)

        result = jitted_mask_handler(action_data, working_mask)

        assert jnp.sum(result) == 25  # 5x5 = 25 cells


class TestActionHandlerFactory:
    """Test the action handler factory function."""

    def test_point_handler_selection(self):
        """Test getting point handler."""
        handler = get_action_handler("point")
        assert handler == point_handler

    def test_bbox_handler_selection(self):
        """Test getting bbox handler."""
        handler = get_action_handler("bbox")
        assert handler == bbox_handler

    def test_mask_handler_selection(self):
        """Test getting mask handler."""
        handler = get_action_handler("mask")
        assert handler == mask_handler

    def test_unknown_format_error(self):
        """Test error for unknown action format."""
        with pytest.raises(ValueError, match="Unknown selection format"):
            get_action_handler("unknown_format")

    def test_handler_jit_compatibility(self):
        """Test that all handlers work with JIT compilation."""
        formats = ["point", "bbox", "mask"]

        for format_name in formats:
            handler = get_action_handler(format_name)

            # Create appropriate test data
            grid_shape = (12, 15)
            if format_name == "point":
                action_data = jnp.array([5, 10])
            elif format_name == "bbox":
                action_data = jnp.array([2, 3, 4, 5])
            else:  # mask formats
                mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
                mask = mask.at[5:8, 10:13].set(True)
                action_data = mask.flatten()

            working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)

            # JIT compile and test
            jitted_handler = jax.jit(handler)
            result = jitted_handler(action_data, working_mask)

            chex.assert_shape(result, grid_shape)
            chex.assert_type(result, jnp.bool_)


class TestValidationFunctions:
    """Test utility functions for validation."""

    def test_validate_point_action_data(self):
        """Test validation of point action data."""
        # Valid data
        action_data = jnp.array([5, 10])
        validate_action_data(action_data, "point")  # Should not raise

        # Invalid data (too few elements)
        action_data = jnp.array([5])
        with pytest.raises(
            ValueError, match="Point action requires at least 2 elements"
        ):
            validate_action_data(action_data, "point")

    def test_validate_bbox_action_data(self):
        """Test validation of bbox action data."""
        # Valid data
        action_data = jnp.array([1, 2, 3, 4])
        validate_action_data(action_data, "bbox")  # Should not raise

        # Invalid data (too few elements)
        action_data = jnp.array([1, 2, 3])
        with pytest.raises(
            ValueError, match="Bbox action requires at least 4 elements"
        ):
            validate_action_data(action_data, "bbox")

    def test_validate_mask_action_data(self):
        """Test validation of mask action data."""
        # Valid data with grid shape
        grid_shape = (20, 15)
        action_data = jnp.ones(300)  # 20*15 = 300
        validate_action_data(action_data, "mask", grid_shape)  # Should not raise

        # Invalid data (too few elements)
        action_data = jnp.ones(200)
        with pytest.raises(
            ValueError, match="Mask action requires at least 300 elements"
        ):
            validate_action_data(action_data, "mask", grid_shape)

    def test_validate_unknown_format(self):
        """Test validation with unknown format."""
        action_data = jnp.array([1, 2])
        with pytest.raises(ValueError, match="Unknown selection format"):
            validate_action_data(action_data, "unknown")


class TestCreateTestActionData:
    """Test utility function for creating test action data."""

    def test_create_point_data(self):
        """Test creating point action data."""
        data = create_test_action_data("point", row=5, col=10)
        expected = jnp.array([5, 10])
        assert jnp.array_equal(data, expected)

        # Test default values
        data = create_test_action_data("point")
        expected = jnp.array([5, 10])  # Default values
        assert jnp.array_equal(data, expected)

    def test_create_bbox_data(self):
        """Test creating bbox action data."""
        data = create_test_action_data("bbox", r1=2, c1=3, r2=4, c2=5)
        expected = jnp.array([2, 3, 4, 5])
        assert jnp.array_equal(data, expected)

    def test_create_mask_data(self):
        """Test creating mask action data."""
        grid_shape = (15, 12)
        data = create_test_action_data(
            "mask", grid_shape=grid_shape, start_row=3, start_col=4, size=2
        )

        # Reshape and check
        mask = data.reshape(grid_shape)
        assert jnp.sum(mask) == 4  # 2x2 = 4 cells
        assert mask[3, 4] == True
        assert mask[4, 5] == True

    def test_create_data_unknown_format(self):
        """Test error for unknown format."""
        with pytest.raises(ValueError, match="Unknown selection format"):
            create_test_action_data("unknown")


class TestIntegration:
    """Integration tests for action handlers."""

    def test_all_handlers_produce_consistent_shapes(self):
        """Test that all handlers produce consistent output shapes."""
        grid_shape = (18, 12)
        working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)

        # Test point handler
        point_data = jnp.array([5, 10])
        point_result = point_handler(point_data, working_mask)

        # Test bbox handler
        bbox_data = jnp.array([2, 3, 4, 5])
        bbox_result = bbox_handler(bbox_data, working_mask)

        # Test mask handler
        mask_data = jnp.ones(grid_shape, dtype=jnp.bool_).flatten()
        mask_result = mask_handler(mask_data, working_mask)

        # All should have same shape and type
        for result in [point_result, bbox_result, mask_result]:
            chex.assert_shape(result, grid_shape)
            chex.assert_type(result, jnp.bool_)

    def test_working_grid_mask_consistency(self):
        """Test that all handlers respect working grid mask consistently."""
        # Create restricted working mask
        grid_shape = (25, 20)
        working_mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
        working_mask = working_mask.at[5:15, 5:15].set(True)

        # Test point handler with point outside working area
        point_data = jnp.array([20, 18])
        point_result = point_handler(point_data, working_mask)
        assert jnp.sum(point_result) == 0

        # Test bbox handler with bbox outside working area
        bbox_data = jnp.array([20, 18, 22, 19])
        bbox_result = bbox_handler(bbox_data, working_mask)
        assert jnp.sum(bbox_result) == 0

        # Test mask handler with mask outside working area
        full_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        mask_result = mask_handler(full_mask.flatten(), working_mask)
        assert jnp.sum(mask_result) == 100  # Only working area (10x10)

    def test_vmap_compatibility(self):
        """Test that handlers work with vmap for batch processing."""
        grid_shape = (15, 12)
        working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)

        # Test point handler with batch
        point_batch = jnp.array([[1, 2], [3, 4], [5, 6]])
        batched_point_handler = jax.vmap(point_handler, in_axes=(0, None))
        results = batched_point_handler(point_batch, working_mask)

        chex.assert_shape(results, (3, *grid_shape))
        assert jnp.sum(results[0]) == 1  # Each result should have 1 cell selected
        assert jnp.sum(results[1]) == 1
        assert jnp.sum(results[2]) == 1
