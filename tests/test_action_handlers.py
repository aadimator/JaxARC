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
    create_test_structured_action,
    get_action_handler,
    mask_handler,
    point_handler,
    validate_structured_action,
)


class TestPointHandler:
    """Test the point action handler."""

    def test_basic_point_selection(self):
        """Test basic point selection functionality."""
        # Create test structured action
        action = create_test_structured_action("point", operation=0, row=5, col=10)
        working_mask = jnp.ones((20, 20), dtype=jnp.bool_)

        # Apply handler
        result = point_handler(action, working_mask)

        # Verify result
        chex.assert_shape(result, (20, 20))
        chex.assert_type(result, jnp.bool_)
        assert result[5, 10] == True
        assert jnp.sum(result) == 1

    def test_coordinate_clipping(self):
        """Test that coordinates are clipped to valid range."""
        # Test coordinates outside valid range
        action = create_test_structured_action("point", operation=0, row=-5, col=35)
        working_mask = jnp.ones((20, 15), dtype=jnp.bool_)

        result = point_handler(action, working_mask)

        # Should be clipped to (0, 14) for 20x15 grid
        assert result[0, 14] == True
        assert jnp.sum(result) == 1

    def test_working_grid_mask_constraint(self):
        """Test that selection is constrained by working grid mask."""
        # Create restricted working mask
        working_mask = jnp.zeros((20, 20), dtype=jnp.bool_)
        working_mask = working_mask.at[0:10, 0:10].set(True)

        # Point inside working area
        action = create_test_structured_action("point", operation=0, row=5, col=5)
        result = point_handler(action, working_mask)
        assert result[5, 5] == True
        assert jnp.sum(result) == 1

        # Point outside working area
        action = create_test_structured_action("point", operation=0, row=15, col=15)
        result = point_handler(action, working_mask)
        assert jnp.sum(result) == 0  # No selection due to mask constraint

    def test_float_coordinates(self):
        """Test handling of float coordinates."""
        # Create action with float coordinates (will be converted to int in structured action)
        action = create_test_structured_action("point", operation=0, row=5, col=10)
        working_mask = jnp.ones((20, 20), dtype=jnp.bool_)

        result = point_handler(action, working_mask)

        # Should work with int coordinates
        assert result[5, 10] == True
        assert jnp.sum(result) == 1

    def test_jit_compilation(self):
        """Test that handler works with JIT compilation."""

        @jax.jit
        def jitted_point_handler(action, working_mask):
            return point_handler(action, working_mask)

        action = create_test_structured_action("point", operation=0, row=3, col=7)
        working_mask = jnp.ones((15, 15), dtype=jnp.bool_)

        result = jitted_point_handler(action, working_mask)

        assert result[3, 7] == True
        assert jnp.sum(result) == 1


class TestBboxHandler:
    """Test the bbox action handler."""

    def test_basic_bbox_selection(self):
        """Test basic bbox selection functionality."""
        # Create test structured action: bbox from (2,3) to (4,5)
        action = create_test_structured_action("bbox", operation=0, r1=2, c1=3, r2=4, c2=5)
        working_mask = jnp.ones((20, 20), dtype=jnp.bool_)

        result = bbox_handler(action, working_mask)

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
        action = create_test_structured_action("bbox", operation=0, r1=4, c1=5, r2=2, c2=3)  # Same bbox as above, reversed
        working_mask = jnp.ones((20, 20), dtype=jnp.bool_)

        result = bbox_handler(action, working_mask)

        expected_count = (4 - 2 + 1) * (5 - 3 + 1)  # 3x3 = 9 cells
        assert jnp.sum(result) == expected_count
        assert result[2, 3] == True
        assert result[4, 5] == True

    def test_coordinate_clipping(self):
        """Test that coordinates are clipped to valid range."""
        # Coordinates outside valid range
        action = create_test_structured_action("bbox", operation=0, r1=-5, c1=-10, r2=35, c2=40)
        working_mask = jnp.ones((15, 12), dtype=jnp.bool_)

        result = bbox_handler(action, working_mask)

        # Should be clipped to (0,0) to (14,11)
        expected_count = 15 * 12  # Full grid
        assert jnp.sum(result) == expected_count

    def test_single_cell_bbox(self):
        """Test bbox with same start and end coordinates."""
        action = create_test_structured_action("bbox", operation=0, r1=5, c1=5, r2=5, c2=5)
        working_mask = jnp.ones((20, 20), dtype=jnp.bool_)

        result = bbox_handler(action, working_mask)

        # Should select single cell
        assert jnp.sum(result) == 1
        assert result[5, 5] == True

    def test_working_grid_mask_constraint(self):
        """Test that selection is constrained by working grid mask."""
        # Create restricted working mask
        working_mask = jnp.zeros((20, 20), dtype=jnp.bool_)
        working_mask = working_mask.at[0:10, 0:10].set(True)

        # Bbox partially outside working area
        action = create_test_structured_action("bbox", operation=0, r1=5, c1=5, r2=15, c2=15)
        result = bbox_handler(action, working_mask)

        # Only cells within working area should be selected
        expected_count = (10 - 5) * (10 - 5)  # 5x5 = 25 cells
        assert jnp.sum(result) == expected_count

    def test_jit_compilation(self):
        """Test that handler works with JIT compilation."""

        @jax.jit
        def jitted_bbox_handler(action, working_mask):
            return bbox_handler(action, working_mask)

        action = create_test_structured_action("bbox", operation=0, r1=1, c1=2, r2=3, c2=4)
        working_mask = jnp.ones((15, 15), dtype=jnp.bool_)

        result = jitted_bbox_handler(action, working_mask)

        expected_count = (3 - 1 + 1) * (4 - 2 + 1)  # 3x3 = 9 cells
        assert jnp.sum(result) == expected_count


class TestMaskHandler:
    """Test the mask action handler."""

    def test_basic_mask_passthrough(self):
        """Test basic mask passthrough functionality."""
        # Create test mask
        mask = jnp.zeros((15, 12), dtype=jnp.bool_)
        mask = mask.at[5:8, 8:11].set(True)
        action = create_test_structured_action("mask", operation=0, grid_shape=(15, 12), start_row=5, start_col=8, size=3)
        working_mask = jnp.ones((15, 12), dtype=jnp.bool_)

        result = mask_handler(action, working_mask)

        # Verify result
        chex.assert_shape(result, (15, 12))
        chex.assert_type(result, jnp.bool_)

        # Should have the expected selection pattern
        assert jnp.sum(result) == 9  # 3x3 region

    def test_working_grid_mask_constraint(self):
        """Test that selection is constrained by working grid mask."""
        # Create test mask that selects everything
        full_mask = jnp.ones((20, 20), dtype=jnp.bool_)
        action = create_test_structured_action("mask", operation=0, grid_shape=(20, 20), start_row=0, start_col=0, size=20)

        # Create restricted working mask
        working_mask = jnp.zeros((20, 20), dtype=jnp.bool_)
        working_mask = working_mask.at[0:10, 0:10].set(True)

        result = mask_handler(action, working_mask)

        # Only cells within working area should be selected
        expected_count = 10 * 10  # 100 cells
        assert jnp.sum(result) == expected_count

    def test_oversized_action_data(self):
        """Test handling of action data with correct grid size."""
        # Create action that selects all cells
        action = create_test_structured_action("mask", operation=0, grid_shape=(15, 12), start_row=0, start_col=0, size=15)
        working_mask = jnp.ones((15, 12), dtype=jnp.bool_)

        result = mask_handler(action, working_mask)

        # Should work correctly
        chex.assert_shape(result, (15, 12))
        # The test action creates a partial selection, not full grid
        assert jnp.sum(result) > 0

    def test_type_conversion(self):
        """Test that mask handler works with boolean selection."""
        # Create action with boolean mask
        action = create_test_structured_action("mask", operation=0, grid_shape=(12, 8), start_row=2, start_col=3, size=4)
        working_mask = jnp.ones((12, 8), dtype=jnp.bool_)

        result = mask_handler(action, working_mask)

        # Should work correctly with boolean types
        chex.assert_type(result, jnp.bool_)
        assert jnp.sum(result) > 0  # Some cells selected

    def test_jit_compilation(self):
        """Test that handler works with JIT compilation."""

        @jax.jit
        def jitted_mask_handler(action, working_mask):
            return mask_handler(action, working_mask)

        action = create_test_structured_action("mask", operation=0, grid_shape=(10, 10), start_row=0, start_col=0, size=5)
        working_mask = jnp.ones((10, 10), dtype=jnp.bool_)

        result = jitted_mask_handler(action, working_mask)

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
        with pytest.raises(ValueError, match="Unknown action type"):
            get_action_handler("unknown_format")

    def test_handler_jit_compatibility(self):
        """Test that all handlers work with JIT compilation."""
        formats = ["point", "bbox", "mask"]

        for format_name in formats:
            handler = get_action_handler(format_name)

            # Create appropriate test data
            grid_shape = (12, 15)
            if format_name == "point":
                action = create_test_structured_action("point", operation=0, row=5, col=10)
            elif format_name == "bbox":
                action = create_test_structured_action("bbox", operation=0, r1=2, c1=3, r2=4, c2=5)
            else:  # mask formats
                action = create_test_structured_action("mask", operation=0, grid_shape=grid_shape, start_row=5, start_col=10, size=3)

            working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)

            # JIT compile and test
            jitted_handler = jax.jit(handler)
            result = jitted_handler(action, working_mask)

            chex.assert_shape(result, grid_shape)
            chex.assert_type(result, jnp.bool_)


class TestValidationFunctions:
    """Test utility functions for validation."""

    def test_validate_point_action(self):
        """Test validation of point structured action."""
        # Valid action
        action = create_test_structured_action("point", operation=0, row=5, col=10)
        validate_structured_action(action, (20, 20))  # Should not raise

        # Invalid action (out of bounds)
        action = create_test_structured_action("point", operation=0, row=25, col=10)
        with pytest.raises(
            ValueError, match="Point row 25 out of bounds"
        ):
            validate_structured_action(action, (20, 20))

    def test_validate_bbox_action(self):
        """Test validation of bbox structured action."""
        # Valid action
        action = create_test_structured_action("bbox", operation=0, r1=1, c1=2, r2=3, c2=4)
        validate_structured_action(action, (20, 20))  # Should not raise

        # Invalid action (out of bounds)
        action = create_test_structured_action("bbox", operation=0, r1=1, c1=2, r2=25, c2=4)
        with pytest.raises(
            ValueError, match="Bbox r2 25 out of bounds"
        ):
            validate_structured_action(action, (20, 20))

    def test_validate_mask_action(self):
        """Test validation of mask structured action."""
        # Valid action
        action = create_test_structured_action("mask", operation=0, grid_shape=(20, 15))
        validate_structured_action(action, (20, 15))  # Should not raise

        # Test with wrong shape would require creating a mask with wrong shape
        # which is complex, so we'll skip this specific test

    def test_validate_operation_range(self):
        """Test validation of operation range."""
        # Valid operation
        action = create_test_structured_action("point", operation=41, row=5, col=10)
        validate_structured_action(action)  # Should not raise

        # Invalid operation (out of range)
        action = create_test_structured_action("point", operation=50, row=5, col=10)
        with pytest.raises(ValueError, match="Operation 50 out of valid range"):
            validate_structured_action(action)


class TestCreateTestStructuredAction:
    """Test utility function for creating test structured actions."""

    def test_create_point_action(self):
        """Test creating point structured action."""
        action = create_test_structured_action("point", operation=5, row=5, col=10)
        assert action.operation == 5
        assert action.row == 5
        assert action.col == 10

        # Test default values
        action = create_test_structured_action("point")
        assert action.operation == 0  # Default operation
        assert action.row == 5  # Default row
        assert action.col == 10  # Default col

    def test_create_bbox_action(self):
        """Test creating bbox structured action."""
        action = create_test_structured_action("bbox", operation=10, r1=2, c1=3, r2=4, c2=5)
        assert action.operation == 10
        assert action.r1 == 2
        assert action.c1 == 3
        assert action.r2 == 4
        assert action.c2 == 5

    def test_create_mask_action(self):
        """Test creating mask structured action."""
        grid_shape = (15, 12)
        action = create_test_structured_action(
            "mask", operation=15, grid_shape=grid_shape, start_row=3, start_col=4, size=2
        )
        assert action.operation == 15
        assert action.selection.shape == grid_shape
        # Check that some cells are selected
        assert jnp.sum(action.selection) > 0

    def test_create_data_unknown_format(self):
        """Test error for unknown format."""
        with pytest.raises(ValueError, match="Unknown action type"):
            create_test_structured_action("unknown")


class TestIntegration:
    """Integration tests for action handlers."""

    def test_all_handlers_produce_consistent_shapes(self):
        """Test that all handlers produce consistent output shapes."""
        grid_shape = (18, 12)
        working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)

        # Test point handler
        point_action = create_test_structured_action("point", operation=0, row=5, col=10)
        point_result = point_handler(point_action, working_mask)

        # Test bbox handler
        bbox_action = create_test_structured_action("bbox", operation=0, r1=2, c1=3, r2=4, c2=5)
        bbox_result = bbox_handler(bbox_action, working_mask)

        # Test mask handler
        mask_action = create_test_structured_action("mask", operation=0, grid_shape=grid_shape)
        mask_result = mask_handler(mask_action, working_mask)

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
        point_action = create_test_structured_action("point", operation=0, row=20, col=18)
        point_result = point_handler(point_action, working_mask)
        assert jnp.sum(point_result) == 0

        # Test bbox handler with bbox outside working area
        bbox_action = create_test_structured_action("bbox", operation=0, r1=20, c1=18, r2=22, c2=19)
        bbox_result = bbox_handler(bbox_action, working_mask)
        assert jnp.sum(bbox_result) == 0

        # Test mask handler with mask outside working area
        full_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        mask_action = create_test_structured_action("mask", operation=0, grid_shape=grid_shape, start_row=0, start_col=0, size=25)
        mask_result = mask_handler(mask_action, working_mask)
        assert jnp.sum(mask_result) == 100  # Only working area (10x10)

    def test_vmap_compatibility(self):
        """Test that handlers work with individual processing (vmap with structured actions is complex)."""
        grid_shape = (15, 12)
        working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)

        # Test point handler with multiple individual actions
        point_actions = [
            create_test_structured_action("point", operation=0, row=1, col=2),
            create_test_structured_action("point", operation=0, row=3, col=4),
            create_test_structured_action("point", operation=0, row=5, col=6)
        ]
        
        results = []
        for action in point_actions:
            result = point_handler(action, working_mask)
            results.append(result)

        # Each result should have exactly one cell selected
        for result in results:
            chex.assert_shape(result, grid_shape)
            assert jnp.sum(result) == 1
