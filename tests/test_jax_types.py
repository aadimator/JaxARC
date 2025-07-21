"""
Tests for JAXTyping definitions in jax_types.py.

This module tests that the centralized JAXTyping definitions work correctly
and provide proper type annotations for JAX arrays.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxarc.utils.jax_types import (
    BboxCoords,
    ColorValue,
    ContinuousSelectionArray,
    EpisodeDone,
    GridArray,
    MaskArray,
    OperationId,
    PointCoords,
    SelectionArray,
    SimilarityScore,
    StepCount,
    TaskIndex,
)


class TestCoreGridTypes:
    """Test core grid type definitions."""

    def test_grid_array_creation(self):
        """Test GridArray type with 2D integer array."""
        # Create a valid grid array
        grid: GridArray = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)

        assert grid.shape == (2, 3)
        assert grid.dtype == jnp.int32
        assert isinstance(grid, jnp.ndarray)

    def test_mask_array_creation(self):
        """Test MaskArray type with 2D boolean array."""
        # Create a valid mask array
        mask: MaskArray = jnp.array([[True, False, True], [False, True, False]])

        assert mask.shape == (2, 3)
        assert mask.dtype == jnp.bool_
        assert isinstance(mask, jnp.ndarray)

    def test_selection_array_creation(self):
        """Test SelectionArray type with 2D boolean array."""
        # Create a valid selection array
        selection: SelectionArray = jnp.array([[True, False], [False, True]])

        assert selection.shape == (2, 2)
        assert selection.dtype == jnp.bool_
        assert isinstance(selection, jnp.ndarray)

    def test_continuous_selection_array_creation(self):
        """Test ContinuousSelectionArray type with 2D float array."""
        # Create a valid continuous selection array
        selection: ContinuousSelectionArray = jnp.array([[0.5, 0.0], [1.0, 0.3]])

        assert selection.shape == (2, 2)
        assert selection.dtype in (jnp.float32, jnp.float64)
        assert isinstance(selection, jnp.ndarray)


class TestBatchTypes:
    """Test batch type definitions using *batch modifier."""

    def test_batch_grid_array_creation(self):
        """Test GridArray type with 3D integer array (batched)."""
        # Create a valid batch grid array using the unified GridArray type
        batch_grids: GridArray = jnp.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=jnp.int32
        )

        assert batch_grids.shape == (2, 2, 2)  # batch, height, width
        assert batch_grids.dtype == jnp.int32
        assert isinstance(batch_grids, jnp.ndarray)

    def test_batch_mask_array_creation(self):
        """Test MaskArray type with 3D boolean array (batched)."""
        # Create a valid batch mask array using the unified MaskArray type
        batch_masks: MaskArray = jnp.array(
            [[[True, False], [False, True]], [[False, True], [True, False]]]
        )

        assert batch_masks.shape == (2, 2, 2)  # batch, height, width
        assert batch_masks.dtype == jnp.bool_
        assert isinstance(batch_masks, jnp.ndarray)

    def test_single_vs_batch_compatibility(self):
        """Test that the same type works for both single and batched arrays."""
        # Single grid
        single_grid: GridArray = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)

        # Batched grids
        batch_grids: GridArray = jnp.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=jnp.int32
        )

        # Both should work with the same type annotation
        assert single_grid.shape == (2, 2)  # height, width
        assert batch_grids.shape == (2, 2, 2)  # batch, height, width

        # Test with scoring types too
        single_score: SimilarityScore = jnp.array(0.85, dtype=jnp.float32)
        batch_scores: SimilarityScore = jnp.array([0.85, 0.92, 0.78], dtype=jnp.float32)

        assert single_score.shape == ()  # scalar
        assert batch_scores.shape == (3,)  # batch


class TestActionTypes:
    """Test action-related type definitions."""

    def test_point_coords_creation(self):
        """Test PointCoords type with 1D integer array."""
        # Create valid point coordinates
        point: PointCoords = jnp.array([5, 10], dtype=jnp.int32)

        assert point.shape == (2,)
        assert point.dtype == jnp.int32
        assert isinstance(point, jnp.ndarray)

    def test_bbox_coords_creation(self):
        """Test BboxCoords type with 1D integer array."""
        # Create valid bounding box coordinates
        bbox: BboxCoords = jnp.array([1, 2, 5, 8], dtype=jnp.int32)

        assert bbox.shape == (4,)
        assert bbox.dtype == jnp.int32
        assert isinstance(bbox, jnp.ndarray)

    def test_operation_id_creation(self):
        """Test OperationId type with scalar integer."""
        # Create valid operation ID
        op_id: OperationId = jnp.array(15, dtype=jnp.int32)

        assert op_id.shape == ()  # scalar
        assert op_id.dtype == jnp.int32
        assert isinstance(op_id, jnp.ndarray)


class TestScoringTypes:
    """Test scoring and value type definitions."""

    def test_similarity_score_creation(self):
        """Test SimilarityScore type with scalar float."""
        # Create valid similarity score
        score: SimilarityScore = jnp.array(0.85, dtype=jnp.float32)

        assert score.shape == ()  # scalar
        assert score.dtype in (jnp.float32, jnp.float64)
        assert isinstance(score, jnp.ndarray)


class TestEnvironmentStateTypes:
    """Test environment state type definitions."""

    def test_step_count_creation(self):
        """Test StepCount type with scalar integer."""
        # Create valid step count
        steps: StepCount = jnp.array(42, dtype=jnp.int32)

        assert steps.shape == ()  # scalar
        assert steps.dtype == jnp.int32
        assert isinstance(steps, jnp.ndarray)

    def test_episode_done_creation(self):
        """Test EpisodeDone type with scalar boolean."""
        # Create valid episode done flag
        done: EpisodeDone = jnp.array(True)

        assert done.shape == ()  # scalar
        assert done.dtype == jnp.bool_
        assert isinstance(done, jnp.ndarray)

    def test_task_index_creation(self):
        """Test TaskIndex type with scalar integer."""
        # Create valid task index
        task_idx: TaskIndex = jnp.array(123, dtype=jnp.int32)

        assert task_idx.shape == ()  # scalar
        assert task_idx.dtype == jnp.int32
        assert isinstance(task_idx, jnp.ndarray)


class TestUtilityTypes:
    """Test utility type definitions."""

    def test_color_value_creation(self):
        """Test ColorValue type with scalar integer."""
        # Create valid color value
        color: ColorValue = jnp.array(7, dtype=jnp.int32)

        assert color.shape == ()  # scalar
        assert color.dtype == jnp.int32
        assert isinstance(color, jnp.ndarray)


class TestTypeCompatibility:
    """Test that types work with JAX operations."""

    def test_grid_operations_with_types(self):
        """Test that typed arrays work with JAX operations."""
        # Create typed arrays
        grid1: GridArray = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        grid2: GridArray = jnp.array([[5, 6], [7, 8]], dtype=jnp.int32)
        mask: MaskArray = jnp.array([[True, False], [True, True]])

        # Test JAX operations work with typed arrays
        result = jnp.where(mask, grid1, grid2)
        assert result.shape == (2, 2)
        assert result.dtype == jnp.int32

        # Test array operations
        sum_result = jnp.sum(grid1)
        assert isinstance(sum_result, jnp.ndarray)

        # Test masking operations
        masked_result = grid1 * mask
        assert masked_result.shape == (2, 2)

    def test_batch_operations_with_types(self):
        """Test that batch typed arrays work with JAX operations."""
        # Create batch typed arrays using the unified GridArray type
        batch_grids: GridArray = jnp.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=jnp.int32
        )

        # Test vmap-style operations
        result = jnp.sum(batch_grids, axis=(1, 2))  # Sum each grid in batch
        assert result.shape == (2,)  # One sum per batch item
        assert result.dtype == jnp.int32


class TestImportAccess:
    """Test that types can be imported from utils package."""

    def test_import_from_utils(self):
        """Test that types can be imported from jaxarc.utils."""

        # Create arrays using imported types
        grid: GridArray = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        mask: MaskArray = jnp.array([[True, False], [True, True]])
        selection: SelectionArray = jnp.array([[False, True], [True, False]])

        assert grid.shape == (2, 2)
        assert mask.shape == (2, 2)
        assert selection.shape == (2, 2)

    def test_import_from_jax_types_directly(self):
        """Test that types can be imported directly from jax_types module."""

        # Create arrays using directly imported types
        grid: GridArray = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        mask: MaskArray = jnp.array([[True, True], [False, True]])
        op: OperationId = jnp.array(10, dtype=jnp.int32)

        assert grid.shape == (2, 2)
        assert mask.shape == (2, 2)
        assert op.shape == ()
