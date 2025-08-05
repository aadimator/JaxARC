"""
Structured action classes for JaxARC environments.

This module provides JAX-compatible structured action classes using Equinox modules
to replace dictionary-based actions. These structured actions are designed for
optimal performance with JAX transformations (jit, vmap, pmap) and provide
type safety and validation.

Key Features:
- Equinox modules for JAX compatibility and automatic PyTree registration
- Type-safe action representations with validation
- Efficient conversion to selection masks
- Support for point, bbox, and mask-based actions
- JIT-compilable validation methods
"""

from __future__ import annotations

import abc
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp

from ..utils.jax_types import MaskArray, OperationId, SelectionArray


class BaseAction(eqx.Module):
    """Base class for all structured actions.
    
    All action types must inherit from this base class and implement
    the to_selection_mask method for converting actions to grid selections.
    
    Attributes:
        operation: ARCLE operation ID (0-41)
        action_type: Type identifier for JAX compatibility (0=point, 1=bbox, 2=mask)
    """
    
    operation: OperationId
    action_type: jnp.int32
    
    @abc.abstractmethod
    def to_selection_mask(self, grid_shape: tuple[int, int]) -> SelectionArray:
        """Convert action to selection mask.
        
        Args:
            grid_shape: Shape of the grid (height, width)
            
        Returns:
            Boolean mask with same shape as grid indicating selected cells
        """
        pass
    
    @abc.abstractmethod
    def validate(self, grid_shape: tuple[int, int], max_operations: int = 42) -> 'BaseAction':
        """Validate action parameters and return validated action.
        
        Args:
            grid_shape: Shape of the grid (height, width)
            max_operations: Maximum number of operations (default 42 for ops 0-41)
            
        Returns:
            Validated action with clipped/corrected parameters
        """
        pass


class PointAction(BaseAction):
    """Point-based action using single coordinate.
    
    This action type selects a single point on the grid using row and column
    coordinates. It's the most memory-efficient action format.
    
    Attributes:
        operation: ARCLE operation ID (0-41)
        action_type: Type identifier (always 0 for PointAction)
        row: Row coordinate (0-based)
        col: Column coordinate (0-based)
    """
    
    operation: jnp.int32
    action_type: jnp.int32  # Always 0 for PointAction
    row: jnp.int32
    col: jnp.int32
    
    def to_selection_mask(self, grid_shape: tuple[int, int]) -> SelectionArray:
        """Convert point coordinates to selection mask.
        
        Args:
            grid_shape: Shape of the grid (height, width)
            
        Returns:
            Boolean mask with single point selected
        """
        grid_height, grid_width = grid_shape
        
        # Create mask with single point
        mask = jnp.zeros((grid_height, grid_width), dtype=jnp.bool_)
        
        # Ensure coordinates are within bounds before setting
        valid_row = jnp.clip(self.row, 0, grid_height - 1)
        valid_col = jnp.clip(self.col, 0, grid_width - 1)
        
        mask = mask.at[valid_row, valid_col].set(True)
        return mask
    
    def validate(self, grid_shape: tuple[int, int], max_operations: int = 42) -> 'PointAction':
        """Validate point action parameters.
        
        Args:
            grid_shape: Shape of the grid (height, width)
            max_operations: Maximum number of operations
            
        Returns:
            Validated point action with clipped coordinates and operation
        """
        grid_height, grid_width = grid_shape
        
        # Clip operation to valid range
        valid_operation = jnp.clip(self.operation, 0, max_operations - 1)
        
        # Clip coordinates to valid range
        valid_row = jnp.clip(self.row, 0, grid_height - 1)
        valid_col = jnp.clip(self.col, 0, grid_width - 1)
        
        return PointAction(
            operation=valid_operation,
            action_type=jnp.array(0, dtype=jnp.int32),
            row=valid_row,
            col=valid_col
        )


class BboxAction(BaseAction):
    """Bounding box action using rectangular selection.
    
    This action type selects a rectangular region on the grid using
    two corner coordinates (top-left and bottom-right).
    
    Attributes:
        operation: ARCLE operation ID (0-41)
        action_type: Type identifier (always 1 for BboxAction)
        r1: Top-left row coordinate
        c1: Top-left column coordinate
        r2: Bottom-right row coordinate
        c2: Bottom-right column coordinate
    """
    
    operation: jnp.int32
    action_type: jnp.int32  # Always 1 for BboxAction
    r1: jnp.int32
    c1: jnp.int32
    r2: jnp.int32
    c2: jnp.int32
    
    def to_selection_mask(self, grid_shape: tuple[int, int]) -> SelectionArray:
        """Convert bounding box coordinates to selection mask.
        
        Args:
            grid_shape: Shape of the grid (height, width)
            
        Returns:
            Boolean mask with rectangular region selected
        """
        grid_height, grid_width = grid_shape
        
        # Ensure proper ordering and clipping
        min_r = jnp.minimum(self.r1, self.r2)
        max_r = jnp.maximum(self.r1, self.r2)
        min_c = jnp.minimum(self.c1, self.c2)
        max_c = jnp.maximum(self.c1, self.c2)
        
        # Clip to grid bounds
        min_r = jnp.clip(min_r, 0, grid_height - 1)
        max_r = jnp.clip(max_r, 0, grid_height - 1)
        min_c = jnp.clip(min_c, 0, grid_width - 1)
        max_c = jnp.clip(max_c, 0, grid_width - 1)
        
        # Create coordinate meshes
        rows = jnp.arange(grid_height)
        cols = jnp.arange(grid_width)
        row_mesh, col_mesh = jnp.meshgrid(rows, cols, indexing="ij")
        
        # Create bbox mask (inclusive bounds)
        mask = (
            (row_mesh >= min_r)
            & (row_mesh <= max_r)
            & (col_mesh >= min_c)
            & (col_mesh <= max_c)
        )
        
        return mask
    
    def validate(self, grid_shape: tuple[int, int], max_operations: int = 42) -> 'BboxAction':
        """Validate bounding box action parameters.
        
        Args:
            grid_shape: Shape of the grid (height, width)
            max_operations: Maximum number of operations
            
        Returns:
            Validated bbox action with clipped coordinates and operation
        """
        grid_height, grid_width = grid_shape
        
        # Clip operation to valid range
        valid_operation = jnp.clip(self.operation, 0, max_operations - 1)
        
        # Clip coordinates to valid range
        valid_r1 = jnp.clip(self.r1, 0, grid_height - 1)
        valid_c1 = jnp.clip(self.c1, 0, grid_width - 1)
        valid_r2 = jnp.clip(self.r2, 0, grid_height - 1)
        valid_c2 = jnp.clip(self.c2, 0, grid_width - 1)
        
        return BboxAction(
            operation=valid_operation,
            action_type=jnp.array(1, dtype=jnp.int32),
            r1=valid_r1,
            c1=valid_c1,
            r2=valid_r2,
            c2=valid_c2
        )


class MaskAction(BaseAction):
    """Mask-based action using arbitrary selection.
    
    This action type allows arbitrary selection patterns using a boolean
    mask that directly specifies which cells are selected.
    
    Attributes:
        operation: ARCLE operation ID (0-41)
        action_type: Type identifier (always 2 for MaskAction)
        selection: Boolean mask indicating selected cells
    """
    
    operation: jnp.int32
    action_type: jnp.int32  # Always 2 for MaskAction
    selection: SelectionArray
    
    def to_selection_mask(self, grid_shape: tuple[int, int]) -> SelectionArray:
        """Return the selection mask directly.
        
        Args:
            grid_shape: Shape of the grid (height, width) - used for validation
            
        Returns:
            The selection mask (assumed to already match grid_shape)
        """
        # For simplicity, assume the mask already has the correct shape
        # This is reasonable since masks are typically created with the correct shape
        return self.selection
    
    def validate(self, grid_shape: tuple[int, int], max_operations: int = 42) -> 'MaskAction':
        """Validate mask action parameters.
        
        Args:
            grid_shape: Shape of the grid (height, width)
            max_operations: Maximum number of operations
            
        Returns:
            Validated mask action with clipped operation
        """
        # Clip operation to valid range
        valid_operation = jnp.clip(self.operation, 0, max_operations - 1)
        
        # Return with validated operation (assume selection is already correct shape)
        return MaskAction(
            operation=valid_operation,
            action_type=jnp.array(2, dtype=jnp.int32),
            selection=self.selection
        )


# Type alias for any structured action
StructuredAction = Union[PointAction, BboxAction, MaskAction]


def create_point_action(operation, row, col) -> PointAction:
    """Create a point action with the given parameters.
    
    This function supports both single and batched inputs. When batched inputs
    are provided, all fields will have the same batch dimension.
    
    Args:
        operation: ARCLE operation ID (0-41). Can be int or JAX array.
        row: Row coordinate. Can be int or JAX array.
        col: Column coordinate. Can be int or JAX array.
        
    Returns:
        PointAction instance with consistent batch dimensions
    """
    # Convert inputs to JAX arrays
    operation_array = jnp.array(operation, dtype=jnp.int32)
    row_array = jnp.array(row, dtype=jnp.int32)
    col_array = jnp.array(col, dtype=jnp.int32)
    
    # Create action_type with the same shape as operation
    # For point actions, action_type is always 0
    if operation_array.shape == ():
        # Scalar case
        action_type_array = jnp.array(0, dtype=jnp.int32)
    else:
        # Batched case - create array of 0s with same shape as operation
        action_type_array = jnp.zeros_like(operation_array, dtype=jnp.int32)
    
    return PointAction(
        operation=operation_array,
        action_type=action_type_array,
        row=row_array,
        col=col_array
    )


def create_bbox_action(operation, r1, c1, r2, c2) -> BboxAction:
    """Create a bounding box action with the given parameters.
    
    This function supports both single and batched inputs. When batched inputs
    are provided, all fields will have the same batch dimension.
    
    Args:
        operation: ARCLE operation ID (0-41). Can be int or JAX array.
        r1: Top-left row coordinate. Can be int or JAX array.
        c1: Top-left column coordinate. Can be int or JAX array.
        r2: Bottom-right row coordinate. Can be int or JAX array.
        c2: Bottom-right column coordinate. Can be int or JAX array.
        
    Returns:
        BboxAction instance with consistent batch dimensions
    """
    # Convert inputs to JAX arrays
    operation_array = jnp.array(operation, dtype=jnp.int32)
    r1_array = jnp.array(r1, dtype=jnp.int32)
    c1_array = jnp.array(c1, dtype=jnp.int32)
    r2_array = jnp.array(r2, dtype=jnp.int32)
    c2_array = jnp.array(c2, dtype=jnp.int32)
    
    # Create action_type with the same shape as operation
    # For bbox actions, action_type is always 1
    if operation_array.shape == ():
        # Scalar case
        action_type_array = jnp.array(1, dtype=jnp.int32)
    else:
        # Batched case - create array of 1s with same shape as operation
        action_type_array = jnp.ones_like(operation_array, dtype=jnp.int32)
    
    return BboxAction(
        operation=operation_array,
        action_type=action_type_array,
        r1=r1_array,
        c1=c1_array,
        r2=r2_array,
        c2=c2_array
    )


def create_mask_action(operation, selection: SelectionArray) -> MaskAction:
    """Create a mask action with the given parameters.
    
    This function supports both single and batched inputs. When batched inputs
    are provided, all fields will have the same batch dimension.
    
    Args:
        operation: ARCLE operation ID (0-41). Can be int or JAX array.
        selection: Boolean mask indicating selected cells
        
    Returns:
        MaskAction instance with consistent batch dimensions
    """
    # Convert operation to JAX array
    operation_array = jnp.array(operation, dtype=jnp.int32)
    
    # Create action_type with the same shape as operation
    # For mask actions, action_type is always 2
    if operation_array.shape == ():
        # Scalar case
        action_type_array = jnp.array(2, dtype=jnp.int32)
    else:
        # Batched case - create array of 2s with same shape as operation
        action_type_array = jnp.full_like(operation_array, 2, dtype=jnp.int32)
    
    return MaskAction(
        operation=operation_array,
        action_type=action_type_array,
        selection=selection
    )


