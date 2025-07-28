"""
JAX-compatible error handling using Equinox error utilities.

This module provides runtime error handling that works within JAX transformations
using equinox.error_if and equinox.branched_error_if. It supports environment
variable configuration for different error handling modes.

Key Features:
- JAX-compatible error checking using equinox.error_if
- Branched error handling with specific error messages
- Environment variable configuration (EQX_ON_ERROR)
- Action validation with runtime error checking
- Grid operation validation with detailed diagnostics
- Debugging support with breakpoint mode

Environment Variables:
- EQX_ON_ERROR: Controls error handling behavior
  - "raise" (default): Raise runtime errors
  - "nan": Return NaN values and continue
  - "breakpoint": Open debugger on errors
- EQX_ON_ERROR_BREAKPOINT_FRAMES: Number of frames to show in breakpoint mode

Examples:
    ```python
    import os
    from jaxarc.utils.error_handling import (
        JAXErrorHandler,
        setup_error_environment,
        validate_action,
        validate_grid_operation
    )

    # Setup error handling
    setup_error_environment()

    # Validate actions
    validated_action = validate_action(action, config)

    # Validate grid operations
    validated_state = validate_grid_operation(state, operation_id)

    # Use custom error checking
    result = JAXErrorHandler.check_bounds(
        value, min_val=0, max_val=10, 
        error_msg="Value out of bounds"
    )
    ```
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from ..state import ArcEnvState
from ..envs.config import JaxArcConfig
from ..envs.structured_actions import StructuredAction, PointAction, BboxAction, MaskAction
from .jax_types import OperationId


class JAXErrorHandler:
    """JAX-compatible error handling using Equinox error utilities.
    
    This class provides static methods for runtime error checking that work
    within JAX transformations. It uses equinox.error_if and equinox.branched_error_if
    for JAX-compatible error handling.
    
    The error handling behavior can be controlled via the EQX_ON_ERROR environment
    variable:
    - "raise": Raise runtime errors (default)
    - "nan": Return NaN values and continue execution
    - "breakpoint": Open debugger on errors
    """

    ERROR_MODES = {
        "raise": "Raise runtime errors",
        "nan": "Return NaN and continue",
        "breakpoint": "Open debugger"
    }

    @staticmethod
    @eqx.filter_jit
    def check_bounds(
        value: Union[Int[Array, ""], float, int],
        min_val: Union[Int[Array, ""], float, int],
        max_val: Union[Int[Array, ""], float, int],
        error_msg: str = "Value out of bounds"
    ) -> Union[Int[Array, ""], float, int]:
        """Check if value is within bounds with JAX-compatible error handling.
        
        Args:
            value: Value to check
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (exclusive)
            error_msg: Error message to display
            
        Returns:
            The original value if valid
            
        Raises:
            RuntimeError: If value is out of bounds (when EQX_ON_ERROR="raise")
            
        Examples:
            ```python
            # Check operation ID bounds
            op_id = JAXErrorHandler.check_bounds(
                operation_id, 0, 42, "Operation ID out of range"
            )
            
            # Check grid coordinates
            row = JAXErrorHandler.check_bounds(
                row, 0, grid_height, "Row coordinate out of bounds"
            )
            ```
        """
        # Convert to JAX arrays for consistent handling
        if not isinstance(value, jnp.ndarray):
            value = jnp.array(value)
        if not isinstance(min_val, jnp.ndarray):
            min_val = jnp.array(min_val)
        if not isinstance(max_val, jnp.ndarray):
            max_val = jnp.array(max_val)
            
        # Check bounds condition
        out_of_bounds = (value < min_val) | (value >= max_val)
        
        return eqx.error_if(value, out_of_bounds, error_msg)

    @staticmethod
    @eqx.filter_jit
    def check_array_shape(
        array: Array,
        expected_shape: tuple,
        error_msg: str = "Array shape mismatch"
    ) -> Array:
        """Check if array has expected shape.
        
        Args:
            array: Array to check
            expected_shape: Expected shape tuple
            error_msg: Error message to display
            
        Returns:
            The original array if shape is correct
            
        Examples:
            ```python
            # Check grid shape
            grid = JAXErrorHandler.check_array_shape(
                grid, (30, 30), "Grid must be 30x30"
            )
            ```
        """
        # Check each dimension
        shape_mismatch = jnp.array(False)
        
        if len(array.shape) != len(expected_shape):
            shape_mismatch = jnp.array(True)
        else:
            for i, (actual, expected) in enumerate(zip(array.shape, expected_shape)):
                if expected is not None:  # Allow None for flexible dimensions
                    shape_mismatch = shape_mismatch | (actual != expected)
        
        return eqx.error_if(array, shape_mismatch, error_msg)

    @staticmethod
    @eqx.filter_jit
    def check_array_dtype(
        array: Array,
        expected_dtype: jnp.dtype,
        error_msg: str = "Array dtype mismatch"
    ) -> Array:
        """Check if array has expected dtype.
        
        Args:
            array: Array to check
            expected_dtype: Expected dtype
            error_msg: Error message to display
            
        Returns:
            The original array if dtype is correct
            
        Examples:
            ```python
            # Check grid dtype
            grid = JAXErrorHandler.check_array_dtype(
                grid, jnp.int32, "Grid must be int32"
            )
            ```
        """
        dtype_mismatch = array.dtype != expected_dtype
        return eqx.error_if(array, dtype_mismatch, error_msg)

    @staticmethod
    @eqx.filter_jit
    def validate_operation_id(
        operation_id: OperationId,
        max_operations: int = 42,
        error_msg: Optional[str] = None
    ) -> OperationId:
        """Validate operation ID is within valid range.
        
        Args:
            operation_id: Operation ID to validate
            max_operations: Maximum number of operations (exclusive)
            error_msg: Custom error message
            
        Returns:
            Validated operation ID
            
        Examples:
            ```python
            op_id = JAXErrorHandler.validate_operation_id(operation_id)
            ```
        """
        if error_msg is None:
            error_msg = f"Operation ID must be in range [0, {max_operations})"
            
        return JAXErrorHandler.check_bounds(
            operation_id, 0, max_operations, error_msg
        )

    @staticmethod
    @eqx.filter_jit
    def validate_grid_coordinates(
        row: Int[Array, ""],
        col: Int[Array, ""],
        grid_height: int,
        grid_width: int
    ) -> tuple[Int[Array, ""], Int[Array, ""]]:
        """Validate grid coordinates are within bounds.
        
        Args:
            row: Row coordinate
            col: Column coordinate
            grid_height: Grid height
            grid_width: Grid width
            
        Returns:
            Tuple of validated (row, col) coordinates
            
        Examples:
            ```python
            row, col = JAXErrorHandler.validate_grid_coordinates(
                row, col, 30, 30
            )
            ```
        """
        validated_row = JAXErrorHandler.check_bounds(
            row, 0, grid_height, f"Row must be in range [0, {grid_height})"
        )
        validated_col = JAXErrorHandler.check_bounds(
            col, 0, grid_width, f"Column must be in range [0, {grid_width})"
        )
        
        return validated_row, validated_col

    @staticmethod
    @eqx.filter_jit
    def validate_bbox_coordinates(
        r1: Int[Array, ""], c1: Int[Array, ""],
        r2: Int[Array, ""], c2: Int[Array, ""],
        grid_height: int, grid_width: int
    ) -> tuple[Int[Array, ""], Int[Array, ""], Int[Array, ""], Int[Array, ""]]:
        """Validate bounding box coordinates.
        
        Args:
            r1, c1: Top-left coordinates
            r2, c2: Bottom-right coordinates
            grid_height: Grid height
            grid_width: Grid width
            
        Returns:
            Tuple of validated (r1, c1, r2, c2) coordinates
            
        Examples:
            ```python
            r1, c1, r2, c2 = JAXErrorHandler.validate_bbox_coordinates(
                r1, c1, r2, c2, 30, 30
            )
            ```
        """
        # Validate individual coordinates
        r1 = JAXErrorHandler.check_bounds(r1, 0, grid_height, "r1 out of bounds")
        c1 = JAXErrorHandler.check_bounds(c1, 0, grid_width, "c1 out of bounds")
        r2 = JAXErrorHandler.check_bounds(r2, 0, grid_height, "r2 out of bounds")
        c2 = JAXErrorHandler.check_bounds(c2, 0, grid_width, "c2 out of bounds")
        
        # Validate bbox ordering
        r1 = eqx.error_if(r1, r1 > r2, "r1 must be <= r2")
        c1 = eqx.error_if(c1, c1 > c2, "c1 must be <= c2")
        
        return r1, c1, r2, c2

    @staticmethod
    @eqx.filter_jit
    def branched_operation_error(
        state: ArcEnvState,
        has_error: Bool[Array, ""],
        error_type: Int[Array, ""]
    ) -> ArcEnvState:
        """Handle operation errors with specific error messages.
        
        Args:
            state: Current environment state
            has_error: Whether an error occurred
            error_type: Type of error (index into error messages)
            
        Returns:
            State (unchanged if no error)
            
        Examples:
            ```python
            state = JAXErrorHandler.branched_operation_error(
                state, has_error, error_type
            )
            ```
        """
        error_messages = [
            "Fill operation failed: invalid color value",
            "Flood fill failed: no valid selection",
            "Move operation failed: invalid direction",
            "Rotate operation failed: invalid angle",
            "Copy operation failed: no selection",
            "Paste operation failed: empty clipboard",
            "Clear operation failed: no selection",
            "Invert operation failed: invalid grid state",
            "Mirror operation failed: invalid axis",
            "Transpose operation failed: non-square selection",
            "Scale operation failed: invalid scale factor",
            "Crop operation failed: invalid bounds",
            "Extend operation failed: invalid extension",
            "Pattern operation failed: invalid pattern",
            "Filter operation failed: invalid filter type",
            "Transform operation failed: invalid transformation",
            "Composite operation failed: incompatible operations",
            "Validation operation failed: invalid state",
            "Unknown operation error"
        ]
        
        return eqx.branched_error_if(state, has_error, error_type, error_messages)


def setup_error_environment() -> None:
    """Setup error handling environment variables with defaults.
    
    This function configures the error handling environment variables
    if they are not already set, providing sensible defaults for
    development and production use.
    
    Examples:
        ```python
        # Setup error handling at application startup
        setup_error_environment()
        
        # Override for debugging
        import os
        os.environ["EQX_ON_ERROR"] = "breakpoint"
        setup_error_environment()
        ```
    """
    # Set default error handling mode
    if "EQX_ON_ERROR" not in os.environ:
        os.environ["EQX_ON_ERROR"] = "raise"
    
    # Set default frame count for breakpoints
    if "EQX_ON_ERROR_BREAKPOINT_FRAMES" not in os.environ:
        os.environ["EQX_ON_ERROR_BREAKPOINT_FRAMES"] = "3"
    
    # Validate error mode
    error_mode = os.environ["EQX_ON_ERROR"]
    if error_mode not in JAXErrorHandler.ERROR_MODES:
        valid_modes = list(JAXErrorHandler.ERROR_MODES.keys())
        raise ValueError(
            f"Invalid EQX_ON_ERROR mode '{error_mode}'. "
            f"Valid modes: {valid_modes}"
        )


@eqx.filter_jit
def validate_action(action: StructuredAction, config: JaxArcConfig) -> StructuredAction:
    """Validate structured action with runtime error checking.
    
    This function validates all aspects of a structured action including
    operation ID, coordinates, and format-specific constraints.
    
    Args:
        action: Structured action to validate
        config: Environment configuration
        
    Returns:
        Validated action (unchanged if valid)
        
    Raises:
        RuntimeError: If action is invalid (when EQX_ON_ERROR="raise")
        
    Examples:
        ```python
        # Validate point action
        point_action = PointAction(operation=0, row=5, col=7)
        validated = validate_action(point_action, config)
        
        # Validate bbox action
        bbox_action = BboxAction(operation=1, r1=2, c1=3, r2=8, c2=9)
        validated = validate_action(bbox_action, config)
        ```
    """
    # Validate operation ID
    action = eqx.tree_at(
        lambda a: a.operation,
        action,
        JAXErrorHandler.validate_operation_id(action.operation)
    )
    
    # Get grid dimensions from config
    grid_height = config.dataset.max_grid_height
    grid_width = config.dataset.max_grid_width
    
    # Format-specific validation
    if isinstance(action, PointAction):
        # Validate point coordinates
        validated_row, validated_col = JAXErrorHandler.validate_grid_coordinates(
            action.row, action.col, grid_height, grid_width
        )
        action = eqx.tree_at(
            lambda a: (a.row, a.col),
            action,
            (validated_row, validated_col)
        )
        
    elif isinstance(action, BboxAction):
        # Validate bbox coordinates
        r1, c1, r2, c2 = JAXErrorHandler.validate_bbox_coordinates(
            action.r1, action.c1, action.r2, action.c2,
            grid_height, grid_width
        )
        action = eqx.tree_at(
            lambda a: (a.r1, a.c1, a.r2, a.c2),
            action,
            (r1, c1, r2, c2)
        )
        
    elif isinstance(action, MaskAction):
        # Validate mask shape
        expected_shape = (grid_height, grid_width)
        action = eqx.tree_at(
            lambda a: a.selection,
            action,
            JAXErrorHandler.check_array_shape(
                action.selection, expected_shape, 
                f"Mask selection must have shape {expected_shape}"
            )
        )
        
        # Validate mask dtype
        action = eqx.tree_at(
            lambda a: a.selection,
            action,
            JAXErrorHandler.check_array_dtype(
                action.selection, jnp.bool_, 
                "Mask selection must be boolean"
            )
        )
    
    return action


@eqx.filter_jit
def validate_grid_operation(
    state: ArcEnvState, 
    operation_id: OperationId
) -> ArcEnvState:
    """Validate grid operation with specific error messages.
    
    This function validates that a grid operation can be performed on the
    current state, checking for common error conditions.
    
    Args:
        state: Current environment state
        operation_id: Operation to validate
        
    Returns:
        State (unchanged if valid)
        
    Raises:
        RuntimeError: If operation cannot be performed
        
    Examples:
        ```python
        # Validate before performing operation
        validated_state = validate_grid_operation(state, operation_id)
        ```
    """
    # Validate operation ID
    validated_op_id = JAXErrorHandler.validate_operation_id(operation_id)
    
    # Check if operation is allowed
    operation_allowed = state.is_operation_allowed(validated_op_id)
    state = eqx.error_if(
        state, 
        ~operation_allowed,
        f"Operation {validated_op_id} is not currently allowed"
    )
    
    # Operation-specific validation
    has_selection = jnp.any(state.selected)
    has_clipboard = jnp.any(state.clipboard != 0)
    
    # Define error conditions for different operations
    # This is a simplified example - real implementation would have more conditions
    error_conditions = jnp.array([
        # Fill operations (0-9) need selection
        (validated_op_id < 10) & (~has_selection),
        # Copy operations (10-19) need selection  
        ((validated_op_id >= 10) & (validated_op_id < 20)) & (~has_selection),
        # Paste operations (20-29) need clipboard
        ((validated_op_id >= 20) & (validated_op_id < 30)) & (~has_clipboard),
        # Transform operations (30-39) need selection
        ((validated_op_id >= 30) & (validated_op_id < 40)) & (~has_selection),
    ])
    
    # Check if any error condition is met
    has_error = jnp.any(error_conditions)
    error_type = jnp.argmax(error_conditions)  # Get first error type
    
    return JAXErrorHandler.branched_operation_error(state, has_error, error_type)


@eqx.filter_jit
def validate_state_consistency(state: ArcEnvState) -> ArcEnvState:
    """Validate state consistency with detailed error messages.
    
    This function performs comprehensive validation of the environment state
    to ensure all components are consistent and valid.
    
    Args:
        state: Environment state to validate
        
    Returns:
        State (unchanged if valid)
        
    Raises:
        RuntimeError: If state is inconsistent
        
    Examples:
        ```python
        # Validate state after updates
        validated_state = validate_state_consistency(state)
        ```
    """
    # Check grid shape consistency
    state = eqx.error_if(
        state,
        state.working_grid.shape != state.target_grid.shape,
        "Working grid and target grid shapes must match"
    )
    
    # Check mask consistency
    state = eqx.error_if(
        state,
        state.working_grid_mask.shape != state.working_grid.shape,
        "Working grid mask shape must match working grid"
    )
    
    state = eqx.error_if(
        state,
        state.target_grid_mask.shape != state.target_grid.shape,
        "Target grid mask shape must match target grid"
    )
    
    # Check selection consistency
    state = eqx.error_if(
        state,
        state.selected.shape != state.working_grid.shape,
        "Selection mask shape must match working grid"
    )
    
    # Check clipboard consistency
    state = eqx.error_if(
        state,
        state.clipboard.shape != state.working_grid.shape,
        "Clipboard shape must match working grid"
    )
    
    # Check step count bounds
    state = eqx.error_if(
        state,
        state.step_count < 0,
        "Step count cannot be negative"
    )
    
    # Check similarity score bounds
    state = eqx.error_if(
        state,
        (state.similarity_score < 0.0) | (state.similarity_score > 1.0),
        "Similarity score must be in range [0.0, 1.0]"
    )
    
    # Check episode mode validity
    state = eqx.error_if(
        state,
        (state.episode_mode != 0) & (state.episode_mode != 1),
        "Episode mode must be 0 (train) or 1 (test)"
    )
    
    # Check current example index bounds
    max_pairs = jnp.where(
        state.episode_mode == 0,
        len(state.available_demo_pairs),
        len(state.available_test_pairs)
    )
    
    state = eqx.error_if(
        state,
        (state.current_example_idx < 0) | (state.current_example_idx >= max_pairs),
        "Current example index out of bounds"
    )
    
    return state


def get_error_mode() -> str:
    """Get current error handling mode from environment variable.
    
    Returns:
        Current error mode ("raise", "nan", or "breakpoint")
        
    Examples:
        ```python
        mode = get_error_mode()
        if mode == "breakpoint":
            print("Debugging mode enabled")
        ```
    """
    return os.environ.get("EQX_ON_ERROR", "raise")


def set_error_mode(mode: str) -> None:
    """Set error handling mode via environment variable.
    
    Args:
        mode: Error mode ("raise", "nan", or "breakpoint")
        
    Raises:
        ValueError: If mode is invalid
        
    Examples:
        ```python
        # Enable debugging mode
        set_error_mode("breakpoint")
        
        # Enable graceful degradation
        set_error_mode("nan")
        
        # Enable strict error checking
        set_error_mode("raise")
        ```
    """
    if mode not in JAXErrorHandler.ERROR_MODES:
        valid_modes = list(JAXErrorHandler.ERROR_MODES.keys())
        raise ValueError(f"Invalid error mode '{mode}'. Valid modes: {valid_modes}")
    
    os.environ["EQX_ON_ERROR"] = mode


def enable_debug_mode(frames: int = 3) -> None:
    """Enable debugging mode with breakpoints on errors.
    
    Args:
        frames: Number of stack frames to show in debugger
        
    Examples:
        ```python
        # Enable debugging with default frame count
        enable_debug_mode()
        
        # Enable debugging with more context
        enable_debug_mode(frames=5)
        ```
    """
    set_error_mode("breakpoint")
    os.environ["EQX_ON_ERROR_BREAKPOINT_FRAMES"] = str(frames)


def disable_debug_mode() -> None:
    """Disable debugging mode and return to normal error handling.
    
    Examples:
        ```python
        # Disable debugging mode
        disable_debug_mode()
        ```
    """
    set_error_mode("raise")