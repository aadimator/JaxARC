"""
JAX-compatible error handling utilities using Equinox.

This module provides comprehensive error handling capabilities that work within
JAX transformations using equinox.error_if and equinox.branched_error_if.
It supports environment variable-based configuration for different error modes.

Key Features:
- JAX-compatible error checking using equinox.error_if
- Branched error handling with specific error messages
- Environment variable configuration (EQX_ON_ERROR)
- Action validation with detailed error messages
- Grid operation error checking
- Batch processing error diagnosis
"""

from __future__ import annotations

import os
from typing import Any, Callable, Literal, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from loguru import logger

from ..envs.structured_actions import StructuredAction, PointAction, BboxAction, MaskAction
from ..envs.config import JaxArcConfig
from ..state import ArcEnvState
from ..utils.jax_types import GridArray, SelectionArray


# Error mode configuration
ErrorMode = Literal["raise", "nan", "breakpoint"]


class JAXErrorHandler:
    """JAX-compatible error handling using Equinox.
    
    This class provides static methods for error handling that work within
    JAX transformations. It uses equinox.error_if and equinox.branched_error_if
    for runtime error checking that survives JIT compilation.
    """
    
    @staticmethod
    def setup_error_environment() -> None:
        """Setup error handling environment variables.
        
        Sets default error handling mode if not already configured.
        Supports EQX_ON_ERROR environment variable for debugging.
        """
        # Set default error handling mode if not configured
        if "EQX_ON_ERROR" not in os.environ:
            os.environ["EQX_ON_ERROR"] = "raise"
        
        # Set debug frame count for breakpoints
        if "EQX_ON_ERROR_BREAKPOINT_FRAMES" not in os.environ:
            os.environ["EQX_ON_ERROR_BREAKPOINT_FRAMES"] = "3"
        
        logger.debug(f"Error handling mode: {os.environ.get('EQX_ON_ERROR', 'raise')}")
    
    @staticmethod
    @eqx.filter_jit
    def validate_action(action: StructuredAction, config: JaxArcConfig) -> StructuredAction:
        """Validate structured action with runtime error checking.
        
        This function validates action parameters and raises JAX-compatible
        errors for invalid actions. It works with all structured action types.
        
        Args:
            action: Structured action to validate (PointAction, BboxAction, or MaskAction)
            config: Environment configuration for validation parameters
            
        Returns:
            Validated action (same as input if valid)
            
        Raises:
            RuntimeError: If action parameters are invalid (via equinox.error_if)
        """
        # Validate operation bounds
        max_operations = 42  # Operations 0-41
        action = eqx.error_if(
            action,
            (action.operation < 0) | (action.operation >= max_operations),
            f"Invalid operation ID: must be in [0, {max_operations-1}], got operation"
        )
        
        # Type-specific validation using JAX conditionals
        if isinstance(action, PointAction):
            action = JAXErrorHandler._validate_point_action(action, config)
        elif isinstance(action, BboxAction):
            action = JAXErrorHandler._validate_bbox_action(action, config)
        elif isinstance(action, MaskAction):
            action = JAXErrorHandler._validate_mask_action(action, config)
        else:
            # This should not happen with proper typing, but add safety check
            action = eqx.error_if(
                action,
                True,  # Always error for unknown action types
                f"Unknown action type: {type(action)}"
            )
        
        return action
    
    @staticmethod
    @eqx.filter_jit
    def _validate_point_action(action: PointAction, config: JaxArcConfig) -> PointAction:
        """Validate point action coordinates.
        
        Args:
            action: Point action to validate
            config: Environment configuration
            
        Returns:
            Validated point action
        """
        max_height = config.dataset.max_grid_height
        max_width = config.dataset.max_grid_width
        
        # Check row bounds
        action = eqx.error_if(
            action,
            (action.row < 0) | (action.row >= max_height),
            f"Point row out of bounds: must be in [0, {max_height-1}]"
        )
        
        # Check column bounds
        action = eqx.error_if(
            action,
            (action.col < 0) | (action.col >= max_width),
            f"Point col out of bounds: must be in [0, {max_width-1}]"
        )
        
        return action
    
    @staticmethod
    @eqx.filter_jit
    def _validate_bbox_action(action: BboxAction, config: JaxArcConfig) -> BboxAction:
        """Validate bounding box action coordinates.
        
        Args:
            action: Bbox action to validate
            config: Environment configuration
            
        Returns:
            Validated bbox action
        """
        max_height = config.dataset.max_grid_height
        max_width = config.dataset.max_grid_width
        
        # Check all coordinate bounds
        action = eqx.error_if(
            action,
            (action.r1 < 0) | (action.r1 >= max_height),
            f"Bbox r1 out of bounds: must be in [0, {max_height-1}]"
        )
        
        action = eqx.error_if(
            action,
            (action.c1 < 0) | (action.c1 >= max_width),
            f"Bbox c1 out of bounds: must be in [0, {max_width-1}]"
        )
        
        action = eqx.error_if(
            action,
            (action.r2 < 0) | (action.r2 >= max_height),
            f"Bbox r2 out of bounds: must be in [0, {max_height-1}]"
        )
        
        action = eqx.error_if(
            action,
            (action.c2 < 0) | (action.c2 >= max_width),
            f"Bbox c2 out of bounds: must be in [0, {max_width-1}]"
        )
        
        return action
    
    @staticmethod
    @eqx.filter_jit
    def _validate_mask_action(action: MaskAction, config: JaxArcConfig) -> MaskAction:
        """Validate mask action selection.
        
        Args:
            action: Mask action to validate
            config: Environment configuration
            
        Returns:
            Validated mask action
        """
        max_height = config.dataset.max_grid_height
        max_width = config.dataset.max_grid_width
        expected_shape = (max_height, max_width)
        
        # Check mask shape
        actual_shape = action.selection.shape
        shape_matches = (actual_shape[0] == expected_shape[0]) & (actual_shape[1] == expected_shape[1])
        
        action = eqx.error_if(
            action,
            ~shape_matches,
            f"Mask selection shape mismatch: expected {expected_shape}, got shape"
        )
        
        return action
    
    @staticmethod
    @eqx.filter_jit
    def validate_grid_operation(
        state: ArcEnvState, 
        operation_id: jnp.int32,
        selection: SelectionArray
    ) -> ArcEnvState:
        """Validate grid operation with specific error messages.
        
        This function validates that a grid operation can be performed
        on the current state with the given selection.
        
        Args:
            state: Current environment state
            operation_id: Operation ID to validate
            selection: Selection mask for the operation
            
        Returns:
            Validated state (same as input if valid)
            
        Raises:
            RuntimeError: If operation cannot be performed (via equinox.branched_error_if)
        """
        # Define operation-specific error messages
        error_messages = [
            "Fill operation failed: invalid color or selection",
            "Flood fill failed: no valid selection or color",
            "Move operation failed: invalid direction or selection",
            "Rotate operation failed: invalid angle or selection",
            "Copy operation failed: invalid selection area",
            "Paste operation failed: clipboard empty or invalid position",
            "Clear operation failed: invalid selection",
            "Invert operation failed: invalid selection",
            "Mirror operation failed: invalid axis or selection",
            "Transpose operation failed: invalid selection area",
            "Scale operation failed: invalid factor or selection",
            "Crop operation failed: invalid selection bounds",
            "Extend operation failed: invalid direction or amount",
            "Shift operation failed: invalid direction or amount",
            "Wrap operation failed: invalid direction or amount",
            "Overlay operation failed: invalid overlay data",
            "Mask operation failed: invalid mask data",
            "Filter operation failed: invalid filter parameters",
            "Transform operation failed: invalid transformation matrix",
            "Blend operation failed: invalid blend parameters",
            "Extract operation failed: invalid extraction parameters",
            "Insert operation failed: invalid insertion parameters",
            "Replace operation failed: invalid replacement parameters",
            "Swap operation failed: invalid swap parameters",
            "Duplicate operation failed: invalid duplication parameters",
            "Remove operation failed: invalid removal parameters",
            "Resize operation failed: invalid size parameters",
            "Pad operation failed: invalid padding parameters",
            "Trim operation failed: invalid trim parameters",
            "Align operation failed: invalid alignment parameters",
            "Distribute operation failed: invalid distribution parameters",
            "Group operation failed: invalid grouping parameters",
            "Ungroup operation failed: invalid ungrouping parameters",
            "Lock operation failed: invalid lock parameters",
            "Unlock operation failed: invalid unlock parameters",
            "Freeze operation failed: invalid freeze parameters",
            "Thaw operation failed: invalid thaw parameters",
            "Save operation failed: invalid save parameters",
            "Load operation failed: invalid load parameters",
            "Undo operation failed: no operations to undo",
            "Redo operation failed: no operations to redo",
            "Reset operation failed: invalid reset parameters",
            "Submit operation failed: invalid submission state"
        ]
        
        # Basic validation checks
        has_selection = jnp.any(selection)
        valid_operation = (operation_id >= 0) & (operation_id < 42)
        
        # Check for basic operation requirements
        needs_selection_ops = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # Operations that need selection
        needs_selection = jnp.isin(operation_id, needs_selection_ops)
        
        # Determine error condition and type
        selection_error = needs_selection & (~has_selection)
        operation_error = ~valid_operation
        
        # Combine error conditions
        has_error = selection_error | operation_error
        
        # Determine error index (which error message to use)
        error_index = jnp.where(
            operation_error,
            0,  # Use first error message for invalid operations
            jnp.clip(operation_id, 0, len(error_messages) - 1)
        )
        
        # Use branched error for specific operation error messages
        return eqx.branched_error_if(
            state,
            has_error,
            error_index,
            error_messages
        )
    
    @staticmethod
    @eqx.filter_jit
    def validate_state_consistency(state: ArcEnvState) -> ArcEnvState:
        """Validate state consistency with detailed error messages.
        
        This function performs comprehensive state validation to ensure
        all state fields are consistent and valid.
        
        Args:
            state: Environment state to validate
            
        Returns:
            Validated state (same as input if valid)
            
        Raises:
            RuntimeError: If state is inconsistent (via equinox.error_if)
        """
        # Check grid shape consistency
        working_shape = state.working_grid.shape
        target_shape = state.target_grid.shape
        shapes_match = jnp.array_equal(jnp.array(working_shape), jnp.array(target_shape))
        
        state = eqx.error_if(
            state,
            ~shapes_match,
            "Working grid and target grid shapes must match"
        )
        
        # Check mask consistency
        mask_shape = state.working_grid_mask.shape
        grid_mask_matches = jnp.array_equal(jnp.array(working_shape), jnp.array(mask_shape))
        
        state = eqx.error_if(
            state,
            ~grid_mask_matches,
            "Working grid mask shape must match working grid"
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
        valid_modes = jnp.array([0, 1])  # 0=train, 1=test
        mode_valid = jnp.isin(state.episode_mode, valid_modes)
        
        state = eqx.error_if(
            state,
            ~mode_valid,
            "Episode mode must be 0 (train) or 1 (test)"
        )
        
        return state
    
    @staticmethod
    def validate_batch_actions(
        actions: StructuredAction,
        config: JaxArcConfig,
        batch_size: int
    ) -> StructuredAction:
        """Validate batch of structured actions.
        
        This function validates a batch of actions and provides clear
        error messages with batch indices for debugging.
        
        Args:
            actions: Batch of structured actions
            config: Environment configuration
            batch_size: Expected batch size
            
        Returns:
            Validated batch of actions
            
        Note:
            This function is not JIT-compiled to allow for better error reporting
            with batch indices in error messages.
        """
        try:
            # Use vmap to validate each action in the batch
            validate_fn = lambda action: JAXErrorHandler.validate_action(action, config)
            validated_actions = jax.vmap(validate_fn)(actions)
            return validated_actions
            
        except Exception as e:
            # Provide enhanced error message with batch context
            error_msg = f"Batch action validation failed (batch_size={batch_size}): {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    @staticmethod
    def get_error_mode() -> ErrorMode:
        """Get current error handling mode from environment variable.
        
        Returns:
            Current error mode ("raise", "nan", or "breakpoint")
        """
        mode = os.environ.get("EQX_ON_ERROR", "raise").lower()
        if mode in ["raise", "nan", "breakpoint"]:
            return mode
        else:
            logger.warning(f"Unknown error mode '{mode}', defaulting to 'raise'")
            return "raise"
    
    @staticmethod
    def set_error_mode(mode: ErrorMode) -> None:
        """Set error handling mode via environment variable.
        
        Args:
            mode: Error mode to set ("raise", "nan", or "breakpoint")
        """
        if mode not in ["raise", "nan", "breakpoint"]:
            raise ValueError(f"Invalid error mode: {mode}. Must be 'raise', 'nan', or 'breakpoint'")
        
        os.environ["EQX_ON_ERROR"] = mode
        logger.info(f"Error handling mode set to: {mode}")
    
    @staticmethod
    def configure_debugging(
        mode: ErrorMode = "raise",
        breakpoint_frames: int = 3,
        enable_nan_checks: bool = True
    ) -> None:
        """Configure debugging environment for error handling.
        
        Args:
            mode: Error handling mode
            breakpoint_frames: Number of frames to capture for breakpoints
            enable_nan_checks: Whether to enable NaN checking
        """
        JAXErrorHandler.set_error_mode(mode)
        
        # Set breakpoint frame count
        os.environ["EQX_ON_ERROR_BREAKPOINT_FRAMES"] = str(breakpoint_frames)
        
        # Configure NaN checking if requested
        if enable_nan_checks:
            os.environ["JAX_DEBUG_NANS"] = "True"
        
        logger.info(f"Debugging configured: mode={mode}, frames={breakpoint_frames}, nan_checks={enable_nan_checks}")


# Utility functions for common error patterns

@eqx.filter_jit
def assert_positive(value: jnp.ndarray, name: str = "value") -> jnp.ndarray:
    """Assert that a value is positive using JAX-compatible error checking.
    
    Args:
        value: Value to check
        name: Name of the value for error message
        
    Returns:
        The value if positive
        
    Raises:
        RuntimeError: If value is not positive
    """
    return eqx.error_if(
        value,
        value <= 0,
        f"{name} must be positive"
    )


@eqx.filter_jit
def assert_in_range(
    value: jnp.ndarray, 
    min_val: float, 
    max_val: float, 
    name: str = "value"
) -> jnp.ndarray:
    """Assert that a value is within a specified range.
    
    Args:
        value: Value to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error message
        
    Returns:
        The value if in range
        
    Raises:
        RuntimeError: If value is out of range
    """
    return eqx.error_if(
        value,
        (value < min_val) | (value > max_val),
        f"{name} must be in range [{min_val}, {max_val}]"
    )


@eqx.filter_jit
def assert_shape_matches(
    array: jnp.ndarray, 
    expected_shape: tuple[int, ...], 
    name: str = "array"
) -> jnp.ndarray:
    """Assert that an array has the expected shape.
    
    Args:
        array: Array to check
        expected_shape: Expected shape
        name: Name of the array for error message
        
    Returns:
        The array if shape matches
        
    Raises:
        RuntimeError: If shape doesn't match
    """
    actual_shape = array.shape
    
    # Convert to JAX arrays for comparison
    actual_shape_array = jnp.array(actual_shape)
    expected_shape_array = jnp.array(expected_shape)
    
    # Check if shapes match exactly
    shapes_match = jnp.array_equal(actual_shape_array, expected_shape_array)
    
    return eqx.error_if(
        array,
        ~shapes_match,
        f"{name} shape mismatch: expected {expected_shape}"
    )