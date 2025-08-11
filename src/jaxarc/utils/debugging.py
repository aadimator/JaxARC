"""
Debugging utilities for JAX-compatible error handling and diagnosis.

This module provides debugging support that works within JAX transformations,
including support for EQX_ON_ERROR environment variable modes, batch processing
error diagnosis, and frame capture configuration.

Key Features:
- EQX_ON_ERROR=breakpoint debugging mode support
- EQX_ON_ERROR=nan graceful degradation mode
- Batch processing error diagnosis utilities
- Frame capture configuration for debugging
- JAX-compatible debugging callbacks
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from loguru import logger

from ..envs.actions import StructuredAction
from ..envs.config import JaxArcConfig
from ..state import ArcEnvState
from ..utils.jax_types import GridArray, SelectionArray


class DebugConfig:
    """Configuration for debugging behavior.
    
    This class manages debugging configuration including error modes,
    frame capture settings, and batch processing diagnosis options.
    """
    
    def __init__(
        self,
        error_mode: str = "raise",
        breakpoint_frames: int = 3,
        enable_nan_checks: bool = True,
        capture_intermediate_states: bool = False,
        log_batch_errors: bool = True,
        max_error_context: int = 5
    ):
        """Initialize debug configuration.
        
        Args:
            error_mode: Error handling mode ("raise", "nan", "breakpoint")
            breakpoint_frames: Number of frames to capture for breakpoints
            enable_nan_checks: Whether to enable NaN checking
            capture_intermediate_states: Whether to capture intermediate states
            log_batch_errors: Whether to log batch processing errors
            max_error_context: Maximum number of error contexts to capture
        """
        self.error_mode = error_mode
        self.breakpoint_frames = breakpoint_frames
        self.enable_nan_checks = enable_nan_checks
        self.capture_intermediate_states = capture_intermediate_states
        self.log_batch_errors = log_batch_errors
        self.max_error_context = max_error_context
        
        # Apply configuration to environment
        self._apply_to_environment()
    
    def _apply_to_environment(self) -> None:
        """Apply debug configuration to environment variables."""
        os.environ["EQX_ON_ERROR"] = self.error_mode
        os.environ["EQX_ON_ERROR_BREAKPOINT_FRAMES"] = str(self.breakpoint_frames)
        
        if self.enable_nan_checks:
            os.environ["JAX_DEBUG_NANS"] = "True"
        else:
            os.environ.pop("JAX_DEBUG_NANS", None)
        
        logger.debug(f"Applied debug configuration: {self}")
    
    def __repr__(self) -> str:
        return (
            f"DebugConfig(error_mode={self.error_mode}, "
            f"breakpoint_frames={self.breakpoint_frames}, "
            f"enable_nan_checks={self.enable_nan_checks})"
        )


class BatchErrorDiagnostics:
    """Utilities for diagnosing errors in batch processing.
    
    This class provides tools for identifying and debugging errors that
    occur during batch processing of environments or actions.
    """
    
    @staticmethod
    def diagnose_batch_action_errors(
        actions: StructuredAction,
        config: JaxArcConfig,
        batch_size: int
    ) -> Dict[str, Any]:
        """Diagnose errors in a batch of actions.
        
        Args:
            actions: Batch of structured actions
            config: Environment configuration
            batch_size: Expected batch size
            
        Returns:
            Dictionary containing diagnostic information
        """
        diagnostics = {
            "batch_size": batch_size,
            "action_type": type(actions).__name__,
            "errors": [],
            "valid_actions": 0,
            "invalid_actions": 0,
            "error_indices": [],
            "error_details": []
        }
        
        try:
            # Check each action individually to identify problematic ones
            for i in range(batch_size):
                try:
                    # Extract single action from batch
                    if hasattr(actions, 'operation'):
                        if actions.operation.ndim > 0:
                            single_action = jax.tree_map(lambda x: x[i], actions)
                        else:
                            single_action = actions
                    else:
                        single_action = actions
                    
                    # Validate single action
                    from .error_handling import JAXErrorHandler
                    JAXErrorHandler.validate_action(single_action, config)
                    diagnostics["valid_actions"] += 1
                    
                except Exception as e:
                    diagnostics["invalid_actions"] += 1
                    diagnostics["error_indices"].append(i)
                    diagnostics["error_details"].append({
                        "index": i,
                        "error": str(e),
                        "action_data": BatchErrorDiagnostics._extract_action_data(single_action)
                    })
                    
        except Exception as e:
            diagnostics["errors"].append(f"Batch diagnosis failed: {str(e)}")
        
        return diagnostics
    
    @staticmethod
    def _extract_action_data(action: StructuredAction) -> Dict[str, Any]:
        """Extract action data for error reporting.
        
        Args:
            action: Structured action to extract data from
            
        Returns:
            Dictionary containing action data
        """
        try:
            data = {"operation": int(action.operation)}
            
            if hasattr(action, 'row'):
                data.update({"row": int(action.row), "col": int(action.col)})
            elif hasattr(action, 'r1'):
                data.update({
                    "r1": int(action.r1), "c1": int(action.c1),
                    "r2": int(action.r2), "c2": int(action.c2)
                })
            elif hasattr(action, 'selection'):
                data.update({
                    "selection_shape": action.selection.shape,
                    "selection_sum": int(jnp.sum(action.selection))
                })
            
            return data
            
        except Exception as e:
            return {"extraction_error": str(e)}
    
    @staticmethod
    def diagnose_batch_state_errors(
        states: ArcEnvState,
        batch_size: int
    ) -> Dict[str, Any]:
        """Diagnose errors in a batch of environment states.
        
        Args:
            states: Batch of environment states
            batch_size: Expected batch size
            
        Returns:
            Dictionary containing diagnostic information
        """
        diagnostics = {
            "batch_size": batch_size,
            "errors": [],
            "valid_states": 0,
            "invalid_states": 0,
            "error_indices": [],
            "error_details": []
        }
        
        try:
            # Check each state individually
            for i in range(batch_size):
                try:
                    # Extract single state from batch
                    single_state = jax.tree_map(lambda x: x[i] if x.ndim > 0 else x, states)
                    
                    # Validate single state
                    from .error_handling import JAXErrorHandler
                    JAXErrorHandler.validate_state_consistency(single_state)
                    diagnostics["valid_states"] += 1
                    
                except Exception as e:
                    diagnostics["invalid_states"] += 1
                    diagnostics["error_indices"].append(i)
                    diagnostics["error_details"].append({
                        "index": i,
                        "error": str(e),
                        "state_summary": BatchErrorDiagnostics._extract_state_summary(single_state)
                    })
                    
        except Exception as e:
            diagnostics["errors"].append(f"Batch state diagnosis failed: {str(e)}")
        
        return diagnostics
    
    @staticmethod
    def _extract_state_summary(state: ArcEnvState) -> Dict[str, Any]:
        """Extract state summary for error reporting.
        
        Args:
            state: Environment state to summarize
            
        Returns:
            Dictionary containing state summary
        """
        try:
            return {
                "step_count": int(state.step_count),
                "episode_done": bool(state.episode_done),
                "similarity_score": float(state.similarity_score),
                "episode_mode": int(state.episode_mode),
                "working_grid_shape": state.working_grid.shape,
                "target_grid_shape": state.target_grid.shape
            }
        except Exception as e:
            return {"extraction_error": str(e)}


class DebugCallbacks:
    """JAX-compatible debugging callbacks.
    
    This class provides debugging callbacks that work within JAX transformations
    using jax.debug.callback for logging and state inspection.
    """
    
    @staticmethod
    def log_action_debug(action: StructuredAction, step: int) -> None:
        """Log action debug information using JAX debug callback.
        
        Args:
            action: Structured action to log
            step: Current step number
        """
        def _log_callback(action_data, step_data):
            logger.debug(f"Step {step_data}: Action {action_data}")
        
        # Extract action data for logging
        action_data = {
            "operation": action.operation,
            "type": type(action).__name__
        }
        
        if hasattr(action, 'row'):
            action_data.update({"row": action.row, "col": action.col})
        elif hasattr(action, 'r1'):
            action_data.update({
                "r1": action.r1, "c1": action.c1,
                "r2": action.r2, "c2": action.c2
            })
        
        jax.debug.callback(_log_callback, action_data, step)
    
    @staticmethod
    def log_state_debug(state: ArcEnvState, step: int) -> None:
        """Log state debug information using JAX debug callback.
        
        Args:
            state: Environment state to log
            step: Current step number
        """
        def _log_callback(state_data, step_data):
            logger.debug(f"Step {step_data}: State {state_data}")
        
        # Extract state data for logging
        state_data = {
            "step_count": state.step_count,
            "similarity_score": state.similarity_score,
            "episode_done": state.episode_done,
            "episode_mode": state.episode_mode
        }
        
        jax.debug.callback(_log_callback, state_data, step)
    
    @staticmethod
    def log_error_context(
        error_msg: str,
        context: Dict[str, Any]
    ) -> None:
        """Log error context using JAX debug callback.
        
        Args:
            error_msg: Error message
            context: Additional context information
        """
        def _log_callback(msg, ctx):
            logger.error(f"Error: {msg}, Context: {ctx}")
        
        jax.debug.callback(_log_callback, error_msg, context)


class FrameCapture:
    """Frame capture utilities for debugging.
    
    This class provides utilities for capturing and analyzing stack frames
    during error conditions, particularly useful for breakpoint debugging.
    """
    
    @staticmethod
    def capture_frames(max_frames: int = 10) -> List[Dict[str, Any]]:
        """Capture current stack frames for debugging.
        
        Args:
            max_frames: Maximum number of frames to capture
            
        Returns:
            List of frame information dictionaries
        """
        frames = []
        
        try:
            # Get current stack
            stack = traceback.extract_stack()
            
            # Capture up to max_frames, excluding this function
            for frame in stack[:-1][-max_frames:]:
                frames.append({
                    "filename": frame.filename,
                    "lineno": frame.lineno,
                    "name": frame.name,
                    "line": frame.line
                })
                
        except Exception as e:
            logger.warning(f"Failed to capture frames: {e}")
            frames.append({"error": str(e)})
        
        return frames
    
    @staticmethod
    def format_frames(frames: List[Dict[str, Any]]) -> str:
        """Format captured frames for display.
        
        Args:
            frames: List of frame information dictionaries
            
        Returns:
            Formatted string representation of frames
        """
        if not frames:
            return "No frames captured"
        
        formatted = ["Stack trace:"]
        for i, frame in enumerate(frames):
            if "error" in frame:
                formatted.append(f"  Frame {i}: {frame['error']}")
            else:
                formatted.append(
                    f"  Frame {i}: {frame['filename']}:{frame['lineno']} "
                    f"in {frame['name']}() - {frame['line']}"
                )
        
        return "\n".join(formatted)


class InteractiveDebugger:
    """Interactive debugging utilities.
    
    This class provides utilities for interactive debugging sessions,
    particularly useful when EQX_ON_ERROR=breakpoint is set.
    """
    
    @staticmethod
    def setup_breakpoint_debugging() -> None:
        """Setup environment for breakpoint debugging."""
        os.environ["EQX_ON_ERROR"] = "breakpoint"
        os.environ["EQX_ON_ERROR_BREAKPOINT_FRAMES"] = "5"
        
        logger.info("Breakpoint debugging enabled. Errors will trigger interactive debugger.")
    
    @staticmethod
    def setup_nan_debugging() -> None:
        """Setup environment for NaN debugging."""
        os.environ["EQX_ON_ERROR"] = "nan"
        os.environ["JAX_DEBUG_NANS"] = "True"
        
        logger.info("NaN debugging enabled. Errors will return NaN values for graceful degradation.")
    
    @staticmethod
    def inspect_action(action: StructuredAction) -> Dict[str, Any]:
        """Inspect structured action for debugging.
        
        Args:
            action: Structured action to inspect
            
        Returns:
            Dictionary containing action inspection data
        """
        inspection = {
            "type": type(action).__name__,
            "operation": int(action.operation),
            "valid": True,
            "issues": []
        }
        
        try:
            # Check operation bounds
            if action.operation < 0 or action.operation >= 42:
                inspection["valid"] = False
                inspection["issues"].append(f"Invalid operation: {action.operation}")
            
            # Type-specific inspection
            if hasattr(action, 'row'):
                inspection.update({
                    "row": int(action.row),
                    "col": int(action.col)
                })
                if action.row < 0 or action.col < 0:
                    inspection["valid"] = False
                    inspection["issues"].append("Negative coordinates")
                    
            elif hasattr(action, 'r1'):
                inspection.update({
                    "r1": int(action.r1), "c1": int(action.c1),
                    "r2": int(action.r2), "c2": int(action.c2)
                })
                if any(coord < 0 for coord in [action.r1, action.c1, action.r2, action.c2]):
                    inspection["valid"] = False
                    inspection["issues"].append("Negative coordinates")
                    
            elif hasattr(action, 'selection'):
                inspection.update({
                    "selection_shape": action.selection.shape,
                    "selection_sum": int(jnp.sum(action.selection)),
                    "selection_dtype": str(action.selection.dtype)
                })
                if action.selection.dtype != jnp.bool_:
                    inspection["valid"] = False
                    inspection["issues"].append(f"Invalid selection dtype: {action.selection.dtype}")
            
        except Exception as e:
            inspection["valid"] = False
            inspection["issues"].append(f"Inspection error: {str(e)}")
        
        return inspection
    
    @staticmethod
    def inspect_state(state: ArcEnvState) -> Dict[str, Any]:
        """Inspect environment state for debugging.
        
        Args:
            state: Environment state to inspect
            
        Returns:
            Dictionary containing state inspection data
        """
        inspection = {
            "valid": True,
            "issues": [],
            "summary": {}
        }
        
        try:
            # Basic state information
            inspection["summary"] = {
                "step_count": int(state.step_count),
                "episode_done": bool(state.episode_done),
                "similarity_score": float(state.similarity_score),
                "episode_mode": int(state.episode_mode),
                "working_grid_shape": state.working_grid.shape,
                "target_grid_shape": state.target_grid.shape
            }
            
            # Validation checks
            if state.step_count < 0:
                inspection["valid"] = False
                inspection["issues"].append("Negative step count")
            
            if not (0.0 <= state.similarity_score <= 1.0):
                inspection["valid"] = False
                inspection["issues"].append(f"Invalid similarity score: {state.similarity_score}")
            
            if state.episode_mode not in [0, 1]:
                inspection["valid"] = False
                inspection["issues"].append(f"Invalid episode mode: {state.episode_mode}")
            
            # Shape consistency checks
            if state.working_grid.shape != state.target_grid.shape:
                inspection["valid"] = False
                inspection["issues"].append("Grid shape mismatch")
            
            if state.working_grid_mask.shape != state.working_grid.shape:
                inspection["valid"] = False
                inspection["issues"].append("Mask shape mismatch")
                
        except Exception as e:
            inspection["valid"] = False
            inspection["issues"].append(f"Inspection error: {str(e)}")
        
        return inspection


# Global debug configuration instance
_debug_config: Optional[DebugConfig] = None


def get_debug_config() -> DebugConfig:
    """Get global debug configuration instance.
    
    Returns:
        Global debug configuration instance
    """
    global _debug_config
    if _debug_config is None:
        _debug_config = DebugConfig()
    return _debug_config


def configure_debugging(
    error_mode: str = "raise",
    breakpoint_frames: int = 3,
    enable_nan_checks: bool = True,
    **kwargs
) -> DebugConfig:
    """Configure global debugging settings.
    
    Args:
        error_mode: Error handling mode ("raise", "nan", "breakpoint")
        breakpoint_frames: Number of frames to capture for breakpoints
        enable_nan_checks: Whether to enable NaN checking
        **kwargs: Additional debug configuration options
        
    Returns:
        Configured debug configuration instance
    """
    global _debug_config
    _debug_config = DebugConfig(
        error_mode=error_mode,
        breakpoint_frames=breakpoint_frames,
        enable_nan_checks=enable_nan_checks,
        **kwargs
    )
    return _debug_config


def reset_debugging() -> None:
    """Reset debugging configuration to defaults."""
    global _debug_config
    _debug_config = None
    
    # Clear environment variables
    for var in ["EQX_ON_ERROR", "EQX_ON_ERROR_BREAKPOINT_FRAMES", "JAX_DEBUG_NANS"]:
        os.environ.pop(var, None)
    
    logger.info("Debugging configuration reset to defaults")