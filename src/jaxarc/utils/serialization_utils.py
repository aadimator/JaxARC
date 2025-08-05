"""Shared serialization utilities for JAX arrays and environment data.

This module provides reusable serialization functions that can be used across
the codebase for converting JAX arrays, environment states, and actions to
JSON-serializable formats.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from loguru import logger

from .task_manager import get_global_task_manager


def serialize_jax_array(arr: jnp.ndarray | np.ndarray) -> np.ndarray:
    """Safely serialize JAX array to numpy array.

    Args:
        arr: JAX or numpy array to serialize

    Returns:
        NumPy array copy of the input
    """
    try:
        if isinstance(arr, jnp.ndarray):
            return np.asarray(arr)
        if isinstance(arr, np.ndarray):
            return arr.copy()
        return np.asarray(arr)
    except Exception as e:
        logger.warning(f"Failed to serialize array: {e}")
        return np.array([])


def serialize_action(action: dict[str, Any] | Any) -> dict[str, Any]:
    """Serialize action (dictionary or structured action) for safe callback usage.

    Args:
        action: Action dictionary or structured action to serialize

    Returns:
        Dictionary with serialized action data
    """
    try:
        # Handle structured actions (Equinox modules)
        if hasattr(action, '__dict__') and not isinstance(action, dict):
            # This is a structured action - extract fields
            serialized = {}
            
            # Get all fields from the structured action
            if hasattr(action, 'operation'):
                serialized['operation'] = serialize_jax_array(action.operation)
            
            # Handle different action types
            if hasattr(action, 'row') and hasattr(action, 'col'):
                # PointAction
                serialized['action_type'] = 'point'
                serialized['row'] = serialize_jax_array(action.row)
                serialized['col'] = serialize_jax_array(action.col)
            elif hasattr(action, 'r1') and hasattr(action, 'c1') and hasattr(action, 'r2') and hasattr(action, 'c2'):
                # BboxAction
                serialized['action_type'] = 'bbox'
                serialized['r1'] = serialize_jax_array(action.r1)
                serialized['c1'] = serialize_jax_array(action.c1)
                serialized['r2'] = serialize_jax_array(action.r2)
                serialized['c2'] = serialize_jax_array(action.c2)
            elif hasattr(action, 'selection'):
                # MaskAction
                serialized['action_type'] = 'mask'
                serialized['selection'] = serialize_jax_array(action.selection)
            else:
                # Unknown structured action type
                serialized['action_type'] = 'unknown'
                serialized['raw'] = str(action)
            
            return serialized
        
        # Handle dictionary actions (legacy)
        elif isinstance(action, dict):
            serialized = {}
            for key, value in action.items():
                if isinstance(value, (jnp.ndarray, np.ndarray)):
                    serialized[key] = serialize_jax_array(value)
                elif isinstance(value, (int, float, bool, str)):
                    serialized[key] = value
                else:
                    serialized[key] = str(value)
            return serialized
        
        # Handle other types
        else:
            return {'raw': str(action), 'type': type(action).__name__}
            
    except Exception as e:
        logger.warning(f"Failed to serialize action: {e}")
        return {'error': str(e), 'type': type(action).__name__}


def serialize_arc_state(state: "ArcEnvState") -> dict[str, Any]:
    """Serialize ArcEnvState for safe callback usage.

    Args:
        state: Environment state to serialize

    Returns:
        Dictionary with serialized state data
    """
    try:
        return {
            # Core grid data
            "working_grid": serialize_jax_array(state.working_grid),
            "working_grid_mask": serialize_jax_array(state.working_grid_mask),
            "target_grid": serialize_jax_array(state.target_grid),
            "target_grid_mask": serialize_jax_array(state.target_grid_mask),
            
            # Episode management
            "step_count": int(state.step_count),
            "episode_done": bool(state.episode_done),
            "current_example_idx": int(state.current_example_idx),
            
            # Grid operations
            "selected": serialize_jax_array(state.selected),
            "clipboard": serialize_jax_array(state.clipboard),
            "similarity_score": float(state.similarity_score),
            
            # Enhanced functionality fields
            "episode_mode": int(state.episode_mode),
            "available_demo_pairs": serialize_jax_array(state.available_demo_pairs),
            "available_test_pairs": serialize_jax_array(state.available_test_pairs),
            "demo_completion_status": serialize_jax_array(state.demo_completion_status),
            "test_completion_status": serialize_jax_array(state.test_completion_status),
            "action_history": serialize_jax_array(state.action_history),
            "action_history_length": int(state.action_history_length),
            "allowed_operations_mask": serialize_jax_array(state.allowed_operations_mask),
        }
    except Exception as e:
        logger.warning(f"Failed to serialize ArcEnvState: {e}")
        return {}


def serialize_object(obj: Any) -> Any:
    """Recursively serialize objects for JSON compatibility.
    
    This method handles JAX arrays, numpy arrays, and other complex objects
    by converting them to JSON-serializable formats.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable representation
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    
    if isinstance(obj, (list, tuple)):
        return [serialize_object(item) for item in obj]
    
    if isinstance(obj, dict):
        return {str(k): serialize_object(v) for k, v in obj.items()}
    
    # Handle JAX/NumPy arrays
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    
    # Handle JAX arrays that might not have tolist
    if hasattr(obj, '__array__'):
        try:
            return np.asarray(obj).tolist()
        except Exception:
            pass
    
    # Handle objects with __dict__
    if hasattr(obj, '__dict__'):
        return serialize_object(obj.__dict__)
    
    # Fallback to string representation
    return str(obj)


# Note: extract_task_id_from_index has been moved to task_manager.py
# Import it from there: from jaxarc.utils.task_manager import extract_task_id_from_index