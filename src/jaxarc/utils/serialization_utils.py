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

from jaxarc.state import ArcEnvState

from .pytree import filter_arrays_from_state
from .task_manager import extract_task_id_from_index


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
        if hasattr(action, "__dict__") and not isinstance(action, dict):
            # This is a structured action - extract fields
            serialized = {}

            # Get all fields from the structured action
            if hasattr(action, "operation"):
                serialized["operation"] = serialize_jax_array(action.operation)

            # Handle different action types
            if hasattr(action, "row") and hasattr(action, "col"):
                # PointAction
                serialized["action_type"] = "point"
                serialized["row"] = serialize_jax_array(action.row)
                serialized["col"] = serialize_jax_array(action.col)
            elif (
                hasattr(action, "r1")
                and hasattr(action, "c1")
                and hasattr(action, "r2")
                and hasattr(action, "c2")
            ):
                # BboxAction
                serialized["action_type"] = "bbox"
                serialized["r1"] = serialize_jax_array(action.r1)
                serialized["c1"] = serialize_jax_array(action.c1)
                serialized["r2"] = serialize_jax_array(action.r2)
                serialized["c2"] = serialize_jax_array(action.c2)
            elif hasattr(action, "selection"):
                # MaskAction
                serialized["action_type"] = "mask"
                serialized["selection"] = serialize_jax_array(action.selection)
            else:
                # Unknown structured action type
                serialized["action_type"] = "unknown"
                serialized["raw"] = str(action)

            return serialized

        # Handle dictionary actions (legacy)
        if isinstance(action, dict):
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
        return {"raw": str(action), "type": type(action).__name__}

    except Exception as e:
        logger.warning(f"Failed to serialize action: {e}")
        return {"error": str(e), "type": type(action).__name__}


def serialize_arc_state(state: ArcEnvState) -> dict[str, Any]:
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
            "allowed_operations_mask": serialize_jax_array(
                state.allowed_operations_mask
            ),
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
    if hasattr(obj, "tolist"):
        return obj.tolist()

    # Handle JAX arrays that might not have tolist
    if hasattr(obj, "__array__"):
        try:
            return np.asarray(obj).tolist()
        except Exception:
            pass

    # Handle objects with __dict__
    if hasattr(obj, "__dict__"):
        return serialize_object(obj.__dict__)

    # Fallback to string representation
    return str(obj)


def serialize_log_step(step_data: dict[str, Any]) -> dict[str, Any]:
    """Convert step data to serializable format using existing utilities.

    Args:
        step_data: Raw step data with potential JAX arrays

    Returns:
        Serialized step data suitable for JSON/pickle storage
    """
    serialized = {}

    for key, value in step_data.items():
        if key in ["before_state", "after_state"]:
            # Handle state objects using existing utilities
            serialized[key] = serialize_log_state(value)
        elif key == "action":
            # Use shared action serialization utility
            serialized[key] = serialize_action(value)
        elif key == "info":
            # Info dictionary may contain complex data
            serialized[key] = serialize_info_dict(value)
        else:
            # Handle other fields (step_num, reward, etc.)
            serialized[key] = serialize_object(value)

    return serialized


def serialize_log_state(state: Any) -> dict[str, Any]:
    """Serialize state objects using existing utilities.

    Args:
        state: ArcEnvState or similar state object

    Returns:
        Serialized state representation
    """
    if state is None:
        return {"type": "None"}

    try:
        # Try to use existing pytree utilities for real ArcEnvState objects
        if hasattr(state, "__class__") and "ArcEnvState" in state.__class__.__name__:
            arrays, non_arrays = filter_arrays_from_state(state)

            # Convert arrays to lists for JSON serialization
            serialized_arrays = {}
            for field_name in ["working_grid", "target_grid", "selected", "clipboard"]:
                if hasattr(arrays, field_name):
                    array_value = getattr(arrays, field_name)
                    serialized_arrays[field_name] = serialize_jax_array(array_value)
        else:
            # For mock states or other objects, serialize arrays directly
            serialized_arrays = {}
            for field_name in ["working_grid", "target_grid", "selected", "clipboard"]:
                if hasattr(state, field_name):
                    array_value = getattr(state, field_name)
                    serialized_arrays[field_name] = serialize_jax_array(array_value)

        # Extract task information using existing utilities
        task_info = {}
        if hasattr(state, "task_data") and hasattr(state.task_data, "task_index"):
            try:
                # Use the canonical extract_task_id_from_index that takes JAX arrays
                task_id = extract_task_id_from_index(state.task_data.task_index)
                task_info["task_id"] = (
                    task_id or f"task_{int(state.task_data.task_index.item())}"
                )
                task_info["task_index"] = int(state.task_data.task_index.item())
            except Exception as e:
                logger.debug(f"Could not extract task info: {e}")
                task_info["task_index"] = -1

        # Combine serialized data
        return {
            "type": type(state).__name__,
            "arrays": serialized_arrays,
            "task_info": task_info,
            "step_count": getattr(state, "step_count", 0),
            "episode_done": getattr(state, "episode_done", False),
            "similarity_score": getattr(state, "similarity_score", 0.0),
            "current_example_idx": getattr(state, "current_example_idx", 0),
        }

    except Exception as e:
        logger.warning(f"Failed to serialize state using utilities: {e}")
        # Fallback to basic serialization
        return {
            "type": type(state).__name__,
            "serialization_error": str(e),
            "step_count": getattr(state, "step_count", 0)
            if hasattr(state, "step_count")
            else 0,
        }


def serialize_info_dict(info: dict[str, Any]) -> dict[str, Any]:
    """Serialize info dictionary, preserving metrics structure for automatic extraction.

    This method ensures that the entire info dictionary is serialized while
    maintaining the special structure of info['metrics'] for automatic metric
    extraction by other handlers.

    Args:
        info: Info dictionary from step data containing metrics and other data

    Returns:
        Serialized info dictionary with preserved metrics structure
    """
    if not isinstance(info, dict):
        return {"serialized_value": str(info)}

    serialized = {}

    # Preserve metrics structure for automatic extraction by other handlers
    if "metrics" in info:
        # Serialize metrics dictionary while preserving structure
        serialized["metrics"] = serialize_object(info["metrics"])

    # Serialize all other info fields (entire dictionary approach)
    for key, value in info.items():
        if key != "metrics":  # Already handled above
            serialized[key] = serialize_object(value)

    return serialized
