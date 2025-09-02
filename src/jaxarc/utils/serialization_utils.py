"""Essential serialization utilities for JaxARC.

This module provides core serialization functions for converting JAX arrays,
actions, and logging data to JSON-serializable formats.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from loguru import logger

from jaxarc.state import State


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
    """Serialize action for logging purposes.

    Args:
        action: Action dictionary or structured action to serialize

    Returns:
        Dictionary with serialized action data
    """
    try:
        # Handle structured actions (Equinox modules)
        if hasattr(action, "__dict__") and not isinstance(action, dict):
            serialized = {"type": type(action).__name__}

            # Extract common action fields
            if hasattr(action, "operation"):
                serialized["operation"] = serialize_jax_array(action.operation)
            if hasattr(action, "selection"):
                serialized["selection"] = serialize_jax_array(action.selection)

            return serialized

        # Handle dictionary actions
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

        # Fallback for other types
        return {"raw": str(action), "type": type(action).__name__}

    except Exception as e:
        logger.warning(f"Failed to serialize action: {e}")
        return {"error": str(e), "type": type(action).__name__}


def serialize_object(obj: Any) -> Any:
    """Convert objects to JSON-serializable formats.

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

    if hasattr(obj, "__array__"):
        try:
            return np.asarray(obj).tolist()
        except Exception:
            pass

    # Fallback to string representation
    return str(obj)


def serialize_state(state: State) -> dict[str, Any]:
    """Serialize State for logging.

    Args:
        state: Environment state to serialize

    Returns:
        Dictionary with serialized state data
    """
    try:
        return {
            # Core grid data
            "working_grid": serialize_jax_array(state.working_grid),
            "target_grid": serialize_jax_array(state.target_grid),
            "selected": serialize_jax_array(state.selected),
            "clipboard": serialize_jax_array(state.clipboard),
            # Progress tracking
            "step_count": int(state.step_count),
            "task_idx": int(state.task_idx),
            "pair_idx": int(state.pair_idx),
            "similarity_score": float(state.similarity_score),
            "type": type(state).__name__,
        }
    except Exception as e:
        logger.warning(f"Failed to serialize state: {e}")
        return {"error": str(e), "type": type(state).__name__}


def serialize_log_step(step_data: dict[str, Any]) -> dict[str, Any]:
    """Convert step data to serializable format.

    Args:
        step_data: Raw step data with potential JAX arrays

    Returns:
        Serialized step data suitable for JSON storage
    """
    serialized = {}

    for key, value in step_data.items():
        if key in ["before_state", "after_state"]:
            serialized[key] = serialize_state(value)
        elif key == "action":
            serialized[key] = serialize_action(value)
        elif key == "info":
            # Preserve info structure while serializing contents
            if isinstance(value, dict):
                serialized[key] = {
                    str(k): serialize_object(v) for k, v in value.items()
                }
            else:
                serialized[key] = serialize_object(value)
        else:
            serialized[key] = serialize_object(value)

    return serialized
