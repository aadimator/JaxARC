"""
Task ID Management Utilities for JAX Compatibility.

This module provides utilities to manage the mapping between string task IDs
and integer task indices, enabling JAX-compatible state while preserving
task identification capabilities.

The core issue: JAX requires all state to be composed of arrays, but task IDs
are strings. This module solves this by maintaining a bidirectional mapping
between string IDs and integer indices outside of JAX transformations.
"""

from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path
from typing import Dict, Optional, Set

import chex
import jax.numpy as jnp
from loguru import logger


class TaskIDManager:
    """
    Thread-safe manager for string task ID to integer index mappings.

    This class maintains bidirectional mappings between string task IDs
    and integer indices, enabling JAX-compatible state while preserving
    the ability to identify specific tasks.

    Features:
    - Thread-safe operations
    - Automatic index assignment
    - Bidirectional lookup
    - Persistence support
    - JAX-compatible index generation
    """

    def __init__(self):
        """Initialize the task ID manager."""
        self._lock = threading.RLock()
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}
        self._next_index: int = 0

    def register_task(self, task_id: str) -> int:
        """
        Register a task ID and get its integer index.

        If the task ID is already registered, returns the existing index.
        If it's new, assigns a new index and returns it.

        Args:
            task_id: String identifier for the task

        Returns:
            Integer index for the task (JAX-compatible)
        """
        with self._lock:
            if task_id in self._id_to_index:
                return self._id_to_index[task_id]

            # Assign new index
            index = self._next_index
            self._id_to_index[task_id] = index
            self._index_to_id[index] = task_id
            self._next_index += 1

            logger.debug(f"Registered task '{task_id}' with index {index}")
            return index

    def get_index(self, task_id: str) -> Optional[int]:
        """
        Get the integer index for a task ID.

        Args:
            task_id: String identifier for the task

        Returns:
            Integer index if found, None otherwise
        """
        with self._lock:
            return self._id_to_index.get(task_id)

    def get_task_id(self, index: int) -> Optional[str]:
        """
        Get the task ID for an integer index.

        Args:
            index: Integer index for the task

        Returns:
            String task ID if found, None otherwise
        """
        with self._lock:
            return self._index_to_id.get(index)

    def get_jax_index(self, task_id: str) -> chex.Array:
        """
        Get a JAX-compatible array containing the task index.

        Args:
            task_id: String identifier for the task

        Returns:
            JAX array with the task index (int32 scalar)

        Raises:
            ValueError: If task_id is not registered
        """
        index = self.get_index(task_id)
        if index is None:
            raise ValueError(f"Task ID '{task_id}' not registered")

        return jnp.array(index, dtype=jnp.int32)

    def has_task(self, task_id: str) -> bool:
        """Check if a task ID is registered."""
        with self._lock:
            return task_id in self._id_to_index

    def has_index(self, index: int) -> bool:
        """Check if an index is assigned."""
        with self._lock:
            return index in self._index_to_id

    def get_all_task_ids(self) -> Set[str]:
        """Get all registered task IDs."""
        with self._lock:
            return set(self._id_to_index.keys())

    def get_all_indices(self) -> Set[int]:
        """Get all assigned indices."""
        with self._lock:
            return set(self._index_to_id.keys())

    def num_tasks(self) -> int:
        """Get the number of registered tasks."""
        with self._lock:
            return len(self._id_to_index)

    def clear(self) -> None:
        """Clear all registered tasks and reset indices."""
        with self._lock:
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._next_index = 0
            logger.debug("Cleared all task registrations")

    def save_to_file(self, filepath: str | Path) -> None:
        """
        Save the task ID mappings to a file.

        Args:
            filepath: Path to save the mappings
        """
        filepath = Path(filepath)

        with self._lock:
            data = {
                "id_to_index": self._id_to_index.copy(),
                "index_to_id": {str(k): v for k, v in self._index_to_id.items()},
                "next_index": self._next_index,
            }

        try:
            if filepath.suffix.lower() == ".json":
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
            else:
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)

            logger.info(f"Saved task ID mappings to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save task ID mappings: {e}")
            raise

    def load_from_file(self, filepath: str | Path) -> None:
        """
        Load task ID mappings from a file.

        Args:
            filepath: Path to load the mappings from
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Task ID mapping file not found: {filepath}")

        try:
            if filepath.suffix.lower() == ".json":
                with open(filepath) as f:
                    data = json.load(f)
            else:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)

            with self._lock:
                self._id_to_index = data["id_to_index"].copy()
                self._index_to_id = {int(k): v for k, v in data["index_to_id"].items()}
                self._next_index = data["next_index"]

            logger.info(
                f"Loaded task ID mappings from {filepath} ({self.num_tasks()} tasks)"
            )
        except Exception as e:
            logger.error(f"Failed to load task ID mappings: {e}")
            raise

    def __repr__(self) -> str:
        with self._lock:
            return f"TaskIDManager(num_tasks={len(self._id_to_index)}, next_index={self._next_index})"


# Global task manager instance
_global_task_manager: Optional[TaskIDManager] = None
_global_manager_lock = threading.RLock()


def get_global_task_manager() -> TaskIDManager:
    """
    Get the global task ID manager instance.

    Returns:
        Global TaskIDManager instance (singleton)
    """
    global _global_task_manager

    with _global_manager_lock:
        if _global_task_manager is None:
            _global_task_manager = TaskIDManager()
            logger.debug("Created global task ID manager")
        return _global_task_manager


def set_global_task_manager(manager: TaskIDManager) -> None:
    """
    Set the global task ID manager instance.

    Args:
        manager: TaskIDManager instance to use globally
    """
    global _global_task_manager

    with _global_manager_lock:
        _global_task_manager = manager
        logger.debug("Set global task ID manager")


def register_task_globally(task_id: str) -> int:
    """
    Register a task ID globally and get its integer index.

    Convenience function for the global task manager.

    Args:
        task_id: String identifier for the task

    Returns:
        Integer index for the task (JAX-compatible)
    """
    return get_global_task_manager().register_task(task_id)


def get_task_index_globally(task_id: str) -> Optional[int]:
    """
    Get the integer index for a task ID from the global manager.

    Args:
        task_id: String identifier for the task

    Returns:
        Integer index if found, None otherwise
    """
    return get_global_task_manager().get_index(task_id)


def get_task_id_globally(index: int) -> Optional[str]:
    """
    Get the task ID for an integer index from the global manager.

    Args:
        index: Integer index for the task

    Returns:
        String task ID if found, None otherwise
    """
    return get_global_task_manager().get_task_id(index)


def get_jax_task_index(task_id: str) -> chex.Array:
    """
    Get a JAX-compatible array containing the task index.

    Convenience function for the global task manager.

    Args:
        task_id: String identifier for the task

    Returns:
        JAX array with the task index (int32 scalar)

    Raises:
        ValueError: If task_id is not registered
    """
    return get_global_task_manager().get_jax_index(task_id)


# Utility functions for creating JAX-compatible task data


def create_jax_task_index(task_id: Optional[str] = None) -> chex.Array:
    """
    Create a JAX-compatible task index.

    Args:
        task_id: String task ID to register/lookup, or None for unknown task

    Returns:
        JAX array with task index (int32 scalar)
        -1 is used for unknown/dummy tasks
    """
    if task_id is None:
        return jnp.array(-1, dtype=jnp.int32)

    index = register_task_globally(task_id)
    return jnp.array(index, dtype=jnp.int32)


def extract_task_id_from_index(task_index: chex.Array) -> Optional[str]:
    """
    Extract the original task ID from a JAX task index.

    Args:
        task_index: JAX array containing the task index

    Returns:
        String task ID if found, None for unknown tasks
    """
    index = int(task_index.item())
    if index == -1:
        return None

    return get_task_id_globally(index)


def is_dummy_task_index(task_index: chex.Array) -> bool:
    """
    Check if a task index represents a dummy/unknown task.

    Args:
        task_index: JAX array containing the task index

    Returns:
        True if this is a dummy task (-1), False otherwise
    """
    return int(task_index.item()) == -1


# Context manager for temporary task managers


class TemporaryTaskManager:
    """Context manager for using a temporary task ID manager."""

    def __init__(self, manager: Optional[TaskIDManager] = None):
        """
        Initialize with an optional task manager.

        Args:
            manager: TaskIDManager to use, or None to create a new one
        """
        self.temp_manager = manager or TaskIDManager()
        self.original_manager = None

    def __enter__(self) -> TaskIDManager:
        """Enter the context and set the temporary manager."""
        global _global_task_manager

        with _global_manager_lock:
            self.original_manager = _global_task_manager
            _global_task_manager = self.temp_manager

        return self.temp_manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore the original manager."""
        global _global_task_manager

        with _global_manager_lock:
            _global_task_manager = self.original_manager


# Example usage and testing utilities


def example_usage():
    """Example of how to use the task ID management system."""
    # Create a task manager
    manager = TaskIDManager()

    # Register some tasks
    index1 = manager.register_task("task_001")
    index2 = manager.register_task("task_002")
    index3 = manager.register_task("task_001")  # Same task, same index

    print(f"Task indices: {index1}, {index2}, {index3}")
    print(f"Index1 == Index3: {index1 == index3}")

    # Get JAX-compatible arrays
    jax_index1 = manager.get_jax_index("task_001")
    jax_index2 = manager.get_jax_index("task_002")

    print(f"JAX indices: {jax_index1}, {jax_index2}")

    # Reverse lookup
    task_id1 = manager.get_task_id(index1)
    task_id2 = manager.get_task_id(index2)

    print(f"Task IDs: {task_id1}, {task_id2}")

    # Global manager usage
    global_index = register_task_globally("global_task_001")
    global_task_id = get_task_id_globally(global_index)

    print(f"Global: {global_index} -> {global_task_id}")


if __name__ == "__main__":
    example_usage()
