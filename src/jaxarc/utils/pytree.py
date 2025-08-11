"""
Core PyTree and Equinox manipulation utilities.

This module provides a consolidated set of utilities for working with PyTrees,
drawing from the original `equinox_utils.py` and `pytree_utils.py` modules.
It focuses on generic, reusable functions for tree traversal, updates,
and debugging.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from ..state import ArcEnvState

# Type variables for generic functions
T = TypeVar("T", bound=eqx.Module)


def update_multiple_fields(state: T, **updates) -> T:
    """Update multiple fields efficiently using equinox.tree_at.

    This function provides an optimized way to update multiple fields in an
    Equinox module simultaneously.

    Args:
        state: The Equinox module to update.
        **updates: Field names and their new values.

    Returns:
        A new module with the updated fields.
    """
    if not updates:
        return state

    current_state = state
    for field_name, new_value in updates.items():
        if not hasattr(current_state, field_name):
            raise AttributeError(f"{type(state).__name__} has no field '{field_name}'")
        current_state = eqx.tree_at(
            lambda s, fn=field_name: getattr(s, fn), current_state, new_value
        )

    return current_state


@eqx.filter_jit
def jit_update_multiple_fields(state: T, **updates) -> T:
    """JIT-compiled version of update_multiple_fields for performance.

    Args:
        state: The Equinox module to update.
        **updates: Field names and their new values.

    Returns:
        A new module with the updated fields.
    """
    return update_multiple_fields(state, **updates)


def filter_and_combine(
    tree: PyTree,
) -> Tuple[Tuple[PyTree, PyTree], Callable[[PyTree, PyTree], PyTree]]:
    """Partition a PyTree and return a function to combine it back.

    Args:
        tree: The PyTree to partition.

    Returns:
        A tuple containing the partitioned trees (arrays, non-arrays) and the
        `eqx.combine` function to merge them.
    """
    arrays, non_arrays = eqx.partition(tree, eqx.is_array)
    return (arrays, non_arrays), eqx.combine


def tree_map_with_path(
    fn: Callable[[str, Any], Any],
    tree: PyTree,
    prefix: str = "",
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    """Enhanced tree mapping with path information for debugging.

    Args:
        fn: Function that takes (path, value) and returns a transformed value.
        tree: PyTree to map over.
        prefix: Path prefix for nested calls.
        is_leaf: Optional function to determine leaf nodes.

    Returns:
        The original tree (unchanged); this function is for side effects.
    """

    def _traverse(obj: Any, path: str) -> None:
        fn(path, obj)
        if is_leaf is not None and is_leaf(obj):
            return
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            return
        if isinstance(obj, eqx.Module):
            if hasattr(obj, "__annotations__"):
                for field_name in obj.__annotations__.keys():
                    if hasattr(obj, field_name):
                        field_value = getattr(obj, field_name)
                        child_path = f"{path}.{field_name}" if path else str(field_name)
                        _traverse(field_value, child_path)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                child_path = f"{path}.{key}" if path else str(key)
                _traverse(value, child_path)
            return
        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                child_path = f"{path}[{i}]" if path else f"[{i}]"
                _traverse(item, child_path)
            return
        try:
            children, _ = jtu.tree_flatten(obj)
            for i, child in enumerate(children):
                child_path = f"{path}[{i}]" if path else f"[{i}]"
                _traverse(child, child_path)
        except Exception:
            pass

    _traverse(tree, prefix)
    return tree


def tree_size_info(tree: PyTree) -> Dict[str, Tuple[Any, int]]:
    """Get size information for all arrays in a PyTree.

    Args:
        tree: PyTree to analyze.

    Returns:
        A dictionary mapping paths to (shape, size) tuples.
    """
    size_info = {}

    def collect_info(path: str, value: Any) -> Any:
        if hasattr(value, "shape") and hasattr(value, "size"):
            size_info[path] = (value.shape, value.size)
        return value

    tree_map_with_path(collect_info, tree)
    return size_info


def filter_arrays_from_state(state: ArcEnvState) -> Tuple[PyTree, PyTree]:
    """Filter arrays and non-arrays from state for serialization.
    
    This function separates array and non-array components of the state,
    which is useful for efficient serialization where you want to handle
    arrays and metadata differently.
    
    Args:
        state: ArcEnvState to filter
        
    Returns:
        Tuple of (arrays, non_arrays) PyTrees
        
    Examples:
        ```python
        arrays, non_arrays = filter_arrays_from_state(state)
        
        # Save arrays with binary format
        eqx.tree_serialise_leaves("state_arrays.eqx", arrays)
        
        # Save non-arrays with JSON
        import json
        with open("state_metadata.json", "w") as f:
            json.dump(non_arrays, f)
        ```
    """
    return eqx.partition(state, eqx.is_array)


def create_state_diff(old_state: T, new_state: T) -> Dict[str, Dict[str, Any]]:
    """Create a diff between two Equinox modules for debugging.

    Args:
        old_state: The previous state.
        new_state: The new state.

    Returns:
        A dictionary with diff information.
    """
    differences = {}

    def compare_values(old_val: Any, new_val: Any) -> Dict[str, Any] | None:
        try:
            if hasattr(old_val, "shape") and hasattr(new_val, "shape"):
                if old_val.shape != new_val.shape:
                    return {
                        "type": "shape_change",
                        "old": old_val.shape,
                        "new": new_val.shape,
                    }
                if not jnp.array_equal(old_val, new_val):
                    return {
                        "type": "value_change",
                        "old": old_val,
                        "new": new_val,
                        "max_diff": jnp.max(jnp.abs(old_val - new_val))
                        if jnp.issubdtype(old_val.dtype, jnp.number)
                        else None,
                    }
            else:
                old_item = old_val.item() if hasattr(old_val, "item") else old_val
                new_item = new_val.item() if hasattr(new_val, "item") else new_val
                if old_item != new_item:
                    return {"type": "value_change", "old": old_item, "new": new_item}
        except Exception:
            pass
        return None

    if isinstance(old_state, eqx.Module) and isinstance(new_state, eqx.Module):
        if hasattr(old_state, "__annotations__") and hasattr(
            new_state, "__annotations__"
        ):
            for field_name in old_state.__annotations__.keys():
                if hasattr(old_state, field_name) and hasattr(new_state, field_name):
                    old_val = getattr(old_state, field_name)
                    new_val = getattr(new_state, field_name)
                    diff_result = compare_values(old_val, new_val)
                    if diff_result is not None:
                        differences[field_name] = diff_result
    else:
        old_flat, _ = jtu.tree_flatten(old_state)
        new_flat, _ = jtu.tree_flatten(new_state)
        if len(old_flat) != len(new_flat):
            differences["_structure"] = {
                "type": "structure_change",
                "old_size": len(old_flat),
                "new_size": len(new_flat),
            }
        else:
            for i, (old_val, new_val) in enumerate(zip(old_flat, new_flat)):
                diff_result = compare_values(old_val, new_val)
                if diff_result is not None:
                    differences[f"leaf_{i}"] = diff_result
    return differences


def print_state_summary(state: T, name: str = "State") -> None:
    """Print a summary of an Equinox module's contents for debugging.

    Args:
        state: The Equinox module to summarize.
        name: A name to use in the summary header.
    """
    print(f"\n=== {name} Summary ===")

    def print_info(path: str, value: Any) -> Any:
        if hasattr(value, "shape"):
            dtype_str = str(value.dtype) if hasattr(value, "dtype") else "unknown"
            print(f"  {path}: shape={value.shape}, dtype={dtype_str}")
            if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.number):
                try:
                    print(
                        f"    min={jnp.min(value):.4f}, max={jnp.max(value):.4f}, mean={jnp.mean(value):.4f}"
                    )
                except Exception:
                    pass
        else:
            print(f"  {path}: {type(value).__name__} = {value}")
        return value

    tree_map_with_path(print_info, state)
    print("=" * (len(name) + 12))


def freeze_module(module: T) -> T:
    """Freeze an Equinox module to prevent accidental modification."""
    return eqx.tree_at(lambda x: x, module, replace_fn=lambda x: x)


def unfreeze_module(module: T) -> T:
    """Unfreeze an Equinox module to allow modification."""
    return eqx.tree_at(lambda x: x, module, replace_fn=lambda x: x)


def module_memory_usage(module: T) -> Dict[str, int]:
    """Calculate the memory usage of an Equinox module.

    Args:
        module: The Equinox module to analyze.

    Returns:
        A dictionary with memory usage information.
    """
    memory_info = {"total_bytes": 0, "total_elements": 0, "arrays": {}}

    def count_memory(path: str, value: Any) -> Any:
        if hasattr(value, "nbytes") and hasattr(value, "size"):
            memory_info["arrays"][path] = {
                "bytes": value.nbytes,
                "elements": value.size,
                "shape": value.shape,
                "dtype": str(value.dtype),
            }
            memory_info["total_bytes"] += value.nbytes
            memory_info["total_elements"] += value.size
        return value

    tree_map_with_path(count_memory, module)
    return memory_info


def ensure_jax_compatible(module: T) -> T:
    """Ensure an Equinox module is compatible with JAX transformations."""
    try:
        flat, tree_def = jtu.tree_flatten(module)
        reconstructed = jtu.tree_unflatten(tree_def, flat)
        return reconstructed
    except Exception as e:
        warnings.warn(f"JAX compatibility check failed: {e}")
        return module


def check_jax_transformations(
    module: T, test_fn: Callable[[T], Any]
) -> Dict[str, bool]:
    """Test that an Equinox module works with common JAX transformations.

    Args:
        module: The Equinox module to test.
        test_fn: A function that takes the module and returns some result.

    Returns:
        A dictionary indicating which transformations work.
    """
    results = {}
    try:
        jitted_fn = jax.jit(test_fn)
        jitted_fn(module)
        results["jit"] = True
    except Exception:
        results["jit"] = False
    try:
        batch_module = jtu.tree_map(
            lambda x: jnp.expand_dims(x, 0) if hasattr(x, "shape") else x, module
        )
        vmapped_fn = jax.vmap(test_fn)
        vmapped_fn(batch_module)
        results["vmap"] = True
    except Exception:
        results["vmap"] = False
    try:

        def scalar_test_fn(m):
            result = test_fn(m)
            return (
                jnp.sum(result) if hasattr(result, "shape") else jnp.sum(jnp.array(0.0))
            )

        grad_fn = jax.grad(scalar_test_fn)
        grad_fn(module)
        results["grad"] = True
    except Exception:
        results["grad"] = False
    return results
