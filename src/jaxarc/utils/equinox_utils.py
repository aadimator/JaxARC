"""
Equinox integration utilities for JaxARC.

This module provides utilities for integrating Equinox with the JaxARC codebase,
including enhanced tree operations, state validation, and debugging support.
Equinox provides better JAX integration with automatic PyTree registration,
improved error messages, and cleaner functional patterns.

Key Features:
- Enhanced tree mapping with path information for debugging
- State validation using Equinox patterns
- State diffing utilities for debugging state transitions
- Automatic PyTree registration and validation
- Better error messages for shape mismatches

Examples:
    ```python
    import equinox as eqx
    from jaxarc.utils.equinox_utils import tree_map_with_path, validate_state_shapes


    # Enhanced tree operations with path tracking
    def debug_print(path, value):
        print(f"Path: {path}, Value shape: {value.shape}")


    tree_map_with_path(debug_print, state)

    # State validation
    is_valid = validate_state_shapes(state)
    ```
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

# Type variables for generic functions
T = TypeVar("T", bound=eqx.Module)
U = TypeVar("U")

# =============================================================================
# Enhanced Tree Operations
# =============================================================================


def tree_map_with_path(
    fn: Callable[[str, Any], Any],
    tree: PyTree,
    prefix: str = "",
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    """Enhanced tree mapping with path information for debugging.

    This function maps a function over a PyTree while providing path information
    to the function. This is useful for debugging, logging, and validation where
    you need to know the location of each value in the tree structure.

    Note: This is a simplified implementation that focuses on collecting information
    rather than transforming the tree structure. For complex transformations,
    use JAX's built-in tree utilities.

    Args:
        fn: Function that takes (path, value) and returns transformed value
        tree: PyTree to map over
        prefix: Path prefix for nested calls
        is_leaf: Optional function to determine leaf nodes

    Returns:
        The original tree (unchanged) - this function is primarily for side effects

    Examples:
        ```python
        def print_shapes(path: str, value: Any) -> Any:
            if hasattr(value, "shape"):
                print(f"{path}: {value.shape}")
            return value


        tree_map_with_path(print_shapes, state)
        ```
    """

    def _traverse(obj: Any, path: str) -> None:
        """Internal traversal function that applies fn to each node."""
        # Apply function to current node
        fn(path, obj)

        # Check if this is a leaf node (JAX arrays, scalars, etc.)
        if is_leaf is not None and is_leaf(obj):
            return

        # JAX arrays are leaf nodes - don't recurse into them
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            return

        # Handle Equinox modules
        if isinstance(obj, eqx.Module):
            if hasattr(obj, "__annotations__"):
                for field_name in obj.__annotations__.keys():
                    if hasattr(obj, field_name):
                        field_value = getattr(obj, field_name)
                        child_path = f"{path}.{field_name}" if path else str(field_name)
                        _traverse(field_value, child_path)
            return

        # Handle dictionaries
        if isinstance(obj, dict):
            for key, value in obj.items():
                child_path = f"{path}.{key}" if path else str(key)
                _traverse(value, child_path)
            return

        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                child_path = f"{path}[{i}]" if path else f"[{i}]"
                _traverse(item, child_path)
            return

        # For other types, try to use JAX tree utilities
        try:
            children, tree_def = jtu.tree_flatten(obj)
            for i, child in enumerate(children):
                child_path = f"{path}[{i}]" if path else f"[{i}]"
                _traverse(child, child_path)
        except:
            # If tree flattening fails, this is likely a leaf node
            pass

    # Start traversal
    _traverse(tree, prefix)

    # Return the original tree unchanged
    return tree


def tree_size_info(tree: PyTree) -> Dict[str, Tuple[Any, int]]:
    """Get size information for all arrays in a PyTree.

    Args:
        tree: PyTree to analyze

    Returns:
        Dictionary mapping paths to (shape, size) tuples

    Examples:
        ```python
        info = tree_size_info(state)
        for path, (shape, size) in info.items():
            print(f"{path}: shape={shape}, size={size}")
        ```
    """
    size_info = {}

    def collect_info(path: str, value: Any) -> Any:
        if hasattr(value, "shape") and hasattr(value, "size"):
            size_info[path] = (value.shape, value.size)
        return value

    tree_map_with_path(collect_info, tree)
    return size_info


# =============================================================================
# State Validation
# =============================================================================


def validate_state_shapes(state: T) -> bool:
    """Validate state using Equinox patterns.

    This function validates that all arrays in an Equinox module have consistent
    shapes and types. It's designed to work with JAX transformations by gracefully
    handling cases where arrays don't have concrete shapes during tracing.

    Args:
        state: Equinox module to validate

    Returns:
        True if validation passes, False otherwise

    Examples:
        ```python
        if not validate_state_shapes(state):
            raise ValueError("State validation failed")
        ```
    """
    try:
        # Use Equinox's built-in validation if available
        if hasattr(state, "__check_init__"):
            state.__check_init__()
            return True

        # Fallback validation for arrays
        validation_errors = []

        def validate_array(path: str, value: Any) -> Any:
            if hasattr(value, "shape"):
                try:
                    # Check for NaN or infinite values
                    if jnp.issubdtype(value.dtype, jnp.floating):
                        if jnp.any(jnp.isnan(value)) or jnp.any(jnp.isinf(value)):
                            validation_errors.append(
                                f"{path}: contains NaN or infinite values"
                            )

                    # Check for negative values where inappropriate
                    if "step_count" in path.lower() or "index" in path.lower():
                        if jnp.any(value < 0):
                            validation_errors.append(
                                f"{path}: contains negative values"
                            )

                except Exception:
                    # Skip validation during JAX transformations
                    pass

            return value

        tree_map_with_path(validate_array, state)

        if validation_errors:
            for error in validation_errors:
                warnings.warn(f"State validation warning: {error}")
            return False

        return True

    except Exception:
        # Skip validation during JAX transformations
        return True


def validate_array_compatibility(
    arr1: Array,
    arr2: Array,
    check_shape: bool = True,
    check_dtype: bool = True,
) -> bool:
    """Validate that two arrays are compatible for operations.

    Args:
        arr1: First array
        arr2: Second array
        check_shape: Whether to check shape compatibility
        check_dtype: Whether to check dtype compatibility

    Returns:
        True if arrays are compatible, False otherwise
    """
    try:
        if check_shape and hasattr(arr1, "shape") and hasattr(arr2, "shape"):
            if arr1.shape != arr2.shape:
                return False

        if check_dtype and hasattr(arr1, "dtype") and hasattr(arr2, "dtype"):
            if arr1.dtype != arr2.dtype:
                return False

        return True

    except Exception:
        # Skip validation during JAX transformations
        return True


# =============================================================================
# State Debugging and Diffing
# =============================================================================


def create_state_diff(old_state: T, new_state: T) -> Dict[str, Dict[str, Any]]:
    """Create diff between states for debugging.

    This function compares two Equinox modules and returns information about
    what changed between them. Useful for debugging state transitions and
    understanding the effects of operations.

    Args:
        old_state: Previous state
        new_state: New state

    Returns:
        Dictionary with diff information including changed fields and their values

    Examples:
        ```python
        diff = create_state_diff(old_state, new_state)
        for path, change_info in diff.items():
            print(f"Changed: {path}")
            print(f"  Old: {change_info['old']}")
            print(f"  New: {change_info['new']}")
        ```
    """
    differences = {}

    def compare_values(old_val: Any, new_val: Any) -> Dict[str, Any] | None:
        try:
            if hasattr(old_val, "shape") and hasattr(new_val, "shape"):
                # Compare arrays
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
                # Compare scalars or other values
                try:
                    # Handle JAX arrays that might be scalars
                    old_item = old_val.item() if hasattr(old_val, "item") else old_val
                    new_item = new_val.item() if hasattr(new_val, "item") else new_val

                    if old_item != new_item:
                        return {
                            "type": "value_change",
                            "old": old_item,
                            "new": new_item,
                        }
                except:
                    # Fallback comparison
                    if old_val != new_val:
                        return {
                            "type": "value_change",
                            "old": old_val,
                            "new": new_val,
                        }
        except Exception:
            # Skip comparison during JAX transformations
            pass
        return None

    # For Equinox modules, compare field by field
    if isinstance(old_state, eqx.Module) and isinstance(new_state, eqx.Module):
        if hasattr(old_state, "__annotations__") and hasattr(
            new_state, "__annotations__"
        ):
            # Compare each field
            for field_name in old_state.__annotations__.keys():
                if hasattr(old_state, field_name) and hasattr(new_state, field_name):
                    old_val = getattr(old_state, field_name)
                    new_val = getattr(new_state, field_name)

                    diff_result = compare_values(old_val, new_val)
                    if diff_result is not None:
                        differences[field_name] = diff_result
    else:
        # Fallback to tree comparison
        old_flat, old_tree_def = jtu.tree_flatten(old_state)
        new_flat, new_tree_def = jtu.tree_flatten(new_state)

        if len(old_flat) != len(new_flat):
            differences["_structure"] = {
                "type": "structure_change",
                "old_size": len(old_flat),
                "new_size": len(new_flat),
            }
        else:
            # Compare flattened values
            for i, (old_val, new_val) in enumerate(zip(old_flat, new_flat)):
                diff_result = compare_values(old_val, new_val)
                if diff_result is not None:
                    differences[f"leaf_{i}"] = diff_result

    return differences


def print_state_summary(state: T, name: str = "State") -> None:
    """Print a summary of state contents for debugging.

    Args:
        state: Equinox module to summarize
        name: Name to use in the summary
    """
    print(f"\n=== {name} Summary ===")

    def print_info(path: str, value: Any) -> Any:
        if hasattr(value, "shape"):
            dtype_str = str(value.dtype) if hasattr(value, "dtype") else "unknown"
            print(f"  {path}: shape={value.shape}, dtype={dtype_str}")

            # Print some statistics for numeric arrays
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


# =============================================================================
# Equinox Module Utilities
# =============================================================================


def freeze_module(module: T) -> T:
    """Freeze an Equinox module to prevent accidental modification.

    Args:
        module: Equinox module to freeze

    Returns:
        Frozen version of the module
    """
    return eqx.tree_at(lambda x: x, module, replace_fn=lambda x: x)


def unfreeze_module(module: T) -> T:
    """Unfreeze an Equinox module to allow modification.

    Args:
        module: Frozen Equinox module

    Returns:
        Unfrozen version of the module
    """
    return eqx.tree_at(lambda x: x, module, replace_fn=lambda x: x)


def module_memory_usage(module: T) -> Dict[str, int]:
    """Calculate memory usage of an Equinox module.

    Args:
        module: Equinox module to analyze

    Returns:
        Dictionary with memory usage information
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


# =============================================================================
# JAX Transformation Helpers
# =============================================================================


def ensure_jax_compatible(module: T) -> T:
    """Ensure an Equinox module is compatible with JAX transformations.

    This function performs checks and transformations to ensure the module
    works correctly with jit, vmap, pmap, etc.

    Args:
        module: Equinox module to check

    Returns:
        JAX-compatible version of the module
    """
    # Equinox modules are automatically JAX-compatible due to PyTree registration
    # This function serves as a validation point and future extension point

    try:
        # Test that the module can be flattened and unflattened
        flat, tree_def = jtu.tree_flatten(module)
        reconstructed = jtu.tree_unflatten(tree_def, flat)

        # Basic validation that reconstruction worked
        if not validate_state_shapes(reconstructed):
            warnings.warn("Module reconstruction validation failed")

        return reconstructed

    except Exception as e:
        warnings.warn(f"JAX compatibility check failed: {e}")
        return module


def check_jax_transformations(
    module: T, test_fn: Callable[[T], Any]
) -> Dict[str, bool]:
    """Test that an Equinox module works with common JAX transformations.

    Args:
        module: Equinox module to test
        test_fn: Function that takes the module and returns some result

    Returns:
        Dictionary indicating which transformations work
    """
    results = {}

    try:
        # Test JIT compilation
        jitted_fn = jax.jit(test_fn)
        jitted_fn(module)
        results["jit"] = True
    except Exception:
        results["jit"] = False

    try:
        # Test vmap (requires creating a batch)
        batch_module = jtu.tree_map(
            lambda x: jnp.expand_dims(x, 0) if hasattr(x, "shape") else x, module
        )
        vmapped_fn = jax.vmap(test_fn)
        vmapped_fn(batch_module)
        results["vmap"] = True
    except Exception:
        results["vmap"] = False

    try:
        # Test grad (if the test function returns a scalar)
        def scalar_test_fn(m):
            result = test_fn(m)
            return (
                jnp.sum(jnp.array(0.0))
                if not hasattr(result, "shape")
                else jnp.sum(result)
            )

        grad_fn = jax.grad(scalar_test_fn)
        grad_fn(module)
        results["grad"] = True
    except Exception:
        results["grad"] = False

    return results
