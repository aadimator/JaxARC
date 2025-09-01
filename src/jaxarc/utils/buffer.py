"""
Utilities for building and using a JAX-native task buffer.

This module provides two core helpers:

1) stack_task_list(tasks):
   - Takes a Python sequence of JaxArcTask objects (Equinox Modules)
   - Converts scalar Python ints in each task to JAX 0-d arrays to avoid
     object-array pitfalls
   - Stacks each corresponding field across the list along a new leading
     dimension, returning a batched pytree (the "buffer") of JAX arrays

   The returned buffer has the same pytree structure as a single JaxArcTask,
   except that each leaf now has a leading batch dimension of size N=len(tasks).
   This buffer is JIT- and vmap-friendly and can be indexed using JAX ops.

2) gather_task(buffer, idx):
   - Given a stacked buffer and a JAX scalar index `idx`, returns the
     single-task view by slicing the leading batch dimension from all
     array leaves. Scalar leaves (0-dim) are left as-is.

Additionally, buffer_size(buffer) returns the leading batch size (N) inferred
from a canonical array in the buffer (or the first array-like leaf).

Design notes:
- We avoid keeping a Python list of tasks during JIT. Instead, we build a
  single JAX pytree of arrays once on host and reuse it in compiled code.
- The buffer can be stored directly in EnvParams; vmap(reset, in_axes=(None, 0))
  will broadcast EnvParams and reuse a single buffer copy (no per-batch duplication).
- For multi-device execution with pmap, the buffer is replicated per device.
  Use sharding (pjit) if you need to distribute the buffer.

Example:
    tasks = [parser.get_task_by_id(tid) for tid in ids]  # host-side
    buffer = stack_task_list(tasks)                       # pytree with leading N
    # In JIT:
    idx = jax.random.randint(key, (), 0, buffer_size(buffer))
    single = gather_task(buffer, idx)                    # single-task pytree
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from jaxarc.types import JaxArcTask


def _to_jax_scalar_if_int(x: Any) -> Any:
    """Convert Python ints to JAX int32 scalars to enable numeric stacking.

    Leaves that are already JAX arrays or other types are returned unchanged.
    """
    if isinstance(x, bool):
        # Keep Python bools as-is; typically not present as leaves we stack.
        return x
    if isinstance(x, int):
        return jnp.asarray(x, dtype=jnp.int32)
    return x


def _preprocess_task_for_stacking(task: JaxArcTask) -> JaxArcTask:
    """Prepare a single JaxArcTask for stacking by converting Python ints to JAX scalars."""
    return jax.tree_util.tree_map(_to_jax_scalar_if_int, task)


def stack_task_list(tasks: Sequence[JaxArcTask]) -> Any:
    """Stack a list/sequence of JaxArcTask into a batched pytree buffer.

    Args:
        tasks: Sequence of JaxArcTask (all with identical padded shapes)

    Returns:
        A pytree with the same structure as JaxArcTask where each leaf is
        a JAX array stacked along axis 0, giving a leading batch dimension N=len(tasks).

    Raises:
        ValueError: If the input sequence is empty
    """
    if not tasks:
        raise ValueError("stack_task_list: 'tasks' must be a non-empty sequence")

    # Convert Python ints to JAX scalars so stacking yields numeric arrays
    processed = [_preprocess_task_for_stacking(t) for t in tasks]

    # Stack each corresponding leaf across tasks along a new axis 0
    # Resulting pytree preserves the original structure
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *processed)
    return batched


def gather_task(buffer: Any, idx: jnp.ndarray) -> Any:
    """Gather a single-task view from a stacked buffer by dynamic index.

    This function is JAX-friendly and safe to use inside jit/vmap.

    Args:
        buffer: A batched pytree created by stack_task_list (leading batch axis)
        idx:    JAX scalar index selecting which task to gather

    Returns:
        A single-task pytree with the leading batch dimension removed from all
        array leaves. Scalar (0-dim) leaves are returned unchanged.
    """

    def _gather(x):
        # Handle JAX arrays and NumPy-like with shape attribute
        if hasattr(x, "ndim") and x.ndim > 0:
            return x[idx]
        return x

    return jax.tree_util.tree_map(_gather, buffer)


def buffer_size(buffer: Any) -> int:
    """Infer the leading batch size (N) of a stacked buffer.

    It first tries to read buffer.input_grids_examples.shape[0]. If that is not
    available, it falls back to the first array leaf with ndim > 0.

    Args:
        buffer: A batched pytree created by stack_task_list

    Returns:
        Integer batch size N (Python int)

    Raises:
        ValueError: If no array leaves with a leading dimension are found
    """
    # Prefer a canonical field if present
    try:
        candidate = buffer.input_grids_examples
        if hasattr(candidate, "shape") and len(candidate.shape) > 0:
            return int(candidate.shape[0])
    except Exception:
        pass

    # Fallback: scan leaves to find the first array with ndim > 0
    for leaf in jax.tree_util.tree_leaves(buffer):
        if hasattr(leaf, "shape") and len(leaf.shape) > 0:
            return int(leaf.shape[0])

    raise ValueError(
        "buffer_size: could not infer leading batch size from buffer leaves"
    )


__all__ = [
    "buffer_size",
    "gather_task",
    "stack_task_list",
]
