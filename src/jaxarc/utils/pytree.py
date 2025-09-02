"""
Essential PyTree utilities for JaxARC.

This module provides only the core utilities that are actually used throughout
the codebase, focusing on state updates and serialization support.
"""

from __future__ import annotations

from typing import TypeVar

import equinox as eqx

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

    Example:
        ```python
        new_state = update_multiple_fields(
            state, working_grid=new_grid, step_count=state.step_count + 1, similarity_score=0.85
        )
        ```
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
