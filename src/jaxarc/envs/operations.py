"""Operation definitions and utilities for ARC environment.

This module provides centralized operation definitions, name mappings, and utilities
for converting operation IDs to human-readable names and display text. This is core
environment functionality that supports the action system.
"""

from __future__ import annotations

# Central operation names mapping
OPERATION_NAMES = {
    # Fill operations (0-9)
    0: "Fill 0",
    1: "Fill 1",
    2: "Fill 2",
    3: "Fill 3",
    4: "Fill 4",
    5: "Fill 5",
    6: "Fill 6",
    7: "Fill 7",
    8: "Fill 8",
    9: "Fill 9",
    # Flood fill operations (10-19)
    10: "Flood Fill 0",
    11: "Flood Fill 1",
    12: "Flood Fill 2",
    13: "Flood Fill 3",
    14: "Flood Fill 4",
    15: "Flood Fill 5",
    16: "Flood Fill 6",
    17: "Flood Fill 7",
    18: "Flood Fill 8",
    19: "Flood Fill 9",
    # Movement operations (20-23)
    20: "Move Up",
    21: "Move Down",
    22: "Move Left",
    23: "Move Right",
    # Transformation operations (24-27)
    24: "Rotate CW",
    25: "Rotate CCW",
    26: "Flip H",
    27: "Flip V",
    # Editing operations (28-31)
    28: "Copy",
    29: "Paste",
    30: "Cut",
    31: "Clear",
    # Special operations (32-34)
    32: "Copy Input",
    33: "Resize",
    34: "Submit",
}


def get_operation_name(operation_id: int) -> str:
    """Get the human-readable name for an operation ID.

    Args:
        operation_id: Integer operation ID

    Returns:
        Human-readable operation name

    Raises:
        ValueError: If operation_id is not recognized

    Example:
        ```python
        from jaxarc.envs.operations import get_operation_name

        name = get_operation_name(0)  # "Fill 0"
        name = get_operation_name(24)  # "Rotate CW"
        ```
    """
    if operation_id not in OPERATION_NAMES:
        raise ValueError(f"Unknown operation ID: {operation_id}")

    return OPERATION_NAMES[operation_id]


def get_operation_display_text(operation_id: int) -> str:
    """Get display text for visualization (includes ID and name).

    Args:
        operation_id: Integer operation ID

    Returns:
        Display text in format "Op {id}: {name}"

    Raises:
        ValueError: If operation_id is not recognized

    Example:
        ```python
        from jaxarc.envs.operations import get_operation_display_text

        text = get_operation_display_text(0)  # "Op 0: Fill 0"
        text = get_operation_display_text(34)  # "Op 34: Submit"
        ```
    """
    name = get_operation_name(operation_id)
    return f"Op {operation_id}: {name}"


def is_valid_operation_id(operation_id: int) -> bool:
    """Check if an operation ID is valid.

    Args:
        operation_id: Integer operation ID to check

    Returns:
        True if operation ID is valid, False otherwise

    Example:
        ```python
        from jaxarc.envs.operations import is_valid_operation_id

        assert is_valid_operation_id(0) == True
        assert is_valid_operation_id(34) == True
        assert is_valid_operation_id(35) == False
        ```
    """
    return operation_id in OPERATION_NAMES


def get_all_operation_ids() -> list[int]:
    """Get all valid operation IDs.

    Returns:
        List of all valid operation IDs sorted in ascending order

    Example:
        ```python
        from jaxarc.envs.operations import get_all_operation_ids

        ids = get_all_operation_ids()  # [0, 1, 2, ..., 34]
        ```
    """
    return sorted(OPERATION_NAMES.keys())


def get_operations_by_category() -> dict[str, list[int]]:
    """Get operations grouped by category.

    Returns:
        Dictionary mapping category names to lists of operation IDs

    Example:
        ```python
        from jaxarc.envs.operations import get_operations_by_category

        categories = get_operations_by_category()
        fill_ops = categories["fill"]  # [0, 1, 2, ..., 9]
        movement_ops = categories["movement"]  # [20, 21, 22, 23]
        ```
    """
    return {
        "fill": list(range(10)),
        "flood_fill": list(range(10, 20)),
        "movement": list(range(20, 24)),
        "transformation": list(range(24, 28)),
        "editing": list(range(28, 32)),
        "special": list(range(32, 35)),
    }


def get_operation_category(operation_id: int) -> str:
    """Get the category name for an operation ID.

    Args:
        operation_id: Integer operation ID

    Returns:
        Category name for the operation

    Raises:
        ValueError: If operation_id is not recognized

    Example:
        ```python
        from jaxarc.envs.operations import get_operation_category

        category = get_operation_category(5)  # "fill"
        category = get_operation_category(24)  # "transformation"
        ```
    """
    if not is_valid_operation_id(operation_id):
        raise ValueError(f"Unknown operation ID: {operation_id}")

    categories = get_operations_by_category()
    for category_name, op_ids in categories.items():
        if operation_id in op_ids:
            return category_name

    # This should never happen if is_valid_operation_id works correctly
    raise ValueError(f"Operation ID {operation_id} not found in any category")
