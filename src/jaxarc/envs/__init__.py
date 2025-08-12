"""
JaxARC environments module.

This module provides reinforcement learning environments for solving
ARC (Abstraction and Reasoning Corpus) tasks using JAX-compatible
environments with both functional and class-based APIs.
"""

from __future__ import annotations

# State definition (centralized)
from ..state import ArcEnvState

# Action space controller
from .action_space_controller import ActionSpaceController

# Action handlers
# Structured actions
from .actions import (
    BaseAction,
    BboxAction,
    MaskAction,
    PointAction,
    StructuredAction,
    bbox_handler,
    create_bbox_action,
    create_mask_action,
    create_point_action,
    get_action_handler,
    mask_handler,
    point_handler,
)

# Functional API
from .functional import (
    arc_reset,
    arc_step,
)

# Grid initialization
from .grid_initialization import initialize_working_grids

# Grid operations
from .grid_operations import (
    OPERATION_NAMES,
    execute_grid_operation,
    get_all_operation_ids,
    get_operation_category,
    get_operation_display_text,
    get_operation_name,
    get_operations_by_category,
    is_valid_operation_id,
)

# Action and observation spaces
from .spaces import MultiBinary, Space

__all__ = [
    "OPERATION_NAMES",
    "ActionSpaceController",
    "ArcEnvState",
    "BaseAction",
    "BboxAction",
    "MaskAction",
    "MultiBinary",
    "PointAction",
    "Space",
    "StructuredAction",
    "arc_reset",
    "arc_step",
    "bbox_handler",
    "create_bbox_action",
    "create_mask_action",
    "create_point_action",
    "execute_grid_operation",
    "get_action_handler",
    "get_all_operation_ids",
    "get_operation_category",
    "get_operation_display_text",
    "get_operation_name",
    "get_operations_by_category",
    "initialize_working_grids",
    "is_valid_operation_id",
    "mask_handler",
    "point_handler",
]
