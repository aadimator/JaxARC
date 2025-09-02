"""
JaxARC environments module.

This module provides reinforcement learning environments for solving
ARC (Abstraction and Reasoning Corpus) tasks using JAX-compatible
environments with both functional and simple environment APIs.
"""

from __future__ import annotations

# State definition (centralized)
from ..state import State

# Core types for new functional API
from ..types import EnvParams, TimeStep

# Action space controller
from .action_space_controller import ActionSpaceController

# Action handlers
from .actions import (
    MaskAction,
    create_mask_action,
    mask_handler,
)

# Simple environment interface and wrappers (Xland-Minigrid pattern)
from .environment import Environment

# Functional API
from .functional import (
    reset,
    step,
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
from .wrapper import DmEnvAutoResetWrapper, GymAutoResetWrapper, Wrapper

__all__ = [
    "OPERATION_NAMES",
    "ActionSpaceController",
    "DmEnvAutoResetWrapper",
    "EnvParams",
    "Environment",
    "GymAutoResetWrapper",
    "MaskAction",
    "MultiBinary",
    "Space",
    "State",
    "TimeStep",
    "Wrapper",
    "create_mask_action",
    "execute_grid_operation",
    "get_all_operation_ids",
    "get_operation_category",
    "get_operation_display_text",
    "get_operation_name",
    "get_operations_by_category",
    "initialize_working_grids",
    "is_valid_operation_id",
    "mask_handler",
    "reset",
    "step",
]
