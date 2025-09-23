"""
JaxARC environments module.

This module provides reinforcement learning environments for solving
ARC (Abstraction and Reasoning Corpus) tasks using JAX-compatible
environments with a clean action design.

Key Design Principles:
- **Action-Based Core**: All actions are ultimately processed as Action objects
- **Action Wrappers**: PointActionWrapper and BboxActionWrapper convert other formats to actions
- **Clean Separation**: Core environment only knows about actions, wrappers handle format conversion
- **Extensible Design**: New action formats can be added as wrappers without changing core logic
- **JAX Compatibility**: Functional and object-oriented APIs with full JAX transformation support

Architecture:
- Core environment (`Environment`, functional API) handles only Action objects
- Action wrappers (`PointActionWrapper`, `BboxActionWrapper`) convert other formats to actions
- Grid operations and visualization work with the unified action representation
- Clean separation of concerns allows easy extension with new action wrapper types
"""

from __future__ import annotations

# Action wrappers
from jaxarc.envs.wrappers import (
    AddChannelDimWrapper,
    BboxActionWrapper,
    FlattenDictActionWrapper,
    PointActionWrapper,
)

# State definition (centralized)
from ..state import State

# Core types for new functional API
from ..types import EnvParams, TimeStep

# Complete action system (combined actions + filtering)
from .actions import (
    Action,
    action_handler,
    create_action,
    filter_invalid_operation,
    get_allowed_operations,
    validate_operation,
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
from .spaces import (
    ARCActionSpace,
    BoundedArraySpace,
    DictSpace,
    DiscreteSpace,
    GridSpace,
    MultiBinary,
    SelectionSpace,
    Space,
)

__all__ = [
    "OPERATION_NAMES",
    "ARCActionSpace",
    "Action",
    "AddChannelDimWrapper",
    "BboxActionWrapper",
    "BoundedArraySpace",
    "DictSpace",
    "DiscreteSpace",
    "EnvParams",
    "Environment",
    "FlattenDictActionWrapper",
    "GridSpace",
    "MultiBinary",
    "PointActionWrapper",
    "SelectionSpace",
    "Space",
    "State",
    "TimeStep",
    "action_handler",
    "create_action",
    "execute_grid_operation",
    "filter_invalid_operation",
    "get_all_operation_ids",
    "get_allowed_operations",
    "get_operation_category",
    "get_operation_display_text",
    "get_operation_name",
    "get_operations_by_category",
    "initialize_working_grids",
    "is_valid_operation_id",
    "reset",
    "step",
    "validate_operation",
]