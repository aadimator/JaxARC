"""
JaxARC environments module.

This module provides reinforcement learning environments for solving
ARC (Abstraction and Reasoning Corpus) tasks using JAX-compatible
environments with a clean mask-centric action design.

Key Design Principles:
- **Mask-Based Core**: All actions are ultimately processed as MaskAction objects
- **Action Wrappers**: PointActionWrapper and BboxActionWrapper convert other formats to masks
- **Clean Separation**: Core environment only knows about masks, wrappers handle format conversion
- **Extensible Design**: New action formats can be added as wrappers without changing core logic
- **JAX Compatibility**: Functional and object-oriented APIs with full JAX transformation support

Architecture:
- Core environment (`Environment`, functional API) handles only MaskAction objects
- Action wrappers (`PointActionWrapper`, `BboxActionWrapper`) convert other formats to masks
- Grid operations and visualization work with the unified mask representation
- Clean separation of concerns allows easy extension with new action wrapper types
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

# Action wrappers
from .action_wrappers import (
    BboxActionWrapper,
    PointActionWrapper,
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
    "BboxActionWrapper",
    "DmEnvAutoResetWrapper",
    "EnvParams",
    "Environment",
    "GymAutoResetWrapper",
    "MaskAction",
    "MultiBinary",
    "PointActionWrapper",
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
