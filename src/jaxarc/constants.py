"""Named constants for JaxARC.

This module is intentionally dependency-free (no jaxarc internal imports)
so it can be imported by both config classes and types.py without circular
dependencies.
"""

from __future__ import annotations

# ARC domain constants
NUM_OPERATIONS: int = 35  # Number of ARC operations (0-34)
NUM_COLORS: int = 10  # Number of colors in ARC (0-9)
MAX_GRID_SIZE: int = 30  # Maximum grid dimension in ARC
