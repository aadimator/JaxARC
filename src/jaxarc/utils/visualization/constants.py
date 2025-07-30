"""Constants and color palettes for JaxARC visualization.

This module contains the core constants used throughout the visualization system,
including the ARC color palette and other visualization-related constants.
"""

from __future__ import annotations

# ARC color palette - matches the provided color map
ARC_COLOR_PALETTE: dict[int, str] = {
    0: "#252525",  # 0: black
    1: "#0074D9",  # 1: blue
    2: "#FF4136",  # 2: red
    3: "#37D449",  # 3: green
    4: "#FFDC00",  # 4: yellow
    5: "#E6E6E6",  # 5: grey
    6: "#F012BE",  # 6: pink
    7: "#FF871E",  # 7: orange
    8: "#54D2EB",  # 8: light blue
    9: "#8D1D2C",  # 9: brown
    10: "#FFFFFF",  # 10: white (for padding/invalid)
}