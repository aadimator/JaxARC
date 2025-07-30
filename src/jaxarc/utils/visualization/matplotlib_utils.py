"""Matplotlib integration utilities for JaxARC visualization.

This module provides utilities for setting up matplotlib styling and
integration with the visualization system.
"""

from __future__ import annotations

# Optional matplotlib imports for enhanced visualizations
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def setup_matplotlib_style() -> None:
    """Set up matplotlib and seaborn styling for high-quality visualizations.

    Raises:
        ImportError: If matplotlib or seaborn is not available
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib and seaborn are required for this function. Install with: pip install matplotlib seaborn"
        )

    # Configure matplotlib for high-quality output
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

    # Set up seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")