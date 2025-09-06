"""Logging utilities for JaxARC.

This module provides a simplified logging architecture centered around the
ExperimentLogger class with focused handlers for different logging concerns.
The system removes overengineered components while preserving valuable
debugging capabilities.
"""

from __future__ import annotations

# Individual handlers
from .handlers import (
    FileHandler,
    RichHandler,
    SVGHandler,
    WandbHandler,
)

# Central logging coordinator
from .logger import (
    ExperimentLogger,
    create_episode_summary,
    create_start_log,
    create_step_log,
)

__all__ = [
    "ExperimentLogger",
    "FileHandler",
    "RichHandler",
    "SVGHandler",
    "WandbHandler",
    "create_episode_summary",
    "create_start_log",
    "create_step_log",
]
