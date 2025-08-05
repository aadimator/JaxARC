"""Logging utilities for JaxARC.

This module provides a simplified logging architecture centered around the
ExperimentLogger class with focused handlers for different logging concerns.
The system removes overengineered components while preserving valuable
debugging capabilities.
"""

from __future__ import annotations

# Central logging coordinator
from .experiment_logger import ExperimentLogger

# Individual handlers
from .file_handler import FileHandler
from .svg_handler import SVGHandler
from .rich_handler import RichHandler
from .wandb_handler import WandbHandler

# Legacy structured logger components (for backward compatibility)
from .structured_logger import (
    EpisodeLogEntry,
    LoggingConfig,
    StepLogEntry,
)

__all__ = [
    # New simplified logging architecture
    "ExperimentLogger",
    "FileHandler",
    "SVGHandler", 
    "RichHandler",
    "WandbHandler",
    # Legacy components (for backward compatibility)
    "EpisodeLogEntry",
    "LoggingConfig",
    "StepLogEntry",
]
