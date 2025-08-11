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
from .rich_handler import RichHandler
from .svg_handler import SVGHandler
from .wandb_handler import WandbHandler

__all__ = [
    "ExperimentLogger",
    "FileHandler",
    "RichHandler",
    "SVGHandler",
    "WandbHandler",
]
