"""Structured logging data classes.

This module provides data classes for structured logging that are used by
other components. The actual logging functionality has been moved to FileHandler.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import chex


@chex.dataclass
class LoggingConfig:
    """Configuration for structured logging."""

    structured_logging: bool = True
    log_format: str = "json"  # "json", "hdf5", "pickle"
    compression: bool = True
    include_full_states: bool = False  # For performance
    log_level: str = "INFO"
    async_logging: bool = True
    output_dir: str = "outputs/structured_logs"


@chex.dataclass
class StepLogEntry:
    """Structured log entry for a single step."""

    step_num: int
    timestamp: float
    before_state: Dict[str, Any]  # Serialized state
    action: Dict[str, Any]
    after_state: Dict[str, Any]  # Serialized state
    reward: float
    info: Dict[str, Any]
    visualization_path: Optional[str] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            object.__setattr__(self, "timestamp", time.time())


@chex.dataclass
class EpisodeLogEntry:
    """Structured log entry for a complete episode."""

    episode_num: int
    start_timestamp: float
    end_timestamp: float
    total_steps: int
    total_reward: float
    final_similarity: float
    task_id: str
    config_hash: str
    steps: List[StepLogEntry]
    summary_visualization_path: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.end_timestamp == 0.0:
            object.__setattr__(self, "end_timestamp", time.time())
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


# Note: The StructuredLogger class has been replaced by FileHandler.
# This file now only contains the data classes that are still used by other components.
