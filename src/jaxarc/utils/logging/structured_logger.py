"""Structured logging system for episode data.

This module provides structured logging capabilities for episode replay and analysis,
with support for JSON serialization and compression.
"""

from __future__ import annotations

import gzip
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chex
from loguru import logger

from ..visualization.async_logger import AsyncLogger, AsyncLoggerConfig


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


class StructuredLogger:
    """Structured logging system for episode data.

    This logger provides structured logging capabilities with support for
    episode replay, analysis, and multiple output formats.
    """

    def __init__(self, config: LoggingConfig):
        """Initialize the structured logger.

        Args:
            config: Configuration for structured logging
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Current episode being logged
        self.current_episode: Optional[EpisodeLogEntry] = None
        self.current_steps: List[StepLogEntry] = []

        # Async logger for performance
        self.async_logger: Optional[AsyncLogger] = None
        if config.async_logging:
            async_config = AsyncLoggerConfig(
                queue_size=1000,
                worker_threads=1,  # Single thread for structured logging
                batch_size=5,
                flush_interval=10.0,
                enable_compression=config.compression,
            )
            self.async_logger = AsyncLogger(async_config, self.output_dir)

        logger.info(f"StructuredLogger initialized with format: {config.log_format}")

    def start_episode(
        self,
        episode_num: int,
        task_id: str,
        config_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start logging a new episode.

        Args:
            episode_num: Episode number
            task_id: Identifier for the task
            config_hash: Hash of the configuration
            metadata: Additional metadata for the episode
        """
        if self.current_episode is not None:
            logger.warning(
                f"Starting new episode {episode_num} without ending previous episode {self.current_episode.episode_num}"
            )
            self.end_episode()

        self.current_episode = EpisodeLogEntry(
            episode_num=episode_num,
            start_timestamp=time.time(),
            end_timestamp=0.0,
            total_steps=0,
            total_reward=0.0,
            final_similarity=0.0,
            task_id=task_id,
            config_hash=config_hash,
            steps=[],
            metadata=metadata or {},
        )
        self.current_steps = []

        logger.debug(f"Started logging episode {episode_num} for task {task_id}")

    def log_step(
        self,
        step_num: int,
        before_state: Any,
        action: Dict[str, Any],
        after_state: Any,
        reward: float,
        info: Dict[str, Any],
        visualization_path: Optional[str] = None,
    ) -> None:
        """Log a single step.

        Args:
            step_num: Step number within the episode
            before_state: State before the action
            action: Action taken
            after_state: State after the action
            reward: Reward received
            info: Additional information
            visualization_path: Path to step visualization if available
        """
        if self.current_episode is None:
            logger.warning("Logging step without active episode")
            return

        # Serialize states based on configuration
        before_state_data = self._serialize_state(before_state)
        after_state_data = self._serialize_state(after_state)

        step_entry = StepLogEntry(
            step_num=step_num,
            timestamp=time.time(),
            before_state=before_state_data,
            action=action,
            after_state=after_state_data,
            reward=reward,
            info=info,
            visualization_path=visualization_path,
        )

        self.current_steps.append(step_entry)

        # Update episode totals
        object.__setattr__(
            self.current_episode, "total_steps", self.current_episode.total_steps + 1
        )
        object.__setattr__(
            self.current_episode,
            "total_reward",
            self.current_episode.total_reward + reward,
        )

        # Update final similarity if available in info
        if "similarity" in info:
            object.__setattr__(
                self.current_episode, "final_similarity", info["similarity"]
            )

    def end_episode(self, summary_visualization_path: Optional[str] = None) -> None:
        """End current episode and save log.

        Args:
            summary_visualization_path: Path to episode summary visualization
        """
        if self.current_episode is None:
            logger.warning("Ending episode without active episode")
            return

        # Finalize episode data
        object.__setattr__(self.current_episode, "end_timestamp", time.time())
        object.__setattr__(self.current_episode, "steps", self.current_steps.copy())
        object.__setattr__(
            self.current_episode,
            "summary_visualization_path",
            summary_visualization_path,
        )

        # Save episode log
        self._save_episode_log(self.current_episode)

        logger.info(
            f"Completed episode {self.current_episode.episode_num}: "
            f"{self.current_episode.total_steps} steps, "
            f"reward: {self.current_episode.total_reward:.3f}, "
            f"similarity: {self.current_episode.final_similarity:.3f}"
        )

        # Clear current episode
        self.current_episode = None
        self.current_steps = []

    def _serialize_state(self, state: Any) -> Dict[str, Any]:
        """Serialize state for logging.

        Args:
            state: State to serialize

        Returns:
            Serialized state data
        """
        if not self.config.include_full_states:
            # Only include essential state information for performance
            if hasattr(state, "grid") and hasattr(state, "step_count"):
                return {
                    "grid_shape": getattr(state.grid, "shape", None),
                    "step_count": state.step_count,
                    "type": type(state).__name__,
                }
            return {"type": type(state).__name__}

        # Full state serialization
        try:
            if hasattr(state, "__dict__"):
                return self._serialize_object(state.__dict__)
            return self._serialize_object(state)
        except Exception as e:
            logger.warning(f"Failed to serialize state: {e}")
            return {"serialization_error": str(e), "type": type(state).__name__}

    def _serialize_object(self, obj: Any) -> Any:
        """Recursively serialize objects for JSON compatibility.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable representation
        """
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._serialize_object(item) for item in obj]
        if isinstance(obj, dict):
            return {str(k): self._serialize_object(v) for k, v in obj.items()}
        if hasattr(obj, "tolist"):  # JAX/NumPy arrays
            return obj.tolist()
        if hasattr(obj, "__dict__"):
            return self._serialize_object(obj.__dict__)
        return str(obj)

    def _save_episode_log(self, episode: EpisodeLogEntry) -> None:
        """Save episode log to file.

        Args:
            episode: Episode data to save
        """
        if self.config.async_logging and self.async_logger:
            # Use async logging
            episode_data = self._episode_to_dict(episode)
            self.async_logger.log_entry("episode", episode_data, priority=1)
        else:
            # Synchronous logging
            self._save_episode_sync(episode)

    def _save_episode_sync(self, episode: EpisodeLogEntry) -> None:
        """Synchronously save episode log.

        Args:
            episode: Episode data to save
        """
        # Create episode-specific directory
        episode_dir = self.output_dir / f"episode_{episode.episode_num:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp_str = time.strftime(
            "%Y%m%d_%H%M%S", time.localtime(episode.start_timestamp)
        )
        filename = f"episode_{episode.episode_num:04d}_{timestamp_str}"

        if self.config.log_format == "json":
            filepath = episode_dir / f"{filename}.json"
            if self.config.compression:
                filepath = episode_dir / f"{filename}.json.gz"
            self._save_json(episode, filepath)
        elif self.config.log_format == "pickle":
            filepath = episode_dir / f"{filename}.pkl"
            if self.config.compression:
                filepath = episode_dir / f"{filename}.pkl.gz"
            self._save_pickle(episode, filepath)
        else:
            logger.error(f"Unsupported log format: {self.config.log_format}")

    def _save_json(self, episode: EpisodeLogEntry, filepath: Path) -> None:
        """Save episode as JSON.

        Args:
            episode: Episode data
            filepath: Output file path
        """
        episode_data = self._episode_to_dict(episode)

        try:
            if filepath.suffix == ".gz":
                with gzip.open(filepath, "wt", encoding="utf-8") as f:
                    json.dump(episode_data, f, indent=2, default=str)
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(episode_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save episode JSON: {e}")

    def _save_pickle(self, episode: EpisodeLogEntry, filepath: Path) -> None:
        """Save episode as pickle.

        Args:
            episode: Episode data
            filepath: Output file path
        """
        try:
            if filepath.suffix == ".gz":
                with gzip.open(filepath, "wb") as f:
                    pickle.dump(episode, f)
            else:
                with open(filepath, "wb") as f:
                    pickle.dump(episode, f)
        except Exception as e:
            logger.error(f"Failed to save episode pickle: {e}")

    def _episode_to_dict(self, episode: EpisodeLogEntry) -> Dict[str, Any]:
        """Convert episode to dictionary for serialization.

        Args:
            episode: Episode to convert

        Returns:
            Dictionary representation of episode
        """
        return {
            "episode_num": episode.episode_num,
            "start_timestamp": episode.start_timestamp,
            "end_timestamp": episode.end_timestamp,
            "total_steps": episode.total_steps,
            "total_reward": episode.total_reward,
            "final_similarity": episode.final_similarity,
            "task_id": episode.task_id,
            "config_hash": episode.config_hash,
            "summary_visualization_path": episode.summary_visualization_path,
            "metadata": episode.metadata,
            "steps": [self._step_to_dict(step) for step in episode.steps],
        }

    def _step_to_dict(self, step: StepLogEntry) -> Dict[str, Any]:
        """Convert step to dictionary for serialization.

        Args:
            step: Step to convert

        Returns:
            Dictionary representation of step
        """
        return {
            "step_num": step.step_num,
            "timestamp": step.timestamp,
            "before_state": step.before_state,
            "action": step.action,
            "after_state": step.after_state,
            "reward": step.reward,
            "info": step.info,
            "visualization_path": step.visualization_path,
        }

    def load_episode(self, episode_num: int) -> Optional[EpisodeLogEntry]:
        """Load episode data for replay/analysis.

        Args:
            episode_num: Episode number to load

        Returns:
            Episode data if found, None otherwise
        """
        episode_dir = self.output_dir / f"episode_{episode_num:04d}"
        if not episode_dir.exists():
            logger.warning(f"Episode directory not found: {episode_dir}")
            return None

        # Find episode file
        episode_files = list(episode_dir.glob(f"episode_{episode_num:04d}_*"))
        if not episode_files:
            logger.warning(f"No episode files found in {episode_dir}")
            return None

        # Use the most recent file
        episode_file = max(episode_files, key=lambda p: p.stat().st_mtime)

        try:
            if episode_file.suffix == ".gz":
                if ".json" in episode_file.name:
                    return self._load_json_episode(episode_file)
                if ".pkl" in episode_file.name:
                    return self._load_pickle_episode(episode_file)
            elif episode_file.suffix == ".json":
                return self._load_json_episode(episode_file)
            elif episode_file.suffix == ".pkl":
                return self._load_pickle_episode(episode_file)
            else:
                logger.error(f"Unknown episode file format: {episode_file}")
                return None
        except Exception as e:
            logger.error(f"Failed to load episode {episode_num}: {e}")
            return None

    def _load_json_episode(self, filepath: Path) -> EpisodeLogEntry:
        """Load episode from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Episode data
        """
        if filepath.suffix == ".gz":
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

        # Convert back to dataclasses
        steps = [
            StepLogEntry(
                step_num=step_data["step_num"],
                timestamp=step_data["timestamp"],
                before_state=step_data["before_state"],
                action=step_data["action"],
                after_state=step_data["after_state"],
                reward=step_data["reward"],
                info=step_data["info"],
                visualization_path=step_data.get("visualization_path"),
            )
            for step_data in data["steps"]
        ]

        return EpisodeLogEntry(
            episode_num=data["episode_num"],
            start_timestamp=data["start_timestamp"],
            end_timestamp=data["end_timestamp"],
            total_steps=data["total_steps"],
            total_reward=data["total_reward"],
            final_similarity=data["final_similarity"],
            task_id=data["task_id"],
            config_hash=data["config_hash"],
            steps=steps,
            summary_visualization_path=data.get("summary_visualization_path"),
            metadata=data.get("metadata", {}),
        )

    def _load_pickle_episode(self, filepath: Path) -> EpisodeLogEntry:
        """Load episode from pickle file.

        Args:
            filepath: Path to pickle file

        Returns:
            Episode data
        """
        if filepath.suffix == ".gz":
            with gzip.open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            with open(filepath, "rb") as f:
                return pickle.load(f)

    def list_episodes(self) -> List[int]:
        """List available episode numbers.

        Returns:
            List of episode numbers that have been logged
        """
        episode_dirs = [
            d
            for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        ]

        episode_nums = []
        for episode_dir in episode_dirs:
            try:
                episode_num = int(episode_dir.name.split("_")[1])
                episode_nums.append(episode_num)
            except (IndexError, ValueError):
                continue

        return sorted(episode_nums)

    def get_episode_summary(self, episode_num: int) -> Optional[Dict[str, Any]]:
        """Get summary information for an episode without loading full data.

        Args:
            episode_num: Episode number

        Returns:
            Episode summary or None if not found
        """
        episode = self.load_episode(episode_num)
        if episode is None:
            return None

        return {
            "episode_num": episode.episode_num,
            "total_steps": episode.total_steps,
            "total_reward": episode.total_reward,
            "final_similarity": episode.final_similarity,
            "task_id": episode.task_id,
            "duration": episode.end_timestamp - episode.start_timestamp,
            "metadata": episode.metadata,
        }

    def shutdown(self) -> None:
        """Shutdown the structured logger."""
        if self.current_episode is not None:
            logger.warning("Shutting down with active episode, ending it now")
            self.end_episode()

        if self.async_logger:
            self.async_logger.shutdown()

        logger.info("StructuredLogger shutdown complete")
