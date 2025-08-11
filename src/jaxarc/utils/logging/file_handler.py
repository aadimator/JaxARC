"""Synchronous file logging handler for episode data.

This module provides a simplified file logging handler that replaces the complex
async logging system with straightforward synchronous file operations. It reuses
existing serialization utilities for JAX arrays and provides both JSON and pickle
output formats.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

from loguru import logger

from ..serialization_utils import (
    serialize_log_step,
)


class FileHandler:
    """Synchronous file logging handler for episode data.

    This handler provides simple, synchronous file writing for episode data
    with support for JSON and pickle formats. It reuses existing serialization
    utilities for JAX arrays and maintains compatibility with the existing
    configuration system.

    Note: This is not an equinox.Module because it needs mutable state for
    accumulating episode data across multiple log_step calls.
    """

    def __init__(self, config):
        """Initialize the file handler.

        Args:
            config: Configuration object with storage settings
        """
        # Extract output directory from config
        if hasattr(config, "storage") and hasattr(config.storage, "base_output_dir"):
            base_dir = config.storage.base_output_dir
            logs_subdir = getattr(config.storage, "logs_dir", "logs")
            self.output_dir = Path(base_dir) / logs_subdir
        elif hasattr(config, "debug") and hasattr(config.debug, "output_dir"):
            # Fallback for legacy config structure
            self.output_dir = Path(config.debug.output_dir)
        else:
            # Default fallback
            self.output_dir = Path("outputs/logs")

        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_episode_data = {}

        logger.info(f"FileHandler initialized with output directory: {self.output_dir}")

    def log_task_start(self, task_data: dict[str, Any]) -> None:
        """Log task information at the start of an episode.

        Args:
            task_data: Dictionary containing task information including:
                - task_id: Task identifier
                - task_object: The JaxArcTask object
                - episode_num: Episode number
                - num_train_pairs: Number of training pairs
                - num_test_pairs: Number of test pairs
                - task_stats: Additional task statistics
                - show_test: Whether to show test examples (not used by FileHandler)
        """
        # Store task information in current episode data
        self.current_episode_data["task_info"] = {
            "task_id": task_data.get("task_id"),
            "episode_num": task_data.get("episode_num"),
            "num_train_pairs": task_data.get("num_train_pairs", 0),
            "num_test_pairs": task_data.get("num_test_pairs", 0),
            "task_stats": task_data.get("task_stats", {}),
        }

        # Serialize task object if provided
        task_object = task_data.get("task_object")
        if task_object is not None:
            try:
                # Store basic task structure information
                self.current_episode_data["task_info"]["task_structure"] = {
                    "input_grid_shapes": [],
                    "output_grid_shapes": [],
                    "max_colors_used": 0,
                }

                # Extract grid shapes and color information
                for i in range(getattr(task_object, "num_train_pairs", 0)):
                    if hasattr(task_object, "input_grids_examples"):
                        input_grid = task_object.input_grids_examples[i]
                        self.current_episode_data["task_info"]["task_structure"][
                            "input_grid_shapes"
                        ].append(
                            list(input_grid.shape)
                            if hasattr(input_grid, "shape")
                            else [0, 0]
                        )

                    if hasattr(task_object, "output_grids_examples"):
                        output_grid = task_object.output_grids_examples[i]
                        self.current_episode_data["task_info"]["task_structure"][
                            "output_grid_shapes"
                        ].append(
                            list(output_grid.shape)
                            if hasattr(output_grid, "shape")
                            else [0, 0]
                        )

            except Exception as e:
                logger.warning(f"Failed to serialize task structure: {e}")

        logger.debug(
            f"Logged task start for task {task_data.get('task_id', 'unknown')}"
        )

    def log_step(self, step_data: dict[str, Any]) -> None:
        """Log step data to current episode.

        Args:
            step_data: Dictionary containing step information including:
                - step_num: Step number
                - before_state: State before action
                - after_state: State after action
                - action: Action taken
                - reward: Reward received
                - info: Additional information dictionary
        """
        if "steps" not in self.current_episode_data:
            self.current_episode_data["steps"] = []

        # Serialize step data using existing utilities
        serialized_step = serialize_log_step(step_data)
        self.current_episode_data["steps"].append(serialized_step)

        logger.debug(f"Logged step {step_data.get('step_num', 'unknown')}")

    def log_episode_summary(self, summary_data: dict[str, Any]) -> None:
        """Save complete episode data to file.

        Args:
            summary_data: Dictionary containing episode summary including:
                - episode_num: Episode number
                - total_steps: Total number of steps
                - total_reward: Total reward accumulated
                - final_similarity: Final similarity score
                - task_id: Task identifier
                - success: Whether episode was successful
        """
        episode_num = summary_data.get("episode_num", 0)

        # Combine step data with summary
        complete_episode = {
            **summary_data,
            **self.current_episode_data,
            "timestamp": time.time(),
            "config_hash": self._get_config_hash(),
        }

        # Save in both formats for different use cases
        self._save_json(complete_episode, episode_num)
        self._save_pickle(complete_episode, episode_num)

        logger.info(
            f"Saved episode {episode_num}: "
            f"{len(self.current_episode_data.get('steps', []))} steps, "
            f"reward: {summary_data.get('total_reward', 0):.3f}"
        )

        # Reset for next episode
        self.current_episode_data = {}

    def log_aggregated_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log aggregated batch metrics to file.

        Args:
            metrics: Dictionary of aggregated metrics from batch processing
            step: Current training step/update number
        """
        try:
            # Create batch metrics file if it doesn't exist
            batch_metrics_file = self.output_dir / "batch_metrics.jsonl"

            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Create log entry with timestamp and step information
            log_entry = {"timestamp": time.time(), "step": step, "metrics": metrics}

            # Append to JSONL file (one JSON object per line)
            with Path(batch_metrics_file).open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")

            logger.debug(f"Logged batch metrics to {batch_metrics_file} at step {step}")

        except Exception as e:
            # Handle file writing errors gracefully
            logger.warning(f"File batch logging failed: {e}")

    def log_evaluation_summary(self, eval_data: dict[str, Any]) -> None:
        """Persist evaluation summary to a dedicated JSON file.

        Creates/overwrites a file named evaluation_summary.json in the
        output directory with the provided evaluation data plus timestamp.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "evaluation_summary.json"
            payload = {
                "timestamp": time.time(),
                "evaluation": eval_data,
                "config_hash": self._get_config_hash(),
            }
            with Path(path).open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.info(f"Saved evaluation summary to {path}")
        except Exception as e:
            logger.warning(f"Failed to write evaluation summary: {e}")

    def close(self) -> None:
        """Clean shutdown - save any pending data."""
        if self.current_episode_data:
            # Save incomplete episode data
            try:
                # Ensure output directory exists
                self.output_dir.mkdir(parents=True, exist_ok=True)
                incomplete_path = self.output_dir / "incomplete_episode.json"
                with Path(incomplete_path).open("w", encoding="utf-8") as f:
                    json.dump(self.current_episode_data, f, indent=2, default=str)
                logger.info(f"Saved incomplete episode data to {incomplete_path}")
            except Exception as e:
                logger.error(f"Failed to save incomplete episode data: {e}")

        logger.info("FileHandler shutdown complete")

    def _save_json(self, episode_data: dict[str, Any], episode_num: int) -> None:
        """Save episode data as JSON file.

        Args:
            episode_data: Complete episode data
            episode_num: Episode number for filename
        """
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{episode_num:04d}_{timestamp_str}.json"
        filepath = self.output_dir / filename

        try:
            with Path(filepath).open("w", encoding="utf-8") as f:
                json.dump(episode_data, f, indent=2, default=str)
            logger.debug(f"Saved JSON episode data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save JSON episode data: {e}")

    def _save_pickle(self, episode_data: dict[str, Any], episode_num: int) -> None:
        """Save episode data as pickle file.

        Args:
            episode_data: Complete episode data
            episode_num: Episode number for filename
        """
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{episode_num:04d}_{timestamp_str}.pkl"
        filepath = self.output_dir / filename

        try:
            with Path(filepath).open("wb") as f:
                pickle.dump(episode_data, f)
            logger.debug(f"Saved pickle episode data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save pickle episode data: {e}")

    def _get_config_hash(self) -> str:
        """Generate a hash of the current configuration.

        Returns:
            String hash of configuration for reproducibility tracking
        """
        try:
            import hashlib

            config_str = str(self.config)
            return hashlib.md5(config_str.encode()).hexdigest()[:8]
        except Exception:
            return "unknown"
