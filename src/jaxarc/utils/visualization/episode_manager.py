"""Episode management system for JaxARC visualization.

This module provides episode-based storage management, directory organization,
and cleanup policies for visualization data.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import chex
from loguru import logger


@chex.dataclass
class EpisodeConfig:
    """Configuration for episode management and storage.

    This dataclass defines all settings for organizing and managing
    episode-based visualization storage with validation and serialization.
    """

    # Directory structure settings
    base_output_dir: str = "outputs/episodes"
    run_name: str | None = None  # Auto-generated if None
    episode_dir_format: str = "episode_{episode:04d}"
    step_file_format: str = "step_{step:03d}"

    # Storage limits and policies
    max_episodes_per_run: int = 1000
    cleanup_policy: Literal["oldest_first", "size_based", "manual"] = "size_based"
    max_storage_gb: float = 10.0

    # File management settings
    create_run_subdirs: bool = True
    preserve_empty_dirs: bool = False
    compress_old_episodes: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate all configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate directory paths
        if not self.base_output_dir or not isinstance(self.base_output_dir, str):
            raise ValueError("base_output_dir must be a non-empty string")

        # Validate format strings
        try:
            self.episode_dir_format.format(episode=1)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid episode_dir_format: {e}") from e

        try:
            self.step_file_format.format(step=1)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid step_file_format: {e}") from e

        # Validate numeric limits
        if self.max_episodes_per_run <= 0:
            raise ValueError("max_episodes_per_run must be positive")

        if self.max_storage_gb <= 0:
            raise ValueError("max_storage_gb must be positive")

        # Validate cleanup policy
        valid_policies = {"oldest_first", "size_based", "manual"}
        if self.cleanup_policy not in valid_policies:
            raise ValueError(f"cleanup_policy must be one of {valid_policies}")

        # Validate run_name if provided
        if self.run_name is not None:
            if not isinstance(self.run_name, str) or not self.run_name.strip():
                raise ValueError("run_name must be a non-empty string if provided")

            # Check for invalid characters in run_name
            invalid_chars = set('<>:"/\\|?*')
            if any(char in self.run_name for char in invalid_chars):
                raise ValueError(
                    f"run_name contains invalid characters: {invalid_chars}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "base_output_dir": self.base_output_dir,
            "run_name": self.run_name,
            "episode_dir_format": self.episode_dir_format,
            "step_file_format": self.step_file_format,
            "max_episodes_per_run": self.max_episodes_per_run,
            "cleanup_policy": self.cleanup_policy,
            "max_storage_gb": self.max_storage_gb,
            "create_run_subdirs": self.create_run_subdirs,
            "preserve_empty_dirs": self.preserve_empty_dirs,
            "compress_old_episodes": self.compress_old_episodes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeConfig:
        """Create configuration from dictionary.

        Args:
            data: Dictionary containing configuration parameters

        Returns:
            EpisodeConfig instance

        Raises:
            ValueError: If required keys are missing or invalid
        """
        # Extract known fields, ignoring unknown ones for forward compatibility
        known_fields = {
            "base_output_dir",
            "run_name",
            "episode_dir_format",
            "step_file_format",
            "max_episodes_per_run",
            "cleanup_policy",
            "max_storage_gb",
            "create_run_subdirs",
            "preserve_empty_dirs",
            "compress_old_episodes",
        }

        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        try:
            return cls(**filtered_data)
        except TypeError as e:
            raise ValueError(f"Invalid configuration data: {e}") from e

    def save_to_file(self, file_path: Path | str) -> None:
        """Save configuration to JSON file.

        Args:
            file_path: Path where to save the configuration

        Raises:
            OSError: If file cannot be written
        """
        file_path = Path(file_path)

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        except OSError as e:
            logger.error(f"Failed to save config to {file_path}: {e}")
            raise

    @classmethod
    def load_from_file(cls, file_path: Path | str) -> EpisodeConfig:
        """Load configuration from JSON file.

        Args:
            file_path: Path to the configuration file

        Returns:
            EpisodeConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file contains invalid configuration
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load config from {file_path}: {e}") from e

        return cls.from_dict(data)

    def get_base_path(self) -> Path:
        """Get the base output directory as a Path object.

        Returns:
            Path object for the base output directory
        """
        return Path(self.base_output_dir).expanduser().resolve()

    def generate_run_name(self) -> str:
        """Generate a timestamped run name if none is provided.

        Returns:
            Generated run name with timestamp
        """
        if self.run_name is not None:
            return self.run_name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"

    def validate_storage_path(self, path: Path) -> bool:
        """Validate that a storage path is accessible and writable.

        Args:
            path: Path to validate

        Returns:
            True if path is valid and writable
        """
        try:
            # Check if path exists or can be created
            path.mkdir(parents=True, exist_ok=True)

            # Test write permissions
            test_file = path / ".write_test"
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink()

            return True
        except (OSError, PermissionError):
            return False

    def estimate_storage_usage(self, path: Path) -> float:
        """Estimate storage usage in GB for a given path.

        Args:
            path: Path to analyze

        Returns:
            Storage usage in GB
        """
        if not path.exists():
            return 0.0

        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    file_path = Path(dirpath) / filename
                    try:
                        total_size += file_path.stat().st_size
                    except (OSError, FileNotFoundError):
                        # Skip files that can't be accessed
                        continue
        except (OSError, PermissionError):
            logger.warning(f"Could not access some files in {path}")

        return total_size / (1024**3)  # Convert bytes to GB


class EpisodeManager:
    """Manages episode-based storage and organization.

    This class handles directory creation, file organization, and cleanup
    for episode-based visualization data storage.
    """

    def __init__(self, config: EpisodeConfig):
        """Initialize episode manager with configuration.

        Args:
            config: Episode configuration settings
        """
        self.config = config
        self.current_run_dir: Path | None = None
        self.current_episode_dir: Path | None = None
        self.current_run_name: str | None = None
        self.current_episode_num: int | None = None

        # Validate base directory on initialization
        base_path = self.config.get_base_path()
        if not self.config.validate_storage_path(base_path):
            raise ValueError(f"Cannot access or write to base directory: {base_path}")

    def start_new_run(self, run_name: str | None = None) -> Path:
        """Start a new training run with timestamped directory.

        Args:
            run_name: Optional custom run name. If None, uses config or generates one.

        Returns:
            Path to the created run directory

        Raises:
            OSError: If directory cannot be created
            ValueError: If run_name is invalid
        """
        # Use provided name, config name, or generate one
        if run_name is not None:
            if not isinstance(run_name, str) or not run_name.strip():
                raise ValueError("run_name must be a non-empty string")
            self.current_run_name = run_name.strip()
        else:
            self.current_run_name = self.config.generate_run_name()

        # Create run directory
        base_path = self.config.get_base_path()
        self.current_run_dir = base_path / self.current_run_name

        try:
            self.current_run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create run directory {self.current_run_dir}: {e}")
            raise

        # Save configuration to run directory
        config_path = self.current_run_dir / "episode_config.json"
        self.config.save_to_file(config_path)

        # Reset episode tracking
        self.current_episode_dir = None
        self.current_episode_num = None

        logger.info(
            f"Started new run: {self.current_run_name} at {self.current_run_dir}"
        )
        return self.current_run_dir

    def start_new_episode(self, episode_num: int) -> Path:
        """Start a new episode within the current run.

        Args:
            episode_num: Episode number (must be non-negative)

        Returns:
            Path to the created episode directory

        Raises:
            ValueError: If no run is active or episode_num is invalid
            OSError: If directory cannot be created
        """
        if self.current_run_dir is None:
            raise ValueError("No active run. Call start_new_run() first.")

        if episode_num < 0:
            raise ValueError("episode_num must be non-negative")

        if episode_num >= self.config.max_episodes_per_run:
            raise ValueError(
                f"episode_num {episode_num} exceeds max_episodes_per_run {self.config.max_episodes_per_run}"
            )

        # Create episode directory
        episode_dir_name = self.config.episode_dir_format.format(episode=episode_num)
        self.current_episode_dir = self.current_run_dir / episode_dir_name

        try:
            self.current_episode_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(
                f"Failed to create episode directory {self.current_episode_dir}: {e}"
            )
            raise

        self.current_episode_num = episode_num

        logger.debug(f"Started episode {episode_num} at {self.current_episode_dir}")
        return self.current_episode_dir

    def get_step_path(self, step_num: int, file_type: str = "svg") -> Path:
        """Get file path for a specific step visualization.

        Args:
            step_num: Step number (must be non-negative)
            file_type: File extension (without dot)

        Returns:
            Path for the step file

        Raises:
            ValueError: If no episode is active or step_num is invalid
        """
        if self.current_episode_dir is None:
            raise ValueError("No active episode. Call start_new_episode() first.")

        if step_num < 0:
            raise ValueError("step_num must be non-negative")

        step_filename = self.config.step_file_format.format(step=step_num)
        return self.current_episode_dir / f"{step_filename}.{file_type}"

    def get_episode_summary_path(self, file_type: str = "svg") -> Path:
        """Get file path for episode summary visualization.

        Args:
            file_type: File extension (without dot)

        Returns:
            Path for the episode summary file

        Raises:
            ValueError: If no episode is active
        """
        if self.current_episode_dir is None:
            raise ValueError("No active episode. Call start_new_episode() first.")

        return self.current_episode_dir / f"summary.{file_type}"

    def get_current_run_info(self) -> dict[str, Any]:
        """Get information about the current run.

        Returns:
            Dictionary with run information
        """
        return {
            "run_name": self.current_run_name,
            "run_dir": str(self.current_run_dir) if self.current_run_dir else None,
            "episode_num": self.current_episode_num,
            "episode_dir": str(self.current_episode_dir)
            if self.current_episode_dir
            else None,
        }

    def list_episodes_in_run(
        self, run_dir: Path | None = None
    ) -> list[tuple[int, Path]]:
        """List all episodes in a run directory.

        Args:
            run_dir: Run directory to scan. Uses current run if None.

        Returns:
            List of (episode_number, episode_path) tuples, sorted by episode number
        """
        if run_dir is None:
            run_dir = self.current_run_dir

        if run_dir is None or not run_dir.exists():
            return []

        episodes = []
        for item in run_dir.iterdir():
            if item.is_dir():
                # Try to extract episode number from directory name
                try:
                    # This is a simple approach - could be made more robust
                    if item.name.startswith("episode_"):
                        episode_str = item.name.replace("episode_", "")
                        episode_num = int(episode_str)
                        episodes.append((episode_num, item))
                except ValueError:
                    # Skip directories that don't match expected format
                    continue

        return sorted(episodes)

    def cleanup_old_data(self) -> None:
        """Clean up old data based on configured policy.

        This method implements the cleanup policy specified in the configuration
        to manage storage usage and maintain the episode limit.
        """
        if self.config.cleanup_policy == "manual":
            logger.debug("Cleanup policy is manual - skipping automatic cleanup")
            return

        base_path = self.config.get_base_path()
        if not base_path.exists():
            return

        current_usage = self.config.estimate_storage_usage(base_path)

        if current_usage <= self.config.max_storage_gb:
            logger.debug(
                f"Storage usage {current_usage:.2f}GB is within limit {self.config.max_storage_gb}GB"
            )
            return

        logger.info(
            f"Storage usage {current_usage:.2f}GB exceeds limit {self.config.max_storage_gb}GB - starting cleanup"
        )

        if self.config.cleanup_policy == "oldest_first":
            self._cleanup_oldest_first(base_path)
        elif self.config.cleanup_policy == "size_based":
            self._cleanup_size_based(base_path)

    def _cleanup_oldest_first(self, base_path: Path) -> None:
        """Clean up oldest runs first until under storage limit.

        Args:
            base_path: Base directory to clean up
        """
        # Get all run directories with their modification times
        runs = []
        for item in base_path.iterdir():
            if item.is_dir():
                try:
                    mtime = item.stat().st_mtime
                    runs.append((mtime, item))
                except OSError:
                    continue

        # Sort by modification time (oldest first)
        runs.sort()

        for mtime, run_dir in runs:
            current_usage = self.config.estimate_storage_usage(base_path)
            if current_usage <= self.config.max_storage_gb:
                break

            # Don't delete current run
            if run_dir == self.current_run_dir:
                continue

            logger.info(f"Removing old run directory: {run_dir}")
            try:
                shutil.rmtree(run_dir)
            except OSError as e:
                logger.error(f"Failed to remove {run_dir}: {e}")

    def _cleanup_size_based(self, base_path: Path) -> None:
        """Clean up largest runs first until under storage limit.

        Args:
            base_path: Base directory to clean up
        """
        # Get all run directories with their sizes
        runs = []
        for item in base_path.iterdir():
            if item.is_dir():
                try:
                    size = self.config.estimate_storage_usage(item)
                    runs.append((size, item))
                except OSError:
                    continue

        # Sort by size (largest first)
        runs.sort(reverse=True)

        for size, run_dir in runs:
            current_usage = self.config.estimate_storage_usage(base_path)
            if current_usage <= self.config.max_storage_gb:
                break

            # Don't delete current run
            if run_dir == self.current_run_dir:
                continue

            logger.info(f"Removing large run directory ({size:.2f}GB): {run_dir}")
            try:
                shutil.rmtree(run_dir)
            except OSError as e:
                logger.error(f"Failed to remove {run_dir}: {e}")

    def force_cleanup_run(self, run_name: str) -> bool:
        """Force cleanup of a specific run directory.

        Args:
            run_name: Name of the run to clean up

        Returns:
            True if cleanup was successful, False otherwise
        """
        base_path = self.config.get_base_path()
        run_dir = base_path / run_name

        if not run_dir.exists():
            logger.warning(f"Run directory does not exist: {run_dir}")
            return False

        # Don't delete current run
        if run_dir == self.current_run_dir:
            logger.warning(f"Cannot delete current active run: {run_name}")
            return False

        try:
            shutil.rmtree(run_dir)
            logger.info(f"Successfully removed run directory: {run_dir}")
            return True
        except OSError as e:
            logger.error(f"Failed to remove run directory {run_dir}: {e}")
            return False
