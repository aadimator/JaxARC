"""
RL-specific visualization and episode management.

Handles training visualization, episode tracking, and RL metrics display.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import chex
import jax.numpy as jnp
import numpy as np
from loguru import logger

from jaxarc.envs.actions import Action
from jaxarc.envs.grid_operations import get_operation_display_text

from .core import (
    ARC_COLOR_PALETTE,
    _extract_grid_data,
    _extract_valid_region,
    add_change_highlighting,
    add_selection_visualization_overlay,
    get_info_metric,
)

if TYPE_CHECKING:
    from jaxarc.types import Grid


# ============================================================================
# SECTION: Episode Management (from episode_manager.py)
# ============================================================================


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


# ============================================================================
# SECTION: RL Visualization (from rl_visualization.py)
# ============================================================================


def get_operation_display_name(
    operation_id: int, action_data: Dict[str, Any] = None
) -> str:
    """Get human-readable operation name from operation ID with context."""
    return get_operation_display_text(operation_id)


def draw_rl_step_svg_enhanced(
    before_grid: Grid,
    after_grid: Grid,
    action: Any,  # Can be Action or dict
    reward: float,
    info: Dict[str, Any],
    step_num: int,
    operation_name: str = "",
    changed_cells: Optional[jnp.ndarray] = None,
    config: Optional[Any] = None,
    max_width: float = 1400.0,
    max_height: float = 700.0,
    task_id: str = "",
    task_pair_index: int = 0,
    total_task_pairs: int = 1,
) -> str:
    """Generate enhanced SVG visualization of a single RL step with more information.

    This enhanced version shows:
    - Before and after grids with improved styling
    - Action selection highlighting
    - Changed cell highlighting
    - Reward information and metrics
    - Operation name and details
    - Step metadata
    - Task context information

    Args:
        before_grid: Grid state before the action
        after_grid: Grid state after the action
        action: Action object or dictionary
        reward: Reward received for this step
        info: Additional information dictionary or StepInfo object
        step_num: Step number in the episode
        operation_name: Human-readable operation name
        changed_cells: Optional mask of cells that changed
        config: Optional visualization configuration
        max_width: Maximum width of the entire visualization
        max_height: Maximum height of the entire visualization
        task_id: Task identifier for context
        task_pair_index: Current task pair index
        total_task_pairs: Total number of task pairs

    Returns:
        SVG string containing the enhanced visualization
    """
    import drawsvg as draw

    # Get color palette from config or use default
    if config and hasattr(config, "get_color_palette"):
        color_palette = config.get_color_palette()
    else:
        color_palette = ARC_COLOR_PALETTE

    # Layout parameters
    top_padding = 100
    bottom_padding = 50
    side_padding = 50
    grid_spacing = 180
    grid_max_width = 280
    grid_max_height = 280

    # Calculate total dimensions
    total_width = 2 * grid_max_width + grid_spacing + 2 * side_padding
    total_height = grid_max_height + top_padding + bottom_padding

    # Create main drawing with background
    drawing = draw.Drawing(total_width, total_height)
    drawing.append(draw.Rectangle(0, 0, total_width, total_height, fill="#f8f9fa"))

    # Add enhanced title with step info
    title_text = f"Step {step_num}"
    if operation_name:
        title_text += f" - {operation_name}"

    drawing.append(
        draw.Text(
            title_text,
            font_size=28,
            x=total_width / 2,
            y=40,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Add task context information
    task_context_text = ""
    if task_id:
        task_context_text = f"Task: {task_id}"
    if total_task_pairs > 1:
        if task_context_text:
            task_context_text += f" | Pair {task_pair_index + 1}/{total_task_pairs}"
        else:
            task_context_text = f"Pair {task_pair_index + 1}/{total_task_pairs}"

    if task_context_text:
        drawing.append(
            draw.Text(
                task_context_text,
                font_size=16,
                x=total_width / 2,
                y=65,
                text_anchor="middle",
                font_family="Anuphan",
                font_weight="400",
                fill="#6c757d",
            )
        )

    # Add reward information (adjusted position for task context)
    reward_color = "#27ae60" if reward > 0 else "#e74c3c" if reward < 0 else "#95a5a6"
    reward_text = f"Reward: {reward:.3f}"
    reward_y = 85 if task_context_text else 70
    drawing.append(
        draw.Text(
            reward_text,
            font_size=20,
            x=total_width / 2,
            y=reward_y,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="500",
            fill=reward_color,
        )
    )

    # Grid positions
    before_x = side_padding
    after_x = side_padding + grid_max_width + grid_spacing
    grids_y = top_padding

    # Helper function to draw enhanced grid
    def draw_enhanced_grid(
        grid: Grid,
        x: float,
        y: float,
        grid_label: str,
        selection_mask: Optional[np.ndarray] = None,
        highlight_changes: bool = False,
        changed_cells: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """Draw an enhanced grid with overlays and styling."""
        grid_data, grid_mask = _extract_grid_data(grid)

        if grid_mask is not None:
            grid_mask = np.asarray(grid_mask)

        # Extract valid region
        valid_grid, (start_row, start_col), (height, width) = _extract_valid_region(
            grid_data, grid_mask
        )

        if height == 0 or width == 0:
            return 0, 0

        # Calculate cell size to fit within max dimensions
        cell_size = min(grid_max_width / width, grid_max_height / height)
        actual_width = width * cell_size
        actual_height = height * cell_size

        # Center the grid within the allocated space
        grid_x = x + (grid_max_width - actual_width) / 2
        grid_y = y + (grid_max_height - actual_height) / 2

        # Draw grid background
        drawing.append(
            draw.Rectangle(
                grid_x - 5,
                grid_y - 5,
                actual_width + 10,
                actual_height + 10,
                fill="white",
                stroke="#dee2e6",
                stroke_width=1,
                rx=5,
            )
        )

        # Draw grid cells
        for i in range(height):
            for j in range(width):
                color_val = int(valid_grid[i, j])

                # Check if cell is valid
                is_valid = True
                if grid_mask is not None:
                    actual_row = start_row + i
                    actual_col = start_col + j
                    if (
                        actual_row < grid_mask.shape[0]
                        and actual_col < grid_mask.shape[1]
                    ):
                        is_valid = grid_mask[actual_row, actual_col]

                if is_valid and 0 <= color_val < len(color_palette.keys()):
                    fill_color = color_palette.get(color_val, "white")
                else:
                    fill_color = "#CCCCCC"

                cell_x = grid_x + j * cell_size
                cell_y = grid_y + i * cell_size

                # Draw cell
                drawing.append(
                    draw.Rectangle(
                        cell_x,
                        cell_y,
                        cell_size,
                        cell_size,
                        fill=fill_color,
                        stroke="#6c757d",
                        stroke_width=0.5,
                    )
                )

        # Add changed cell highlighting after all cells are drawn
        if highlight_changes and changed_cells is not None:
            add_change_highlighting(
                drawing,
                changed_cells,
                grid_x,
                grid_y,
                cell_size,
                start_row,
                start_col,
                height,
                width,
            )

        # Add selection overlay if provided
        if selection_mask is not None and selection_mask.any():
            add_selection_visualization_overlay(
                drawing,
                selection_mask,
                grid_x,
                grid_y,
                cell_size,
                start_row,
                start_col,
                height,
                width,
                selection_color="#00FFFF",  # Bright cyan - very visible
                selection_opacity=0.4,
                border_width=3,
            )

        # Add enhanced grid border
        drawing.append(
            draw.Rectangle(
                grid_x - 3,
                grid_y - 3,
                actual_width + 6,
                actual_height + 6,
                fill="none",
                stroke="#495057",
                stroke_width=2,
                rx=3,
            )
        )

        # Add enhanced grid label with background
        label_bg_width = len(grid_label) * 12 + 20
        drawing.append(
            draw.Rectangle(
                grid_x - 5,
                grid_y + actual_height + 15,
                label_bg_width,
                25,
                fill="#e9ecef",
                stroke="#dee2e6",
                stroke_width=1,
                rx=3,
            )
        )

        drawing.append(
            draw.Text(
                grid_label,
                font_size=16,
                x=grid_x + 5,
                y=grid_y + actual_height + 32,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="600",
                fill="#495057",
            )
        )

        return actual_width, actual_height

    # Extract selection mask from action
    selection_mask = None
    if isinstance(action, Action):
        selection_mask = np.asarray(action.selection)
    elif isinstance(action, tuple) and len(action) >= 3:
        # Handle tuple actions from action wrappers
        grid_height, grid_width = before_grid.shape
        selection_mask = np.zeros((grid_height, grid_width), dtype=bool)

        if len(action) == 3:
            # PointActionWrapper: (operation, row, col)
            _, row, col = action[0], action[1], action[2]
            # Clip coordinates to valid range and set the selected point
            row = max(0, min(int(row), grid_height - 1))
            col = max(0, min(int(col), grid_width - 1))
            selection_mask[row, col] = True
        elif len(action) == 5:
            # BboxActionWrapper: (operation, r1, c1, r2, c2)
            _, r1, c1, r2, c2 = action[0], action[1], action[2], action[3], action[4]
            # Clip coordinates to valid range
            r1 = max(0, min(int(r1), grid_height - 1))
            c1 = max(0, min(int(c1), grid_width - 1))
            r2 = max(0, min(int(r2), grid_height - 1))
            c2 = max(0, min(int(c2), grid_width - 1))
            # Ensure proper ordering (min, max)
            min_r, max_r = min(r1, r2), max(r1, r2)
            min_c, max_c = min(c1, c2), max(c1, c2)
            # Set rectangular region (inclusive bounds)
            selection_mask[min_r : max_r + 1, min_c : max_c + 1] = True
    elif isinstance(action, dict):  # Fallback for old dictionary format
        if "selection" in action:
            selection_mask = np.asarray(action["selection"])

    # Draw before grid with selection overlay
    draw_enhanced_grid(
        before_grid, before_x, grids_y, "Before State", selection_mask=selection_mask
    )

    # Draw after grid with change highlighting
    draw_enhanced_grid(
        after_grid,
        after_x,
        grids_y,
        "After State",
        highlight_changes=True,
        changed_cells=changed_cells,
    )

    # Add enhanced arrow between grids
    arrow_y = grids_y + grid_max_height / 2
    arrow_start_x = before_x + grid_max_width + 30
    arrow_end_x = after_x - 30

    # Arrow shaft
    drawing.append(
        draw.Line(
            arrow_start_x,
            arrow_y,
            arrow_end_x,
            arrow_y,
            stroke="#6c757d",
            stroke_width=3,
        )
    )

    # Arrow head
    drawing.append(
        draw.Lines(
            arrow_end_x - 15,
            arrow_y - 10,
            arrow_end_x - 15,
            arrow_y + 10,
            arrow_end_x,
            arrow_y,
            close=True,
            fill="#6c757d",
        )
    )

    return drawing.as_svg()


def draw_rl_step_svg(
    before_grid: Grid,
    after_grid: Grid,
    action: Dict[str, Any],
    reward: float,
    info: Dict[str, Any],
    step_num: int,
    operation_name: str = "",
    changed_cells: Optional[jnp.ndarray] = None,
    config: Optional[Any] = None,
    **kwargs,
) -> str:
    """Enhanced wrapper for draw_rl_step_svg_enhanced with backward compatibility."""
    return draw_rl_step_svg_enhanced(
        before_grid=before_grid,
        after_grid=after_grid,
        action=action,
        reward=reward,
        info=info,
        step_num=step_num,
        operation_name=operation_name,
        changed_cells=changed_cells,
        config=config,
        **kwargs,
    )


def save_rl_step_visualization(
    state: Any,  # ArcEnvState
    action: dict,
    next_state: Any,  # ArcEnvState
    output_dir: str = "output/rl_steps",
) -> None:
    """JAX callback function to save RL step visualization.

    This function is designed to be used with jax.debug.callback.

    Args:
        state: Environment state before the action
        action: Action dictionary with 'selection' and 'operation' keys
        next_state: Environment state after the action
        output_dir: Directory to save visualization files
    """
    from pathlib import Path

    from jaxarc.types import Grid

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create Grid objects (convert JAX arrays to numpy)
    before_grid = Grid(
        data=np.asarray(state.working_grid),
        mask=np.asarray(state.working_grid_mask),
    )
    after_grid = Grid(
        data=np.asarray(next_state.working_grid),
        mask=np.asarray(next_state.working_grid_mask),
    )

    # Extract action components
    # Note: This handles structured actions, dictionary format, and tuple format for visualization
    if hasattr(action, "operation"):
        operation_id = int(action.operation)
    elif isinstance(action, tuple) and len(action) >= 1:
        # Handle tuple actions from PointActionWrapper: (operation, row, col)
        operation_id = int(action[0])
    else:
        operation_id = int(action["operation"])  # Legacy format for visualization only
    step_number = int(state.step_count)

    # Create dummy reward and info for visualization
    reward = 0.0  # Placeholder since we don't have reward in this context
    info = {"step_count": step_number}  # Basic info

    # Generate visualization
    svg_content = draw_rl_step_svg(
        before_grid=before_grid,
        after_grid=after_grid,
        action=action,
        reward=reward,
        info=info,
        step_num=step_number,
    )

    # Save to file with zero-padded step number
    filename = f"step_{step_number:03d}.svg"
    filepath = Path(output_dir) / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg_content)

    # Log the save (will appear in console during execution)
    logger.info(f"Saved RL step visualization: {filepath}")


def create_action_summary_panel(
    action: Dict[str, Any],
    reward: float,
    info: Dict[str, Any],
    operation_name: str = "",
    width: float = 400,
    height: float = 100,
) -> str:
    """Create an action summary panel as SVG.

    Args:
        action: Action dictionary
        reward: Reward received
        info: Additional information
        operation_name: Human-readable operation name
        width: Panel width
        height: Panel height

    Returns:
        SVG string for the action summary panel
    """
    import drawsvg as draw

    drawing = draw.Drawing(width, height)

    # Panel background
    drawing.append(
        draw.Rectangle(
            0,
            0,
            width,
            height,
            fill="#ffffff",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Title
    drawing.append(
        draw.Text(
            "Action Summary",
            font_size=16,
            x=10,
            y=25,
            text_anchor="start",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Operation info
    if operation_name:
        drawing.append(
            draw.Text(
                f"Operation: {operation_name}",
                font_size=14,
                x=10,
                y=45,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="400",
                fill="#495057",
            )
        )

    # Reward info
    reward_color = "#27ae60" if reward > 0 else "#e74c3c" if reward < 0 else "#95a5a6"
    drawing.append(
        draw.Text(
            f"Reward: {reward:.3f}",
            font_size=14,
            x=10,
            y=65,
            text_anchor="start",
            font_family="Anuphan",
            font_weight="500",
            fill=reward_color,
        )
    )

    # Additional info - check both direct and nested metrics
    similarity_val = get_info_metric(info, "similarity")

    if similarity_val is not None:
        drawing.append(
            draw.Text(
                f"Similarity: {similarity_val:.3f}",
                font_size=12,
                x=10,
                y=85,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="400",
                fill="#6c757d",
            )
        )

    return drawing.as_svg()


def create_metrics_visualization(
    metrics: Dict[str, float],
    width: float = 300,
    height: float = 200,
) -> str:
    """Create a metrics visualization panel.

    Args:
        metrics: Dictionary of metric names to values
        width: Panel width
        height: Panel height

    Returns:
        SVG string for the metrics panel
    """
    import drawsvg as draw

    drawing = draw.Drawing(width, height)

    # Panel background
    drawing.append(
        draw.Rectangle(
            0,
            0,
            width,
            height,
            fill="#ffffff",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Title
    drawing.append(
        draw.Text(
            "Step Metrics",
            font_size=16,
            x=10,
            y=25,
            text_anchor="start",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Display metrics
    y_pos = 50
    for name, value in metrics.items():
        # Metric name
        drawing.append(
            draw.Text(
                f"{name}:",
                font_size=12,
                x=10,
                y=y_pos,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="500",
                fill="#495057",
            )
        )

        # Metric value
        drawing.append(
            draw.Text(
                f"{value:.3f}",
                font_size=12,
                x=width - 10,
                y=y_pos,
                text_anchor="end",
                font_family="Anuphan",
                font_weight="400",
                fill="#6c757d",
            )
        )

        y_pos += 20

        if y_pos > height - 20:
            break

    return drawing.as_svg()
