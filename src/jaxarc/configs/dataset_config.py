from __future__ import annotations

import equinox as eqx
from loguru import logger
from omegaconf import DictConfig

from .validation import (
    ConfigValidationError,
    validate_path_string,
    validate_positive_int,
    validate_string_choice,
)


class DatasetConfig(eqx.Module):
    """Dataset-specific settings and constraints.

    This config contains all dataset-related settings including grid constraints,
    color limits, task sampling, and dataset identification.
    """

    # Dataset identification
    dataset_name: str = "arc-agi-1"
    dataset_path: str = ""
    dataset_repo: str = ""
    parser_entry_point: str = "jaxarc.parsers:ArcAgiParser"
    expected_subdirs: tuple[str, ...] = ("data",)

    # Dataset-specific grid constraints
    max_grid_height: int = 30
    max_grid_width: int = 30
    min_grid_height: int = 3
    min_grid_width: int = 3

    # Color constraints
    max_colors: int = 10
    background_color: int = -1

    # Task Configuration
    max_train_pairs: int = 10
    max_test_pairs: int = 3

    # Task sampling parameters
    task_split: str = "train"
    shuffle_tasks: bool = True

    def validate(self) -> tuple[str, ...]:
        """Validate dataset configuration and return tuple of errors."""
        errors: list[str] = []

        try:
            # Validate dataset name
            if not self.dataset_name.strip():
                errors.append("dataset_name cannot be empty")

            # Validate dataset path
            validate_path_string(self.dataset_path, "dataset_path")

            # Validate grid dimensions
            validate_positive_int(self.max_grid_height, "max_grid_height")
            validate_positive_int(self.max_grid_width, "max_grid_width")
            validate_positive_int(self.min_grid_height, "min_grid_height")
            validate_positive_int(self.min_grid_width, "min_grid_width")

            # Validate task pair counts
            validate_positive_int(self.max_train_pairs, "max_train_pairs")
            validate_positive_int(self.max_test_pairs, "max_test_pairs")

            if self.max_train_pairs > 20:
                logger.warning(f"max_train_pairs is very large: {self.max_train_pairs}")
            if self.max_test_pairs > 5:
                logger.warning(f"max_test_pairs is very large: {self.max_test_pairs}")

            # Validate reasonable bounds
            if self.max_grid_height > 200:
                logger.warning(f"max_grid_height is very large: {self.max_grid_height}")
            if self.max_grid_width > 200:
                logger.warning(f"max_grid_width is very large: {self.max_grid_width}")

            # Validate color constraints
            validate_positive_int(self.max_colors, "max_colors")

            # Validate background_color: -1 is valid for padding, 0-9 are valid ARC colors
            if not isinstance(self.background_color, int):
                errors.append(
                    f"background_color must be an integer, got {type(self.background_color).__name__}"
                )
            elif self.background_color < -1:
                errors.append(
                    f"background_color must be >= -1 (for padding) or a valid color index, got {self.background_color}"
                )

            if self.max_colors < 2:
                errors.append("max_colors must be at least 2")
            if self.max_colors > 50:
                logger.warning(f"max_colors is very large: {self.max_colors}")

            # Validate task split
            valid_splits = [
                "train",
                "eval",
                "test",
                "all",
                "training",
                "evaluation",
                "corpus",
            ]
            validate_string_choice(self.task_split, "task_split", tuple(valid_splits))

            # Cross-field validation
            if self.max_grid_height < self.min_grid_height:
                errors.append(
                    f"max_grid_height ({self.max_grid_height}) < min_grid_height ({self.min_grid_height})"
                )

            if self.max_grid_width < self.min_grid_width:
                errors.append(
                    f"max_grid_width ({self.max_grid_width}) < min_grid_width ({self.min_grid_width})"
                )

            # Validate background_color against max_colors (but allow -1 for padding)
            if self.background_color >= 0 and self.background_color >= self.max_colors:
                errors.append(
                    f"background_color ({self.background_color}) must be < max_colors ({self.max_colors}) when >= 0"
                )

        except ConfigValidationError as e:
            errors.append(str(e))

        return tuple(errors)

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            msg = f"DatasetConfig must be hashable for JAX compatibility: {e}"
            raise ValueError(msg) from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> DatasetConfig:
        """Create dataset config from Hydra DictConfig."""
        return cls(
            dataset_name=cfg.get("dataset_name", "arc-agi-1"),
            dataset_path=cfg.get("dataset_path", ""),
            dataset_repo=cfg.get("dataset_repo", ""),
            parser_entry_point=cfg.get(
                "parser_entry_point", "jaxarc.parsers:ArcAgiParser"
            ),
            expected_subdirs=tuple(cfg.get("expected_subdirs", ["data"])),
            max_grid_height=cfg.get("max_grid_height", 30),
            max_grid_width=cfg.get("max_grid_width", 30),
            min_grid_height=cfg.get("min_grid_height", 3),
            min_grid_width=cfg.get("min_grid_width", 3),
            max_colors=cfg.get("max_colors", 10),
            background_color=cfg.get("background_color", -1),
            task_split=cfg.get("task_split", "train"),
            max_train_pairs=cfg.get("max_train_pairs", 10),
            max_test_pairs=cfg.get("max_test_pairs", 3),
            shuffle_tasks=cfg.get("shuffle_tasks", True),
        )
