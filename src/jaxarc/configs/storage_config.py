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


class StorageConfig(eqx.Module):
    """All storage, output, and file management settings.

    This config contains everything related to file storage, output directories,
    cleanup policies, and file organization. All output paths are managed here.
    """

    # Base storage configuration
    base_output_dir: str = "outputs"
    run_name: str | None = None

    # Output directories for different types of content
    episodes_dir: str = "episodes"
    debug_dir: str = "debug"
    visualization_dir: str = "visualizations"
    logs_dir: str = "logs"

    # Storage limits
    max_episodes_per_run: int = 100
    max_storage_gb: float = 5.0

    # Cleanup settings
    cleanup_policy: str = "size_based"

    # File organization
    create_run_subdirs: bool = True
    clear_output_on_start: bool = True

    def validate(self) -> tuple[str, ...]:
        """Validate storage configuration and return tuple of errors."""
        errors: list[str] = []

        try:
            valid_cleanup_policies = ("none", "size_based", "oldest_first", "manual")
            validate_string_choice(
                self.cleanup_policy, "cleanup_policy", valid_cleanup_policies
            )

            # Validate output directory paths
            validate_path_string(self.base_output_dir, "base_output_dir")
            validate_path_string(self.episodes_dir, "episodes_dir")
            validate_path_string(self.debug_dir, "debug_dir")
            validate_path_string(self.visualization_dir, "visualization_dir")
            validate_path_string(self.logs_dir, "logs_dir")

            # Validate numeric fields
            validate_positive_int(self.max_episodes_per_run, "max_episodes_per_run")

            if not isinstance(self.max_storage_gb, (int, float)):
                msg = f"max_storage_gb must be a number, got {type(self.max_storage_gb).__name__}"
                errors.append(msg)
            elif self.max_storage_gb <= 0:
                errors.append("max_storage_gb must be positive")

            # Validate reasonable bounds
            if self.max_episodes_per_run > 10000:
                logger.warning(
                    f"max_episodes_per_run is very large: {self.max_episodes_per_run}"
                )

            if self.max_storage_gb > 100:
                logger.warning(f"max_storage_gb is very large: {self.max_storage_gb}")

        except ConfigValidationError as e:
            errors.append(str(e))

        return tuple(errors)

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            msg = f"StorageConfig must be hashable for JAX compatibility: {e}"
            raise ValueError(msg) from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> StorageConfig:
        """Create storage config from Hydra DictConfig."""
        return cls(
            base_output_dir=cfg.get("base_output_dir", "outputs"),
            run_name=cfg.get("run_name"),
            episodes_dir=cfg.get("episodes_dir", "episodes"),
            debug_dir=cfg.get("debug_dir", "debug"),
            visualization_dir=cfg.get("visualization_dir", "visualizations"),
            logs_dir=cfg.get("logs_dir", "logs"),
            max_episodes_per_run=cfg.get("max_episodes_per_run", 100),
            max_storage_gb=cfg.get("max_storage_gb", 5.0),
            cleanup_policy=cfg.get("cleanup_policy", "size_based"),
            create_run_subdirs=cfg.get("create_run_subdirs", True),
            clear_output_on_start=cfg.get("clear_output_on_start", True),
        )
