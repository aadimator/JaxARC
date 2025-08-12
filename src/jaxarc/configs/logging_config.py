from __future__ import annotations

import equinox as eqx
from omegaconf import DictConfig

from .validation import (
    ConfigValidationError,
    validate_positive_int,
    validate_string_choice,
)


class LoggingConfig(eqx.Module):
    """All logging behavior and formats.

    This config contains everything related to logging: what to log,
    how to format it, where to write it, and performance settings.
    """

    # What to log (specific content flags)
    log_operations: bool = False
    log_rewards: bool = False

    # Logging frequency and timing
    log_frequency: int = 10  # Log every N steps

    # Batched logging settings
    batched_logging_enabled: bool = False

    # Format and level referenced by cross-validation logic
    log_format: str = "text"
    log_level: str = "INFO"

    # Structured logging toggle referenced by cross-validation logic
    structured_logging: bool = False

    def validate(self) -> tuple[str, ...]:
        """Validate logging configuration and return tuple of errors."""
        errors: list[str] = []

        try:
            # Validate format choices
            valid_formats = ("json", "text", "structured")
            validate_string_choice(self.log_format, "log_format", valid_formats)

            valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR")
            validate_string_choice(self.log_level, "log_level", valid_levels)

            # Validate numeric fields
            validate_positive_int(self.log_frequency, "log_frequency")

        except ConfigValidationError as e:
            errors.append(str(e))

        return tuple(errors)

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            msg = f"LoggingConfig must be hashable for JAX compatibility: {e}"
            raise ValueError(msg) from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> LoggingConfig:
        """Create logging config from Hydra DictConfig."""
        return cls(
            log_operations=cfg.get("log_operations", False),
            log_rewards=cfg.get("log_rewards", False),
            log_frequency=cfg.get("log_frequency", 10),
            batched_logging_enabled=cfg.get("batched_logging_enabled", False),
            log_format=cfg.get("log_format", "text"),
            log_level=cfg.get("log_level", "INFO"),
            structured_logging=cfg.get("structured_logging", False),
        )
