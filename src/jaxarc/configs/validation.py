from __future__ import annotations

from pathlib import Path


class ConfigValidationError(ValueError):
    """Custom exception for configuration validation errors."""


def validate_positive_int(value: int, field_name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int):
        msg = f"{field_name} must be an integer, got {type(value).__name__}"
        raise ConfigValidationError(msg)
    if value <= 0:
        msg = f"{field_name} must be positive, got {value}"
        raise ConfigValidationError(msg)


def validate_non_negative_int(value: int, field_name: str) -> None:
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, int):
        msg = f"{field_name} must be an integer, got {type(value).__name__}"
        raise ConfigValidationError(msg)
    if value < 0:
        msg = f"{field_name} must be non-negative, got {value}"
        raise ConfigValidationError(msg)


def validate_float_range(
    value: float, field_name: str, min_val: float, max_val: float
) -> None:
    """Validate that a float value is within a specified range."""
    if not isinstance(value, (int, float)):
        msg = f"{field_name} must be a number, got {type(value).__name__}"
        raise ConfigValidationError(msg)
    if not min_val <= value <= max_val:
        msg = f"{field_name} must be in range [{min_val}, {max_val}], got {value}"
        raise ConfigValidationError(msg)


def validate_string_choice(
    value: str, field_name: str, choices: tuple[str, ...]
) -> None:
    """Validate that a string value is one of the allowed choices."""
    if not isinstance(value, str):
        msg = f"{field_name} must be a string, got {type(value).__name__}"
        raise ConfigValidationError(msg)
    if value not in choices:
        msg = f"{field_name} must be one of {choices}, got '{value}'"
        raise ConfigValidationError(msg)


def validate_path_string(value: str, field_name: str, must_exist: bool = False) -> None:
    """Validate that a value is a valid path string."""
    if not isinstance(value, str):
        msg = f"{field_name} must be a string, got {type(value).__name__}"
        raise ConfigValidationError(msg)

    # Check for invalid path characters
    invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]
    if any(char in value for char in invalid_chars):
        msg = f"{field_name} contains invalid path characters: {value}"
        raise ConfigValidationError(msg)

    if must_exist and value and not Path(value).exists():
        msg = f"{field_name} path does not exist: {value}"
        raise ConfigValidationError(msg)
