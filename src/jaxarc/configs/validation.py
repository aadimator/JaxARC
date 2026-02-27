from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


class ConfigValidationError(ValueError):
    """Custom exception for configuration validation errors."""


# ---------------------------------------------------------------------------
# Scalar validators
# ---------------------------------------------------------------------------


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


def validate_positive_float(value: float, field_name: str) -> None:
    """Validate that a value is a positive number."""
    if not isinstance(value, (int, float)):
        msg = f"{field_name} must be a number, got {type(value).__name__}"
        raise ConfigValidationError(msg)
    if value <= 0:
        msg = f"{field_name} must be positive, got {value}"
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

    invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]
    if any(char in value for char in invalid_chars):
        msg = f"{field_name} contains invalid path characters: {value}"
        raise ConfigValidationError(msg)

    if must_exist and value and not Path(value).exists():
        msg = f"{field_name} path does not exist: {value}"
        raise ConfigValidationError(msg)


# ---------------------------------------------------------------------------
# Compound validators
# ---------------------------------------------------------------------------


def validate_tuple_elements(
    value: tuple[Any, ...],
    field_name: str,
    *,
    allowed: set[Any] | None = None,
    element_type: type = int,
    int_range: tuple[int, int] | None = None,
    allow_empty: bool = True,
) -> list[str]:
    """Validate elements of a tuple field. Returns a list of error strings."""
    errors: list[str] = []
    if not isinstance(value, tuple):
        errors.append(f"{field_name} must be a tuple, got {type(value).__name__}")
        return errors
    if not allow_empty and not value:
        errors.append(f"{field_name} cannot be empty")
        return errors
    for i, elem in enumerate(value):
        if not isinstance(elem, element_type):
            errors.append(
                f"{field_name}[{i}] must be an integer"
                if element_type is int
                else f"{field_name}[{i}] must be {element_type.__name__}"
            )
        elif allowed is not None and elem not in allowed:
            errors.append(
                f"{field_name}[{i}]: invalid value {elem!r}. Valid: {allowed}"
            )
        elif int_range is not None:
            lo, hi = int_range
            if not lo <= elem < hi:
                errors.append(f"{field_name}[{i}] must be in range [{lo}, {hi})")
    if len(set(value)) != len(value):
        dupes = [x for x in set(value) if value.count(x) > 1]
        errors.append(f"{field_name} contains duplicates: {dupes}")
    return errors


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def ensure_tuple(
    value: Any, *, default: tuple[Any, ...], of_type: type = str
) -> tuple[Any, ...]:
    """Convert *value* to a tuple, handling lists, single values, and iterables.

    Used by config ``__init__`` methods that accept lists from Hydra/YAML and
    must store tuples for Equinox hashability.
    """
    if value is None:
        return default
    if isinstance(value, tuple):
        return value
    if isinstance(value, of_type):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return default


def check_hashable(obj: Any, class_name: str) -> None:
    """Shared ``__check_init__`` logic for all config classes."""
    try:
        hash(obj)
    except TypeError as e:
        msg = f"{class_name} must be hashable for JAX compatibility: {e}"
        raise ValueError(msg) from e
