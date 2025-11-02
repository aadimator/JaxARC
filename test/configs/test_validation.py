"""
Tests for configuration validation utilities.

This module tests the validation functions used across all configuration
modules to ensure proper error handling and validation logic.
"""

from __future__ import annotations

import pytest

from jaxarc.configs.validation import (
    ConfigValidationError,
    validate_float_range,
    validate_non_negative_int,
    validate_path_string,
    validate_positive_int,
    validate_string_choice,
)


class TestConfigValidationError:
    """Test ConfigValidationError exception."""

    def test_config_validation_error_inheritance(self):
        """Test that ConfigValidationError inherits from ValueError."""
        error = ConfigValidationError("test error")
        assert isinstance(error, ValueError)
        assert str(error) == "test error"

    def test_config_validation_error_with_message(self):
        """Test ConfigValidationError with custom message."""
        message = "Custom validation error message"
        error = ConfigValidationError(message)
        assert str(error) == message


class TestValidatePositiveInt:
    """Test validate_positive_int function."""

    def test_valid_positive_integers(self):
        """Test that valid positive integers pass validation."""
        valid_values = [1, 5, 10, 100, 1000]

        for value in valid_values:
            # Should not raise an exception
            validate_positive_int(value, "test_field")

    def test_invalid_zero_value(self):
        """Test that zero value raises ConfigValidationError."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_positive_int(0, "test_field")

        assert "test_field must be positive" in str(exc_info.value)
        assert "got 0" in str(exc_info.value)

    def test_invalid_negative_value(self):
        """Test that negative values raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_positive_int(-5, "test_field")

        assert "test_field must be positive" in str(exc_info.value)
        assert "got -5" in str(exc_info.value)

    def test_invalid_non_integer_type(self):
        """Test that non-integer types raise ConfigValidationError."""
        invalid_values = [1.5, "5", [1], {"value": 1}, None]

        for value in invalid_values:
            with pytest.raises(ConfigValidationError) as exc_info:
                validate_positive_int(value, "test_field")

            assert "test_field must be an integer" in str(exc_info.value)
            assert type(value).__name__ in str(exc_info.value)


class TestValidateNonNegativeInt:
    """Test validate_non_negative_int function."""

    def test_valid_non_negative_integers(self):
        """Test that valid non-negative integers pass validation."""
        valid_values = [0, 1, 5, 10, 100, 1000]

        for value in valid_values:
            # Should not raise an exception
            validate_non_negative_int(value, "test_field")

    def test_invalid_negative_value(self):
        """Test that negative values raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_non_negative_int(-1, "test_field")

        assert "test_field must be non-negative" in str(exc_info.value)
        assert "got -1" in str(exc_info.value)

    def test_invalid_non_integer_type(self):
        """Test that non-integer types raise ConfigValidationError."""
        invalid_values = [1.5, "0", [0], {"value": 0}, None]

        for value in invalid_values:
            with pytest.raises(ConfigValidationError) as exc_info:
                validate_non_negative_int(value, "test_field")

            assert "test_field must be an integer" in str(exc_info.value)
            assert type(value).__name__ in str(exc_info.value)


class TestValidateFloatRange:
    """Test validate_float_range function."""

    def test_valid_float_values_in_range(self):
        """Test that valid float values in range pass validation."""
        # Test various ranges
        validate_float_range(0.5, "test_field", 0.0, 1.0)
        validate_float_range(5.0, "test_field", -10.0, 10.0)
        validate_float_range(-2.5, "test_field", -5.0, 0.0)
        validate_float_range(100, "test_field", 0, 1000)  # int should work too

    def test_valid_boundary_values(self):
        """Test that boundary values pass validation."""
        # Test minimum boundary
        validate_float_range(0.0, "test_field", 0.0, 1.0)

        # Test maximum boundary
        validate_float_range(1.0, "test_field", 0.0, 1.0)

    def test_invalid_below_minimum(self):
        """Test that values below minimum raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_float_range(-0.1, "test_field", 0.0, 1.0)

        assert "test_field must be in range [0.0, 1.0]" in str(exc_info.value)
        assert "got -0.1" in str(exc_info.value)

    def test_invalid_above_maximum(self):
        """Test that values above maximum raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_float_range(1.1, "test_field", 0.0, 1.0)

        assert "test_field must be in range [0.0, 1.0]" in str(exc_info.value)
        assert "got 1.1" in str(exc_info.value)

    def test_invalid_non_numeric_type(self):
        """Test that non-numeric types raise ConfigValidationError."""
        invalid_values = ["0.5", [0.5], {"value": 0.5}, None]

        for value in invalid_values:
            with pytest.raises(ConfigValidationError) as exc_info:
                validate_float_range(value, "test_field", 0.0, 1.0)

            assert "test_field must be a number" in str(exc_info.value)
            assert type(value).__name__ in str(exc_info.value)


class TestValidateStringChoice:
    """Test validate_string_choice function."""

    def test_valid_string_choices(self):
        """Test that valid string choices pass validation."""
        choices = ("option1", "option2", "option3")

        for choice in choices:
            validate_string_choice(choice, "test_field", choices)

    def test_invalid_string_choice(self):
        """Test that invalid string choices raise ConfigValidationError."""
        choices = ("option1", "option2", "option3")

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_string_choice("invalid_option", "test_field", choices)

        assert "test_field must be one of" in str(exc_info.value)
        assert "got 'invalid_option'" in str(exc_info.value)
        assert str(choices) in str(exc_info.value)

    def test_invalid_non_string_type(self):
        """Test that non-string types raise ConfigValidationError."""
        choices = ("option1", "option2", "option3")
        invalid_values = [1, 1.5, ["option1"], {"value": "option1"}, None]

        for value in invalid_values:
            with pytest.raises(ConfigValidationError) as exc_info:
                validate_string_choice(value, "test_field", choices)

            assert "test_field must be a string" in str(exc_info.value)
            assert type(value).__name__ in str(exc_info.value)


class TestValidatePathString:
    """Test validate_path_string function."""

    def test_valid_path_strings(self):
        """Test that valid path strings pass validation."""
        valid_paths = [
            "",  # Empty path is valid
            "relative/path",
            "/absolute/path",
            "path/with/file.txt",
            "path_with_underscores",
            "path-with-dashes",
            "path.with.dots",
        ]

        for path in valid_paths:
            validate_path_string(path, "test_field")

    def test_invalid_path_characters(self):
        """Test that paths with invalid characters raise ConfigValidationError."""
        invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]

        for char in invalid_chars:
            invalid_path = f"invalid{char}path"
            with pytest.raises(ConfigValidationError) as exc_info:
                validate_path_string(invalid_path, "test_field")

            assert "test_field contains invalid path characters" in str(exc_info.value)
            assert invalid_path in str(exc_info.value)

    def test_invalid_non_string_type(self):
        """Test that non-string types raise ConfigValidationError."""
        invalid_values = [1, 1.5, ["path"], {"path": "value"}, None]

        for value in invalid_values:
            with pytest.raises(ConfigValidationError) as exc_info:
                validate_path_string(value, "test_field")

            assert "test_field must be a string" in str(exc_info.value)
            assert type(value).__name__ in str(exc_info.value)

    def test_must_exist_validation(self, tmp_path):
        """Test path existence validation when must_exist=True."""
        # Create a test file
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("test content")

        # Valid: existing file
        validate_path_string(str(test_file), "test_field", must_exist=True)

        # Valid: existing directory
        validate_path_string(str(tmp_path), "test_field", must_exist=True)

        # Invalid: non-existing path
        non_existing_path = tmp_path / "non_existing_file.txt"
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_path_string(str(non_existing_path), "test_field", must_exist=True)

        assert "test_field path does not exist" in str(exc_info.value)
        assert str(non_existing_path) in str(exc_info.value)

    def test_must_exist_with_empty_path(self):
        """Test that empty path with must_exist=True passes validation."""
        # Empty path should pass even with must_exist=True
        validate_path_string("", "test_field", must_exist=True)


class TestValidationIntegration:
    """Test validation functions working together."""

    def test_multiple_validation_calls(self):
        """Test that multiple validation calls work correctly."""
        # This should not raise any exceptions
        validate_positive_int(10, "int_field")
        validate_float_range(0.5, "float_field", 0.0, 1.0)
        validate_string_choice("option1", "string_field", ("option1", "option2"))
        validate_path_string("valid/path", "path_field")

    def test_validation_error_messages_are_descriptive(self):
        """Test that validation error messages contain useful information."""
        # Test that field names are included in error messages
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_positive_int(-1, "my_custom_field_name")

        assert "my_custom_field_name" in str(exc_info.value)

        # Test that values are included in error messages
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_float_range(5.5, "test_field", 0.0, 1.0)

        assert "5.5" in str(exc_info.value)
        assert "[0.0, 1.0]" in str(exc_info.value)

    def test_validation_functions_are_pure(self):
        """Test that validation functions don't modify input values."""
        original_value = 42
        validate_positive_int(original_value, "test_field")
        assert original_value == 42  # Should be unchanged

        original_path = "test/path"
        validate_path_string(original_path, "test_field")
        assert original_path == "test/path"  # Should be unchanged
