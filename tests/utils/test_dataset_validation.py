"""Tests for dataset validation utilities."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from jaxarc.utils.dataset_validation import (
    get_dataset_recommendations,
    validate_dataset_config,
)

# No longer need to patch validate_config since we use config.validate() method


class TestValidateDatasetConfig:
    """Test validate_dataset_config function."""

    def test_validate_conceptarc_config_success(self):
        """Test successful ConceptARC configuration validation."""
        # Create a mock config that should pass ConceptARC validation
        mock_config = Mock()
        mock_config.dataset.max_grid_height = 30
        mock_config.dataset.max_grid_width = 30
        mock_config.action.selection_format = "mask"
        mock_config.dataset.dataset_name = "ConceptARC"
        mock_config.validate.return_value = []  # No validation errors

        # Should not raise any exceptions
        validate_dataset_config(mock_config, "ConceptARC")

        # Verify general validation was called
        mock_config.validate.assert_called_once()

    def test_validate_miniarc_config_success(self):
        """Test successful MiniARC configuration validation."""
        # Create a mock config that should pass MiniARC validation
        mock_config = Mock()
        mock_config.dataset.max_grid_height = 5
        mock_config.dataset.max_grid_width = 5
        mock_config.action.selection_format = "point"
        mock_config.dataset.dataset_name = "MiniARC"
        mock_config.validate.return_value = []  # No validation errors

        # Should not raise any exceptions
        validate_dataset_config(mock_config, "MiniARC")

        # Verify general validation was called
        mock_config.validate.assert_called_once()

    def test_validate_unknown_dataset(self):
        """Test validation with unknown dataset name."""
        mock_config = Mock()
        mock_config.validate.return_value = []  # No validation errors

        with patch("jaxarc.utils.dataset_validation.logger") as mock_logger:
            # Should not raise exception for unknown dataset (just logs warning)
            validate_dataset_config(mock_config, "UnknownDataset")

            # Verify general validation was called
            mock_config.validate.assert_called_once()
            mock_logger.warning.assert_called_once()
            assert (
                "No specific validation available"
                in mock_logger.warning.call_args[0][0]
            )

    def test_validate_config_general_validation_failure(self):
        """Test when general config validation fails."""
        mock_config = Mock()
        mock_config.validate.return_value = ["General validation failed"]

        with pytest.raises(
            ValueError, match="Invalid configuration for ConceptARC"
        ):
            validate_dataset_config(mock_config, "ConceptARC")

    def test_validate_config_import_error(self):
        """Test handling of import errors."""
        mock_config = Mock()
        mock_config.validate.side_effect = ImportError("Module not found")

        with pytest.raises(
            ValueError, match="Invalid configuration for ConceptARC"
        ):
            validate_dataset_config(mock_config, "ConceptARC")

    def test_validate_config_case_insensitive(self):
        """Test that dataset name matching is case insensitive."""
        mock_config = Mock()
        mock_config.dataset.max_grid_height = 30
        mock_config.dataset.max_grid_width = 30
        mock_config.action.selection_format = "mask"
        mock_config.dataset.dataset_name = "ConceptARC"
        mock_config.validate.return_value = []  # No validation errors

        # Test various case combinations
        for dataset_name in ["conceptarc", "CONCEPTARC", "ConceptArc"]:
            validate_dataset_config(mock_config, dataset_name)

            # Should have been called for each test
            assert mock_config.validate.call_count > 0


class TestGetDatasetRecommendations:
    """Test get_dataset_recommendations function."""

    def test_conceptarc_recommendations(self):
        """Test recommendations for ConceptARC dataset."""
        recommendations = get_dataset_recommendations("ConceptARC")

        expected = {
            "dataset.max_grid_height": "30",
            "dataset.max_grid_width": "30",
            "action.selection_format": "mask",
            "dataset.dataset_name": "ConceptARC",
        }

        assert recommendations == expected

    def test_miniarc_recommendations(self):
        """Test recommendations for MiniARC dataset."""
        recommendations = get_dataset_recommendations("MiniARC")

        expected = {
            "dataset.max_grid_height": "5",
            "dataset.max_grid_width": "5",
            "action.selection_format": "point",
            "dataset.dataset_name": "MiniARC",
        }

        assert recommendations == expected

    def test_case_insensitive_recommendations(self):
        """Test that recommendations work with different case."""
        # Test lowercase
        recommendations_lower = get_dataset_recommendations("conceptarc")
        recommendations_upper = get_dataset_recommendations("CONCEPTARC")
        recommendations_mixed = get_dataset_recommendations("ConceptArc")

        expected = {
            "dataset.max_grid_height": "30",
            "dataset.max_grid_width": "30",
            "action.selection_format": "mask",
            "dataset.dataset_name": "ConceptARC",
        }

        assert recommendations_lower == expected
        assert recommendations_upper == expected
        assert recommendations_mixed == expected

    def test_unknown_dataset_recommendations(self):
        """Test recommendations for unknown dataset."""
        with patch("jaxarc.utils.dataset_validation.logger") as mock_logger:
            recommendations = get_dataset_recommendations("UnknownDataset")

            assert recommendations == {}
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert (
                "No recommendations available for dataset: UnknownDataset"
                in warning_msg
            )

    def test_empty_dataset_name_recommendations(self):
        """Test recommendations for empty dataset name."""
        with patch("jaxarc.utils.dataset_validation.logger") as mock_logger:
            recommendations = get_dataset_recommendations("")

            assert recommendations == {}
            mock_logger.warning.assert_called_once()

    def test_recommendations_return_strings(self):
        """Test that all recommendation values are strings."""
        for dataset in ["ConceptARC", "MiniARC"]:
            recommendations = get_dataset_recommendations(dataset)

            for key, value in recommendations.items():
                assert isinstance(key, str), f"Key {key} is not a string"
                assert isinstance(value, str), (
                    f"Value {value} for key {key} is not a string"
                )

    def test_recommendations_hydra_format(self):
        """Test that recommendations are in valid Hydra override format."""
        for dataset in ["ConceptARC", "MiniARC"]:
            recommendations = get_dataset_recommendations(dataset)

            for key, value in recommendations.items():
                # Keys should contain dots for nested config
                assert "." in key, f"Key {key} should be in nested format"

                # Values should not contain spaces (for simple values)
                assert " " not in value or value.startswith('"'), (
                    f"Value {value} should be quoted if it contains spaces"
                )
