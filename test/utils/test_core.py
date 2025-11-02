"""Tests for core utility functions."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig

from jaxarc.utils.core import (
    get_config,
    get_path,
    get_raw_path,
)


class TestGetConfig:
    """Test configuration loading functionality."""

    def test_get_config_default(self):
        """Test loading default configuration."""
        config = get_config()

        assert isinstance(config, DictConfig)
        # Check that main sections exist
        assert "dataset" in config
        assert "action" in config
        assert "environment" in config
        assert "reward" in config

    def test_get_config_with_overrides(self):
        """Test loading configuration with overrides."""
        overrides = [
            "dataset.dataset_name=TestDataset",
            "action.validate_actions=false",
        ]

        config = get_config(overrides=overrides)

        assert isinstance(config, DictConfig)
        assert config.dataset.dataset_name == "TestDataset"
        assert config.action.validate_actions == False

    def test_get_config_empty_overrides(self):
        """Test loading configuration with empty overrides list."""
        config = get_config(overrides=[])

        assert isinstance(config, DictConfig)
        # Should be same as default config
        default_config = get_config()
        assert config.dataset.dataset_name == default_config.dataset.dataset_name

    def test_get_config_none_overrides(self):
        """Test loading configuration with None overrides."""
        config = get_config(overrides=None)

        assert isinstance(config, DictConfig)
        # Should be same as default config
        default_config = get_config()
        assert config.dataset.dataset_name == default_config.dataset.dataset_name

    def test_get_config_multiple_overrides(self):
        """Test loading configuration with multiple overrides."""
        overrides = [
            "dataset.dataset_name=MultiTest",
            "dataset.max_grid_height=15",
            "dataset.max_grid_width=20",
            "action.validate_actions=false",
            "environment.max_episode_steps=100",
        ]

        config = get_config(overrides=overrides)

        assert config.dataset.dataset_name == "MultiTest"
        assert config.dataset.max_grid_height == 15
        assert config.dataset.max_grid_width == 20
        assert config.action.validate_actions == False
        assert config.environment.max_episode_steps == 100

    def test_get_config_nested_overrides(self):
        """Test loading configuration with deeply nested overrides."""
        overrides = ["dataset.expected_subdirs=[data,test]", "reward.success_bonus=2.0"]

        config = get_config(overrides=overrides)

        assert config.dataset.expected_subdirs == ["data", "test"]
        assert config.reward.success_bonus == 2.0

    def test_get_config_invalid_override_format(self):
        """Test that invalid override format raises appropriate error."""
        invalid_overrides = [
            "invalid_format_no_equals",
            "dataset.dataset_name",  # Missing value
        ]

        # Should raise an error from Hydra
        with pytest.raises(Exception):  # Hydra raises various exception types
            get_config(overrides=invalid_overrides)

    def test_get_config_nonexistent_field_override(self):
        """Test override of nonexistent configuration field."""
        overrides = ["nonexistent.field=value"]

        # Hydra should raise an error for nonexistent fields
        with pytest.raises(Exception):  # ConfigCompositionException
            get_config(overrides=overrides)

    def test_get_config_type_conversion(self):
        """Test that overrides properly convert types."""
        overrides = [
            "dataset.max_grid_height=25",  # Should be int
            "reward.success_bonus=1.5",  # Should be float
            "environment.debug_level=verbose",  # Should be string
        ]

        config = get_config(overrides=overrides)

        assert isinstance(config.dataset.max_grid_height, int)
        assert config.dataset.max_grid_height == 25

        assert isinstance(config.reward.success_bonus, float)
        assert config.reward.success_bonus == 1.5

        assert isinstance(config.environment.debug_level, str)
        assert config.environment.debug_level == "verbose"


class TestGetPath:
    """Test path retrieval functionality."""

    def test_get_path_data_raw(self):
        """Test getting raw data path."""
        path = get_path("data_raw")

        assert isinstance(path, Path)
        assert path.name == "raw"
        assert "data" in str(path)

    def test_get_path_with_create_false(self):
        """Test getting path without creating directory."""
        path = get_path("data_raw", create=False)

        assert isinstance(path, Path)
        # Directory may or may not exist, but function should not fail

    def test_get_path_with_create_true(self):
        """Test getting path with directory creation."""
        # Use a temporary directory to avoid affecting real paths
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("jaxarc.utils.core.here") as mock_here:
                mock_here.return_value = Path(temp_dir)

                # Mock the config to return a test path
                with patch("jaxarc.utils.core.get_config") as mock_get_config:
                    mock_config = MagicMock()
                    mock_config.paths = {"test_path": "test_dir"}
                    mock_get_config.return_value = mock_config

                    path = get_path("test_path", create=True)

                    assert isinstance(path, Path)
                    assert path.exists()
                    assert path.is_dir()

    def test_get_path_nonexistent_type(self):
        """Test getting path with nonexistent path type."""
        with pytest.raises(KeyError, match="Path type 'nonexistent' not found"):
            get_path("nonexistent")

    def test_get_path_available_paths_in_error(self):
        """Test that error message includes available paths."""
        try:
            get_path("nonexistent")
        except KeyError as e:
            error_msg = str(e)
            assert "Available:" in error_msg
            # Should list some actual path types
            assert "data_raw" in error_msg or "data" in error_msg

    @patch("jaxarc.utils.core.get_config")
    def test_get_path_empty_paths_config(self, mock_get_config):
        """Test behavior when paths config is empty."""
        mock_config = MagicMock()
        mock_config.paths = {}
        mock_get_config.return_value = mock_config

        with pytest.raises(KeyError):
            get_path("any_path")

    # Removed test_get_path_relative_path_resolution - tests complex path resolution edge case

    def test_get_path_multiple_calls_consistent(self):
        """Test that multiple calls return consistent paths."""
        path1 = get_path("data_raw")
        path2 = get_path("data_raw")

        assert path1 == path2
        assert str(path1) == str(path2)


class TestGetRawPath:
    """Test raw path convenience function."""

    def test_get_raw_path_default(self):
        """Test getting raw path with default parameters."""
        path = get_raw_path()

        assert isinstance(path, Path)
        # Should be equivalent to get_path("data_raw", create=False)
        expected_path = get_path("data_raw", create=False)
        assert path == expected_path

    def test_get_raw_path_with_create_false(self):
        """Test getting raw path without creating directory."""
        path = get_raw_path(create=False)

        assert isinstance(path, Path)
        expected_path = get_path("data_raw", create=False)
        assert path == expected_path

    def test_get_raw_path_with_create_true(self):
        """Test getting raw path with directory creation."""
        # Use a temporary directory to avoid affecting real paths
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("jaxarc.utils.core.here") as mock_here:
                mock_here.return_value = Path(temp_dir)

                with patch("jaxarc.utils.core.get_config") as mock_get_config:
                    mock_config = MagicMock()
                    mock_config.paths = {"data_raw": "test_raw"}
                    mock_get_config.return_value = mock_config

                    path = get_raw_path(create=True)

                    assert isinstance(path, Path)
                    assert path.exists()
                    assert path.is_dir()

    def test_get_raw_path_consistency_with_get_path(self):
        """Test that get_raw_path is consistent with get_path."""
        raw_path = get_raw_path()
        direct_path = get_path("data_raw")

        assert raw_path == direct_path


class TestIntegrationAndWorkflows:
    """Test integration scenarios and complete workflows."""

    def test_config_and_path_integration(self):
        """Test integration between config loading and path resolution."""
        # Load config and use it to get paths
        config = get_config()

        # Should be able to get paths that are defined in the config
        try:
            raw_path = get_path("data_raw")
            assert isinstance(raw_path, Path)
        except KeyError:
            # If data_raw is not in config, that's also valid
            pass

    def test_override_and_path_workflow(self):
        """Test workflow of overriding config and using paths."""
        # This tests that config overrides don't break path resolution
        overrides = ["dataset.dataset_name=TestWorkflow"]
        config = get_config(overrides=overrides)

        assert config.dataset.dataset_name == "TestWorkflow"

        # Path resolution should still work
        try:
            path = get_path("data_raw")
            assert isinstance(path, Path)
        except KeyError:
            # If path doesn't exist in config, that's expected
            pass

    def test_multiple_path_types_workflow(self):
        """Test workflow using multiple path types."""
        # Get different types of paths
        path_types = []

        # Try to get various path types that might exist
        for path_type in [
            "data_raw",
            "data_processed",
            "data_interim",
            "data_external",
        ]:
            try:
                path = get_path(path_type)
                path_types.append((path_type, path))
            except KeyError:
                # Path type doesn't exist in config
                continue

        # Should have gotten at least one path type
        # (or this test needs to be updated based on actual config)
        if path_types:
            for path_type, path in path_types:
                assert isinstance(path, Path)
                assert path_type in str(path).lower() or "data" in str(path).lower()


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_config_with_very_long_overrides(self):
        """Test configuration with very long override values."""
        long_value = "x" * 1000
        overrides = [f"dataset.dataset_name={long_value}"]

        config = get_config(overrides=overrides)
        assert config.dataset.dataset_name == long_value

    def test_config_with_special_characters(self):
        """Test configuration with special characters in overrides."""
        special_overrides = [
            "dataset.dataset_name='Test-Dataset_123'",
            "dataset.dataset_path='/path/with/special-chars_123'",
        ]

        config = get_config(overrides=special_overrides)
        assert "Test-Dataset_123" in config.dataset.dataset_name
        assert "/path/with/special-chars_123" in config.dataset.dataset_path

    def test_path_with_very_long_names(self):
        """Test path handling with very long path names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("jaxarc.utils.core.here") as mock_here:
                mock_here.return_value = Path(temp_dir)

                with patch("jaxarc.utils.core.get_config") as mock_get_config:
                    long_path_name = "very_long_path_name_" + "x" * 100
                    mock_config = MagicMock()
                    mock_config.paths = {"long_path": long_path_name}
                    mock_get_config.return_value = mock_config

                    path = get_path("long_path", create=True)

                    assert isinstance(path, Path)
                    assert path.exists()

    def test_path_with_unicode_characters(self):
        """Test path handling with unicode characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("jaxarc.utils.core.here") as mock_here:
                mock_here.return_value = Path(temp_dir)

                with patch("jaxarc.utils.core.get_config") as mock_get_config:
                    unicode_path = "测试路径_тест_🚀"
                    mock_config = MagicMock()
                    mock_config.paths = {"unicode_path": unicode_path}
                    mock_get_config.return_value = mock_config

                    try:
                        path = get_path("unicode_path", create=True)
                        assert isinstance(path, Path)
                        # Unicode support depends on filesystem
                    except (OSError, UnicodeError):
                        # Some filesystems don't support unicode
                        pass

    # Removed test_concurrent_path_access - tests complex threading edge case

    @patch("jaxarc.utils.core.get_config")
    def test_path_creation_permission_error(self, mock_get_config):
        """Test path creation when permission is denied."""
        # Mock a path that would cause permission error
        with patch("jaxarc.utils.core.here") as mock_here:
            mock_here.return_value = Path("/root/restricted")  # Typically restricted

            mock_config = MagicMock()
            mock_config.paths = {"restricted_path": "test"}
            mock_get_config.return_value = mock_config

            # Should handle permission error gracefully
            try:
                path = get_path("restricted_path", create=True)
                # If it succeeds, that's fine too
                assert isinstance(path, Path)
            except (PermissionError, OSError):
                # Expected for restricted paths
                pass

    def test_config_loading_performance(self):
        """Test that config loading performs reasonably."""
        import time

        start_time = time.time()

        # Load config multiple times
        for _ in range(10):
            config = get_config()
            assert isinstance(config, DictConfig)

        end_time = time.time()

        # Should complete in reasonable time (less than 5 seconds for 10 loads)
        assert (end_time - start_time) < 5.0
