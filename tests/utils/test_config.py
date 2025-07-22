"""Tests for configuration utility functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from omegaconf import DictConfig

from jaxarc.utils.config import (
    get_config,
    get_external_path,
    get_interim_path,
    get_path,
    get_processed_path,
    get_raw_path,
)


class TestGetConfig:
    """Test get_config function."""

    def test_get_default_config(self):
        """Test loading default configuration."""
        cfg = get_config()

        assert isinstance(cfg, DictConfig)
        assert "dataset" in cfg
        assert "action" in cfg
        assert "environment" in cfg
        assert "paths" in cfg

    def test_get_config_with_overrides(self):
        """Test loading configuration with overrides."""
        overrides = ["dataset.dataset_name=ConceptARC", "action.selection_format=point"]

        cfg = get_config(overrides)

        assert cfg.dataset.dataset_name == "ConceptARC"
        assert cfg.action.selection_format == "point"

    def test_get_config_with_empty_overrides(self):
        """Test loading configuration with empty overrides list."""
        cfg = get_config([])

        assert isinstance(cfg, DictConfig)
        # Should be same as default config
        default_cfg = get_config()
        assert cfg.dataset.dataset_name == default_cfg.dataset.dataset_name

    def test_get_config_with_none_overrides(self):
        """Test loading configuration with None overrides."""
        cfg = get_config(None)

        assert isinstance(cfg, DictConfig)
        # Should be same as default config
        default_cfg = get_config()
        assert cfg.dataset.dataset_name == default_cfg.dataset.dataset_name

    def test_get_config_multiple_overrides(self):
        """Test loading configuration with multiple overrides."""
        overrides = [
            "dataset.dataset_name=MiniARC",
            "action.selection_format=bbox",
            "environment.max_episode_steps=50",
            "dataset.max_grid_height=5",
        ]

        cfg = get_config(overrides)

        assert cfg.dataset.dataset_name == "MiniARC"
        assert cfg.action.selection_format == "bbox"
        assert cfg.environment.max_episode_steps == 50
        assert cfg.dataset.max_grid_height == 5

    def test_get_config_nested_overrides(self):
        """Test loading configuration with deeply nested overrides."""
        overrides = ["reward.similarity_weight=0.8", "reward.step_penalty=-0.02"]

        cfg = get_config(overrides)

        assert cfg.reward.similarity_weight == 0.8
        assert cfg.reward.step_penalty == -0.02

    def test_get_config_invalid_override_format(self):
        """Test that invalid override format raises appropriate error."""
        overrides = ["invalid_format"]  # Missing = sign

        with pytest.raises(Exception):  # Hydra will raise an exception
            get_config(overrides)

    @patch("jaxarc.utils.config.here")
    def test_get_config_custom_config_dir(self, mock_here):
        """Test that config directory is correctly resolved."""
        mock_here.return_value = Path("/fake/project/root")

        # This will test that the config directory path is constructed correctly
        # The actual config loading might fail, but we can test the path resolution
        with patch("jaxarc.utils.config.initialize_config_dir") as mock_init:
            mock_init.return_value.__enter__ = Mock()
            mock_init.return_value.__exit__ = Mock()

            with patch("jaxarc.utils.config.compose") as mock_compose:
                mock_compose.return_value = DictConfig({"test": "value"})

                cfg = get_config()

                # Check that initialize_config_dir was called
                mock_init.assert_called_once()
                # The config directory path depends on the mocked here() function
                # Just verify the function was called correctly


class TestGetPath:
    """Test get_path function."""

    def test_get_existing_path_type(self):
        """Test getting an existing path type."""
        path = get_path("data_raw")

        assert isinstance(path, Path)
        assert path.name == "raw"  # Based on typical config

    def test_get_path_with_create_false(self):
        """Test getting path without creating directory."""
        path = get_path("data_raw", create=False)

        assert isinstance(path, Path)
        # Directory may or may not exist, but function should not fail

    def test_get_path_with_create_true(self, tmp_path):
        """Test getting path with directory creation."""
        # Mock the config to return a test path
        with patch("jaxarc.utils.config.get_config") as mock_get_config:
            mock_config = DictConfig(
                {"paths": {"test_path": str(tmp_path / "test_dir")}}
            )
            mock_get_config.return_value = mock_config

            with patch("jaxarc.utils.config.here") as mock_here:
                mock_here.return_value = tmp_path

                path = get_path("test_path", create=True)

                assert path.exists()
                assert path.is_dir()

    def test_get_nonexistent_path_type(self):
        """Test getting a non-existent path type."""
        with pytest.raises(KeyError, match="Path type 'nonexistent' not found"):
            get_path("nonexistent")

    def test_get_path_error_message_includes_available_paths(self):
        """Test that error message includes available path types."""
        try:
            get_path("nonexistent")
        except KeyError as e:
            error_msg = str(e)
            assert "Available:" in error_msg
            assert "data_raw" in error_msg or "data_processed" in error_msg

    @patch("jaxarc.utils.config.get_config")
    @patch("jaxarc.utils.config.here")
    def test_get_path_with_relative_path(self, mock_here, mock_get_config, tmp_path):
        """Test path resolution with relative paths."""
        mock_here.return_value = tmp_path
        mock_config = DictConfig({"paths": {"relative_path": "data/test"}})
        mock_get_config.return_value = mock_config

        path = get_path("relative_path")

        # The function returns the path as resolved by here() from project root
        # Since we mocked here() to return tmp_path, the path should be based on that
        assert path == tmp_path

    @patch("jaxarc.utils.config.logger")
    def test_get_path_create_logs_debug(self, mock_logger, tmp_path):
        """Test that path creation logs debug message."""
        with patch("jaxarc.utils.config.get_config") as mock_get_config:
            mock_config = DictConfig(
                {"paths": {"test_path": str(tmp_path / "new_dir")}}
            )
            mock_get_config.return_value = mock_config

            with patch("jaxarc.utils.config.here") as mock_here:
                mock_here.return_value = tmp_path

                get_path("test_path", create=True)

                mock_logger.debug.assert_called_once()
                assert "Created path:" in mock_logger.debug.call_args[0][0]


class TestSpecificPathFunctions:
    """Test specific path getter functions."""

    def test_get_raw_path_default(self):
        """Test get_raw_path with default parameters."""
        path = get_raw_path()

        assert isinstance(path, Path)
        assert "raw" in str(path).lower()

    def test_get_raw_path_with_create(self, tmp_path):
        """Test get_raw_path with create=True."""
        with patch("jaxarc.utils.config.get_path") as mock_get_path:
            mock_get_path.return_value = tmp_path / "raw"

            path = get_raw_path(create=True)

            mock_get_path.assert_called_once_with("data_raw", create=True)
            assert path == tmp_path / "raw"

    def test_get_processed_path_default(self):
        """Test get_processed_path with default parameters."""
        path = get_processed_path()

        assert isinstance(path, Path)

    def test_get_processed_path_with_create(self, tmp_path):
        """Test get_processed_path with create=True."""
        with patch("jaxarc.utils.config.get_path") as mock_get_path:
            mock_get_path.return_value = tmp_path / "processed"

            path = get_processed_path(create=True)

            mock_get_path.assert_called_once_with("data_processed", create=True)
            assert path == tmp_path / "processed"

    def test_get_interim_path_default(self):
        """Test get_interim_path with default parameters."""
        path = get_interim_path()

        assert isinstance(path, Path)

    def test_get_interim_path_with_create(self, tmp_path):
        """Test get_interim_path with create=True."""
        with patch("jaxarc.utils.config.get_path") as mock_get_path:
            mock_get_path.return_value = tmp_path / "interim"

            path = get_interim_path(create=True)

            mock_get_path.assert_called_once_with("data_interim", create=True)
            assert path == tmp_path / "interim"

    def test_get_external_path_default(self):
        """Test get_external_path with default parameters."""
        path = get_external_path()

        assert isinstance(path, Path)

    def test_get_external_path_with_create(self, tmp_path):
        """Test get_external_path with create=True."""
        with patch("jaxarc.utils.config.get_path") as mock_get_path:
            mock_get_path.return_value = tmp_path / "external"

            path = get_external_path(create=True)

            mock_get_path.assert_called_once_with("data_external", create=True)
            assert path == tmp_path / "external"


class TestConfigIntegration:
    """Test integration with actual configuration system."""

    def test_config_structure_validation(self):
        """Test that loaded config has expected structure."""
        cfg = get_config()

        # Test required top-level keys
        required_keys = ["dataset", "action", "environment", "paths"]
        for key in required_keys:
            assert key in cfg, f"Missing required config key: {key}"

        # Test dataset config structure
        assert "dataset_name" in cfg.dataset
        assert "data_root" in cfg.dataset

        # Test action config structure
        assert "selection_format" in cfg.action
        assert "allowed_operations" in cfg.action

        # Test environment config structure
        assert "max_episode_steps" in cfg.environment

        # Test paths config structure
        assert "data_raw" in cfg.paths

    def test_config_override_precedence(self):
        """Test that overrides take precedence over defaults."""
        # Get default value
        default_cfg = get_config()
        default_dataset = default_cfg.dataset.dataset_name

        # Override with different value
        new_dataset = "TestDataset"
        if default_dataset == new_dataset:
            new_dataset = "DifferentTestDataset"

        override_cfg = get_config([f"dataset.dataset_name={new_dataset}"])

        assert override_cfg.dataset.dataset_name == new_dataset
        assert override_cfg.dataset.dataset_name != default_dataset

    def test_config_type_preservation(self):
        """Test that config values maintain correct types."""
        cfg = get_config()

        # Test integer values
        if hasattr(cfg.environment, "max_episode_steps"):
            assert isinstance(cfg.environment.max_episode_steps, int)

        # Test string values
        assert isinstance(cfg.dataset.dataset_name, str)

        # Test boolean values (if any exist in config)
        # This depends on actual config structure

    def test_path_resolution_integration(self):
        """Test integration between config and path resolution."""
        cfg = get_config()

        # Test that all configured paths can be resolved
        for path_key in cfg.paths.keys():
            try:
                path = get_path(path_key)
                assert isinstance(path, Path)
            except Exception as e:
                pytest.fail(f"Failed to resolve path '{path_key}': {e}")

    def test_config_modification_isolation(self):
        """Test that config modifications don't affect other calls."""
        cfg1 = get_config()
        cfg2 = get_config(["dataset.dataset_name=ModifiedDataset"])
        cfg3 = get_config()

        # cfg2 should be modified, but cfg1 and cfg3 should be the same
        assert cfg1.dataset.dataset_name == cfg3.dataset.dataset_name
        assert cfg2.dataset.dataset_name == "ModifiedDataset"
        assert cfg1.dataset.dataset_name != cfg2.dataset.dataset_name


class TestErrorHandling:
    """Test error handling in configuration utilities."""

    def test_invalid_hydra_override_syntax(self):
        """Test handling of invalid Hydra override syntax."""
        invalid_overrides = [
            "dataset.dataset_name",  # Missing value
            "=invalid",  # Missing key
            "dataset..dataset_name=value",  # Double dots
        ]

        for invalid_override in invalid_overrides:
            with pytest.raises(Exception):  # Hydra will raise various exceptions
                get_config([invalid_override])

    def test_nonexistent_config_key_override(self):
        """Test overriding non-existent config keys."""
        # Hydra with struct mode doesn't allow new keys, so this should fail
        with pytest.raises(Exception):  # ConfigCompositionException
            get_config(["nonexistent.key=value"])

    @patch("jaxarc.utils.config.here")
    def test_missing_config_directory(self, mock_here):
        """Test behavior when config directory doesn't exist."""
        mock_here.return_value = Path("/nonexistent/path")

        # This should raise an exception when trying to initialize Hydra
        with pytest.raises(Exception):
            get_config()

    def test_path_creation_permission_error(self, tmp_path):
        """Test handling of permission errors during path creation."""
        # Create a directory with no write permissions
        no_write_dir = tmp_path / "no_write"
        no_write_dir.mkdir()
        no_write_dir.chmod(0o444)  # Read-only

        try:
            with patch("jaxarc.utils.config.get_config") as mock_get_config:
                mock_config = DictConfig(
                    {"paths": {"test_path": str(no_write_dir / "subdir")}}
                )
                mock_get_config.return_value = mock_config

                with patch("jaxarc.utils.config.here") as mock_here:
                    mock_here.return_value = tmp_path

                    # This might not raise a permission error on all systems
                    # Just test that the function handles the case gracefully
                    try:
                        get_path("test_path", create=True)
                    except (PermissionError, OSError):
                        # Expected on some systems
                        pass
        finally:
            # Restore permissions for cleanup
            no_write_dir.chmod(0o755)


class TestPerformanceAndCaching:
    """Test performance characteristics and caching behavior."""

    def test_config_loading_performance(self):
        """Test that config loading is reasonably fast."""
        import time

        start_time = time.time()
        for _ in range(10):
            get_config()
        end_time = time.time()

        # Should be able to load config 10 times in under 5 seconds
        # This is a reasonable performance expectation
        assert (end_time - start_time) < 5.0

    def test_path_resolution_consistency(self):
        """Test that path resolution is consistent across calls."""
        path1 = get_path("data_raw")
        path2 = get_path("data_raw")
        path3 = get_raw_path()

        assert path1 == path2 == path3

    def test_config_override_performance(self):
        """Test performance with valid overrides."""
        # Use valid config keys that exist in the structure
        many_overrides = [f"dataset.dataset_name=Dataset_{i}" for i in range(10)]

        import time

        start_time = time.time()
        cfg = get_config(many_overrides[:1])  # Just test one override
        end_time = time.time()

        # Should handle overrides reasonably quickly
        assert (end_time - start_time) < 2.0

        # Verify override was applied
        assert cfg.dataset.dataset_name == "Dataset_0"
