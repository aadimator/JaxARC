"""Tests for the download script CLI functionality."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

# Add scripts directory to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from download_dataset import (
    DATASETS,
    DatasetConfig,
    _check_existing_dataset,
    _download_all_datasets,
    _download_single_dataset,
    _log_dataset_info,
    _setup_output_directory,
    app,
)
from jaxarc.utils.dataset_downloader import DatasetDownloadError


class TestDownloadScriptCLI:
    """Test suite for download script CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_arc_agi_1_command_success(self, tmp_path):
        """Test successful ARC-AGI-1 download command."""
        with patch("download_dataset._download_single_dataset") as mock_download:
            result = self.runner.invoke(
                app, ["arc-agi-1", "--output", str(tmp_path), "--force"]
            )
            
            assert result.exit_code == 0
            mock_download.assert_called_once_with("arc-agi-1", tmp_path, True)

    def test_arc_agi_2_command_success(self, tmp_path):
        """Test successful ARC-AGI-2 download command."""
        with patch("download_dataset._download_single_dataset") as mock_download:
            result = self.runner.invoke(
                app, ["arc-agi-2", "--output", str(tmp_path), "--force"]
            )
            
            assert result.exit_code == 0
            mock_download.assert_called_once_with("arc-agi-2", tmp_path, True)

    def test_conceptarc_command_success(self, tmp_path):
        """Test successful ConceptARC download command."""
        with patch("download_dataset._download_single_dataset") as mock_download:
            result = self.runner.invoke(
                app, ["conceptarc", "--output", str(tmp_path)]
            )
            
            assert result.exit_code == 0
            mock_download.assert_called_once_with("conceptarc", tmp_path, False)

    def test_miniarc_command_success(self, tmp_path):
        """Test successful MiniARC download command."""
        with patch("download_dataset._download_single_dataset") as mock_download:
            result = self.runner.invoke(
                app, ["miniarc", "--output", str(tmp_path)]
            )
            
            assert result.exit_code == 0
            mock_download.assert_called_once_with("miniarc", tmp_path, False)

    def test_all_command_success(self, tmp_path):
        """Test successful download all command."""
        with patch("download_dataset._setup_output_directory") as mock_setup:
            with patch("download_dataset._download_all_datasets") as mock_download:
                mock_setup.return_value = tmp_path
                
                result = self.runner.invoke(
                    app, ["all", "--output", str(tmp_path), "--force"]
                )
                
                assert result.exit_code == 0
                mock_setup.assert_called_once_with(tmp_path)
                mock_download.assert_called_once_with(tmp_path, True)

    def test_help_command(self):
        """Test help command shows usage information."""
        result = self.runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "Download ARC datasets from GitHub repositories" in result.output
        assert "Examples:" in result.output
        assert "arc-agi-1" in result.output
        assert "arc-agi-2" in result.output

    def test_arc_agi_1_help(self):
        """Test ARC-AGI-1 command help."""
        result = self.runner.invoke(app, ["arc-agi-1", "--help"])
        
        assert result.exit_code == 0
        assert "Download ARC-AGI-1 dataset from GitHub" in result.output
        assert "fchollet/ARC-AGI" in result.output

    def test_arc_agi_2_help(self):
        """Test ARC-AGI-2 command help."""
        result = self.runner.invoke(app, ["arc-agi-2", "--help"])
        
        assert result.exit_code == 0
        assert "Download ARC-AGI-2 dataset from GitHub" in result.output
        assert "arcprize/ARC-AGI-2" in result.output

    def test_invalid_command(self):
        """Test invalid command shows error."""
        result = self.runner.invoke(app, ["invalid-dataset"])
        
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_download_error_handling(self, tmp_path):
        """Test error handling in download commands."""
        with patch("download_dataset.DatasetDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.download_arc_agi_1.side_effect = DatasetDownloadError("Network error")
            mock_downloader_class.return_value = mock_downloader
            
            with patch("download_dataset.get_raw_path", return_value=tmp_path):
                result = self.runner.invoke(app, ["arc-agi-1"])
                
                # Should exit with error code
                assert result.exit_code == 1
                # Error messages are logged to stderr, not captured in CLI output
                # Just verify the exit code is correct


class TestDownloadScriptHelpers:
    """Test suite for download script helper functions."""

    def test_setup_output_directory_default(self):
        """Test setup output directory with default path."""
        with patch("download_dataset.get_raw_path") as mock_get_path:
            mock_path = Path("/default/path")
            mock_get_path.return_value = mock_path
            
            result = _setup_output_directory(None)
            
            assert result == mock_path
            mock_get_path.assert_called_once_with(create=True)

    def test_setup_output_directory_custom(self, tmp_path):
        """Test setup output directory with custom path."""
        custom_path = tmp_path / "custom"
        
        result = _setup_output_directory(custom_path)
        
        assert result == custom_path
        assert custom_path.exists()

    @patch("download_dataset.sys.exit")
    def test_setup_output_directory_error(self, mock_exit, tmp_path):
        """Test setup output directory with permission error."""
        # Create a path that will cause an OSError
        bad_path = tmp_path / "nonexistent" / "deeply" / "nested" / "path"
        
        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            _setup_output_directory(bad_path)
            
            mock_exit.assert_called_once_with(1)

    def test_check_existing_dataset_exists_no_force(self, tmp_path):
        """Test check existing dataset when dataset exists and no force."""
        dataset_dir = tmp_path / "ARC-AGI-1"
        dataset_dir.mkdir()
        
        result = _check_existing_dataset(dataset_dir, "ARC-AGI-1", False)
        
        assert result is True

    def test_check_existing_dataset_exists_with_force(self, tmp_path):
        """Test check existing dataset when dataset exists with force."""
        dataset_dir = tmp_path / "ARC-AGI-1"
        dataset_dir.mkdir()
        
        result = _check_existing_dataset(dataset_dir, "ARC-AGI-1", True)
        
        assert result is False

    def test_check_existing_dataset_not_exists(self, tmp_path):
        """Test check existing dataset when dataset doesn't exist."""
        dataset_dir = tmp_path / "ARC-AGI-1"
        
        result = _check_existing_dataset(dataset_dir, "ARC-AGI-1", False)
        
        assert result is False

    def test_log_dataset_info_arc_agi(self, tmp_path):
        """Test logging dataset info for ARC-AGI datasets."""
        config = DATASETS["arc-agi-1"]
        
        # This should not raise an exception
        _log_dataset_info(config, tmp_path)

    def test_log_dataset_info_conceptarc(self, tmp_path):
        """Test logging dataset info for ConceptARC."""
        config = DATASETS["conceptarc"]
        
        # This should not raise an exception
        _log_dataset_info(config, tmp_path)

    def test_log_dataset_info_miniarc(self, tmp_path):
        """Test logging dataset info for MiniARC."""
        config = DATASETS["miniarc"]
        
        # This should not raise an exception
        _log_dataset_info(config, tmp_path)

    def test_download_single_dataset_unknown(self):
        """Test download single dataset with unknown dataset key."""
        with patch("download_dataset.sys.exit") as mock_exit:
            # Mock sys.exit to prevent actual exit and allow us to check the call
            mock_exit.side_effect = SystemExit(1)
            
            with pytest.raises(SystemExit):
                _download_single_dataset("unknown-dataset")
            
            mock_exit.assert_called_once_with(1)

    def test_download_single_dataset_existing_no_force(self, tmp_path):
        """Test download single dataset when dataset exists without force."""
        dataset_dir = tmp_path / "ARC-AGI-1"
        dataset_dir.mkdir()
        
        with patch("download_dataset.get_raw_path", return_value=tmp_path):
            # Should return early without downloading
            _download_single_dataset("arc-agi-1", tmp_path, False)

    @patch("download_dataset.sys.exit")
    def test_download_single_dataset_download_error(self, mock_exit, tmp_path):
        """Test download single dataset with download error."""
        with patch("download_dataset.DatasetDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.download_arc_agi_1.side_effect = DatasetDownloadError("Network error")
            mock_downloader_class.return_value = mock_downloader
            
            _download_single_dataset("arc-agi-1", tmp_path, True)
            
            mock_exit.assert_called_once_with(1)

    def test_download_single_dataset_success(self, tmp_path):
        """Test successful single dataset download."""
        with patch("download_dataset.DatasetDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.download_arc_agi_1.return_value = tmp_path / "ARC-AGI-1"
            mock_downloader_class.return_value = mock_downloader
            
            # Should complete without error
            _download_single_dataset("arc-agi-1", tmp_path, True)
            
            mock_downloader.download_arc_agi_1.assert_called_once()

    def test_download_all_datasets_success(self, tmp_path):
        """Test successful download of all datasets."""
        with patch("download_dataset.DatasetDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.download_arc_agi_1.return_value = tmp_path / "ARC-AGI-1"
            mock_downloader.download_arc_agi_2.return_value = tmp_path / "ARC-AGI-2"
            mock_downloader.download_conceptarc.return_value = tmp_path / "ConceptARC"
            mock_downloader.download_miniarc.return_value = tmp_path / "MiniARC"
            mock_downloader_class.return_value = mock_downloader
            
            _download_all_datasets(tmp_path, True)
            
            # All download methods should be called
            mock_downloader.download_arc_agi_1.assert_called_once()
            mock_downloader.download_arc_agi_2.assert_called_once()
            mock_downloader.download_conceptarc.assert_called_once()
            mock_downloader.download_miniarc.assert_called_once()

    def test_download_all_datasets_existing_no_force(self, tmp_path):
        """Test download all datasets when some exist without force."""
        # Create existing dataset directories
        (tmp_path / "ARC-AGI-1").mkdir()
        (tmp_path / "ConceptARC").mkdir()
        
        with patch("download_dataset.DatasetDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.download_arc_agi_2.return_value = tmp_path / "ARC-AGI-2"
            mock_downloader.download_miniarc.return_value = tmp_path / "MiniARC"
            mock_downloader_class.return_value = mock_downloader
            
            _download_all_datasets(tmp_path, False)
            
            # Only non-existing datasets should be downloaded
            mock_downloader.download_arc_agi_1.assert_not_called()
            mock_downloader.download_conceptarc.assert_not_called()
            mock_downloader.download_arc_agi_2.assert_called_once()
            mock_downloader.download_miniarc.assert_called_once()

    def test_download_all_datasets_error(self, tmp_path):
        """Test download all datasets with error."""
        with patch("download_dataset.DatasetDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.download_arc_agi_1.side_effect = DatasetDownloadError("Network error")
            mock_downloader_class.return_value = mock_downloader
            
            with pytest.raises(DatasetDownloadError):
                _download_all_datasets(tmp_path, True)


class TestDatasetConfig:
    """Test suite for DatasetConfig dataclass."""

    def test_dataset_config_creation(self):
        """Test DatasetConfig creation and immutability."""
        config = DatasetConfig(
            name="test",
            display_name="Test Dataset",
            downloader_method="download_test",
            expected_structure={"tasks": 100},
            next_steps=["Step 1", "Step 2"]
        )
        
        assert config.name == "test"
        assert config.display_name == "Test Dataset"
        assert config.downloader_method == "download_test"
        assert config.expected_structure == {"tasks": 100}
        assert config.next_steps == ["Step 1", "Step 2"]
        
        # Test immutability
        with pytest.raises(AttributeError):
            config.name = "modified"

    def test_datasets_configuration(self):
        """Test that all dataset configurations are properly defined."""
        required_datasets = ["arc-agi-1", "arc-agi-2", "conceptarc", "miniarc"]
        
        for dataset_key in required_datasets:
            assert dataset_key in DATASETS
            config = DATASETS[dataset_key]
            
            # Verify all required fields are present
            assert config.name
            assert config.display_name
            assert config.downloader_method
            assert config.expected_structure
            assert config.next_steps
            
            # Verify downloader method names match expected pattern
            if dataset_key == "arc-agi-1":
                assert config.downloader_method == "download_arc_agi_1"
            elif dataset_key == "arc-agi-2":
                assert config.downloader_method == "download_arc_agi_2"
            elif dataset_key == "conceptarc":
                assert config.downloader_method == "download_conceptarc"
            elif dataset_key == "miniarc":
                assert config.downloader_method == "download_miniarc"

    def test_arc_agi_1_config(self):
        """Test ARC-AGI-1 specific configuration."""
        config = DATASETS["arc-agi-1"]
        
        assert config.name == "arc-agi-1"
        assert config.display_name == "ARC-AGI-1"
        assert config.expected_structure["training_tasks"] == 400
        assert config.expected_structure["evaluation_tasks"] == 400
        assert "data/training/" in config.expected_structure["training_dir"]
        assert "data/evaluation/" in config.expected_structure["evaluation_dir"]

    def test_arc_agi_2_config(self):
        """Test ARC-AGI-2 specific configuration."""
        config = DATASETS["arc-agi-2"]
        
        assert config.name == "arc-agi-2"
        assert config.display_name == "ARC-AGI-2"
        assert config.expected_structure["training_tasks"] == 1000
        assert config.expected_structure["evaluation_tasks"] == 120
        assert "data/training/" in config.expected_structure["training_dir"]
        assert "data/evaluation/" in config.expected_structure["evaluation_dir"]

    def test_conceptarc_config(self):
        """Test ConceptARC specific configuration."""
        config = DATASETS["conceptarc"]
        
        assert config.name == "conceptarc"
        assert config.display_name == "ConceptARC"
        assert config.expected_structure["concept_groups"] == 16
        assert config.expected_structure["tasks_per_group"] == 10
        assert "corpus/" in config.expected_structure["data_dir"]

    def test_miniarc_config(self):
        """Test MiniARC specific configuration."""
        config = DATASETS["miniarc"]
        
        assert config.name == "miniarc"
        assert config.display_name == "MiniARC"
        assert config.expected_structure["total_tasks"] == "400+"
        assert config.expected_structure["optimization"] == "5x5 grids"
        assert "data/" in config.expected_structure["data_dir"]


class TestDownloadScriptIntegration:
    """Integration tests for download script functionality."""

    def test_cli_integration_arc_agi_1(self, tmp_path):
        """Test CLI integration for ARC-AGI-1 download."""
        runner = CliRunner()
        
        with patch("download_dataset.DatasetDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.download_arc_agi_1.return_value = tmp_path / "ARC-AGI-1"
            mock_downloader_class.return_value = mock_downloader
            
            result = runner.invoke(
                app, ["arc-agi-1", "--output", str(tmp_path), "--force"]
            )
            
            assert result.exit_code == 0
            mock_downloader.download_arc_agi_1.assert_called_once()

    def test_cli_integration_error_handling(self, tmp_path):
        """Test CLI integration error handling."""
        runner = CliRunner()
        
        with patch("download_dataset.DatasetDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.download_arc_agi_1.side_effect = DatasetDownloadError("Test error")
            mock_downloader_class.return_value = mock_downloader
            
            result = runner.invoke(
                app, ["arc-agi-1", "--output", str(tmp_path), "--force"]
            )
            
            assert result.exit_code == 1

    def test_cli_parameter_validation(self):
        """Test CLI parameter validation."""
        runner = CliRunner()
        
        # Test with invalid output directory path
        result = runner.invoke(
            app, ["arc-agi-1", "--output", "/invalid/path/that/cannot/be/created"]
        )
        
        # Should handle the error gracefully
        assert result.exit_code == 1