"""Tests for dataset downloader functionality."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from jaxarc.utils.dataset_downloader import DatasetDownloader, DatasetDownloadError


class TestDatasetDownloader:
    """Test suite for DatasetDownloader class."""

    def test_init_default_output_dir(self):
        """Test DatasetDownloader initialization with default output directory."""
        downloader = DatasetDownloader()
        assert downloader.output_dir == Path.cwd()

    def test_init_custom_output_dir(self, tmp_path):
        """Test DatasetDownloader initialization with custom output directory."""
        downloader = DatasetDownloader(tmp_path)
        assert downloader.output_dir == tmp_path
        assert tmp_path.exists()

    @patch("jaxarc.utils.dataset_downloader.subprocess.run")
    @patch("jaxarc.utils.dataset_downloader.shutil.rmtree")
    def test_download_conceptarc_success(self, mock_rmtree, mock_run, tmp_path):
        """Test successful ConceptARC download."""
        # Setup mocks for multiple subprocess calls
        mock_run.side_effect = [
            Mock(returncode=0, stdout="git version 2.0", stderr=""),  # git --version
            Mock(returncode=0, stdout="", stderr=""),                 # ping (success)
            Mock(returncode=0, stdout="Success", stderr="")           # git clone
        ]
        
        # Create expected directory structure
        target_dir = tmp_path / "ConceptARC"
        corpus_dir = target_dir / "corpus"
        corpus_dir.mkdir(parents=True)
        
        # Create some concept group directories
        for concept in ["AboveBelow", "Center", "CleanUp", "CompleteShape", "Copy"]:
            (corpus_dir / concept).mkdir()
            # Add a dummy task file
            (corpus_dir / concept / "task1.json").write_text('{"test": "data"}')

        downloader = DatasetDownloader(tmp_path)
        
        with patch.object(downloader, "_validate_conceptarc_structure"):
            result = downloader.download_conceptarc()
        
        assert result == target_dir
        assert mock_run.call_count == 3
        # Check that git clone was called
        git_clone_call = mock_run.call_args_list[2]
        assert "git" in git_clone_call[0][0][0]
        assert "clone" in git_clone_call[0][0][1]

    @patch("jaxarc.utils.dataset_downloader.subprocess.run")
    @patch("jaxarc.utils.dataset_downloader.shutil.rmtree")
    def test_download_miniarc_success(self, mock_rmtree, mock_run, tmp_path):
        """Test successful MiniARC download."""
        # Setup mocks for multiple subprocess calls
        mock_run.side_effect = [
            Mock(returncode=0, stdout="git version 2.0", stderr=""),  # git --version
            Mock(returncode=0, stdout="", stderr=""),                 # ping (success)
            Mock(returncode=0, stdout="Success", stderr="")           # git clone
        ]
        
        # Create expected directory structure
        target_dir = tmp_path / "MiniARC"
        data_dir = target_dir / "data"
        data_dir.mkdir(parents=True)
        
        # Create some JSON files
        for i in range(150):
            (data_dir / f"task_{i}.json").write_text('{"test": "data"}')

        downloader = DatasetDownloader(tmp_path)
        
        with patch.object(downloader, "_validate_miniarc_structure"):
            result = downloader.download_miniarc()
        
        assert result == target_dir
        assert mock_run.call_count == 3
        # Check that git clone was called
        git_clone_call = mock_run.call_args_list[2]
        assert "git" in git_clone_call[0][0][0]
        assert "clone" in git_clone_call[0][0][1]

    @patch("jaxarc.utils.dataset_downloader.subprocess.run")
    def test_git_not_available(self, mock_run, tmp_path):
        """Test error when git is not available."""
        mock_run.side_effect = FileNotFoundError("git not found")
        
        downloader = DatasetDownloader(tmp_path)
        
        with pytest.raises(DatasetDownloadError, match="Git is not installed"):
            downloader.download_conceptarc()

    @patch("jaxarc.utils.dataset_downloader.subprocess.run")
    def test_git_clone_failure(self, mock_run, tmp_path):
        """Test error when git clone fails."""
        # First call (git --version) succeeds
        # Second call (ping) succeeds
        # Third call (git clone) fails
        mock_run.side_effect = [
            Mock(returncode=0, stdout="git version 2.0", stderr=""),
            Mock(returncode=0, stdout="", stderr=""),  # ping success
            subprocess.CalledProcessError(1, "git clone", stderr="Repository not found")
        ]
        
        downloader = DatasetDownloader(tmp_path)
        
        with pytest.raises(DatasetDownloadError, match="Failed to download ConceptARC"):
            downloader.download_conceptarc()

    @patch("jaxarc.utils.dataset_downloader.subprocess.run")
    def test_git_clone_timeout(self, mock_run, tmp_path):
        """Test error when git clone times out."""
        # First call (git --version) succeeds
        # Second call (ping) succeeds
        # Third call (git clone) times out
        mock_run.side_effect = [
            Mock(returncode=0, stdout="git version 2.0", stderr=""),
            Mock(returncode=0, stdout="", stderr=""),  # ping success
            subprocess.TimeoutExpired("git clone", 300)
        ]
        
        downloader = DatasetDownloader(tmp_path)
        
        with pytest.raises(DatasetDownloadError, match="Failed to download ConceptARC"):
            downloader.download_conceptarc()

    @patch("jaxarc.utils.dataset_downloader.shutil.disk_usage")
    def test_insufficient_disk_space(self, mock_disk_usage, tmp_path):
        """Test error when insufficient disk space."""
        # Mock disk usage to return very low free space
        mock_disk_usage.return_value = Mock(free=50 * 1024 * 1024)  # 50MB
        
        downloader = DatasetDownloader(tmp_path)
        
        with pytest.raises(DatasetDownloadError, match="Insufficient disk space"):
            downloader.download_conceptarc()

    def test_no_write_permissions(self, tmp_path):
        """Test error when no write permissions."""
        # Create a directory with no write permissions
        no_write_dir = tmp_path / "no_write"
        no_write_dir.mkdir()
        no_write_dir.chmod(0o444)  # Read-only
        
        downloader = DatasetDownloader(no_write_dir)
        
        try:
            with pytest.raises(DatasetDownloadError, match="No write permission"):
                downloader.download_conceptarc()
        finally:
            # Restore permissions for cleanup
            no_write_dir.chmod(0o755)

    def test_validate_conceptarc_structure_missing_corpus(self, tmp_path):
        """Test ConceptARC validation with missing corpus directory."""
        target_dir = tmp_path / "ConceptARC"
        target_dir.mkdir()
        
        downloader = DatasetDownloader(tmp_path)
        
        with pytest.raises(DatasetDownloadError, match="ConceptARC corpus directory not found"):
            downloader._validate_conceptarc_structure(target_dir)

    def test_validate_conceptarc_structure_few_concept_groups(self, tmp_path):
        """Test ConceptARC validation with few concept groups (should warn)."""
        target_dir = tmp_path / "ConceptARC"
        corpus_dir = target_dir / "corpus"
        corpus_dir.mkdir(parents=True)
        
        # Create only 3 concept groups (less than expected 5)
        for concept in ["AboveBelow", "Center", "CleanUp"]:
            (corpus_dir / concept).mkdir()
        
        downloader = DatasetDownloader(tmp_path)
        
        # Should not raise exception but should log warning
        # We'll test this by checking that it doesn't raise an exception
        # The warning is logged but we can't easily capture loguru logs in tests
        downloader._validate_conceptarc_structure(target_dir)

    def test_validate_miniarc_structure_missing_data(self, tmp_path):
        """Test MiniARC validation with missing data directory."""
        target_dir = tmp_path / "MiniARC"
        target_dir.mkdir()
        
        downloader = DatasetDownloader(tmp_path)
        
        with pytest.raises(DatasetDownloadError, match="MiniARC data directory not found"):
            downloader._validate_miniarc_structure(target_dir)

    def test_validate_miniarc_structure_alternative_structure(self, tmp_path):
        """Test MiniARC validation with alternative directory structure."""
        target_dir = tmp_path / "MiniARC"
        miniarc_subdir = target_dir / "MiniARC"
        miniarc_subdir.mkdir(parents=True)
        
        # Create some JSON files in the subdirectory
        for i in range(150):
            (miniarc_subdir / f"task_{i}.json").write_text('{"test": "data"}')
        
        downloader = DatasetDownloader(tmp_path)
        
        # Should not raise exception
        downloader._validate_miniarc_structure(target_dir)

    def test_validate_miniarc_structure_few_json_files(self, tmp_path):
        """Test MiniARC validation with few JSON files (should warn)."""
        target_dir = tmp_path / "MiniARC"
        data_dir = target_dir / "data"
        data_dir.mkdir(parents=True)
        
        # Create only 50 JSON files (less than expected 100)
        for i in range(50):
            (data_dir / f"task_{i}.json").write_text('{"test": "data"}')
        
        downloader = DatasetDownloader(tmp_path)
        
        # Should not raise exception but should log warning
        # We'll test this by checking that it doesn't raise an exception
        # The warning is logged but we can't easily capture loguru logs in tests
        downloader._validate_miniarc_structure(target_dir)

    def test_custom_target_directory(self, tmp_path):
        """Test download with custom target directory."""
        custom_dir = tmp_path / "custom_conceptarc"
        
        downloader = DatasetDownloader(tmp_path)
        
        with patch.object(downloader, "_clone_repository") as mock_clone:
            mock_clone.return_value = custom_dir
            result = downloader.download_conceptarc(custom_dir)
        
        assert result == custom_dir
        mock_clone.assert_called_once_with(
            "https://github.com/victorvikram/ConceptARC.git",
            custom_dir,
            "ConceptARC"
        )

    @patch("jaxarc.utils.dataset_downloader.subprocess.run")
    def test_network_connectivity_check_failure(self, mock_run, tmp_path):
        """Test network connectivity check failure (should warn but continue)."""
        # Mock git --version to succeed
        # Mock ping to fail
        mock_run.side_effect = [
            Mock(returncode=0, stdout="git version 2.0", stderr=""),  # git --version
            Mock(returncode=1, stdout="", stderr="ping failed"),      # ping
            Mock(returncode=0, stdout="Success", stderr="")           # git clone
        ]
        
        downloader = DatasetDownloader(tmp_path)
        
        # Mock the validation methods to avoid directory structure issues
        with patch.object(downloader, "_validate_conceptarc_structure"):
            with patch.object(downloader, "_validate_download"):
                result = downloader.download_conceptarc()
        
        # Should complete successfully despite network connectivity warning
        assert result == tmp_path / "ConceptARC"
        assert mock_run.call_count == 3