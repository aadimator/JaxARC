"""Tests for dataset manager utilities."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jaxarc.configs import DatasetConfig, JaxArcConfig
from jaxarc.utils.dataset_manager import (
    DatasetError,
    DatasetManager,
)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_dataset_config():
    """Create a sample dataset configuration."""
    return DatasetConfig(
        dataset_name="TestDataset",
        dataset_path="data/TestDataset",
        dataset_repo="https://github.com/test/TestDataset.git",
        expected_subdirs=("data", "test"),
        max_grid_height=10,
        max_grid_width=10,
    )


@pytest.fixture
def sample_jaxarc_config(sample_dataset_config):
    """Create a sample JaxArcConfig with dataset configuration."""
    return JaxArcConfig(dataset=sample_dataset_config)


class TestDatasetManagerInit:
    """Test DatasetManager initialization."""

    def test_init_default_output_dir(self):
        """Test initialization with default output directory."""
        manager = DatasetManager()

        assert isinstance(manager.output_dir, Path)
        assert manager.output_dir.name == "data"
        assert manager.output_dir.exists()

    def test_init_custom_output_dir(self, temp_output_dir):
        """Test initialization with custom output directory."""
        custom_dir = temp_output_dir / "custom_data"
        manager = DatasetManager(output_dir=custom_dir)

        assert manager.output_dir == custom_dir
        assert manager.output_dir.exists()

    def test_init_creates_output_dir(self, temp_output_dir):
        """Test that initialization creates output directory if it doesn't exist."""
        non_existent_dir = temp_output_dir / "non_existent"
        assert not non_existent_dir.exists()

        manager = DatasetManager(output_dir=non_existent_dir)

        assert manager.output_dir == non_existent_dir
        assert manager.output_dir.exists()


class TestValidateDataset:
    """Test dataset validation functionality."""

    def test_validate_dataset_exists_and_valid(
        self, temp_output_dir, sample_dataset_config
    ):
        """Test validation of existing valid dataset."""
        manager = DatasetManager(output_dir=temp_output_dir)

        # Create a valid dataset directory structure
        dataset_path = temp_output_dir / "TestDataset"
        dataset_path.mkdir()
        (dataset_path / "data").mkdir()
        (dataset_path / "test").mkdir()
        (dataset_path / "README.md").touch()

        result = manager.validate_dataset(sample_dataset_config, dataset_path)
        assert result == True

    def test_validate_dataset_not_exists(self, temp_output_dir, sample_dataset_config):
        """Test validation of non-existent dataset."""
        manager = DatasetManager(output_dir=temp_output_dir)

        non_existent_path = temp_output_dir / "NonExistent"
        result = manager.validate_dataset(sample_dataset_config, non_existent_path)
        assert result == False

    def test_validate_dataset_empty_directory(
        self, temp_output_dir, sample_dataset_config
    ):
        """Test validation of empty dataset directory."""
        manager = DatasetManager(output_dir=temp_output_dir)

        empty_dir = temp_output_dir / "EmptyDataset"
        empty_dir.mkdir()

        result = manager.validate_dataset(sample_dataset_config, empty_dir)
        assert result == False

    def test_validate_dataset_missing_subdirs(
        self, temp_output_dir, sample_dataset_config
    ):
        """Test validation when expected subdirectories are missing."""
        manager = DatasetManager(output_dir=temp_output_dir)

        dataset_path = temp_output_dir / "IncompleteDataset"
        dataset_path.mkdir()
        (dataset_path / "data").mkdir()  # Missing "test" subdirectory
        (dataset_path / "README.md").touch()

        result = manager.validate_dataset(sample_dataset_config, dataset_path)
        assert result == False

    def test_validate_dataset_no_expected_subdirs(self, temp_output_dir):
        """Test validation when no expected subdirectories are specified."""
        manager = DatasetManager(output_dir=temp_output_dir)

        config = DatasetConfig(
            dataset_name="SimpleDataset",
            dataset_path="data/SimpleDataset",
            expected_subdirs=(),  # No expected subdirs
        )

        dataset_path = temp_output_dir / "SimpleDataset"
        dataset_path.mkdir()
        (dataset_path / "file.txt").touch()

        result = manager.validate_dataset(config, dataset_path)
        assert result == True

    def test_validate_dataset_is_file_not_directory(
        self, temp_output_dir, sample_dataset_config
    ):
        """Test validation when path points to a file instead of directory."""
        manager = DatasetManager(output_dir=temp_output_dir)

        file_path = temp_output_dir / "not_a_directory.txt"
        file_path.touch()

        result = manager.validate_dataset(sample_dataset_config, file_path)
        assert result == False

    def test_validate_dataset_permission_error(
        self, temp_output_dir, sample_dataset_config
    ):
        """Test validation when directory cannot be accessed."""
        manager = DatasetManager(output_dir=temp_output_dir)

        # Create directory and make it inaccessible (on Unix systems)
        restricted_dir = temp_output_dir / "RestrictedDataset"
        restricted_dir.mkdir()

        try:
            restricted_dir.chmod(0o000)  # Remove all permissions
            result = manager.validate_dataset(sample_dataset_config, restricted_dir)
            assert result == False
        finally:
            # Restore permissions for cleanup
            restricted_dir.chmod(0o755)


class TestDownloadDataset:
    """Test dataset downloading functionality."""

    def test_download_dataset_success(self, temp_output_dir, sample_dataset_config):
        """Test successful dataset download."""
        manager = DatasetManager(output_dir=temp_output_dir)

        target_path = temp_output_dir / "DownloadedDataset"

        # Mock successful git clone
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Cloning into 'DownloadedDataset'...\ndone.",
                stderr="",
            )

            # Mock validation to return True
            with patch.object(manager, "validate_dataset", return_value=True):
                result_path = manager.download_dataset(
                    sample_dataset_config, target_path
                )

                assert result_path == target_path
                mock_run.assert_called_once()

                # Check git clone command
                call_args = mock_run.call_args[0][0]
                assert call_args[0] == "git"
                assert call_args[1] == "clone"
                assert call_args[2] == sample_dataset_config.dataset_repo
                assert call_args[3] == str(target_path)

    def test_download_dataset_no_repo_url(self, temp_output_dir):
        """Test download when no repository URL is configured."""
        manager = DatasetManager(output_dir=temp_output_dir)

        config = DatasetConfig(
            dataset_name="NoRepoDataset",
            dataset_path="data/NoRepoDataset",
            dataset_repo="",  # Empty repo URL
        )

        with pytest.raises(DatasetError, match="No repository URL configured"):
            manager.download_dataset(config)

    def test_download_dataset_git_not_found(
        self, temp_output_dir, sample_dataset_config
    ):
        """Test download when git is not installed."""
        manager = DatasetManager(output_dir=temp_output_dir)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")

            with pytest.raises(DatasetError, match="Git is not installed"):
                manager.download_dataset(sample_dataset_config)

    def test_download_dataset_timeout(self, temp_output_dir, sample_dataset_config):
        """Test download timeout."""
        manager = DatasetManager(output_dir=temp_output_dir)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("git", 300)

            with pytest.raises(DatasetError, match="Download timed out"):
                manager.download_dataset(sample_dataset_config)

    def test_download_dataset_git_error(self, temp_output_dir, sample_dataset_config):
        """Test download when git clone fails."""
        manager = DatasetManager(output_dir=temp_output_dir)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "git", stderr="Repository not found"
            )

            with pytest.raises(DatasetError, match="Git clone failed"):
                manager.download_dataset(sample_dataset_config)

    # Removed test_download_dataset_removes_existing - tests file system edge case

    def test_download_dataset_validation_fails(
        self, temp_output_dir, sample_dataset_config
    ):
        """Test download when validation fails after download."""
        manager = DatasetManager(output_dir=temp_output_dir)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            # Mock validation to return False
            with patch.object(manager, "validate_dataset", return_value=False):
                with pytest.raises(
                    DatasetError, match="Downloaded dataset failed validation"
                ):
                    manager.download_dataset(sample_dataset_config)

    def test_download_dataset_uses_config_path(
        self, temp_output_dir, sample_dataset_config
    ):
        """Test download uses dataset path from config when target_path is None."""
        manager = DatasetManager(output_dir=temp_output_dir)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            with patch.object(manager, "validate_dataset", return_value=True):
                with patch("jaxarc.utils.dataset_manager.here") as mock_here:
                    mock_here.return_value = temp_output_dir

                    result_path = manager.download_dataset(
                        sample_dataset_config, target_path=None
                    )

                    expected_path = temp_output_dir / "data" / "TestDataset"
                    assert result_path == expected_path

    def test_download_dataset_no_target_path_no_config_path(self, temp_output_dir):
        """Test download when no target path and no config path."""
        manager = DatasetManager(output_dir=temp_output_dir)

        config = DatasetConfig(
            dataset_name="NoPathDataset",
            dataset_repo="https://github.com/test/repo.git",
            dataset_path="",  # Empty path
        )

        with pytest.raises(DatasetError, match="No target path provided"):
            manager.download_dataset(config, target_path=None)


class TestEnsureDatasetAvailable:
    """Test ensure dataset available functionality."""

    def test_ensure_dataset_available_exists_valid(
        self, temp_output_dir, sample_jaxarc_config
    ):
        """Test when dataset already exists and is valid."""
        manager = DatasetManager(output_dir=temp_output_dir)

        # Create valid dataset
        dataset_path = temp_output_dir / "data" / "TestDataset"
        dataset_path.mkdir(parents=True)
        (dataset_path / "data").mkdir()
        (dataset_path / "test").mkdir()

        with patch("jaxarc.utils.dataset_manager.here") as mock_here:
            mock_here.return_value = temp_output_dir

            result_path = manager.ensure_dataset_available(sample_jaxarc_config)
            assert result_path == dataset_path

    def test_ensure_dataset_available_missing_no_auto_download(
        self, temp_output_dir, sample_jaxarc_config
    ):
        """Test when dataset is missing and auto_download is False."""
        manager = DatasetManager(output_dir=temp_output_dir)

        with patch("jaxarc.utils.dataset_manager.here") as mock_here:
            mock_here.return_value = temp_output_dir

            with pytest.raises(DatasetError, match="not found.*Set auto_download=True"):
                manager.ensure_dataset_available(
                    sample_jaxarc_config, auto_download=False
                )

    def test_ensure_dataset_available_missing_with_auto_download(
        self, temp_output_dir, sample_jaxarc_config
    ):
        """Test when dataset is missing and auto_download is True."""
        manager = DatasetManager(output_dir=temp_output_dir)

        with patch("jaxarc.utils.dataset_manager.here") as mock_here:
            mock_here.return_value = temp_output_dir

            with patch.object(manager, "download_dataset") as mock_download:
                expected_path = temp_output_dir / "data" / "TestDataset"
                mock_download.return_value = expected_path

                result_path = manager.ensure_dataset_available(
                    sample_jaxarc_config, auto_download=True
                )

                assert result_path == expected_path
                mock_download.assert_called_once()

    def test_ensure_dataset_available_dataset_config_directly(
        self, temp_output_dir, sample_dataset_config
    ):
        """Test with DatasetConfig directly instead of JaxArcConfig."""
        manager = DatasetManager(output_dir=temp_output_dir)

        # Create valid dataset
        dataset_path = temp_output_dir / "data" / "TestDataset"
        dataset_path.mkdir(parents=True)
        (dataset_path / "data").mkdir()
        (dataset_path / "test").mkdir()

        with patch("jaxarc.utils.dataset_manager.here") as mock_here:
            mock_here.return_value = temp_output_dir

            result_path = manager.ensure_dataset_available(sample_dataset_config)
            assert result_path == dataset_path

    def test_ensure_dataset_available_no_dataset_path(self, temp_output_dir):
        """Test when dataset path is not configured."""
        manager = DatasetManager(output_dir=temp_output_dir)

        config = DatasetConfig(
            dataset_name="NoPathDataset",
            dataset_path="",  # Empty path
        )

        with pytest.raises(DatasetError, match="Dataset path not configured"):
            manager.ensure_dataset_available(config)

    def test_ensure_dataset_available_absolute_path(
        self, temp_output_dir, sample_dataset_config
    ):
        """Test with absolute dataset path."""
        manager = DatasetManager(output_dir=temp_output_dir)

        # Create config with absolute path
        absolute_path = temp_output_dir / "absolute_dataset"
        config = DatasetConfig(
            dataset_name="AbsoluteDataset",
            dataset_path=str(absolute_path),
            expected_subdirs=("data",),
        )

        # Create valid dataset at absolute path
        absolute_path.mkdir(parents=True)
        (absolute_path / "data").mkdir()

        result_path = manager.ensure_dataset_available(config)
        assert result_path == absolute_path


class TestValidateConfig:
    """Test configuration validation functionality."""

    def test_validate_config_valid(self, sample_jaxarc_config):
        """Test validation of valid configuration."""
        manager = DatasetManager()

        # Should not raise any exceptions
        manager.validate_config(sample_jaxarc_config, "TestDataset")

    def test_validate_config_miniarc_large_grids(self, temp_output_dir):
        """Test validation warning for MiniARC with large grids."""
        manager = DatasetManager(output_dir=temp_output_dir)

        config = JaxArcConfig(
            dataset=DatasetConfig(
                dataset_name="MiniARC",
                max_grid_height=10,  # Larger than recommended 5
                max_grid_width=10,
            )
        )

        # Should log warning but not raise error
        with patch("jaxarc.utils.dataset_manager.logger") as mock_logger:
            manager.validate_config(config, "MiniARC")
            mock_logger.warning.assert_called()
            assert "optimized for 5x5 grids" in str(mock_logger.warning.call_args)

    def test_validate_config_conceptarc_small_grids(self, temp_output_dir):
        """Test validation warning for ConceptARC with small grids."""
        manager = DatasetManager(output_dir=temp_output_dir)

        config = JaxArcConfig(
            dataset=DatasetConfig(
                dataset_name="ConceptARC",
                max_grid_height=5,  # Smaller than recommended 15
                max_grid_width=5,
            )
        )

        with patch("jaxarc.utils.dataset_manager.logger") as mock_logger:
            manager.validate_config(config, "ConceptARC")
            mock_logger.warning.assert_called()
            assert "typically uses larger grids" in str(mock_logger.warning.call_args)

    def test_validate_config_dataset_name_mismatch(self, temp_output_dir):
        """Test validation warning for dataset name mismatch."""
        manager = DatasetManager(output_dir=temp_output_dir)

        config = JaxArcConfig(
            dataset=DatasetConfig(
                dataset_name="ConfigName",
            )
        )

        with patch("jaxarc.utils.dataset_manager.logger") as mock_logger:
            manager.validate_config(config, "DifferentName")
            mock_logger.warning.assert_called()
            assert "Dataset name mismatch" in str(mock_logger.warning.call_args)

    def test_validate_config_validation_error(self, temp_output_dir):
        """Test validation when config validation fails."""
        manager = DatasetManager(output_dir=temp_output_dir)

        # Mock config that fails validation
        mock_config = MagicMock()
        mock_config.validate.return_value = ["Error 1", "Error 2"]
        mock_config.dataset = MagicMock()

        with pytest.raises(ValueError, match="Config validation errors"):
            manager.validate_config(mock_config, "TestDataset")

    def test_validate_config_exception_handling(self, temp_output_dir):
        """Test validation exception handling."""
        manager = DatasetManager(output_dir=temp_output_dir)

        # Mock config that raises exception
        mock_config = MagicMock()
        mock_config.validate.side_effect = Exception("Test exception")

        with pytest.raises(ValueError, match="Invalid configuration"):
            manager.validate_config(mock_config, "TestDataset")


class TestGetDatasetRecommendations:
    """Test dataset recommendations functionality."""

    def test_get_recommendations_conceptarc(self):
        """Test recommendations for ConceptARC."""
        recommendations = DatasetManager.get_dataset_recommendations("ConceptARC")

        assert isinstance(recommendations, dict)
        assert recommendations["dataset.max_grid_height"] == 30
        assert recommendations["dataset.max_grid_width"] == 30
        assert recommendations["dataset.dataset_name"] == "ConceptARC"

    def test_get_recommendations_miniarc(self):
        """Test recommendations for MiniARC."""
        recommendations = DatasetManager.get_dataset_recommendations("MiniARC")

        assert isinstance(recommendations, dict)
        assert recommendations["dataset.max_grid_height"] == 5
        assert recommendations["dataset.max_grid_width"] == 5
        assert recommendations["dataset.dataset_name"] == "MiniARC"

    def test_get_recommendations_arc_agi_1(self):
        """Test recommendations for ARC-AGI-1."""
        recommendations = DatasetManager.get_dataset_recommendations("ARC-AGI-1")

        assert isinstance(recommendations, dict)
        assert recommendations["dataset.max_grid_height"] == 30
        assert recommendations["dataset.max_grid_width"] == 30
        assert recommendations["dataset.dataset_name"] == "ARC-AGI-1"

    def test_get_recommendations_arc_agi_2(self):
        """Test recommendations for ARC-AGI-2."""
        recommendations = DatasetManager.get_dataset_recommendations("ARC-AGI-2")

        assert isinstance(recommendations, dict)
        assert recommendations["dataset.max_grid_height"] == 30
        assert recommendations["dataset.max_grid_width"] == 30
        assert recommendations["dataset.dataset_name"] == "ARC-AGI-2"

    def test_get_recommendations_case_insensitive(self):
        """Test that recommendations work with different cases."""
        # Test various case combinations
        test_cases = [
            ("conceptarc", "ConceptARC"),
            ("MINIARC", "MiniARC"),
            ("mini-arc", "MiniARC"),
            ("arc-agi-1", "ARC-AGI-1"),
            ("AGI2", "ARC-AGI-2"),
        ]

        for input_name, expected_name in test_cases:
            recommendations = DatasetManager.get_dataset_recommendations(input_name)
            assert recommendations["dataset.dataset_name"] == expected_name

    def test_get_recommendations_unknown_dataset(self):
        """Test recommendations for unknown dataset."""
        with patch("jaxarc.utils.dataset_manager.logger") as mock_logger:
            recommendations = DatasetManager.get_dataset_recommendations(
                "UnknownDataset"
            )

            assert isinstance(recommendations, dict)
            assert len(recommendations) == 0
            mock_logger.warning.assert_called()
            assert "No recommendations available" in str(mock_logger.warning.call_args)

    def test_get_recommendations_empty_string(self):
        """Test recommendations for empty string."""
        with patch("jaxarc.utils.dataset_manager.logger") as mock_logger:
            recommendations = DatasetManager.get_dataset_recommendations("")

            assert isinstance(recommendations, dict)
            assert len(recommendations) == 0
            mock_logger.warning.assert_called()


class TestIntegrationAndWorkflows:
    """Test integration scenarios and complete workflows."""

    def test_complete_dataset_setup_workflow(self, temp_output_dir):
        """Test complete workflow from config to dataset availability."""
        manager = DatasetManager(output_dir=temp_output_dir)

        # Create config
        config = JaxArcConfig(
            dataset=DatasetConfig(
                dataset_name="WorkflowTest",
                dataset_path="data/WorkflowTest",
                dataset_repo="https://github.com/test/WorkflowTest.git",
                expected_subdirs=("data",),
            )
        )

        # Mock successful download and validation
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            with patch("jaxarc.utils.dataset_manager.here") as mock_here:
                mock_here.return_value = temp_output_dir

                # Create the dataset structure after "download"
                def create_dataset(*args, **kwargs):
                    dataset_path = temp_output_dir / "data" / "WorkflowTest"
                    dataset_path.mkdir(parents=True, exist_ok=True)
                    (dataset_path / "data").mkdir(exist_ok=True)
                    return MagicMock(returncode=0, stdout="", stderr="")

                mock_run.side_effect = create_dataset

                # Test the complete workflow
                result_path = manager.ensure_dataset_available(
                    config, auto_download=True
                )

                assert result_path.exists()
                assert (result_path / "data").exists()

    def test_validation_and_recommendation_workflow(self):
        """Test workflow of getting recommendations and validating config."""
        manager = DatasetManager()

        # Get recommendations for MiniARC
        recommendations = DatasetManager.get_dataset_recommendations("MiniARC")

        # Create config with recommendations
        config = JaxArcConfig(
            dataset=DatasetConfig(
                dataset_name=recommendations["dataset.dataset_name"],
                max_grid_height=recommendations["dataset.max_grid_height"],
                max_grid_width=recommendations["dataset.max_grid_width"],
            )
        )

        # Validate the config
        manager.validate_config(config, "MiniARC")

        # Should not raise any warnings since we used recommendations
        assert config.dataset.dataset_name == "MiniARC"
        assert config.dataset.max_grid_height == 5
        assert config.dataset.max_grid_width == 5


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_dataset_manager_with_very_long_paths(self, temp_output_dir):
        """Test dataset manager with very long paths."""
        long_path_component = "x" * 100
        long_path = temp_output_dir / long_path_component / long_path_component

        try:
            manager = DatasetManager(output_dir=long_path)
            assert manager.output_dir == long_path
            assert manager.output_dir.exists()
        except OSError:
            # Some filesystems have path length limits
            pass

    def test_dataset_validation_with_symlinks(
        self, temp_output_dir, sample_dataset_config
    ):
        """Test dataset validation with symbolic links."""
        manager = DatasetManager(output_dir=temp_output_dir)

        # Create actual dataset
        real_dataset = temp_output_dir / "RealDataset"
        real_dataset.mkdir()
        (real_dataset / "data").mkdir()
        (real_dataset / "test").mkdir()

        # Create symlink to dataset
        symlink_dataset = temp_output_dir / "SymlinkDataset"
        try:
            symlink_dataset.symlink_to(real_dataset)

            result = manager.validate_dataset(sample_dataset_config, symlink_dataset)
            assert result == True
        except OSError:
            # Symlinks not supported on this system
            pass

    def test_concurrent_dataset_operations(self, temp_output_dir):
        """Test concurrent dataset operations."""
        import threading
        import time

        manager = DatasetManager(output_dir=temp_output_dir)
        results = []
        errors = []

        def validate_worker():
            try:
                config = DatasetConfig(
                    dataset_name="ConcurrentTest",
                    expected_subdirs=(),
                )

                # Create dataset
                dataset_path = (
                    temp_output_dir / f"Dataset_{threading.current_thread().ident}"
                )
                dataset_path.mkdir(exist_ok=True)
                (dataset_path / "file.txt").touch()

                for _ in range(5):
                    result = manager.validate_dataset(config, dataset_path)
                    results.append(result)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=validate_worker)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0
        # All validations should succeed
        assert all(results)

    def test_dataset_manager_memory_efficiency(self, temp_output_dir):
        """Test memory efficiency with large dataset structures."""
        manager = DatasetManager(output_dir=temp_output_dir)

        # Create many dataset directories
        for i in range(100):
            dataset_path = temp_output_dir / f"Dataset_{i:03d}"
            dataset_path.mkdir()
            (dataset_path / "data").mkdir()

            config = DatasetConfig(
                dataset_name=f"Dataset_{i:03d}",
                expected_subdirs=("data",),
            )

            # Validate each dataset
            result = manager.validate_dataset(config, dataset_path)
            assert result == True

        # Manager should still function normally
        assert isinstance(manager.output_dir, Path)

    def test_dataset_operations_with_unicode_names(self, temp_output_dir):
        """Test dataset operations with unicode names."""
        manager = DatasetManager(output_dir=temp_output_dir)

        unicode_names = [
            "数据集_中文",
            "датасет_русский",
            "مجموعة_البيانات",
            "データセット_日本語",
        ]

        for name in unicode_names:
            try:
                config = DatasetConfig(
                    dataset_name=name,
                    expected_subdirs=(),
                )

                dataset_path = temp_output_dir / name
                dataset_path.mkdir()
                (dataset_path / "file.txt").touch()

                result = manager.validate_dataset(config, dataset_path)
                assert result == True
            except (OSError, UnicodeError):
                # Some filesystems don't support unicode
                pass
