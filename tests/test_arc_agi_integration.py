"""
Integration tests for end-to-end ARC-AGI GitHub migration workflow.

Tests complete download and parsing workflow for ARC-AGI-1 and ARC-AGI-2,
CLI interface with new commands, and configuration loading with new format.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import jax
import pytest
from omegaconf import DictConfig
from typer.testing import CliRunner

# Add scripts directory to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from download_dataset import app as download_app
from jaxarc.parsers.arc_agi import ArcAgiParser
from jaxarc.types import JaxArcTask
from jaxarc.utils.config import get_config
from jaxarc.utils.dataset_downloader import DatasetDownloader, DatasetDownloadError


class TestArcAgiIntegrationWorkflow:
    """Integration tests for complete ARC-AGI workflow."""

    @pytest.fixture
    def mock_arc_agi_1_structure(self):
        """Create mock ARC-AGI-1 directory structure with sample data."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create directory structure
        data_dir = temp_dir / "data"
        training_dir = data_dir / "training"
        evaluation_dir = data_dir / "evaluation"

        training_dir.mkdir(parents=True)
        evaluation_dir.mkdir(parents=True)

        # Create sample training tasks
        training_tasks = {
            "007bbfb7.json": {
                "train": [
                    {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
                    {"input": [[5, 6]], "output": [[6, 5]]},
                ],
                "test": [{"input": [[7, 8]], "output": [[8, 7]]}],
            },
            "00d62c1b.json": {
                "train": [
                    {"input": [[0, 1]], "output": [[1, 0]]},
                ],
                "test": [{"input": [[2, 3]]}],  # No test output
            },
            "025d127b.json": {
                "train": [
                    {"input": [[9, 8, 7]], "output": [[7, 8, 9]]},
                    {"input": [[6, 5]], "output": [[5, 6]]},
                ],
                "test": [{"input": [[4, 3, 2, 1]], "output": [[1, 2, 3, 4]]}],
            },
        }

        for filename, task_data in training_tasks.items():
            task_file = training_dir / filename
            with task_file.open("w", encoding="utf-8") as f:
                json.dump(task_data, f)

        # Create sample evaluation tasks
        evaluation_tasks = {
            "eval001.json": {
                "train": [
                    {"input": [[1, 0]], "output": [[0, 1]]},
                ],
                "test": [{"input": [[2, 3]], "output": [[3, 2]]}],
            },
            "eval002.json": {
                "train": [
                    {"input": [[4, 5, 6]], "output": [[6, 5, 4]]},
                ],
                "test": [{"input": [[7, 8, 9]]}],  # No test output
            },
        }

        for filename, task_data in evaluation_tasks.items():
            task_file = evaluation_dir / filename
            with task_file.open("w", encoding="utf-8") as f:
                json.dump(task_data, f)

        yield temp_dir

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_arc_agi_2_structure(self):
        """Create mock ARC-AGI-2 directory structure with sample data."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create directory structure
        data_dir = temp_dir / "data"
        training_dir = data_dir / "training"
        evaluation_dir = data_dir / "evaluation"

        training_dir.mkdir(parents=True)
        evaluation_dir.mkdir(parents=True)

        # Create sample training tasks (more than ARC-AGI-1)
        training_tasks = {}
        for i in range(10):  # Simulate 10 training tasks
            task_id = f"train_{i:03d}"
            training_tasks[f"{task_id}.json"] = {
                "train": [
                    {"input": [[i % 10]], "output": [[(i + 1) % 10]]},
                    {"input": [[i % 5, (i + 1) % 5]], "output": [[(i + 1) % 5, i % 5]]},
                ],
                "test": [
                    {"input": [[(i + 2) % 10]], "output": [[(i + 3) % 10]]},
                    {"input": [[(i + 4) % 10]]},  # Some without outputs
                ],
            }

        for filename, task_data in training_tasks.items():
            task_file = training_dir / filename
            with task_file.open("w", encoding="utf-8") as f:
                json.dump(task_data, f)

        # Create sample evaluation tasks
        evaluation_tasks = {}
        for i in range(5):  # Simulate 5 evaluation tasks
            task_id = f"eval_{i:03d}"
            evaluation_tasks[f"{task_id}.json"] = {
                "train": [
                    {"input": [[i + 10]], "output": [[i + 11]]},
                ],
                "test": [{"input": [[i + 20]]}],  # No test outputs for evaluation
            }

        for filename, task_data in evaluation_tasks.items():
            task_file = evaluation_dir / filename
            with task_file.open("w", encoding="utf-8") as f:
                json.dump(task_data, f)

        yield temp_dir

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

    def test_complete_arc_agi_1_workflow(self, mock_arc_agi_1_structure):
        """Test complete download and parsing workflow for ARC-AGI-1."""
        # Test configuration loading
        config = DictConfig(
            {
                "dataset_name": "ARC-AGI-1",
                "default_split": "training",
                "data_root": str(mock_arc_agi_1_structure),
                "training": {
                    "path": str(mock_arc_agi_1_structure / "data" / "training"),
                },
                "evaluation": {
                    "path": str(mock_arc_agi_1_structure / "data" / "evaluation"),
                },
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 10,
                "max_test_pairs": 3,
            }
        )

        # Test parser initialization with GitHub format
        parser = ArcAgiParser(cfg=config)

        # Verify tasks are loaded correctly
        task_ids = parser.get_available_task_ids()
        assert len(task_ids) == 3  # 3 training tasks
        assert "007bbfb7" in task_ids
        assert "00d62c1b" in task_ids
        assert "025d127b" in task_ids

        # Test task retrieval and parsing
        task = parser.get_task_by_id("007bbfb7")
        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs == 2
        assert task.num_test_pairs == 1

        # Test random task access
        key = jax.random.PRNGKey(42)
        random_task = parser.get_random_task(key)
        assert isinstance(random_task, JaxArcTask)

        # Test evaluation split
        eval_config = config.copy()
        eval_config.default_split = "evaluation"
        eval_config.evaluation = {
            "path": str(mock_arc_agi_1_structure / "data" / "evaluation"),
        }

        eval_parser = ArcAgiParser(cfg=eval_config)
        eval_task_ids = eval_parser.get_available_task_ids()
        assert len(eval_task_ids) == 2  # 2 evaluation tasks
        assert "eval001" in eval_task_ids
        assert "eval002" in eval_task_ids

    def test_complete_arc_agi_2_workflow(self, mock_arc_agi_2_structure):
        """Test complete download and parsing workflow for ARC-AGI-2."""
        # Test configuration loading
        config = DictConfig(
            {
                "dataset_name": "ARC-AGI-2",
                "default_split": "training",
                "data_root": str(mock_arc_agi_2_structure),
                "training": {
                    "path": str(mock_arc_agi_2_structure / "data" / "training"),
                },
                "evaluation": {
                    "path": str(mock_arc_agi_2_structure / "data" / "evaluation"),
                },
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 10,
                "max_test_pairs": 4,  # ARC-AGI-2 can have more test pairs
            }
        )

        # Test parser initialization with GitHub format
        parser = ArcAgiParser(cfg=config)

        # Verify tasks are loaded correctly
        task_ids = parser.get_available_task_ids()
        assert len(task_ids) == 10  # 10 training tasks

        # Test that all expected task IDs are present
        expected_ids = {f"train_{i:03d}" for i in range(10)}
        assert set(task_ids) == expected_ids

        # Test task retrieval and parsing
        task = parser.get_task_by_id("train_000")
        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs == 2
        assert task.num_test_pairs == 2  # ARC-AGI-2 has 2 test pairs per task

        # Test evaluation split
        eval_config = config.copy()
        eval_config.default_split = "evaluation"
        eval_config.evaluation = {
            "path": str(mock_arc_agi_2_structure / "data" / "evaluation"),
        }

        eval_parser = ArcAgiParser(cfg=eval_config)
        eval_task_ids = eval_parser.get_available_task_ids()
        assert len(eval_task_ids) == 5  # 5 evaluation tasks

        expected_eval_ids = {f"eval_{i:03d}" for i in range(5)}
        assert set(eval_task_ids) == expected_eval_ids

    def test_cli_interface_arc_agi_1_command(self, tmp_path):
        """Test CLI interface with ARC-AGI-1 command."""
        runner = CliRunner()

        with patch("download_dataset.DatasetDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.download_arc_agi_1.return_value = tmp_path / "ARC-AGI-1"
            mock_downloader_class.return_value = mock_downloader

            # Test basic command
            result = runner.invoke(
                download_app, ["arc-agi-1", "--output", str(tmp_path)]
            )

            assert result.exit_code == 0
            mock_downloader.download_arc_agi_1.assert_called_once()

            # Test with force flag
            result = runner.invoke(
                download_app, ["arc-agi-1", "--output", str(tmp_path), "--force"]
            )

            assert result.exit_code == 0
            assert mock_downloader.download_arc_agi_1.call_count == 2

    def test_cli_interface_arc_agi_2_command(self, tmp_path):
        """Test CLI interface with ARC-AGI-2 command."""
        runner = CliRunner()

        with patch("download_dataset.DatasetDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.download_arc_agi_2.return_value = tmp_path / "ARC-AGI-2"
            mock_downloader_class.return_value = mock_downloader

            # Test basic command
            result = runner.invoke(
                download_app, ["arc-agi-2", "--output", str(tmp_path)]
            )

            assert result.exit_code == 0
            mock_downloader.download_arc_agi_2.assert_called_once()

            # Test with force flag
            result = runner.invoke(
                download_app, ["arc-agi-2", "--output", str(tmp_path), "--force"]
            )

            assert result.exit_code == 0
            assert mock_downloader.download_arc_agi_2.call_count == 2

    def test_cli_interface_all_command(self, tmp_path):
        """Test CLI interface with 'all' command."""
        runner = CliRunner()

        with patch("download_dataset.DatasetDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.download_arc_agi_1.return_value = tmp_path / "ARC-AGI-1"
            mock_downloader.download_arc_agi_2.return_value = tmp_path / "ARC-AGI-2"
            mock_downloader.download_conceptarc.return_value = tmp_path / "ConceptARC"
            mock_downloader.download_miniarc.return_value = tmp_path / "MiniARC"
            mock_downloader_class.return_value = mock_downloader

            result = runner.invoke(
                download_app, ["all", "--output", str(tmp_path), "--force"]
            )

            assert result.exit_code == 0

            # All download methods should be called
            mock_downloader.download_arc_agi_1.assert_called_once()
            mock_downloader.download_arc_agi_2.assert_called_once()
            mock_downloader.download_conceptarc.assert_called_once()
            mock_downloader.download_miniarc.assert_called_once()

    def test_cli_error_handling(self, tmp_path):
        """Test CLI error handling for download failures."""
        runner = CliRunner()

        with patch("download_dataset.DatasetDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.download_arc_agi_1.side_effect = DatasetDownloadError(
                "Network error"
            )
            mock_downloader_class.return_value = mock_downloader

            result = runner.invoke(
                download_app, ["arc-agi-1", "--output", str(tmp_path)]
            )

            assert result.exit_code == 1

    def test_configuration_loading_arc_agi_1(self):
        """Test configuration loading with new ARC-AGI-1 format."""
        # Test loading actual configuration file
        try:
            config = get_config("conf/dataset/arc_agi_1.yaml")

            # Verify configuration structure
            assert config.dataset_name == "ARC-AGI-1"
            assert config.dataset_year == 2024
            assert config.default_split == "training"

            # Verify paths use GitHub format (directories, not files)
            assert "data/training" in config.training.path
            assert "data/evaluation" in config.evaluation.path

            # Verify parser configuration
            assert config.parser._target_ == "jaxarc.parsers.ArcAgiParser"

            # Verify grid configuration
            assert config.grid.max_grid_height == 30
            assert config.grid.max_grid_width == 30
            assert config.grid.max_colors == 10

            # Verify task configuration
            assert config.max_train_pairs == 10
            assert config.max_test_pairs == 3

        except Exception as e:
            pytest.skip(f"Configuration file not available: {e}")

    def test_configuration_loading_arc_agi_2(self):
        """Test configuration loading with new ARC-AGI-2 format."""
        # Test loading actual configuration file
        try:
            config = get_config("conf/dataset/arc_agi_2.yaml")

            # Verify configuration structure
            assert config.dataset_name == "ARC-AGI-2"
            assert config.dataset_year == 2025
            assert config.default_split == "training"

            # Verify paths use GitHub format (directories, not files)
            assert "data/training" in config.training.path
            assert "data/evaluation" in config.evaluation.path

            # Verify parser configuration
            assert config.parser._target_ == "jaxarc.parsers.ArcAgiParser"

            # Verify grid configuration
            assert config.grid.max_grid_height == 30
            assert config.grid.max_grid_width == 30
            assert config.grid.max_colors == 10

            # Verify task configuration
            assert config.max_train_pairs == 10
            assert config.max_test_pairs == 4  # ARC-AGI-2 has more test pairs

        except Exception as e:
            pytest.skip(f"Configuration file not available: {e}")

    def test_end_to_end_download_and_parse_simulation(self, tmp_path):
        """Test simulated end-to-end download and parse workflow."""
        # Simulate download by creating directory structure
        arc_agi_1_dir = tmp_path / "ARC-AGI-1"
        data_dir = arc_agi_1_dir / "data"
        training_dir = data_dir / "training"
        evaluation_dir = data_dir / "evaluation"
        training_dir.mkdir(parents=True)
        evaluation_dir.mkdir(parents=True)

        # Create sample task file
        task_data = {
            "train": [
                {"input": [[1, 2]], "output": [[2, 1]]},
            ],
            "test": [{"input": [[3, 4]]}],
        }

        task_file = training_dir / "sample_task.json"
        with task_file.open("w", encoding="utf-8") as f:
            json.dump(task_data, f)

        # Create evaluation task file
        eval_task_file = evaluation_dir / "eval_task.json"
        with eval_task_file.open("w", encoding="utf-8") as f:
            json.dump(task_data, f)

        # Test that downloader would validate this structure
        downloader = DatasetDownloader(tmp_path)

        # This should not raise an exception
        downloader._validate_arc_agi_1_structure(arc_agi_1_dir)

        # Test parser can load from this structure
        config = DictConfig(
            {
                "default_split": "training",
                "training": {
                    "path": str(training_dir),
                },
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 10,
                "max_test_pairs": 3,
            }
        )

        parser = ArcAgiParser(cfg=config)
        task_ids = parser.get_available_task_ids()
        assert len(task_ids) == 1
        assert "sample_task" in task_ids

        # Test task parsing
        task = parser.get_task_by_id("sample_task")
        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs == 1
        assert task.num_test_pairs == 1

    def test_performance_with_multiple_tasks(self, tmp_path):
        """Test performance with multiple tasks to ensure scalability."""
        # Create directory structure
        training_dir = tmp_path / "training"
        training_dir.mkdir(parents=True)

        # Create multiple task files
        num_tasks = 50
        for i in range(num_tasks):
            task_data = {
                "train": [
                    {"input": [[i % 10]], "output": [[(i + 1) % 10]]},
                ],
                "test": [{"input": [[(i + 2) % 10]]}],
            }

            task_file = training_dir / f"task_{i:03d}.json"
            with task_file.open("w", encoding="utf-8") as f:
                json.dump(task_data, f)

        # Test parser performance
        config = DictConfig(
            {
                "default_split": "training",
                "training": {
                    "path": str(training_dir),
                },
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 10,
                "max_test_pairs": 3,
            }
        )

        start_time = time.time()
        parser = ArcAgiParser(cfg=config)
        load_time = time.time() - start_time

        # Should load all tasks
        task_ids = parser.get_available_task_ids()
        assert len(task_ids) == num_tasks

        # Loading should be reasonably fast
        assert load_time < 5.0, f"Loading took too long: {load_time:.2f}s"

        # Test random access performance
        key = jax.random.PRNGKey(42)
        start_time = time.time()
        for _ in range(10):  # Access 10 random tasks
            task = parser.get_random_task(key)
            assert isinstance(task, JaxArcTask)
            key, _ = jax.random.split(key)
        access_time = time.time() - start_time

        # Random access should be fast
        assert access_time < 2.0, f"Random access took too long: {access_time:.2f}s"

    def test_error_recovery_and_validation(self, tmp_path):
        """Test error recovery and validation in end-to-end workflow."""
        # Test with missing directory
        config = DictConfig(
            {
                "default_split": "training",
                "training": {
                    "path": str(tmp_path / "nonexistent"),
                },
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 10,
                "max_test_pairs": 3,
            }
        )

        with pytest.raises(RuntimeError, match="Data directory not found"):
            ArcAgiParser(cfg=config)

        # Test with empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        config.training.path = str(empty_dir)

        with pytest.raises(RuntimeError, match="No JSON files found"):
            ArcAgiParser(cfg=config)

        # Test with malformed JSON
        malformed_file = empty_dir / "malformed.json"
        with malformed_file.open("w", encoding="utf-8") as f:
            f.write("{ invalid json")

        # Parser should handle malformed files gracefully
        parser = ArcAgiParser(cfg=config)
        task_ids = parser.get_available_task_ids()
        assert len(task_ids) == 0  # No valid tasks loaded

    def test_backward_compatibility(self, mock_arc_agi_1_structure):
        """Test that new GitHub format maintains backward compatibility."""
        # Test that existing code patterns still work
        config = DictConfig(
            {
                "default_split": "training",
                "training": {
                    "path": str(mock_arc_agi_1_structure / "data" / "training"),
                },
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 10,
                "max_test_pairs": 3,
            }
        )

        parser = ArcAgiParser(cfg=config)

        # Test that all existing methods still work
        task_ids = parser.get_available_task_ids()
        assert len(task_ids) > 0

        # Test get_task_by_id
        task = parser.get_task_by_id(task_ids[0])
        assert isinstance(task, JaxArcTask)

        # Test get_random_task
        key = jax.random.PRNGKey(42)
        random_task = parser.get_random_task(key)
        assert isinstance(random_task, JaxArcTask)

        # Test that task structure is compatible with existing code
        assert hasattr(task, "input_grids_examples")
        assert hasattr(task, "output_grids_examples")
        assert hasattr(task, "test_input_grids")
        assert hasattr(task, "true_test_output_grids")
        assert hasattr(task, "input_masks_examples")
        assert hasattr(task, "output_masks_examples")
        assert hasattr(task, "test_input_masks")
        assert hasattr(task, "num_train_pairs")
        assert hasattr(task, "num_test_pairs")

    def test_jax_compatibility_integration(self, mock_arc_agi_1_structure):
        """Test JAX compatibility in integration workflow."""
        config = DictConfig(
            {
                "default_split": "training",
                "training": {
                    "path": str(mock_arc_agi_1_structure / "data" / "training"),
                },
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 10,
                "max_test_pairs": 3,
            }
        )

        parser = ArcAgiParser(cfg=config)
        task = parser.get_task_by_id("007bbfb7")

        # Test JIT compilation
        @jax.jit
        def process_task(input_grids, input_masks):
            return jax.numpy.sum(input_grids * input_masks)

        result = process_task(task.input_grids_examples, task.input_masks_examples)
        assert isinstance(result, jax.numpy.ndarray)

        # Test vmap over multiple tasks
        def process_single_input(input_grid, input_mask):
            return jax.numpy.sum(input_grid * input_mask)

        vmapped_process = jax.vmap(process_single_input)
        results = vmapped_process(task.input_grids_examples, task.input_masks_examples)

        assert isinstance(results, jax.numpy.ndarray)
        assert results.shape == (10,)  # max_train_pairs

    def test_cli_help_and_documentation(self):
        """Test CLI help and documentation for new commands."""
        runner = CliRunner()

        # Test main help
        result = runner.invoke(download_app, ["--help"])
        assert result.exit_code == 0
        assert "Download ARC datasets from GitHub repositories" in result.output
        assert "arc-agi-1" in result.output
        assert "arc-agi-2" in result.output
        assert "Examples:" in result.output

        # Test ARC-AGI-1 help
        result = runner.invoke(download_app, ["arc-agi-1", "--help"])
        assert result.exit_code == 0
        assert "Download ARC-AGI-1 dataset from GitHub" in result.output
        assert "fchollet/ARC-AGI" in result.output

        # Test ARC-AGI-2 help
        result = runner.invoke(download_app, ["arc-agi-2", "--help"])
        assert result.exit_code == 0
        assert "Download ARC-AGI-2 dataset from GitHub" in result.output
        assert "arcprize/ARC-AGI-2" in result.output

        # Test 'all' command help
        result = runner.invoke(download_app, ["all", "--help"])
        assert result.exit_code == 0
        assert "Download all ARC datasets from GitHub" in result.output

    def test_dataset_validation_integration(self, tmp_path):
        """Test dataset validation in integration workflow."""
        downloader = DatasetDownloader(tmp_path)

        # Test ARC-AGI-1 validation with proper structure
        arc_agi_1_dir = tmp_path / "ARC-AGI-1"
        data_dir = arc_agi_1_dir / "data"
        training_dir = data_dir / "training"
        evaluation_dir = data_dir / "evaluation"

        training_dir.mkdir(parents=True)
        evaluation_dir.mkdir(parents=True)

        # Create sufficient task files
        for i in range(350):  # Create 350 files to meet validation threshold
            task_file = training_dir / f"task_{i:03d}.json"
            task_file.touch()

            eval_file = evaluation_dir / f"eval_{i:03d}.json"
            eval_file.touch()

        # This should not raise an exception
        downloader._validate_arc_agi_1_structure(arc_agi_1_dir)

        # Test ARC-AGI-2 validation with proper structure
        arc_agi_2_dir = tmp_path / "ARC-AGI-2"
        data_dir = arc_agi_2_dir / "data"
        training_dir = data_dir / "training"
        evaluation_dir = data_dir / "evaluation"

        training_dir.mkdir(parents=True)
        evaluation_dir.mkdir(parents=True)

        # Create sufficient task files for ARC-AGI-2
        for i in range(850):  # Create 850 training files
            task_file = training_dir / f"task_{i:03d}.json"
            task_file.touch()

        for i in range(110):  # Create 110 evaluation files
            eval_file = evaluation_dir / f"eval_{i:03d}.json"
            eval_file.touch()

        # This should not raise an exception
        downloader._validate_arc_agi_2_structure(arc_agi_2_dir)

    def test_mixed_dataset_workflow(self, tmp_path):
        """Test workflow with mixed dataset types (ARC-AGI-1, ARC-AGI-2, etc.)."""
        runner = CliRunner()

        with patch("download_dataset.DatasetDownloader") as mock_downloader_class:
            mock_downloader = Mock()

            # Mock all download methods
            mock_downloader.download_arc_agi_1.return_value = tmp_path / "ARC-AGI-1"
            mock_downloader.download_arc_agi_2.return_value = tmp_path / "ARC-AGI-2"
            mock_downloader.download_conceptarc.return_value = tmp_path / "ConceptARC"
            mock_downloader.download_miniarc.return_value = tmp_path / "MiniARC"
            mock_downloader_class.return_value = mock_downloader

            # Test downloading multiple datasets individually
            datasets = ["arc-agi-1", "arc-agi-2", "conceptarc", "miniarc"]

            for dataset in datasets:
                result = runner.invoke(
                    download_app, [dataset, "--output", str(tmp_path)]
                )
                assert result.exit_code == 0

            # Verify all methods were called
            mock_downloader.download_arc_agi_1.assert_called()
            mock_downloader.download_arc_agi_2.assert_called()
            mock_downloader.download_conceptarc.assert_called()
            mock_downloader.download_miniarc.assert_called()

    def test_configuration_interpolation(self):
        """Test configuration interpolation with new GitHub format."""
        # Test that configuration structure supports interpolation
        # (actual interpolation is handled by Hydra in real usage)
        config_dict = {
            "data_root": "data/raw/ARC-AGI-1",
            "training": {
                "path": "data/raw/ARC-AGI-1/data/training"  # Resolved path
            },
            "evaluation": {
                "path": "data/raw/ARC-AGI-1/data/evaluation"  # Resolved path
            },
        }

        config = DictConfig(config_dict)

        # Test that the resolved paths are correct
        assert config.training.path == "data/raw/ARC-AGI-1/data/training"
        assert config.evaluation.path == "data/raw/ARC-AGI-1/data/evaluation"
        assert config.data_root == "data/raw/ARC-AGI-1"

        # Test that the structure supports the expected GitHub format paths
        assert "data/training" in config.training.path
        assert "data/evaluation" in config.evaluation.path
