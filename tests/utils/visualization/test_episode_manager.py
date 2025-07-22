"""Tests for episode management functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from jaxarc.utils.visualization.episode_manager import (
    EpisodeConfig,
    EpisodeManager,
)


class TestEpisodeConfig:
    """Test EpisodeConfig dataclass."""

    def test_default_config(self):
        """Test default episode configuration."""
        config = EpisodeConfig()

        assert config.base_output_dir == "outputs/episodes"
        assert config.run_name is None
        assert config.episode_dir_format == "episode_{episode:04d}"
        assert config.step_file_format == "step_{step:03d}"
        assert config.max_episodes_per_run == 1000
        assert config.cleanup_policy == "size_based"
        assert config.max_storage_gb == 10.0
        assert config.create_run_subdirs is True
        assert config.preserve_empty_dirs is False
        assert config.compress_old_episodes is False

    def test_custom_config(self):
        """Test custom episode configuration."""
        config = EpisodeConfig(
            base_output_dir="custom/output",
            run_name="test_run",
            episode_dir_format="ep_{episode:02d}",
            step_file_format="s_{step:02d}",
            max_episodes_per_run=50,
            cleanup_policy="oldest_first",
            max_storage_gb=5.0,
            create_run_subdirs=False,
            preserve_empty_dirs=True,
            compress_old_episodes=True,
        )

        assert config.base_output_dir == "custom/output"
        assert config.run_name == "test_run"
        assert config.episode_dir_format == "ep_{episode:02d}"
        assert config.step_file_format == "s_{step:02d}"
        assert config.max_episodes_per_run == 50
        assert config.cleanup_policy == "oldest_first"
        assert config.max_storage_gb == 5.0
        assert config.create_run_subdirs is False
        assert config.preserve_empty_dirs is True
        assert config.compress_old_episodes is True

    def test_get_base_path(self):
        """Test getting base path."""
        config = EpisodeConfig(base_output_dir="test/output")

        base_path = config.get_base_path()

        assert isinstance(base_path, Path)
        assert base_path.name == "output"
        assert "test" in str(base_path)

    def test_config_validation_episode_format(self):
        """Test episode directory format validation."""
        # Valid format
        config = EpisodeConfig(episode_dir_format="ep_{episode:02d}")
        assert config.episode_dir_format == "ep_{episode:02d}"

        # Invalid format - missing required field
        with pytest.raises(ValueError, match="Invalid episode_dir_format"):
            EpisodeConfig(episode_dir_format="ep_{invalid:02d}")

    def test_config_validation_step_format(self):
        """Test step file format validation."""
        # Valid format
        config = EpisodeConfig(step_file_format="s_{step:02d}")
        assert config.step_file_format == "s_{step:02d}"

        # Invalid format
        with pytest.raises(ValueError, match="Invalid step_file_format"):
            EpisodeConfig(step_file_format="s_{invalid:02d}")

    def test_config_validation_numeric_limits(self):
        """Test numeric limit validation."""
        # Invalid max_episodes_per_run
        with pytest.raises(ValueError, match="max_episodes_per_run must be positive"):
            EpisodeConfig(max_episodes_per_run=0)

        # Invalid max_storage_gb
        with pytest.raises(ValueError, match="max_storage_gb must be positive"):
            EpisodeConfig(max_storage_gb=-1.0)

    def test_config_validation_cleanup_policy(self):
        """Test cleanup policy validation."""
        # Valid policies
        for policy in ["oldest_first", "size_based", "manual"]:
            config = EpisodeConfig(cleanup_policy=policy)
            assert config.cleanup_policy == policy

        # Invalid policy
        with pytest.raises(ValueError, match="cleanup_policy must be one of"):
            EpisodeConfig(cleanup_policy="invalid")

    def test_config_validation_run_name(self):
        """Test run name validation."""
        # Valid run name
        config = EpisodeConfig(run_name="valid_run")
        assert config.run_name == "valid_run"

        # Empty run name
        with pytest.raises(ValueError, match="run_name must be a non-empty string"):
            EpisodeConfig(run_name="")

        # Run name with invalid characters
        with pytest.raises(ValueError, match="run_name contains invalid characters"):
            EpisodeConfig(run_name="run<name>")

    def test_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = EpisodeConfig(base_output_dir="test/output", run_name="test_run")

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["base_output_dir"] == "test/output"
        assert config_dict["run_name"] == "test_run"
        assert "episode_dir_format" in config_dict
        assert "step_file_format" in config_dict

    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "base_output_dir": "test/output",
            "run_name": "test_run",
            "max_episodes_per_run": 500,
            "unknown_field": "ignored",  # Should be ignored
        }

        config = EpisodeConfig.from_dict(config_dict)

        assert config.base_output_dir == "test/output"
        assert config.run_name == "test_run"
        assert config.max_episodes_per_run == 500
        # Unknown field should be ignored

    def test_from_dict_invalid(self):
        """Test configuration creation from invalid dictionary."""
        invalid_dict = {
            "max_episodes_per_run": -1  # Invalid value
        }

        with pytest.raises(ValueError, match="max_episodes_per_run must be positive"):
            EpisodeConfig.from_dict(invalid_dict)

    def test_save_and_load_config(self):
        """Test saving and loading configuration to/from file."""
        config = EpisodeConfig(
            base_output_dir="test/output", run_name="test_run", max_episodes_per_run=500
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            # Save config
            config.save_to_file(config_path)
            assert config_path.exists()

            # Load config
            loaded_config = EpisodeConfig.load_from_file(config_path)

            assert loaded_config.base_output_dir == config.base_output_dir
            assert loaded_config.run_name == config.run_name
            assert loaded_config.max_episodes_per_run == config.max_episodes_per_run

    def test_load_config_nonexistent_file(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            EpisodeConfig.load_from_file("nonexistent.json")


class TestEpisodeManager:
    """Test EpisodeManager class."""

    def test_episode_manager_initialization(self):
        """Test episode manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)

            manager = EpisodeManager(config)

            assert manager.config == config
            assert manager.current_run_dir is None
            assert manager.current_episode_dir is None
            assert manager.current_run_name is None
            assert manager.current_episode_num is None

    def test_episode_manager_initialization_invalid_path(self):
        """Test initialization with invalid base path."""
        # Use a path that doesn't exist and can't be created
        config = EpisodeConfig(base_output_dir="/invalid/path/that/cannot/be/created")

        with pytest.raises(
            ValueError, match="Cannot access or write to base directory"
        ):
            EpisodeManager(config)

    def test_start_new_run(self):
        """Test starting a new run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)
            manager = EpisodeManager(config)

            run_dir = manager.start_new_run("test_run")

            assert manager.current_run_name == "test_run"
            assert manager.current_run_dir == run_dir
            assert run_dir.exists()
            assert run_dir.name == "test_run"

            # Should create config file
            config_file = run_dir / "episode_config.json"
            assert config_file.exists()

    def test_start_new_run_auto_generated_name(self):
        """Test starting run with auto-generated name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)
            manager = EpisodeManager(config)

            run_dir = manager.start_new_run()

            assert manager.current_run_name is not None
            assert manager.current_run_name.startswith("run_")
            assert run_dir.exists()

    def test_start_new_run_invalid_name(self):
        """Test starting run with invalid name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)
            manager = EpisodeManager(config)

            with pytest.raises(ValueError, match="run_name must be a non-empty string"):
                manager.start_new_run("")

    def test_start_new_episode(self):
        """Test starting a new episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)
            manager = EpisodeManager(config)

            # Start run first
            manager.start_new_run("test_run")

            episode_dir = manager.start_new_episode(1)

            assert manager.current_episode_num == 1
            assert manager.current_episode_dir == episode_dir
            assert episode_dir.exists()
            assert episode_dir.name == "episode_0001"

    def test_start_new_episode_no_active_run(self):
        """Test starting episode without active run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)
            manager = EpisodeManager(config)

            with pytest.raises(ValueError, match="No active run"):
                manager.start_new_episode(1)

    def test_start_new_episode_invalid_number(self):
        """Test starting episode with invalid episode number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)
            manager = EpisodeManager(config)

            manager.start_new_run("test_run")

            # Negative episode number
            with pytest.raises(ValueError, match="episode_num must be non-negative"):
                manager.start_new_episode(-1)

            # Episode number exceeds limit
            with pytest.raises(ValueError, match="exceeds max_episodes_per_run"):
                manager.start_new_episode(2000)  # Default limit is 1000

    def test_get_step_path(self):
        """Test getting step file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)
            manager = EpisodeManager(config)

            # Start run and episode first
            manager.start_new_run("test_run")
            manager.start_new_episode(1)

            step_path = manager.get_step_path(5, "svg")

            # Check the path components rather than exact path due to symlink resolution
            assert step_path.name == "step_005.svg"
            assert step_path.parent.name == "episode_0001"
            assert step_path.parent.parent.name == "test_run"

    def test_get_step_path_no_active_episode(self):
        """Test getting step path without active episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)
            manager = EpisodeManager(config)

            with pytest.raises(ValueError, match="No active episode"):
                manager.get_step_path(5, "svg")

    def test_get_step_path_invalid_step_number(self):
        """Test getting step path with invalid step number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)
            manager = EpisodeManager(config)

            manager.start_new_run("test_run")
            manager.start_new_episode(1)

            with pytest.raises(ValueError, match="step_num must be non-negative"):
                manager.get_step_path(-1, "svg")

    def test_get_episode_summary_path(self):
        """Test getting episode summary path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)
            manager = EpisodeManager(config)

            # Start run and episode first
            manager.start_new_run("test_run")
            manager.start_new_episode(1)

            summary_path = manager.get_episode_summary_path("html")

            # Check the path components rather than exact path due to symlink resolution
            assert summary_path.name == "summary.html"
            assert summary_path.parent.name == "episode_0001"
            assert summary_path.parent.parent.name == "test_run"

    def test_get_episode_summary_path_no_active_episode(self):
        """Test getting summary path without active episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)
            manager = EpisodeManager(config)

            with pytest.raises(ValueError, match="No active episode"):
                manager.get_episode_summary_path("svg")

    def test_config_validation_methods(self):
        """Test config validation helper methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)

            # Test generate_run_name
            run_name = config.generate_run_name()
            assert isinstance(run_name, str)
            assert run_name.startswith("run_")

            # Test with custom run_name
            config_with_name = EpisodeConfig(base_output_dir=tmpdir, run_name="custom")
            assert config_with_name.generate_run_name() == "custom"

            # Test validate_storage_path
            valid_path = Path(tmpdir) / "valid"
            assert config.validate_storage_path(valid_path) is True

            # Test estimate_storage_usage
            usage = config.estimate_storage_usage(Path(tmpdir))
            assert isinstance(usage, float)
            assert usage >= 0.0


class TestEpisodeManagerIntegration:
    """Test episode manager integration scenarios."""

    def test_full_episode_workflow(self):
        """Test complete episode workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)
            manager = EpisodeManager(config)

            # Start run and episode
            run_dir = manager.start_new_run("test_run")
            episode_dir = manager.start_new_episode(1)

            assert run_dir.exists()
            assert episode_dir.exists()

            # Get step paths and create files
            step1_path = manager.get_step_path(1, "svg")
            step2_path = manager.get_step_path(2, "svg")
            summary_path = manager.get_episode_summary_path("svg")

            step1_path.touch()
            step2_path.touch()
            summary_path.touch()

            assert step1_path.exists()
            assert step2_path.exists()
            assert summary_path.exists()

    def test_multiple_episodes_in_run(self):
        """Test multiple episodes within a single run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(base_output_dir=tmpdir)
            manager = EpisodeManager(config)

            # Start run
            manager.start_new_run("multi_episode_run")

            # Create multiple episodes
            for i in range(1, 4):
                episode_dir = manager.start_new_episode(i)
                assert episode_dir.exists()
                assert episode_dir.name == f"episode_{i:04d}"

                # Create a step file in each episode
                step_path = manager.get_step_path(1, "svg")
                step_path.touch()
                assert step_path.exists()

    def test_run_configuration_persistence(self):
        """Test that run configuration is saved and can be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EpisodeConfig(
                base_output_dir=tmpdir,
                max_episodes_per_run=500,
                cleanup_policy="oldest_first",
            )
            manager = EpisodeManager(config)

            # Start run (should save config)
            run_dir = manager.start_new_run("config_test")

            # Check that config file was created
            config_file = run_dir / "episode_config.json"
            assert config_file.exists()

            # Load and verify config
            loaded_config = EpisodeConfig.load_from_file(config_file)
            assert loaded_config.max_episodes_per_run == 500
            assert loaded_config.cleanup_policy == "oldest_first"


if __name__ == "__main__":
    pytest.main([__file__])
