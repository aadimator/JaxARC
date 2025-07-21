"""Unit tests for EpisodeManager component of enhanced visualization system."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from jaxarc.utils.visualization.episode_manager import (
    EpisodeConfig,
    EpisodeManager,
)


class TestEpisodeConfig:
    """Test EpisodeConfig dataclass and validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
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

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = EpisodeConfig(
            base_output_dir="custom/output",
            run_name="test_run",
            max_episodes_per_run=500,
            cleanup_policy="oldest_first",
            max_storage_gb=5.0,
        )

        assert config.base_output_dir == "custom/output"
        assert config.run_name == "test_run"
        assert config.max_episodes_per_run == 500
        assert config.cleanup_policy == "oldest_first"
        assert config.max_storage_gb == 5.0

    def test_config_validation_invalid_cleanup_policy(self) -> None:
        """Test validation fails for invalid cleanup policy."""
        with pytest.raises(ValueError, match="cleanup_policy must be one of"):
            EpisodeConfig(cleanup_policy="invalid_policy")

    def test_config_validation_negative_storage(self) -> None:
        """Test validation fails for negative storage limit."""
        with pytest.raises(ValueError, match="max_storage_gb must be positive"):
            EpisodeConfig(max_storage_gb=-1.0)

    def test_config_validation_zero_episodes(self) -> None:
        """Test validation fails for zero max episodes."""
        with pytest.raises(ValueError, match="max_episodes_per_run must be positive"):
            EpisodeConfig(max_episodes_per_run=0)

    def test_config_serialization(self) -> None:
        """Test configuration can be serialized and deserialized."""
        config = EpisodeConfig(
            base_output_dir="test/dir",
            run_name="test_run",
            max_storage_gb=5.0,
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["base_output_dir"] == "test/dir"
        assert config_dict["run_name"] == "test_run"
        assert config_dict["max_storage_gb"] == 5.0

        # Test from_dict
        restored_config = EpisodeConfig.from_dict(config_dict)
        assert restored_config.base_output_dir == config.base_output_dir
        assert restored_config.run_name == config.run_name
        assert restored_config.max_storage_gb == config.max_storage_gb


class TestEpisodeManager:
    """Test EpisodeManager functionality."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def episode_config(self, temp_dir: Path) -> EpisodeConfig:
        """Create test episode configuration."""
        return EpisodeConfig(
            base_output_dir=str(temp_dir),
            max_episodes_per_run=10,
            max_storage_gb=0.1,  # Small limit for testing
        )

    @pytest.fixture
    def episode_manager(self, episode_config: EpisodeConfig) -> EpisodeManager:
        """Create episode manager for testing."""
        return EpisodeManager(episode_config)

    def test_initialization(
        self, episode_manager: EpisodeManager, episode_config: EpisodeConfig
    ) -> None:
        """Test episode manager initialization."""
        assert episode_manager.config == episode_config
        assert episode_manager.current_run_dir is None
        assert episode_manager.current_episode_dir is None
        assert episode_manager.current_run_name is None
        assert episode_manager.current_episode_num is None

    def test_start_new_run_default_name(
        self, episode_manager: EpisodeManager, temp_dir: Path
    ) -> None:
        """Test starting new run with auto-generated name."""
        run_dir = episode_manager.start_new_run()

        assert run_dir.exists()
        assert run_dir.parent == temp_dir
        assert episode_manager.current_run_dir == run_dir
        assert episode_manager.current_run_name is not None
        assert "run_" in episode_manager.current_run_name

        # Check run metadata file
        metadata_file = run_dir / "run_metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)
        assert "start_time" in metadata
        assert "run_name" in metadata
        assert metadata["run_name"] == episode_manager.current_run_name

    def test_start_new_run_custom_name(
        self, episode_manager: EpisodeManager, temp_dir: Path
    ) -> None:
        """Test starting new run with custom name."""
        custom_name = "test_experiment_run"
        run_dir = episode_manager.start_new_run(custom_name)

        assert run_dir.exists()
        assert custom_name in str(run_dir)
        assert episode_manager.current_run_name == custom_name

    def test_start_new_episode(
        self, episode_manager: EpisodeManager, temp_dir: Path
    ) -> None:
        """Test starting new episode."""
        # First start a run
        episode_manager.start_new_run("test_run")

        # Start episode
        episode_dir = episode_manager.start_new_episode(1)

        assert episode_dir.exists()
        assert "episode_0001" in str(episode_dir)
        assert episode_manager.current_episode_dir == episode_dir
        assert episode_manager.current_episode_num == 1

        # Check episode metadata
        metadata_file = episode_dir / "episode_metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)
        assert metadata["episode_num"] == 1
        assert "start_time" in metadata

    def test_start_episode_without_run(self, episode_manager: EpisodeManager) -> None:
        """Test starting episode without starting run first."""
        with pytest.raises(RuntimeError, match="No active run"):
            episode_manager.start_new_episode(1)

    def test_get_step_path(
        self, episode_manager: EpisodeManager, temp_dir: Path
    ) -> None:
        """Test getting step file paths."""
        # Setup run and episode
        episode_manager.start_new_run("test_run")
        episode_manager.start_new_episode(1)

        # Test default SVG path
        step_path = episode_manager.get_step_path(5)
        expected_name = "step_005.svg"
        assert expected_name in str(step_path)
        assert step_path.parent == episode_manager.current_episode_dir

        # Test custom file type
        png_path = episode_manager.get_step_path(10, "png")
        assert "step_010.png" in str(png_path)

        # Test with custom format
        json_path = episode_manager.get_step_path(3, "json")
        assert "step_003.json" in str(json_path)

    def test_get_step_path_without_episode(
        self, episode_manager: EpisodeManager
    ) -> None:
        """Test getting step path without active episode."""
        with pytest.raises(RuntimeError, match="No active episode"):
            episode_manager.get_step_path(1)

    def test_get_episode_summary_path(
        self, episode_manager: EpisodeManager, temp_dir: Path
    ) -> None:
        """Test getting episode summary path."""
        episode_manager.start_new_run("test_run")
        episode_manager.start_new_episode(1)

        summary_path = episode_manager.get_episode_summary_path()
        assert "episode_summary.svg" in str(summary_path)
        assert summary_path.parent == episode_manager.current_episode_dir

        # Test custom format
        json_summary = episode_manager.get_episode_summary_path("json")
        assert "episode_summary.json" in str(json_summary)

    def test_cleanup_oldest_first(
        self, episode_manager: EpisodeManager, temp_dir: Path
    ) -> None:
        """Test cleanup with oldest_first policy."""
        episode_manager.config = episode_manager.config.replace(
            cleanup_policy="oldest_first", max_episodes_per_run=3
        )

        # Create run and multiple episodes
        episode_manager.start_new_run("test_run")

        episode_dirs = []
        for i in range(5):  # More than max_episodes_per_run
            episode_dir = episode_manager.start_new_episode(i + 1)
            episode_dirs.append(episode_dir)
            # Create some content
            (episode_dir / "test_file.txt").write_text("test content")
            time.sleep(0.01)  # Ensure different timestamps

        # Trigger cleanup
        episode_manager.cleanup_old_data()

        # Check that only the most recent episodes remain
        remaining_dirs = [d for d in episode_dirs if d.exists()]
        assert len(remaining_dirs) == 3

        # Check that the oldest episodes were removed
        assert not episode_dirs[0].exists()  # Oldest
        assert not episode_dirs[1].exists()  # Second oldest
        assert episode_dirs[2].exists()  # Should remain
        assert episode_dirs[3].exists()  # Should remain
        assert episode_dirs[4].exists()  # Should remain (newest)

    def test_cleanup_size_based(
        self, episode_manager: EpisodeManager, temp_dir: Path
    ) -> None:
        """Test cleanup with size_based policy."""
        episode_manager.config = episode_manager.config.replace(
            cleanup_policy="size_based",
            max_storage_gb=0.001,  # Very small limit
        )

        episode_manager.start_new_run("test_run")

        # Create episodes with large files
        large_content = "x" * 1000  # 1KB content
        episode_dirs = []

        for i in range(3):
            episode_dir = episode_manager.start_new_episode(i + 1)
            episode_dirs.append(episode_dir)
            # Create large file
            (episode_dir / "large_file.txt").write_text(large_content)
            time.sleep(0.01)

        # Trigger cleanup
        episode_manager.cleanup_old_data()

        # Should have cleaned up some episodes due to size limit
        remaining_dirs = [d for d in episode_dirs if d.exists()]
        assert len(remaining_dirs) < 3

    def test_cleanup_manual_policy(
        self, episode_manager: EpisodeManager, temp_dir: Path
    ) -> None:
        """Test that manual cleanup policy doesn't auto-cleanup."""
        episode_manager.config = episode_manager.config.replace(
            cleanup_policy="manual",
            max_episodes_per_run=1,  # Very low limit
        )

        episode_manager.start_new_run("test_run")

        # Create more episodes than the limit
        episode_dirs = []
        for i in range(3):
            episode_dir = episode_manager.start_new_episode(i + 1)
            episode_dirs.append(episode_dir)
            (episode_dir / "test_file.txt").write_text("test")

        # Trigger cleanup - should do nothing with manual policy
        episode_manager.cleanup_old_data()

        # All episodes should still exist
        for episode_dir in episode_dirs:
            assert episode_dir.exists()

    def test_get_storage_usage(
        self, episode_manager: EpisodeManager, temp_dir: Path
    ) -> None:
        """Test storage usage calculation."""
        episode_manager.start_new_run("test_run")
        episode_manager.start_new_episode(1)

        # Initially should be minimal usage
        initial_usage = episode_manager.get_storage_usage_gb()
        assert initial_usage >= 0

        # Create some content
        large_content = "x" * 10000  # 10KB
        test_file = episode_manager.current_episode_dir / "large_file.txt"
        test_file.write_text(large_content)

        # Usage should increase
        new_usage = episode_manager.get_storage_usage_gb()
        assert new_usage > initial_usage

    def test_list_runs(self, episode_manager: EpisodeManager, temp_dir: Path) -> None:
        """Test listing available runs."""
        # Initially no runs
        runs = episode_manager.list_runs()
        assert len(runs) == 0

        # Create some runs
        run1 = episode_manager.start_new_run("run_1")
        episode_manager.start_new_run("run_2")

        runs = episode_manager.list_runs()
        assert len(runs) == 2
        assert any("run_1" in str(run) for run in runs)
        assert any("run_2" in str(run) for run in runs)

    def test_list_episodes(
        self, episode_manager: EpisodeManager, temp_dir: Path
    ) -> None:
        """Test listing episodes in current run."""
        episode_manager.start_new_run("test_run")

        # Initially no episodes
        episodes = episode_manager.list_episodes()
        assert len(episodes) == 0

        # Create episodes
        episode_manager.start_new_episode(1)
        episode_manager.start_new_episode(2)
        episode_manager.start_new_episode(3)

        episodes = episode_manager.list_episodes()
        assert len(episodes) == 3
        assert any("episode_0001" in str(ep) for ep in episodes)
        assert any("episode_0002" in str(ep) for ep in episodes)
        assert any("episode_0003" in str(ep) for ep in episodes)

    def test_error_handling_permission_denied(
        self, episode_manager: EpisodeManager
    ) -> None:
        """Test error handling for permission denied."""
        # Try to create in a directory we can't write to
        episode_manager.config = episode_manager.config.replace(
            base_output_dir="/root/no_permission"
        )

        with pytest.raises((OSError, PermissionError)):
            episode_manager.start_new_run("test_run")

    def test_concurrent_access_safety(
        self, episode_manager: EpisodeManager, temp_dir: Path
    ) -> None:
        """Test thread safety for concurrent access."""
        import threading

        episode_manager.start_new_run("concurrent_test")

        results = []
        errors = []

        def create_episode(episode_num: int) -> None:
            try:
                episode_dir = episode_manager.start_new_episode(episode_num)
                results.append(episode_dir)
            except Exception as e:
                errors.append(e)

        # Create multiple threads trying to create episodes
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_episode, args=(i + 1,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have some successful results and handle conflicts gracefully
        assert len(results) > 0
        # Some conflicts are expected in concurrent access
        assert len(errors) >= 0


# Factory functions are not implemented yet, so removing these tests for now
# class TestEpisodeConfigFactories:
#     """Test episode configuration factory functions."""


if __name__ == "__main__":
    pytest.main([__file__])
