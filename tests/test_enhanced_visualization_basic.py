"""Basic tests for enhanced visualization system components."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from jaxarc.utils.visualization.episode_manager import EpisodeConfig, EpisodeManager


class TestBasicEpisodeManager:
    """Basic tests for episode manager functionality."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_episode_config_creation(self) -> None:
        """Test basic episode config creation."""
        config = EpisodeConfig()

        assert config.base_output_dir == "outputs/episodes"
        assert config.max_episodes_per_run == 1000
        assert config.cleanup_policy == "size_based"
        assert config.max_storage_gb == 10.0

    def test_episode_manager_creation(self, temp_dir: Path) -> None:
        """Test basic episode manager creation."""
        config = EpisodeConfig(base_output_dir=str(temp_dir))
        manager = EpisodeManager(config)

        assert manager.config == config
        assert manager.current_run_dir is None
        assert manager.current_episode_dir is None


if __name__ == "__main__":
    pytest.main([__file__])
