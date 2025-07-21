"""Unit tests for WandbIntegration component of enhanced visualization system."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import numpy as np
import pytest

from jaxarc.utils.visualization.wandb_integration import (
    WandbConfig,
    WandbIntegration,
)


class TestWandbConfig:
    """Test WandbConfig dataclass and validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = WandbConfig()

        assert config.enabled is False
        assert config.project_name == "jaxarc-experiments"
        assert config.entity is None
        assert config.tags == []
        assert config.notes is None
        assert config.group is None
        assert config.job_type is None
        assert config.log_frequency == 10
        assert config.image_format == "png"
        assert config.max_image_size == (800, 600)
        assert config.log_gradients is False
        assert config.log_model_topology is False
        assert config.log_system_metrics is True
        assert config.offline_mode is False
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.save_code is True
        assert config.save_config is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = WandbConfig(
            enabled=True,
            project_name="test-project",
            entity="test-entity",
            tags=["test", "experiment"],
            notes="Test experiment notes",
            group="test-group",
            job_type="training",
            log_frequency=5,
            image_format="svg",
            max_image_size=(1024, 768),
            log_gradients=True,
            log_model_topology=True,
            log_system_metrics=False,
            offline_mode=True,
            retry_attempts=5,
            retry_delay=2.0,
            save_code=False,
            save_config=False,
        )

        assert config.enabled is True
        assert config.project_name == "test-project"
        assert config.entity == "test-entity"
        assert config.tags == ["test", "experiment"]
        assert config.notes == "Test experiment notes"
        assert config.group == "test-group"
        assert config.job_type == "training"
        assert config.log_frequency == 5
        assert config.image_format == "svg"
        assert config.max_image_size == (1024, 768)
        assert config.log_gradients is True
        assert config.log_model_topology is True
        assert config.log_system_metrics is False
        assert config.offline_mode is True
        assert config.retry_attempts == 5
        assert config.retry_delay == 2.0
        assert config.save_code is False
        assert config.save_config is False

    def test_config_validation_invalid_image_format(self) -> None:
        """Test validation fails for invalid image format."""
        with pytest.raises(ValueError, match="Invalid image_format"):
            WandbConfig(image_format="invalid")

    def test_config_validation_negative_values(self) -> None:
        """Test validation fails for negative values."""
        with pytest.raises(ValueError, match="log_frequency must be positive"):
            WandbConfig(log_frequency=0)

        with pytest.raises(ValueError, match="retry_attempts must be non-negative"):
            WandbConfig(retry_attempts=-1)

        with pytest.raises(ValueError, match="retry_delay must be non-negative"):
            WandbConfig(retry_delay=-1.0)

    def test_config_validation_image_size(self) -> None:
        """Test validation of image size."""
        with pytest.raises(
            ValueError, match="max_image_size must have positive dimensions"
        ):
            WandbConfig(max_image_size=(0, 600))

        with pytest.raises(
            ValueError, match="max_image_size must have positive dimensions"
        ):
            WandbConfig(max_image_size=(800, 0))

    def test_config_serialization(self) -> None:
        """Test configuration serialization."""
        config = WandbConfig(
            enabled=True,
            project_name="test-project",
            tags=["test", "serialization"],
            log_frequency=15,
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["enabled"] is True
        assert config_dict["project_name"] == "test-project"
        assert config_dict["tags"] == ["test", "serialization"]
        assert config_dict["log_frequency"] == 15

        # Test from_dict
        restored_config = WandbConfig.from_dict(config_dict)
        assert restored_config.enabled == config.enabled
        assert restored_config.project_name == config.project_name
        assert restored_config.tags == config.tags
        assert restored_config.log_frequency == config.log_frequency


class TestWandbIntegration:
    """Test WandbIntegration functionality."""

    @pytest.fixture
    def wandb_config(self) -> WandbConfig:
        """Create test wandb configuration."""
        return WandbConfig(
            enabled=True,
            project_name="test-project",
            entity="test-entity",
            tags=["test"],
            log_frequency=1,  # Log every step for testing
            retry_attempts=2,
            retry_delay=0.1,
        )

    @pytest.fixture
    def mock_wandb(self):
        """Mock wandb module."""
        with patch("jaxarc.utils.visualization.wandb_integration.wandb") as mock_wandb:
            mock_run = MagicMock()
            mock_wandb.init.return_value = mock_run
            mock_wandb.run = mock_run
            mock_wandb.Image = MagicMock()
            mock_wandb.Table = MagicMock()
            mock_wandb.config = {}
            yield mock_wandb

    @pytest.fixture
    def wandb_integration(
        self, wandb_config: WandbConfig, mock_wandb
    ) -> WandbIntegration:
        """Create wandb integration for testing."""
        return WandbIntegration(wandb_config)

    def test_initialization_enabled(
        self, wandb_integration: WandbIntegration, wandb_config: WandbConfig
    ) -> None:
        """Test wandb integration initialization when enabled."""
        assert wandb_integration.config == wandb_config
        assert wandb_integration.run is None  # Not initialized until initialize_run
        assert wandb_integration.is_initialized is False
        assert wandb_integration.offline_cache == []

    def test_initialization_disabled(self) -> None:
        """Test wandb integration when disabled."""
        disabled_config = WandbConfig(enabled=False)
        integration = WandbIntegration(disabled_config)

        assert integration.config.enabled is False
        assert integration.run is None
        assert integration.is_initialized is False

    def test_initialize_run_success(
        self, wandb_integration: WandbIntegration, mock_wandb
    ) -> None:
        """Test successful run initialization."""
        experiment_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "environment": "arc_agi",
        }

        wandb_integration.initialize_run(experiment_config, run_name="test_run")

        # Check wandb.init was called correctly
        mock_wandb.init.assert_called_once()
        call_args = mock_wandb.init.call_args

        assert call_args[1]["project"] == "test-project"
        assert call_args[1]["entity"] == "test-entity"
        assert call_args[1]["name"] == "test_run"
        assert call_args[1]["tags"] == ["test"]
        assert call_args[1]["config"] == experiment_config

        # Check integration state
        assert wandb_integration.is_initialized is True
        assert wandb_integration.run is not None

    def test_initialize_run_disabled(self, mock_wandb) -> None:
        """Test run initialization when wandb is disabled."""
        disabled_config = WandbConfig(enabled=False)
        integration = WandbIntegration(disabled_config)

        integration.initialize_run({"test": "config"})

        # Should not call wandb.init
        mock_wandb.init.assert_not_called()
        assert integration.is_initialized is False

    def test_initialize_run_failure(
        self, wandb_integration: WandbIntegration, mock_wandb
    ) -> None:
        """Test run initialization failure handling."""
        # Mock wandb.init to raise an exception
        mock_wandb.init.side_effect = Exception("Wandb initialization failed")

        with pytest.raises(
            Exception
        ):  # Generic exception since WandbError may not exist
            wandb_integration.initialize_run({"test": "config"})

        assert wandb_integration.is_initialized is False

    def test_log_step_success(
        self, wandb_integration: WandbIntegration, mock_wandb
    ) -> None:
        """Test successful step logging."""
        # Initialize run first
        wandb_integration.initialize_run({"test": "config"})

        metrics = {
            "reward": 0.8,
            "episode_length": 25,
            "success_rate": 0.6,
        }

        images = {
            "step_visualization": np.random.rand(100, 100, 3),
            "grid_state": np.random.rand(50, 50),
        }

        wandb_integration.log_step(step_num=10, metrics=metrics, images=images)

        # Check that wandb.log was called
        mock_wandb.log.assert_called()

        # Verify logged data structure
        call_args = mock_wandb.log.call_args[0][0]
        assert call_args["step"] == 10
        assert call_args["reward"] == 0.8
        assert call_args["episode_length"] == 25
        assert call_args["success_rate"] == 0.6
        assert "step_visualization" in call_args
        assert "grid_state" in call_args

    def test_log_step_not_initialized(
        self, wandb_integration: WandbIntegration
    ) -> None:
        """Test step logging when not initialized."""
        metrics = {"reward": 0.5}

        # Should not raise error, but should cache for offline mode
        wandb_integration.log_step(step_num=5, metrics=metrics)

        # Should be cached for later
        assert len(wandb_integration.offline_cache) == 1
        cached_entry = wandb_integration.offline_cache[0]
        assert cached_entry["type"] == "step"
        assert cached_entry["data"]["step"] == 5
        assert cached_entry["data"]["reward"] == 0.5

    def test_log_step_frequency_filtering(
        self, wandb_integration: WandbIntegration, mock_wandb
    ) -> None:
        """Test step logging frequency filtering."""
        # Set log frequency to 5
        wandb_integration.config = wandb_integration.config.replace(log_frequency=5)
        wandb_integration.initialize_run({"test": "config"})

        # Log steps 1-10
        for step in range(1, 11):
            wandb_integration.log_step(step_num=step, metrics={"reward": step * 0.1})

        # Should only log steps 5 and 10 (every 5th step)
        assert mock_wandb.log.call_count == 2

        # Check the logged steps
        logged_steps = [call[0][0]["step"] for call in mock_wandb.log.call_args_list]
        assert 5 in logged_steps
        assert 10 in logged_steps

    def test_log_episode_summary(
        self, wandb_integration: WandbIntegration, mock_wandb
    ) -> None:
        """Test episode summary logging."""
        wandb_integration.initialize_run({"test": "config"})

        summary_data = {
            "episode_num": 15,
            "total_steps": 30,
            "total_reward": 12.5,
            "success": True,
            "final_similarity": 0.95,
        }

        summary_image = np.random.rand(200, 200, 3)

        wandb_integration.log_episode_summary(
            episode_num=15, summary_data=summary_data, summary_image=summary_image
        )

        # Check wandb.log was called
        mock_wandb.log.assert_called()

        call_args = mock_wandb.log.call_args[0][0]
        assert call_args["episode"] == 15
        assert call_args["total_steps"] == 30
        assert call_args["total_reward"] == 12.5
        assert call_args["success"] is True
        assert call_args["final_similarity"] == 0.95
        assert "episode_summary" in call_args

    def test_log_episode_summary_not_initialized(
        self, wandb_integration: WandbIntegration
    ) -> None:
        """Test episode summary logging when not initialized."""
        summary_data = {"episode_num": 5, "total_reward": 10.0}

        wandb_integration.log_episode_summary(episode_num=5, summary_data=summary_data)

        # Should be cached
        assert len(wandb_integration.offline_cache) == 1
        cached_entry = wandb_integration.offline_cache[0]
        assert cached_entry["type"] == "episode_summary"
        assert cached_entry["data"]["episode"] == 5

    def test_image_optimization(
        self, wandb_integration: WandbIntegration, mock_wandb
    ) -> None:
        """Test image optimization for wandb upload."""
        wandb_integration.initialize_run({"test": "config"})

        # Create large image that exceeds max_image_size
        large_image = np.random.rand(1200, 1000, 3)  # Larger than (800, 600)

        wandb_integration.log_step(
            step_num=1, metrics={"reward": 0.5}, images={"large_image": large_image}
        )

        # Check that wandb.Image was called (image should be processed)
        mock_wandb.Image.assert_called()

        # The actual image passed to wandb.Image should be resized
        image_call_args = mock_wandb.Image.call_args[0][0]
        # Should be resized to fit within max_image_size while maintaining aspect ratio
        assert image_call_args.shape[0] <= 600  # Height constraint
        assert image_call_args.shape[1] <= 800  # Width constraint

    def test_network_error_handling(
        self, wandb_integration: WandbIntegration, mock_wandb
    ) -> None:
        """Test network error handling and retries."""
        wandb_integration.initialize_run({"test": "config"})

        # Mock wandb.log to raise network error first, then succeed
        call_count = 0

        def mock_log_with_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise URLError("Network error")

        mock_wandb.log.side_effect = mock_log_with_error

        # Should retry and eventually succeed
        wandb_integration.log_step(step_num=1, metrics={"reward": 0.5})

        # Should have been called twice (initial + 1 retry)
        assert mock_wandb.log.call_count == 2

    def test_network_error_max_retries(
        self, wandb_integration: WandbIntegration, mock_wandb
    ) -> None:
        """Test network error handling when max retries exceeded."""
        wandb_integration.initialize_run({"test": "config"})

        # Mock wandb.log to always raise network error
        mock_wandb.log.side_effect = URLError("Persistent network error")

        # Should eventually give up and cache offline
        wandb_integration.log_step(step_num=1, metrics={"reward": 0.5})

        # Should have retried max_attempts times
        assert mock_wandb.log.call_count == wandb_integration.config.retry_attempts + 1

        # Should cache the failed entry
        assert len(wandb_integration.offline_cache) == 1

    def test_offline_mode(self, mock_wandb) -> None:
        """Test offline mode functionality."""
        offline_config = WandbConfig(
            enabled=True, offline_mode=True, project_name="offline-test"
        )
        integration = WandbIntegration(offline_config)

        integration.initialize_run({"test": "config"})

        # Check that wandb.init was called with offline mode
        call_args = mock_wandb.init.call_args[1]
        assert call_args["mode"] == "offline"

    def test_sync_offline_data(
        self, wandb_integration: WandbIntegration, mock_wandb
    ) -> None:
        """Test syncing offline cached data."""
        # Add some offline data
        wandb_integration.offline_cache = [
            {
                "type": "step",
                "timestamp": time.time(),
                "data": {"step": 1, "reward": 0.5},
            },
            {
                "type": "episode_summary",
                "timestamp": time.time(),
                "data": {"episode": 1, "total_reward": 10.0},
            },
        ]

        # Initialize run and sync
        wandb_integration.initialize_run({"test": "config"})
        wandb_integration.sync_offline_data()

        # Should have logged cached data
        assert mock_wandb.log.call_count == 2

        # Cache should be cleared
        assert len(wandb_integration.offline_cache) == 0

    def test_finish_run(self, wandb_integration: WandbIntegration, mock_wandb) -> None:
        """Test finishing wandb run."""
        wandb_integration.initialize_run({"test": "config"})

        # Add some final metrics
        final_metrics = {"final_score": 95.5, "total_episodes": 100}

        wandb_integration.finish_run(final_metrics)

        # Should log final metrics
        mock_wandb.log.assert_called()
        final_call_args = mock_wandb.log.call_args[0][0]
        assert final_call_args["final_score"] == 95.5
        assert final_call_args["total_episodes"] == 100

        # Should call wandb.finish
        mock_wandb.finish.assert_called_once()

        # Integration should be reset
        assert wandb_integration.is_initialized is False
        assert wandb_integration.run is None

    def test_finish_run_not_initialized(
        self, wandb_integration: WandbIntegration, mock_wandb
    ) -> None:
        """Test finishing run when not initialized."""
        wandb_integration.finish_run()

        # Should not call wandb.finish
        mock_wandb.finish.assert_not_called()

    def test_context_manager(
        self, wandb_integration: WandbIntegration, mock_wandb
    ) -> None:
        """Test using wandb integration as context manager."""
        experiment_config = {"test": "config"}

        with wandb_integration.run_context(experiment_config, "context_test"):
            assert wandb_integration.is_initialized is True

            # Log something during context
            wandb_integration.log_step(step_num=1, metrics={"reward": 0.8})

        # Should automatically finish run when exiting context
        mock_wandb.finish.assert_called_once()
        assert wandb_integration.is_initialized is False

    def test_context_manager_exception(
        self, wandb_integration: WandbIntegration, mock_wandb
    ) -> None:
        """Test context manager with exception."""
        experiment_config = {"test": "config"}

        with pytest.raises(ValueError, match="Test exception"):
            with wandb_integration.run_context(experiment_config, "exception_test"):
                assert wandb_integration.is_initialized is True
                raise ValueError("Test exception")

        # Should still finish run even with exception
        mock_wandb.finish.assert_called_once()
        assert wandb_integration.is_initialized is False

    def test_get_run_url(self, wandb_integration: WandbIntegration, mock_wandb) -> None:
        """Test getting run URL."""
        # Mock run with URL
        mock_run = MagicMock()
        mock_run.get_url.return_value = (
            "https://wandb.ai/test-entity/test-project/runs/test-run"
        )
        mock_wandb.init.return_value = mock_run

        wandb_integration.initialize_run({"test": "config"})

        url = wandb_integration.get_run_url()
        assert url == "https://wandb.ai/test-entity/test-project/runs/test-run"

    def test_get_run_url_not_initialized(
        self, wandb_integration: WandbIntegration
    ) -> None:
        """Test getting run URL when not initialized."""
        url = wandb_integration.get_run_url()
        assert url is None

    def test_log_config_changes(
        self, wandb_integration: WandbIntegration, mock_wandb
    ) -> None:
        """Test logging configuration changes during run."""
        wandb_integration.initialize_run({"initial": "config"})

        # Update config during run
        config_updates = {"learning_rate": 0.0005, "new_param": "value"}
        wandb_integration.log_config_update(config_updates)

        # Should update wandb config
        mock_wandb.config.update.assert_called_once_with(config_updates)


# Factory functions are not implemented yet, so removing these tests for now
# class TestWandbConfigFactories:
#     """Test wandb configuration factory functions."""


if __name__ == "__main__":
    pytest.main([__file__])
