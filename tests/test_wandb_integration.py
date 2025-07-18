"""Tests for Weights & Biases integration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jaxarc.utils.visualization.wandb_integration import (
    WandbConfig,
    WandbIntegration,
    create_development_wandb_config,
    create_research_wandb_config,
    create_wandb_config,
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
            log_frequency=5,
            image_format="svg",
            max_image_size=(1024, 768),
            offline_mode=True
        )
        
        assert config.enabled is True
        assert config.project_name == "test-project"
        assert config.entity == "test-entity"
        assert config.tags == ["test", "experiment"]
        assert config.log_frequency == 5
        assert config.image_format == "svg"
        assert config.max_image_size == (1024, 768)
        assert config.offline_mode is True
    
    def test_invalid_image_format(self) -> None:
        """Test validation of image format."""
        with pytest.raises(ValueError, match="Invalid image_format"):
            WandbConfig(image_format="invalid")
    
    def test_invalid_log_frequency(self) -> None:
        """Test validation of log frequency."""
        with pytest.raises(ValueError, match="log_frequency must be positive"):
            WandbConfig(log_frequency=0)
        
        with pytest.raises(ValueError, match="log_frequency must be positive"):
            WandbConfig(log_frequency=-1)
    
    def test_invalid_max_image_size(self) -> None:
        """Test validation of max image size."""
        with pytest.raises(ValueError, match="max_image_size must be tuple"):
            WandbConfig(max_image_size=(800,))  # Only one dimension
        
        with pytest.raises(ValueError, match="max_image_size must be tuple"):
            WandbConfig(max_image_size=(0, 600))  # Zero dimension
        
        with pytest.raises(ValueError, match="max_image_size must be tuple"):
            WandbConfig(max_image_size=(-100, 600))  # Negative dimension


class TestWandbIntegration:
    """Test WandbIntegration class functionality."""
    
    def test_init_disabled(self) -> None:
        """Test initialization with disabled config."""
        config = WandbConfig(enabled=False)
        integration = WandbIntegration(config)
        
        assert integration.config == config
        assert integration.run is None
        assert not integration.is_available
        assert not integration.is_initialized
    
    @patch('jaxarc.utils.visualization.wandb_integration.logger')
    def test_init_wandb_unavailable(self, mock_logger: MagicMock) -> None:
        """Test initialization when wandb is not available."""
        config = WandbConfig(enabled=True)
        
        with patch.dict('sys.modules', {'wandb': None}):
            integration = WandbIntegration(config)
        
        assert not integration.is_available
        mock_logger.warning.assert_called_once()
        assert "wandb not available" in mock_logger.warning.call_args[0][0]
    
    @patch('jaxarc.utils.visualization.wandb_integration.logger')
    def test_init_wandb_available(self, mock_logger: MagicMock) -> None:
        """Test initialization when wandb is available."""
        config = WandbConfig(enabled=True)
        mock_wandb = MagicMock()
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            integration = WandbIntegration(config)
        
        assert integration.is_available
        mock_logger.info.assert_called_with("Wandb successfully imported and available")
    
    def test_initialize_run_disabled(self) -> None:
        """Test run initialization when wandb is disabled."""
        config = WandbConfig(enabled=False)
        integration = WandbIntegration(config)
        
        result = integration.initialize_run({"test": "config"})
        
        assert result is False
        assert integration.run is None
    
    @patch('jaxarc.utils.visualization.wandb_integration.logger')
    def test_initialize_run_success(self, mock_logger: MagicMock) -> None:
        """Test successful run initialization."""
        config = WandbConfig(enabled=True, project_name="test-project")
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.name = "test-run"
        mock_run.id = "test-id"
        mock_wandb.init.return_value = mock_run
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            integration = WandbIntegration(config)
            result = integration.initialize_run({"test": "config"}, run_name="test-run")
        
        assert result is True
        assert integration.run == mock_run
        assert integration.is_initialized
        
        mock_wandb.init.assert_called_once_with(
            project="test-project",
            entity=None,
            name="test-run",
            id=None,
            tags=[],
            notes=None,
            group=None,
            job_type=None,
            config={"test": "config"},
            save_code=True,
            resume=None
        )
    
    def test_log_step_not_available(self) -> None:
        """Test step logging when wandb is not available."""
        config = WandbConfig(enabled=False)
        integration = WandbIntegration(config)
        
        result = integration.log_step(1, {"reward": 0.5})
        
        assert result is False
    
    def test_log_step_frequency_check(self) -> None:
        """Test step logging frequency check."""
        config = WandbConfig(enabled=True, log_frequency=10)
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            integration = WandbIntegration(config)
            integration.run = mock_run
            integration._last_log_step = 0
            
            # Should not log (frequency not met)
            result = integration.log_step(5, {"reward": 0.5})
            assert result is True
            mock_run.log.assert_not_called()
            
            # Should log (frequency met)
            result = integration.log_step(10, {"reward": 0.8})
            assert result is True
            mock_run.log.assert_called_once()
    
    def test_log_step_force_log(self) -> None:
        """Test forced step logging."""
        config = WandbConfig(enabled=True, log_frequency=10)
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            integration = WandbIntegration(config)
            integration.run = mock_run
            integration._last_log_step = 0
            
            # Force log even though frequency not met
            result = integration.log_step(5, {"reward": 0.5}, force_log=True)
            assert result is True
            mock_run.log.assert_called_once_with({"step": 5, "reward": 0.5}, step=5)
    
    def test_log_episode_summary(self) -> None:
        """Test episode summary logging."""
        config = WandbConfig(enabled=True)
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            integration = WandbIntegration(config)
            integration.run = mock_run
            
            summary_data = {
                "total_reward": 10.5,
                "steps": 25,
                "success": True
            }
            
            result = integration.log_episode_summary(1, summary_data)
            assert result is True
            
            expected_data = {
                "episode": 1,
                "total_reward": 10.5,
                "steps": 25,
                "success": True
            }
            mock_run.log.assert_called_once_with(expected_data, step=1)
    
    def test_log_config_update(self) -> None:
        """Test configuration update logging."""
        config = WandbConfig(enabled=True)
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            integration = WandbIntegration(config)
            integration.run = mock_run
            
            config_update = {"learning_rate": 0.001}
            result = integration.log_config_update(config_update)
            
            assert result is True
            mock_run.config.update.assert_called_once_with(config_update)
    
    def test_finish_run(self) -> None:
        """Test run finishing."""
        config = WandbConfig(enabled=True)
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            integration = WandbIntegration(config)
            integration.run = mock_run
            
            integration.finish_run()
            
            mock_run.finish.assert_called_once()
            assert integration.run is None
    
    def test_properties(self) -> None:
        """Test integration properties."""
        config = WandbConfig(enabled=True)
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.get_url.return_value = "https://wandb.ai/test/run"
        mock_run.id = "test-id"
        mock_run.name = "test-name"
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            integration = WandbIntegration(config)
            integration.run = mock_run
            
            assert integration.run_url == "https://wandb.ai/test/run"
            assert integration.run_id == "test-id"
            assert integration.run_name == "test-name"


class TestWandbConfigFactories:
    """Test wandb configuration factory functions."""
    
    def test_create_wandb_config(self) -> None:
        """Test basic wandb config creation."""
        config = create_wandb_config(enabled=True, project_name="test-project")
        
        assert config.enabled is True
        assert config.project_name == "test-project"
        assert isinstance(config, WandbConfig)
    
    def test_create_research_wandb_config(self) -> None:
        """Test research-optimized wandb config."""
        config = create_research_wandb_config(
            project_name="research-project",
            entity="research-team"
        )
        
        assert config.enabled is True
        assert config.project_name == "research-project"
        assert config.entity == "research-team"
        assert config.log_frequency == 5  # More frequent for research
        assert config.image_format == "both"
        assert config.log_system_metrics is True
        assert config.save_code is True
        assert config.save_config is True
        assert "research" in config.tags
        assert "jaxarc" in config.tags
    
    def test_create_development_wandb_config(self) -> None:
        """Test development-optimized wandb config."""
        config = create_development_wandb_config(project_name="dev-project")
        
        assert config.enabled is True
        assert config.project_name == "dev-project"
        assert config.log_frequency == 20  # Less frequent for development
        assert config.image_format == "png"  # Faster PNG only
        assert config.log_system_metrics is False
        assert config.save_code is False  # Don't save code during development
        assert config.offline_mode is True  # Work offline during development
        assert "development" in config.tags
        assert "jaxarc" in config.tags


class TestWandbIntegrationErrorHandling:
    """Test error handling in wandb integration."""
    
    @patch('jaxarc.utils.visualization.wandb_integration.logger')
    def test_initialize_run_error(self, mock_logger: MagicMock) -> None:
        """Test error handling during run initialization."""
        config = WandbConfig(enabled=True)
        mock_wandb = MagicMock()
        mock_wandb.init.side_effect = Exception("Network error")
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            integration = WandbIntegration(config)
            result = integration.initialize_run({"test": "config"})
        
        assert result is False
        assert not integration.is_available
        mock_logger.error.assert_called_once()
        assert "Failed to initialize wandb run" in mock_logger.error.call_args[0][0]
    
    @patch('jaxarc.utils.visualization.wandb_integration.logger')
    def test_log_step_error(self, mock_logger: MagicMock) -> None:
        """Test error handling during step logging."""
        config = WandbConfig(enabled=True, retry_attempts=1)
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.log.side_effect = Exception("Network error")
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            integration = WandbIntegration(config)
            integration.run = mock_run
            
            result = integration.log_step(1, {"reward": 0.5}, force_log=True)
        
        assert result is False
        mock_logger.error.assert_called()
    
    @patch('jaxarc.utils.visualization.wandb_integration.logger')
    def test_finish_run_error(self, mock_logger: MagicMock) -> None:
        """Test error handling during run finishing."""
        config = WandbConfig(enabled=True)
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.finish.side_effect = Exception("Network error")
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            integration = WandbIntegration(config)
            integration.run = mock_run
            
            integration.finish_run()
        
        assert integration.run is None  # Should still be set to None
        mock_logger.error.assert_called_once()
        assert "Error finishing wandb run" in mock_logger.error.call_args[0][0]


class TestWandbIntegrationRetryLogic:
    """Test retry logic in wandb integration."""
    
    @patch('jaxarc.utils.visualization.wandb_integration.time.sleep')
    @patch('jaxarc.utils.visualization.wandb_integration.logger')
    def test_retry_logic_success_on_second_attempt(
        self, 
        mock_logger: MagicMock,
        mock_sleep: MagicMock
    ) -> None:
        """Test successful retry after initial failure."""
        config = WandbConfig(enabled=True, retry_attempts=3, retry_delay=0.1)
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        
        # Fail first time, succeed second time
        mock_run.log.side_effect = [Exception("Network error"), None]
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            integration = WandbIntegration(config)
            integration.run = mock_run
            
            result = integration.log_step(1, {"reward": 0.5}, force_log=True)
        
        assert result is True
        assert mock_run.log.call_count == 2
        mock_logger.warning.assert_called_once()
        mock_sleep.assert_called_once_with(0.1)  # First retry delay
    
    @patch('jaxarc.utils.visualization.wandb_integration.time.sleep')
    @patch('jaxarc.utils.visualization.wandb_integration.logger')
    def test_retry_logic_all_attempts_fail(
        self, 
        mock_logger: MagicMock,
        mock_sleep: MagicMock
    ) -> None:
        """Test retry logic when all attempts fail."""
        config = WandbConfig(enabled=True, retry_attempts=2, retry_delay=0.1)
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.log.side_effect = Exception("Network error")
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            integration = WandbIntegration(config)
            integration.run = mock_run
            
            result = integration.log_step(1, {"reward": 0.5}, force_log=True)
        
        assert result is False
        assert mock_run.log.call_count == 2  # All retry attempts
        mock_logger.warning.assert_called_once()  # First retry warning
        mock_logger.error.assert_called_once()    # Final failure error
        mock_sleep.assert_called_once_with(0.1)   # First retry delay


if __name__ == "__main__":
    pytest.main([__file__])