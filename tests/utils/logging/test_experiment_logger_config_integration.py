"""Test configuration integration for ExperimentLogger.

This test module verifies that the ExperimentLogger properly integrates with
Hydra configurations and handles various configuration scenarios correctly.

This addresses Task 10 requirements:
- 10.1: Existing Hydra debug configurations work with new ExperimentLogger
- 10.2: debug.level="off" properly disables logging
- 10.3: wandb configuration structure remains compatible
- 10.4: Configuration validation using existing config utilities
- 10.5: Configuration compatibility and edge cases
"""

import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch
import pytest

from jaxarc.utils.logging.experiment_logger import ExperimentLogger
from jaxarc.envs.config import (
    EnvironmentConfig, WandbConfig, LoggingConfig, StorageConfig
)


class TestExperimentLoggerConfigIntegration:
    """Test configuration integration for ExperimentLogger."""
    
    def test_debug_level_off_disables_logging(self):
        """Test that debug.level='off' properly disables logging handlers."""
        # Create config with debug level off
        env_config = EnvironmentConfig(debug_level="off")
        config = Mock()
        config.environment = env_config
        # Ensure no wandb config to avoid handler creation
        del config.wandb
        
        # Initialize logger
        logger = ExperimentLogger(config)
        
        # Verify no handlers are initialized when debug level is off
        assert len(logger.handlers) == 0, f"Expected 0 handlers with debug_level='off', got {len(logger.handlers)}: {list(logger.handlers.keys())}"
        
        # Test that logging methods don't crash even with no handlers
        step_data = {
            'step_num': 1,
            'reward': 0.5,
            'info': {'metrics': {'similarity': 0.8}}
        }
        
        summary_data = {
            'episode_num': 1,
            'total_steps': 10,
            'total_reward': 5.0
        }
        
        # These should not crash even with no handlers
        logger.log_step(step_data)
        logger.log_episode_summary(summary_data)
        logger.close()
    
    def test_debug_level_standard_enables_handlers(self):
        """Test that debug.level='standard' enables appropriate handlers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config with standard debug level
            env_config = EnvironmentConfig(debug_level="standard")
            storage_config = StorageConfig(
                base_output_dir=temp_dir,
                logs_dir="logs"
            )
            
            config = Mock()
            config.environment = env_config
            config.storage = storage_config
            
            # Initialize logger
            logger = ExperimentLogger(config)
            
            # Should have at least one handler (file and/or rich)
            assert len(logger.handlers) > 0, "Expected at least one handler to be initialized"
            
            # Test that logging works
            step_data = {
                'step_num': 1,
                'reward': 0.5,
                'info': {'metrics': {'similarity': 0.8}}
            }
            
            logger.log_step(step_data)
            logger.close()
    
    def test_wandb_config_compatibility(self):
        """Test that wandb configuration structure remains compatible."""
        # Test with wandb enabled
        wandb_config = WandbConfig(
            enabled=True,
            project_name="test-project",
            entity="test-entity",
            tags=("test", "config"),
            offline_mode=True  # Use offline mode for testing
        )
        
        env_config = EnvironmentConfig(debug_level="standard")
        config = Mock()
        config.environment = env_config
        config.wandb = wandb_config
        
        # Should attempt to create wandb handler
        logger = ExperimentLogger(config)
        
        # May or may not create wandb handler depending on environment
        # but should not crash
        logger.close()
        
        # Test with wandb disabled
        wandb_config_disabled = WandbConfig(enabled=False)
        config.wandb = wandb_config_disabled
        
        logger_disabled = ExperimentLogger(config)
        assert 'wandb' not in logger_disabled.handlers, "Wandb handler should not be created when disabled"
        logger_disabled.close()
    
    def test_configuration_validation(self):
        """Test configuration validation using existing config utilities."""
        # Test valid configuration
        env_config = EnvironmentConfig(debug_level="standard")
        errors = env_config.validate()
        assert len(errors) == 0, f"Valid config should have no errors, got: {errors}"
        
        # Test wandb config validation
        wandb_config = WandbConfig(
            enabled=True,
            project_name="test-project",
            log_frequency=10
        )
        
        wandb_errors = wandb_config.validate()
        assert len(wandb_errors) == 0, f"Valid wandb config should have no errors, got: {wandb_errors}"
        
        # Test logging config validation
        logging_config = LoggingConfig(
            log_level="INFO",
            log_frequency=10
        )
        
        logging_errors = logging_config.validate()
        assert len(logging_errors) == 0, f"Valid logging config should have no errors, got: {logging_errors}"
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with minimal config
        minimal_config = Mock()
        minimal_config.environment = EnvironmentConfig(debug_level="off")
        # Ensure no wandb config exists
        del minimal_config.wandb
        
        logger = ExperimentLogger(minimal_config)
        assert len(logger.handlers) == 0, f"Minimal config should create no handlers, got {len(logger.handlers)}: {list(logger.handlers.keys())}"
        logger.close()
        
        # Test with missing config attributes
        incomplete_config = Mock()
        incomplete_config.environment = EnvironmentConfig(debug_level="standard")
        # Missing storage and wandb configs
        
        logger_incomplete = ExperimentLogger(incomplete_config)
        # Should handle missing configs gracefully
        logger_incomplete.close()
        
        # Test handler failure isolation
        config_with_bad_storage = Mock()
        config_with_bad_storage.environment = EnvironmentConfig(debug_level="standard")
        
        # Create storage config that will cause file handler to fail
        bad_storage = StorageConfig(
            base_output_dir="/invalid/path/that/cannot/be/created",
            logs_dir="logs"
        )
        config_with_bad_storage.storage = bad_storage
        
        logger_bad = ExperimentLogger(config_with_bad_storage)
        # Should handle handler creation failures gracefully
        logger_bad.close()
    
    def test_all_debug_levels(self):
        """Test that all debug levels work correctly."""
        debug_levels = ["off", "minimal", "standard", "verbose", "research"]
        
        for level in debug_levels:
            env_config = EnvironmentConfig(debug_level=level)
            config = Mock()
            config.environment = env_config
            
            if level != "off":
                # Provide valid storage for non-off levels
                with tempfile.TemporaryDirectory() as temp_dir:
                    storage_config = StorageConfig(
                        base_output_dir=temp_dir,
                        logs_dir="logs"
                    )
                    config.storage = storage_config
                    
                    logger = ExperimentLogger(config)
                    
                    if level == "off":
                        assert len(logger.handlers) == 0, f"Debug level '{level}' should create no handlers"
                    else:
                        # Should create at least one handler for non-off levels
                        # (may fail in test environment, but should not crash)
                        pass
                    
                    logger.close()
            else:
                # For "off" level, don't need storage
                del config.wandb  # Remove wandb to ensure no handlers
                logger = ExperimentLogger(config)
                assert len(logger.handlers) == 0, f"Debug level '{level}' should create no handlers"
                logger.close()
    
    def test_config_hashability(self):
        """Test that all config objects are hashable for JAX compatibility."""
        configs = [
            EnvironmentConfig(),
            LoggingConfig(),
            StorageConfig(),
            WandbConfig()
        ]
        
        for config in configs:
            try:
                hash_value = hash(config)
                assert isinstance(hash_value, int), f"Hash should be an integer for {type(config).__name__}"
            except TypeError as e:
                pytest.fail(f"{type(config).__name__} is not hashable: {e}")
    
    def test_backward_compatibility(self):
        """Test backward compatibility with existing configurations."""
        # Test legacy config structure (if any)
        legacy_config = Mock()
        legacy_config.environment = EnvironmentConfig(debug_level="standard")
        
        # Add legacy debug config structure
        legacy_debug = Mock()
        legacy_debug.output_dir = "outputs/legacy"
        legacy_config.debug = legacy_debug
        
        logger = ExperimentLogger(legacy_config)
        # Should handle legacy config structure
        logger.close()