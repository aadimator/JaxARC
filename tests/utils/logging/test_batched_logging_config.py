"""Unit tests for batched logging configuration system.

This module tests the configuration loading, validation, and error handling
for batched logging settings, including the removal of deprecated async
logging fields.
"""

from __future__ import annotations

import pytest
from omegaconf import DictConfig, OmegaConf

from jaxarc.envs.config import LoggingConfig, ConfigValidationError


class TestBatchedLoggingConfig:
    """Test cases for batched logging configuration."""
    
    def test_default_batched_logging_config(self):
        """Test default batched logging configuration values."""
        config = LoggingConfig()
        
        # Test default values for batched logging
        assert config.batched_logging_enabled is False
        assert config.sampling_enabled is True
        assert config.num_samples == 3
        assert config.sample_frequency == 50
        
        # Test default aggregated metrics selection
        assert config.log_aggregated_rewards is True
        assert config.log_aggregated_similarity is True
        assert config.log_loss_metrics is True
        assert config.log_gradient_norms is True
        assert config.log_episode_lengths is True
        assert config.log_success_rates is True
    
    def test_batched_logging_config_from_hydra(self):
        """Test creating batched logging config from Hydra DictConfig."""
        hydra_config = OmegaConf.create({
            'structured_logging': True,
            'log_format': 'json',
            'log_level': 'INFO',
            'batched_logging_enabled': True,
            'sampling_enabled': True,
            'num_samples': 5,
            'sample_frequency': 25,
            'log_aggregated_rewards': True,
            'log_aggregated_similarity': False,
            'log_loss_metrics': True,
            'log_gradient_norms': False,
            'log_episode_lengths': True,
            'log_success_rates': True
        })
        
        config = LoggingConfig.from_hydra(hydra_config)
        
        # Test batched logging settings
        assert config.batched_logging_enabled is True
        assert config.sampling_enabled is True
        assert config.num_samples == 5
        assert config.sample_frequency == 25
        
        # Test aggregated metrics selection
        assert config.log_aggregated_rewards is True
        assert config.log_aggregated_similarity is False
        assert config.log_loss_metrics is True
        assert config.log_gradient_norms is False
        assert config.log_episode_lengths is True
        assert config.log_success_rates is True
    
    def test_batched_logging_config_validation_success(self):
        """Test successful validation of batched logging configuration."""
        config = LoggingConfig(
            batched_logging_enabled=True,
            num_samples=10,
            sample_frequency=100
        )
        
        errors = config.validate()
        assert len(errors) == 0
    
    def test_batched_logging_config_validation_errors(self):
        """Test validation errors for invalid batched logging configuration."""
        # Test negative num_samples
        config = LoggingConfig(
            batched_logging_enabled=True,
            num_samples=-1,
            sample_frequency=50
        )
        
        errors = config.validate()
        assert len(errors) > 0
        assert any("num_samples must be positive" in error for error in errors)
        
        # Test zero sample_frequency
        config = LoggingConfig(
            batched_logging_enabled=True,
            num_samples=3,
            sample_frequency=0
        )
        
        errors = config.validate()
        assert len(errors) > 0
        assert any("sample_frequency must be positive" in error for error in errors)
        
        # Test zero num_samples
        config = LoggingConfig(
            batched_logging_enabled=True,
            num_samples=0,
            sample_frequency=50
        )
        
        errors = config.validate()
        assert len(errors) > 0
        assert any("num_samples must be positive" in error for error in errors)
    
    def test_batched_logging_config_validation_warnings(self):
        """Test validation warnings for large batched logging values."""
        # Test large num_samples warning - we can't easily test loguru warnings
        # but we can test that validation doesn't return errors for large values
        config = LoggingConfig(
            batched_logging_enabled=True,
            num_samples=2000,
            sample_frequency=50
        )
        
        errors = config.validate()
        assert len(errors) == 0  # No errors, just warnings (which we can't easily capture)
    
    def test_async_logging_fields_removed(self):
        """Test that deprecated async logging fields are not present in LoggingConfig."""
        config = LoggingConfig()
        
        # These fields should not exist in the new configuration
        deprecated_fields = [
            'queue_size',
            'worker_threads', 
            'batch_size',  # Note: this is logging batch_size, not training batch_size
            'flush_interval',
            'enable_compression'  # async-specific compression
        ]
        
        for field in deprecated_fields:
            assert not hasattr(config, field), f"Deprecated field '{field}' should not exist"
    
    def test_async_logging_fields_not_in_from_hydra(self):
        """Test that from_hydra ignores deprecated async logging fields."""
        hydra_config = OmegaConf.create({
            'structured_logging': True,
            'log_format': 'json',
            'batched_logging_enabled': True,
            # These deprecated fields should be ignored
            'queue_size': 1000,
            'worker_threads': 4,
            'batch_size': 100,
            'flush_interval': 5.0,
            'enable_compression': True
        })
        
        # Should not raise an error and should ignore deprecated fields
        config = LoggingConfig.from_hydra(hydra_config)
        
        assert config.structured_logging is True
        assert config.log_format == 'json'
        assert config.batched_logging_enabled is True
        
        # Deprecated fields should not be set
        deprecated_fields = ['queue_size', 'worker_threads', 'batch_size', 'flush_interval', 'enable_compression']
        for field in deprecated_fields:
            assert not hasattr(config, field)
    
    def test_batched_logging_config_hashability(self):
        """Test that LoggingConfig with batched logging settings is hashable for JAX compatibility."""
        config = LoggingConfig(
            batched_logging_enabled=True,
            sampling_enabled=True,
            num_samples=5,
            sample_frequency=25,
            log_aggregated_rewards=True,
            log_aggregated_similarity=False
        )
        
        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)
        
        # Test that identical configs have same hash
        config2 = LoggingConfig(
            batched_logging_enabled=True,
            sampling_enabled=True,
            num_samples=5,
            sample_frequency=25,
            log_aggregated_rewards=True,
            log_aggregated_similarity=False
        )
        
        assert hash(config) == hash(config2)
    
    def test_batched_logging_disabled_validation(self):
        """Test validation when batched logging is disabled."""
        config = LoggingConfig(
            batched_logging_enabled=False,
            num_samples=-1,  # Invalid, but should not matter when disabled
            sample_frequency=0  # Invalid, but should not matter when disabled
        )
        
        # When batched logging is disabled, validation should not check batched parameters
        # This is the current behavior - only validate when enabled
        errors = config.validate()
        assert len(errors) == 0  # No errors when disabled
    
    def test_configuration_error_messages(self):
        """Test that configuration validation provides clear error messages."""
        config = LoggingConfig(
            batched_logging_enabled=True,
            num_samples=-5,
            sample_frequency=-10
        )
        
        errors = config.validate()
        
        # Check that error messages are descriptive
        error_text = ' '.join(errors)
        assert 'num_samples' in error_text
        assert 'positive' in error_text
        # Note: sample_frequency validation is done by validate_positive_int which throws ConfigValidationError
        # so it should be caught and added to errors
    
    def test_partial_hydra_config(self):
        """Test from_hydra with partial configuration (missing batched logging fields)."""
        hydra_config = OmegaConf.create({
            'structured_logging': True,
            'log_format': 'json',
            # Missing batched logging fields - should use defaults
        })
        
        config = LoggingConfig.from_hydra(hydra_config)
        
        # Should use default values for missing fields
        assert config.batched_logging_enabled is False
        assert config.sampling_enabled is True
        assert config.num_samples == 3
        assert config.sample_frequency == 50
        assert config.log_aggregated_rewards is True
    
    def test_edge_case_values(self):
        """Test edge case values for batched logging configuration."""
        # Test minimum valid values
        config = LoggingConfig(
            batched_logging_enabled=True,
            num_samples=1,
            sample_frequency=1
        )
        
        errors = config.validate()
        assert len(errors) == 0
        
        # Test very large but valid values
        config = LoggingConfig(
            batched_logging_enabled=True,
            num_samples=999,  # Just under warning threshold
            sample_frequency=10000
        )
        
        errors = config.validate()
        assert len(errors) == 0


class TestConfigurationMigration:
    """Test cases for configuration migration from async logging."""
    
    def test_old_config_compatibility(self):
        """Test that old configurations without batched logging fields still work."""
        # Simulate an old configuration that doesn't have batched logging fields
        old_hydra_config = OmegaConf.create({
            'structured_logging': True,
            'log_format': 'json',
            'log_level': 'INFO',
            'compression': True,
            'include_full_states': False,
            'log_operations': False,
            'log_grid_changes': False,
            'log_rewards': False,
            'log_episode_start': True,
            'log_episode_end': True,
            'log_key_moments': True,
            'log_frequency': 10
        })
        
        config = LoggingConfig.from_hydra(old_hydra_config)
        
        # Should work without errors and use defaults for new fields
        assert config.structured_logging is True
        assert config.log_format == 'json'
        assert config.batched_logging_enabled is False  # Default
        assert config.sampling_enabled is True  # Default
        assert config.num_samples == 3  # Default
    
    def test_mixed_old_new_config(self):
        """Test configuration with mix of old and new fields."""
        mixed_config = OmegaConf.create({
            # Old fields
            'structured_logging': True,
            'log_format': 'json',
            'log_frequency': 5,
            
            # New batched logging fields
            'batched_logging_enabled': True,
            'num_samples': 10,
            'sample_frequency': 100,
            'log_aggregated_rewards': False,
            
            # Deprecated fields that should be ignored
            'queue_size': 500,
            'worker_threads': 2
        })
        
        config = LoggingConfig.from_hydra(mixed_config)
        
        # Old fields should be preserved
        assert config.structured_logging is True
        assert config.log_format == 'json'
        assert config.log_frequency == 5
        
        # New fields should be set
        assert config.batched_logging_enabled is True
        assert config.num_samples == 10
        assert config.sample_frequency == 100
        assert config.log_aggregated_rewards is False
        
        # Deprecated fields should not exist
        assert not hasattr(config, 'queue_size')
        assert not hasattr(config, 'worker_threads')


class TestBatchedLoggingYAMLConfig:
    """Test cases for YAML configuration file loading."""
    
    def test_batched_yaml_config_structure(self):
        """Test that batched.yaml configuration can be loaded properly."""
        # Simulate the expected structure of batched.yaml
        yaml_content = {
            'structured_logging': True,
            'log_format': 'json',
            'log_level': 'INFO',
            'compression': True,
            'include_full_states': False,
            'batched_logging_enabled': True,
            'log_frequency': 10,
            'sampling_enabled': True,
            'num_samples': 3,
            'sample_frequency': 50,
            'log_aggregated_rewards': True,
            'log_aggregated_similarity': True,
            'log_loss_metrics': True,
            'log_gradient_norms': True,
            'log_episode_lengths': True,
            'log_success_rates': True
        }
        
        hydra_config = OmegaConf.create(yaml_content)
        config = LoggingConfig.from_hydra(hydra_config)
        
        # Verify all fields are loaded correctly
        assert config.structured_logging is True
        assert config.log_format == 'json'
        assert config.batched_logging_enabled is True
        assert config.log_frequency == 10
        assert config.sampling_enabled is True
        assert config.num_samples == 3
        assert config.sample_frequency == 50
        assert config.log_aggregated_rewards is True
        assert config.log_aggregated_similarity is True
        assert config.log_loss_metrics is True
        assert config.log_gradient_norms is True
        assert config.log_episode_lengths is True
        assert config.log_success_rates is True
    
    def test_yaml_config_validation(self):
        """Test validation of YAML-loaded configuration."""
        yaml_content = {
            'batched_logging_enabled': True,
            'num_samples': 5,
            'sample_frequency': 25,
            'log_aggregated_rewards': True
        }
        
        hydra_config = OmegaConf.create(yaml_content)
        config = LoggingConfig.from_hydra(hydra_config)
        
        errors = config.validate()
        assert len(errors) == 0
    
    def test_yaml_config_with_invalid_values(self):
        """Test YAML configuration with invalid values."""
        yaml_content = {
            'batched_logging_enabled': True,
            'num_samples': -1,  # Invalid
            'sample_frequency': 0,  # Invalid
            'log_format': 'invalid_format'  # Invalid
        }
        
        hydra_config = OmegaConf.create(yaml_content)
        config = LoggingConfig.from_hydra(hydra_config)
        
        errors = config.validate()
        assert len(errors) > 0
        
        # Should contain errors - check for any of the invalid fields
        error_text = ' '.join(errors)
        assert ('num_samples' in error_text or 
                'sample_frequency' in error_text or 
                'log_format' in error_text)