"""Tests for enhanced visualization configuration integration."""

import pytest
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from jaxarc.utils.visualization.config_validation import validate_config
from jaxarc.utils.visualization.config_composition import quick_compose
from jaxarc.utils.visualization.config_migration import ConfigMigrator, migrate_legacy_config


class TestConfigurationIntegration:
    """Test configuration integration with existing systems."""
    
    def test_debug_off_integration(self):
        """Test debug=off configuration integration."""
        config = quick_compose(debug_level="off", validate=True)
        
        # Should have visualization disabled
        assert config.visualization.debug_level == "off"
        assert config.visualization.enabled == False
        
        # Should have minimal storage requirements
        assert config.storage.max_storage_gb <= 2.0
        
        # Should have local logging only
        assert config.wandb.enabled == False
    
    def test_debug_standard_integration(self):
        """Test debug=standard configuration integration."""
        config = quick_compose(debug_level="standard", validate=True)
        
        # Should have standard visualization
        assert config.visualization.debug_level == "standard"
        assert config.visualization.enabled == True
        assert config.visualization.show_operation_names == True
        assert config.visualization.highlight_changes == True
        
        # Should have reasonable storage limits
        assert config.storage.max_storage_gb >= 1.0
        
        # Should support both local and wandb logging
        assert hasattr(config, 'logging')
    
    def test_debug_verbose_integration(self):
        """Test debug=verbose configuration integration."""
        config = quick_compose(
            debug_level="verbose", 
            storage_type="research",
            validate=True
        )
        
        # Should have verbose visualization
        assert config.visualization.debug_level == "verbose"
        assert config.visualization.log_frequency == 1  # Every step
        assert config.visualization.show_coordinates == True
        
        # Should have larger storage limits for research
        assert config.storage.max_storage_gb >= 10.0
        assert config.storage.cleanup_policy in ["manual", "oldest_first"]
    
    def test_storage_compatibility(self):
        """Test storage configuration compatibility."""
        # Development storage should work with all debug levels
        for debug_level in ["off", "minimal", "standard"]:
            config = quick_compose(
                debug_level=debug_level,
                storage_type="development",
                validate=True
            )
            assert config.storage.max_storage_gb <= 5.0
        
        # Research storage should work with verbose/full debug
        for debug_level in ["verbose", "full"]:
            config = quick_compose(
                debug_level=debug_level,
                storage_type="research",
                validate=True
            )
            assert config.storage.max_storage_gb >= 10.0
    
    def test_wandb_integration_compatibility(self):
        """Test wandb integration compatibility."""
        # Basic wandb should work with standard debug
        config = quick_compose(
            debug_level="standard",
            logging_type="wandb_basic",
            validate=True
        )
        
        assert config.wandb.enabled == True
        assert config.wandb.project_name == "jaxarc-experiments"
        assert config.wandb.log_frequency == 10
        
        # Full wandb should work with verbose debug
        config = quick_compose(
            debug_level="verbose",
            logging_type="wandb_full",
            validate=True
        )
        
        assert config.wandb.enabled == True
        assert config.wandb.project_name == "jaxarc-research"
        assert config.wandb.log_frequency == 5
    
    def test_legacy_migration_integration(self):
        """Test legacy configuration migration."""
        # Create legacy config
        legacy_config = OmegaConf.create({
            "log_rl_steps": True,
            "rl_steps_output_dir": "outputs/legacy_debug",
            "clear_output_dir": True,
            "dataset": "arc_agi_1",
            "seed": 42
        })
        
        # Migrate configuration
        migrated = migrate_legacy_config(legacy_config)
        
        # Should have new visualization settings
        assert hasattr(migrated, 'visualization')
        assert migrated.visualization.debug_level == "standard"
        assert migrated.visualization.output_dir == "outputs/legacy_debug"
        
        # Should have storage settings
        assert hasattr(migrated, 'storage')
        assert migrated.storage.cleanup_policy == "size_based"
        assert migrated.storage.auto_cleanup == True
        
        # Should preserve other settings
        assert migrated.dataset == "arc_agi_1"
        assert migrated.seed == 42
    
    def test_override_compatibility(self):
        """Test configuration overrides work correctly."""
        config = quick_compose(
            debug_level="standard",
            overrides=[
                "visualization.show_coordinates=true",
                "storage.max_storage_gb=15.0",
                "wandb.enabled=true",
                "wandb.project_name=custom-project"
            ],
            validate=True
        )
        
        # Overrides should be applied
        assert config.visualization.show_coordinates == True
        assert config.storage.max_storage_gb == 15.0
        assert config.wandb.enabled == True
        assert config.wandb.project_name == "custom-project"
    
    def test_cross_validation_warnings(self):
        """Test cross-validation between config sections."""
        # This should generate warnings about storage limits
        config = OmegaConf.create({
            "visualization": {
                "debug_level": "full",
                "enabled": True
            },
            "storage": {
                "max_storage_gb": 0.5  # Too small for full debug
            }
        })
        
        errors = validate_config(config)
        
        # Should have warnings about storage being too small
        storage_warnings = [e for e in errors if "storage" in e.field.lower()]
        assert len(storage_warnings) > 0
    
    def test_backward_compatibility(self):
        """Test backward compatibility with existing configurations."""
        # Should be able to create config without new sections
        minimal_config = OmegaConf.create({
            "dataset": "arc_agi_1",
            "seed": 42
        })
        
        # Validation should not fail on missing sections
        errors = validate_config(minimal_config)
        # Should not have critical errors for missing optional sections
        critical_errors = [e for e in errors if "required" in e.message.lower()]
        assert len(critical_errors) == 0


class TestConfigMigrator:
    """Test configuration migration utilities."""
    
    def test_detect_legacy_config(self):
        """Test legacy configuration detection."""
        migrator = ConfigMigrator()
        
        # Should detect legacy fields
        legacy_config = OmegaConf.create({
            "log_rl_steps": True,
            "rl_steps_output_dir": "outputs/debug"
        })
        assert migrator.detect_legacy_config(legacy_config) == True
        
        # Should not detect new format as legacy
        new_config = OmegaConf.create({
            "visualization": {"debug_level": "standard"},
            "storage": {"cleanup_policy": "size_based"}
        })
        assert migrator.detect_legacy_config(new_config) == False
    
    def test_migration_suggestions(self):
        """Test migration suggestions generation."""
        migrator = ConfigMigrator()
        
        legacy_config = OmegaConf.create({
            "log_rl_steps": True,
            "rl_steps_output_dir": "outputs/old_debug",
            "clear_output_dir": False
        })
        
        suggestions = migrator.suggest_migration(legacy_config)
        
        # Should suggest replacing legacy fields
        assert "log_rl_steps" in suggestions["detected_legacy_fields"]
        assert "rl_steps_output_dir" in suggestions["detected_legacy_fields"]
        assert "clear_output_dir" in suggestions["detected_legacy_fields"]
        
        # Should have recommended actions
        assert len(suggestions["recommended_actions"]) > 0
        
        # Should have new config structure
        assert "visualization" in suggestions["new_config_structure"]
        assert "storage" in suggestions["new_config_structure"]
    
    def test_migration_guide_generation(self):
        """Test migration guide generation."""
        migrator = ConfigMigrator()
        
        legacy_config = OmegaConf.create({
            "log_rl_steps": True,
            "rl_steps_output_dir": "outputs/debug"
        })
        
        guide = migrator.create_migration_guide(legacy_config)
        
        # Should contain key sections
        assert "Migration Guide" in guide
        assert "Legacy fields detected" in guide
        assert "Recommended actions" in guide
        assert "New configuration structure" in guide
        assert "Migration steps" in guide


@pytest.mark.integration
class TestEndToEndIntegration:
    """Test end-to-end configuration integration."""
    
    def test_complete_workflow(self):
        """Test complete configuration workflow."""
        # 1. Start with legacy config
        legacy_config = OmegaConf.create({
            "log_rl_steps": True,
            "rl_steps_output_dir": "outputs/test_debug",
            "clear_output_dir": True,
            "dataset": "arc_agi_2",
            "seed": 123
        })
        
        # 2. Migrate to new format
        migrated_config = migrate_legacy_config(legacy_config)
        
        # 3. Validate migrated config
        errors = validate_config(migrated_config)
        assert len(errors) == 0, f"Validation errors: {errors}"
        
        # 4. Compose with additional settings
        final_config = quick_compose(
            debug_level="standard",
            storage_type="development",
            overrides=[
                f"dataset={migrated_config.dataset}",
                f"seed={migrated_config.seed}"
            ],
            validate=True
        )
        
        # 5. Verify final configuration
        assert final_config.visualization.debug_level == "standard"
        assert final_config.storage.cleanup_policy in ["size_based", "oldest_first"]
        assert final_config.dataset == "arc_agi_2"
        assert final_config.seed == 123
    
    def test_hydra_compatibility(self):
        """Test compatibility with Hydra configuration system."""
        # This would require actual Hydra setup, so we'll test the structure
        config = quick_compose(debug_level="standard", validate=True)
        
        # Should have all required sections for Hydra
        required_sections = ["visualization", "storage", "logging", "wandb"]
        for section in required_sections:
            assert hasattr(config, section), f"Missing section: {section}"
        
        # Should be serializable to YAML (required for Hydra)
        yaml_str = OmegaConf.to_yaml(config)
        assert len(yaml_str) > 0
        
        # Should be deserializable from YAML
        reloaded_config = OmegaConf.create(yaml_str)
        assert reloaded_config.visualization.debug_level == "standard"