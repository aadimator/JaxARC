"""Integration tests for wandb configuration with Hydra."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from jaxarc.utils.visualization.wandb_integration import WandbConfig


class TestWandbConfigIntegration:
    """Test wandb configuration integration with Hydra."""

    def test_wandb_config_from_dict(self) -> None:
        """Test creating WandbConfig from dictionary (Hydra-style)."""
        config_dict = {
            "enabled": True,
            "project_name": "test-project",
            "entity": "test-entity",
            "tags": ["test", "integration"],
            "log_frequency": 5,
            "image_format": "svg",
            "max_image_size": [1024, 768],
            "offline_mode": True,
            "retry_attempts": 2,
            "save_code": False,
        }

        # Convert to OmegaConf (simulating Hydra)
        omega_config = OmegaConf.create(config_dict)

        # Create WandbConfig from the dictionary
        wandb_config = WandbConfig(**omega_config)

        assert wandb_config.enabled is True
        assert wandb_config.project_name == "test-project"
        assert wandb_config.entity == "test-entity"
        assert wandb_config.tags == ["test", "integration"]
        assert wandb_config.log_frequency == 5
        assert wandb_config.image_format == "svg"
        assert wandb_config.max_image_size == (1024, 768)
        assert wandb_config.offline_mode is True
        assert wandb_config.retry_attempts == 2
        assert wandb_config.save_code is False

    def test_wandb_config_partial_dict(self) -> None:
        """Test creating WandbConfig with partial configuration (using defaults)."""
        config_dict = {
            "enabled": True,
            "project_name": "partial-test",
            "log_frequency": 15,
        }

        omega_config = OmegaConf.create(config_dict)
        wandb_config = WandbConfig(**omega_config)

        # Check specified values
        assert wandb_config.enabled is True
        assert wandb_config.project_name == "partial-test"
        assert wandb_config.log_frequency == 15

        # Check defaults are preserved
        assert wandb_config.entity is None
        assert wandb_config.tags == []
        assert wandb_config.image_format == "png"
        assert wandb_config.max_image_size == (800, 600)
        assert wandb_config.offline_mode is False

    def test_wandb_config_validation_from_dict(self) -> None:
        """Test that validation works when creating from dictionary."""
        config_dict = {"enabled": True, "image_format": "invalid_format"}

        omega_config = OmegaConf.create(config_dict)

        with pytest.raises(ValueError, match="Invalid image_format"):
            WandbConfig(**omega_config)

    def test_nested_config_structure(self) -> None:
        """Test wandb config as part of larger configuration structure."""
        full_config = {
            "experiment": {"name": "test_experiment", "seed": 42},
            "wandb": {
                "enabled": True,
                "project_name": "nested-test",
                "tags": ["nested", "config"],
                "log_frequency": 8,
            },
            "training": {"epochs": 100, "batch_size": 32},
        }

        omega_config = OmegaConf.create(full_config)

        # Extract wandb config
        wandb_config = WandbConfig(**omega_config.wandb)

        assert wandb_config.enabled is True
        assert wandb_config.project_name == "nested-test"
        assert wandb_config.tags == ["nested", "config"]
        assert wandb_config.log_frequency == 8

    def test_config_serialization(self) -> None:
        """Test that WandbConfig can be serialized back to dict/YAML."""
        original_config = WandbConfig(
            enabled=True,
            project_name="serialization-test",
            tags=["serialize", "test"],
            log_frequency=12,
            image_format="both",
            offline_mode=True,
        )

        # Convert to dict (simulating what would be logged to wandb)
        config_dict = {
            "enabled": original_config.enabled,
            "project_name": original_config.project_name,
            "tags": original_config.tags,
            "log_frequency": original_config.log_frequency,
            "image_format": original_config.image_format,
            "offline_mode": original_config.offline_mode,
            "max_image_size": original_config.max_image_size,
            "retry_attempts": original_config.retry_attempts,
            "save_code": original_config.save_code,
        }

        # Verify we can recreate the config
        recreated_config = WandbConfig(**config_dict)

        assert recreated_config.enabled == original_config.enabled
        assert recreated_config.project_name == original_config.project_name
        assert recreated_config.tags == original_config.tags
        assert recreated_config.log_frequency == original_config.log_frequency
        assert recreated_config.image_format == original_config.image_format
        assert recreated_config.offline_mode == original_config.offline_mode


class TestWandbConfigFiles:
    """Test the actual wandb configuration files."""

    def test_local_only_config_structure(self) -> None:
        """Test that local_only.yaml has correct structure."""
        config_path = Path("conf/logging/local_only.yaml")
        assert config_path.exists(), "local_only.yaml config file should exist"

        # Load and parse the config
        with open(config_path) as f:
            content = f.read()

        # Basic structure checks
        assert "wandb:" in content
        assert "enabled: false" in content
        assert "logging:" in content
        assert "episode:" in content
        assert "async_logger:" in content

    def test_wandb_basic_config_structure(self) -> None:
        """Test that wandb_basic.yaml has correct structure."""
        config_path = Path("conf/logging/wandb_basic.yaml")
        assert config_path.exists(), "wandb_basic.yaml config file should exist"

        with open(config_path) as f:
            content = f.read()

        # Basic structure checks
        assert "wandb:" in content
        assert "enabled: true" in content
        assert "project_name:" in content
        assert "log_frequency:" in content
        assert "image_format:" in content

    def test_wandb_full_config_structure(self) -> None:
        """Test that wandb_full.yaml has correct structure."""
        config_path = Path("conf/logging/wandb_full.yaml")
        assert config_path.exists(), "wandb_full.yaml config file should exist"

        with open(config_path) as f:
            content = f.read()

        # Basic structure checks
        assert "wandb:" in content
        assert "enabled: true" in content
        assert "log_gradients: true" in content
        assert "log_model_topology: true" in content
        assert 'image_format: "both"' in content


if __name__ == "__main__":
    pytest.main([__file__])
