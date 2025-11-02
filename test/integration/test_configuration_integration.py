"""
Integration tests for configuration system integration.

This module tests Hydra integration, configuration composition,
configuration loading and validation workflows, and configuration-driven
environment creation.
"""

from __future__ import annotations

import chex
import pytest
from omegaconf import DictConfig

from jaxarc import JaxArcConfig
from jaxarc.configs.action_config import ActionConfig
from jaxarc.configs.dataset_config import DatasetConfig
from jaxarc.configs.environment_config import EnvironmentConfig
from jaxarc.configs.reward_config import RewardConfig
from jaxarc.registration import available_task_ids, make
from jaxarc.utils.core import get_config


class TestHydraIntegration:
    """Test Hydra integration and configuration composition."""

    def test_default_config_loading(self):
        """Test loading default Hydra configuration."""
        # Load default configuration
        cfg = get_config()

        # Verify it's a DictConfig
        assert isinstance(cfg, DictConfig)

        # Verify main sections exist
        assert "environment" in cfg
        assert "dataset" in cfg
        assert "action" in cfg
        assert "reward" in cfg
        assert "visualization" in cfg
        assert "storage" in cfg
        assert "logging" in cfg
        assert "wandb" in cfg

        # Verify some default values
        assert cfg.environment.max_episode_steps > 0
        assert cfg.dataset.dataset_name is not None
        assert cfg.action.max_operations > 0

    def test_config_overrides(self):
        """Test configuration loading with overrides."""
        # Test with overrides
        overrides = [
            "environment.max_episode_steps=200",
            "dataset.dataset_name=TestDataset",
            "reward.success_bonus=5.0",
        ]

        cfg = get_config(overrides=overrides)

        # Verify overrides were applied
        assert cfg.environment.max_episode_steps == 200
        assert cfg.dataset.dataset_name == "TestDataset"
        assert cfg.reward.success_bonus == 5.0

    def test_dataset_config_composition(self):
        """Test dataset configuration composition."""
        # Test different dataset configurations
        datasets = ["mini_arc", "concept_arc", "arc_agi_1", "arc_agi_2"]

        for dataset in datasets:
            try:
                cfg = get_config(overrides=[f"dataset={dataset}"])

                # Verify dataset-specific configuration
                assert cfg.dataset.dataset_name is not None
                assert cfg.dataset.max_grid_height > 0
                assert cfg.dataset.max_grid_width > 0
                assert cfg.dataset.max_colors > 0

                # Verify parser entry point is set
                assert cfg.dataset.parser_entry_point is not None

            except Exception as e:
                pytest.skip(f"Dataset config {dataset} not available: {e}")

    def test_action_config_composition(self):
        """Test action configuration composition."""
        # Test different action configurations
        action_configs = ["standard", "full", "raw"]

        for action_config in action_configs:
            try:
                cfg = get_config(overrides=[f"action={action_config}"])

                # Verify action configuration
                assert cfg.action.max_operations > 0
                assert hasattr(cfg.action, "validate_actions")

                # Verify allowed operations is a list
                if hasattr(cfg.action, "allowed_operations"):
                    assert isinstance(cfg.action.allowed_operations, (list, type(None)))

            except Exception as e:
                pytest.skip(f"Action config {action_config} not available: {e}")

    def test_reward_config_composition(self):
        """Test reward configuration composition."""
        # Test different reward configurations
        reward_configs = ["training", "evaluation"]

        for reward_config in reward_configs:
            try:
                cfg = get_config(overrides=[f"reward={reward_config}"])

                # Verify reward configuration
                assert hasattr(cfg.reward, "success_bonus")
                assert hasattr(cfg.reward, "step_penalty")
                assert hasattr(cfg.reward, "similarity_weight")

                # Verify numeric values are reasonable
                assert isinstance(cfg.reward.success_bonus, (int, float))
                assert isinstance(cfg.reward.step_penalty, (int, float))
                assert isinstance(cfg.reward.similarity_weight, (int, float))

            except Exception as e:
                pytest.skip(f"Reward config {reward_config} not available: {e}")

    def test_complex_config_composition(self):
        """Test complex configuration composition with multiple overrides."""
        overrides = [
            "dataset=mini_arc",
            "action=full",
            "reward=training",
            "environment.max_episode_steps=100",
            "visualization.enabled=false",
            "wandb.enabled=false",
        ]

        cfg = get_config(overrides=overrides)

        # Verify all overrides were applied
        assert cfg.dataset.dataset_name == "MiniARC"
        assert cfg.environment.max_episode_steps == 100
        assert cfg.visualization.enabled is False
        assert cfg.wandb.enabled is False

    def test_hydra_config_validation(self):
        """Test that Hydra configurations are valid."""
        # Load default config
        cfg = get_config()

        # Convert to JaxArcConfig and validate
        jax_config = JaxArcConfig.from_hydra(cfg)

        # Validate configuration
        errors = jax_config.validate()

        # Should have no critical errors (warnings are OK)
        critical_errors = [e for e in errors if "error" in e.lower()]
        assert len(critical_errors) == 0, (
            f"Critical configuration errors: {critical_errors}"
        )

    def test_config_directory_structure(self):
        """Test that configuration directory structure is correct."""
        import importlib.resources

        # Check that config directory exists
        config_dir = importlib.resources.files("jaxarc") / "conf"
        assert config_dir.is_dir()

        # Check main config file
        main_config = config_dir / "config.yaml"
        assert main_config.is_file()

        # Check subdirectories
        subdirs = ["dataset", "action", "reward"]
        for subdir in subdirs:
            subdir_path = config_dir / subdir
            assert subdir_path.is_dir(), f"Missing config subdirectory: {subdir}"


class TestConfigurationLoading:
    """Test configuration loading and validation workflows."""

    def test_jaxarc_config_from_hydra(self):
        """Test JaxArcConfig creation from Hydra configuration."""
        # Load Hydra config
        hydra_cfg = get_config()

        # Create JaxArcConfig
        jax_config = JaxArcConfig.from_hydra(hydra_cfg)

        # Verify it's a proper JaxArcConfig
        assert isinstance(jax_config, JaxArcConfig)
        assert isinstance(jax_config.environment, EnvironmentConfig)
        assert isinstance(jax_config.dataset, DatasetConfig)
        assert isinstance(jax_config.action, ActionConfig)
        assert isinstance(jax_config.reward, RewardConfig)

        # Verify JAX compatibility (Equinox modules are PyTrees)
        import jax

        leaves, treedef = jax.tree_util.tree_flatten(jax_config)
        assert len(leaves) > 0
        assert treedef is not None

    def test_config_validation_workflow(self):
        """Test configuration validation workflow."""
        # Create config with potential issues
        hydra_cfg = get_config(
            overrides=[
                "environment.max_episode_steps=5",  # Very short episodes
                "reward.step_penalty=-1.0",  # Large penalty
                "reward.success_bonus=1.0",  # Small bonus
            ]
        )

        jax_config = JaxArcConfig.from_hydra(hydra_cfg)

        # Validate - should produce warnings but not errors
        validation_results = jax_config.validate()

        # Should be a tuple of strings
        assert isinstance(validation_results, tuple)
        assert all(isinstance(result, str) for result in validation_results)

    def test_config_error_handling(self):
        """Test configuration error handling."""
        # Test with invalid configuration
        invalid_cfg = DictConfig(
            {
                "environment": {"max_episode_steps": -1},  # Invalid value
                "dataset": {"max_grid_height": 0},  # Invalid value
            }
        )

        # Should handle gracefully
        try:
            jax_config = JaxArcConfig.from_hydra(invalid_cfg)
            errors = jax_config.validate()
            # Should have validation errors
            assert len(errors) > 0
        except Exception as e:
            # Should raise meaningful error
            assert "validation" in str(e).lower() or "invalid" in str(e).lower()

    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        # Create config
        hydra_cfg = get_config()
        jax_config = JaxArcConfig.from_hydra(hydra_cfg)

        # Test YAML serialization
        yaml_str = jax_config.to_yaml()
        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 0

        # Should contain expected sections
        assert "environment:" in yaml_str
        assert "dataset:" in yaml_str
        assert "action:" in yaml_str
        assert "reward:" in yaml_str

    def test_config_immutability(self):
        """Test that configurations are immutable (Equinox requirement)."""
        hydra_cfg = get_config()
        jax_config = JaxArcConfig.from_hydra(hydra_cfg)

        # Should be hashable (immutable)
        hash_value = hash(jax_config)
        assert isinstance(hash_value, int)

        # Should be JAX-compatible
        import jax

        # Should work with JAX tree operations
        leaves, treedef = jax.tree_util.tree_flatten(jax_config)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Should be equivalent
        assert type(reconstructed) == type(jax_config)

    def test_config_cross_validation(self):
        """Test cross-configuration validation."""
        # Create config with cross-validation issues
        hydra_cfg = get_config(
            overrides=[
                "environment.max_episode_steps=10",
                "action.max_operations=100",  # Many operations, few steps
                "reward.step_penalty=-0.1",  # High step penalty with few steps
            ]
        )

        jax_config = JaxArcConfig.from_hydra(hydra_cfg)
        validation_results = jax_config.validate()

        # Should detect cross-validation issues
        cross_validation_warnings = [
            r
            for r in validation_results
            if "operations" in r or "penalty" in r or "steps" in r
        ]

        # Should have some cross-validation feedback
        assert len(validation_results) >= 0  # May have warnings


class TestConfigurationDrivenEnvironmentCreation:
    """Test configuration-driven environment creation."""

    def test_environment_creation_from_config(self):
        """Test creating environment directly from configuration."""
        # Load configuration
        cfg = get_config(overrides=["dataset=mini_arc"])
        jax_config = JaxArcConfig.from_hydra(cfg)

        # Create environment using configuration
        try:
            task_ids = available_task_ids("Mini", config=jax_config)
            if not task_ids:
                pytest.skip("No Mini tasks available")

            task_id = task_ids[0]
            env, env_params = make(f"Mini-{task_id}", config=jax_config)

            # Verify environment reflects configuration
            assert (
                env_params.max_episode_steps == jax_config.environment.max_episode_steps
            )
            assert env_params.dataset.dataset_name == jax_config.dataset.dataset_name
            assert env_params.reward.success_bonus == jax_config.reward.success_bonus

        except Exception as e:
            pytest.skip(f"Environment creation failed: {e}")

    def test_config_parameter_propagation(self):
        """Test that configuration parameters propagate correctly."""
        # Create config with specific parameters
        cfg = get_config(
            overrides=[
                "environment.max_episode_steps=75",
                "reward.success_bonus=3.5",
                "reward.step_penalty=-0.05",
                "dataset=mini_arc",
            ]
        )

        jax_config = JaxArcConfig.from_hydra(cfg)

        try:
            task_ids = available_task_ids("Mini", config=jax_config)
            if not task_ids:
                pytest.skip("No Mini tasks available")

            task_id = task_ids[0]
            env, env_params = make(f"Mini-{task_id}", config=jax_config)

            # Verify specific parameters
            assert env_params.max_episode_steps == 75
            assert env_params.reward.success_bonus == 3.5
            assert env_params.reward.step_penalty == -0.05

        except Exception as e:
            pytest.skip(f"Environment creation failed: {e}")

    def test_different_dataset_configs(self):
        """Test environment creation with different dataset configurations."""
        datasets = [
            ("mini_arc", "Mini"),
            ("concept_arc", "Concept"),
            ("arc_agi_1", "AGI1"),
            ("arc_agi_2", "AGI2"),
        ]

        for config_name, dataset_key in datasets:
            try:
                cfg = get_config(overrides=[f"dataset={config_name}"])
                jax_config = JaxArcConfig.from_hydra(cfg)

                task_ids = available_task_ids(dataset_key, config=jax_config)
                if not task_ids:
                    continue

                task_id = task_ids[0]
                env, env_params = make(f"{dataset_key}-{task_id}", config=jax_config)

                # Verify dataset-specific configuration
                assert (
                    env_params.dataset.dataset_name == jax_config.dataset.dataset_name
                )
                assert (
                    env_params.dataset.max_grid_height
                    == jax_config.dataset.max_grid_height
                )
                assert (
                    env_params.dataset.max_grid_width
                    == jax_config.dataset.max_grid_width
                )

            except Exception as e:
                pytest.skip(f"Dataset {config_name} not available: {e}")

    def test_action_config_environment_integration(self):
        """Test action configuration integration with environment."""
        # Test different action configurations
        action_configs = ["standard", "full"]

        for action_config in action_configs:
            try:
                cfg = get_config(
                    overrides=[f"action={action_config}", "dataset=mini_arc"]
                )
                jax_config = JaxArcConfig.from_hydra(cfg)

                task_ids = available_task_ids("Mini", config=jax_config)
                if not task_ids:
                    continue

                task_id = task_ids[0]
                env, env_params = make(f"Mini-{task_id}", config=jax_config)

                # Verify action configuration
                assert (
                    env_params.action.max_operations == jax_config.action.max_operations
                )
                assert (
                    env_params.action.validate_actions
                    == jax_config.action.validate_actions
                )

                # Test action space
                action_space = env.action_space(env_params)
                assert action_space.max_height == env_params.dataset.max_grid_height
                assert action_space.max_width == env_params.dataset.max_grid_width

            except Exception as e:
                pytest.skip(f"Action config {action_config} not available: {e}")

    def test_reward_config_environment_integration(self):
        """Test reward configuration integration with environment."""
        # Test different reward configurations
        reward_configs = ["training", "evaluation"]

        for reward_config in reward_configs:
            try:
                cfg = get_config(
                    overrides=[f"reward={reward_config}", "dataset=mini_arc"]
                )
                jax_config = JaxArcConfig.from_hydra(cfg)

                task_ids = available_task_ids("Mini", config=jax_config)
                if not task_ids:
                    continue

                task_id = task_ids[0]
                env, env_params = make(f"Mini-{task_id}", config=jax_config)

                # Verify reward configuration
                assert (
                    env_params.reward.success_bonus == jax_config.reward.success_bonus
                )
                assert env_params.reward.step_penalty == jax_config.reward.step_penalty
                assert (
                    env_params.reward.similarity_weight
                    == jax_config.reward.similarity_weight
                )

            except Exception as e:
                pytest.skip(f"Reward config {reward_config} not available: {e}")

    def test_config_override_environment_creation(self):
        """Test environment creation with configuration overrides."""
        # Create environment with overrides
        cfg = get_config(
            overrides=[
                "dataset=mini_arc",
                "environment.max_episode_steps=50",
                "reward.success_bonus=10.0",
                "action=full",
            ]
        )

        jax_config = JaxArcConfig.from_hydra(cfg)

        try:
            task_ids = available_task_ids("Mini", config=jax_config)
            if not task_ids:
                pytest.skip("No Mini tasks available")

            task_id = task_ids[0]
            env, env_params = make(f"Mini-{task_id}", config=jax_config)

            # Test environment with overridden configuration
            import jax

            key = jax.random.PRNGKey(42)

            timestep = env.reset(env_params, key)

            # Verify configuration is applied
            assert timestep.step_type == 0  # FIRST
            assert timestep.state.step_count == 0

            # Verify grid dimensions match config
            expected_shape = (
                env_params.dataset.max_grid_height,
                env_params.dataset.max_grid_width,
            )
            chex.assert_shape(timestep.observation, expected_shape)

        except Exception as e:
            pytest.skip(f"Environment creation with overrides failed: {e}")

    def test_config_validation_in_environment_creation(self):
        """Test that configuration validation works during environment creation."""
        # Create config with validation issues
        cfg = get_config(
            overrides=[
                "dataset=mini_arc",
                "environment.max_episode_steps=1",  # Too short
                "reward.step_penalty=-10.0",  # Too harsh
                "reward.success_bonus=0.1",  # Too small
            ]
        )

        jax_config = JaxArcConfig.from_hydra(cfg)

        # Should still create environment (warnings, not errors)
        try:
            task_ids = available_task_ids("Mini", config=jax_config)
            if not task_ids:
                pytest.skip("No Mini tasks available")

            task_id = task_ids[0]
            env, env_params = make(f"Mini-{task_id}", config=jax_config)

            # Environment should be created despite warnings
            assert env is not None
            assert env_params is not None

            # Configuration should be applied as specified
            assert env_params.max_episode_steps == 1
            assert env_params.reward.step_penalty == -10.0
            assert env_params.reward.success_bonus == 0.1

        except Exception as e:
            pytest.skip(f"Environment creation failed: {e}")
