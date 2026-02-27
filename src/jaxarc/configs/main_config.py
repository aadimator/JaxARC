from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# Import canonical constants to avoid magic numbers
from jaxarc.constants import NUM_COLORS

from .action_config import ActionConfig
from .dataset_config import DatasetConfig
from .environment_config import EnvironmentConfig
from .grid_initialization_config import GridInitializationConfig
from .logging_config import LoggingConfig
from .reward_config import RewardConfig
from .storage_config import StorageConfig
from .validation import ConfigValidationError, check_hashable
from .visualization_config import VisualizationConfig
from .wandb_config import WandbConfig


class JaxArcConfig(eqx.Module):
    """Unified configuration for JaxARC using Equinox.

    Main container that unifies all configuration aspects.
    """

    environment: EnvironmentConfig
    dataset: DatasetConfig
    action: ActionConfig
    reward: RewardConfig
    grid_initialization: GridInitializationConfig
    visualization: VisualizationConfig
    storage: StorageConfig
    logging: LoggingConfig
    wandb: WandbConfig

    def __init__(
        self,
        environment: EnvironmentConfig | None = None,
        dataset: DatasetConfig | None = None,
        action: ActionConfig | None = None,
        reward: RewardConfig | None = None,
        grid_initialization: GridInitializationConfig | None = None,
        visualization: VisualizationConfig | None = None,
        storage: StorageConfig | None = None,
        logging: LoggingConfig | None = None,
        wandb: WandbConfig | None = None,
    ):
        self.environment = environment or EnvironmentConfig()
        self.dataset = dataset or DatasetConfig()
        self.action = action or ActionConfig()
        self.reward = reward or RewardConfig()
        self.grid_initialization = grid_initialization or GridInitializationConfig()
        self.visualization = visualization or VisualizationConfig.from_hydra(
            DictConfig({})
        )
        self.storage = storage or StorageConfig()
        self.logging = logging or LoggingConfig()
        self.wandb = wandb or WandbConfig.from_hydra(DictConfig({}))

    def __check_init__(self):
        check_hashable(self, "JaxArcConfig")

    # Sub-config field names for iteration
    _SUB_CONFIGS = (
        "environment",
        "dataset",
        "action",
        "reward",
        "grid_initialization",
        "visualization",
        "storage",
        "logging",
        "wandb",
    )

    def validate(self) -> tuple[str, ...]:
        """Validate all components and cross-config consistency."""
        all_errors: list[str] = []
        for name in self._SUB_CONFIGS:
            all_errors.extend(getattr(self, name).validate())
        all_errors.extend(self._validate_cross_config_consistency())
        return tuple(all_errors)

    def _validate_cross_config_consistency(self) -> tuple[str, ...]:
        errors: list[str] = []
        warnings: list[str] = []

        try:
            self._validate_debug_level_consistency(warnings)
            self._validate_wandb_consistency(errors, warnings)
            self._validate_action_environment_consistency(warnings)
            self._validate_reward_consistency(warnings)
            self._validate_dataset_consistency(warnings)
            self._validate_logging_consistency(warnings)

            for warning in warnings:
                logger.warning(warning)
        except (ValueError, TypeError, ConfigValidationError) as e:
            errors.append(f"Cross-configuration validation error: {e}")

        return tuple(errors)

    def _validate_debug_level_consistency(self, warnings: list[str]) -> None:
        debug_level = self.environment.debug_level

        if debug_level == "off":
            if self.visualization.enabled:
                warnings.append(
                    "Debug level is 'off' but visualization is enabled - consider disabling visualization for better performance"
                )
            if self.logging.log_operations or self.logging.log_rewards:
                warnings.append(
                    "Debug level is 'off' but detailed logging is enabled - consider reducing log level"
                )

    def _validate_wandb_consistency(
        self, errors: list[str], warnings: list[str]
    ) -> None:
        if self.wandb.enabled:
            if not self.wandb.project_name.strip():
                errors.append("WandB enabled but project_name is empty")

            if getattr(self.logging, "log_level", "INFO") == "ERROR":
                warnings.append(
                    "WandB enabled but log level is ERROR - may miss important metrics"
                )

    def _validate_action_environment_consistency(self, warnings: list[str]) -> None:
        if self.action.max_operations > 50 and self.environment.max_episode_steps < 20:
            warnings.append(
                "Many operations available but few episode steps - may not explore action space effectively"
            )

    def _validate_reward_consistency(self, warnings: list[str]) -> None:
        if abs(self.reward.step_penalty) * self.environment.max_episode_steps > abs(
            self.reward.success_bonus
        ):
            warnings.append(
                "Cumulative step penalties may exceed success bonus - consider adjusting reward balance"
            )

    def _validate_dataset_consistency(self, warnings: list[str]) -> None:
        max_grid_area = self.dataset.max_grid_height * self.dataset.max_grid_width
        if max_grid_area > 400 and self.environment.max_episode_steps < 100:
            warnings.append(
                "Large grids with short episodes may not provide enough time for complex tasks"
            )

        if self.dataset.max_colors > NUM_COLORS and self.action.allowed_operations:
            fill_ops = [op for op in self.action.allowed_operations if 0 <= op <= 9]
            if len(fill_ops) < self.dataset.max_colors:
                warnings.append(
                    f"Dataset allows {self.dataset.max_colors} colors but only {len(fill_ops)} fill operations available"
                )

    def _validate_logging_consistency(self, warnings: list[str]) -> None:
        if getattr(
            self.logging, "structured_logging", False
        ) and self.logging.log_format not in [
            "json",
            "structured",
        ]:
            warnings.append(
                f"Structured logging enabled but format is '{self.logging.log_format}' - consider using 'json' or 'structured'"
            )

        detailed_logging = self.logging.log_operations or self.logging.log_rewards
        if detailed_logging and self.logging.log_level in ["ERROR", "WARNING"]:
            warnings.append(
                "Detailed content logging enabled but log level may suppress the logs"
            )

    def to_yaml(self) -> str:
        try:
            config_dict = {
                name: self._config_to_dict(getattr(self, name))
                for name in self._SUB_CONFIGS
            }
            return yaml.dump(
                config_dict,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                encoding=None,
            )
        except Exception as e:
            msg = f"Failed to export configuration to YAML: {e}"
            raise ConfigValidationError(msg) from e

    def to_yaml_file(self, yaml_path: str | Path) -> None:
        try:
            yaml_path = Path(yaml_path)
            yaml_path.parent.mkdir(parents=True, exist_ok=True)

            yaml_content = self.to_yaml()
            with yaml_path.open("w", encoding="utf-8") as f:
                f.write(yaml_content)
        except Exception as e:
            msg = f"Failed to save configuration to YAML file: {e}"
            raise ConfigValidationError(msg) from e

    def _config_to_dict(self, config: eqx.Module) -> dict[str, Any]:
        return {
            name: self._serialize_value(getattr(config, name))
            for name in getattr(config, "__annotations__", {})
            if hasattr(config, name)
        }

    def _serialize_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if hasattr(value, "_content"):
            try:
                return OmegaConf.to_container(value, resolve=True)
            except (AttributeError, ValueError, TypeError):
                return str(value)
        else:
            return str(value)

    @classmethod
    def from_hydra(cls, hydra_config: DictConfig) -> JaxArcConfig:
        try:
            cfg_classes = {
                "environment": EnvironmentConfig,
                "dataset": DatasetConfig,
                "action": ActionConfig,
                "reward": RewardConfig,
                "grid_initialization": GridInitializationConfig,
                "visualization": VisualizationConfig,
                "storage": StorageConfig,
                "logging": LoggingConfig,
                "wandb": WandbConfig,
            }
            sub_cfgs = {
                name: cfg_cls.from_hydra(hydra_config.get(name, DictConfig({})))
                for name, cfg_cls in cfg_classes.items()
            }
            return cls(**sub_cfgs)
        except ConfigValidationError:
            raise
        except Exception as e:
            msg = f"Failed to create configuration from Hydra: {e}"
            raise ConfigValidationError(msg) from e
