from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from .action_config import ActionConfig
from .dataset_config import DatasetConfig
from .environment_config import EnvironmentConfig
from .grid_initialization_config import GridInitializationConfig
from .logging_config import LoggingConfig
from .reward_config import RewardConfig
from .storage_config import StorageConfig
from .validation import ConfigValidationError
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
        try:
            hash(self)
        except TypeError as e:
            msg = f"JaxArcConfig must be hashable for JAX compatibility: {e}"
            raise ValueError(msg) from e

    def validate(self) -> tuple[str, ...]:
        """Validate all components and cross-config consistency."""
        all_errors: list[str] = []

        all_errors.extend(self.environment.validate())
        all_errors.extend(self.dataset.validate())
        all_errors.extend(self.action.validate())
        all_errors.extend(self.reward.validate())
        all_errors.extend(self.grid_initialization.validate())
        all_errors.extend(self.visualization.validate())
        all_errors.extend(self.storage.validate())
        all_errors.extend(self.logging.validate())
        all_errors.extend(self.wandb.validate())

        cross_validation_errors = self._validate_cross_config_consistency()
        all_errors.extend(cross_validation_errors)

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

        if self.dataset.max_colors > 10 and self.action.allowed_operations:
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
                "environment": self._config_to_dict(self.environment),
                "dataset": self._config_to_dict(self.dataset),
                "action": self._config_to_dict(self.action),
                "reward": self._config_to_dict(self.reward),
                "visualization": self._config_to_dict(self.visualization),
                "storage": self._config_to_dict(self.storage),
                "logging": self._config_to_dict(self.logging),
                "wandb": self._config_to_dict(self.wandb),
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
        result: dict[str, Any] = {}

        for field_name in getattr(config, "__annotations__", {}):
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                if value is None:
                    result[field_name] = None
                elif isinstance(value, (list, tuple)):
                    result[field_name] = self._serialize_value(list(value))
                elif hasattr(value, "__dict__") and hasattr(value, "_content"):
                    result[field_name] = self._serialize_value(value)
                else:
                    result[field_name] = self._serialize_value(value)

        return result

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
            environment_cfg = EnvironmentConfig.from_hydra(
                hydra_config.get("environment", DictConfig({}))
            )
            dataset_cfg = DatasetConfig.from_hydra(
                hydra_config.get("dataset", DictConfig({}))
            )
            action_cfg = ActionConfig.from_hydra(
                hydra_config.get("action", DictConfig({}))
            )
            reward_cfg = RewardConfig.from_hydra(
                hydra_config.get("reward", DictConfig({}))
            )
            grid_init_cfg = GridInitializationConfig.from_hydra(
                hydra_config.get("grid_initialization", DictConfig({}))
            )
            visualization_cfg = VisualizationConfig.from_hydra(
                hydra_config.get("visualization", DictConfig({}))
            )
            storage_cfg = StorageConfig.from_hydra(
                hydra_config.get("storage", DictConfig({}))
            )
            logging_cfg = LoggingConfig.from_hydra(
                hydra_config.get("logging", DictConfig({}))
            )
            wandb_cfg = WandbConfig.from_hydra(
                hydra_config.get("wandb", DictConfig({}))
            )

            return cls(
                environment=environment_cfg,
                dataset=dataset_cfg,
                action=action_cfg,
                reward=reward_cfg,
                grid_initialization=grid_init_cfg,
                visualization=visualization_cfg,
                storage=storage_cfg,
                logging=logging_cfg,
                wandb=wandb_cfg,
            )
        except Exception as e:
            if isinstance(e, ConfigValidationError):
                raise
            msg = f"Failed to create configuration from Hydra: {e}"
            raise ConfigValidationError(msg) from e
