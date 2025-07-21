"""Hydra configuration composition utilities for visualization system."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .config_validation import format_validation_errors, validate_config


class ConfigComposer:
    """Handles Hydra configuration composition for visualization system."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize config composer.

        Args:
            config_dir: Path to configuration directory. If None, uses default.
        """
        if config_dir is None:
            # Default to project conf directory
            config_dir = Path(__file__).parent.parent.parent.parent.parent / "conf"

        self.config_dir = Path(config_dir).resolve()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure Hydra is initialized with our config directory."""
        if not self._initialized:
            # Clear any existing Hydra instance
            GlobalHydra.instance().clear()

            # Initialize with our config directory
            initialize_config_dir(config_dir=str(self.config_dir), version_base=None)
            self._initialized = True

    def compose_config(
        self,
        config_name: str = "config",
        overrides: Optional[List[str]] = None,
        return_hydra_config: bool = False,
    ) -> DictConfig:
        """Compose configuration with optional overrides.

        Args:
            config_name: Name of the main config file (without .yaml)
            overrides: List of override strings (e.g., ["debug=on", "visualization.debug_level=verbose"])
            return_hydra_config: Whether to return the full hydra config

        Returns:
            Composed configuration
        """
        self._ensure_initialized()

        if overrides is None:
            overrides = []

        try:
            config = compose(
                config_name=config_name,
                overrides=overrides,
                return_hydra_config=return_hydra_config,
            )
            return config
        except Exception as e:
            raise ValueError(f"Failed to compose configuration: {e}")

    def compose_visualization_config(
        self,
        debug_level: str = "standard",
        storage_type: str = "development",
        logging_type: str = "local_only",
        overrides: Optional[List[str]] = None,
    ) -> DictConfig:
        """Compose configuration with visualization-specific defaults.

        Args:
            debug_level: Visualization debug level (off, minimal, standard, verbose, full)
            storage_type: Storage configuration type (development, research, production)
            logging_type: Logging configuration type (local_only, wandb_basic, wandb_full)
            overrides: Additional override strings

        Returns:
            Composed configuration with visualization settings
        """
        # Map debug levels to actual config names
        debug_level_map = {
            "off": "debug_off",
            "minimal": "debug_minimal",
            "standard": "debug_standard",
            "verbose": "debug_verbose",
            "full": "debug_full",
        }

        # Get the actual config name
        vis_config_name = debug_level_map.get(debug_level, f"debug_{debug_level}")

        # Load individual config files manually to avoid conflicts
        vis_config_path = self.config_dir / "visualization" / f"{vis_config_name}.yaml"
        storage_config_path = self.config_dir / "storage" / f"{storage_type}.yaml"
        logging_config_path = self.config_dir / "logging" / f"{logging_type}.yaml"

        # Load configs
        configs_to_merge = []

        if vis_config_path.exists():
            vis_config = OmegaConf.load(vis_config_path)
            configs_to_merge.append({"visualization": vis_config})

        if storage_config_path.exists():
            storage_config = OmegaConf.load(storage_config_path)
            configs_to_merge.append({"storage": storage_config})

        if logging_config_path.exists():
            logging_config = OmegaConf.load(logging_config_path)
            # Merge the logging config directly (it contains multiple sections)
            configs_to_merge.append(logging_config)

        # Start with empty config
        merged_config = OmegaConf.create({})

        # Merge all configs
        for config in configs_to_merge:
            merged_config = OmegaConf.merge(merged_config, config)

        # Apply overrides
        if overrides:
            for override in overrides:
                if "=" in override:
                    key, value = override.split("=", 1)
                    # Parse value
                    try:
                        # Try to parse as YAML value
                        parsed_value = OmegaConf.create({key: value})[key]
                    except:
                        # Fall back to string
                        parsed_value = value

                    # Set the value
                    OmegaConf.set(merged_config, key, parsed_value)

        return merged_config

    def validate_composed_config(self, config: DictConfig) -> DictConfig:
        """Validate a composed configuration and return it if valid.

        Args:
            config: Configuration to validate

        Returns:
            The same configuration if valid

        Raises:
            ValueError: If configuration is invalid
        """
        errors = validate_config(config)
        if errors:
            error_message = format_validation_errors(errors)
            raise ValueError(f"Invalid composed configuration:\n{error_message}")

        return config

    def compose_and_validate(
        self, config_name: str = "config", overrides: Optional[List[str]] = None
    ) -> DictConfig:
        """Compose and validate configuration in one step.

        Args:
            config_name: Name of the main config file
            overrides: List of override strings

        Returns:
            Validated composed configuration
        """
        config = self.compose_config(config_name=config_name, overrides=overrides)
        return self.validate_composed_config(config)

    def get_available_configs(self) -> Dict[str, List[str]]:
        """Get available configuration options for each category.

        Returns:
            Dictionary mapping category names to available config files
        """
        categories = {
            "visualization": [],
            "storage": [],
            "logging": [],
            "debug": [],
            "dataset": [],
            "action": [],
            "environment": [],
            "reward": [],
        }

        for category in categories:
            category_dir = self.config_dir / category
            if category_dir.exists():
                for config_file in category_dir.glob("*.yaml"):
                    categories[category].append(config_file.stem)

        return categories

    def create_override_suggestions(
        self, field_path: str, current_value: Any
    ) -> List[str]:
        """Create override suggestions for a given field.

        Args:
            field_path: Dot-separated path to the field (e.g., "visualization.debug_level")
            current_value: Current value of the field

        Returns:
            List of suggested override strings
        """
        suggestions = []

        # Common visualization overrides
        if "visualization" in field_path:
            if "debug_level" in field_path:
                levels = ["off", "minimal", "standard", "verbose", "full"]
                for level in levels:
                    if level != current_value:
                        suggestions.append(f"{field_path}={level}")

            elif "output_formats" in field_path:
                formats = [["svg"], ["png"], ["svg", "png"], ["svg", "png", "html"]]
                for fmt in formats:
                    if fmt != current_value:
                        suggestions.append(f"{field_path}=[{','.join(fmt)}]")

        # Storage overrides
        elif "storage" in field_path:
            if "cleanup_policy" in field_path:
                policies = ["oldest_first", "size_based", "manual"]
                for policy in policies:
                    if policy != current_value:
                        suggestions.append(f"{field_path}={policy}")

        # Wandb overrides
        elif "wandb" in field_path:
            if "enabled" in field_path:
                suggestions.append(f"{field_path}={not current_value}")
            elif "log_frequency" in field_path:
                frequencies = [1, 5, 10, 20, 50]
                for freq in frequencies:
                    if freq != current_value:
                        suggestions.append(f"{field_path}={freq}")

        return suggestions

    def cleanup(self) -> None:
        """Clean up Hydra global state."""
        if self._initialized:
            GlobalHydra.instance().clear()
            self._initialized = False


def create_config_composer(
    config_dir: Optional[Union[str, Path]] = None,
) -> ConfigComposer:
    """Create a configuration composer instance.

    Args:
        config_dir: Path to configuration directory

    Returns:
        ConfigComposer instance
    """
    return ConfigComposer(config_dir=config_dir)


def quick_compose(
    debug_level: str = "standard",
    storage_type: str = "development",
    logging_type: str = "local_only",
    overrides: Optional[List[str]] = None,
    validate: bool = True,
) -> DictConfig:
    """Quick configuration composition with common defaults.

    Args:
        debug_level: Visualization debug level
        storage_type: Storage configuration type
        logging_type: Logging configuration type
        overrides: Additional overrides
        validate: Whether to validate the composed config

    Returns:
        Composed (and optionally validated) configuration
    """
    composer = create_config_composer()

    try:
        config = composer.compose_visualization_config(
            debug_level=debug_level,
            storage_type=storage_type,
            logging_type=logging_type,
            overrides=overrides,
        )

        if validate:
            config = composer.validate_composed_config(config)

        return config

    finally:
        composer.cleanup()


def get_config_help() -> str:
    """Get help text for configuration options.

    Returns:
        Formatted help text
    """
    composer = create_config_composer()

    try:
        available = composer.get_available_configs()

        help_lines = ["Available Configuration Options:", "=" * 35, ""]

        for category, configs in available.items():
            if configs:
                help_lines.append(f"{category.title()}:")
                for config in sorted(configs):
                    help_lines.append(f"  - {config}")
                help_lines.append("")

        help_lines.extend(
            [
                "Example Usage:",
                "  # Basic composition",
                "  config = quick_compose(debug_level='verbose', storage_type='research')",
                "",
                "  # With overrides",
                "  config = quick_compose(",
                "      debug_level='standard',",
                "      overrides=['wandb.enabled=true', 'visualization.show_coordinates=true']",
                "  )",
                "",
                "Override Format:",
                "  - Simple values: 'field=value'",
                "  - Nested fields: 'section.field=value'",
                "  - Lists: 'field=[item1,item2]'",
                "  - Booleans: 'field=true' or 'field=false'",
            ]
        )

        return "\n".join(help_lines)

    finally:
        composer.cleanup()
