"""Configuration migration utilities for enhanced visualization system."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from .config_composition import create_config_composer


class ConfigMigrator:
    """Handles migration from old to new configuration formats."""

    def __init__(self):
        self.migration_warnings: List[str] = []

    def migrate_legacy_debug_config(self, config: DictConfig) -> DictConfig:
        """Migrate legacy debug configuration to new format.

        Args:
            config: Configuration with legacy debug settings

        Returns:
            Migrated configuration with new visualization settings
        """
        migrated = OmegaConf.create(config)

        # Check for legacy debug settings
        if hasattr(config, "log_rl_steps"):
            if config.log_rl_steps:
                # Migrate to standard debug level
                if not hasattr(migrated, "visualization"):
                    migrated.visualization = OmegaConf.create({})

                if not hasattr(migrated.visualization, "debug_level"):
                    migrated.visualization.debug_level = "standard"
                    self.migration_warnings.append(
                        "Migrated 'log_rl_steps=true' to 'visualization.debug_level=standard'"
                    )
            else:
                # Migrate to off
                if not hasattr(migrated, "visualization"):
                    migrated.visualization = OmegaConf.create({})

                if not hasattr(migrated.visualization, "debug_level"):
                    migrated.visualization.debug_level = "off"
                    self.migration_warnings.append(
                        "Migrated 'log_rl_steps=false' to 'visualization.debug_level=off'"
                    )

        # Migrate output directory settings
        if hasattr(config, "rl_steps_output_dir"):
            if not hasattr(migrated, "visualization"):
                migrated.visualization = OmegaConf.create({})

            if not hasattr(migrated.visualization, "output_dir"):
                migrated.visualization.output_dir = config.rl_steps_output_dir
                self.migration_warnings.append(
                    "Migrated 'rl_steps_output_dir' to 'visualization.output_dir'"
                )

        # Migrate clear output directory setting
        if hasattr(config, "clear_output_dir"):
            if not hasattr(migrated, "storage"):
                migrated.storage = OmegaConf.create({})

            if config.clear_output_dir:
                migrated.storage.cleanup_policy = "size_based"
                migrated.storage.auto_cleanup = True
            else:
                migrated.storage.cleanup_policy = "manual"
                migrated.storage.auto_cleanup = False

            self.migration_warnings.append(
                "Migrated 'clear_output_dir' to storage cleanup settings"
            )

        return migrated

    def detect_legacy_config(self, config: DictConfig) -> bool:
        """Detect if configuration uses legacy format.

        Args:
            config: Configuration to check

        Returns:
            True if legacy format detected
        """
        legacy_fields = ["log_rl_steps", "rl_steps_output_dir", "clear_output_dir"]

        return any(hasattr(config, field) for field in legacy_fields)

    def suggest_migration(self, config: DictConfig) -> Dict[str, Any]:
        """Suggest migration steps for legacy configuration.

        Args:
            config: Legacy configuration

        Returns:
            Dictionary with migration suggestions
        """
        suggestions = {
            "detected_legacy_fields": [],
            "recommended_actions": [],
            "new_config_structure": {},
        }

        # Detect legacy fields
        if hasattr(config, "log_rl_steps"):
            suggestions["detected_legacy_fields"].append("log_rl_steps")
            if config.log_rl_steps:
                suggestions["recommended_actions"].append(
                    "Replace 'log_rl_steps: true' with 'debug: standard' in defaults"
                )
                suggestions["new_config_structure"]["visualization"] = {
                    "debug_level": "standard"
                }
            else:
                suggestions["recommended_actions"].append(
                    "Replace 'log_rl_steps: false' with 'debug: off' in defaults"
                )
                suggestions["new_config_structure"]["visualization"] = {
                    "debug_level": "off"
                }

        if hasattr(config, "rl_steps_output_dir"):
            suggestions["detected_legacy_fields"].append("rl_steps_output_dir")
            suggestions["recommended_actions"].append(
                "Move 'rl_steps_output_dir' to 'visualization.output_dir'"
            )
            if "visualization" not in suggestions["new_config_structure"]:
                suggestions["new_config_structure"]["visualization"] = {}
            suggestions["new_config_structure"]["visualization"]["output_dir"] = (
                config.rl_steps_output_dir
            )

        if hasattr(config, "clear_output_dir"):
            suggestions["detected_legacy_fields"].append("clear_output_dir")
            suggestions["recommended_actions"].append(
                "Replace 'clear_output_dir' with storage cleanup policy"
            )
            suggestions["new_config_structure"]["storage"] = {
                "cleanup_policy": "size_based" if config.clear_output_dir else "manual",
                "auto_cleanup": bool(config.clear_output_dir),
            }

        return suggestions

    def create_migration_guide(self, config: DictConfig) -> str:
        """Create a migration guide for legacy configuration.

        Args:
            config: Legacy configuration

        Returns:
            Formatted migration guide text
        """
        if not self.detect_legacy_config(config):
            return "Configuration is already using the new format."

        suggestions = self.suggest_migration(config)

        guide_lines = [
            "Configuration Migration Guide",
            "=" * 30,
            "",
            "Legacy fields detected:",
        ]

        for field in suggestions["detected_legacy_fields"]:
            guide_lines.append(f"  - {field}")

        guide_lines.extend(
            [
                "",
                "Recommended actions:",
            ]
        )

        for action in suggestions["recommended_actions"]:
            guide_lines.append(f"  - {action}")

        guide_lines.extend(["", "New configuration structure:", "```yaml"])

        new_config_yaml = OmegaConf.to_yaml(
            OmegaConf.create(suggestions["new_config_structure"])
        )
        guide_lines.append(new_config_yaml.rstrip())

        guide_lines.extend(
            [
                "```",
                "",
                "Migration steps:",
                "1. Update your config file with the new structure above",
                "2. Remove the legacy fields",
                "3. Test the new configuration with: jaxarc-config validate",
                "4. Update any scripts that reference the old field names",
            ]
        )

        return "\n".join(guide_lines)

    def auto_migrate_config_file(
        self, input_file: Path, output_file: Optional[Path] = None, backup: bool = True
    ) -> Path:
        """Automatically migrate a configuration file.

        Args:
            input_file: Path to legacy configuration file
            output_file: Path for migrated file (default: same as input)
            backup: Whether to create backup of original file

        Returns:
            Path to migrated configuration file
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {input_file}")

        # Load legacy config
        legacy_config = OmegaConf.load(input_file)

        if not self.detect_legacy_config(legacy_config):
            warnings.warn(f"No legacy fields detected in {input_file}")
            return input_file

        # Create backup if requested
        if backup:
            backup_file = input_file.with_suffix(f"{input_file.suffix}.backup")
            backup_file.write_text(input_file.read_text())

        # Migrate configuration
        migrated_config = self.migrate_legacy_debug_config(legacy_config)

        # Determine output file
        if output_file is None:
            output_file = input_file

        # Save migrated config
        OmegaConf.save(migrated_config, output_file)

        # Print warnings
        if self.migration_warnings:
            print("Migration completed with the following changes:")
            for warning in self.migration_warnings:
                print(f"  - {warning}")

        return output_file

    def get_migration_warnings(self) -> List[str]:
        """Get list of migration warnings."""
        return self.migration_warnings.copy()

    def clear_warnings(self) -> None:
        """Clear migration warnings."""
        self.migration_warnings.clear()


def migrate_legacy_config(config: DictConfig) -> DictConfig:
    """Convenience function to migrate legacy configuration.

    Args:
        config: Legacy configuration

    Returns:
        Migrated configuration
    """
    migrator = ConfigMigrator()
    return migrator.migrate_legacy_debug_config(config)


def check_config_compatibility(config: DictConfig) -> Dict[str, Any]:
    """Check configuration compatibility with all dataset/action/environment configs.

    Args:
        config: Configuration to check

    Returns:
        Compatibility report
    """
    composer = create_config_composer()

    try:
        # Get available configs
        available = composer.get_available_configs()

        compatibility_report = {
            "compatible_combinations": [],
            "incompatible_combinations": [],
            "warnings": [],
        }

        # Test combinations with different datasets
        for dataset in available.get("dataset", []):
            for action in available.get("action", []):
                for environment in available.get("environment", []):
                    try:
                        # Try to compose with this combination
                        test_overrides = [
                            f"dataset={dataset}",
                            f"action={action}",
                            f"environment={environment}",
                        ]

                        test_config = composer.compose_config(overrides=test_overrides)

                        # Merge with our config
                        merged_config = OmegaConf.merge(test_config, config)

                        # Try to validate
                        from .config_validation import validate_config

                        errors = validate_config(merged_config)

                        if not errors:
                            compatibility_report["compatible_combinations"].append(
                                {
                                    "dataset": dataset,
                                    "action": action,
                                    "environment": environment,
                                }
                            )
                        else:
                            compatibility_report["incompatible_combinations"].append(
                                {
                                    "dataset": dataset,
                                    "action": action,
                                    "environment": environment,
                                    "errors": [error.message for error in errors],
                                }
                            )

                    except Exception as e:
                        compatibility_report["incompatible_combinations"].append(
                            {
                                "dataset": dataset,
                                "action": action,
                                "environment": environment,
                                "errors": [str(e)],
                            }
                        )

        return compatibility_report

    finally:
        composer.cleanup()


def create_config_documentation() -> str:
    """Create documentation for the new configuration system.

    Returns:
        Formatted documentation text
    """
    doc_lines = [
        "Enhanced Visualization Configuration Documentation",
        "=" * 50,
        "",
        "## Overview",
        "",
        "The enhanced visualization system provides hierarchical configuration",
        "through Hydra with support for multiple debug levels, storage policies,",
        "and logging integrations.",
        "",
        "## Configuration Structure",
        "",
        "```",
        "conf/",
        "├── visualization/          # Debug level configurations",
        "│   ├── debug_off.yaml     # No visualization",
        "│   ├── debug_minimal.yaml # Episode summaries only",
        "│   ├── debug_standard.yaml# Key steps and changes",
        "│   ├── debug_verbose.yaml # All steps and actions",
        "│   └── debug_full.yaml    # Complete state dumps",
        "├── storage/               # Storage configurations",
        "│   ├── development.yaml   # Dev-friendly settings",
        "│   ├── research.yaml      # Research-optimized",
        "│   └── production.yaml    # Production-safe",
        "└── logging/               # Logging configurations",
        "    ├── local_only.yaml    # Local file logging",
        "    ├── wandb_basic.yaml   # Basic wandb integration",
        "    └── wandb_full.yaml    # Full wandb logging",
        "```",
        "",
        "## Usage Examples",
        "",
        "### Basic Usage",
        "```bash",
        "# Use standard debug level with development storage",
        "python script.py debug=on",
        "",
        "# Use verbose debug with research storage and wandb",
        "python script.py debug=verbose logging=wandb_basic",
        "```",
        "",
        "### Advanced Configuration",
        "```bash",
        "# Override specific settings",
        "python script.py debug=standard \\",
        "  visualization.show_coordinates=true \\",
        "  storage.max_storage_gb=10.0 \\",
        "  wandb.enabled=true",
        "```",
        "",
        "## Migration from Legacy Configuration",
        "",
        "Old format:",
        "```yaml",
        "log_rl_steps: true",
        "rl_steps_output_dir: outputs/debug",
        "clear_output_dir: true",
        "```",
        "",
        "New format:",
        "```yaml",
        "defaults:",
        "  - debug: standard  # or minimal, verbose, full",
        "```",
        "",
        "## Configuration Validation",
        "",
        "Use the configuration CLI to validate settings:",
        "```bash",
        "# Validate composed configuration",
        "python -m jaxarc.utils.visualization.config_cli validate",
        "",
        "# Check specific combination",
        "python -m jaxarc.utils.visualization.config_cli check-compatibility \\",
        "  standard research wandb_basic",
        "```",
        "",
        "## Best Practices",
        "",
        "1. **Development**: Use `debug=minimal` or `debug=standard` with `storage=development`",
        "2. **Research**: Use `debug=verbose` with `storage=research` and `logging=wandb_basic`",
        "3. **Production**: Use `debug=off` or `debug=minimal` with `storage=production`",
        "4. **Performance**: Monitor storage usage with verbose/full debug levels",
        "5. **Wandb**: Enable wandb logging for experiment tracking and collaboration",
    ]

    return "\n".join(doc_lines)
