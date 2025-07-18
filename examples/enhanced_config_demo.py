#!/usr/bin/env python3
"""
Enhanced Configuration System Demo

This script demonstrates the new enhanced visualization configuration system
with Hydra integration, validation, and migration utilities.
"""

import sys
from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jaxarc.utils.visualization.config_validation import validate_config, format_validation_errors
from jaxarc.utils.visualization.config_composition import quick_compose, get_config_help
from jaxarc.utils.visualization.config_migration import (
    ConfigMigrator, 
    migrate_legacy_config,
    check_config_compatibility,
    create_config_documentation
)

console = Console()


def demo_config_composition():
    """Demonstrate configuration composition."""
    console.print(Panel("[bold blue]Configuration Composition Demo[/bold blue]"))
    
    # Example 1: Quick compose with defaults
    console.print("\n[yellow]Example 1: Quick compose with defaults[/yellow]")
    config1 = quick_compose(debug_level="standard")
    console.print(f"Debug level: {config1.visualization.debug_level}")
    console.print(f"Storage policy: {config1.storage.cleanup_policy}")
    console.print(f"Wandb enabled: {config1.wandb.enabled}")
    
    # Example 2: Research configuration
    console.print("\n[yellow]Example 2: Research configuration[/yellow]")
    config2 = quick_compose(
        debug_level="verbose",
        storage_type="research", 
        logging_type="wandb_basic"
    )
    console.print(f"Debug level: {config2.visualization.debug_level}")
    console.print(f"Max storage: {config2.storage.max_storage_gb} GB")
    console.print(f"Wandb project: {config2.wandb.project_name}")
    
    # Example 3: Custom overrides
    console.print("\n[yellow]Example 3: Custom overrides[/yellow]")
    config3 = quick_compose(
        debug_level="standard",
        overrides=[
            "visualization.show_coordinates=true",
            "storage.max_storage_gb=15.0",
            "wandb.enabled=true"
        ]
    )
    console.print(f"Show coordinates: {config3.visualization.show_coordinates}")
    console.print(f"Max storage: {config3.storage.max_storage_gb} GB")
    console.print(f"Wandb enabled: {config3.wandb.enabled}")


def demo_config_validation():
    """Demonstrate configuration validation."""
    console.print(Panel("[bold blue]Configuration Validation Demo[/bold blue]"))
    
    # Example 1: Valid configuration
    console.print("\n[yellow]Example 1: Valid configuration[/yellow]")
    valid_config = quick_compose(debug_level="standard", validate=False)
    errors = validate_config(valid_config)
    if not errors:
        console.print("[green]✓ Configuration is valid![/green]")
    else:
        console.print("[red]✗ Unexpected validation errors[/red]")
    
    # Example 2: Invalid configuration
    console.print("\n[yellow]Example 2: Invalid configuration[/yellow]")
    invalid_config = OmegaConf.create({
        "visualization": {
            "debug_level": "invalid_level",
            "output_formats": ["invalid_format"],
            "memory_limit_mb": -100
        },
        "storage": {
            "cleanup_policy": "invalid_policy",
            "max_storage_gb": "not_a_number"
        }
    })
    
    errors = validate_config(invalid_config)
    if errors:
        console.print("[red]Found validation errors (as expected):[/red]")
        console.print(format_validation_errors(errors))


def demo_legacy_migration():
    """Demonstrate legacy configuration migration."""
    console.print(Panel("[bold blue]Legacy Configuration Migration Demo[/bold blue]"))
    
    # Create a legacy configuration
    legacy_config = OmegaConf.create({
        "log_rl_steps": True,
        "rl_steps_output_dir": "outputs/old_debug",
        "clear_output_dir": True,
        "dataset": "arc_agi_1",
        "seed": 42
    })
    
    console.print("\n[yellow]Legacy configuration:[/yellow]")
    legacy_yaml = OmegaConf.to_yaml(legacy_config)
    syntax = Syntax(legacy_yaml, "yaml", theme="monokai")
    console.print(syntax)
    
    # Migrate the configuration
    migrator = ConfigMigrator()
    migrated_config = migrator.migrate_legacy_debug_config(legacy_config)
    
    console.print("\n[yellow]Migrated configuration:[/yellow]")
    migrated_yaml = OmegaConf.to_yaml(migrated_config)
    syntax = Syntax(migrated_yaml, "yaml", theme="monokai")
    console.print(syntax)
    
    # Show migration warnings
    warnings = migrator.get_migration_warnings()
    if warnings:
        console.print("\n[yellow]Migration warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  - {warning}")
    
    # Show migration guide
    console.print("\n[yellow]Migration guide:[/yellow]")
    guide = migrator.create_migration_guide(legacy_config)
    console.print(Panel(guide, title="Migration Guide"))


def demo_compatibility_checking():
    """Demonstrate configuration compatibility checking."""
    console.print(Panel("[bold blue]Configuration Compatibility Demo[/bold blue]"))
    
    # Test different debug levels with storage constraints
    test_cases = [
        ("minimal", "development"),
        ("standard", "development"), 
        ("verbose", "development"),  # Should warn about storage
        ("full", "research"),
        ("verbose", "production")
    ]
    
    table = Table(title="Compatibility Test Results")
    table.add_column("Debug Level", style="cyan")
    table.add_column("Storage Type", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Notes", style="yellow")
    
    for debug_level, storage_type in test_cases:
        try:
            config = quick_compose(
                debug_level=debug_level,
                storage_type=storage_type,
                validate=True
            )
            status = "✓ Compatible"
            notes = "No issues"
        except Exception as e:
            status = "✗ Issues"
            notes = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
        
        table.add_row(debug_level, storage_type, status, notes)
    
    console.print(table)


def demo_hydra_integration():
    """Demonstrate Hydra integration."""
    console.print(Panel("[bold blue]Hydra Integration Demo[/bold blue]"))
    
    console.print("\n[yellow]Available configuration options:[/yellow]")
    help_text = get_config_help()
    console.print(Panel(help_text, title="Configuration Help"))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main demo function with Hydra integration."""
    console.print(Panel(
        "[bold green]Enhanced Visualization Configuration System Demo[/bold green]",
        subtitle="Demonstrating new configuration capabilities"
    ))
    
    # Show current configuration
    console.print("\n[yellow]Current Hydra configuration:[/yellow]")
    current_yaml = OmegaConf.to_yaml(cfg)
    syntax = Syntax(current_yaml, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Current Configuration"))
    
    # Run demos
    try:
        demo_config_composition()
        demo_config_validation()
        demo_legacy_migration()
        demo_compatibility_checking()
        demo_hydra_integration()
        
        console.print(Panel(
            "[bold green]All demos completed successfully![/bold green]\n\n"
            "Key takeaways:\n"
            "• Use quick_compose() for programmatic configuration\n"
            "• Validate configurations before use\n"
            "• Migrate legacy configs with ConfigMigrator\n"
            "• Check compatibility between different settings\n"
            "• Use Hydra overrides for flexible configuration",
            title="Demo Summary"
        ))
        
    except Exception as e:
        console.print(f"[red]Demo failed with error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()