"""Command-line interface for visualization configuration management."""

import sys
from pathlib import Path
from typing import List, Optional

import typer
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from .config_composition import create_config_composer, quick_compose, get_config_help
from .config_validation import validate_config, format_validation_errors

app = typer.Typer(help="JaxARC Visualization Configuration Management")
console = Console()


@app.command()
def validate(
    config_file: Optional[Path] = typer.Argument(None, help="Path to config file to validate"),
    config_name: str = typer.Option("config", help="Config name to compose and validate"),
    overrides: Optional[List[str]] = typer.Option(None, "--override", "-o", help="Configuration overrides")
) -> None:
    """Validate a configuration file or composed configuration."""
    
    try:
        if config_file:
            # Validate existing file
            if not config_file.exists():
                console.print(f"[red]Error: Config file not found: {config_file}[/red]")
                raise typer.Exit(1)
            
            config = OmegaConf.load(config_file)
            console.print(f"[blue]Validating config file: {config_file}[/blue]")
        
        else:
            # Compose and validate
            composer = create_config_composer()
            try:
                config = composer.compose_config(config_name=config_name, overrides=overrides or [])
                console.print(f"[blue]Validating composed config: {config_name}[/blue]")
            finally:
                composer.cleanup()
        
        # Perform validation
        errors = validate_config(config)
        
        if not errors:
            console.print("[green]✓ Configuration is valid![/green]")
        else:
            console.print("[red]✗ Configuration validation failed:[/red]")
            console.print(format_validation_errors(errors))
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def compose(
    config_name: str = typer.Option("config", help="Base config name"),
    debug_level: str = typer.Option("standard", help="Debug level (off, minimal, standard, verbose, full)"),
    storage_type: str = typer.Option("development", help="Storage type (development, research, production)"),
    logging_type: str = typer.Option("local_only", help="Logging type (local_only, wandb_basic, wandb_full)"),
    overrides: Optional[List[str]] = typer.Option(None, "--override", "-o", help="Additional overrides"),
    output: Optional[Path] = typer.Option(None, "--output", "-f", help="Save composed config to file"),
    validate_config: bool = typer.Option(True, "--validate/--no-validate", help="Validate composed config")
) -> None:
    """Compose a configuration with specified options."""
    
    try:
        config = quick_compose(
            debug_level=debug_level,
            storage_type=storage_type,
            logging_type=logging_type,
            overrides=overrides or [],
            validate=validate_config
        )
        
        if output:
            # Save to file
            OmegaConf.save(config, output)
            console.print(f"[green]Configuration saved to: {output}[/green]")
        else:
            # Print to console
            yaml_str = OmegaConf.to_yaml(config)
            syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="Composed Configuration"))
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_configs() -> None:
    """List available configuration options."""
    
    try:
        composer = create_config_composer()
        try:
            available = composer.get_available_configs()
        finally:
            composer.cleanup()
        
        table = Table(title="Available Configuration Options")
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Available Configs", style="magenta")
        
        for category, configs in available.items():
            if configs:
                table.add_row(category.title(), ", ".join(sorted(configs)))
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def help_overrides() -> None:
    """Show help for configuration overrides."""
    
    help_text = get_config_help()
    console.print(Panel(help_text, title="Configuration Help"))


@app.command()
def check_compatibility(
    debug_level: str = typer.Argument(..., help="Debug level to check"),
    storage_type: str = typer.Argument(..., help="Storage type to check"),
    logging_type: str = typer.Argument(..., help="Logging type to check")
) -> None:
    """Check compatibility between configuration options."""
    
    try:
        config = quick_compose(
            debug_level=debug_level,
            storage_type=storage_type,
            logging_type=logging_type,
            validate=True
        )
        
        console.print("[green]✓ Configuration combination is compatible![/green]")
        
        # Show some key settings
        table = Table(title="Key Configuration Settings")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="magenta")
        
        if hasattr(config, 'visualization'):
            table.add_row("Debug Level", config.visualization.debug_level)
            if hasattr(config.visualization, 'output_formats'):
                table.add_row("Output Formats", ", ".join(config.visualization.output_formats))
        
        if hasattr(config, 'storage'):
            table.add_row("Storage Policy", config.storage.cleanup_policy)
            table.add_row("Max Storage", f"{config.storage.max_storage_gb} GB")
        
        if hasattr(config, 'wandb'):
            table.add_row("Wandb Enabled", str(config.wandb.enabled))
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[red]✗ Configuration incompatible: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def create_preset(
    name: str = typer.Argument(..., help="Name for the preset"),
    debug_level: str = typer.Option("standard", help="Debug level"),
    storage_type: str = typer.Option("development", help="Storage type"),
    logging_type: str = typer.Option("local_only", help="Logging type"),
    overrides: Optional[List[str]] = typer.Option(None, "--override", "-o", help="Additional overrides"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory (default: conf/presets)")
) -> None:
    """Create a preset configuration file."""
    
    try:
        # Determine output directory
        if output_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent.parent / "conf"
            output_dir = config_dir / "presets"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{name}.yaml"
        
        # Compose configuration
        config = quick_compose(
            debug_level=debug_level,
            storage_type=storage_type,
            logging_type=logging_type,
            overrides=overrides or [],
            validate=True
        )
        
        # Add preset metadata
        preset_config = {
            "_preset_info": {
                "name": name,
                "description": f"Preset with {debug_level} debug, {storage_type} storage, {logging_type} logging",
                "debug_level": debug_level,
                "storage_type": storage_type,
                "logging_type": logging_type
            }
        }
        
        # Merge with composed config
        final_config = OmegaConf.merge(OmegaConf.create(preset_config), config)
        
        # Save preset
        OmegaConf.save(final_config, output_file)
        console.print(f"[green]Preset '{name}' created: {output_file}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error creating preset: {e}[/red]")
        raise typer.Exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()