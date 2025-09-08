#!/usr/bin/env python3
"""
ARCLE vs JaxARC Performance Benchmark

Comprehensive benchmarking system comparing ARCLE (Gymnasium-based) against JaxARC (JAX-based)
performance following NAVIX's proven benchmarking patterns. Provides two key scaling analyses:
1. Performance scaling with number of timesteps (like NAVIX speed benchmark)
2. Throughput scaling with number of parallel environments (like NAVIX throughput benchmark)

Usage:
    pixi run -e bench python benchmarks/arcle_vs_jaxarc.py
    pixi run -e bench python benchmarks/arcle_vs_jaxarc.py --timestep-powers "1,2,3,4" --num-runs 3
    pixi run -e bench python benchmarks/arcle_vs_jaxarc.py --batch-powers "0,1,2,3,4" --fixed-steps 500
"""

import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import platform
import sys

# Core libraries
import jax
import jax.numpy as jnp
import numpy as np

# Statistical analysis
import scipy.stats as stats
from scipy import __version__ as scipy_version

# Visualization
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# CLI interface
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

# JaxARC imports
try:
    from jaxarc import JaxArcConfig
    from jaxarc.registration import make, available_task_ids
    from jaxarc.envs.action_wrappers import BboxActionWrapper

    JAXARC_AVAILABLE = True
except ImportError as e:
    JAXARC_AVAILABLE = False
    JAXARC_IMPORT_ERROR = str(e)

# ARCLE imports (optional)
try:
    import gymnasium
    from arcle import MiniARCLoader

    ARCLE_AVAILABLE = True
except ImportError as e:
    ARCLE_AVAILABLE = False
    ARCLE_IMPORT_ERROR = str(e)

# Initialize console for rich output
console = Console()

# CLI app
app = typer.Typer(
    name="arcle-vs-jaxarc-benchmark",
    help="Comprehensive benchmark comparing ARCLE vs JaxARC performance",
    add_completion=False,
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    timestep_powers: List[int]  # Powers of 10 for timestep scaling
    batch_powers: List[int]  # Powers of 2 for throughput scaling
    num_runs: int  # Number of runs per configuration
    fixed_steps: int  # Fixed steps for throughput benchmark
    output_dir: Path  # Output directory
    save_plots: bool  # Generate matplotlib plots
    save_json: bool  # Save detailed JSON results
    benchmark_type: str  # "timestep", "throughput", or "both"


@dataclass
class TimingResult:
    """Results from timing measurements."""

    mean_time: float  # Mean execution time
    std_time: float  # Standard deviation
    mean_sps: float  # Mean steps per second
    std_sps: float  # SPS standard deviation
    compilation_time: float  # JIT compilation time (JaxARC only)
    raw_times: List[float]  # Raw timing data
    raw_sps: List[float]  # Raw SPS data


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    timestep_scaling: Dict[int, Dict[str, TimingResult]]
    throughput_scaling: Dict[int, Dict[str, TimingResult]]
    system_info: Dict  # System information
    config: BenchmarkConfig  # Benchmark configuration
    timestamp: str  # Execution timestamp


def get_system_info() -> Dict[str, Any]:
    """Collect comprehensive system information."""
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "matplotlib_version": matplotlib.__version__,
    }

    # JAX device information
    try:
        devices = jax.devices()
        info["jax_devices"] = [str(device) for device in devices]
        info["jax_default_backend"] = jax.default_backend()

        # JAX configuration flags
        info["jax_config"] = {
            "jax_enable_x64": jax.config.jax_enable_x64,
            "jax_platform_name": jax.lib.xla_bridge.get_backend().platform,
        }
    except Exception as e:
        info["jax_error"] = str(e)

    # Library availability
    info["jaxarc_available"] = JAXARC_AVAILABLE
    info["arcle_available"] = ARCLE_AVAILABLE

    if not JAXARC_AVAILABLE:
        info["jaxarc_error"] = JAXARC_IMPORT_ERROR
    if not ARCLE_AVAILABLE:
        info["arcle_error"] = ARCLE_IMPORT_ERROR

    return info


def setup_jaxarc_environment():
    """Setup JaxARC using modern registration-based API.

    Creates a JaxARC environment with minimal overhead configuration:
    - Uses MiniARC dataset for fast testing
    - Raw action configuration for minimal operations
    - Disabled logging and visualization for performance
    - BboxActionWrapper for bbox-style actions

    Returns:
        Tuple of (wrapped_env, env_params, task_id) where:
        - wrapped_env: BboxActionWrapper around base environment
        - env_params: Environment parameters with task buffer
        - task_id: Selected task ID for reference
    """
    if not JAXARC_AVAILABLE:
        raise ImportError(f"JaxARC not available: {JAXARC_IMPORT_ERROR}")

    try:
        from jaxarc.utils.core import get_config

        # Create config with minimal overhead overrides
        config = JaxArcConfig.from_hydra(
            get_config(
                overrides=[
                    "dataset=mini_arc",  # Use MiniARC for fast testing
                    "action=raw",  # Minimal action set
                    "wandb.enabled=false",  # Disable experiment tracking
                    "logging.log_operations=false",  # Disable operation logging
                    "logging.log_rewards=false",  # Disable reward logging
                    "visualization.enabled=false",  # Disable visualization
                    "environment.debug_level=off",  # Disable debug output
                ]
            )
        )

        # Get available task IDs (auto-download if needed)
        available_ids = available_task_ids("Mini", config=config, auto_download=True)
        if not available_ids:
            raise ValueError("No MiniARC tasks available")

        # Use first available task for consistent benchmarking
        task_id = available_ids[0]
        console.print(f"Using MiniARC task: {task_id}")

        # Create environment using registration system
        env, env_params = make(f"Mini-{task_id}", config=config)

        # Wrap with BboxActionWrapper for bbox-style actions
        wrapped_env = BboxActionWrapper(env)

        return wrapped_env, env_params, task_id

    except Exception as e:
        console.print(f"[red]Failed to setup JaxARC environment: {e}[/red]")
        raise


def setup_arcle_environment():
    """Setup ARCLE with equivalent configuration."""
    if not ARCLE_AVAILABLE:
        raise ImportError(f"ARCLE not available: {ARCLE_IMPORT_ERROR}")

    # TODO: Implement ARCLE environment setup
    # This will be implemented in task 3
    raise NotImplementedError("ARCLE environment setup will be implemented in task 3")


def benchmark_timestep_scaling(
    timestep_powers: List[int], num_runs: int = 5
) -> Dict[int, Dict[str, TimingResult]]:
    """
    Benchmark performance scaling with number of timesteps.

    Tests single environment with varying episode lengths from 10^1 to 10^6 steps.
    Similar to NAVIX speed.py approach.

    Args:
        timestep_powers: Powers of 10 to test (e.g., [1, 2, 3] for 10, 100, 1000 steps)
        num_runs: Number of runs per configuration for statistical analysis

    Returns:
        Dictionary mapping timesteps to timing results for each framework
    """
    console.print("[bold blue]Starting timestep scaling benchmark...[/bold blue]")

    results = {}

    for power in timestep_powers:
        timesteps = 10**power
        console.print(f"Testing {timesteps:,} timesteps...")

        results[timesteps] = {}

        # TODO: Implement actual benchmarking logic
        # This will be implemented in task 5
        console.print(
            f"  [yellow]Timestep scaling for {timesteps} steps not yet implemented[/yellow]"
        )

    return results


def benchmark_throughput_scaling(
    batch_powers: List[int], fixed_steps: int = 1000, num_runs: int = 5
) -> Dict[int, Dict[str, TimingResult]]:
    """
    Benchmark throughput scaling with parallel environments.

    Tests varying batch sizes with fixed episode length.
    Similar to NAVIX throughput.py approach.

    Args:
        batch_powers: Powers of 2 to test (e.g., [0, 1, 2] for 1, 2, 4 environments)
        fixed_steps: Fixed number of steps per environment
        num_runs: Number of runs per configuration for statistical analysis

    Returns:
        Dictionary mapping batch sizes to timing results for each framework
    """
    console.print("[bold blue]Starting throughput scaling benchmark...[/bold blue]")

    results = {}

    for power in batch_powers:
        batch_size = 2**power
        console.print(f"Testing batch size {batch_size}...")

        results[batch_size] = {}

        # TODO: Implement actual benchmarking logic
        # This will be implemented in task 6
        console.print(
            f"  [yellow]Throughput scaling for batch size {batch_size} not yet implemented[/yellow]"
        )

    return results


def analyze_results(results: Dict) -> Dict:
    """
    Analyze benchmark results with statistical measures.

    Args:
        results: Raw benchmark results

    Returns:
        Statistical analysis including speedup ratios and confidence intervals
    """
    # TODO: Implement statistical analysis
    # This will be implemented in task 7
    console.print("[yellow]Statistical analysis not yet implemented[/yellow]")
    return {}


def create_timestep_plot(results: Dict, output_dir: Path):
    """Create NAVIX-style timestep scaling visualization."""
    # TODO: Implement timestep scaling plot
    # This will be implemented in task 8
    console.print(
        "[yellow]Timestep scaling plot generation not yet implemented[/yellow]"
    )


def create_throughput_plot(results: Dict, output_dir: Path):
    """Create NAVIX-style throughput scaling visualization."""
    # TODO: Implement throughput scaling plot
    # This will be implemented in task 8
    console.print(
        "[yellow]Throughput scaling plot generation not yet implemented[/yellow]"
    )


def save_results(results: BenchmarkResults, output_dir: Path):
    """Save comprehensive benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = results.timestamp.replace(":", "-").replace(" ", "_")

    # Save timestep scaling results
    if results.timestep_scaling:
        timestep_file = output_dir / f"timestep_scaling_{timestamp}.json"
        with open(timestep_file, "w") as f:
            json.dump(
                {
                    "timestep_scaling": results.timestep_scaling,
                    "system_info": results.system_info,
                    "config": asdict(results.config),
                    "timestamp": results.timestamp,
                },
                f,
                indent=2,
                default=str,
            )
        console.print(f"Timestep results saved to: {timestep_file}")

    # Save throughput scaling results
    if results.throughput_scaling:
        throughput_file = output_dir / f"throughput_scaling_{timestamp}.json"
        with open(throughput_file, "w") as f:
            json.dump(
                {
                    "throughput_scaling": results.throughput_scaling,
                    "system_info": results.system_info,
                    "config": asdict(results.config),
                    "timestamp": results.timestamp,
                },
                f,
                indent=2,
                default=str,
            )
        console.print(f"Throughput results saved to: {throughput_file}")


@app.command()
def main(
    benchmark_type: str = typer.Option(
        "both",
        "--benchmark-type",
        help="Type of benchmark to run: 'timestep', 'throughput', or 'both'",
    ),
    timestep_powers: str = typer.Option(
        "1,2,3,4,5,6",
        "--timestep-powers",
        help="Comma-separated powers of 10 for timestep scaling (e.g., '1,2,3' for 10,100,1000 steps)",
    ),
    batch_powers: str = typer.Option(
        "0,1,2,3,4,5,6,7,8,9,10",
        "--batch-powers",
        help="Comma-separated powers of 2 for throughput scaling (e.g., '0,1,2' for 1,2,4 environments)",
    ),
    num_runs: int = typer.Option(
        5,
        "--num-runs",
        help="Number of runs per configuration for statistical analysis",
    ),
    fixed_steps: int = typer.Option(
        1000, "--fixed-steps", help="Fixed number of steps for throughput benchmark"
    ),
    output_dir: str = typer.Option(
        "benchmarks/results",
        "--output-dir",
        help="Output directory for results and plots",
    ),
    save_plots: bool = typer.Option(
        True, "--save-plots/--no-plots", help="Generate and save matplotlib plots"
    ),
    save_json: bool = typer.Option(
        True, "--save-json/--no-json", help="Save detailed JSON results"
    ),
):
    """
    Run comprehensive ARCLE vs JaxARC performance benchmark.

    This benchmark provides two key scaling analyses following NAVIX methodology:
    1. Timestep scaling: Performance vs episode length (10 to 1M steps)
    2. Throughput scaling: Performance vs parallel environments (1 to max feasible)
    """
    # Parse CLI arguments
    try:
        timestep_powers_list = [int(x.strip()) for x in timestep_powers.split(",")]
        batch_powers_list = [int(x.strip()) for x in batch_powers.split(",")]
    except ValueError as e:
        console.print(f"[red]Error parsing powers: {e}[/red]")
        raise typer.Exit(1)

    # Validate benchmark type
    if benchmark_type not in ["timestep", "throughput", "both"]:
        console.print(f"[red]Invalid benchmark type: {benchmark_type}[/red]")
        console.print("Valid options: 'timestep', 'throughput', 'both'")
        raise typer.Exit(1)

    # Create configuration
    config = BenchmarkConfig(
        timestep_powers=timestep_powers_list,
        batch_powers=batch_powers_list,
        num_runs=num_runs,
        fixed_steps=fixed_steps,
        output_dir=Path(output_dir),
        save_plots=save_plots,
        save_json=save_json,
        benchmark_type=benchmark_type,
    )

    # Display system information
    console.print("[bold green]System Information[/bold green]")
    system_info = get_system_info()

    info_table = Table(title="System Configuration")
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Value", style="white")

    for key, value in system_info.items():
        if isinstance(value, (list, dict)):
            value = str(value)
        info_table.add_row(key, str(value))

    console.print(info_table)

    # Check library availability
    if not JAXARC_AVAILABLE:
        console.print(f"[red]JaxARC not available: {JAXARC_IMPORT_ERROR}[/red]")
        raise typer.Exit(1)

    if not ARCLE_AVAILABLE:
        console.print(f"[yellow]ARCLE not available: {ARCLE_IMPORT_ERROR}[/yellow]")
        console.print("[yellow]Will run JaxARC-only benchmarks[/yellow]")

    # Initialize results
    results = BenchmarkResults(
        timestep_scaling={},
        throughput_scaling={},
        system_info=system_info,
        config=config,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Run benchmarks based on type
    try:
        if benchmark_type in ["timestep", "both"]:
            results.timestep_scaling = benchmark_timestep_scaling(
                timestep_powers_list, num_runs
            )

        if benchmark_type in ["throughput", "both"]:
            results.throughput_scaling = benchmark_throughput_scaling(
                batch_powers_list, fixed_steps, num_runs
            )

        # Analyze results
        if results.timestep_scaling or results.throughput_scaling:
            analysis = analyze_results(results)
            console.print("[bold green]Analysis complete[/bold green]")

        # Generate plots
        if save_plots:
            if results.timestep_scaling:
                create_timestep_plot(results.timestep_scaling, config.output_dir)
            if results.throughput_scaling:
                create_throughput_plot(results.throughput_scaling, config.output_dir)

        # Save results
        if save_json:
            save_results(results, config.output_dir)

        console.print("[bold green]Benchmark complete![/bold green]")

    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
