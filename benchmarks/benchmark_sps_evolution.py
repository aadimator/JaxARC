"""JaxARC Performance Evolution and Hardware Comparison Analysis

This script creates two distinct types of performance visualizations:

1. Mac Evolution Analysis: Shows performance improvements across development phases
   - Baseline → Phase 1 → Phase 2 on Mac M3 Max

2. Hardware Comparison Analysis: Compares Phase 2 performance across different hardware
   - Mac M3 Max vs RTX 3090 vs H100

Data Sources:
- Mac Baseline: baseline_benchmark_20250812_232058.json
- Mac Phase 1: benchmark_phase1_20250812_234006.json
- Mac Phase 2: benchmark_phase2_mac_20250814_095248.json
- RTX 3090 Phase 2: benchmark_phase2_gpu_20250813_151950.json
- H100 Phase 2: benchmark_phase2_kailash_20250814_123928.json
"""

# %%
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from pyprojroot import here

# Set up professional styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 20,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#2D3748",
    }
)

# Professional color schemes
MAC_EVOLUTION_COLORS = {
    "Baseline": "#E53E3E",  # Red
    "Phase 1": "#DD6B20",  # Orange
    "Phase 2": "#38A169",  # Green
}

HARDWARE_COLORS = {
    "Mac M2 Pro": "#3182CE",  # Professional Blue
    "RTX 3090": "#805AD5",  # Professional Purple
    "H100": "#D69E2E",  # Professional Gold
    "H100 Scan": "#E53E3E",  # Professional Red for scan
}


def format_sps(x, pos):
    """Format SPS values for better readability."""
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def format_batch_size(x, pos):
    """Format batch sizes for better readability."""
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{int(x)}"


def load_json_file(filename):
    """Load a specific JSON benchmark file."""
    results_dir = here() / "benchmarks" / "results"
    file_path = results_dir / filename

    if not file_path.exists():
        print(f"⚠️  File not found: {filename}")
        return None

    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None


def extract_single_env_performance(data):
    """Extract single environment performance from benchmark data."""
    # Try different possible keys for single environment data
    single_keys = ["single_environment", "single_environment_arc"]

    for key in single_keys:
        if key in data:
            return data[key]["steps_per_second"]

    return None


def extract_batch_performance(data):
    """Extract batch scaling performance from benchmark data."""
    # Try different possible keys for batch data
    batch_keys = ["batch_scaling", "batch_scaling_arc"]

    for key in batch_keys:
        if key in data:
            batch_data = data[key]
            batches = []
            sps_values = []

            for batch_str, metrics in batch_data.items():
                try:
                    batch_size = int(batch_str)
                    sps = metrics["steps_per_second"]
                    batches.append(batch_size)
                    sps_values.append(sps)
                except (ValueError, KeyError):
                    continue

            # Sort by batch size
            if batches and sps_values:
                sorted_data = sorted(zip(batches, sps_values))
                return zip(*sorted_data)

    return [], []


def extract_scan_performance(data):
    """Extract functional scan performance from benchmark data."""
    if "functional_scan" in data:
        scan_data = data["functional_scan"]
        batches = []
        sps_values = []

        for batch_str, metrics in scan_data.items():
            try:
                batch_size = int(batch_str)
                sps = metrics["steps_per_second"]
                batches.append(batch_size)
                sps_values.append(sps)
            except (ValueError, KeyError):
                continue

        # Sort by batch size
        if batches and sps_values:
            sorted_data = sorted(zip(batches, sps_values))
            return zip(*sorted_data)

    return [], []


def get_hardware_name(data):
    """Extract hardware name from system info."""
    if "system_info" not in data:
        return "Unknown"

    system_info = data["system_info"]

    # Check if it's GPU
    if "jax_devices" in system_info:
        devices = system_info["jax_devices"]
        if devices.get("platform") == "gpu" and "types" in devices:
            device_type = devices["types"][0]
            if "RTX 3090" in device_type:
                return "RTX 3090"
            if "H100" in device_type:
                return "H100"

    # Check if it's Mac
    platform = system_info.get("platform", "")
    if "macOS" in platform and "arm64" in platform:
        return "Mac M2 Pro"

    return "Unknown"


def plot_mac_evolution():
    """Create plots showing Mac performance evolution across development phases."""
    print("\n🍎 Analyzing Mac Performance Evolution...")
    print("=" * 50)

    # Load Mac evolution data
    baseline_data = load_json_file("baseline_benchmark_20250812_232058.json")
    phase1_data = load_json_file("benchmark_phase1_20250812_234006.json")
    phase2_data = load_json_file("benchmark_phase2_mac_20250814_095248.json")

    if not all([baseline_data, phase1_data, phase2_data]):
        print("❌ Could not load all Mac evolution data files")
        return

    # Extract single environment performance
    baseline_single = extract_single_env_performance(baseline_data)
    phase1_single = extract_single_env_performance(phase1_data)
    phase2_single = extract_single_env_performance(phase2_data)

    # Extract batch performance
    baseline_batches, baseline_sps = extract_batch_performance(baseline_data)
    phase1_batches, phase1_sps = extract_batch_performance(phase1_data)
    phase2_batches, phase2_sps = extract_batch_performance(phase2_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Single Environment Evolution
    phases = ["Baseline", "Phase 1", "Phase 2"]
    single_values = [baseline_single, phase1_single, phase2_single]

    if all(v is not None for v in single_values):
        bars1 = ax1.bar(
            phases,
            single_values,
            color=[MAC_EVOLUTION_COLORS[phase] for phase in phases],
            alpha=0.85,
            edgecolor="white",
            linewidth=2.5,
            width=0.6,
        )

        # Add gradient effect
        for bar, phase in zip(bars1, phases):
            bar.set_edgecolor("#2D3748")
            bar.set_linewidth(1.5)

        # Add value labels with improved styling
        for bar, value in zip(bars1, single_values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + height * 0.03,
                f"{value:,.0f}\nSPS",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
                color="#2D3748",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    edgecolor="#CBD5E0",
                    alpha=0.9,
                    linewidth=1,
                ),
            )

        ax1.set_title(
            "Single Environment Performance Evolution\nMac M2 Pro Development Progress",
            fontweight="bold",
            pad=25,
            color="#2D3748",
        )
        ax1.set_ylabel("Steps Per Second", fontweight="bold", color="#2D3748")
        ax1.set_xlabel("Development Phase", fontweight="bold", color="#2D3748")
        ax1.yaxis.set_major_formatter(FuncFormatter(format_sps))
        ax1.grid(True, alpha=0.2, axis="y", linestyle="-", linewidth=0.8)
        ax1.set_ylim(0, max(single_values) * 1.2)
    else:
        ax1.text(
            0.5,
            0.5,
            "Single Environment Data\nNot Available",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    # Plot 2: Batch Scaling Evolution
    if baseline_batches and phase1_batches and phase2_batches:
        ax2.plot(
            baseline_batches,
            baseline_sps,
            "o-",
            linewidth=3.5,
            markersize=9,
            label="Baseline",
            color=MAC_EVOLUTION_COLORS["Baseline"],
            alpha=0.9,
            markerfacecolor="white",
            markeredgewidth=2,
            markeredgecolor=MAC_EVOLUTION_COLORS["Baseline"],
        )
        ax2.plot(
            phase1_batches,
            phase1_sps,
            "s-",
            linewidth=3.5,
            markersize=9,
            label="Phase 1",
            color=MAC_EVOLUTION_COLORS["Phase 1"],
            alpha=0.9,
            markerfacecolor="white",
            markeredgewidth=2,
            markeredgecolor=MAC_EVOLUTION_COLORS["Phase 1"],
        )
        ax2.plot(
            phase2_batches,
            phase2_sps,
            "^-",
            linewidth=3.5,
            markersize=9,
            label="Phase 2",
            color=MAC_EVOLUTION_COLORS["Phase 2"],
            alpha=0.9,
            markerfacecolor="white",
            markeredgewidth=2,
            markeredgecolor=MAC_EVOLUTION_COLORS["Phase 2"],
        )

        # Add value annotations for higher batch sizes
        annotation_batches = [1000, 5000, 10000]
        for batches, sps_values, color in [
            (baseline_batches, baseline_sps, MAC_EVOLUTION_COLORS["Baseline"]),
            (phase1_batches, phase1_sps, MAC_EVOLUTION_COLORS["Phase 1"]),
            (phase2_batches, phase2_sps, MAC_EVOLUTION_COLORS["Phase 2"]),
        ]:
            for i, (batch, sps) in enumerate(zip(batches, sps_values)):
                if batch in annotation_batches:
                    ax2.annotate(
                        f"{sps:,.0f}",
                        (batch, sps),
                        xytext=(5, 10),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color=color,
                        alpha=0.8,
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor=color,
                            alpha=0.8,
                            linewidth=1,
                        ),
                    )

        ax2.set_xscale("log")
        ax2.set_xlabel("Batch Size (log scale)", fontweight="bold", color="#2D3748")
        ax2.set_ylabel("Steps Per Second", fontweight="bold", color="#2D3748")
        ax2.set_title(
            "Batch Performance Scaling Evolution\nMac M2 Pro Development Progress",
            fontweight="bold",
            pad=25,
            color="#2D3748",
        )
        ax2.yaxis.set_major_formatter(FuncFormatter(format_sps))
        ax2.xaxis.set_major_formatter(FuncFormatter(format_batch_size))
        ax2.legend(
            framealpha=0.95,
            fontsize=12,
            loc="upper left",
            fancybox=True,
            shadow=True,
            edgecolor="#CBD5E0",
        )
        ax2.grid(True, alpha=0.2, which="both", linestyle="-", linewidth=0.8)
    else:
        ax2.text(
            0.5,
            0.5,
            "Batch Scaling Data\nNot Available",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    plt.suptitle(
        "JaxARC Mac Performance Evolution\nBaseline → Phase 1 → Phase 2 Development Progress",
        fontsize=18,
        fontweight="bold",
        y=0.96,
    )
    plt.tight_layout()
    plt.show()

    # Create linear scale version
    plot_mac_evolution_linear(baseline_data, phase1_data, phase2_data)

    # Print evolution summary
    print("\n📊 MAC EVOLUTION SUMMARY:")
    print("-" * 30)
    if all(v is not None for v in single_values):
        for phase, value in zip(phases, single_values):
            improvement = (
                ((value / baseline_single - 1) * 100) if phase != "Baseline" else 0
            )
            print(
                f"{phase:>10}: {value:>8,.0f} SPS"
                + (f" (+{improvement:.1f}%)" if improvement > 0 else "")
            )

        total_improvement = (phase2_single / baseline_single - 1) * 100
        print(
            f"\n🚀 Total Evolution: {total_improvement:.1f}% improvement from Baseline to Phase 2"
        )

    # Print peak batch performance evolution
    if baseline_sps and phase1_sps and phase2_sps:
        baseline_peak = max(baseline_sps)
        phase1_peak = max(phase1_sps)
        phase2_peak = max(phase2_sps)

        print("\n📈 PEAK BATCH PERFORMANCE:")
        print("-" * 30)
        print(f"{'Baseline':>10}: {baseline_peak:>10,.0f} SPS")
        phase1_batch_improvement = (phase1_peak / baseline_peak - 1) * 100
        print(
            f"{'Phase 1':>10}: {phase1_peak:>10,.0f} SPS (+{phase1_batch_improvement:.1f}%)"
        )
        phase2_batch_improvement = (phase2_peak / baseline_peak - 1) * 100
        print(
            f"{'Phase 2':>10}: {phase2_peak:>10,.0f} SPS (+{phase2_batch_improvement:.1f}%)"
        )


def plot_mac_evolution_linear(baseline_data, phase1_data, phase2_data):
    """Create linear-scale version of Mac evolution plots."""
    print("\n📏 Creating Linear Scale Mac Evolution...")

    # Extract batch performance
    baseline_batches, baseline_sps = extract_batch_performance(baseline_data)
    phase1_batches, phase1_sps = extract_batch_performance(phase1_data)
    phase2_batches, phase2_sps = extract_batch_performance(phase2_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Focus on lower batch sizes (linear scale)
    max_batch_for_linear = 2000

    if baseline_batches and phase1_batches and phase2_batches:
        # Filter data for smaller batch sizes
        for batches, sps_values, phase, color in [
            (
                baseline_batches,
                baseline_sps,
                "Baseline",
                MAC_EVOLUTION_COLORS["Baseline"],
            ),
            (phase1_batches, phase1_sps, "Phase 1", MAC_EVOLUTION_COLORS["Phase 1"]),
            (phase2_batches, phase2_sps, "Phase 2", MAC_EVOLUTION_COLORS["Phase 2"]),
        ]:
            filtered_data = [
                (b, s) for b, s in zip(batches, sps_values) if b <= max_batch_for_linear
            ]
            if filtered_data:
                filtered_batches, filtered_sps = zip(*filtered_data)
                marker = (
                    "o" if phase == "Baseline" else ("s" if phase == "Phase 1" else "^")
                )
                ax1.plot(
                    filtered_batches,
                    filtered_sps,
                    marker + "-",
                    linewidth=3.5,
                    markersize=9,
                    label=phase,
                    color=color,
                    alpha=0.9,
                    markerfacecolor="white",
                    markeredgewidth=2,
                    markeredgecolor=color,
                )

                # Add value annotations for key points in linear view
                for batch, sps in zip(filtered_batches, filtered_sps):
                    if batch in [1000, 2000] or batch == max(filtered_batches):
                        ax1.annotate(
                            f"{sps:,.0f}",
                            (batch, sps),
                            xytext=(5, 10),
                            textcoords="offset points",
                            ha="left",
                            va="bottom",
                            fontsize=9,
                            fontweight="bold",
                            color=color,
                            alpha=0.8,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                edgecolor=color,
                                alpha=0.8,
                                linewidth=1,
                            ),
                        )

        ax1.set_xlabel("Batch Size (linear scale)", fontweight="bold", color="#2D3748")
        ax1.set_ylabel("Steps Per Second", fontweight="bold", color="#2D3748")
        ax1.set_title(
            "Mac Evolution (Small Batches)\nLinear Scale View",
            fontweight="bold",
            pad=25,
            color="#2D3748",
        )
        ax1.yaxis.set_major_formatter(FuncFormatter(format_sps))
        ax1.legend(
            framealpha=0.95,
            fontsize=12,
            loc="upper left",
            fancybox=True,
            shadow=True,
            edgecolor="#CBD5E0",
        )
        ax1.grid(True, alpha=0.2, linestyle="-", linewidth=0.8)

    # Plot 2: Full range linear scale
    if baseline_batches and phase1_batches and phase2_batches:
        for batches, sps_values, phase, color in [
            (
                baseline_batches,
                baseline_sps,
                "Baseline",
                MAC_EVOLUTION_COLORS["Baseline"],
            ),
            (phase1_batches, phase1_sps, "Phase 1", MAC_EVOLUTION_COLORS["Phase 1"]),
            (phase2_batches, phase2_sps, "Phase 2", MAC_EVOLUTION_COLORS["Phase 2"]),
        ]:
            marker = (
                "o" if phase == "Baseline" else ("s" if phase == "Phase 1" else "^")
            )
            ax2.plot(
                batches,
                sps_values,
                marker + "-",
                linewidth=3.5,
                markersize=7,
                label=phase,
                color=color,
                alpha=0.9,
                markerfacecolor="white",
                markeredgewidth=2,
                markeredgecolor=color,
            )

            # Add value annotations for peak performance and key points
            peak_idx = sps_values.index(max(sps_values))
            peak_batch = batches[peak_idx]
            peak_sps = sps_values[peak_idx]
            ax2.annotate(
                f"{peak_sps:,.0f}",
                (peak_batch, peak_sps),
                xytext=(5, 10),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=color,
                alpha=0.9,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.9,
                    linewidth=1.5,
                ),
            )

        ax2.set_xlabel("Batch Size (linear scale)", fontweight="bold", color="#2D3748")
        ax2.set_ylabel("Steps Per Second", fontweight="bold", color="#2D3748")
        ax2.set_title(
            "Mac Evolution (Full Range)\nLinear Scale View",
            fontweight="bold",
            pad=25,
            color="#2D3748",
        )
        ax2.yaxis.set_major_formatter(FuncFormatter(format_sps))
        ax2.xaxis.set_major_formatter(FuncFormatter(format_batch_size))
        ax2.legend(
            framealpha=0.95,
            fontsize=12,
            loc="upper left",
            fancybox=True,
            shadow=True,
            edgecolor="#CBD5E0",
        )
        ax2.grid(True, alpha=0.2, linestyle="-", linewidth=0.8)

    plt.suptitle(
        "JaxARC Mac Evolution - Linear Scale Views\nDetailed Development Progress",
        fontsize=18,
        fontweight="bold",
        y=0.96,
    )
    plt.tight_layout()
    plt.show()


def plot_hardware_comparison():
    """Create plots comparing Phase 2 performance across different hardware."""
    print("\n🖥️  Analyzing Hardware Performance Comparison...")
    print("=" * 50)

    # Load Phase 2 data for all hardware
    mac_data = load_json_file("benchmark_phase2_mac_20250814_095248.json")
    rtx_data = load_json_file("benchmark_phase2_gpu_20250813_151950.json")
    h100_data = load_json_file("benchmark_phase2_kailash_20250814_123928.json")
    hardware_data = {"Mac M2 Pro": mac_data, "RTX 3090": rtx_data, "H100": h100_data}

    # Remove None entries
    hardware_data = {k: v for k, v in hardware_data.items() if v is not None}

    if not hardware_data:
        print("❌ Could not load hardware comparison data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Extract single environment performance
    hardware_names = list(hardware_data.keys())
    single_performance = []

    for name, data in hardware_data.items():
        single_sps = extract_single_env_performance(data)
        single_performance.append(single_sps if single_sps is not None else 0)

    # Plot 1: Single Environment Comparison
    if any(perf > 0 for perf in single_performance):
        bars1 = ax1.bar(
            hardware_names,
            single_performance,
            color=[HARDWARE_COLORS[name] for name in hardware_names],
            alpha=0.85,
            edgecolor="#2D3748",
            linewidth=1.5,
            width=0.6,
        )

        # Add value labels with improved styling
        for bar, value in zip(bars1, single_performance):
            if value > 0:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + height * 0.03,
                    f"{value:,.0f}\nSPS",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=12,
                    color="#2D3748",
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor="white",
                        edgecolor="#CBD5E0",
                        alpha=0.9,
                        linewidth=1,
                    ),
                )

        ax1.set_title(
            "Single Environment Performance\nPhase 2 Hardware Comparison",
            fontweight="bold",
            pad=25,
            color="#2D3748",
        )
        ax1.set_ylabel("Steps Per Second", fontweight="bold", color="#2D3748")
        ax1.set_xlabel("Hardware Platform", fontweight="bold", color="#2D3748")
        ax1.yaxis.set_major_formatter(FuncFormatter(format_sps))
        ax1.grid(True, alpha=0.2, axis="y", linestyle="-", linewidth=0.8)
        ax1.set_ylim(0, max(single_performance) * 1.2 if single_performance else 1000)
    else:
        ax1.text(
            0.5,
            0.5,
            "Single Environment Data\nNot Available",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    # Plot 2: Batch Scaling Comparison
    ax2.set_xscale("log")

    for name, data in hardware_data.items():
        batches, sps_values = extract_batch_performance(data)
        if batches and sps_values:
            ax2.plot(
                batches,
                sps_values,
                "o-",
                linewidth=3.5,
                markersize=8,
                label=name,
                color=HARDWARE_COLORS[name],
                alpha=0.9,
                markerfacecolor="white",
                markeredgewidth=2,
                markeredgecolor=HARDWARE_COLORS[name],
            )

            # Add value annotations for higher batch sizes
            annotation_batches = [1000, 5000, 10000, 50000]
            for batch, sps in zip(batches, sps_values):
                if batch in annotation_batches or batch == max(batches):
                    ax2.annotate(
                        f"{sps:,.0f}",
                        (batch, sps),
                        xytext=(5, 10),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color=HARDWARE_COLORS[name],
                        alpha=0.8,
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor=HARDWARE_COLORS[name],
                            alpha=0.8,
                            linewidth=1,
                        ),
                    )

    ax2.set_xlabel("Batch Size (log scale)", fontweight="bold", color="#2D3748")
    ax2.set_ylabel("Steps Per Second", fontweight="bold", color="#2D3748")
    ax2.set_title(
        "Batch Performance Scaling\nPhase 2 Hardware Comparison",
        fontweight="bold",
        pad=25,
        color="#2D3748",
    )
    ax2.yaxis.set_major_formatter(FuncFormatter(format_sps))
    ax2.xaxis.set_major_formatter(FuncFormatter(format_batch_size))
    ax2.legend(
        framealpha=0.95,
        fontsize=12,
        loc="upper left",
        fancybox=True,
        shadow=True,
        edgecolor="#CBD5E0",
    )
    ax2.grid(True, alpha=0.2, which="both", linestyle="-", linewidth=0.8)

    plt.suptitle(
        "JaxARC Phase 2 Hardware Comparison\nMac M2 Pro vs RTX 3090 vs H100",
        fontsize=18,
        fontweight="bold",
        y=0.96,
    )
    plt.tight_layout()
    plt.show()

    # Create linear scale version
    plot_hardware_comparison_linear(hardware_data)

    # Print hardware comparison summary
    print("\n📊 HARDWARE COMPARISON SUMMARY:")
    print("-" * 40)

    # Single environment summary
    if any(perf > 0 for perf in single_performance):
        print("Single Environment Performance:")
        for name, perf in zip(hardware_names, single_performance):
            if perf > 0:
                print(f"  {name:>12}: {perf:>8,.0f} SPS")

        # Calculate relative performance
        if "Mac M2 Pro" in hardware_names:
            mac_perf = single_performance[hardware_names.index("Mac M2 Pro")]
            if mac_perf > 0:
                print("\nSpeedup vs Mac M2 Pro:")
                for name, perf in zip(hardware_names, single_performance):
                    if name != "Mac M2 Pro" and perf > 0:
                        speedup = perf / mac_perf
                        print(f"  {name:>12}: {speedup:>8.1f}×")

    # Peak batch performance summary
    print("\nPeak Batch Performance:")
    peak_data = []
    for name, data in hardware_data.items():
        batches, sps_values = extract_batch_performance(data)
        if sps_values:
            peak_sps = max(sps_values)
            peak_batch = batches[sps_values.index(peak_sps)]
            peak_data.append((name, peak_sps, peak_batch))
            print(f"  {name:>12}: {peak_sps:>10,.0f} SPS (Batch {peak_batch:,})")

    # Hardware efficiency analysis
    if peak_data and "Mac M2 Pro" in [item[0] for item in peak_data]:
        mac_peak = next(item[1] for item in peak_data if item[0] == "Mac M2 Pro")
        print("\nPeak Performance vs Mac M2 Pro:")
        for name, peak_sps, peak_batch in peak_data:
            if name != "Mac M2 Pro":
                speedup = peak_sps / mac_peak
                print(f"  {name:>12}: {speedup:>8.1f}× faster")


def plot_hardware_comparison_linear(hardware_data):
    """Create linear-scale version of hardware comparison plots."""
    print("\n📏 Creating Linear Scale Hardware Comparison...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Focus on lower batch sizes (linear scale)
    max_batch_for_linear = 2000  # Focus on smaller batch sizes for linear view

    for name, data in hardware_data.items():
        batches, sps_values = extract_batch_performance(data)
        if batches and sps_values:
            # Filter for smaller batch sizes
            filtered_data = [
                (b, s) for b, s in zip(batches, sps_values) if b <= max_batch_for_linear
            ]
            if filtered_data:
                filtered_batches, filtered_sps = zip(*filtered_data)
                ax1.plot(
                    filtered_batches,
                    filtered_sps,
                    "o-",
                    linewidth=3.5,
                    markersize=8,
                    label=name,
                    color=HARDWARE_COLORS[name],
                    alpha=0.9,
                    markerfacecolor="white",
                    markeredgewidth=2,
                    markeredgecolor=HARDWARE_COLORS[name],
                )

                # Add value annotations for key points in linear view
                for batch, sps in zip(filtered_batches, filtered_sps):
                    if batch in [1000, 2000] or batch == max(filtered_batches):
                        ax1.annotate(
                            f"{sps:,.0f}",
                            (batch, sps),
                            xytext=(5, 10),
                            textcoords="offset points",
                            ha="left",
                            va="bottom",
                            fontsize=9,
                            fontweight="bold",
                            color=HARDWARE_COLORS[name],
                            alpha=0.8,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                edgecolor=HARDWARE_COLORS[name],
                                alpha=0.8,
                                linewidth=1,
                            ),
                        )

    ax1.set_xlabel("Batch Size (linear scale)", fontweight="bold", color="#2D3748")
    ax1.set_ylabel("Steps Per Second", fontweight="bold", color="#2D3748")
    ax1.set_title(
        "Batch Performance Scaling (Small Batches)\nLinear Scale View",
        fontweight="bold",
        pad=25,
        color="#2D3748",
    )
    ax1.yaxis.set_major_formatter(FuncFormatter(format_sps))
    ax1.legend(
        framealpha=0.95,
        fontsize=12,
        loc="upper left",
        fancybox=True,
        shadow=True,
        edgecolor="#CBD5E0",
    )
    ax1.grid(True, alpha=0.2, linestyle="-", linewidth=0.8)

    # Plot 2: Full range linear scale
    for name, data in hardware_data.items():
        batches, sps_values = extract_batch_performance(data)
        if batches and sps_values:
            ax2.plot(
                batches,
                sps_values,
                "o-",
                linewidth=3.5,
                markersize=6,
                label=name,
                color=HARDWARE_COLORS[name],
                alpha=0.9,
                markerfacecolor="white",
                markeredgewidth=2,
                markeredgecolor=HARDWARE_COLORS[name],
            )

            # Add value annotations for peak performance
            peak_idx = sps_values.index(max(sps_values))
            peak_batch = batches[peak_idx]
            peak_sps = sps_values[peak_idx]
            ax2.annotate(
                f"{peak_sps:,.0f}",
                (peak_batch, peak_sps),
                xytext=(5, 10),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=HARDWARE_COLORS[name],
                alpha=0.9,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=HARDWARE_COLORS[name],
                    alpha=0.9,
                    linewidth=1.5,
                ),
            )

    ax2.set_xlabel("Batch Size (linear scale)", fontweight="bold", color="#2D3748")
    ax2.set_ylabel("Steps Per Second", fontweight="bold", color="#2D3748")
    ax2.set_title(
        "Batch Performance Scaling (Full Range)\nLinear Scale View",
        fontweight="bold",
        pad=25,
        color="#2D3748",
    )
    ax2.yaxis.set_major_formatter(FuncFormatter(format_sps))
    ax2.xaxis.set_major_formatter(FuncFormatter(format_batch_size))
    ax2.legend(
        framealpha=0.95,
        fontsize=12,
        loc="upper left",
        fancybox=True,
        shadow=True,
        edgecolor="#CBD5E0",
    )
    ax2.grid(True, alpha=0.2, linestyle="-", linewidth=0.8)

    plt.suptitle(
        "JaxARC Hardware Comparison - Linear Scale Views\nDetailed Performance Analysis",
        fontsize=18,
        fontweight="bold",
        y=0.96,
    )
    plt.tight_layout()
    plt.show()


def plot_acceleration_factors():
    """Create a focused plot showing GPU acceleration factors."""
    print("\n🚀 Analyzing GPU Acceleration Factors...")
    print("=" * 50)

    # Load data
    mac_data = load_json_file("benchmark_phase2_mac_20250814_095248.json")
    rtx_data = load_json_file("benchmark_phase2_gpu_20250813_151950.json")
    h100_data = load_json_file("benchmark_phase2_kailash_20250814_123928.json")

    if not mac_data:
        print("❌ Need Mac data as baseline for acceleration factors")
        return

    # Extract Mac performance as baseline
    mac_batches, mac_sps = extract_batch_performance(mac_data)
    if not mac_batches:
        print("❌ No Mac batch data available for baseline")
        return

    # Create batch size to SPS mapping for Mac
    mac_performance = dict(zip(mac_batches, mac_sps))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Common batch sizes for comparison
    common_batches = sorted(mac_performance.keys())[:8]  # Limit to first 8 for clarity

    rtx_speedups = []
    h100_speedups = []
    batch_labels = []

    # Calculate speedups for RTX 3090
    if rtx_data:
        rtx_batches, rtx_sps = extract_batch_performance(rtx_data)
        rtx_performance = dict(zip(rtx_batches, rtx_sps)) if rtx_batches else {}

        for batch in common_batches:
            if batch in mac_performance and batch in rtx_performance:
                speedup = rtx_performance[batch] / mac_performance[batch]
                rtx_speedups.append(speedup)
                if not batch_labels or len(batch_labels) < len(rtx_speedups):
                    batch_labels.append(format_batch_size(batch, None))

    # Calculate speedups for H100
    if h100_data:
        h100_batches, h100_sps = extract_batch_performance(h100_data)
        h100_performance = dict(zip(h100_batches, h100_sps)) if h100_batches else {}

        for batch in common_batches:
            if batch in mac_performance and batch in h100_performance:
                speedup = h100_performance[batch] / mac_performance[batch]
                h100_speedups.append(speedup)

    # Ensure both lists have same length
    min_length = min(len(rtx_speedups), len(h100_speedups), len(batch_labels))
    rtx_speedups = rtx_speedups[:min_length]
    h100_speedups = h100_speedups[:min_length]
    batch_labels = batch_labels[:min_length]

    if not rtx_speedups and not h100_speedups:
        print("❌ No acceleration data available")
        return

    x = np.arange(len(batch_labels))
    width = 0.35

    # Create bars with improved styling
    if rtx_speedups:
        bars1 = ax.bar(
            x - width / 2,
            rtx_speedups,
            width,
            label="RTX 3090",
            color=HARDWARE_COLORS["RTX 3090"],
            alpha=0.85,
            edgecolor="#2D3748",
            linewidth=1.5,
        )

        # Add value labels
        for bar, speedup in zip(bars1, rtx_speedups):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05,
                f"{speedup:.1f}×",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
                color="#2D3748",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="#CBD5E0",
                    alpha=0.9,
                ),
            )

    if h100_speedups:
        bars2 = ax.bar(
            x + width / 2,
            h100_speedups,
            width,
            label="H100",
            color=HARDWARE_COLORS["H100"],
            alpha=0.85,
            edgecolor="#2D3748",
            linewidth=1.5,
        )

        # Add value labels
        for bar, speedup in zip(bars2, h100_speedups):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05,
                f"{speedup:.1f}×",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
                color="#2D3748",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="#CBD5E0",
                    alpha=0.9,
                ),
            )

    # Add baseline reference
    ax.axhline(
        y=1,
        color="black",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Mac M2 Pro Baseline (1×)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(batch_labels, rotation=45, ha="right")
    ax.set_xlabel("Batch Size", fontweight="bold", color="#2D3748")
    ax.set_ylabel("Speedup Factor vs Mac M2 Pro", fontweight="bold", color="#2D3748")
    ax.set_title(
        "GPU Acceleration Factors\nPerformance Multiplier vs Mac M2 Pro Baseline",
        fontweight="bold",
        pad=25,
        color="#2D3748",
    )
    ax.legend(
        framealpha=0.95,
        fontsize=12,
        loc="upper left",
        fancybox=True,
        shadow=True,
        edgecolor="#CBD5E0",
    )
    ax.grid(True, alpha=0.2, axis="y", linestyle="-", linewidth=0.8)

    plt.tight_layout()
    plt.show()

    # Print acceleration summary
    print("\n🚀 GPU ACCELERATION SUMMARY:")
    print("-" * 30)
    if rtx_speedups:
        avg_rtx = np.mean(rtx_speedups)
        max_rtx = max(rtx_speedups)
        print(f"RTX 3090: {avg_rtx:.1f}× avg, {max_rtx:.1f}× max speedup")

    if h100_speedups:
        avg_h100 = np.mean(h100_speedups)
        max_h100 = max(h100_speedups)
        print(f"H100:     {avg_h100:.1f}× avg, {max_h100:.1f}× max speedup")


def plot_h100_comparison():
    """Create dedicated plots comparing H100 regular vs H100 scan performance."""
    print("\n⚡ Analyzing H100 vs H100 Scan Comparison...")
    print("=" * 50)

    # Load H100 data
    h100_data = load_json_file("benchmark_phase2_kailash_20250814_123928.json")
    h100_scan_data = load_json_file("kailash_scan_benchmark_20250814_124443.json")

    if not h100_data or not h100_scan_data:
        print("❌ Could not load H100 comparison data")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))

    # Extract performance data
    h100_batches, h100_sps = extract_batch_performance(h100_data)
    h100_scan_batches, h100_scan_sps = extract_scan_performance(h100_scan_data)

    # Plot 1: Log scale comparison
    if h100_batches and h100_scan_batches:
        ax1.plot(
            h100_batches,
            h100_sps,
            "o-",
            linewidth=3.5,
            markersize=8,
            label="H100 Regular",
            color=HARDWARE_COLORS["H100"],
            alpha=0.9,
            markerfacecolor="white",
            markeredgewidth=2,
            markeredgecolor=HARDWARE_COLORS["H100"],
        )
        ax1.plot(
            h100_scan_batches,
            h100_scan_sps,
            "s-",
            linewidth=3.5,
            markersize=8,
            label="H100 Scan",
            color=HARDWARE_COLORS["H100 Scan"],
            alpha=0.9,
            markerfacecolor="white",
            markeredgewidth=2,
            markeredgecolor=HARDWARE_COLORS["H100 Scan"],
        )

        # Add value annotations for higher batch sizes
        annotation_batches = [10000, 50000, 100000]
        for batches, sps_values, color in [
            (h100_batches, h100_sps, HARDWARE_COLORS["H100"]),
            (h100_scan_batches, h100_scan_sps, HARDWARE_COLORS["H100 Scan"]),
        ]:
            for batch, sps in zip(batches, sps_values):
                if batch in annotation_batches:
                    ax1.annotate(
                        f"{sps:,.0f}",
                        (batch, sps),
                        xytext=(5, 10),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color=color,
                        alpha=0.8,
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor=color,
                            alpha=0.8,
                            linewidth=1,
                        ),
                    )

        ax1.set_xscale("log")
        ax1.set_xlabel("Batch Size (log scale)", fontweight="bold", color="#2D3748")
        ax1.set_ylabel("Steps Per Second", fontweight="bold", color="#2D3748")
        ax1.set_title(
            "H100 Regular vs Scan\nLog Scale Comparison",
            fontweight="bold",
            pad=20,
            color="#2D3748",
        )
        ax1.yaxis.set_major_formatter(FuncFormatter(format_sps))
        ax1.xaxis.set_major_formatter(FuncFormatter(format_batch_size))
        ax1.legend(framealpha=0.95, fontsize=12)
        ax1.grid(True, alpha=0.2, which="both")

    # Plot 2: Linear scale comparison (focus on smaller batches)
    max_batch_linear = 10000
    if h100_batches and h100_scan_batches:
        h100_filtered = [
            (b, s) for b, s in zip(h100_batches, h100_sps) if b <= max_batch_linear
        ]
        scan_filtered = [
            (b, s)
            for b, s in zip(h100_scan_batches, h100_scan_sps)
            if b <= max_batch_linear
        ]

        if h100_filtered:
            h100_filt_batches, h100_filt_sps = zip(*h100_filtered)
            ax2.plot(
                h100_filt_batches,
                h100_filt_sps,
                "o-",
                linewidth=3.5,
                markersize=8,
                label="H100 Regular",
                color=HARDWARE_COLORS["H100"],
                alpha=0.9,
                markerfacecolor="white",
                markeredgewidth=2,
                markeredgecolor=HARDWARE_COLORS["H100"],
            )

        if scan_filtered:
            scan_filt_batches, scan_filt_sps = zip(*scan_filtered)
            ax2.plot(
                scan_filt_batches,
                scan_filt_sps,
                "s-",
                linewidth=3.5,
                markersize=8,
                label="H100 Scan",
                color=HARDWARE_COLORS["H100 Scan"],
                alpha=0.9,
                markerfacecolor="white",
                markeredgewidth=2,
                markeredgecolor=HARDWARE_COLORS["H100 Scan"],
            )

            # Add value annotations for key points in linear view
            for batch, sps in zip(scan_filt_batches, scan_filt_sps):
                if batch in [1000, 5000, 10000]:
                    ax2.annotate(
                        f"{sps:,.0f}",
                        (batch, sps),
                        xytext=(5, 10),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color=HARDWARE_COLORS["H100 Scan"],
                        alpha=0.8,
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor=HARDWARE_COLORS["H100 Scan"],
                            alpha=0.8,
                            linewidth=1,
                        ),
                    )

        if h100_filtered:
            # Add annotations for H100 Regular as well
            for batch, sps in zip(h100_filt_batches, h100_filt_sps):
                if batch in [1000, 5000, 10000]:
                    ax2.annotate(
                        f"{sps:,.0f}",
                        (batch, sps),
                        xytext=(5, -15),
                        textcoords="offset points",
                        ha="left",
                        va="top",
                        fontsize=9,
                        fontweight="bold",
                        color=HARDWARE_COLORS["H100"],
                        alpha=0.8,
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor=HARDWARE_COLORS["H100"],
                            alpha=0.8,
                            linewidth=1,
                        ),
                    )

        ax2.set_xlabel("Batch Size (linear scale)", fontweight="bold", color="#2D3748")
        ax2.set_ylabel("Steps Per Second", fontweight="bold", color="#2D3748")
        ax2.set_title(
            "H100 Regular vs Scan (≤10K batches)\nLinear Scale Comparison",
            fontweight="bold",
            pad=20,
            color="#2D3748",
        )
        ax2.yaxis.set_major_formatter(FuncFormatter(format_sps))
        ax2.xaxis.set_major_formatter(FuncFormatter(format_batch_size))
        ax2.legend(framealpha=0.95, fontsize=12)
        ax2.grid(True, alpha=0.2)

    # Plot 3: Speedup factors (H100 Scan vs H100 Regular)
    if h100_batches and h100_scan_batches:
        h100_performance = dict(zip(h100_batches, h100_sps))
        scan_performance = dict(zip(h100_scan_batches, h100_scan_sps))

        common_batches = sorted(set(h100_batches) & set(h100_scan_batches))[:8]
        speedups = []
        batch_labels = []

        for batch in common_batches:
            if batch in h100_performance and batch in scan_performance:
                speedup = scan_performance[batch] / h100_performance[batch]
                speedups.append(speedup)
                batch_labels.append(format_batch_size(batch, None))

        if speedups:
            bars = ax3.bar(
                range(len(speedups)),
                speedups,
                color=HARDWARE_COLORS["H100 Scan"],
                alpha=0.85,
                edgecolor="#2D3748",
                linewidth=1.5,
            )

            for bar, speedup in zip(bars, speedups):
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.05,
                    f"{speedup:.1f}×",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=11,
                    color="#2D3748",
                )

            ax3.axhline(
                y=1,
                color="black",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label="H100 Regular Baseline (1×)",
            )
            ax3.set_xticks(range(len(batch_labels)))
            ax3.set_xticklabels(batch_labels, rotation=45, ha="right")
            ax3.set_xlabel("Batch Size", fontweight="bold", color="#2D3748")
            ax3.set_ylabel(
                "Speedup Factor (Scan vs Regular)", fontweight="bold", color="#2D3748"
            )
            ax3.set_title(
                "H100 Scan Speedup vs Regular\nPerformance Multiplier",
                fontweight="bold",
                pad=20,
                color="#2D3748",
            )
            ax3.legend(framealpha=0.95, fontsize=12)
            ax3.grid(True, alpha=0.2, axis="y")

    # Plot 4: Performance summary
    if h100_sps and h100_scan_sps:
        h100_peak = max(h100_sps)
        scan_peak = max(h100_scan_sps)

        categories = ["Peak Performance"]
        h100_vals = [h100_peak]
        scan_vals = [scan_peak]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax4.bar(
            x - width / 2,
            h100_vals,
            width,
            label="H100 Regular",
            color=HARDWARE_COLORS["H100"],
            alpha=0.85,
            edgecolor="#2D3748",
        )
        bars2 = ax4.bar(
            x + width / 2,
            scan_vals,
            width,
            label="H100 Scan",
            color=HARDWARE_COLORS["H100 Scan"],
            alpha=0.85,
            edgecolor="#2D3748",
        )

        # Add value labels
        for bar, val in zip(bars1, h100_vals):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                height + height * 0.02,
                f"{val:,.0f}\nSPS",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
                color="#2D3748",
            )

        for bar, val in zip(bars2, scan_vals):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                height + height * 0.02,
                f"{val:,.0f}\nSPS",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
                color="#2D3748",
            )

        improvement = scan_peak / h100_peak
        ax4.text(
            0.5,
            0.95,
            f"Scan is {improvement:.1f}× faster\n({scan_peak:,.0f} vs {h100_peak:,.0f} SPS)",
            transform=ax4.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="#2D3748",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )

        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.set_ylabel("Steps Per Second", fontweight="bold", color="#2D3748")
        ax4.set_title(
            "Peak Performance Summary\nH100 Regular vs Scan",
            fontweight="bold",
            pad=20,
            color="#2D3748",
        )
        ax4.yaxis.set_major_formatter(FuncFormatter(format_sps))
        ax4.legend(framealpha=0.95, fontsize=12)
        ax4.grid(True, alpha=0.2, axis="y")

    plt.suptitle(
        "JaxARC H100 Performance Comparison\nRegular Implementation vs Optimized Scan",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.show()

    # Print summary
    if h100_sps and h100_scan_sps:
        h100_peak = max(h100_sps)
        scan_peak = max(h100_scan_sps)
        improvement = scan_peak / h100_peak

        print("\n⚡ H100 COMPARISON SUMMARY:")
        print("-" * 40)
        print(f"H100 Regular Peak: {h100_peak:>10,.0f} SPS")
        print(f"H100 Scan Peak:    {scan_peak:>10,.0f} SPS")
        print(f"Scan Improvement:  {improvement:>10.1f}× faster")
        print(f"Absolute Gain:     {scan_peak - h100_peak:>10,.0f} SPS")


def main():
    """Main execution function."""
    print("🎯 JaxARC Performance Analysis")
    print("=" * 60)
    print("📊 Generating two types of performance visualizations:")
    print("   1. Mac Evolution: Baseline → Phase 1 → Phase 2")
    print("   2. Hardware Comparison: Mac M2 Pro vs RTX 3090 vs H100 (Phase 2)")

    # Create Mac evolution plots
    plot_mac_evolution()

    # Create hardware comparison plots
    plot_hardware_comparison()

    # Create acceleration factors plot
    plot_acceleration_factors()

    # Create H100 comparison plots
    plot_h100_comparison()

    print("\n✅ Analysis complete!")
    print("🎨 All visualizations generated successfully.")


if __name__ == "__main__":
    main()

# %%
