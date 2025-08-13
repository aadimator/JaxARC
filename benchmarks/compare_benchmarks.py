#!/usr/bin/env python3
"""
Benchmark Comparison Utility for JaxARC

This script compares benchmark results across different optimization phases
to track performance improvements and regressions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def load_benchmark_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all benchmark results from the results directory."""
    results = []

    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                data["filename"] = json_file.name
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    # Sort by timestamp
    results.sort(key=lambda x: x.get("timestamp", 0))
    return results


def format_timestamp(timestamp: float) -> str:
    """Format timestamp for display."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def calculate_improvement(old_value: float, new_value: float) -> str:
    """Calculate and format performance improvement."""
    if old_value == 0:
        return "N/A"

    improvement = (new_value - old_value) / old_value * 100
    if improvement > 0:
        return f"+{improvement:.1f}%"
    return f"{improvement:.1f}%"


def compare_single_env_performance(results: List[Dict[str, Any]]) -> None:
    """Compare single environment performance across benchmarks."""
    print("\nSingle Environment Performance:")
    print("-" * 50)
    print(f"{'Date':<19} {'Description':<30} {'SPS':<10} {'Improvement':<12}")
    print("-" * 50)

    prev_sps = None
    for result in results:
        timestamp = format_timestamp(result.get("timestamp", 0))
        description = result.get("description", "No description")[:28]
        tag = result.get("tag", "")

        if tag:
            description = f"[{tag}] {description}"[:28]

        single_env = result.get("single_environment", {})
        sps = single_env.get("steps_per_second", 0)

        improvement = ""
        if prev_sps is not None:
            improvement = calculate_improvement(prev_sps, sps)

        print(f"{timestamp} {description:<30} {sps:<10.0f} {improvement:<12}")
        prev_sps = sps


def compare_batch_performance(
    results: List[Dict[str, Any]], batch_size: int = 1000
) -> None:
    """Compare batch environment performance for a specific batch size."""
    print(f"\nBatch Environment Performance (Batch Size {batch_size}):")
    print("-" * 50)
    print(f"{'Date':<19} {'Description':<30} {'SPS':<10} {'Improvement':<12}")
    print("-" * 50)

    prev_sps = None
    for result in results:
        timestamp = format_timestamp(result.get("timestamp", 0))
        description = result.get("description", "No description")[:28]
        tag = result.get("tag", "")

        if tag:
            description = f"[{tag}] {description}"[:28]

        batch_scaling = result.get("batch_scaling", {})
        batch_data = batch_scaling.get(str(batch_size), {})
        sps = batch_data.get("steps_per_second", 0)

        improvement = ""
        if prev_sps is not None and prev_sps > 0:
            improvement = calculate_improvement(prev_sps, sps)

        print(f"{timestamp} {description:<30} {sps:<10.0f} {improvement:<12}")
        prev_sps = sps


def show_batch_scaling(results: List[Dict[str, Any]], result_index: int = -1) -> None:
    """Show batch scaling for a specific benchmark result."""
    if not results:
        print("No benchmark results found.")
        return

    result = results[result_index]
    timestamp = format_timestamp(result.get("timestamp", 0))
    description = result.get("description", "No description")

    print("\nBatch Scaling Analysis:")
    print(f"Date: {timestamp}")
    print(f"Description: {description}")
    print("-" * 40)
    print(f"{'Batch Size':<12} {'SPS':<12} {'Efficiency':<12}")
    print("-" * 40)

    batch_scaling = result.get("batch_scaling", {})
    single_sps = result.get("single_environment", {}).get("steps_per_second", 1)

    for batch_size in sorted([int(k) for k in batch_scaling.keys() if k.isdigit()]):
        batch_data = batch_scaling[str(batch_size)]
        sps = batch_data.get("steps_per_second", 0)
        efficiency = sps / (single_sps * batch_size) * 100 if single_sps > 0 else 0

        print(f"{batch_size:<12} {sps:<12.0f} {efficiency:<12.1f}%")


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare JaxARC benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/compare_benchmarks.py
  python benchmarks/compare_benchmarks.py --batch-size 500
  python benchmarks/compare_benchmarks.py --scaling-only
        """,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory containing benchmark results (default: benchmarks/results)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size to compare (default: 1000)",
    )
    parser.add_argument(
        "--scaling-only",
        action="store_true",
        help="Only show batch scaling for the latest result",
    )

    args = parser.parse_args()

    # Load benchmark results
    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}")
        return

    results = load_benchmark_results(args.results_dir)

    if not results:
        print(f"No benchmark results found in {args.results_dir}")
        return

    print("=" * 60)
    print("JAXARC BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"Found {len(results)} benchmark results")

    if args.scaling_only:
        show_batch_scaling(results)
    else:
        compare_single_env_performance(results)
        compare_batch_performance(results, args.batch_size)

        if len(results) > 1:
            print("\nLatest Batch Scaling:")
            show_batch_scaling(results, -1)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
