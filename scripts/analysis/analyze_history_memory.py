from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import typer

from jaxarc.configs.history_config import HistoryConfig
from jaxarc.utils.jax_types import get_selection_data_size

# Standalone analysis utilities for HistoryConfig memory planning.
# This file intentionally lives outside the library path and can be executed
# as a one-off helper using the project's Python environment.


def format_bytes(bytes_value: int) -> str:
    if bytes_value == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    value = float(bytes_value)
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    return (
        f"{int(value)} {units[unit_index]}"
        if unit_index == 0
        else f"{value:.2f} {units[unit_index]}"
    )


def estimate_memory_usage(
    config: HistoryConfig,
    selection_format: str = "mask",
    max_grid_height: int = 30,
    max_grid_width: int = 30,
    dtype_size: int = 4,  # bytes per float32
) -> dict:
    if not config.enabled:
        return {
            "total_bytes": 0,
            "human_readable": "0 B (history disabled)",
            "breakdown": {"history_disabled": True},
        }

    selection_size = get_selection_data_size(
        selection_format, max_grid_height, max_grid_width
    )
    metadata_fields = 4  # operation_id, timestamp, pair_index, valid

    record_size = (
        (selection_size + metadata_fields)
        if config.store_selection_data
        else metadata_fields
    )
    history_bytes = config.max_history_length * record_size * dtype_size

    intermediate_grids_bytes = 0
    if config.store_intermediate_grids:
        grid_size = max_grid_height * max_grid_width
        intermediate_grids_bytes = (
            config.max_history_length * grid_size * 2 * dtype_size
        )

    total_bytes = history_bytes + intermediate_grids_bytes
    breakdown = {
        "action_records_bytes": history_bytes,
        "intermediate_grids_bytes": intermediate_grids_bytes,
        "record_size_fields": record_size,
        "selection_data_fields": selection_size if config.store_selection_data else 0,
        "metadata_fields": metadata_fields,
        "max_history_length": config.max_history_length,
        "selection_format": selection_format,
        "grid_dimensions": f"{max_grid_height}x{max_grid_width}",
    }
    return {
        "total_bytes": total_bytes,
        "human_readable": format_bytes(total_bytes),
        "breakdown": breakdown,
    }


def compare_memory_configurations(
    base: HistoryConfig,
    others: Iterable[HistoryConfig],
    selection_format: str = "mask",
    max_grid_height: int = 30,
    max_grid_width: int = 30,
) -> dict:
    base_usage = estimate_memory_usage(
        base, selection_format, max_grid_height, max_grid_width
    )
    comparisons = []
    for i, other_cfg in enumerate(others):
        usage = estimate_memory_usage(
            other_cfg, selection_format, max_grid_height, max_grid_width
        )
        relative_change = (
            (usage["total_bytes"] - base_usage["total_bytes"])
            / base_usage["total_bytes"]
            if base_usage["total_bytes"] > 0
            else (float("inf") if usage["total_bytes"] > 0 else 0.0)
        )
        comparisons.append(
            {
                "config_index": i,
                "config": cfg_to_dict(other_cfg),
                "memory": usage,
                "relative_to_base": {
                    "bytes_difference": usage["total_bytes"]
                    - base_usage["total_bytes"],
                    "percentage_change": relative_change * 100,
                    "is_more_efficient": usage["total_bytes"]
                    < base_usage["total_bytes"],
                },
            }
        )

    most_efficient = (
        min(comparisons, key=lambda x: x["memory"]["total_bytes"])
        if comparisons
        else None
    )
    least_efficient = (
        max(comparisons, key=lambda x: x["memory"]["total_bytes"])
        if comparisons
        else None
    )

    return {
        "base_config": {"config": cfg_to_dict(base), "memory": base_usage},
        "comparisons": comparisons,
        "summary": {
            "most_efficient": most_efficient,
            "least_efficient": least_efficient,
            "total_configs_compared": len(comparisons),
        },
    }


def get_recommended_config(
    selection_format: str = "mask",
    max_grid_height: int = 30,
    max_grid_width: int = 30,
    memory_budget_mb: float = 100.0,
    use_case: str = "development",  # "development", "training", "evaluation"
) -> HistoryConfig:
    memory_budget_bytes = memory_budget_mb * 1024 * 1024

    if use_case == "development":
        base = HistoryConfig(True, 100, True, False, True)
    elif use_case == "training":
        base = HistoryConfig(True, 1000, True, False, True)
    elif use_case == "evaluation":
        base = HistoryConfig(True, 500, False, False, True)
    else:
        base = HistoryConfig()

    usage = estimate_memory_usage(
        base, selection_format, max_grid_height, max_grid_width
    )
    if usage["total_bytes"] <= memory_budget_bytes:
        return base

    if base.store_intermediate_grids:
        reduced = HistoryConfig(
            base.enabled,
            base.max_history_length,
            base.store_selection_data,
            False,
            base.compress_repeated_actions,
        )
        if (
            estimate_memory_usage(
                reduced, selection_format, max_grid_height, max_grid_width
            )["total_bytes"]
            <= memory_budget_bytes
        ):
            return reduced
        base = reduced

    if base.store_selection_data:
        reduced = HistoryConfig(
            base.enabled,
            base.max_history_length,
            False,
            base.store_intermediate_grids,
            base.compress_repeated_actions,
        )
        if (
            estimate_memory_usage(
                reduced, selection_format, max_grid_height, max_grid_width
            )["total_bytes"]
            <= memory_budget_bytes
        ):
            return reduced
        base = reduced

    selection_size = get_selection_data_size(
        selection_format, max_grid_height, max_grid_width
    )
    metadata_fields = 4
    record_size = (
        metadata_fields
        if not base.store_selection_data
        else selection_size + metadata_fields
    )
    dtype_size = 4
    max_affordable_length = max(
        1, int(memory_budget_bytes / (record_size * dtype_size))
    )

    return HistoryConfig(
        base.enabled,
        max_affordable_length,
        base.store_selection_data,
        base.store_intermediate_grids,
        base.compress_repeated_actions,
    )


def cfg_to_dict(config: HistoryConfig) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "max_history_length": config.max_history_length,
        "store_selection_data": config.store_selection_data,
        "store_intermediate_grids": config.store_intermediate_grids,
        "compress_repeated_actions": config.compress_repeated_actions,
    }


def main(
    selection_format: str = typer.Option(
        "mask", "--selection-format", case_sensitive=False
    ),
    grid_h: int = typer.Option(30, "--grid-h"),
    grid_w: int = typer.Option(30, "--grid-w"),
    budget_mb: float = typer.Option(100.0, "--budget-mb"),
    use_case: str = typer.Option("development", "--use-case", case_sensitive=False),
    custom: bool = typer.Option(
        False, "--custom", help="Use custom HistoryConfig values"
    ),
    enabled: bool = typer.Option(True, "--enabled"),
    max_history: int = typer.Option(1000, "--max-history"),
    store_selection: bool = typer.Option(True, "--store-selection"),
    store_grids: bool = typer.Option(False, "--store-grids"),
    compress: bool = typer.Option(True, "--compress"),
    compare: bool = typer.Option(
        False, "--compare", help="Run a small comparison sweep"
    ),
):
    if custom or use_case == "custom":
        config = HistoryConfig(
            enabled=enabled,
            max_history_length=int(max_history),
            store_selection_data=store_selection,
            store_intermediate_grids=store_grids,
            compress_repeated_actions=compress,
        )
    else:
        config = get_recommended_config(
            selection_format=selection_format,
            max_grid_height=grid_h,
            max_grid_width=grid_w,
            memory_budget_mb=budget_mb,
            use_case=use_case,
        )

    result = estimate_memory_usage(config, selection_format, grid_h, grid_w)
    typer.echo(json.dumps({"config": cfg_to_dict(config), "usage": result}, indent=2))

    if compare:
        sweep = [
            config,
            HistoryConfig(
                config.enabled,
                max(config.max_history_length // 2, 1),
                config.store_selection_data,
                config.store_intermediate_grids,
                config.compress_repeated_actions,
            ),
            HistoryConfig(
                config.enabled,
                config.max_history_length,
                not config.store_selection_data,
                config.store_intermediate_grids,
                config.compress_repeated_actions,
            ),
            HistoryConfig(
                config.enabled,
                max(config.max_history_length // 4, 1),
                False,
                False,
                True,
            ),
        ]
        cmp_res = compare_memory_configurations(
            config, sweep, selection_format, grid_h, grid_w
        )
        typer.echo(json.dumps(cmp_res, indent=2))


if __name__ == "__main__":
    typer.run(main)
