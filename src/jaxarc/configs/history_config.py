from __future__ import annotations

import equinox as eqx
from omegaconf import DictConfig

from ..utils.jax_types import (
    get_selection_data_size,
)


class HistoryConfig(eqx.Module):
    """Configuration for action history tracking.

    This configuration controls how action history is stored and managed,
    providing options for memory optimization and different storage formats.

    Attributes:
        enabled: Whether action history tracking is enabled
        max_history_length: Maximum number of actions to store in history
        store_selection_data: Whether to store full selection data (memory-intensive)
        store_intermediate_grids: Whether to store intermediate grid states (very memory-intensive)
        compress_repeated_actions: Whether to compress repeated identical actions

    Examples:
        ```python
        # Memory-efficient configuration
        config = HistoryConfig(
            enabled=True,
            max_history_length=500,
            store_selection_data=False,  # Save memory
            store_intermediate_grids=False,
            compress_repeated_actions=True
        )

        # Full tracking configuration
        config = HistoryConfig(
            enabled=True,
            max_history_length=1000,
            store_selection_data=True,
            store_intermediate_grids=False,  # Still too memory-intensive for most uses
            compress_repeated_actions=False
        )
        ```
    """

    enabled: bool = True
    max_history_length: int = 1000
    store_selection_data: bool = True
    store_intermediate_grids: bool = False  # Memory-intensive option
    compress_repeated_actions: bool = True

    @classmethod
    def from_hydra(cls, hydra_config: DictConfig) -> HistoryConfig:
        """Create HistoryConfig from Hydra DictConfig.
        Args:
            hydra_config: Hydra configuration dictionary
        Returns:
            HistoryConfig instance
        """
        return cls(
            enabled=hydra_config.get("enabled", True),
            max_history_length=hydra_config.get("max_history_length", 1000),
            store_selection_data=hydra_config.get("store_selection_data", True),
            store_intermediate_grids=hydra_config.get(
                "store_intermediate_grids", False
            ),
            compress_repeated_actions=hydra_config.get(
                "compress_repeated_actions", True
            ),
        )

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_history_length <= 0:
            msg = f"max_history_length must be positive, got {self.max_history_length}"
            raise ValueError(
                msg
            )

        if self.max_history_length > 10000:
            # Warn about very large history lengths
            import warnings

            warnings.warn(
                f"Large history length ({self.max_history_length}) may impact memory usage",
                UserWarning,
            )

    def estimate_memory_usage(
        self,
        selection_format: str = "mask",
        max_grid_height: int = 30,
        max_grid_width: int = 30,
        dtype_size: int = 4,  # bytes per float32
    ) -> dict:
        """Estimate memory usage for different configuration options.

        This method calculates the approximate memory usage for action history
        storage based on the configuration parameters and dataset characteristics.

        Args:
            selection_format: Selection format ("point", "bbox", "mask")
            max_grid_height: Maximum grid height for the dataset
            max_grid_width: Maximum grid width for the dataset
            dtype_size: Size in bytes of the data type used (4 for float32)

        Returns:
            Dictionary containing memory usage estimates in bytes and human-readable format

        Examples:
            ```python
            # Estimate memory for MiniARC with point selection
            config = HistoryConfig(max_history_length=1000, store_selection_data=True)
            usage = config.estimate_memory_usage("point", 5, 5)
            print(f"Memory usage: {usage['human_readable']}")

            # Compare different configurations
            config_full = HistoryConfig(store_selection_data=True, store_intermediate_grids=True)
            config_minimal = HistoryConfig(store_selection_data=False, store_intermediate_grids=False)

            usage_full = config_full.estimate_memory_usage("mask", 30, 30)
            usage_minimal = config_minimal.estimate_memory_usage("mask", 30, 30)

            print(f"Full storage: {usage_full['human_readable']}")
            print(f"Minimal storage: {usage_minimal['human_readable']}")
            ```
        """
        if not self.enabled:
            return {
                "total_bytes": 0,
                "human_readable": "0 B (history disabled)",
                "breakdown": {"history_disabled": True},
            }

        # Calculate base record size
        selection_size = get_selection_data_size(
            selection_format, max_grid_height, max_grid_width
        )
        metadata_fields = 4  # operation_id, timestamp, pair_index, valid

        if self.store_selection_data:
            record_size = selection_size + metadata_fields
        else:
            # Only store metadata if selection data is disabled
            record_size = metadata_fields

        # Base history storage
        history_bytes = self.max_history_length * record_size * dtype_size

        # Intermediate grids storage (very memory-intensive)
        intermediate_grids_bytes = 0
        if self.store_intermediate_grids:
            grid_size = max_grid_height * max_grid_width
            # Store both working grid and target grid for each action
            intermediate_grids_bytes = (
                self.max_history_length * grid_size * 2 * dtype_size
            )

        total_bytes = history_bytes + intermediate_grids_bytes

        # Create breakdown
        breakdown = {
            "action_records_bytes": history_bytes,
            "intermediate_grids_bytes": intermediate_grids_bytes,
            "record_size_fields": record_size,
            "selection_data_fields": selection_size if self.store_selection_data else 0,
            "metadata_fields": metadata_fields,
            "max_history_length": self.max_history_length,
            "selection_format": selection_format,
            "grid_dimensions": f"{max_grid_height}x{max_grid_width}",
        }

        return {
            "total_bytes": total_bytes,
            "human_readable": self._format_bytes(total_bytes),
            "breakdown": breakdown,
        }

    def compare_memory_configurations(
        self,
        other_configs: list[HistoryConfig],
        selection_format: str = "mask",
        max_grid_height: int = 30,
        max_grid_width: int = 30,
    ) -> dict:
        """Compare memory usage across different history configurations.

        This method provides a comparative analysis of memory usage for different
        configuration options, helping users choose the most appropriate settings.

        Args:
            other_configs: List of other HistoryConfig instances to compare against
            selection_format: Selection format for comparison
            max_grid_height: Maximum grid height
            max_grid_width: Maximum grid width

        Returns:
            Dictionary containing comparative memory usage analysis

        Examples:
            ```python
            # Compare different configuration strategies
            base_config = HistoryConfig(max_history_length=1000, store_selection_data=True)

            configs_to_compare = [
                HistoryConfig(max_history_length=500, store_selection_data=True),
                HistoryConfig(max_history_length=1000, store_selection_data=False),
                HistoryConfig(max_history_length=2000, store_selection_data=True),
            ]

            comparison = base_config.compare_memory_configurations(configs_to_compare, "point", 5, 5)

            for i, result in enumerate(comparison['comparisons']):
                print(f"Config {i}: {result['memory']['human_readable']}")
            ```
        """
        base_usage = self.estimate_memory_usage(
            selection_format, max_grid_height, max_grid_width
        )

        comparisons = []
        for i, config in enumerate(other_configs):
            usage = config.estimate_memory_usage(
                selection_format, max_grid_height, max_grid_width
            )

            # Calculate relative difference
            if base_usage["total_bytes"] > 0:
                relative_change = (
                    usage["total_bytes"] - base_usage["total_bytes"]
                ) / base_usage["total_bytes"]
            else:
                relative_change = float("inf") if usage["total_bytes"] > 0 else 0.0

            comparisons.append(
                {
                    "config_index": i,
                    "config": {
                        "enabled": config.enabled,
                        "max_history_length": config.max_history_length,
                        "store_selection_data": config.store_selection_data,
                        "store_intermediate_grids": config.store_intermediate_grids,
                        "compress_repeated_actions": config.compress_repeated_actions,
                    },
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

        # Find most and least memory-efficient configurations
        if comparisons:
            most_efficient = min(comparisons, key=lambda x: x["memory"]["total_bytes"])
            least_efficient = max(comparisons, key=lambda x: x["memory"]["total_bytes"])
        else:
            most_efficient = least_efficient = None

        return {
            "base_config": {
                "config": {
                    "enabled": self.enabled,
                    "max_history_length": self.max_history_length,
                    "store_selection_data": self.store_selection_data,
                    "store_intermediate_grids": self.store_intermediate_grids,
                    "compress_repeated_actions": self.compress_repeated_actions,
                },
                "memory": base_usage,
            },
            "comparisons": comparisons,
            "summary": {
                "most_efficient": most_efficient,
                "least_efficient": least_efficient,
                "total_configs_compared": len(comparisons),
            },
        }

    def get_recommended_config(
        self,
        selection_format: str = "mask",
        max_grid_height: int = 30,
        max_grid_width: int = 30,
        memory_budget_mb: float = 100.0,
        use_case: str = "development",  # "development", "training", "evaluation"
    ) -> HistoryConfig:
        """Get recommended configuration based on memory budget and use case.

        This method provides intelligent configuration recommendations based on
        memory constraints and intended use case.

        Args:
            selection_format: Selection format for sizing calculations
            max_grid_height: Maximum grid height
            max_grid_width: Maximum grid width
            memory_budget_mb: Memory budget in megabytes
            use_case: Intended use case ("development", "training", "evaluation")

        Returns:
            Recommended HistoryConfig instance

        Examples:
            ```python
            # Get config for development with limited memory
            config = HistoryConfig().get_recommended_config(
                selection_format="point",
                max_grid_height=5,
                max_grid_width=5,
                memory_budget_mb=10.0,
                use_case="development"
            )

            # Get config for training with more memory
            config = HistoryConfig().get_recommended_config(
                selection_format="mask",
                max_grid_height=30,
                max_grid_width=30,
                memory_budget_mb=500.0,
                use_case="training"
            )
            ```
        """
        memory_budget_bytes = memory_budget_mb * 1024 * 1024  # Convert MB to bytes

        # Use case specific defaults
        if use_case == "development":
            base_config = HistoryConfig(
                enabled=True,
                max_history_length=100,
                store_selection_data=True,
                store_intermediate_grids=False,
                compress_repeated_actions=True,
            )
        elif use_case == "training":
            base_config = HistoryConfig(
                enabled=True,
                max_history_length=1000,
                store_selection_data=True,
                store_intermediate_grids=False,
                compress_repeated_actions=True,
            )
        elif use_case == "evaluation":
            base_config = HistoryConfig(
                enabled=True,
                max_history_length=500,
                store_selection_data=False,  # Less memory for evaluation
                store_intermediate_grids=False,
                compress_repeated_actions=True,
            )
        else:
            # Default configuration
            base_config = HistoryConfig()

        # Check if base config fits in budget
        usage = base_config.estimate_memory_usage(
            selection_format, max_grid_height, max_grid_width
        )

        if usage["total_bytes"] <= memory_budget_bytes:
            return base_config

        # If base config exceeds budget, try to reduce memory usage
        # Start with the most memory-intensive options

        # First, disable intermediate grids if enabled
        if base_config.store_intermediate_grids:
            reduced_config = HistoryConfig(
                enabled=base_config.enabled,
                max_history_length=base_config.max_history_length,
                store_selection_data=base_config.store_selection_data,
                store_intermediate_grids=False,
                compress_repeated_actions=base_config.compress_repeated_actions,
            )
            usage = reduced_config.estimate_memory_usage(
                selection_format, max_grid_height, max_grid_width
            )
            if usage["total_bytes"] <= memory_budget_bytes:
                return reduced_config
            base_config = reduced_config

        # Next, disable selection data storage
        if base_config.store_selection_data:
            reduced_config = HistoryConfig(
                enabled=base_config.enabled,
                max_history_length=base_config.max_history_length,
                store_selection_data=False,
                store_intermediate_grids=base_config.store_intermediate_grids,
                compress_repeated_actions=base_config.compress_repeated_actions,
            )
            usage = reduced_config.estimate_memory_usage(
                selection_format, max_grid_height, max_grid_width
            )
            if usage["total_bytes"] <= memory_budget_bytes:
                return reduced_config
            base_config = reduced_config

        # Finally, reduce history length
        selection_size = get_selection_data_size(
            selection_format, max_grid_height, max_grid_width
        )
        metadata_fields = 4
        record_size = (
            metadata_fields
            if not base_config.store_selection_data
            else selection_size + metadata_fields
        )
        dtype_size = 4  # float32

        max_affordable_length = int(memory_budget_bytes / (record_size * dtype_size))
        max_affordable_length = max(1, max_affordable_length)  # At least 1 action

        return HistoryConfig(
            enabled=base_config.enabled,
            max_history_length=max_affordable_length,
            store_selection_data=base_config.store_selection_data,
            store_intermediate_grids=base_config.store_intermediate_grids,
            compress_repeated_actions=base_config.compress_repeated_actions,
        )

    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format byte count into human-readable string.

        Args:
            bytes_value: Number of bytes

        Returns:
            Human-readable string representation
        """
        if bytes_value == 0:
            return "0 B"

        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        value = float(bytes_value)

        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024
            unit_index += 1

        if unit_index == 0:
            return f"{int(value)} {units[unit_index]}"
        return f"{value:.2f} {units[unit_index]}"
