"""
Action history tracking for the enhanced ARC environment.

This module provides the ActionHistoryTracker class for managing action sequences
with JAX-compatible fixed-size storage. It supports circular buffer behavior,
proper indexing, and overflow handling while maintaining full JAX compatibility.

Key Features:
- Fixed-size action storage with circular buffer behavior
- JAX-compatible data structures with static shapes
- Configurable history length and storage format
- Efficient action sequence retrieval and management
- Memory-optimized storage with optional selection data compression

Examples:
    ```python
    from jaxarc.envs.action_history import ActionHistoryTracker, HistoryConfig
    from jaxarc.state import ArcEnvState
    
    # Create history tracker with configuration
    config = HistoryConfig(
        enabled=True,
        max_history_length=1000,
        store_selection_data=True,
        compress_repeated_actions=True
    )
    
    tracker = ActionHistoryTracker()
    
    # Add action to history
    from jaxarc.envs.structured_actions import create_mask_action
    action = create_mask_action(operation=operation_id, selection=selection_data)
    new_state = tracker.add_action(state, action, config)
    
    # Retrieve action sequence
    sequence = tracker.get_action_sequence(new_state, start_idx=0, end_idx=10)
    
    # Clear history for new episode
    clean_state = tracker.clear_history(new_state)
    ```
"""

from __future__ import annotations

from typing import Optional

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int
from omegaconf import DictConfig

from ..state import ArcEnvState
from ..utils.jax_types import (
    NUM_OPERATIONS,
    ActionSequence,
    EpisodeIndex,
    HistoryLength,
    OperationId,
    SelectionData,
    StepCount,
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
    def from_hydra(cls, hydra_config: DictConfig) -> "HistoryConfig":
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
            store_intermediate_grids=hydra_config.get("store_intermediate_grids", False),
            compress_repeated_actions=hydra_config.get("compress_repeated_actions", True),
        )
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_history_length <= 0:
            raise ValueError(f"max_history_length must be positive, got {self.max_history_length}")
        
        if self.max_history_length > 10000:
            # Warn about very large history lengths
            import warnings
            warnings.warn(
                f"Large history length ({self.max_history_length}) may impact memory usage",
                UserWarning
            )
    
    def estimate_memory_usage(
        self,
        selection_format: str = "mask",
        max_grid_height: int = 30,
        max_grid_width: int = 30,
        dtype_size: int = 4  # bytes per float32
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
                "breakdown": {"history_disabled": True}
            }
        
        # Calculate base record size
        selection_size = get_selection_data_size(selection_format, max_grid_height, max_grid_width)
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
            intermediate_grids_bytes = self.max_history_length * grid_size * 2 * dtype_size
        
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
            "breakdown": breakdown
        }
    
    def compare_memory_configurations(
        self,
        other_configs: list["HistoryConfig"],
        selection_format: str = "mask",
        max_grid_height: int = 30,
        max_grid_width: int = 30
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
        base_usage = self.estimate_memory_usage(selection_format, max_grid_height, max_grid_width)
        
        comparisons = []
        for i, config in enumerate(other_configs):
            usage = config.estimate_memory_usage(selection_format, max_grid_height, max_grid_width)
            
            # Calculate relative difference
            if base_usage["total_bytes"] > 0:
                relative_change = (usage["total_bytes"] - base_usage["total_bytes"]) / base_usage["total_bytes"]
            else:
                relative_change = float('inf') if usage["total_bytes"] > 0 else 0.0
            
            comparisons.append({
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
                    "bytes_difference": usage["total_bytes"] - base_usage["total_bytes"],
                    "percentage_change": relative_change * 100,
                    "is_more_efficient": usage["total_bytes"] < base_usage["total_bytes"],
                }
            })
        
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
            }
        }
    
    def get_recommended_config(
        self,
        selection_format: str = "mask",
        max_grid_height: int = 30,
        max_grid_width: int = 30,
        memory_budget_mb: float = 100.0,
        use_case: str = "development"  # "development", "training", "evaluation"
    ) -> "HistoryConfig":
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
                compress_repeated_actions=True
            )
        elif use_case == "training":
            base_config = HistoryConfig(
                enabled=True,
                max_history_length=1000,
                store_selection_data=True,
                store_intermediate_grids=False,
                compress_repeated_actions=True
            )
        elif use_case == "evaluation":
            base_config = HistoryConfig(
                enabled=True,
                max_history_length=500,
                store_selection_data=False,  # Less memory for evaluation
                store_intermediate_grids=False,
                compress_repeated_actions=True
            )
        else:
            # Default configuration
            base_config = HistoryConfig()
        
        # Check if base config fits in budget
        usage = base_config.estimate_memory_usage(selection_format, max_grid_height, max_grid_width)
        
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
                compress_repeated_actions=base_config.compress_repeated_actions
            )
            usage = reduced_config.estimate_memory_usage(selection_format, max_grid_height, max_grid_width)
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
                compress_repeated_actions=base_config.compress_repeated_actions
            )
            usage = reduced_config.estimate_memory_usage(selection_format, max_grid_height, max_grid_width)
            if usage["total_bytes"] <= memory_budget_bytes:
                return reduced_config
            base_config = reduced_config
        
        # Finally, reduce history length
        selection_size = get_selection_data_size(selection_format, max_grid_height, max_grid_width)
        metadata_fields = 4
        record_size = metadata_fields if not base_config.store_selection_data else selection_size + metadata_fields
        dtype_size = 4  # float32
        
        max_affordable_length = int(memory_budget_bytes / (record_size * dtype_size))
        max_affordable_length = max(1, max_affordable_length)  # At least 1 action
        
        return HistoryConfig(
            enabled=base_config.enabled,
            max_history_length=max_affordable_length,
            store_selection_data=base_config.store_selection_data,
            store_intermediate_grids=base_config.store_intermediate_grids,
            compress_repeated_actions=base_config.compress_repeated_actions
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
        else:
            return f"{value:.2f} {units[unit_index]}"


@chex.dataclass
class ActionRecord:
    """Comprehensive action record with selection data and metadata.
    
    This structure represents a single action in the history with all the
    information needed to reconstruct, analyze, and validate the action sequence.
    The selection_data size is optimized based on the selection format and
    dataset configuration for maximum efficiency and semantic meaning.
    
    The record implements proper padding and masking for JAX compatibility,
    ensuring static shapes while supporting variable-length data through
    the valid flag and appropriate padding strategies.
    
    Attributes:
        selection_data: Optimally sized selection data based on format
        operation_id: ARC operation ID (0-34)
        timestamp: Step count when action was taken
        pair_index: Which demo/test pair this action was on
        valid: Whether this record contains valid data (for padding)
        
    Selection Data Formats:
        - Point actions: [row, col] (2 elements) - preserves exact coordinates
        - Bbox actions: [r1, c1, r2, c2] (4 elements) - preserves exact bbox
        - Mask actions: flattened mask (grid_size^2 elements) - full spatial info
        
    JAX Compatibility:
        - All fields use static shapes with appropriate padding
        - Variable-length data handled through masking with 'valid' flag
        - Pure functional methods for validation and manipulation
        - JIT-compilable validation and helper methods
        
    Examples:
        # MiniARC point action: 2 + 4 = 6 total fields
        record = ActionRecord(
            selection_data=jnp.array([2.0, 3.0]),  # [row, col]
            operation_id=jnp.array(15),
            timestamp=jnp.array(10),
            pair_index=jnp.array(0),
            valid=jnp.array(True)
        )
        
        # Full ARC mask action: 900 + 4 = 904 total fields
        record = ActionRecord(
            selection_data=flattened_mask,  # 900 elements
            operation_id=jnp.array(25),
            timestamp=jnp.array(15),
            pair_index=jnp.array(1),
            valid=jnp.array(True)
        )
        
        # Validate record integrity
        is_valid = record.validate_integrity("mask", 30, 30)
        
        # Extract selection coordinates for point actions
        coords = record.get_selection_coordinates("point")
    """
    
    selection_data: SelectionData  # Flattened selection (point/bbox/mask)
    operation_id: OperationId
    timestamp: StepCount
    pair_index: EpisodeIndex
    valid: Bool[Array, ""]  # Whether this record contains valid data
    
    def validate_integrity(
        self,
        selection_format: str = "mask",
        max_grid_height: int = 30,
        max_grid_width: int = 30
    ) -> Bool[Array, ""]:
        """Validate the integrity of this action record.
        
        This method performs comprehensive validation of the action record,
        checking that all fields are within valid ranges and that the
        selection data is properly formatted for the specified format.
        
        Args:
            selection_format: Expected selection format ("point", "bbox", "mask")
            max_grid_height: Maximum grid height for validation
            max_grid_width: Maximum grid width for validation
            
        Returns:
            JAX boolean scalar indicating if the record is valid
            
        Examples:
            ```python
            # Validate point action record
            is_valid = record.validate_integrity("point", 5, 5)
            
            # Validate mask action record
            is_valid = record.validate_integrity("mask", 30, 30)
            ```
        """
        # Use JAX where instead of Python if for JIT compatibility
        base_valid = jnp.where(
            self.valid,
            jnp.array(True),
            jnp.array(False)
        )
        
        # Validate operation ID range
        operation_valid = (self.operation_id >= 0) & (self.operation_id < NUM_OPERATIONS)
        
        # Validate timestamp is non-negative
        timestamp_valid = self.timestamp >= 0
        
        # Validate pair index is non-negative
        pair_index_valid = self.pair_index >= 0
        
        # Validate selection data based on format
        selection_valid = self._validate_selection_data(
            selection_format, max_grid_height, max_grid_width
        )
        
        return base_valid & operation_valid & timestamp_valid & pair_index_valid & selection_valid
    
    def validate_integrity_detailed(
        self,
        selection_format: str = "mask",
        max_grid_height: int = 30,
        max_grid_width: int = 30
    ) -> tuple[bool, str]:
        """Detailed validation with error messages (not JIT-compatible).
        
        This method provides detailed validation with specific error messages,
        but is not JIT-compatible due to string operations and Python control flow.
        
        Args:
            selection_format: Expected selection format ("point", "bbox", "mask")
            max_grid_height: Maximum grid height for validation
            max_grid_width: Maximum grid width for validation
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not bool(self.valid):
            return False, "Record is marked as invalid (padding record)"
        
        # Validate operation ID range
        if not (0 <= int(self.operation_id) < NUM_OPERATIONS):
            return False, f"Operation ID {int(self.operation_id)} is out of range [0, {NUM_OPERATIONS-1}]"
        
        # Validate timestamp is non-negative
        if int(self.timestamp) < 0:
            return False, f"Timestamp {int(self.timestamp)} is negative"
        
        # Validate pair index is non-negative
        if int(self.pair_index) < 0:
            return False, f"Pair index {int(self.pair_index)} is negative"
        
        # Validate selection data based on format
        expected_size = get_selection_data_size(selection_format, max_grid_height, max_grid_width)
        if self.selection_data.shape[0] != expected_size:
            return False, f"Selection data size {self.selection_data.shape[0]} != expected {expected_size}"
        
        if selection_format == "point":
            if expected_size >= 2:
                row, col = float(self.selection_data[0]), float(self.selection_data[1])
                if not (0 <= row < max_grid_height):
                    return False, f"Point row {row} is out of bounds [0, {max_grid_height})"
                if not (0 <= col < max_grid_width):
                    return False, f"Point col {col} is out of bounds [0, {max_grid_width})"
        elif selection_format == "bbox":
            if expected_size >= 4:
                r1, c1, r2, c2 = [float(x) for x in self.selection_data[:4]]
                if not (0 <= r1 < max_grid_height and 0 <= r2 < max_grid_height):
                    return False, f"Bbox rows [{r1}, {r2}] are out of bounds [0, {max_grid_height})"
                if not (0 <= c1 < max_grid_width and 0 <= c2 < max_grid_width):
                    return False, f"Bbox cols [{c1}, {c2}] are out of bounds [0, {max_grid_width})"
                if not (r1 <= r2 and c1 <= c2):
                    return False, f"Bbox coordinates not properly ordered: ({r1}, {c1}) to ({r2}, {c2})"
        elif selection_format == "mask":
            if not jnp.all((self.selection_data >= 0.0) & (self.selection_data <= 1.0)):
                return False, "Mask values are not in [0, 1] range"
        
        return True, "Valid"
    
    def _validate_selection_data(
        self,
        selection_format: str,
        max_grid_height: int,
        max_grid_width: int
    ) -> Bool[Array, ""]:
        """Validate selection data for the specified format.
        
        Args:
            selection_format: Selection format to validate against
            max_grid_height: Maximum grid height
            max_grid_width: Maximum grid width
            
        Returns:
            JAX boolean scalar indicating if selection data is valid
        """
        expected_size = get_selection_data_size(selection_format, max_grid_height, max_grid_width)
        
        # Check if selection data has the expected size
        size_valid = self.selection_data.shape[0] == expected_size
        
        # For JAX compatibility, we'll do basic validation that works for all formats
        # More specific validation can be done outside of JIT-compiled functions
        
        # Basic validation: all values should be finite and non-negative
        basic_valid = jnp.all(jnp.isfinite(self.selection_data)) & jnp.all(self.selection_data >= 0.0)
        
        # For point and bbox formats, check bounds
        bounds_valid = jnp.where(
            expected_size <= 4,  # Point (2) or bbox (4) format
            jnp.all(self.selection_data < jnp.maximum(max_grid_height, max_grid_width)),
            jnp.all(self.selection_data <= 1.0)  # Mask format should be in [0, 1]
        )
        
        return size_valid & basic_valid & bounds_valid
    
    def is_valid_record(self) -> Bool[Array, ""]:
        """Check if this is a valid (non-padding) record.
        
        Returns:
            JAX boolean scalar indicating if this record contains valid data
        """
        return self.valid
    
    def get_selection_coordinates(self, selection_format: str) -> Array:
        """Extract selection coordinates based on format.
        
        This method extracts meaningful coordinate information from the
        selection data based on the specified format.
        
        Args:
            selection_format: Selection format ("point", "bbox", "mask")
            
        Returns:
            JAX array containing coordinates:
            - Point: [row, col]
            - Bbox: [r1, c1, r2, c2]  
            - Mask: indices of selected cells (flattened)
            
        Examples:
            ```python
            # Get point coordinates
            coords = record.get_selection_coordinates("point")  # [row, col]
            
            # Get bbox coordinates
            coords = record.get_selection_coordinates("bbox")   # [r1, c1, r2, c2]
            
            # Get mask selection indices
            coords = record.get_selection_coordinates("mask")   # [idx1, idx2, ...]
            ```
        """
        if selection_format == "point":
            return self.selection_data[:2]  # [row, col]
        elif selection_format == "bbox":
            return self.selection_data[:4]  # [r1, c1, r2, c2]
        elif selection_format == "mask":
            # Return indices of selected cells (where mask > 0.5)
            return jnp.where(self.selection_data > 0.5, size=self.selection_data.shape[0])[0]
        else:
            error_msg = f"Unknown selection format: {selection_format}"
            raise ValueError(error_msg)
    
    def get_metadata_summary(self) -> dict:
        """Get a summary of the record metadata.
        
        Note: This method converts JAX arrays to Python types for readability.
        For JAX-compatible operations, access fields directly.
        
        Returns:
            Dictionary containing record metadata
        """
        return {
            "operation_id": int(self.operation_id),
            "timestamp": int(self.timestamp),
            "pair_index": int(self.pair_index),
            "valid": bool(self.valid),
            "selection_data_size": self.selection_data.shape[0],
        }
    
    def matches_pair(self, pair_index: int) -> Bool[Array, ""]:
        """Check if this record was taken on the specified pair.
        
        Args:
            pair_index: Pair index to check against
            
        Returns:
            JAX boolean scalar indicating if record matches the pair
        """
        return self.pair_index == pair_index
    
    def is_after_timestamp(self, timestamp: int) -> Bool[Array, ""]:
        """Check if this record was taken after the specified timestamp.
        
        Args:
            timestamp: Timestamp to compare against
            
        Returns:
            JAX boolean scalar indicating if record is after timestamp
        """
        return self.timestamp > timestamp
    
    def is_before_timestamp(self, timestamp: int) -> Bool[Array, ""]:
        """Check if this record was taken before the specified timestamp.
        
        Args:
            timestamp: Timestamp to compare against
            
        Returns:
            JAX boolean scalar indicating if record is before timestamp
        """
        return self.timestamp < timestamp
    
    @staticmethod
    def create_invalid_record(
        selection_format: str = "mask",
        max_grid_height: int = 30,
        max_grid_width: int = 30
    ) -> "ActionRecord":
        """Create an invalid (padding) record for JAX compatibility.
        
        This static method creates a properly sized but invalid record
        that can be used for padding in fixed-size arrays.
        
        Args:
            selection_format: Selection format for sizing
            max_grid_height: Maximum grid height
            max_grid_width: Maximum grid width
            
        Returns:
            ActionRecord with valid=False and zero-filled data
        """
        selection_size = get_selection_data_size(selection_format, max_grid_height, max_grid_width)
        
        return ActionRecord(
            selection_data=jnp.zeros(selection_size, dtype=jnp.float32),
            operation_id=jnp.array(-1, dtype=jnp.int32),
            timestamp=jnp.array(-1, dtype=jnp.int32),
            pair_index=jnp.array(-1, dtype=jnp.int32),
            valid=jnp.array(False)
        )
    
    @staticmethod
    def create_from_action(
        action: dict,
        timestamp: int | Array,
        pair_index: int | Array,
        selection_format: str = "mask",
        max_grid_height: int = 30,
        max_grid_width: int = 30
    ) -> "ActionRecord":
        """Create an ActionRecord from an action dictionary.
        
        This static method provides a convenient way to create ActionRecord
        instances from action dictionaries with proper validation and formatting.
        
        Args:
            action: Action dictionary containing selection and operation data
            timestamp: Step count when action was taken (int or JAX array)
            pair_index: Index of the pair this action was taken on (int or JAX array)
            selection_format: Selection format ("point", "bbox", "mask")
            max_grid_height: Maximum grid height
            max_grid_width: Maximum grid width
            
        Returns:
            ActionRecord instance with properly formatted data
            
        Examples:
            ```python
            # Create from point action
            from jaxarc.envs.structured_actions import PointAction, MaskAction
            action = PointAction(operation=jnp.array(15, dtype=jnp.int32), row=jnp.array(5, dtype=jnp.int32), col=jnp.array(10, dtype=jnp.int32))
            record = ActionRecord.create_from_action(
                action, timestamp=100, pair_index=0, selection_format="point"
            )
            
            # Create from mask action
            action = MaskAction(operation=jnp.array(25, dtype=jnp.int32), selection=selection_mask)
            record = ActionRecord.create_from_action(
                action, timestamp=150, pair_index=1, selection_format="mask"
            )
            ```
        """
        # Extract operation ID from structured action or dictionary
        if hasattr(action, 'operation'):
            # Structured action
            operation_id = action.operation
        else:
            raise ValueError(f"Unsupported action type: {type(action)}. Only structured actions are supported.")
        
        # Format selection data based on format
        selection_data = ActionRecord._format_selection_data(
            action, selection_format, max_grid_height, max_grid_width
        )
        
        return ActionRecord(
            selection_data=selection_data,
            operation_id=operation_id,
            timestamp=jnp.array(timestamp, dtype=jnp.int32),
            pair_index=jnp.array(pair_index, dtype=jnp.int32),
            valid=jnp.array(True)
        )
    
    @staticmethod
    def _format_selection_data(
        action,  # Union[StructuredAction, dict]
        selection_format: str,
        max_grid_height: int,
        max_grid_width: int
    ) -> SelectionData:
        """Format selection data from structured action or action dictionary.
        
        Args:
            action: Structured action or action dictionary
            selection_format: Selection format
            max_grid_height: Maximum grid height
            max_grid_width: Maximum grid width
            
        Returns:
            Properly formatted selection data array
        """
        selection_size = get_selection_data_size(selection_format, max_grid_height, max_grid_width)
        
        # Handle structured actions
        if hasattr(action, 'operation'):
            from .structured_actions import PointAction, BboxAction, MaskAction
            
            if selection_format == "point":
                if isinstance(action, PointAction):
                    return jnp.array([action.row, action.col], dtype=jnp.float32)
                else:
                    # For non-point actions in point format, use zeros
                    return jnp.zeros(2, dtype=jnp.float32)
                    
            elif selection_format == "bbox":
                if isinstance(action, BboxAction):
                    return jnp.array([action.r1, action.c1, action.r2, action.c2], dtype=jnp.float32)
                else:
                    # For non-bbox actions in bbox format, use zeros
                    return jnp.zeros(4, dtype=jnp.float32)
                    
            elif selection_format == "mask":
                if isinstance(action, MaskAction):
                    flattened = action.selection.flatten().astype(jnp.float32)
                    if flattened.shape[0] == selection_size:
                        return flattened
                    elif flattened.shape[0] < selection_size:
                        padded = jnp.zeros(selection_size, dtype=jnp.float32)
                        return padded.at[:flattened.shape[0]].set(flattened)
                    else:
                        return flattened[:selection_size]
                else:
                    # For non-mask actions, convert to mask using to_selection_mask
                    grid_shape = (max_grid_height, max_grid_width)
                    mask = action.to_selection_mask(grid_shape)
                    flattened = mask.flatten().astype(jnp.float32)
                    if flattened.shape[0] == selection_size:
                        return flattened
                    elif flattened.shape[0] < selection_size:
                        padded = jnp.zeros(selection_size, dtype=jnp.float32)
                        return padded.at[:flattened.shape[0]].set(flattened)
                    else:
                        return flattened[:selection_size]
            else:
                raise ValueError(f"Unknown selection format: {selection_format}")
        
        else:
            raise ValueError(f"Unsupported action type: {type(action)}. Only structured actions are supported.")


# =============================================================================
# ActionRecord Validation and Utility Functions
# =============================================================================

def validate_action_record_array(
    records: ActionSequence,
    selection_format: str = "mask",
    max_grid_height: int = 30,
    max_grid_width: int = 30
) -> Bool[Array, "sequence_length"]:
    """Validate an array of action records.
    
    This function validates each record in an array of action records,
    returning a boolean mask indicating which records are valid.
    
    Args:
        records: Array of action records to validate
        selection_format: Expected selection format
        max_grid_height: Maximum grid height for validation
        max_grid_width: Maximum grid width for validation
        
    Returns:
        Boolean array indicating which records are valid
        
    Examples:
        ```python
        # Validate a sequence of records
        valid_mask = validate_action_record_array(history_sequence, "point", 5, 5)
        
        # Filter to only valid records
        valid_records = history_sequence[valid_mask]
        ```
    """
    if records.shape[0] == 0:
        return jnp.array([], dtype=jnp.bool_)
    
    # Convert array records back to ActionRecord instances for validation
    selection_size = get_selection_data_size(selection_format, max_grid_height, max_grid_width)
    
    def validate_single_record(record_array):
        # Extract fields from array representation
        selection_data = record_array[:selection_size]
        operation_id = jnp.array(record_array[selection_size], dtype=jnp.int32)
        timestamp = jnp.array(record_array[selection_size + 1], dtype=jnp.int32)
        pair_index = jnp.array(record_array[selection_size + 2], dtype=jnp.int32)
        valid = jnp.array(record_array[selection_size + 3], dtype=jnp.bool_)
        
        # Create temporary ActionRecord for validation
        temp_record = ActionRecord(
            selection_data=selection_data,
            operation_id=operation_id,
            timestamp=timestamp,
            pair_index=pair_index,
            valid=valid
        )
        
        return temp_record.validate_integrity(selection_format, max_grid_height, max_grid_width)
    
    # Vectorize validation across all records
    return jax.vmap(validate_single_record)(records)


def filter_valid_records(
    records: ActionSequence,
    selection_format: str = "mask",
    max_grid_height: int = 30,
    max_grid_width: int = 30
) -> ActionSequence:
    """Filter an array of records to only include valid ones.
    
    Args:
        records: Array of action records
        selection_format: Selection format for validation
        max_grid_height: Maximum grid height
        max_grid_width: Maximum grid width
        
    Returns:
        Array containing only valid records
    """
    valid_mask = validate_action_record_array(records, selection_format, max_grid_height, max_grid_width)
    return records[valid_mask]


def count_valid_records(
    records: ActionSequence,
    selection_format: str = "mask",
    max_grid_height: int = 30,
    max_grid_width: int = 30
) -> Int[Array, ""]:
    """Count the number of valid records in an array.
    
    Args:
        records: Array of action records
        selection_format: Selection format for validation
        max_grid_height: Maximum grid height
        max_grid_width: Maximum grid width
        
    Returns:
        JAX scalar array containing the count of valid records
    """
    valid_mask = validate_action_record_array(records, selection_format, max_grid_height, max_grid_width)
    return jnp.sum(valid_mask)


def get_record_statistics(
    records: ActionSequence,
    selection_format: str = "mask",
    max_grid_height: int = 30,
    max_grid_width: int = 30
) -> dict:
    """Get statistics about an array of action records.
    
    Note: This function converts JAX arrays to Python types for readability.
    
    Args:
        records: Array of action records
        selection_format: Selection format for validation
        max_grid_height: Maximum grid height
        max_grid_width: Maximum grid width
        
    Returns:
        Dictionary containing record statistics
    """
    if records.shape[0] == 0:
        return {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "validity_rate": 0.0,
        }
    
    valid_mask = validate_action_record_array(records, selection_format, max_grid_height, max_grid_width)
    total_records = records.shape[0]
    valid_records = int(jnp.sum(valid_mask))
    invalid_records = total_records - valid_records
    
    return {
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "validity_rate": valid_records / total_records if total_records > 0 else 0.0,
    }


def create_padded_record_array(
    records: list[ActionRecord],
    target_length: int,
    selection_format: str = "mask",
    max_grid_height: int = 30,
    max_grid_width: int = 30
) -> ActionSequence:
    """Create a padded array of action records for JAX compatibility.
    
    This function takes a list of ActionRecord instances and creates a
    fixed-size JAX array with proper padding for records that don't exist.
    
    Args:
        records: List of ActionRecord instances
        target_length: Desired length of the output array
        selection_format: Selection format for padding records
        max_grid_height: Maximum grid height
        max_grid_width: Maximum grid width
        
    Returns:
        Fixed-size JAX array containing the records with padding
        
    Examples:
        ```python
        # Create padded array from record list
        records = [record1, record2, record3]
        padded_array = create_padded_record_array(records, 1000, "point", 5, 5)
        ```
    """
    # Determine target record size based on selection format
    target_record_size = get_action_record_fields(selection_format, max_grid_height, max_grid_width)
    
    # Convert records to array format
    record_arrays = []
    for record in records:
        record_array = _record_to_array_helper(record, target_record_size)
        record_arrays.append(record_array)
    
    # Pad with invalid records if necessary
    while len(record_arrays) < target_length:
        invalid_record = ActionRecord.create_invalid_record(
            selection_format, max_grid_height, max_grid_width
        )
        invalid_array = _record_to_array_helper(invalid_record, target_record_size)
        record_arrays.append(invalid_array)
    
    # Truncate if too long
    record_arrays = record_arrays[:target_length]
    
    return jnp.array(record_arrays)


def _record_to_array_helper(record: ActionRecord, target_size: int = None) -> Array:
    """Helper function to convert ActionRecord to array format.
    
    Args:
        record: ActionRecord to convert
        target_size: Target size for the output array (for padding)
        
    Returns:
        Flattened array representation of the record, padded if necessary
    """
    base_array = jnp.concatenate([
        record.selection_data,
        jnp.array([record.operation_id], dtype=jnp.float32),
        jnp.array([record.timestamp], dtype=jnp.float32),
        jnp.array([record.pair_index], dtype=jnp.float32),
        jnp.array([record.valid], dtype=jnp.float32),
    ])
    
    # Pad to target size if specified
    if target_size is not None and base_array.shape[0] < target_size:
        padding_size = target_size - base_array.shape[0]
        padding = jnp.zeros(padding_size, dtype=jnp.float32)
        return jnp.concatenate([base_array, padding])
    
    return base_array


def array_to_action_record(
    record_array: Array,
    selection_format: str = "mask",
    max_grid_height: int = 30,
    max_grid_width: int = 30
) -> ActionRecord:
    """Convert array format back to ActionRecord.
    
    Args:
        record_array: Flattened array representation
        selection_format: Selection format for reconstruction
        max_grid_height: Maximum grid height
        max_grid_width: Maximum grid width
        
    Returns:
        Reconstructed ActionRecord instance
    """
    selection_size = get_selection_data_size(selection_format, max_grid_height, max_grid_width)
    
    selection_data = record_array[:selection_size]
    operation_id = jnp.array(record_array[selection_size], dtype=jnp.int32)
    timestamp = jnp.array(record_array[selection_size + 1], dtype=jnp.int32)
    pair_index = jnp.array(record_array[selection_size + 2], dtype=jnp.int32)
    valid = jnp.array(record_array[selection_size + 3], dtype=jnp.bool_)
    
    return ActionRecord(
        selection_data=selection_data,
        operation_id=operation_id,
        timestamp=timestamp,
        pair_index=pair_index,
        valid=valid
    )


class ActionHistoryTracker:
    """Tracks action sequences with JAX-compatible fixed-size storage.
    
    This class manages a circular buffer of action records, providing efficient
    storage and retrieval of action sequences while maintaining JAX compatibility
    through static shapes and pure functions.
    
    Key Features:
    - Circular buffer with automatic overflow handling
    - JAX-compatible pure functions for all operations
    - Configuration-aware storage optimization
    - Preserves original selection format (point/bbox) when beneficial
    - Efficient action sequence extraction
    - Dynamic sizing based on dataset and selection format
    
    The tracker uses a fixed-size array to store action records, with a circular
    buffer approach to handle overflow. The size of each record is optimized
    based on the selection format and dataset configuration:
    - Point selections: 2 + 4 = 6 fields (row, col + metadata)
    - Bbox selections: 4 + 4 = 8 fields (r1, c1, r2, c2 + metadata)  
    - Mask selections: grid_size^2 + 4 fields (flattened mask + metadata)
    
    Examples:
        ```python
        # Configuration-aware tracker
        tracker = ActionHistoryTracker()
        
        # Add action to history (preserves original format)
        from jaxarc.envs.structured_actions import PointAction
        action = PointAction(operation=jnp.array(15, dtype=jnp.int32), row=jnp.array(5, dtype=jnp.int32), col=jnp.array(10, dtype=jnp.int32))  # Point action
        new_state = tracker.add_action(state, action, config)
        
        # Get recent actions with original selection data
        recent = tracker.get_action_sequence(new_state, start_idx=-10)
        
        # Clear history
        clean_state = tracker.clear_history(new_state)
        ```
    """
    
    def add_action(
        self,
        state: ArcEnvState,
        action: dict,
        config: HistoryConfig,
        selection_format: str = "mask",
        max_grid_height: int = 30,
        max_grid_width: int = 30
    ) -> ArcEnvState:
        """Add action to history with configuration-aware optimal storage.
        
        This method adds a new action to the history buffer using an optimized storage
        format based on the selection format and dataset configuration. It preserves
        the most semantically meaningful information for agents.
        
        Args:
            state: Current environment state
            action: Action dictionary (format depends on selection_format)
            config: History configuration settings
            selection_format: Selection format ("point", "bbox", "mask")
            max_grid_height: Maximum grid height for the dataset
            max_grid_width: Maximum grid width for the dataset
            
        Returns:
            New state with updated action history
            
        Examples:
            ```python
            # Add point-based action (stores [row, col] directly)
            from jaxarc.envs.structured_actions import PointAction, BboxAction, MaskAction
            action = PointAction(operation=jnp.array(15, dtype=jnp.int32), row=jnp.array(5, dtype=jnp.int32), col=jnp.array(10, dtype=jnp.int32))
            new_state = tracker.add_action(state, action, config, "point", 5, 5)
            
            # Add bbox-based action (stores [r1, c1, r2, c2] directly)
            action = BboxAction(operation=jnp.array(20, dtype=jnp.int32), r1=jnp.array(1, dtype=jnp.int32), c1=jnp.array(2, dtype=jnp.int32), r2=jnp.array(8, dtype=jnp.int32), c2=jnp.array(9, dtype=jnp.int32))
            new_state = tracker.add_action(state, action, config, "bbox", 30, 30)
            
            # Add mask-based action (stores flattened mask)
            action = MaskAction(operation=jnp.array(28, dtype=jnp.int32), selection=selection_mask)
            new_state = tracker.add_action(state, action, config, "mask", 30, 30)
            ```
        """
        if not config.enabled:
            return state
            

        
        # Create action record using the enhanced factory method
        record = ActionRecord.create_from_action(
            action,
            timestamp=state.step_count,  # Keep as JAX array
            pair_index=state.current_example_idx,  # Keep as JAX array
            selection_format=selection_format,
            max_grid_height=max_grid_height,
            max_grid_width=max_grid_width
        )
        
        # Convert record to array format for storage
        target_size = state.action_history.shape[1]  # Get the buffer's record size
        record_array = _record_to_array_helper(record, target_size)
        
        # For circular buffer, we need to track both the current length and total actions added
        # The current_length is capped at buffer size, but we need total count for indexing
        max_length = state.action_history.shape[0]
        
        # Use step_count as a proxy for total actions added (assuming one action per step)
        # This gives us the true insertion index for circular buffer
        total_actions = state.step_count
        insert_idx = total_actions % max_length
        
        # Update history array
        new_history = state.action_history.at[insert_idx].set(record_array)
        
        # Update history length (capped at max_length for circular buffer)
        new_length = jnp.minimum(state.action_history_length + 1, max_length)
        
        # Return updated state using PyTree utilities
        from jaxarc.utils.pytree_utils import update_multiple_fields
        
        return update_multiple_fields(
            state,
            action_history=new_history,
            action_history_length=new_length
        )
    
    def get_action_sequence(
        self,
        state: ArcEnvState,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ) -> ActionSequence:
        """Extract action sequence from history with proper indexing.
        
        This method retrieves a sequence of actions from the history buffer.
        For JAX compatibility, this simplified version returns the valid portion
        of the history buffer based on the history length.
        
        Args:
            state: Current environment state
            start_idx: Starting index (0-based, negative for relative to end)
            end_idx: Ending index (exclusive, None for end of history)
            
        Returns:
            JAX array containing the requested action sequence
            
        Examples:
            ```python
            # Get all actions
            all_actions = tracker.get_action_sequence(state)
            
            # Get last 10 actions
            recent = tracker.get_action_sequence(state, start_idx=-10)
            
            # Get actions 5-15
            middle = tracker.get_action_sequence(state, start_idx=5, end_idx=15)
            ```
        """
        # For JAX compatibility, we return the valid portion of the history
        # The caller can handle more complex slicing if needed
        history_length = state.action_history_length
        
        # Return the valid portion of the history (up to history_length)
        # We use a mask to zero out invalid entries
        valid_mask = jnp.arange(state.action_history.shape[0]) < history_length
        valid_mask = valid_mask[:, None]  # Broadcast to match record fields
        
        # Apply mask to zero out invalid entries
        masked_history = state.action_history * valid_mask
        
        # For now, return the full masked history
        # More sophisticated slicing can be added later if needed
        return masked_history
    
    def clear_history(self, state: ArcEnvState) -> ArcEnvState:
        """Clear action history for new episode.
        
        This method resets the action history buffer, preparing it for a new
        episode. It maintains the buffer structure but resets the length counter
        and marks all records as invalid.
        
        Args:
            state: Current environment state
            
        Returns:
            New state with cleared action history
            
        Examples:
            ```python
            # Clear history at episode start
            clean_state = tracker.clear_history(state)
            
            # Verify history is empty
            assert clean_state.action_history_length == 0
            ```
        """
        # Reset history length to 0
        new_length = jnp.array(0, dtype=jnp.int32)
        
        # Optionally clear the history array (not strictly necessary due to length tracking)
        # But helps with debugging and ensures clean state
        new_history = jnp.zeros_like(state.action_history)
        
        from jaxarc.utils.pytree_utils import update_multiple_fields
        
        return update_multiple_fields(
            state,
            action_history=new_history,
            action_history_length=new_length
        )
    

    
    def _array_to_record(
        self, 
        array_data: Array,
        selection_format: str = "mask",
        max_grid_height: int = 30,
        max_grid_width: int = 30
    ) -> ActionRecord:
        """Convert array format back to ActionRecord structure.
        
        This method reconstructs an ActionRecord from its flattened array
        representation stored in the history buffer, using the configuration
        to determine the correct field boundaries.
        
        Args:
            array_data: Flattened array representation
            selection_format: Selection format used for this record
            max_grid_height: Maximum grid height for the dataset
            max_grid_width: Maximum grid width for the dataset
            
        Returns:
            Reconstructed ActionRecord
        """
        return array_to_action_record(array_data, selection_format, max_grid_height, max_grid_width)
    
    def get_action_count(self, state: ArcEnvState) -> HistoryLength:
        """Get the current number of actions in history.
        
        Args:
            state: Current environment state
            
        Returns:
            Number of actions currently stored in history
        """
        return state.action_history_length
    
    def is_history_full(self, state: ArcEnvState) -> Bool[Array, ""]:
        """Check if the history buffer is full.
        
        Args:
            state: Current environment state
            
        Returns:
            True if history buffer is at maximum capacity
        """
        return state.action_history_length >= state.action_history.shape[0]
    
    def get_history_capacity(self, state: ArcEnvState) -> int:
        """Get the maximum capacity of the history buffer.
        
        Args:
            state: Current environment state
            
        Returns:
            Maximum number of actions that can be stored
        """
        return state.action_history.shape[0]
    
    def get_recent_actions(
        self,
        state: ArcEnvState,
        count: int = 10
    ) -> ActionSequence:
        """Get the most recent N actions from history.
        
        Convenience method for getting recent actions without manual indexing.
        
        Args:
            state: Current environment state
            count: Number of recent actions to retrieve
            
        Returns:
            Array containing the most recent actions
        """
        return self.get_action_sequence(state, start_idx=-count)
    
    def get_actions_for_pair(
        self,
        state: ArcEnvState,
        pair_index: int
    ) -> ActionSequence:
        """Get all actions taken on a specific demonstration/test pair.
        
        This method filters the action history to return only actions
        that were taken on the specified pair index.
        
        Args:
            state: Current environment state
            pair_index: Index of the pair to filter by
            
        Returns:
            Array containing actions for the specified pair
        """
        # Get full history
        full_history = self.get_action_sequence(state)
        
        if full_history.shape[0] == 0:
            return full_history
        
        # Filter by pair index
        # Pair index is stored at position (selection_size + 2)
        # We need to determine selection size from the history buffer dimensions
        record_fields = state.action_history.shape[1]
        selection_size = record_fields - 4  # Subtract metadata fields
        pair_indices = full_history[:, selection_size + 2]
        mask = pair_indices == pair_index
        
        # Return filtered actions
        return full_history[mask]
    
    def get_history_summary(self, state: ArcEnvState) -> dict:
        """Get a summary of the current action history.
        
        This method provides useful statistics and information about
        the current state of the action history buffer.
        
        Args:
            state: Current environment state
            
        Returns:
            Dictionary containing history summary information
        """
        history_length = int(state.action_history_length)
        capacity = self.get_history_capacity(state)
        
        summary = {
            "length": history_length,
            "capacity": capacity,
            "is_full": bool(self.is_history_full(state)),
            "utilization": history_length / capacity if capacity > 0 else 0.0,
        }
        
        if history_length > 0:
            # Get operation statistics
            full_history = self.get_action_sequence(state)
            # Operation ID is at position (selection_size + 0)
            record_fields = state.action_history.shape[1]
            selection_size = record_fields - 4  # Subtract metadata fields
            operations = full_history[:, selection_size]  # Operation IDs
            
            summary.update({
                "unique_operations": int(len(jnp.unique(operations))),
                "most_recent_operation": int(operations[-1]) if len(operations) > 0 else -1,
            })
        
        return summary


# Utility functions for easier integration

def create_action_history_tracker_for_config(config) -> ActionHistoryTracker:
    """Create an ActionHistoryTracker configured for the given environment config.
    
    This is a convenience function that extracts the necessary configuration
    parameters and returns a properly configured ActionHistoryTracker.
    
    Args:
        config: JaxArcConfig or similar configuration object
        
    Returns:
        ActionHistoryTracker instance
        
    Examples:
        ```python
        from jaxarc.envs.config_factory import create_development_config
        from jaxarc.envs.action_history import create_action_history_tracker_for_config
        
        config = create_development_config(selection_format="point", max_grid_height=5)
        tracker = create_action_history_tracker_for_config(config)
        ```
    """
    return ActionHistoryTracker()


def add_action_to_state(
    state: ArcEnvState,
    action: dict,
    config,
    history_config: Optional[HistoryConfig] = None
) -> ArcEnvState:
    """Convenience function to add action to state with proper configuration.
    
    This function automatically extracts the necessary configuration parameters
    and adds the action to the state's history using the optimal storage format.
    
    Args:
        state: Current environment state
        action: Action dictionary (format depends on config.action.selection_format)
        config: JaxArcConfig or similar configuration object
        history_config: Optional history configuration (uses default if None)
        
    Returns:
        New state with updated action history
        
    Examples:
        ```python
        # Point action with MiniARC
        from jaxarc.envs.structured_actions import PointAction, BboxAction
        action = PointAction(operation=jnp.array(15, dtype=jnp.int32), row=jnp.array(2, dtype=jnp.int32), col=jnp.array(3, dtype=jnp.int32))
        new_state = add_action_to_state(state, action, config)
        
        # Bbox action with full ARC
        action = BboxAction(operation=jnp.array(20, dtype=jnp.int32), r1=jnp.array(1, dtype=jnp.int32), c1=jnp.array(2, dtype=jnp.int32), r2=jnp.array(8, dtype=jnp.int32), c2=jnp.array(9, dtype=jnp.int32))
        new_state = add_action_to_state(state, action, config)
        ```
    """
    if history_config is None:
        history_config = HistoryConfig()
    
    tracker = ActionHistoryTracker()
    return tracker.add_action(
        state,
        action,
        history_config,
        config.action.selection_format,
        config.dataset.max_grid_height,
        config.dataset.max_grid_width
    )