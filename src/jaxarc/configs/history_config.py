from __future__ import annotations

import warnings

import equinox as eqx
from omegaconf import DictConfig


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

    Notes:
        This class intentionally contains only data fields and minimal
        validation helpers for JAX compatibility. Any analysis or advisory
        utilities (e.g., memory estimation/comparison) live outside the
        library codebase in scripts/analysis.
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
            raise ValueError(msg)

        if self.max_history_length > 10000:
            # Warn about very large history lengths
            warnings.warn(
                f"Large history length ({self.max_history_length}) may impact memory usage",
                UserWarning,
                stacklevel=2,
            )
