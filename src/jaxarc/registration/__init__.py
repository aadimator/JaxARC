"""
Registration system for JaxARC environments.

This package provides a lean registry that maps simple dataset keys to
environment specs. Dataset parsing and task loading are no longer part of this
module. Environments are expected to be constructed with buffer-based EnvParams
(JAX-native, JIT-friendly) and not depend on parsers at runtime.

Core ideas:
- A global registry maps dataset keys (e.g., "Mini", "Concept", "AGI1", "AGI2") to EnvSpec definitions.
- No parser entry points or subset inference live here anymore.
- `make(id, **kwargs)` only parses the dataset key and returns the environment and parameters
  built from provided kwargs (e.g., a prebuilt buffer in EnvParams or an explicit params).
- Named subsets can be registered (e.g., `register_subset("Mini", "easy", [...])`) and then
  selected via `make("Mini-easy")` to load exactly those tasks. This makes it easy to publish
  curated benchmarks and implement curriculum learning.

Typical usage:
    from jaxarc.registration import make
    # Build EnvParams with a pre-stacked task buffer outside this module.
    env, params = make("Mini", params=my_params)

Notes:
- This module keeps a single way of doing things: buffer-based, JIT-friendly EnvParams.
- Dataset downloading/parsing and subset handling should be done outside this module.
"""

from __future__ import annotations

from typing import Any

from .registry import EnvRegistry, EnvSpec
from .subset_loader import (
    load_all_subsets_for_dataset,
    load_subset,
    load_subset_if_needed,
)

# -----------------------------------------------------------------------------
# Module-level singleton API
# -----------------------------------------------------------------------------

_registry = EnvRegistry()

# Default bootstrap: register common dataset IDs with minimal specs
_registry.register(id="Mini", max_episode_steps=100)
_registry.register(id="Concept", max_episode_steps=100)
_registry.register(id="AGI1", max_episode_steps=100)
_registry.register(id="AGI2", max_episode_steps=100)


def register(
    id: str,
    entry_point: str | None = None,
    env_entry: str = "jaxarc.envs:Environment",
    max_episode_steps: int = 100,
    **kwargs: Any,
) -> None:
    """Register an environment spec in the global registry."""
    _registry.register(
        id=id,
        env_entry=env_entry,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )


def make(id: str, **kwargs: Any) -> tuple[Any, Any]:
    """Create an environment instance and EnvParams using a registered spec.

    See EnvRegistry.make for details on supported kwargs.
    """
    return _registry.make(id, **kwargs)


def register_subset(
    dataset_key: str, name: str, task_ids: list[str] | tuple[str, ...]
) -> None:
    """Register a named subset for a dataset key, enabling IDs like 'Mini-easy'."""
    _registry.register_subset(dataset_key, name, task_ids)


def get_subset_task_ids(
    dataset_key: str,
    selector: str = "all",
    config: Any | None = None,
    auto_download: bool = False,
) -> list[str]:
    """Get task IDs for a specific subset without creating an environment.

    This allows users to query what tasks will be loaded before calling make().

    Args:
        dataset_key: Dataset name (Mini, Concept, AGI1, AGI2)
        selector: Subset selector ('all', 'train', 'easy', task_id, etc.)
        config: Optional config
        auto_download: Download dataset if missing

    Returns:
        List of task IDs that will be loaded

    Examples:
        >>> get_subset_task_ids("Mini", "all")
        ['Most_Common_color_l6ab0lf3xztbyxsu3p', ...]

        >>> get_subset_task_ids("Mini", "easy")
        ['task1', 'task2', 'task3']

        >>> get_subset_task_ids("Mini", "Most_Common_color_l6ab0lf3xztbyxsu3p")
        ['Most_Common_color_l6ab0lf3xztbyxsu3p']
    """
    return _registry.get_subset_task_ids(
        dataset_key, selector=selector, config=config, auto_download=auto_download
    )


def available_task_ids(
    dataset_key: str, config: Any | None = None, auto_download: bool = False
) -> list[str]:
    """List all available task IDs (equivalent to get_subset_task_ids with selector='all')."""
    return _registry.get_subset_task_ids(
        dataset_key, selector="all", config=config, auto_download=auto_download
    )


def available_named_subsets(
    dataset_key: str, include_builtin: bool = True
) -> tuple[str, ...]:
    """List available subset names for a dataset (includes built-in selectors by default).

    Args:
        dataset_key: Dataset name (Mini, Concept, AGI1, AGI2)
        include_builtin: Include built-in selectors like 'all', 'train', 'eval' (default: True)

    Returns:
        Tuple of subset names

    Examples:
        >>> available_named_subsets("Mini")
        ('all', 'easy', 'eval', 'train')

        >>> available_named_subsets("Mini", include_builtin=False)
        ('easy',)  # Only custom subsets
    """
    return _registry.available_named_subsets(
        dataset_key, include_builtin=include_builtin
    )


def subset_task_ids(dataset_key: str, name: str) -> tuple[str, ...]:
    """Return the task IDs registered under a named subset.

    This only works for explicitly registered subsets (via register_subset).
    For more flexible queries, use get_subset_task_ids() instead.
    """
    return _registry.subset_task_ids(dataset_key, name)


__all__ = [
    "EnvRegistry",
    "EnvSpec",
    "_registry",
    "available_named_subsets",
    "available_task_ids",
    "get_subset_task_ids",
    "load_all_subsets_for_dataset",
    "load_subset",
    "load_subset_if_needed",
    "make",
    "register",
    "register_subset",
    "subset_task_ids",
]
