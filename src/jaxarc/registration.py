"""
Registration system for JaxARC environments.

This module now provides a lean registry that maps simple dataset keys to
environment specs. Dataset parsing and task loading are no longer part of this
module. Environments are expected to be constructed with buffer-based EnvParams
(JAX-native, JIT-friendly) and not depend on parsers at runtime.

Core ideas:
- A global registry maps dataset keys (e.g., "Mini", "Concept", "AGI1", "AGI2") to EnvSpec definitions.
- No parser entry points or subset inference live here anymore.
- `make(id, **kwargs)` only parses the dataset key and returns the environment and parameters
  built from provided kwargs (e.g., a prebuilt buffer in EnvParams or an explicit params).

Typical usage:
    from jaxarc.registration import make
    # Build EnvParams with a pre-stacked task buffer outside this module.
    env, params = make("Mini", params=my_params)

Notes:
- This module keeps a single way of doing things: buffer-based, JIT-friendly EnvParams.
- Dataset downloading/parsing and subset handling should be done outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import equinox as eqx
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from loguru import logger


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class EnvSpec:
    """Environment specification for registration."""
    id: str
    env_entry: str = "jaxarc.envs:Environment"
    max_episode_steps: int = 100
    kwargs: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Registry implementation
# -----------------------------------------------------------------------------

class EnvRegistry:
    """Global environment registry with gym-like semantics."""

    def __init__(self) -> None:
        self._specs: Dict[str, EnvSpec] = {}

    def register(
        self,
        id: str,
        env_entry: str = "jaxarc.envs:Environment",
        max_episode_steps: int = 100,
        **kwargs: Any,
    ) -> None:
        """Register a new environment specification.

        Args:
            id: Unique environment ID (e.g., "JaxARC-Mini-v0")
            entry_point: Dotted path or colon path to class/factory (e.g., "jaxarc.envs:Environment")
            max_episode_steps: Default max steps for this environment family
            **kwargs: Additional metadata stored with the spec
        """
        self._specs[id] = EnvSpec(
            id=id,
            env_entry=env_entry,
            max_episode_steps=int(max_episode_steps),
            kwargs=dict(kwargs),
        )

    def make(self, id: str, **kwargs: Any) -> Tuple[Any, Any]:
        """Create an environment instance and parameters for a registered spec.

        Expected kwargs:
            - params: EnvParams (preferred; buffer-based, JIT-friendly)
            - env_entry: str (optional) override of environment entry point

        Returns:
            (env, params) tuple:
                env: Environment instance
                params: EnvParams provided directly
        """
        dataset_key, modifiers = self._parse_id(id)

        if dataset_key not in self._specs:
            raise ValueError(f"Environment '{dataset_key}' is not registered")

        spec = self._specs[dataset_key]

        # Instantiate environment (spec.env_entry or override)
        env_entry = kwargs.get("env_entry", spec.env_entry)
        env_obj = self._import_from_entry_point(env_entry)
        env = env_obj() if self._is_class(env_obj) else env_obj

        # If params explicitly provided, use them
        if "params" in kwargs and kwargs["params"] is not None:
            return env, kwargs["params"]

        raise ValueError(
            "EnvParams must be provided. This registry no longer supports parser-based construction."
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _parse_id(self, id: str) -> tuple[str, dict[str, str]]:
        """Parse environment ID and extract modifiers.

        Conventions:
            - Accepts either:
              - DatasetID                 (no selector)
              - DatasetID-{Selector}      (with selector)
            - When selector is present, all remaining tokens after DatasetID
              are joined to form the single Selector string.
        """
        tokens = id.split("-", 1)
        dataset_key = tokens[0]
        selector = tokens[1] if len(tokens) > 1 else ""
        modifiers: dict[str, str] = {}
        if selector:
            modifiers["selector"] = selector
        return dataset_key, modifiers

    @staticmethod
    def _is_class(obj: Any) -> bool:
        try:
            import inspect
            return inspect.isclass(obj)
        except Exception:
            return False

    @staticmethod
    def _import_from_entry_point(entry_point: str) -> Any:
        """Import an object from an entry point string.

        Supports:
            - "package.module:object"
            - "package.module.Object"

        Raises:
            ValueError: If the entry point format is invalid or import fails
        """
        module_name: Optional[str] = None
        attr_name: Optional[str] = None

        if ":" in entry_point:
            module_name, attr_name = entry_point.split(":", 1)
        else:
            # Split by last dot to separate module and attribute
            parts = entry_point.split(".")
            if len(parts) < 2:
                raise ValueError(f"Invalid entry_point '{entry_point}'")
            module_name = ".".join(parts[:-1])
            attr_name = parts[-1]

        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, attr_name)
            return obj
        except Exception as e:
            raise ValueError(f"Failed to import '{entry_point}': {e}") from e

    @staticmethod
    def _prepare_config(config: Optional[Any], max_episode_steps: int, dataset_key: str) -> Any:
        """Prepare a JaxArcConfig, applying overrides when possible.

        - If no config provided, instantiate a default JaxArcConfig.
        - Ensure environment.max_episode_steps matches the spec if possible.
        - Normalize dataset configuration based on spec.dataset_key.
        """
        # Import locally to avoid hard dependency at module import time
        try:
            from jaxarc.configs.main_config import JaxArcConfig
            from jaxarc.configs.environment_config import EnvironmentConfig
            from jaxarc.configs.dataset_config import DatasetConfig
        except Exception as e:
            raise ValueError(
                "Could not import configuration types. Ensure configurations "
                "are available or provide a ready 'config' object."
            ) from e

        cfg = config if config is not None else JaxArcConfig()

        # Enforce max_episode_steps
        try:
            setattr(cfg, "environment", EnvironmentConfig(max_episode_steps=max_episode_steps))
        except Exception:
            pass

        # Best-effort dataset normalization
        try:
            ds: DatasetConfig = getattr(cfg, "dataset")
            # Update dataset_name to normalized value (e.g., "MiniARC", "ConceptARC", "ARC-AGI-1", "ARC-AGI-2")
            _, _, normalized_name = EnvRegistry._resolve_dataset_meta(dataset_key)
            ds.dataset_name = normalized_name
            setattr(cfg, "dataset", ds)
        except Exception:
            pass

        return cfg

    @staticmethod
    def _resolve_episode_mode(episode_mode: Optional[int], selector: Optional[str]) -> int:
        """Resolve episode mode using explicit value; selector no longer used."""
        if episode_mode is not None:
            return int(episode_mode)
        return 0

    @staticmethod
    def _resolve_dataset_meta(dataset_key: str) -> tuple[str, str, str]:
        """Deprecated: dataset meta resolution removed."""
        return ("", "", dataset_key)

    def _maybe_adjust_task_split(self, config: Any, dataset_key: str, selector: Optional[str]) -> None:
        """Adjust config.dataset.task_split based on selector for AGI datasets."""
        try:
            ds = getattr(config, "dataset")
            sel = (selector or "").lower()
            if dataset_key.lower() in ("agi1", "arc-agi-1", "agi-1", "agi_1", "agi2", "arc-agi-2", "agi-2", "agi_2"):
                if sel in ("train", "training"):
                    setattr(ds, "task_split", "train")
                elif sel in ("eval", "evaluation", "test", "corpus"):
                    setattr(ds, "task_split", "evaluation")
                setattr(config, "dataset", ds)
        except Exception:
            # Best-effort only
            pass

    @staticmethod
    def _infer_subset_ids(parser: Any, dataset_key: str, selector: str) -> tuple[str, ...]:
        """Infer a tuple of task IDs for standard named subsets.

        Supports:
            - Mini: 'train'/'eval'/'all' => all task IDs
            - Concept: concept group names; 'train'/'eval'/'all' => all task IDs
            - AGI1/AGI2: 'train'/'eval' => current split's available task IDs
        """
        try:
            key = dataset_key.lower()
            sel = selector.lower()

            # ConceptARC named subsets
            if key in ("concept", "conceptarc", "concept-arc"):
                if sel in ("train", "training", "eval", "evaluation", "test", "corpus", "all"):
                    return tuple(parser.get_available_task_ids())
                if hasattr(parser, "get_concept_groups") and hasattr(parser, "get_tasks_in_concept"):
                    concepts = set(parser.get_concept_groups())
                    if selector in concepts:
                        return tuple(parser.get_tasks_in_concept(selector))
                return tuple()

            # MiniARC subsets: treat train/eval/all as "all tasks"
            if key in ("mini", "miniarc", "mini-arc"):
                if sel in ("train", "training", "eval", "evaluation", "test", "corpus", "all"):
                    return tuple(parser.get_available_task_ids())
                return tuple()

            # AGI subsets: use current parser split's available IDs
            if key in ("agi1", "arc-agi-1", "agi-1", "agi_1", "agi2", "arc-agi-2", "agi-2", "agi_2"):
                if sel in ("train", "training", "eval", "evaluation", "test", "corpus"):
                    return tuple(parser.get_available_task_ids())
                return tuple()

            # Fallback: if selector is a concrete task id, ensure it exists
            if hasattr(parser, "get_available_task_ids"):
                ids = parser.get_available_task_ids()
                if selector in ids:
                    return (selector,)
            return tuple()
        except Exception:
            return tuple()

    @staticmethod
    def _ensure_dataset_available(config: Any, dataset_key: str, auto_download: bool) -> Any:
        """Deprecated: dataset availability/download no longer handled here.

        Functional: returns (potentially) updated config without in-place mutation.
        """
        try:
            # Access DatasetConfig
            ds = getattr(config, "dataset")
            _ = getattr(ds, "dataset_path", "")
        except Exception as e:
            logger.warning(f"Unable to access dataset configuration: {e}")
        return config


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


def make(id: str, **kwargs: Any) -> Tuple[Any, Any]:
    """Create an environment instance and EnvParams using a registered spec.

    See EnvRegistry.make for details on supported kwargs.
    """
    return _registry.make(id, **kwargs)


def get_subset_ids(
    dataset: str,
    selector: str,
    config: Optional[Any] = None,
    auto_download: bool = False,
) -> tuple[str, ...]:
    """Resolve a named subset to concrete task IDs for a dataset/selector.

    This utility instantiates the appropriate parser (with inferred entry from dataset),
    ensures the dataset is available (optionally auto-downloading), and returns a tuple
    of task IDs matching the selector semantics.

    Supported selectors by dataset:
    - Mini: 'train'/'eval'/'all' => all available task IDs; or a concrete task id
    - Concept: a concept group name (e.g., 'AboveBelow') => group task IDs; 'train'/'eval'/'all' => all IDs
    - AGI1/AGI2: 'train'/'eval' => current split's task IDs; or a concrete task id

    Args:
        dataset: Dataset key ('Mini', 'Concept', 'AGI1', 'AGI2', or aliases)
        selector: Named subset or concrete task id
        config: Optional JaxArcConfig to control dataset location/settings
        auto_download: If True, attempts dataset download when missing

    Returns:
        Tuple of task IDs (may be empty if selector cannot be resolved)
    """
    # Resolve parser entry and normalize dataset name
    parser_entry, _downloader_method, _normalized_name = EnvRegistry._resolve_dataset_meta(dataset)

    # Prepare config (best-effort) and align split for AGI datasets based on selector
    prepared_cfg = EnvRegistry._prepare_config(config, max_episode_steps=100, dataset_key=dataset)
    try:
        ds = getattr(prepared_cfg, "dataset")
        sel = selector.lower()
        if dataset.lower() in ("agi1", "arc-agi-1", "agi-1", "agi_1", "agi2", "arc-agi-2", "agi-2", "agi_2"):
            if sel in ("train", "training"):
                setattr(ds, "task_split", "train")
            elif sel in ("eval", "evaluation", "test", "corpus"):
                setattr(ds, "task_split", "evaluation")
            setattr(prepared_cfg, "dataset", ds)
    except Exception:
        # best-effort only
        pass

    raise NotImplementedError("get_subset_ids has been removed; use buffer-based construction outside registration.")
    # When auto_download=True, require a dataset_path and download into that location
    if auto_download:
        try:
            ds = getattr(prepared_cfg, "dataset")
            dspath = getattr(ds, "dataset_path", "")
        except Exception:
            dspath = ""
        if not dspath:
            raise ValueError(
                "When auto_download=True, please set config.dataset.dataset_path to the desired download directory."
            )
    prepared_cfg = EnvRegistry._ensure_dataset_available(prepared_cfg, dataset, auto_download)

    # Instantiate parser
    parser_obj = EnvRegistry._import_from_entry_point(parser_entry)
    is_class = EnvRegistry._is_class(parser_obj)
    parser = parser_obj(prepared_cfg.dataset) if is_class else parser_obj

    key = dataset.lower()
    sel_lower = selector.lower()

    # Compute subset ids per dataset family
    try:
        if key in ("concept", "conceptarc", "concept-arc"):
            if sel_lower in ("train", "training", "eval", "evaluation", "test", "corpus", "all"):
                return tuple(parser.get_available_task_ids())
            if hasattr(parser, "get_concept_groups") and hasattr(parser, "get_tasks_in_concept"):
                concepts = set(parser.get_concept_groups())
                if selector in concepts:
                    return tuple(parser.get_tasks_in_concept(selector))
            ids = parser.get_available_task_ids()
            return (selector,) if selector in ids else tuple()

        if key in ("mini", "miniarc", "mini-arc"):
            if sel_lower in ("train", "training", "eval", "evaluation", "test", "corpus", "all"):
                return tuple(parser.get_available_task_ids())
            ids = parser.get_available_task_ids()
            return (selector,) if selector in ids else tuple()

        if key in ("agi1", "arc-agi-1", "agi-1", "agi_1", "agi2", "arc-agi-2", "agi-2", "agi_2"):
            if sel_lower in ("train", "training", "eval", "evaluation", "test", "corpus"):
                return tuple(parser.get_available_task_ids())
            ids = parser.get_available_task_ids()
            return (selector,) if selector in ids else tuple()

        # Generic fallback for custom datasets with available IDs
        if hasattr(parser, "get_available_task_ids"):
            ids = parser.get_available_task_ids()
            return (selector,) if selector in ids else tuple()

        return tuple()
    except Exception:
        return tuple()
