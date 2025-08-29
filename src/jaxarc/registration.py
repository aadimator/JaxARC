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

from dataclasses import dataclass, field
import importlib
import equinox as eqx
from pathlib import Path
from pyprojroot import here
from typing import Any, Callable, Dict, Optional, Tuple
from loguru import logger
from jaxarc.utils import DatasetDownloader, DatasetDownloadError
from jaxarc.utils.buffer import stack_task_list


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
        # Named subset registry: maps normalized dataset key -> subset name -> tuple of task IDs
        self._subsets: Dict[str, Dict[str, tuple[str, ...]]] = {}

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

    def register_subset(self, dataset_key: str, name: str, task_ids: list[str] | tuple[str, ...]) -> None:
        """Register a named subset (e.g., 'Mini-easy') that maps to specific task IDs.

        Args:
            dataset_key: Base dataset key (e.g., 'Mini', 'Concept', 'AGI1', 'AGI2' or synonyms)
            name: Subset name (e.g., 'easy', 'hard', 'my-benchmark')
            task_ids: Sequence of task IDs to include in this subset
        """
        key = self._normalize_dataset_key(dataset_key)
        sel = name.strip().lower()
        if not sel:
            raise ValueError("Subset name must be non-empty")
        ids_tuple: tuple[str, ...] = tuple(task_ids) if not isinstance(task_ids, tuple) else task_ids
        if key not in self._subsets:
            self._subsets[key] = {}
        self._subsets[key][sel] = ids_tuple

    def available_named_subsets(self, dataset_key: str) -> tuple[str, ...]:
        """Return names of registered subsets for a dataset key."""
        key = self._normalize_dataset_key(dataset_key)
        return tuple(sorted(self._subsets.get(key, {}).keys()))

    def subset_task_ids(self, dataset_key: str, name: str) -> tuple[str, ...]:
        """Return the task IDs registered for a named subset (e.g., 'Mini', 'easy')."""
        return self._get_named_subset_ids(dataset_key, name)

    def available_task_ids(self, dataset_key: str, config: Optional[Any] = None, auto_download: bool = False) -> list[str]:
        """Return all available task IDs for a dataset key after ensuring dataset availability."""
        spec_key = self._canonical_spec_key(dataset_key)
        if spec_key not in self._specs:
            raise ValueError(f"Environment '{spec_key}' is not registered")

        spec = self._specs[spec_key]
        cfg = self._prepare_config(config, spec.max_episode_steps, spec_key)
        cfg = self._ensure_dataset_available(cfg, spec_key, auto_download)

        parser_entry, _, _ = self._resolve_dataset_meta(spec_key)
        parser_obj = self._import_from_entry_point(parser_entry)
        parser = parser_obj(cfg.dataset) if self._is_class(parser_obj) else parser_obj
        return parser.get_available_task_ids() if hasattr(parser, "get_available_task_ids") else []

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

        # Prepare config and dataset availability
        config = self._prepare_config(kwargs.get("config"), spec.max_episode_steps, dataset_key)
        auto_download = bool(kwargs.get("auto_download", False))

        # Parse selector (may be empty)
        selector = modifiers.get("selector", "")

        # Adjust split for AGI datasets based on selector
        self._maybe_adjust_task_split(config, dataset_key, selector)

        # Ensure dataset exists on disk (optionally download)
        config = self._ensure_dataset_available(config, dataset_key, auto_download)

        # Instantiate the dataset parser from dataset key
        parser_entry, _, _ = self._resolve_dataset_meta(dataset_key)
        parser_obj = self._import_from_entry_point(parser_entry)
        parser = parser_obj(config.dataset) if self._is_class(parser_obj) else parser_obj

        # Resolve episode mode (0=train, 1=eval)
        episode_mode = self._resolve_episode_mode(kwargs.get("episode_mode"), selector)

        # Helper: get all ids and concept groups if available
        def _available_ids(p):
            return p.get_available_task_ids() if hasattr(p, "get_available_task_ids") else []

        def _concept_groups(p):
            return p.get_concept_groups() if hasattr(p, "get_concept_groups") else []

        ids: list[str] = []

        key_l = dataset_key.lower()
        sel_l = selector.lower()

        # Standard synonyms for full sets
        is_full = sel_l in ("", "all", "train", "training", "eval", "evaluation", "test", "corpus")

        # First priority: check for a registered named subset like 'Mini-easy'
        named_ids = ()
        if selector:
            named_ids = self._get_named_subset_ids(dataset_key, selector)
        if named_ids:
            ids = list(named_ids)
        elif key_l in ("concept", "conceptarc", "concept-arc"):
            if is_full:
                ids = _available_ids(parser)
            else:
                # If selector matches a concept group, select that group
                concepts = set(_concept_groups(parser))
                if selector in concepts and hasattr(parser, "get_tasks_in_concept"):
                    ids = parser.get_tasks_in_concept(selector)
                else:
                    # Fall back to single task id if available
                    avail = _available_ids(parser)
                    if selector in avail:
                        ids = [selector]
                    else:
                        raise ValueError(f"Unknown Concept selector '{selector}'. Not a concept group or task id.")
        elif key_l in ("mini", "miniarc", "mini-arc"):
            if is_full:
                ids = _available_ids(parser)
            else:
                avail = _available_ids(parser)
                if selector in avail:
                    ids = [selector]
                else:
                    raise ValueError(f"Unknown Mini selector '{selector}'. Provide 'all' or a valid task id.")
        elif key_l in ("agi1", "arc-agi-1", "agi-1", "agi_1", "agi2", "arc-agi-2", "agi-2", "agi_2"):
            # For split-like selectors, task_split was already adjusted; use all ids
            if is_full:
                ids = _available_ids(parser)
            else:
                # Try to find the specific task id in current split; if not found, try the opposite split
                avail = _available_ids(parser)
                if selector in avail:
                    ids = [selector]
                else:
                    # Try opposite split
                    try:
                        ds = getattr(config, "dataset")
                        current_split = getattr(ds, "task_split", "train")
                        opposite = "evaluation" if current_split in ("train", "training") else "train"
                        setattr(ds, "task_split", opposite)
                        setattr(config, "dataset", ds)
                        parser2 = parser_obj(config.dataset) if self._is_class(parser_obj) else parser_obj
                        avail2 = _available_ids(parser2)
                        if selector in avail2:
                            parser = parser2
                            ids = [selector]
                        else:
                            raise ValueError(f"Task id '{selector}' not found in either split for {dataset_key}.")
                    except Exception as e:
                        raise ValueError(f"Failed to resolve AGI task selector '{selector}': {e}") from e
        else:
            # Generic fallback
            avail = _available_ids(parser)
            if is_full:
                ids = avail
            elif selector in avail:
                ids = [selector]
            else:
                raise ValueError(f"Unknown dataset key '{dataset_key}' or selector '{selector}'")

        if not ids:
            raise ValueError("No tasks resolved for the given selector.")

        # Build stacked buffer using parser, handling cross-split lookups for AGI datasets if needed
        tasks = self._get_tasks_for_ids(parser, parser_obj, config, dataset_key, ids)
        buf = stack_task_list(tasks)

        # Create EnvParams with buffer
        from jaxarc.types import EnvParams as _EnvParams
        params = _EnvParams.from_config(
            config=config,
            buffer=buf,
            subset_indices=None,
            episode_mode=episode_mode,
        )
        return env, params

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
    def _normalize_dataset_key(dataset_key: str) -> str:
        """Normalize dataset key to a canonical lowercase token for internal mapping."""
        key = dataset_key.lower()
        if key in ("mini", "miniarc", "mini-arc"):
            return "mini"
        if key in ("concept", "conceptarc", "concept-arc"):
            return "concept"
        if key in ("agi1", "arc-agi-1", "agi-1", "agi_1"):
            return "agi1"
        if key in ("agi2", "arc-agi-2", "agi-2", "agi_2"):
            return "agi2"
        return key

    @staticmethod
    def _canonical_spec_key(dataset_key: str) -> str:
        """Map a dataset key (including synonyms) to a registered spec key."""
        key = EnvRegistry._normalize_dataset_key(dataset_key)
        if key == "mini":
            return "Mini"
        if key == "concept":
            return "Concept"
        if key == "agi1":
            return "AGI1"
        if key == "agi2":
            return "AGI2"
        # Fallback: assume caller provided exact registered key
        return dataset_key

    def _get_named_subset_ids(self, dataset_key: str, selector: str) -> tuple[str, ...]:
        """Fetch named subset IDs if registered for the dataset_key/selector pair."""
        key = self._normalize_dataset_key(dataset_key)
        subsets = self._subsets.get(key, {})
        return subsets.get(selector.lower(), tuple())

    def _get_tasks_for_ids(self, parser: Any, parser_entry_obj: Any, config: Any, dataset_key: str, ids: list[str]) -> list[Any]:
        """Load tasks by ID using the current parser. For AGI datasets, missing IDs are looked up in the opposite split."""
        tasks: list[Any] = []
        missing: list[str] = []
        for tid in ids:
            try:
                tasks.append(parser.get_task_by_id(tid))
            except Exception:
                missing.append(tid)
        if not missing:
            return tasks

        key_l = dataset_key.lower()
        if key_l in ("agi1", "arc-agi-1", "agi-1", "agi_1", "agi2", "arc-agi-2", "agi-2", "agi_2"):
            try:
                ds = getattr(config, "dataset")
                current_split = getattr(ds, "task_split", "train")
                opposite = "evaluation" if current_split in ("train", "training") else "train"
                setattr(ds, "task_split", opposite)
                setattr(config, "dataset", ds)
                parser2 = parser_entry_obj(config.dataset) if self._is_class(parser_entry_obj) else parser_entry_obj
                still_missing: list[str] = []
                for tid in list(missing):
                    try:
                        tasks.append(parser2.get_task_by_id(tid))
                    except Exception:
                        still_missing.append(tid)
                missing = still_missing
            except Exception:
                # Fall through to error below
                pass

        if missing:
            raise ValueError(f"Some task ids were not found for dataset '{dataset_key}': {missing}")
        return tasks

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
            from jaxarc.utils.config import get_config
            from jaxarc.configs.environment_config import EnvironmentConfig
            from jaxarc.configs.dataset_config import DatasetConfig
        except Exception as e:
            raise ValueError(
                "Could not import configuration types. Ensure configurations "
                "are available or provide a ready 'config' object."
            ) from e

        cfg = config if config is not None else JaxArcConfig.from_hydra(get_config())

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
        """Resolve episode mode using explicit value or selector token (train/eval)."""
        if episode_mode is not None:
            return int(episode_mode)
        if not selector:
            return 0
        sel = selector.lower()
        if sel in ("train", "training"):
            return 0
        if sel in ("eval", "evaluation", "test", "corpus"):
            return 1
        return 0

    @staticmethod
    def _resolve_dataset_meta(dataset_key: str) -> tuple[str, str, str]:
        """Map dataset key to (parser_entry, downloader_method, normalized_dataset_name)."""
        key = dataset_key.lower()
        if key in ("mini", "miniarc", "mini-arc"):
            return ("jaxarc.parsers:MiniArcParser", "download_miniarc", "MiniARC")
        if key in ("concept", "conceptarc", "concept-arc"):
            return ("jaxarc.parsers:ConceptArcParser", "download_conceptarc", "ConceptARC")
        if key in ("agi1", "arc-agi-1", "agi-1", "agi_1"):
            return ("jaxarc.parsers:ArcAgiParser", "download_arc_agi_1", "ARC-AGI-1")
        if key in ("agi2", "arc-agi-2", "agi-2", "agi_2"):
            return ("jaxarc.parsers:ArcAgiParser", "download_arc_agi_2", "ARC-AGI-2")
        # Fallback assumes dataset_key is already normalized and no downloader available
        return ("jaxarc.parsers:ArcAgiParser", "", dataset_key)

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
        """Ensure dataset exists at config.dataset.dataset_path. Optionally download if missing.
        Functional: returns (potentially) updated config without in-place mutation.
        """
        try:
            ds = getattr(config, "dataset")
            dataset_path = getattr(ds, "dataset_path", "")
            path = Path(dataset_path) if dataset_path else None
        except Exception as e:
            logger.warning(f"Unable to access dataset configuration: {e}")
            return config

        _, downloader_method, normalized_name = EnvRegistry._resolve_dataset_meta(dataset_key)

        # If path is missing or does not exist
        if not path or not path.exists():
            msg = f"Dataset path is missing or not found for '{normalized_name}'"
            if not auto_download or not downloader_method:
                logger.warning(msg + " - set auto_download=True to attempt download.")
                raise ValueError("Dataset not available on disk and auto_download is disabled.")
            # Attempt download into config.dataset.dataset_path (create default if empty)
            try:
                # Decide target directory:
                # - If dataset_path provided, use it
                # - Otherwise, set a default under ./data/{normalized_name} and persist it to config immutably
                if dataset_path:
                    target_dir = here(dataset_path)
                else:
                    default_dir = here() / "data" / normalized_name
                    target_dir = default_dir
                    # immutably set dataset_path and dataset_name
                    import equinox as eqx  # local import to avoid top-level dependency cycles
                    ds = eqx.tree_at(lambda d: d.dataset_path, ds, str(default_dir))
                    ds = eqx.tree_at(lambda d: d.dataset_name, ds, normalized_name)
                    config = eqx.tree_at(lambda c: c.dataset, config, ds)

                # Ensure parent directories exist
                target_dir.parent.mkdir(parents=True, exist_ok=True)

                # Download into the chosen target directory
                downloader = DatasetDownloader(output_dir=target_dir.parent)
                download_fn = getattr(downloader, downloader_method)
                target_dir = download_fn(target_dir=target_dir)

                # Update dataset_path and dataset_name to reflect the final target location (immutably)
                import equinox as eqx  # local import to avoid top-level dependency cycles
                ds = getattr(config, "dataset")
                ds = eqx.tree_at(lambda d: d.dataset_path, ds, str(target_dir))
                ds = eqx.tree_at(lambda d: d.dataset_name, ds, normalized_name)
                config = eqx.tree_at(lambda c: c.dataset, config, ds)
                logger.info(f"Downloaded dataset '{normalized_name}' to: {target_dir}")
            except DatasetDownloadError as e:
                logger.error(f"Automatic dataset download failed: {e}")
                raise

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

def register_subset(dataset_key: str, name: str, task_ids: list[str] | tuple[str, ...]) -> None:
    """Register a named subset for a dataset key, enabling IDs like 'Mini-easy'."""
    _registry.register_subset(dataset_key, name, task_ids)

def available_task_ids(dataset_key: str, config: Optional[Any] = None, auto_download: bool = False) -> list[str]:
    """List available task IDs for the given dataset key (after ensuring availability)."""
    return _registry.available_task_ids(dataset_key, config=config, auto_download=auto_download)

def available_named_subsets(dataset_key: str) -> tuple[str, ...]:
    """List names of registered named subsets for a dataset key (e.g., ('easy', 'hard'))."""
    return _registry.available_named_subsets(dataset_key)

def subset_task_ids(dataset_key: str, name: str) -> tuple[str, ...]:
    """Return the task IDs registered under the named subset for the dataset."""
    return _registry.subset_task_ids(dataset_key, name)
