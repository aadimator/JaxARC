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

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from jaxarc.utils import DatasetError, DatasetManager
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

    def register_subset(
        self, dataset_key: str, name: str, task_ids: list[str] | tuple[str, ...]
    ) -> None:
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
        ids_tuple: tuple[str, ...] = (
            tuple(task_ids) if not isinstance(task_ids, tuple) else task_ids
        )
        if key not in self._subsets:
            self._subsets[key] = {}
        self._subsets[key][sel] = ids_tuple

    def available_named_subsets(
        self, dataset_key: str, include_builtin: bool = True
    ) -> tuple[str, ...]:
        """Return names of available subsets for a dataset.

        Args:
            dataset_key: Dataset name (Mini, Concept, AGI1, AGI2)
            include_builtin: Include built-in selectors ('all', 'train', 'eval')
                            and concept groups (default: True)

        Returns:
            Tuple of subset names, sorted alphabetically

        Examples:
            >>> available_named_subsets("Mini")
            ('all',)  # Mini doesn't have train/eval splits

            >>> available_named_subsets("Concept")
            ('AboveBelow', 'Center', 'all', ...)  # Includes concept groups

            >>> available_named_subsets("AGI1")
            ('all', 'eval', 'train')  # AGI has splits

            >>> available_named_subsets("Mini", include_builtin=False)
            ()  # Only custom subsets
        """
        key = self._normalize_dataset_key(dataset_key)

        # Start with manually registered subsets
        subsets = set(self._subsets.get(key, {}).keys())

        if include_builtin:
            # Add 'all' for everyone
            subsets.add("all")

            # Only AGI datasets have train/eval splits
            if key in ("agi1", "agi2"):
                subsets.update(["train", "eval"])

            # Add concept groups for ConceptARC
            if key == "concept":
                try:
                    # Try to get concept groups if dataset is available
                    spec_key = self._canonical_spec_key(dataset_key)
                    if spec_key in self._specs:
                        spec = self._specs[spec_key]
                        cfg = self._prepare_config(
                            None, spec.max_episode_steps, spec_key
                        )
                        try:
                            cfg = self._ensure_dataset_available(
                                cfg, spec_key, auto_download=False
                            )
                            parser = self._create_parser(cfg)
                            if hasattr(parser, "get_concept_groups"):
                                concepts = parser.get_concept_groups()
                                subsets.update(concepts)
                        except Exception:
                            # Dataset not available, skip concept groups
                            pass
                except Exception:
                    # If we can't load concepts, just continue
                    pass

        return tuple(sorted(subsets))

    def get_subset_task_ids(
        self,
        dataset_key: str,
        selector: str = "all",
        config: Optional[Any] = None,
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
            ['task1', 'task2', 'task3']  # Only tasks in 'easy' subset

            >>> get_subset_task_ids("Concept", "Center")
            ['Center_001', 'Center_002', ...]  # Tasks in Center concept

            >>> get_subset_task_ids("Mini", "Most_Common_color_l6ab0lf3xztbyxsu3p")
            ['Most_Common_color_l6ab0lf3xztbyxsu3p']  # Single task
        """
        spec_key = self._canonical_spec_key(dataset_key)
        if spec_key not in self._specs:
            msg = f"Environment '{spec_key}' is not registered"
            raise ValueError(msg)

        spec = self._specs[spec_key]
        cfg = self._prepare_config(config, spec.max_episode_steps, spec_key)

        # Adjust split for AGI datasets (returns modified config)
        cfg = self._maybe_adjust_task_split(cfg, dataset_key, selector)

        # Ensure dataset available and create parser
        cfg = self._ensure_dataset_available(cfg, spec_key, auto_download)
        parser = self._create_parser(cfg)

        # Use unified resolution
        return self._resolve_selector_to_task_ids(dataset_key, selector, parser)

    def subset_task_ids(self, dataset_key: str, name: str) -> tuple[str, ...]:
        """Return the task IDs registered for a named subset (e.g., 'Mini', 'easy')."""
        return self._get_named_subset_ids(dataset_key, name)

    def available_task_ids(
        self,
        dataset_key: str,
        config: Optional[Any] = None,
        auto_download: bool = False,
    ) -> list[str]:
        """Return all available task IDs for a dataset key after ensuring dataset availability."""
        spec_key = self._canonical_spec_key(dataset_key)
        if spec_key not in self._specs:
            raise ValueError(f"Environment '{spec_key}' is not registered")

        spec = self._specs[spec_key]
        cfg = self._prepare_config(config, spec.max_episode_steps, spec_key)
        cfg = self._ensure_dataset_available(cfg, spec_key, auto_download)

        dataset_config = cfg.dataset
        parser_entry = getattr(
            dataset_config, "parser_entry_point", "jaxarc.parsers:ArcAgiParser"
        )
        parser_obj = self._import_from_entry_point(parser_entry)
        parser = parser_obj(cfg.dataset) if self._is_class(parser_obj) else parser_obj
        return (
            parser.get_available_task_ids()
            if hasattr(parser, "get_available_task_ids")
            else []
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

        # If params explicitly provided, use them
        if "params" in kwargs and kwargs["params"] is not None:
            return env_obj(
                config=kwargs["config"], buffer=kwargs["params"].buffer
            ), kwargs["params"]

        # Prepare config and dataset availability
        config = self._prepare_config(
            kwargs.get("config"), spec.max_episode_steps, dataset_key
        )
        auto_download = bool(kwargs.get("auto_download", False))

        # Parse selector (may be empty)
        selector = modifiers.get("selector", "")

        # Adjust split for AGI datasets based on selector (returns modified config)
        config = self._maybe_adjust_task_split(config, dataset_key, selector)

        # Ensure dataset exists on disk (optionally download)
        config = self._ensure_dataset_available(config, dataset_key, auto_download)

        # Instantiate the dataset parser from config
        parser = self._create_parser(config)

        # For AGI datasets, we may need the parser_obj for cross-split lookups
        dataset_config = config.dataset
        parser_entry = getattr(
            dataset_config, "parser_entry_point", "jaxarc.parsers:ArcAgiParser"
        )
        parser_obj = self._import_from_entry_point(parser_entry)

        # Resolve episode mode (0=train, 1=eval)
        episode_mode = self._resolve_episode_mode(kwargs.get("episode_mode"), selector)

        # UNIFIED RESOLUTION - works for all selector types
        try:
            ids = self._resolve_selector_to_task_ids(
                dataset_key, selector if selector else "all", parser
            )
        except ValueError as e:
            msg = f"Failed to resolve '{id}': {e}"
            raise ValueError(msg) from e

        if not ids:
            msg = "No tasks resolved for the given selector."
            raise ValueError(msg)

        # Build stacked buffer using parser, handling cross-split lookups for AGI datasets if needed
        tasks = self._get_tasks_for_ids(parser, parser_obj, config, dataset_key, ids)
        buf = stack_task_list(tasks)

        env = env_obj(config=config, buffer=buf, episode_mode=episode_mode)

        return env, env.params

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _resolve_selector_to_task_ids(
        self, dataset_key: str, selector: str, parser: Any
    ) -> list[str]:
        """Resolve any selector to a list of task IDs.

        Priority order:
        1. Named subset (e.g., 'easy' from register_subset)
        2. Built-in selectors ('all', 'train', 'eval')
        3. Concept groups (ConceptARC: 'AboveBelow', 'Center', etc.)
        4. Single task ID (e.g., 'Most_Common_color_l6ab0lf3xztbyxsu3p')

        Args:
            dataset_key: Dataset key (Mini, Concept, AGI1, AGI2)
            selector: Selector string from make("Dataset-{selector}")
            parser: Initialized parser instance

        Returns:
            List of resolved task IDs

        Raises:
            ValueError: If selector cannot be resolved
        """
        # 1. Check named subsets first (highest priority)
        named_ids = self._get_named_subset_ids(dataset_key, selector)
        if named_ids:
            return list(named_ids)

        # 2. Check built-in selectors
        sel_l = selector.lower()
        if sel_l in (
            "",
            "all",
            "train",
            "training",
            "eval",
            "evaluation",
            "test",
            "corpus",
        ):
            return self._get_all_task_ids(parser)

        # 3. Concept-specific: check concept groups
        key_l = self._normalize_dataset_key(dataset_key)
        if key_l == "concept":
            if hasattr(parser, "get_concept_groups") and hasattr(
                parser, "get_tasks_in_concept"
            ):
                concepts = parser.get_concept_groups()
                if selector in concepts:
                    return list(parser.get_tasks_in_concept(selector))

        # 4. Try as single task ID
        all_ids = self._get_all_task_ids(parser)
        if selector in all_ids:
            return [selector]

        # 5. Failed to resolve - provide helpful error
        available_options = self._describe_available_selectors(dataset_key, parser)
        raise ValueError(
            f"Unknown selector '{selector}' for {dataset_key}.\n"
            f"Available options: {available_options}"
        )

    def _get_all_task_ids(self, parser: Any) -> list[str]:
        """Get all available task IDs from parser."""
        if hasattr(parser, "get_available_task_ids"):
            return parser.get_available_task_ids()
        return []

    def _describe_available_selectors(self, dataset_key: str, parser: Any) -> str:
        """Create a helpful description of valid selectors for error messages."""
        # Get all available named subsets (includes built-ins, custom subsets, and concepts)
        named = self.available_named_subsets(dataset_key, include_builtin=True)

        options = [f"'{n}'" for n in named] if named else []

        # Add note about task IDs
        options.append("or any valid task ID")

        return ", ".join(options)

    def _create_parser(self, config: Any) -> Any:
        """Create parser instance from config.

        Extracted to eliminate duplication across dataset branches.
        """
        dataset_config = config.dataset
        parser_entry = getattr(
            dataset_config, "parser_entry_point", "jaxarc.parsers:ArcAgiParser"
        )
        parser_obj = self._import_from_entry_point(parser_entry)
        return parser_obj(config.dataset) if self._is_class(parser_obj) else parser_obj

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

    def _get_tasks_for_ids(
        self,
        parser: Any,
        parser_entry_obj: Any,
        config: Any,
        dataset_key: str,
        ids: list[str],
    ) -> list[Any]:
        """Load tasks by ID using the current parser. For AGI datasets, missing IDs are looked up in the opposite split."""
        import equinox as eqx

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
        if key_l in (
            "agi1",
            "arc-agi-1",
            "agi-1",
            "agi_1",
            "agi2",
            "arc-agi-2",
            "agi-2",
            "agi_2",
        ):
            try:
                ds = config.dataset
                current_split = getattr(ds, "task_split", "train")
                opposite = (
                    "evaluation" if current_split in ("train", "training") else "train"
                )
                logger.debug(
                    f"Looking for {len(missing)} missing tasks in opposite split '{opposite}'"
                )

                # Properly update immutable config using eqx.tree_at
                ds_opposite = eqx.tree_at(lambda d: d.task_split, ds, opposite)
                config_opposite = eqx.tree_at(lambda c: c.dataset, config, ds_opposite)

                # Create parser for opposite split
                parser2 = (
                    parser_entry_obj(config_opposite.dataset)
                    if self._is_class(parser_entry_obj)
                    else parser_entry_obj
                )

                still_missing: list[str] = []
                for tid in list(missing):
                    try:
                        tasks.append(parser2.get_task_by_id(tid))
                        logger.debug(
                            f"Found task '{tid}' in opposite split '{opposite}'"
                        )
                    except Exception:
                        still_missing.append(tid)
                missing = still_missing
            except Exception as e:
                logger.warning(f"Failed to lookup missing tasks in opposite split: {e}")
                # Fall through to error below

        if missing:
            raise ValueError(
                f"Some task ids were not found for dataset '{dataset_key}': {missing}"
            )
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
    def _prepare_config(
        config: Optional[Any], max_episode_steps: int, dataset_key: str
    ) -> Any:
        """Prepare a JaxArcConfig, applying overrides when possible.

        - If no config provided, instantiate a default JaxArcConfig.
        - Ensure environment.max_episode_steps matches the spec if possible.
        - Normalize dataset configuration based on spec.dataset_key.
        """
        # Import locally to avoid hard dependency at module import time
        try:
            from jaxarc.configs.environment_config import EnvironmentConfig
            from jaxarc.configs.main_config import JaxArcConfig
            from jaxarc.utils.core import get_config
        except Exception as e:
            raise ValueError(
                "Could not import configuration types. Ensure configurations "
                "are available or provide a ready 'config' object."
            ) from e
        # If config not provided, prefer a safe construction path that avoids Hydra re-init
        if config is None:
            try:
                # Detect if a Hydra app is already initialized in this process
                from hydra.core.global_hydra import GlobalHydra  # type: ignore

                gh = GlobalHydra.instance()
                hydra_active = gh.is_initialized()
            except Exception:
                hydra_active = False

            if hydra_active:
                # Avoid re-initializing Hydra: build a default config directly
                cfg = JaxArcConfig()
            else:
                # Standalone usage: use Hydra defaults
                cfg = JaxArcConfig.from_hydra(get_config())
        else:
            cfg = config

        # Enforce max_episode_steps
        try:
            cfg.environment = EnvironmentConfig(max_episode_steps=max_episode_steps)
        except Exception:
            pass

        # Best-effort dataset normalization is handled later by _ensure_dataset_available

        return cfg

    @staticmethod
    def _resolve_episode_mode(
        episode_mode: Optional[int], selector: Optional[str]
    ) -> int:
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
    def _load_dataset_config(dataset_key: str) -> Any:
        """Load dataset config from packaged YAML without initializing Hydra.

        This avoids conflicts when JaxARC is embedded inside an existing Hydra app.
        Returns a DatasetConfig instance for the requested dataset key.
        """
        try:
            # Map normalized key -> dataset YAML file name inside jaxarc/conf/dataset
            key_lower = dataset_key.lower()
            file_name: Optional[str] = None
            if key_lower in ("mini", "miniarc", "mini-arc"):
                file_name = "mini_arc.yaml"
            elif key_lower in ("concept", "conceptarc", "concept-arc"):
                file_name = "concept_arc.yaml"
            elif key_lower in ("agi1", "arc-agi-1", "agi-1", "agi_1"):
                file_name = "arc_agi_1.yaml"
            elif key_lower in ("agi2", "arc-agi-2", "agi-2", "agi_2"):
                file_name = "arc_agi_2.yaml"
            else:
                raise ValueError(f"Unknown dataset key: {dataset_key}")

            # Load YAML via importlib.resources to avoid file path issues
            import importlib.resources as pkg_resources
            import io

            import yaml
            from omegaconf import DictConfig, OmegaConf

            from jaxarc.configs.dataset_config import DatasetConfig

            dataset_dir = pkg_resources.files("jaxarc") / "conf" / "dataset"
            yaml_path = dataset_dir / file_name
            # Read text and convert to DictConfig
            with yaml_path.open("r", encoding="utf-8") as f:
                yaml_text = f.read()
            data = yaml.safe_load(io.StringIO(yaml_text)) or {}
            cfg: DictConfig = OmegaConf.create(data)
            return DatasetConfig.from_hydra(cfg)
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset config for '{dataset_key}': {e}"
            ) from e

    def _maybe_adjust_task_split(
        self, config: Any, dataset_key: str, selector: Optional[str]
    ) -> Any:
        """Adjust config.dataset.task_split based on selector for AGI datasets.

        Returns the modified config (necessary because equinox objects are immutable).
        """
        try:
            import equinox as eqx

            sel = (selector or "").lower()
            if dataset_key.lower() in (
                "agi1",
                "arc-agi-1",
                "agi-1",
                "agi_1",
                "agi2",
                "arc-agi-2",
                "agi-2",
                "agi_2",
            ):
                new_split = None
                if sel in ("train", "training"):
                    logger.debug(
                        f"Adjusting task_split to 'train' for selector '{sel}'"
                    )
                    new_split = "train"
                elif sel in ("eval", "evaluation", "test", "corpus"):
                    logger.debug(
                        f"Adjusting task_split to 'evaluation' for selector '{sel}'"
                    )
                    new_split = "evaluation"

                if new_split is not None:
                    # Use eqx.tree_at to properly modify immutable config
                    ds = eqx.tree_at(lambda d: d.task_split, config.dataset, new_split)
                    config = eqx.tree_at(lambda c: c.dataset, config, ds)

            return config
        except Exception:
            # Best-effort only
            return config

    @staticmethod
    def _infer_subset_ids(
        parser: Any, dataset_key: str, selector: str
    ) -> tuple[str, ...]:
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
                if sel in (
                    "train",
                    "training",
                    "eval",
                    "evaluation",
                    "test",
                    "corpus",
                    "all",
                ):
                    return tuple(parser.get_available_task_ids())
                if hasattr(parser, "get_concept_groups") and hasattr(
                    parser, "get_tasks_in_concept"
                ):
                    concepts = set(parser.get_concept_groups())
                    if selector in concepts:
                        return tuple(parser.get_tasks_in_concept(selector))
                return tuple()

            # MiniARC subsets: treat train/eval/all as "all tasks"
            if key in ("mini", "miniarc", "mini-arc"):
                if sel in (
                    "train",
                    "training",
                    "eval",
                    "evaluation",
                    "test",
                    "corpus",
                    "all",
                ):
                    return tuple(parser.get_available_task_ids())
                return tuple()

            # AGI subsets: use current parser split's available IDs
            if key in (
                "agi1",
                "arc-agi-1",
                "agi-1",
                "agi_1",
                "agi2",
                "arc-agi-2",
                "agi-2",
                "agi_2",
            ):
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
    def _ensure_dataset_available(
        config: Any, dataset_key: str, auto_download: bool
    ) -> Any:
        """Ensure dataset exists and matches the requested dataset key.

        - If config.dataset exists but doesn't match the requested key, replace it.
        - Load dataset config directly from packaged YAML (no Hydra init).
        - Ensure files are present via DatasetManager and fix dataset_path.
        """
        import equinox as eqx

        from jaxarc.configs.dataset_config import DatasetConfig

        manager = DatasetManager()

        # Load the desired dataset configuration from YAML
        desired_ds: DatasetConfig = EnvRegistry._load_dataset_config(dataset_key)

        # Preserve task_split from current config if it was modified (e.g., by _maybe_adjust_task_split)
        # This must happen BEFORE replacing the dataset config
        current_ds = getattr(config, "dataset", None)
        if isinstance(current_ds, DatasetConfig):
            same = (
                str(current_ds.dataset_name).strip().lower()
                == str(desired_ds.dataset_name).strip().lower()
            )

            # Preserve task_split if it differs from default (was modified by _maybe_adjust_task_split)
            if hasattr(current_ds, "task_split") and hasattr(desired_ds, "task_split"):
                if current_ds.task_split != desired_ds.task_split:
                    logger.debug(
                        f"Preserving task_split='{current_ds.task_split}' (was modified by selector)"
                    )
                    desired_ds = eqx.tree_at(
                        lambda d: d.task_split, desired_ds, current_ds.task_split
                    )

            if not same:
                logger.debug(
                    f"Overriding provided DatasetConfig '{current_ds.dataset_name}' with '{desired_ds.dataset_name}' from key '{dataset_key}'."
                )

            config = eqx.tree_at(lambda c: c.dataset, config, desired_ds)
        else:
            # No valid dataset found in config, set to desired
            config = eqx.tree_at(lambda c: c.dataset, config, desired_ds)

        # Now ensure the dataset files are on disk and update dataset_path
        try:
            dataset_path = manager.ensure_dataset_available(
                config, auto_download=auto_download
            )
            ds = config.dataset
            ds = eqx.tree_at(lambda d: d.dataset_path, ds, str(dataset_path))
            config = eqx.tree_at(lambda c: c.dataset, config, ds)
            return config
        except DatasetError as e:
            logger.error(f"Dataset management failed: {e}")
            raise ValueError(f"Dataset not available: {e}") from e


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


def register_subset(
    dataset_key: str, name: str, task_ids: list[str] | tuple[str, ...]
) -> None:
    """Register a named subset for a dataset key, enabling IDs like 'Mini-easy'."""
    _registry.register_subset(dataset_key, name, task_ids)


def get_subset_task_ids(
    dataset_key: str,
    selector: str = "all",
    config: Optional[Any] = None,
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
    dataset_key: str, config: Optional[Any] = None, auto_download: bool = False
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
