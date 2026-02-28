"""Core registry implementation for JaxARC environments.

Contains the EnvSpec dataclass and EnvRegistry class with instance methods
that manage specs and subsets. Standalone config/subset functions are
imported from sibling modules.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from jaxarc.utils.buffer import stack_task_list

from .config_prep import (
    ensure_dataset_available,
    maybe_adjust_task_split,
    prepare_config,
    resolve_episode_mode,
)

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
                        cfg = prepare_config(None, spec.max_episode_steps, spec_key)
                        try:
                            cfg = ensure_dataset_available(
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
        cfg = prepare_config(config, spec.max_episode_steps, spec_key)

        # Adjust split for AGI datasets (returns modified config)
        cfg = maybe_adjust_task_split(cfg, dataset_key, selector)

        # Ensure dataset available and create parser
        cfg = ensure_dataset_available(cfg, spec_key, auto_download)
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
        cfg = prepare_config(config, spec.max_episode_steps, spec_key)
        cfg = ensure_dataset_available(cfg, spec_key, auto_download)

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
        config = prepare_config(
            kwargs.get("config"), spec.max_episode_steps, dataset_key
        )
        auto_download = bool(kwargs.get("auto_download", False))

        # Parse selector (may be empty)
        selector = modifiers.get("selector", "")

        # Adjust split for AGI datasets based on selector (returns modified config)
        config = maybe_adjust_task_split(config, dataset_key, selector)

        # Ensure dataset exists on disk (optionally download)
        config = ensure_dataset_available(config, dataset_key, auto_download)

        # Instantiate the dataset parser from config
        parser = self._create_parser(config)

        # For AGI datasets, we may need the parser_obj for cross-split lookups
        dataset_config = config.dataset
        parser_entry = getattr(
            dataset_config, "parser_entry_point", "jaxarc.parsers:ArcAgiParser"
        )
        parser_obj = self._import_from_entry_point(parser_entry)

        # Resolve episode mode (0=train, 1=eval)
        episode_mode = resolve_episode_mode(kwargs.get("episode_mode"), selector)

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
