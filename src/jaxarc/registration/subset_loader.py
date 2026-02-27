"""Subset inference and YAML-based subset loading for JaxARC registration.

Provides:
- ``infer_subset_ids``: Infer task IDs from standard named subsets.
- ``load_subset``: Load a named subset from a YAML file on disk.
- ``load_subset_if_needed``: Load a subset only if not already registered.
- ``load_all_subsets_for_dataset``: Discover and load all YAML subsets for a dataset.

YAML file format (in ``{config_root}/env/jaxarc/subsets/{Dataset}/{name}.yaml``)::

    task_ids:
      - "task_id_1"
      - "task_id_2"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# YAML subset loading
# ---------------------------------------------------------------------------


def _find_config_root() -> Path | None:
    """Find the config root directory using pyprojroot.

    Returns the ``configs/`` directory under the project root, or None.
    """
    try:
        from pyprojroot import here  # type: ignore[import-untyped]

        project_root = here()
        configs_dir = project_root / "configs"
        if configs_dir.exists() and configs_dir.is_dir():
            return configs_dir
        logger.debug(f"Project root at {project_root}, but no configs/ directory")
    except Exception as exc:
        logger.debug(f"Could not find project root via pyprojroot: {exc}")
    return None


def load_subset(
    name: str,
    dataset: str,
    config_root: Path | None = None,
) -> list[str] | None:
    """Load task IDs for a named subset from a YAML file.

    Args:
        name: Subset name (e.g. ``"easy"``).
        dataset: Dataset key (e.g. ``"Mini"``, ``"AGI1"``).
        config_root: Path to the ``configs/`` directory.  When *None*,
            :pep:`pyprojroot` is used to locate it automatically.

    Returns:
        List of task ID strings, or *None* if the file was not found or
        could not be parsed.
    """
    if config_root is None:
        config_root = _find_config_root()
        if config_root is None:
            logger.debug("Could not locate configs directory for subset loading")
            return None

    subset_file = config_root / "env" / "jaxarc" / "subsets" / dataset / f"{name}.yaml"
    if not subset_file.exists():
        logger.debug(f"Subset file not found: {subset_file}")
        return None

    try:
        import yaml  # type: ignore[import-untyped]

        with subset_file.open() as fh:
            data = yaml.safe_load(fh)

        if not data or "task_ids" not in data or not data["task_ids"]:
            logger.debug(f"Empty subset {dataset}/{name} in {subset_file}")
            return None

        task_ids: list[str] = [str(tid) for tid in data["task_ids"]]
        return task_ids
    except Exception as exc:
        logger.warning(f"Failed to load subset {dataset}/{name}: {exc}")
        return None


def load_subset_if_needed(
    name: str,
    dataset: str,
    config_root: Path | None = None,
) -> bool:
    """Load and register a subset only if it is not already registered.

    Returns True when the subset is available (either already present or
    freshly loaded).
    """
    # Deferred import to prevent circular dependency
    from jaxarc.registration import available_named_subsets, register_subset

    if name.lower() in available_named_subsets(dataset):
        return True

    task_ids = load_subset(name, dataset, config_root)
    if task_ids is None:
        return False

    register_subset(dataset, name, task_ids)
    logger.info(f"Registered subset '{dataset}-{name}' ({len(task_ids)} tasks)")
    return True


def load_all_subsets_for_dataset(
    dataset: str,
    config_root: Path | None = None,
) -> int:
    """Discover and register all YAML-defined subsets for *dataset*.

    Returns the number of subsets successfully loaded.
    """
    from jaxarc.registration import register_subset

    if config_root is None:
        config_root = _find_config_root()
        if config_root is None:
            return 0

    dataset_dir = config_root / "env" / "jaxarc" / "subsets" / dataset
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        logger.debug(f"No subsets directory for {dataset}: {dataset_dir}")
        return 0

    count = 0
    for yaml_file in sorted(dataset_dir.glob("*.yaml")):
        subset_name = yaml_file.stem
        task_ids = load_subset(subset_name, dataset, config_root)
        if task_ids is not None:
            register_subset(dataset, subset_name, task_ids)
            logger.info(
                f"Registered subset '{dataset}-{subset_name}' ({len(task_ids)} tasks)"
            )
            count += 1
    return count


# ---------------------------------------------------------------------------
# Standard subset inference (parser-based)
# ---------------------------------------------------------------------------


def infer_subset_ids(parser: Any, dataset_key: str, selector: str) -> tuple[str, ...]:
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
