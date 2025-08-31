"""
Host-side logging utilities for building enriched logging payloads.

These helpers are intended to be executed outside of JAX-transformed code (on the
host) and provide a consistent, best-effort way to construct the dictionaries
that logging handlers expect. The library avoids implicit discovery of large
objects (like `EnvParams.buffer`) inside logging code — callers should explicitly
pass `params` when task-level metadata is desired.

Key utilities:
- build_step_logging_payload(...) : Create a step_data dict suitable for
  ExperimentLogger.log_step and visualization handlers.
- build_task_metadata_from_params(...) : Given EnvParams and an integer task_idx,
  extract lightweight task metadata (train/test pair counts and optional task id).

Usage (host-side):
    payload = build_step_logging_payload(
        params=params,                # optional, host-only
        before_state=state_before,
        after_state=state_after,
        action=action,
        reward=reward,
        info=info,
        step_num=step,
        episode_num=episode,
        include_task_meta=True,
    )
    experiment_logger.log_step(payload)

Note:
- These functions are intentionally defensive and best-effort: they never raise
  on missing fields and they avoid deep copies of large arrays unless explicitly
  requested by the caller.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger

# Import lightweight helpers from project (host-side safe)
from jaxarc.utils.task_manager import extract_task_id_from_index, get_task_id_globally

try:
    # Import types for type hints only; these imports should be fine on host side
    from jaxarc.types import EnvParams
    from jaxarc.state import State
except Exception:  # pragma: no cover - defensively handle import environments
    EnvParams = Any  # type: ignore
    State = Any  # type: ignore


def _to_python_int(x: Any) -> Optional[int]:
    """Convert various scalar-like values (numpy, JAX, Python) to Python int.

    Returns None if conversion fails.
    """
    if x is None:
        return None
    try:
        # numpy scalars, python ints
        return int(np.asarray(x).item())
    except Exception:
        try:
            return int(x)
        except Exception:
            return None


def to_python_int(x: Any) -> Optional[int]:
    """Public helper: convert scalar-like value to Python int or return None.

    This is a small, well-tested helper intended for use across logging handlers
    and other host-side utilities to normalize scalar values coming from JAX /
    numpy / Python objects.
    """
    return _to_python_int(x)


def _to_python_float(x: Any) -> Optional[float]:
    """Convert various scalar-like values (numpy, JAX, Python) to Python float.

    Returns None if conversion fails.
    """
    if x is None:
        return None
    try:
        return float(np.asarray(x).item())
    except Exception:
        try:
            return float(x)
        except Exception:
            return None


def to_python_float(x: Any) -> Optional[float]:
    """Public helper: convert scalar-like value to Python float or return None."""
    return _to_python_float(x)


def to_python_scalar(x: Any) -> Optional[Any]:
    """Coerce a scalar-like value to a native Python scalar when possible.

    This function attempts to extract a Python scalar from JAX / numpy scalar
    wrappers, falling back to common coercions. It returns None on failure.

    Ordering:
      - Try numpy/JAX .item() extraction first (covers numpy and jax scalars).
      - Then try int(), float(), bool() coercions in that order.
      - Finally return the original value if it is already a Python scalar.
    """
    if x is None:
        return None

    # Fast path for common Python scalars
    if isinstance(x, (int, float, bool, str)):
        return x

    # Try numpy / JAX extraction
    try:
        arr = np.asarray(x)
        # Only attempt .item() on 0-d arrays
        if hasattr(arr, "shape") and arr.shape == ():
            return arr.item()
    except Exception:
        pass

    # Try primitive coercions as a last resort
    try:
        return int(x)
    except Exception:
        pass

    try:
        return float(x)
    except Exception:
        pass

    try:
        return bool(x)
    except Exception:
        pass

    # Give up
    return None


def build_task_metadata_from_params(params: EnvParams, task_idx: Any) -> Dict[str, Any]:
    """
    Extract lightweight metadata for a given task index from EnvParams.buffer.

    This function is host-side only and will index into the stacked `params.buffer`
    (a pytree of arrays) in a best-effort manner. It returns small scalar metadata
    only (counts and task id) — it intentionally avoids returning full grids
    unless a compact parsed `task_object` is discoverable.

    The returned dict has been extended to include:
      - task_object: A minimal parsed task object (if available) or a small dict
                     describing input/output examples.
      - task_structure: small summary with input/output grid shapes and
                        an estimated max color value where possible.

    Args:
        params: EnvParams containing a `buffer` attribute (stacked pytree)
        task_idx: Scalar index (int / numpy scalar) selecting the task

    Returns:
        dict with keys:
            - task_idx (int)
            - task_id (str | None)
            - num_train_pairs (int | None)
            - num_test_pairs (int | None)
            - task_object (object | None)          # best-effort, may be a small dict
            - task_structure (dict | None)         # input/output shapes, max_colors_used (best-effort)
    """
    out: Dict[str, Any] = {
        "task_idx": _to_python_int(task_idx),
        "task_id": None,
        "num_train_pairs": None,
        "num_test_pairs": None,
        "task_object": None,
        "task_structure": None,
    }

    if params is None:
        return out

    # Defensive: ensure buffer exists
    buf = getattr(params, "buffer", None)
    if buf is None:
        return out

    # Helper to coerce index to int when needed
    idx_int = _to_python_int(task_idx)

    # Try to index into buffer fields in a robust way and also attempt to extract
    # a parsed task object (if present) or at least small descriptive fields.
    try:
        # Use attribute access where available for counts and task index
        num_train_pairs = getattr(buf, "num_train_pairs", None)
        num_test_pairs = getattr(buf, "num_test_pairs", None)
        task_index = getattr(buf, "task_index", None)

        # Coerce and slice counts if they are sequences/arrays
        if num_train_pairs is not None:
            try:
                if hasattr(num_train_pairs, "__len__") and idx_int is not None:
                    out["num_train_pairs"] = _to_python_int(num_train_pairs[idx_int])
                else:
                    out["num_train_pairs"] = _to_python_int(num_train_pairs)
            except Exception:
                out["num_train_pairs"] = _to_python_int(num_train_pairs)

        if num_test_pairs is not None:
            try:
                if hasattr(num_test_pairs, "__len__") and idx_int is not None:
                    out["num_test_pairs"] = _to_python_int(num_test_pairs[idx_int])
                else:
                    out["num_test_pairs"] = _to_python_int(num_test_pairs)
            except Exception:
                out["num_test_pairs"] = _to_python_int(num_test_pairs)

        # Resolve human-readable task id using task_index field or provided task_idx
        resolved_task_index = None
        if task_index is not None:
            try:
                if hasattr(task_index, "__len__") and idx_int is not None:
                    resolved_task_index = _to_python_int(task_index[idx_int])
                else:
                    resolved_task_index = _to_python_int(task_index)
            except Exception:
                resolved_task_index = idx_int

        if resolved_task_index is None:
            resolved_task_index = idx_int

        # Map numeric index to string id via global manager helper (best-effort)
        try:
            if resolved_task_index is not None:
                tid = get_task_id_globally(resolved_task_index)
                out["task_id"] = tid
            else:
                out["task_id"] = None
        except Exception:
            out["task_id"] = None

        # ----- Attempt to find a parsed task object or minimal task description -----
        parsed_task = None
        try:
            # Common candidate attribute names where parsers might store parsed tasks
            candidate_attrs = [
                "parsed_tasks",
                "tasks",
                "task_objects",
                "task_list",
                "examples",
                "examples_list",
            ]
            for attr in candidate_attrs:
                candidate = getattr(buf, attr, None)
                if candidate is None:
                    continue
                # If it's indexable, try to pull the single parsed task
                try:
                    if idx_int is not None and hasattr(candidate, "__len__"):
                        parsed_task = candidate[idx_int]
                    else:
                        parsed_task = candidate
                except Exception:
                    parsed_task = None
                if parsed_task is not None:
                    break

            # If we didn't find a parsed task above, try to build a compact
            # minimal description from available example fields (common names).
            if parsed_task is None:
                # Try canonical example fields
                input_examples = getattr(buf, "input_grids_examples", None)
                output_examples = getattr(buf, "output_grids_examples", None)

                if input_examples is not None or output_examples is not None:
                    mini = {"input_grids_examples": None, "output_grids_examples": None, "num_train_pairs": None, "num_test_pairs": None}
                    try:
                        if input_examples is not None:
                            if idx_int is not None and hasattr(input_examples, "__len__"):
                                mini["input_grids_examples"] = input_examples[idx_int]
                            else:
                                mini["input_grids_examples"] = input_examples
                    except Exception:
                        mini["input_grids_examples"] = None
                    try:
                        if output_examples is not None:
                            if idx_int is not None and hasattr(output_examples, "__len__"):
                                mini["output_grids_examples"] = output_examples[idx_int]
                            else:
                                mini["output_grids_examples"] = output_examples
                    except Exception:
                        mini["output_grids_examples"] = None

                    mini["num_train_pairs"] = out.get("num_train_pairs")
                    mini["num_test_pairs"] = out.get("num_test_pairs")
                    parsed_task = mini

            # If still nothing, leave parsed_task as None
            if parsed_task is not None:
                out["task_object"] = parsed_task

                # Try to compute a very small task_structure summary (shapes + max color)
                try:
                    shapes_in = []
                    shapes_out = []
                    max_color = None

                    def _shape_of_grid(g):
                        try:
                            arr = np.asarray(g)
                            return list(arr.shape)
                        except Exception:
                            return None

                    def _max_color_in_grid(g):
                        try:
                            arr = np.asarray(g)
                            if arr.size == 0:
                                return None
                            return int(arr.max())
                        except Exception:
                            return None

                    # Extract shapes from parsed_task fields where possible
                    pin = parsed_task.get("input_grids_examples") if isinstance(parsed_task, dict) else getattr(parsed_task, "input_grids_examples", None)
                    pout = parsed_task.get("output_grids_examples") if isinstance(parsed_task, dict) else getattr(parsed_task, "output_grids_examples", None)

                    if pin is not None:
                        try:
                            for i, g in enumerate(pin):
                                s = _shape_of_grid(g)
                                if s is not None:
                                    shapes_in.append(s)
                                # update max_color from a couple of examples only to keep cheap
                                if max_color is None:
                                    mc = _max_color_in_grid(g)
                                    if mc is not None:
                                        max_color = mc
                        except Exception:
                            pass

                    if pout is not None:
                        try:
                            for i, g in enumerate(pout):
                                s = _shape_of_grid(g)
                                if s is not None:
                                    shapes_out.append(s)
                                if max_color is None:
                                    mc = _max_color_in_grid(g)
                                    if mc is not None:
                                        max_color = mc
                        except Exception:
                            pass

                    # Fallback: try to glean shapes directly from single canonical buffers
                    if not shapes_in and hasattr(buf, "input_grid_shape"):
                        try:
                            shapes_in = [list(getattr(buf, "input_grid_shape"))]
                        except Exception:
                            pass

                    out["task_structure"] = {
                        "input_grid_shapes": shapes_in if shapes_in else None,
                        "output_grid_shapes": shapes_out if shapes_out else None,
                        "max_colors_used": int(max_color) if max_color is not None else None,
                    }
                except Exception:
                    out["task_structure"] = None

        except Exception:
            # Don't fail the overall metadata extraction if task object probing fails
            out["task_object"] = out.get("task_object", None)
            out["task_structure"] = out.get("task_structure", None)

    except Exception as e:  # pragma: no cover - defensive host code
        logger.debug(f"build_task_metadata_from_params failed to extract metadata: {e}")

    return out


def build_step_logging_payload(
    *,
    before_state: Optional[State] = None,
    after_state: Optional[State] = None,
    action: Any = None,
    reward: Any = None,
    info: Any = None,
    step_num: Optional[int] = None,
    episode_num: Optional[int] = None,
    params: Optional[EnvParams] = None,
    include_task_meta: bool = False,
    include_grids: bool = False,
) -> Dict[str, Any]:
    """
    Construct a logging payload for a single environment step.

    This helper gathers commonly used fields into a single dict suitable for
    passing to ExperimentLogger.log_step or other handlers.

    Important: This function runs on the host only. It does NOT attempt to
    access JAX-traced structures inside JIT; callers must call it outside JAX.

    Args:
        before_state: State before step (optional)
        after_state: State after step (optional)
        action: Action object (structured or dict)
        reward: Reward scalar
        info: Additional info (dict or StepInfo-like object)
        step_num: Step number within episode (optional)
        episode_num: Episode number (optional)
        params: EnvParams (optional). If provided and include_task_meta=True, the
                helper will attempt to extract task-level metadata (counts and id).
        include_task_meta: If True, attempt to include task-level metadata (best-effort)
        include_grids: If True, attach small copies of input/target/working grids
                       for visualization. Caller should prefer passing minimal data.

    Returns:
        Dictionary with normalized logging fields.
    """
    payload: Dict[str, Any] = {}

    # Core timing/ids
    if step_num is not None:
        payload["step_num"] = int(step_num)
    if episode_num is not None:
        payload["episode_num"] = int(episode_num)

    # States and action/reward/info
    payload["before_state"] = before_state
    payload["after_state"] = after_state
    payload["action"] = action
    payload["reward"] = float(reward) if (reward is not None and not isinstance(reward, dict)) else reward

    # Normalize `info` into a plain dict and best-effort populate common
    # metrics so visualization handlers can rely on a consistent shape.
    info_dict: Dict[str, Any] = {}
    if info is None:
        info_dict = {}
    elif isinstance(info, dict):
        # shallow copy to avoid mutating caller's dict
        info_dict = dict(info)
    else:
        # object-like info (e.g., StepInfo). Extract known attributes.
        try:
            metrics_obj = getattr(info, "metrics", None)
            if metrics_obj is not None:
                if isinstance(metrics_obj, dict):
                    info_dict["metrics"] = dict(metrics_obj)
                else:
                    # try to pick a few fields off the metrics object
                    m = {}
                    for k in ("similarity", "similarity_improvement"):
                        v = getattr(metrics_obj, k, None)
                        if v is not None:
                            m[k] = v
                    if m:
                        info_dict["metrics"] = m
            # Copy direct attributes where present
            for k in ("similarity", "similarity_improvement", "step_count", "current_pair_index"):
                v = getattr(info, k, None)
                if v is not None:
                    info_dict[k] = v
        except Exception:
            info_dict = {}

    # Best-effort enrichment from before/after states when keys are missing.
    try:
        # Ensure metrics dict exists
        info_metrics = info_dict.get("metrics")
        if info_metrics is None:
            info_metrics = {}
            info_dict["metrics"] = info_metrics

        # similarity: prefer metrics, then direct info keys, then state attributes
        if info_metrics.get("similarity") is None and info_dict.get("similarity") is None:
            sim_val = None
            if after_state is not None and hasattr(after_state, "similarity_score"):
                sim_val = getattr(after_state, "similarity_score")
            elif before_state is not None and hasattr(before_state, "similarity_score"):
                sim_val = getattr(before_state, "similarity_score")
            if sim_val is not None:
                try:
                    info_metrics["similarity"] = float(np.asarray(sim_val).item())
                except Exception:
                    try:
                        info_metrics["similarity"] = float(sim_val)
                    except Exception:
                        info_metrics["similarity"] = None

        # similarity_improvement: compute if possible
        if info_metrics.get("similarity_improvement") is None and info_dict.get("similarity_improvement") is None:
            try:
                a = getattr(after_state, "similarity_score", None) if after_state is not None else None
                b = getattr(before_state, "similarity_score", None) if before_state is not None else None
                if a is not None and b is not None:
                    aval = float(np.asarray(a).item()) if hasattr(a, "item") else float(a)
                    bval = float(np.asarray(b).item()) if hasattr(b, "item") else float(b)
                    info_metrics["similarity_improvement"] = aval - bval
            except Exception:
                pass

        # step_count: prefer info.step_count, then after_state.step_count, then before_state.step_count
        if info_dict.get("step_count") is None:
            sc = None
            if hasattr(after_state, "step_count") and after_state is not None:
                sc = getattr(after_state, "step_count")
            elif hasattr(before_state, "step_count") and before_state is not None:
                sc = getattr(before_state, "step_count")
            if sc is not None:
                try:
                    info_dict["step_count"] = int(np.asarray(sc).item()) if hasattr(sc, "item") else int(sc)
                except Exception:
                    info_dict["step_count"] = None

        # current_pair_index: prefer info-provided keys, then state.pair_idx
        if info_dict.get("current_pair_index") is None:
            # Accept multiple aliases supplied by callers
            candidate = None
            for alias in ("current_pair_index", "pair_idx", "task_pair_index"):
                if alias in info_dict and info_dict[alias] is not None:
                    candidate = info_dict[alias]
                    break
            if candidate is None:
                if hasattr(after_state, "pair_idx") and after_state is not None:
                    candidate = getattr(after_state, "pair_idx")
                elif hasattr(before_state, "pair_idx") and before_state is not None:
                    candidate = getattr(before_state, "pair_idx")
            if candidate is not None:
                try:
                    info_dict["current_pair_index"] = int(np.asarray(candidate).item()) if hasattr(candidate, "item") else int(candidate)
                except Exception:
                    info_dict["current_pair_index"] = None
    except Exception:
        # never let enrichment raise
        pass

    payload["info"] = info_dict
    # Include params in the payload so downstream handlers (SVG, rich, etc.)
    # can do best-effort lookups from the buffer if explicit counts are missing.
    payload["params"] = params

    # Best-effort extraction of small scalar indices from states (task_idx / pair_idx)
    try:
        if before_state is not None and hasattr(before_state, "task_idx"):
            payload["task_idx"] = _to_python_int(getattr(before_state, "task_idx"))
        elif after_state is not None and hasattr(after_state, "task_idx"):
            payload["task_idx"] = _to_python_int(getattr(after_state, "task_idx"))
    except Exception:
        payload["task_idx"] = None

    try:
        if before_state is not None and hasattr(before_state, "pair_idx"):
            payload["task_pair_index"] = _to_python_int(getattr(before_state, "pair_idx"))
        elif after_state is not None and hasattr(after_state, "pair_idx"):
            payload["task_pair_index"] = _to_python_int(getattr(after_state, "pair_idx"))
    except Exception:
        payload["task_pair_index"] = None

    # Human-readable id resolution (best-effort). Try several sources in priority:
    #  1) extract_task_id_from_index (works for JAX array indices)
    #  2) params.buffer via build_task_metadata_from_params (best-effort host-side extraction)
    #  3) numeric fallback label "task_{idx}" only if nothing else available
    try:
        if "task_idx" in payload and payload["task_idx"] is not None:
            task_id = None
            try:
                # Prefer the array-aware extractor which may return the original string id.
                task_id = extract_task_id_from_index(payload["task_idx"])
            except Exception:
                task_id = None

            # If extractor did not return a string id, try to extract from params.buffer
            # (best-effort, host-side). Do not mutate params.
            if task_id is None:
                try:
                    if params is not None:
                        meta = build_task_metadata_from_params(params, payload["task_idx"])
                        tid = meta.get("task_id")
                        if tid:
                            task_id = tid
                except Exception:
                    task_id = None

            # Final fallback: fabricate numeric label only if nothing better exists
            if task_id is not None:
                payload["task_id"] = task_id
            else:
                try:
                    payload["task_id"] = f"task_{int(payload['task_idx'])}"
                except Exception:
                    payload["task_id"] = None
        else:
            payload["task_id"] = None
    except Exception:
        payload["task_id"] = None

    # Optionally include task-level metadata from params if requested.
    # NOTE: this remains best-effort and will not mutate params or copy large buffers.
    if include_task_meta and params is not None and "task_idx" in payload and payload["task_idx"] is not None:
        try:
            task_meta = build_task_metadata_from_params(params, payload["task_idx"])
            payload["total_task_pairs"] = {
                "train": task_meta.get("num_train_pairs"),
                "test": task_meta.get("num_test_pairs"),
            }
            payload["task_id"] = payload.get("task_id") or task_meta.get("task_id")
        except Exception:
            payload["total_task_pairs"] = None

    # Always attempt to compute a single scalar total for pair-count display when
    # params and task_idx are available. This makes it easy for handlers to show
    # "Pair X/Y" without parsing container structures.
    if params is not None and "task_idx" in payload and payload["task_idx"] is not None:
        try:
            # Reuse task_meta if already computed above; otherwise, compute it.
            try:
                task_meta  # type: ignore[name-defined]
            except Exception:
                task_meta = build_task_metadata_from_params(params, payload["task_idx"])

            # Determine episode mode (0=train, 1=test) robustly, handling numpy/jax scalars.
            ep_mode = getattr(params, "episode_mode", None)
            try:
                if ep_mode is None:
                    ep_mode_val = 0
                else:
                    ep_mode_val = int(ep_mode.item()) if hasattr(ep_mode, "item") else int(ep_mode)
            except Exception:
                try:
                    ep_mode_val = int(ep_mode)
                except Exception:
                    ep_mode_val = 0

            # Select appropriate total based on episode mode, with sensible fallbacks.
            selected = None
            if ep_mode_val == 1:
                selected = task_meta.get("num_test_pairs")
            else:
                selected = task_meta.get("num_train_pairs")

            # If selected is missing, pick any available count from task_meta.
            if selected is None:
                for v in (task_meta.get("num_train_pairs"), task_meta.get("num_test_pairs")):
                    if v is not None:
                        selected = v
                        break

            payload["total_task_pairs_selected"] = _to_python_int(selected)
        except Exception:
            payload["total_task_pairs_selected"] = None

    # Optionally include grids (caller opted in). These can be large; keep them as-is.
    # We make copies for safety (numpy conversion) where possible.
    if include_grids:
        try:
            def _safe_copy_grid(arr):
                if arr is None:
                    return None
                try:
                    return np.asarray(arr).copy()
                except Exception:
                    return arr

            if before_state is not None:
                payload.setdefault("grids", {})["before_working_grid"] = _safe_copy_grid(getattr(before_state, "working_grid", None))
            if after_state is not None:
                payload.setdefault("grids", {})["after_working_grid"] = _safe_copy_grid(getattr(after_state, "working_grid", None))
            # Also include input/target if states expose them
            if before_state is not None and hasattr(before_state, "input_grid"):
                payload.setdefault("grids", {})["input_grid"] = _safe_copy_grid(getattr(before_state, "input_grid"))
            if before_state is not None and hasattr(before_state, "target_grid"):
                payload.setdefault("grids", {})["target_grid"] = _safe_copy_grid(getattr(before_state, "target_grid"))
        except Exception:
            # never fail the caller due to logging enrichment
            logger.debug("Failed to attach grids to logging payload")

    return payload


# Convenience helper: build an episode summary payload suitable for handlers
def build_episode_summary_payload(
    *,
    episode_num: Any = None,
    step_data: Optional[list[dict[str, Any]]] = None,
    total_steps: Optional[int] = None,
    total_reward: Optional[float] = None,
    final_similarity: Optional[float] = None,
    success: Optional[bool] = None,
    task_id: Optional[str] = None,
    params: Optional[EnvParams] = None,
    include_serialized_steps: bool = True,
) -> dict[str, Any]:
    """
    Assemble an episode summary payload that is compatible with the ExperimentLogger
    handlers (FileHandler, SVGHandler, WandbHandler, RichHandler).

    This utility is intentionally defensive and best-effort:
      - Coerces scalar-like values to native Python scalars where possible.
      - Attempts to serialize provided step_data using the project's
        serialization utility when `include_serialized_steps` is True.
      - Attempts to resolve a task_id from provided values, step payloads,
        or params via `build_task_metadata_from_params`.

    Args:
        episode_num: Episode number (scalar-like)
        step_data: List of step payload dicts (as produced by build_step_logging_payload)
        total_steps: Optional override for total number of steps
        total_reward: Optional total reward
        final_similarity: Final similarity metric (optional)
        success: Whether the episode succeeded (optional)
        task_id: Optional task id override
        params: Optional EnvParams (host-side) to extract task-level metadata
        include_serialized_steps: If True, attempt to serialize each step using
            `serialize_log_step` from serialization utilities (best-effort).

    Returns:
        Dict suitable for passing to ExperimentLogger.log_episode_summary or
        handlers that expect episode-level payloads.
    """
    payload: dict[str, Any] = {}

    # Normalize simple scalars using helpers
    try:
        if episode_num is not None:
            payload["episode_num"] = to_python_int(episode_num)
        if total_steps is not None:
            payload["total_steps"] = to_python_int(total_steps)
        if total_reward is not None:
            payload["total_reward"] = to_python_float(total_reward)
        if final_similarity is not None:
            # Accept scalar-like values
            try:
                payload["final_similarity"] = float(to_python_scalar(final_similarity))
            except Exception:
                payload["final_similarity"] = to_python_float(final_similarity)
        if success is not None:
            # Coerce bool-like values to plain bool where possible
            try:
                if isinstance(success, (bool, int)):
                    payload["success"] = bool(success)
                else:
                    v = to_python_scalar(success)
                    payload["success"] = bool(v) if v is not None else None
            except Exception:
                payload["success"] = None
    except Exception:
        # Never raise from enrichment helpers
        pass

    # Attach explicit task_id if provided
    if task_id is not None:
        payload["task_id"] = task_id

    # If step_data present, attempt to derive a couple of common fields and optionally serialize steps
    serialized_steps: Optional[list[dict[str, Any]]] = None
    if step_data:
        # Best-effort: derive total_steps if missing
        try:
            if payload.get("total_steps") is None:
                payload["total_steps"] = int(len(step_data))
        except Exception:
            pass

        # Attempt to resolve task_id from first step if missing
        if payload.get("task_id") is None:
            try:
                first = step_data[0] if isinstance(step_data, (list, tuple)) and len(step_data) > 0 else None
                if first is not None:
                    # Common keys: task_id or task_idx -> use extract helper if numeric index present
                    if isinstance(first, dict) and "task_id" in first and first.get("task_id") is not None:
                        payload["task_id"] = first.get("task_id")
                    elif isinstance(first, dict) and "task_idx" in first and first.get("task_idx") is not None:
                        try:
                            meta = build_task_metadata_from_params(params, first.get("task_idx")) if params is not None else {}
                            tid = meta.get("task_id") if isinstance(meta, dict) else None
                            payload["task_id"] = tid or f"task_{to_python_int(first.get('task_idx'))}"
                        except Exception:
                            # final fallback: try numeric formatting
                            try:
                                payload["task_id"] = f"task_{int(first.get('task_idx'))}"
                            except Exception:
                                payload["task_id"] = None
            except Exception:
                payload["task_id"] = payload.get("task_id", None)

        # Optionally serialize step payloads for handlers that need them
        if include_serialized_steps:
            try:
                # Lazy import to avoid top-level cycles; if unavailable we fall back to safe serialization
                try:
                    from jaxarc.utils.serialization_utils import serialize_log_step  # type: ignore
                except Exception:
                    serialize_log_step = None  # type: ignore

                serialized_steps = []
                for s in step_data:
                    if serialize_log_step is not None:
                        try:
                            serialized_steps.append(serialize_log_step(s))
                            continue
                        except Exception:
                            # fallback to generic object serialization
                            pass
                    # Fallback: shallow safe-serialize known fields
                    try:
                        entry = {}
                        entry["step_num"] = to_python_int(s.get("step_num")) if isinstance(s, dict) else None
                        entry["reward"] = to_python_scalar(s.get("reward")) if isinstance(s, dict) else None
                        entry["task_id"] = s.get("task_id") if isinstance(s, dict) and "task_id" in s else payload.get("task_id")
                        # Keep before/after state out of serialized steps unless already serialized by caller
                        # Provide a minimal compatibility wrapper for visualization functions expecting serialized steps
                        if isinstance(s, dict) and "after_state" in s:
                            try:
                                # Prefer already-serialized structure if present
                                from jaxarc.utils.serialization_utils import serialize_log_state  # type: ignore
                                entry["after_state"] = serialize_log_state(s.get("after_state"))
                            except Exception:
                                entry["after_state"] = str(s.get("after_state"))
                        serialized_steps.append(entry)
                    except Exception:
                        serialized_steps.append({"raw": str(s)})
            except Exception:
                serialized_steps = None

    # Attach step_data (prefer serialized when available)
    if serialized_steps is not None:
        payload["step_data"] = serialized_steps
    elif step_data is not None:
        # As a last resort attach original step_data (caller likely already serialized)
        payload["step_data"] = step_data

    # Compute reward and similarity progressions for visualization handlers.
    # We attempt to extract rewards and similarity values from serialized steps
    # (preferred) and fall back to raw step entries when necessary.
    reward_progression: list[float] = []
    similarity_progression: list[Optional[float]] = []
    try:
        sd = payload.get("step_data", []) or []
        for entry in sd:
            # Normalize entry to dict-like access where possible
            try:
                # Attempt to get reward
                r_val = None
                if isinstance(entry, dict):
                    if "reward" in entry:
                        r_val = to_python_float(entry.get("reward"))
                    else:
                        # Some serialized formats nest metrics under info.metrics
                        info = entry.get("info")
                        if isinstance(info, dict):
                            metrics = info.get("metrics")
                            if isinstance(metrics, dict) and "reward" in metrics:
                                r_val = to_python_float(metrics.get("reward"))
                # Fallback to numeric coercion on the entire entry
                if r_val is None:
                    # Last-resort default to 0.0
                    r_val = 0.0
            except Exception:
                r_val = 0.0
            try:
                reward_progression.append(float(r_val) if r_val is not None else 0.0)
            except Exception:
                reward_progression.append(0.0)

            # Attempt to get similarity
            s_val = None
            try:
                if isinstance(entry, dict):
                    info = entry.get("info")
                    if isinstance(info, dict):
                        metrics = info.get("metrics")
                        if isinstance(metrics, dict) and "similarity" in metrics:
                            s_val = to_python_float(metrics.get("similarity"))
                    # Some serialized step formats include an `after_state` dict with `similarity_score`
                    after = entry.get("after_state")
                    if s_val is None and isinstance(after, dict):
                        s_val = to_python_float(after.get("similarity_score"))
                    # Also accept a top-level 'similarity' key if present
                    if s_val is None and "similarity" in entry:
                        s_val = to_python_float(entry.get("similarity"))
                # If entry is not a dict, give up gracefully
            except Exception:
                s_val = None

            # Keep raw similarity (may be None) for downstream logic before coercion
            similarity_progression.append(s_val if s_val is not None else None)
    except Exception:
        # Never fail summary construction due to progression extraction
        reward_progression = []
        similarity_progression = []

    # Coerce similarity progression to numeric floats: fill None with last known or 0.0
    similarity_numeric: list[float] = []
    last_val = 0.0
    try:
        for v in similarity_progression:
            if v is None:
                similarity_numeric.append(float(last_val))
            else:
                try:
                    fv = float(v)
                    similarity_numeric.append(fv)
                    last_val = fv
                except Exception:
                    similarity_numeric.append(float(last_val))
    except Exception:
        similarity_numeric = []

    # Detect key moments: indices where reward delta magnitude is large (best-effort)
    key_moments: list[int] = []
    try:
        if len(reward_progression) >= 2:
            deltas = [abs(reward_progression[i] - reward_progression[i - 1]) for i in range(1, len(reward_progression))]
            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                threshold = max(0.0, avg_delta * 1.5)
                for i, d in enumerate(deltas, start=1):
                    if d >= threshold:
                        key_moments.append(i)
    except Exception:
        key_moments = []

    # Ensure payload contains the progressions and key_moments for visualizers
    # Pad reward and similarity arrays to at least length 2 so charts render correctly
    try:
        if not reward_progression:
            reward_progression = [0.0, 0.0]
        elif len(reward_progression) == 1:
            reward_progression = [float(reward_progression[0]), float(reward_progression[0])]
    except Exception:
        reward_progression = [0.0, 0.0]

    try:
        if not similarity_numeric:
            similarity_numeric = [0.0, 0.0]
        elif len(similarity_numeric) == 1:
            similarity_numeric = [float(similarity_numeric[0]), float(similarity_numeric[0])]
    except Exception:
        similarity_numeric = [0.0, 0.0]

    payload["reward_progression"] = reward_progression
    payload["similarity_progression"] = similarity_numeric
    payload["key_moments"] = key_moments

    # Ensure final_similarity exists for visualizers: prefer existing value, otherwise use last similarity
    try:
        if payload.get("final_similarity") is None:
            payload["final_similarity"] = float(similarity_numeric[-1]) if similarity_numeric else 0.0
    except Exception:
        payload["final_similarity"] = 0.0

    # If still missing total_steps, try to infer from attached step_data
    try:
        if payload.get("total_steps") is None and "step_data" in payload and payload["step_data"] is not None:
            payload["total_steps"] = int(len(payload["step_data"]))
    except Exception:
        payload["total_steps"] = payload.get("total_steps", None)

    # Provide an opportunity for handlers to access the original params for further best-effort extraction
    if params is not None:
        payload["params"] = params

    return payload


# Expose a small API tuple for convenience
__all__ = [
    "build_step_logging_payload",
    "build_task_metadata_from_params",
    "build_episode_summary_payload",
    # exported scalar coercion helpers for reuse across logging handlers
    "to_python_int",
    "to_python_float",
    "to_python_scalar",
]
