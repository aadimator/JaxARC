"""
Core logging functionality for JaxARC experiments.

Provides main logger class and utilities.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Union

import jax
import numpy as np
from loguru import logger

try:
    from jaxarc.configs import JaxArcConfig
    from jaxarc.state import State
    from jaxarc.types import EnvParams, JaxArcTask
    from jaxarc.utils.task_manager import (
        extract_task_id_from_index,
        get_task_id_globally,
    )
except ImportError:
    # Handle missing imports gracefully
    State = Any
    EnvParams = Any
    JaxArcTask = Any
    JaxArcConfig = Any
    extract_task_id_from_index = lambda x: f"task_{x}"
    get_task_id_globally = lambda x: f"task_{x}"


# ============================================================================
# SECTION: Logging Utilities (from logging_utils.py)
# ============================================================================


def to_python_int(x: Any) -> Optional[int]:
    """Convert scalar-like value to Python int or None."""
    if x is None:
        return None
    try:
        return int(np.asarray(x).item())
    except Exception:
        try:
            return int(x)
        except Exception:
            return None


def to_python_float(x: Any) -> Optional[float]:
    """Convert scalar-like value to Python float or None."""
    if x is None:
        return None
    try:
        return float(np.asarray(x).item())
    except Exception:
        try:
            return float(x)
        except Exception:
            return None


def to_python_scalar(x: Any) -> Any:
    """Convert JAX/numpy scalar to Python scalar, return as-is if conversion fails."""
    if x is None or isinstance(x, (int, float, bool, str)):
        return x

    try:
        arr = np.asarray(x)
        if arr.shape == ():
            return arr.item()
    except Exception:
        pass

    return x


def create_start_log(
    params: EnvParams,
    task_idx: Union[int, Any] = None,
    state: Optional[State] = None,
    episode_num: int = 0,
) -> Dict[str, Any]:
    """Extract task data for logging.

    Args:
        params: Environment parameters with task buffer
        task_idx: Task index (required if state not provided)
        state: Environment state containing task_idx
        episode_num: Episode number

    Returns:
        Dict with task data for logging
    """
    # Get task index
    if task_idx is None and state is not None:
        task_idx = state.task_idx
    if task_idx is None:
        raise ValueError("Either task_idx or state must be provided")

    idx = to_python_int(task_idx)
    if params is None or params.buffer is None:
        raise ValueError("EnvParams must contain a valid buffer")

    # Extract task from buffer
    try:
        single = jax.tree_util.tree_map(lambda x: x[idx], params.buffer)
        task_object = JaxArcTask(
            input_grids_examples=single.input_grids_examples,
            input_masks_examples=single.input_masks_examples,
            output_grids_examples=single.output_grids_examples,
            output_masks_examples=single.output_masks_examples,
            num_train_pairs=single.num_train_pairs,
            test_input_grids=single.test_input_grids,
            test_input_masks=single.test_input_masks,
            true_test_output_grids=single.true_test_output_grids,
            true_test_output_masks=single.true_test_output_masks,
            num_test_pairs=single.num_test_pairs,
            task_index=single.task_index,
        )

        # Get task ID
        try:
            task_id = task_object.get_task_id()
        except Exception:
            task_id = extract_task_id_from_index(idx)

    except Exception as e:
        logger.error(f"Failed to extract task data: {e}")
        raise ValueError(f"Could not extract task at index {idx}") from e

    return {
        "task_object": task_object,
        "task_idx": idx,
        "task_id": task_id,
        "num_train_pairs": to_python_int(single.num_train_pairs),
        "num_test_pairs": to_python_int(single.num_test_pairs),
        "episode_num": episode_num,
        "show_test": True,
    }


def create_step_log(
    timestep,
    state,
    action,
    step_num: int,
    episode_num: int,
    prev_state=None,
    env_params=None,
) -> Dict[str, Any]:
    """Create step logging payload.

    Args:
        timestep: TimeStep from environment (no longer contains state)
        state: Current state from environment step
        action: Action taken
        step_num: Step number
        episode_num: Episode number
        prev_state: Previous state (optional)
        env_params: Environment parameters (optional)

    Returns:
        Dict for step logging
    """
    # Prefer a canonical mask-based action provided by wrappers via timestep.extras
    canonical_action = None
    try:
        if hasattr(timestep, "extras") and isinstance(timestep.extras, dict):
            ca = timestep.extras.get("canonical_action")
            if isinstance(ca, dict) and "operation" in ca and "selection" in ca:
                canonical_action = ca
    except (AttributeError, KeyError, TypeError):
        canonical_action = None

    log_data = {
        "step_num": step_num,
        "episode_num": episode_num,
        # Use canonical mask action when available so visualization layers receive selection directly
        "action": canonical_action if canonical_action is not None else action,
        "reward": to_python_float(timestep.reward),
        "before_state": prev_state,
        "after_state": state,
        "params": env_params,
    }

    # Extract info from timestep extras if available
    info = {}
    if (
        hasattr(timestep, "extras")
        and timestep.extras is not None
        and isinstance(timestep.extras, dict)
    ):
        info.update(timestep.extras)

    # Add similarity metrics from current state
    if state and hasattr(state, "similarity_score"):
        similarity = to_python_float(state.similarity_score)
        if similarity is not None:
            info["similarity"] = similarity

    # Calculate similarity improvement
    if (
        prev_state
        and hasattr(prev_state, "similarity_score")
        and state
        and hasattr(state, "similarity_score")
    ):
        prev_sim = to_python_float(prev_state.similarity_score) or 0.0
        curr_sim = to_python_float(state.similarity_score) or 0.0
        info["similarity_improvement"] = curr_sim - prev_sim

    log_data["info"] = info

    # Extract state metadata from current state
    if state:
        if hasattr(state, "task_idx"):
            task_idx = to_python_int(state.task_idx)
            log_data["task_idx"] = task_idx
            try:
                log_data["task_id"] = extract_task_id_from_index(task_idx)
            except Exception:
                log_data["task_id"] = f"task_{task_idx}"

        if hasattr(state, "pair_idx"):
            log_data["task_pair_index"] = to_python_int(state.pair_idx)

        if hasattr(state, "step_count"):
            log_data["step_count"] = to_python_int(state.step_count)

    return log_data


def create_episode_summary(
    episode_num: int,
    step_logs: list[dict],
    env_params=None,
) -> Dict[str, Any]:
    """Create episode summary from step logs.

    Args:
        episode_num: Episode number
        step_logs: List of step log dicts
        env_params: Environment parameters (optional)

    Returns:
        Dict for episode summary logging
    """
    summary = {
        "episode_num": episode_num,
        "total_steps": len(step_logs),
        "step_data": step_logs,
        "params": env_params,
    }

    if not step_logs:
        summary.update(
            {
                "total_reward": 0.0,
                "final_similarity": 0.0,
                "reward_progression": [],
                "similarity_progression": [],
            }
        )
        return summary

    # Extract metrics from step logs
    rewards = []
    similarities = []

    for step in step_logs:
        reward = to_python_float(step.get("reward", 0.0)) or 0.0
        rewards.append(reward)

        # Get similarity from info or state
        similarity = 0.0
        info = step.get("info", {})
        if isinstance(info, dict) and "similarity" in info:
            similarity = to_python_float(info["similarity"]) or 0.0

        similarities.append(similarity)

    summary.update(
        {
            "total_reward": sum(rewards),
            "final_similarity": similarities[-1] if similarities else 0.0,
            "reward_progression": rewards,
            "similarity_progression": similarities,
        }
    )

    # Add task ID from first step
    if step_logs and "task_id" in step_logs[0]:
        summary["task_id"] = step_logs[0]["task_id"]

    return summary


# ============================================================================
# SECTION: Experiment Logger (from experiment_logger.py)
# ============================================================================


class ExperimentLogger:
    """Central logging coordinator with handler-based architecture.

    This class serves as the single entry point for all logging operations
    in JaxARC. It manages a set of handlers for different logging concerns
    and provides error isolation to ensure that failures in one handler
    don't affect others.

    Note: This is a regular Python class (not equinox.Module) because it needs
    mutable state to manage handlers and doesn't need to be JAX-compatible.

    Attributes:
        config: JaxARC configuration object
        handlers: Dictionary of active handler instances
    """

    def __init__(self, config: JaxArcConfig):
        """Initialize logger with handlers based on configuration.

        Args:
            config: JaxARC configuration object containing logging settings
        """
        self.config = config
        self.handlers = self._initialize_handlers()
        self._episode_counter = 0  # Sequential episode counter for batched logging
        # Preferred episode number (filled by log_task_start or left None). When set,
        # steps missing an explicit episode_num will be logged into this episode.
        self._preferred_episode_num: int | None = None
        # Pending task start payload saved when caller asks to log task start but
        # doesn't yet provide an episode_num. This will be flushed when the first
        # step with an episode_num arrives.
        self._pending_task_start: dict[str, Any] | None = None

        logger.info(f"ExperimentLogger initialized with {len(self.handlers)} handlers")

    def _initialize_handlers(self) -> dict[str, Any]:
        """Initialize handlers based on configuration settings.

        This method creates handler instances based on the configuration,
        with graceful fallback if handler initialization fails.

        Returns:
            Dictionary mapping handler names to handler instances
        """
        handlers: dict[str, Any] = {}

        # Get debug level from environment config
        debug_level = "off"
        if hasattr(self.config, "environment") and hasattr(
            self.config.environment, "debug_level"
        ):
            debug_level = self.config.environment.debug_level

        # Early return if logging is completely disabled
        if debug_level == "off":
            logger.debug("Logging disabled (debug_level='off')")
            return handlers

        try:
            # File logging handler - enabled for minimal and verbose
            if debug_level in ["minimal", "verbose"]:
                handlers["file"] = self._create_file_handler()
                logger.debug("FileHandler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize FileHandler: {e}")

        try:
            # SVG visualization handler - now controlled explicitly by visualization.enabled
            if hasattr(self.config, "visualization") and getattr(
                self.config.visualization, "enabled", False
            ):
                handlers["svg"] = self._create_svg_handler()
                logger.debug("SVGHandler initialized (visualization.enabled=True)")
        except Exception as e:
            logger.warning(f"Failed to initialize SVGHandler: {e}")

        try:
            # Console output handler - enabled for minimal and verbose
            if debug_level in ["minimal", "verbose"]:
                handlers["rich"] = self._create_rich_handler()
                logger.debug("RichHandler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize RichHandler: {e}")

        try:
            # Wandb integration handler - enabled if wandb config exists and is enabled
            # Only initialize if explicitly enabled in config
            if (
                hasattr(self.config, "wandb")
                and hasattr(self.config.wandb, "enabled")
                and self.config.wandb.enabled is True
            ):
                handlers["wandb"] = self._create_wandb_handler()
                logger.debug("WandbHandler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize WandbHandler: {e}")

        return handlers

    def _create_file_handler(self) -> Any:
        """Create FileHandler instance.

        Returns:
            FileHandler instance
        """
        from .handlers import FileHandler

        return FileHandler(self.config)

    def _create_svg_handler(self) -> Any:
        """Create SVGHandler instance.

        Returns:
            SVGHandler instance
        """
        from .handlers import SVGHandler

        return SVGHandler(self.config)

    def _create_rich_handler(self) -> Any:
        """Create RichHandler instance.

        Returns:
            RichHandler instance
        """
        from .handlers import RichHandler

        return RichHandler(self.config)

    def _create_wandb_handler(self) -> Any:
        """Create WandbHandler instance.

        Returns:
            WandbHandler instance
        """
        from .handlers import WandbHandler

        return WandbHandler(self.config.wandb)

    def log_step(self, step_data: dict[str, Any]) -> None:
        """Log step data through all active handlers.

        This method calls the log_step method on all active handlers,
        with error isolation to ensure that failures in one handler
        don't affect others.

        Args:
            step_data: Dictionary containing step information with keys:
                - step_num: Step number within episode
                - before_state: State before action
                - after_state: State after action (passed separately from timestep)
                - action: Action taken
                - reward: Reward received
                - info: Additional information including metrics
        """
        # Best-effort enrichment of step_data with task/pair context to support
        # visualization and logging handlers that expect task metadata.
        before_state = step_data.get("before_state")
        after_state = step_data.get("after_state")

        task_idx = None
        pair_idx = None
        task_id = None
        total_task_pairs = None

        def _as_int(x):
            # Delegate scalar coercion to shared helper that handles numpy/jax scalars.
            # Keeps the original semantics but centralizes coercion logic.
            try:
                return to_python_int(x)
            except Exception:
                return None

        try:
            # Prefer before_state for stable attribution, fall back to after_state
            if before_state is not None and hasattr(before_state, "task_idx"):
                task_idx = before_state.task_idx
            elif after_state is not None and hasattr(after_state, "task_idx"):
                task_idx = after_state.task_idx

            if before_state is not None and hasattr(before_state, "pair_idx"):
                pair_idx = before_state.pair_idx
            elif after_state is not None and hasattr(after_state, "pair_idx"):
                pair_idx = after_state.pair_idx

            # Resolve a human-readable task id if possible.
            # Try the canonical extractor first; if that fails, fall back to the
            # global task manager using a plain Python int index.
            if task_idx is not None:
                try:
                    # Prefer the array-aware extractor which may return the original string id.
                    task_id = extract_task_id_from_index(task_idx)
                except Exception:
                    task_id = None

                # If extractor returned None, try the global manager using a python int index.
                if task_id is None:
                    try:
                        iid = _as_int(task_idx)
                        if iid is not None:
                            task_id = get_task_id_globally(iid)
                        else:
                            task_id = None
                    except Exception:
                        task_id = None

            # Do not attempt to extract params or inspect params.buffer here.
            # The caller must provide any task-level metadata in step_data.
            total_task_pairs = None
        except Exception:
            # Best-effort only; do not allow logging enrichment to raise
            task_idx = task_idx  # no-op to clarify we intentionally ignore errors

        # Build enhanced step_data passed to handlers (preserve original keys)
        enhanced_step_data = dict(step_data)

        # Handle payload task id vs resolved task id with simplified rules:
        # - If the caller provided an explicit task_id that is non-synthetic, prefer it.
        # - If caller provided a synthetic label like "task_<n>", prefer a resolved task_id
        #   from state/manager when available; otherwise keep the synthetic label.
        # - If caller did not provide a task_id, use the resolved task_id when available.
        payload_task_id = step_data.get("task_id")
        resolved_task_id = task_id  # best-effort resolved from state above

        try:
            is_synthetic = False
            if isinstance(payload_task_id, str):
                # Simple synthetic pattern check: task_<digits>
                if re.match(r"^task_\d+$", payload_task_id):
                    is_synthetic = True

            final_task_id = None
            if payload_task_id is not None:
                # If payload provided a non-synthetic id, prefer it
                if isinstance(payload_task_id, str) and not is_synthetic:
                    final_task_id = payload_task_id
                else:
                    # payload is synthetic or not a string; prefer resolved id if we have one
                    final_task_id = resolved_task_id or payload_task_id
            else:
                # No payload task id provided; use resolved id if available, otherwise None
                final_task_id = resolved_task_id

            enhanced_step_data["task_id"] = final_task_id
        except Exception:
            # Be defensive: ensure we always set the key even on error
            enhanced_step_data["task_id"] = step_data.get("task_id")

        try:
            # Accept several possible payload keys for pair index, prefer explicit ones.
            if "task_pair_index" in step_data:
                enhanced_step_data["task_pair_index"] = to_python_int(
                    step_data.get("task_pair_index")
                )
            elif "task_pair_idx" in step_data:
                enhanced_step_data["task_pair_index"] = to_python_int(
                    step_data.get("task_pair_idx")
                )
            else:
                enhanced_step_data["task_pair_index"] = (
                    to_python_int(pair_idx) if pair_idx is not None else None
                )

            # Prefer the pre-selected scalar total when available; otherwise fall back
            # to any legacy container or the computed best-effort value.
            enhanced_step_data["total_task_pairs"] = step_data.get(
                "total_task_pairs", total_task_pairs
            )
            # Also allow an explicit scalar chosen by the payload builder.
            # Coerce if it's a scalar-like value, otherwise leave dicts unchanged.
            raw_selected = step_data.get("total_task_pairs_selected")
            enhanced_step_data["total_task_pairs_selected"] = (
                to_python_int(raw_selected) if raw_selected is not None else None
            )

        except Exception:
            # If enrichment fails, preserve original step_data keys as-is
            pass

        # Normalize provided episode_num when present; otherwise apply preferred episode
        # from a prior `log_task_start` call. This ensures steps and task visuals land
        # in the same episode directory and that episode numbers are host Python ints.
        try:
            # Prefer a previously-declared preferred episode number (from log_task_start)
            # so steps and task visuals are placed consistently in the same episode
            # directory. If no preferred episode is set, fall back to the step-provided
            # episode_num when available.
            pref = getattr(self, "_preferred_episode_num", None)
            if pref is not None:
                enhanced_step_data["episode_num"] = int(pref)
            elif (
                "episode_num" in step_data and step_data.get("episode_num") is not None
            ):
                coerced = to_python_int(step_data.get("episode_num"))
                if coerced is not None:
                    enhanced_step_data["episode_num"] = int(coerced)
        except Exception:
            # Be defensive: leave payload unchanged on error
            pass

        # If we have a pending task_start that we deferred earlier because there was
        # no episode_num, flush it now that we know which episode to place the task overview in.
        try:
            pending = getattr(self, "_pending_task_start", None)
            ep_for_pending = enhanced_step_data.get("episode_num")
            if pending is not None and ep_for_pending is not None:
                try:
                    pending_payload = dict(pending)
                    # Attach resolved episode number
                    pending_payload["episode_num"] = int(ep_for_pending)
                    # Ensure handlers receive any params present on the step payload if not already set
                    if (
                        "params" not in pending_payload
                        and "params" in enhanced_step_data
                    ):
                        pending_payload["params"] = enhanced_step_data.get("params")
                    for handler_name, handler in self.handlers.items():
                        try:
                            if hasattr(handler, "log_task_start"):
                                handler.log_task_start(pending_payload)
                        except Exception as e:
                            logger.warning(
                                f"Handler {handler_name} failed in deferred log_task_start: {e}"
                            )
                finally:
                    # Clear the pending payload after attempting flush (best-effort)
                    self._pending_task_start = None
        except Exception:
            # Never allow pending flush to raise and disrupt step logging
            logger.debug("Deferred task start flush failed; continuing")

        for handler_name, handler in self.handlers.items():
            try:
                if hasattr(handler, "log_step"):
                    # Pass enriched data so visualization handlers can render useful context
                    handler.log_step(enhanced_step_data)
            except Exception as e:
                # Log error but continue with other handlers
                logger.warning(f"Handler {handler_name} failed in log_step: {e}")

    def log_task_start(self, task_data: dict[str, Any], show_test: bool = True) -> None:
        """Log task information when an episode starts.

        This method calls the log_task_start method on all active handlers,
        with error isolation to ensure that failures in one handler
        don't affect others.

        Args:
            task_data: Dictionary containing task information with keys:
                - task_id: Task identifier
                - task_object: The JaxArcTask object
                - episode_num: Episode number
                - num_train_pairs: Number of training pairs
                - num_test_pairs: Number of test pairs
                - task_stats: Additional task statistics
            show_test: Whether to show test examples in visualizations (default: True)
        """
        # Compute episode number preference for task visuals.
        # Historically many callers expect a simple `episode_0000` when they only call
        # log_task_start(metadata) without an explicit episode number. Defaulting to 0
        # in that case produces better, predictable outputs (a single run/episode dir).
        raw_ep = task_data.get("episode_num")
        if raw_ep is not None:
            ep_num = to_python_int(raw_ep)
        else:
            # Default to episode 0 when caller didn't supply an episode number.
            ep_num = 0

        # Remember the preferred episode so subsequent steps that omit episode_num
        # will be logged into the same episode directory.
        try:
            self._preferred_episode_num = to_python_int(ep_num)
        except Exception:
            self._preferred_episode_num = None

        # Add show_test and explicit episode_num (might be None) so handlers behave consistently.
        enhanced_task_data = {
            **task_data,
            "show_test": show_test,
            "episode_num": ep_num,
        }

        for handler_name, handler in self.handlers.items():
            try:
                if hasattr(handler, "log_task_start"):
                    handler.log_task_start(enhanced_task_data)
            except Exception as e:
                # Log error but continue with other handlers
                logger.warning(f"Handler {handler_name} failed in log_task_start: {e}")

    def log_episode_summary(self, summary_data: dict[str, Any]) -> None:
        """Log episode summary through all active handlers.

        This method calls the log_episode_summary method on all active handlers,
        with error isolation to ensure that failures in one handler
        don't affect others.

        Args:
            summary_data: Dictionary containing episode summary with keys:
                - episode_num: Episode number
                - total_steps: Total number of steps
                - total_reward: Total reward accumulated
                - final_similarity: Final similarity score
                - success: Whether episode was successful
                - task_id: Task identifier
        """
        for handler_name, handler in self.handlers.items():
            try:
                if hasattr(handler, "log_episode_summary"):
                    handler.log_episode_summary(summary_data)
            except Exception as e:
                # Log error but continue with other handlers
                logger.warning(
                    f"Handler {handler_name} failed in log_episode_summary: {e}"
                )

    def log_batch_step(self, batch_data: dict[str, Any]) -> None:
        """Log data from a batched training step.

        This method handles batched training data by aggregating metrics and
        sampling episodes for detailed logging. It provides frequency-based
        control for both aggregation and sampling to minimize performance impact.

        Args:
            batch_data: Dictionary containing:
                - update_step: Current training update number
                - episode_returns: Array of episode returns [batch_size]
                - episode_lengths: Array of episode lengths [batch_size]
                - similarity_scores: Array of similarity scores [batch_size]
                - policy_loss: Scalar policy loss
                - value_loss: Scalar value loss
                - gradient_norm: Scalar gradient norm
                - success_mask: Boolean array of episode successes [batch_size]
                - Optional: task_ids, initial_states, final_states for detailed logging
        """
        if not hasattr(self.config, "logging"):
            logger.warning("No logging configuration found, skipping batch logging")
            return

        update_step = batch_data.get("update_step", 0)

        # Log aggregated metrics at specified frequency
        if (
            self.config.logging.batched_logging_enabled
            and update_step % self.config.logging.log_frequency == 0
        ):
            try:
                for handler_name, handler in self.handlers.items():
                    try:
                        if hasattr(handler, "log_aggregated_metrics"):
                            handler.log_aggregated_metrics(batch_data, update_step)
                    except Exception as e:
                        logger.warning(
                            f"Handler {handler_name} failed in log_aggregated_metrics: {e}"
                        )
            except Exception as e:
                logger.warning(f"Failed to aggregate batch metrics: {e}")

    def log_evaluation_summary(self, eval_data: dict[str, Any]) -> None:
        """Log a final evaluation summary through all active handlers.

        This method is intended to be called once at the end of training / evaluation
        for a task (or set of tasks). Handlers may optionally implement
        ``log_evaluation_summary``; handlers lacking the method are skipped
        gracefully.

        Args:
            eval_data: Dictionary containing evaluation summary.
        """
        for handler_name, handler in self.handlers.items():
            try:
                if hasattr(handler, "log_evaluation_summary"):
                    handler.log_evaluation_summary(eval_data)
            except Exception as e:
                logger.warning(
                    f"Handler {handler_name} failed in log_evaluation_summary: {e}"
                )

    def close(self) -> None:
        """Clean shutdown of all handlers.

        This method calls the close method on all active handlers to ensure
        proper cleanup of resources like file handles, network connections, etc.
        """
        for handler_name, handler in self.handlers.items():
            try:
                if hasattr(handler, "close"):
                    handler.close()
                    logger.debug(f"Handler {handler_name} closed successfully")
            except Exception as e:
                logger.warning(f"Handler {handler_name} failed to close: {e}")

        logger.info("ExperimentLogger shutdown complete")
