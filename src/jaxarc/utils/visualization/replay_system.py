"""Episode replay and analysis system.

This module provides functionality to replay episodes from structured logs,
reconstruct states, regenerate visualizations, and perform analysis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chex
from loguru import logger

from ...state import ArcEnvState
from ..logging.structured_logger import (
    EpisodeLogEntry,
    StepLogEntry,
    StructuredLogger,
)
from .rl_visualization import draw_rl_step_svg
from .visualizer import VisualizationConfig


@chex.dataclass
class ReplayConfig:
    """Configuration for episode replay."""

    validate_integrity: bool = True
    regenerate_visualizations: bool = False
    output_dir: str = "outputs/replay"
    max_episodes_to_load: int = 100
    include_step_details: bool = True
    comparison_metrics: List[str] = None

    def __post_init__(self):
        if self.comparison_metrics is None:
            object.__setattr__(
                self, "comparison_metrics", ["reward", "similarity", "steps"]
            )


@chex.dataclass
class ReplayValidationResult:
    """Result of replay validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    episode_num: int
    total_steps: int

    def __post_init__(self):
        if self.errors is None:
            object.__setattr__(self, "errors", [])
        if self.warnings is None:
            object.__setattr__(self, "warnings", [])


class EpisodeReplaySystem:
    """System for replaying and analyzing episodes from structured logs."""

    def __init__(
        self,
        structured_logger: StructuredLogger,
        config: ReplayConfig,
        visualization_config: Optional[VisualizationConfig] = None,
    ):
        """Initialize the replay system.

        Args:
            structured_logger: Logger instance for loading episode data
            config: Configuration for replay behavior
            visualization_config: Configuration for visualization regeneration
        """
        self.structured_logger = structured_logger
        self.config = config
        self.visualization_config = visualization_config or VisualizationConfig()

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache for loaded episodes
        self._episode_cache: Dict[int, EpisodeLogEntry] = {}

        logger.info(
            f"EpisodeReplaySystem initialized with output dir: {self.output_dir}"
        )

    def load_episode(
        self, episode_num: int, use_cache: bool = True
    ) -> Optional[EpisodeLogEntry]:
        """Load episode data from structured logs.

        Args:
            episode_num: Episode number to load
            use_cache: Whether to use cached data if available

        Returns:
            Episode data if found, None otherwise
        """
        if use_cache and episode_num in self._episode_cache:
            return self._episode_cache[episode_num]

        episode = self.structured_logger.load_episode(episode_num)
        if episode is not None and use_cache:
            self._episode_cache[episode_num] = episode

        return episode

    def validate_episode_integrity(
        self, episode: EpisodeLogEntry
    ) -> ReplayValidationResult:
        """Validate episode data integrity for replay.

        Args:
            episode: Episode data to validate

        Returns:
            Validation result with errors and warnings
        """
        errors = []
        warnings = []

        # Basic structure validation
        if episode.total_steps != len(episode.steps):
            errors.append(
                f"Step count mismatch: expected {episode.total_steps}, got {len(episode.steps)}"
            )

        if episode.end_timestamp <= episode.start_timestamp:
            errors.append("Invalid timestamps: end_timestamp <= start_timestamp")

        # Step sequence validation
        expected_step_nums = list(range(len(episode.steps)))
        actual_step_nums = [step.step_num for step in episode.steps]
        if actual_step_nums != expected_step_nums:
            errors.append(
                f"Step sequence invalid: expected {expected_step_nums}, got {actual_step_nums}"
            )

        # Timestamp validation
        prev_timestamp = episode.start_timestamp
        for i, step in enumerate(episode.steps):
            if step.timestamp < prev_timestamp:
                warnings.append(
                    f"Step {i} timestamp {step.timestamp} is before previous {prev_timestamp}"
                )
            prev_timestamp = step.timestamp

        # State data validation
        for i, step in enumerate(episode.steps):
            if not step.before_state or not step.after_state:
                warnings.append(f"Step {i} missing state data")

            if not step.action:
                warnings.append(f"Step {i} missing action data")

        # Reward validation
        calculated_total_reward = sum(step.reward for step in episode.steps)
        if abs(calculated_total_reward - episode.total_reward) > 1e-6:
            warnings.append(
                f"Total reward mismatch: calculated {calculated_total_reward}, "
                f"recorded {episode.total_reward}"
            )

        return ReplayValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            episode_num=episode.episode_num,
            total_steps=episode.total_steps,
        )

    def reconstruct_state_from_log(
        self, step_log: StepLogEntry, use_before_state: bool = True
    ) -> Optional[ArcEnvState]:
        """Reconstruct ArcEnvState from logged step data.

        Args:
            step_log: Step log entry containing state data
            use_before_state: Whether to use before_state (True) or after_state (False)

        Returns:
            Reconstructed state or None if reconstruction fails
        """
        try:
            state_data = (
                step_log.before_state if use_before_state else step_log.after_state
            )

            if not state_data or "type" not in state_data:
                logger.warning(f"Insufficient state data for step {step_log.step_num}")
                return None

            # For minimal state logging, we can't fully reconstruct
            if not self.structured_logger.config.include_full_states:
                logger.warning("Cannot reconstruct state from minimal logging")
                return None

            # Reconstruct state from full data
            return self._reconstruct_full_state(state_data)

        except Exception as e:
            logger.error(
                f"Failed to reconstruct state for step {step_log.step_num}: {e}"
            )
            return None

    def _reconstruct_full_state(
        self, state_data: Dict[str, Any]
    ) -> Optional[ArcEnvState]:
        """Reconstruct state from full state data.

        Args:
            state_data: Full state data dictionary

        Returns:
            Reconstructed ArcEnvState or None if failed
        """
        try:
            # This is a simplified reconstruction - in practice, you'd need
            # to handle the full complexity of state serialization/deserialization

            # For now, return None as we need the actual task data and proper
            # deserialization logic which would be complex to implement fully
            logger.warning("Full state reconstruction not yet implemented")
            return None

        except Exception as e:
            logger.error(f"Failed to reconstruct full state: {e}")
            return None

    def regenerate_step_visualization(
        self,
        step_log: StepLogEntry,
        episode_num: int,
        output_path: Optional[Path] = None,
    ) -> Optional[str]:
        """Regenerate visualization for a single step from log data.

        Args:
            step_log: Step log entry
            episode_num: Episode number for file naming
            output_path: Custom output path, or None for default

        Returns:
            Path to generated visualization or None if failed
        """
        if output_path is None:
            output_path = (
                self.output_dir
                / f"episode_{episode_num:04d}"
                / f"step_{step_log.step_num:03d}_replay.svg"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Try to reconstruct states
            before_state = self.reconstruct_state_from_log(
                step_log, use_before_state=True
            )
            after_state = self.reconstruct_state_from_log(
                step_log, use_before_state=False
            )

            if before_state is None or after_state is None:
                # Fallback to basic visualization from available data
                return self._generate_basic_step_visualization(step_log, output_path)

            # Generate full visualization with reconstructed states
            svg_content = draw_rl_step_svg(
                before_state=before_state,
                after_state=after_state,
                action=step_log.action,
                reward=step_log.reward,
                info=step_log.info,
                step_num=step_log.step_num,
            )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(svg_content)

            logger.debug(
                f"Regenerated visualization for step {step_log.step_num}: {output_path}"
            )
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to regenerate step visualization: {e}")
            return None

    def _generate_basic_step_visualization(
        self, step_log: StepLogEntry, output_path: Path
    ) -> Optional[str]:
        """Generate basic visualization from limited log data.

        Args:
            step_log: Step log entry
            output_path: Output file path

        Returns:
            Path to generated visualization or None if failed
        """
        try:
            # Create a simple text-based visualization with available data
            content = f"""
            <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
                <rect width="400" height="300" fill="white" stroke="black"/>
                <text x="20" y="30" font-family="monospace" font-size="14">
                    Step {step_log.step_num} (Replay - Limited Data)
                </text>
                <text x="20" y="60" font-family="monospace" font-size="12">
                    Timestamp: {time.strftime("%H:%M:%S", time.localtime(step_log.timestamp))}
                </text>
                <text x="20" y="80" font-family="monospace" font-size="12">
                    Reward: {step_log.reward:.3f}
                </text>
                <text x="20" y="100" font-family="monospace" font-size="12">
                    Action: {step_log.action.get("operation", "Unknown")}
                </text>
                <text x="20" y="140" font-family="monospace" font-size="10" fill="red">
                    Note: Full state reconstruction not available
                </text>
                <text x="20" y="160" font-family="monospace" font-size="10" fill="red">
                    Enable include_full_states in logging config for complete replay
                </text>
            </svg>
            """

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to generate basic visualization: {e}")
            return None

    def replay_episode(
        self,
        episode_num: int,
        validate: bool = None,
        regenerate_visualizations: bool = None,
    ) -> Optional[Dict[str, Any]]:
        """Replay a complete episode from logs.

        Args:
            episode_num: Episode number to replay
            validate: Whether to validate episode integrity (uses config default if None)
            regenerate_visualizations: Whether to regenerate visualizations (uses config default if None)

        Returns:
            Replay result with metadata and paths, or None if failed
        """
        validate = validate if validate is not None else self.config.validate_integrity
        regenerate_visualizations = (
            regenerate_visualizations
            if regenerate_visualizations is not None
            else self.config.regenerate_visualizations
        )

        # Load episode data
        episode = self.load_episode(episode_num)
        if episode is None:
            logger.error(f"Failed to load episode {episode_num}")
            return None

        logger.info(f"Replaying episode {episode_num} ({episode.total_steps} steps)")

        # Validate if requested
        validation_result = None
        if validate:
            validation_result = self.validate_episode_integrity(episode)
            if not validation_result.is_valid:
                logger.error(
                    f"Episode {episode_num} failed validation: {validation_result.errors}"
                )
                if (
                    not self.config.validate_integrity
                ):  # Continue if validation not required
                    return None

        # Create episode replay directory
        episode_replay_dir = self.output_dir / f"episode_{episode_num:04d}_replay"
        episode_replay_dir.mkdir(parents=True, exist_ok=True)

        # Regenerate visualizations if requested
        visualization_paths = []
        if regenerate_visualizations:
            logger.info(f"Regenerating visualizations for episode {episode_num}")
            for step in episode.steps:
                viz_path = self.regenerate_step_visualization(step, episode_num)
                if viz_path:
                    visualization_paths.append(viz_path)

        # Create episode summary
        summary = {
            "episode_num": episode.episode_num,
            "task_id": episode.task_id,
            "total_steps": episode.total_steps,
            "total_reward": episode.total_reward,
            "final_similarity": episode.final_similarity,
            "duration": episode.end_timestamp - episode.start_timestamp,
            "start_time": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(episode.start_timestamp)
            ),
            "end_time": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(episode.end_timestamp)
            ),
            "config_hash": episode.config_hash,
            "metadata": episode.metadata,
            "validation_result": validation_result,
            "visualization_paths": visualization_paths,
            "replay_timestamp": time.time(),
        }

        # Save replay summary
        summary_path = episode_replay_dir / "replay_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(
            f"Episode {episode_num} replay completed: {len(visualization_paths)} visualizations generated"
        )
        return summary

    def list_available_episodes(self) -> List[int]:
        """List all available episodes for replay.

        Returns:
            List of episode numbers available for replay
        """
        return self.structured_logger.list_episodes()

    def get_episode_summaries(
        self, episode_nums: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Get summary information for multiple episodes.

        Args:
            episode_nums: Specific episode numbers to get summaries for, or None for all

        Returns:
            List of episode summaries
        """
        if episode_nums is None:
            episode_nums = self.list_available_episodes()

        summaries = []
        for episode_num in episode_nums[: self.config.max_episodes_to_load]:
            summary = self.structured_logger.get_episode_summary(episode_num)
            if summary:
                summaries.append(summary)

        return summaries

    def find_episodes_by_criteria(
        self,
        min_reward: Optional[float] = None,
        max_reward: Optional[float] = None,
        min_similarity: Optional[float] = None,
        max_similarity: Optional[float] = None,
        task_id: Optional[str] = None,
        min_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> List[int]:
        """Find episodes matching specific criteria.

        Args:
            min_reward: Minimum total reward
            max_reward: Maximum total reward
            min_similarity: Minimum final similarity
            max_similarity: Maximum final similarity
            task_id: Specific task ID to match
            min_steps: Minimum number of steps
            max_steps: Maximum number of steps

        Returns:
            List of episode numbers matching criteria
        """
        matching_episodes = []
        summaries = self.get_episode_summaries()

        for summary in summaries:
            # Check all criteria
            if min_reward is not None and summary["total_reward"] < min_reward:
                continue
            if max_reward is not None and summary["total_reward"] > max_reward:
                continue
            if (
                min_similarity is not None
                and summary["final_similarity"] < min_similarity
            ):
                continue
            if (
                max_similarity is not None
                and summary["final_similarity"] > max_similarity
            ):
                continue
            if task_id is not None and summary["task_id"] != task_id:
                continue
            if min_steps is not None and summary["total_steps"] < min_steps:
                continue
            if max_steps is not None and summary["total_steps"] > max_steps:
                continue

            matching_episodes.append(summary["episode_num"])

        return matching_episodes

    def clear_cache(self) -> None:
        """Clear the episode cache to free memory."""
        self._episode_cache.clear()
        logger.debug("Episode cache cleared")
