"""SVG visualization generation handler for JaxARC logging.

This module provides the SVGHandler class that consolidates SVG generation logic
from rl_visualization.py and episode_visualization.py into a single handler for
the simplified logging architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover
    import drawsvg as draw  # type: ignore[import-not-found]
try:  # Optional dependency for SVG creation at runtime
    import drawsvg as draw  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency path
    draw = None  # type: ignore[assignment]
    logger.warning("drawsvg not available - SVG generation degraded")

from ...types import Grid
from ..visualization.episode_manager import EpisodeConfig, EpisodeManager
from ..visualization.episode_visualization import draw_enhanced_episode_summary_svg
from ..visualization.rl_visualization import (
    draw_rl_step_svg_enhanced,
    get_operation_display_name,
)
from ..visualization.task_visualization import draw_parsed_task_data_svg
from ..visualization.utils import detect_changed_cells


class SVGHandler:
    """SVG visualization generation handler.

    Note: Regular Python class that can freely use file I/O,
    string manipulation, and SVG generation libraries.

    This handler consolidates SVG generation logic from the visualization
    modules and provides a clean interface for the ExperimentLogger.
    """

    def __init__(self, config: Any):
        """Initialize SVG handler with configuration.

        Args:
            config: JaxArcConfig containing visualization and storage settings
        """
        self.config = config

        # Initialize episode manager for file path management
        # Create episode config from storage config
        episode_config = EpisodeConfig(
            base_output_dir=getattr(config.storage, "base_output_dir", "outputs"),
            run_name=getattr(config.storage, "run_name", None),
            max_episodes_per_run=getattr(config.storage, "max_episodes_per_run", 1000),
            max_storage_gb=getattr(config.storage, "max_storage_gb", 10.0),
            cleanup_policy=getattr(config.storage, "cleanup_policy", "size_based"),
        )

        self.episode_manager = EpisodeManager(episode_config)

        # Track current episode for file management
        self.current_episode_num: int | None = None
        self.current_run_started = False

        logger.debug("SVGHandler initialized")

    def start_run(self, run_name: str | None = None) -> None:
        """Start a new run for SVG generation.

        Args:
            run_name: Optional custom run name
        """
        try:
            self.episode_manager.start_new_run(run_name)
            self.current_run_started = True
            logger.info(
                f"SVGHandler started new run: {self.episode_manager.current_run_name}"
            )
        except Exception as e:
            logger.error(f"Failed to start SVG run: {e}")

    def start_episode(self, episode_num: int) -> None:
        """Start a new episode for SVG generation.

        Args:
            episode_num: Episode number
        """
        try:
            if not self.current_run_started:
                self.start_run()

            self.episode_manager.start_new_episode(episode_num)
            self.current_episode_num = episode_num
            logger.debug(f"SVGHandler started episode {episode_num}")
        except Exception as e:
            logger.error(f"Failed to start SVG episode {episode_num}: {e}")

    def log_task_start(self, task_data: dict[str, Any]) -> None:
        """Generate and save task visualization at episode start.

        Args:
            task_data: Dictionary containing task information including:
                - task_id: Task identifier
                - task_object: The JaxArcTask object
                - episode_num: Episode number
                - show_test: Whether to show test examples (default: True)
        """
        # Check if task visualization is enabled
        if not self._should_generate_task_svg():
            return

        try:
            episode_num = task_data.get("episode_num", 0)
            task_object = task_data.get("task_object")
            show_test = task_data.get("show_test", True)

            # Ensure episode is started
            if self.current_episode_num != episode_num:
                self.start_episode(episode_num)

            if task_object is None:
                logger.warning("No task object provided for task visualization")
                return

            # Generate task SVG using existing function with configurable test display
            svg_drawing = draw_parsed_task_data_svg(
                task_object,
                width=30.0,
                height=20.0,
                include_test=show_test,  # Configurable test example display
            )

            # Save task SVG
            task_path = self.episode_manager.current_episode_dir / "task_overview.svg"
            task_path.parent.mkdir(parents=True, exist_ok=True)

            with Path.open(task_path, "w", encoding="utf-8") as f:
                f.write(svg_drawing.as_svg())

            logger.debug(f"Saved task overview SVG to {task_path}")

        except Exception as e:
            logger.error(f"Failed to generate task SVG: {e}")

    def log_step(self, step_data: dict[str, Any]) -> None:
        """Generate and save step visualization.

        Args:
            step_data: Dictionary containing step information including:
                - step_num: Step number
                - before_state: Environment state before action
                - after_state: Environment state after action
                - action: Action taken
                - reward: Reward received
                - info: Additional information dictionary
        """
        # Check if step visualization is enabled
        if not self._should_generate_step_svg():
            return

        try:
            step_num = step_data.get("step_num", 0)
            episode_num = step_data.get("episode_num")

            # Ensure episode is started
            if episode_num is not None and self.current_episode_num != episode_num:
                self.start_episode(episode_num)
            elif self.current_episode_num is None:
                # Default to episode 0 if not specified
                self.start_episode(0)

            # Extract required data
            before_state = step_data.get("before_state")
            after_state = step_data.get("after_state")
            action = step_data.get("action")
            reward = step_data.get("reward", 0.0)
            info = step_data.get("info", {})

            if before_state is None or after_state is None or action is None:
                logger.warning(
                    f"Missing required data for step {step_num} SVG generation"
                )
                return

            # Create Grid objects from states
            before_grid = self._extract_grid_from_state(before_state)
            after_grid = self._extract_grid_from_state(after_state)

            if before_grid is None or after_grid is None:
                logger.warning(f"Could not extract grids for step {step_num}")
                return

            # Detect changed cells
            changed_cells = detect_changed_cells(before_grid, after_grid)

            # Get operation name
            operation_id = self._extract_operation_id(action)
            operation_name = get_operation_display_name(operation_id, action)

            # Extract additional context
            task_id = step_data.get("task_id", "")
            task_pair_index = step_data.get("task_pair_index", 0)
            total_task_pairs = step_data.get("total_task_pairs", 1)

            # Filter info dictionary to only include known visualization keys
            # This makes SVGHandler ignore unknown keys gracefully
            filtered_info = self._filter_info_for_visualization(info)

            # Generate SVG content
            svg_content = draw_rl_step_svg_enhanced(
                before_grid=before_grid,
                after_grid=after_grid,
                action=action,
                reward=reward,
                info=filtered_info,
                step_num=step_num,
                operation_name=operation_name,
                changed_cells=changed_cells,
                config=self.config,
                task_id=task_id,
                task_pair_index=task_pair_index,
                total_task_pairs=total_task_pairs,
            )

            # Save SVG file
            svg_path = self.episode_manager.get_step_path(step_num, "svg")
            svg_path.parent.mkdir(parents=True, exist_ok=True)

            with Path.open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg_content)

            logger.debug(f"Saved step {step_num} SVG to {svg_path}")

        except Exception as e:
            logger.error(
                f"Failed to generate step {step_data.get('step_num', 'unknown')} SVG: {e}"
            )

    def log_episode_summary(self, summary_data: dict[str, Any]) -> None:
        """Generate and save episode summary visualization.

        This method handles both traditional episode summaries and episode summary data
        from batched training sampling. It supports initial vs final state visualization
        when state data is available and gracefully degrades when not provided.

        Args:
            summary_data: Dictionary containing episode summary information including:
                - episode_num: Episode number
                - total_steps: Total number of steps
                - total_reward: Total reward received
                - final_similarity: Final similarity score
                - success: Whether episode was successful
                - task_id: Task identifier
                - environment_id: Environment ID (for batched sampling)
                - initial_state: Initial state (optional, for batched sampling)
                - final_state: Final state (optional, for batched sampling)
                - step_data: List of step data for visualization (traditional mode)
        """
        # Check if episode summary visualization is enabled
        if not self._should_generate_episode_summary():
            return

        try:
            episode_num = summary_data.get("episode_num", 0)
            environment_id = summary_data.get("environment_id")

            # Ensure episode is started
            if self.current_episode_num != episode_num:
                self.start_episode(episode_num)

            # Handle batched sampling mode vs traditional mode
            if environment_id is not None:
                # Batched sampling mode - generate visualization from initial/final states
                self._generate_batched_episode_summary(summary_data, environment_id)
            else:
                # Traditional mode - use step data for full episode visualization
                self._generate_traditional_episode_summary(summary_data)

        except Exception as e:
            logger.error(f"Failed to generate episode summary SVG: {e}")

    def _generate_batched_episode_summary(
        self, summary_data: dict[str, Any], environment_id: int
    ) -> None:
        """Generate episode summary for batched sampling mode.

        Args:
            summary_data: Episode summary data from batched sampling
            environment_id: Environment ID within the batch
        """
        try:
            episode_num = summary_data.get("episode_num", 0)
            initial_state = summary_data.get("initial_state")
            final_state = summary_data.get("final_state")

            # Check if we have state data for visualization
            if initial_state is None and final_state is None:
                # Graceful degradation - create text-only summary
                self._generate_text_only_summary(summary_data, environment_id)
                return

            # Extract grids from states when available
            initial_grid = None
            final_grid = None

            if initial_state is not None:
                initial_grid = self._extract_grid_from_state(initial_state)

            if final_state is not None:
                final_grid = self._extract_grid_from_state(final_state)

            # Generate SVG content for initial vs final state comparison
            svg_content = self._create_batched_summary_svg(
                summary_data=summary_data,
                initial_grid=initial_grid,
                final_grid=final_grid,
                environment_id=environment_id,
            )

            # Save summary SVG with environment ID in filename
            filename_suffix = (
                f"_env_{environment_id}" if environment_id is not None else ""
            )
            summary_path = (
                self.episode_manager.current_episode_dir
                / f"episode_summary{filename_suffix}.svg"
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            with Path.open(summary_path, "w", encoding="utf-8") as f:
                f.write(svg_content)

            logger.debug(f"Saved batched episode summary SVG to {summary_path}")

        except Exception as e:
            logger.error(f"Failed to generate batched episode summary: {e}")

    def _generate_traditional_episode_summary(
        self, summary_data: dict[str, Any]
    ) -> None:
        """Generate episode summary for traditional mode with step data.

        Args:
            summary_data: Traditional episode summary data
        """
        try:
            episode_num = summary_data.get("episode_num", 0)
            step_data = summary_data.get("step_data", [])

            # Convert dictionary to object for compatibility with visualization function
            class SummaryDataWrapper:
                def __init__(self, data_dict):
                    # Set default values for expected attributes
                    self.episode_num = 0
                    self.total_steps = 0
                    self.total_reward = 0.0
                    self.final_similarity = 0.0
                    self.success = False
                    self.task_id = "Unknown"
                    self.reward_progression = []
                    self.similarity_progression = []
                    self.key_moments = []

                    # Override with actual data
                    for key, value in data_dict.items():
                        setattr(self, key, value)

            summary_obj = SummaryDataWrapper(summary_data)

            # Generate episode summary SVG using existing function
            svg_content = draw_enhanced_episode_summary_svg(
                summary_data=summary_obj,
                step_data=step_data,
                config=self.config,
            )

            # Save summary SVG
            summary_path = self.episode_manager.get_episode_summary_path("svg")
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            with Path.open(summary_path, "w", encoding="utf-8") as f:
                f.write(svg_content)

            logger.debug(
                f"Saved traditional episode {episode_num} summary SVG to {summary_path}"
            )

        except Exception as e:
            logger.error(f"Failed to generate traditional episode summary: {e}")

    def _generate_text_only_summary(
        self, summary_data: dict[str, Any], environment_id: int
    ) -> None:
        """Generate text-only summary when state data is not available.

        Args:
            summary_data: Episode summary data
            environment_id: Environment ID within the batch
        """
        try:
            # Create simple text-based summary
            summary_text = self._create_text_summary(summary_data, environment_id)

            # Save as text file instead of SVG
            filename_suffix = (
                f"_env_{environment_id}" if environment_id is not None else ""
            )
            summary_path = (
                self.episode_manager.current_episode_dir
                / f"episode_summary{filename_suffix}.txt"
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            with Path.open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_text)

            logger.debug(f"Saved text-only episode summary to {summary_path}")

        except Exception as e:
            logger.error(f"Failed to generate text-only summary: {e}")

    def _create_batched_summary_svg(
        self,
        summary_data: dict[str, Any],
        initial_grid: Grid | None,
        final_grid: Grid | None,
        environment_id: int,
    ) -> str:
        """Create SVG content for batched episode summary.

        Args:
            summary_data: Episode summary data
            initial_grid: Initial grid state (optional)
            final_grid: Final grid state (optional)
            environment_id: Environment ID within the batch

        Returns:
            SVG content as string
        """
        try:
            if draw is None:  # type: ignore[truthy-function]
                logger.error("drawsvg not installed; cannot create batched summary SVG")
                return '<svg width="400" height="200"><text x="200" y="100" text-anchor="middle">drawsvg not installed</text></svg>'

            # Create drawing with appropriate size
            width = 800
            height = 400
            d = draw.Drawing(width, height)

            # Add title
            episode_num = summary_data.get("episode_num", 0)
            task_id = summary_data.get("task_id", "Unknown")
            title = f"Episode {episode_num} - Environment {environment_id} - Task: {task_id}"
            d.append(
                draw.Text(
                    title,
                    20,
                    x=width // 2,
                    y=30,
                    text_anchor="middle",
                    font_family="Arial",
                    font_size=16,
                    font_weight="bold",
                )
            )

            # Add summary statistics
            stats_y = 60
            stats = [
                f"Total Reward: {summary_data.get('total_reward', 0):.3f}",
                f"Total Steps: {summary_data.get('total_steps', 0)}",
                f"Final Similarity: {summary_data.get('final_similarity', 0):.3f}",
                f"Success: {summary_data.get('success', False)}",
            ]

            for i, stat in enumerate(stats):
                d.append(
                    draw.Text(
                        stat,
                        14,
                        x=50,
                        y=stats_y + i * 20,
                        font_family="Arial",
                        font_size=12,
                    )
                )

            # Draw grids if available
            grid_y = 150
            grid_size = 200

            if initial_grid is not None:
                # Draw initial grid
                self._draw_grid_in_svg(
                    d,
                    initial_grid,
                    x=100,
                    y=grid_y,
                    size=grid_size,
                    title="Initial State",
                )

            if final_grid is not None:
                # Draw final grid
                x_offset = 450 if initial_grid is not None else 300
                self._draw_grid_in_svg(
                    d,
                    final_grid,
                    x=x_offset,
                    y=grid_y,
                    size=grid_size,
                    title="Final State",
                )

            # Add note about batched mode
            d.append(
                draw.Text(
                    "Generated from batched training sample",
                    12,
                    x=width // 2,
                    y=height - 20,
                    text_anchor="middle",
                    font_family="Arial",
                    font_size=10,
                    fill="gray",
                )
            )

            return d.as_svg()

        except Exception as e:
            logger.error(f"Failed to create batched summary SVG: {e}")
            # Return minimal SVG on error
            return f'<svg width="400" height="200"><text x="200" y="100" text-anchor="middle">Error generating visualization: {e}</text></svg>'

    def _draw_grid_in_svg(
        self, drawing, grid: Grid, x: int, y: int, size: int, title: str
    ) -> None:
        """Draw a grid in the SVG drawing.

        Args:
            drawing: DrawSVG drawing object
            grid: Grid object to draw
            x: X position
            y: Y position
            size: Size of the grid visualization
            title: Title for the grid
        """
        try:
            if draw is None:  # type: ignore[truthy-function]
                logger.error("drawsvg not installed; cannot draw grid")
                return

            # Add title
            drawing.append(
                draw.Text(
                    title,
                    12,
                    x=x + size // 2,
                    y=y - 10,
                    text_anchor="middle",
                    font_family="Arial",
                    font_size=12,
                    font_weight="bold",
                )
            )

            # Get grid data
            grid_data = (
                np.asarray(grid.data) if hasattr(grid, "data") else np.asarray(grid)
            )
            rows, cols = grid_data.shape

            # Calculate cell size
            cell_size = min(size // max(rows, cols), 20)

            # Color mapping for ARC (0-9 colors)
            colors = [
                "#000000",
                "#0074D9",
                "#FF4136",
                "#2ECC40",
                "#FFDC00",
                "#AAAAAA",
                "#F012BE",
                "#FF851B",
                "#7FDBFF",
                "#870C25",
            ]

            # Draw grid cells
            for i in range(rows):
                for j in range(cols):
                    cell_x = x + j * cell_size
                    cell_y = y + i * cell_size
                    color_idx = int(grid_data[i, j]) % len(colors)

                    # Draw cell
                    drawing.append(
                        draw.Rectangle(
                            cell_x,
                            cell_y,
                            cell_size,
                            cell_size,
                            fill=colors[color_idx],
                            stroke="white",
                            stroke_width=1,
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to draw grid in SVG: {e}")

    def _create_text_summary(
        self, summary_data: dict[str, Any], environment_id: int
    ) -> str:
        """Create text-based summary when visualization is not possible.

        Args:
            summary_data: Episode summary data
            environment_id: Environment ID within the batch

        Returns:
            Text summary as string
        """
        lines = [
            f"Episode Summary - Environment {environment_id}",
            "=" * 50,
            f"Episode Number: {summary_data.get('episode_num', 0)}",
            f"Task ID: {summary_data.get('task_id', 'Unknown')}",
            f"Total Reward: {summary_data.get('total_reward', 0):.3f}",
            f"Total Steps: {summary_data.get('total_steps', 0)}",
            f"Final Similarity: {summary_data.get('final_similarity', 0):.3f}",
            f"Success: {summary_data.get('success', False)}",
            "",
            "Note: This summary was generated from batched training data.",
            "State visualization is not available as intermediate states",
            "are not stored during batched training for performance reasons.",
        ]

        return "\n".join(lines)

    def close(self) -> None:
        """Clean shutdown of SVG handler."""
        try:
            # Perform any cleanup if needed
            logger.debug("SVGHandler closed")
        except Exception as e:
            logger.error(f"Error during SVGHandler shutdown: {e}")

    def _should_generate_step_svg(self) -> bool:
        """Check if step SVG generation is enabled based on configuration."""
        try:
            # Check visualization config
            if hasattr(self.config, "visualization"):
                viz_config = self.config.visualization
                if getattr(viz_config, "enabled", True) and getattr(
                    viz_config, "step_visualizations", True
                ):
                    return True
            return False
        except Exception:
            return False

    def _should_generate_task_svg(self) -> bool:
        """Check if task SVG generation is enabled based on configuration."""
        try:
            # Check visualization config
            # Check visualization config
            if hasattr(self.config, "visualization"):
                viz_config = self.config.visualization
                if getattr(viz_config, "enabled", True):
                    return True
            return False
        except Exception:
            return True

    def _should_generate_episode_summary(self) -> bool:
        """Check if episode summary generation is enabled based on configuration."""
        try:
            # Check visualization config
            if hasattr(self.config, "visualization"):
                viz_config = self.config.visualization
                if getattr(viz_config, "enabled", True) and getattr(
                    viz_config, "episode_summaries", True
                ):
                    return True
            return False
        except Exception:
            return True

    def _extract_grid_from_state(self, state: Any) -> Grid | None:
        """Extract Grid object from environment state."""
        try:
            if hasattr(state, "working_grid") and hasattr(state, "working_grid_mask"):
                return Grid(
                    data=np.asarray(state.working_grid),
                    mask=np.asarray(state.working_grid_mask),
                )
            if hasattr(state, "data") and hasattr(state, "mask"):
                return state  # Already Grid-like
            if isinstance(state, (np.ndarray, list)):
                return Grid(data=np.asarray(state), mask=None)
            logger.warning(f"Unknown state format: {type(state)}")
            return None
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to extract grid from state: {e}")
            return None

    def _extract_operation_id(self, action: Any) -> int:
        """Extract operation ID from action."""
        try:
            if hasattr(action, "operation"):
                op_val = action.operation
            elif isinstance(action, dict) and "operation" in action:
                op_val = action["operation"]
            else:
                logger.warning(
                    f"Could not extract operation from action: {type(action)}"
                )
                return 0
            if hasattr(op_val, "item"):
                op_val = op_val.item()
            return int(op_val)
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to extract operation ID: {e}")
            return 0

    def _filter_info_for_visualization(self, info: Any) -> dict[str, Any]:
        """Filter info object (StepInfo or dict) to include keys relevant for visualization."""
        # Known keys used by visualization functions
        known_viz_keys = {
            "success",
            "similarity",
            "similarity_improvement",
            "step_count",
            "operation_type",
            "episode_mode",
            "current_pair_index",
        }

        filtered_info = {}
        # Handle both the new StepInfo object and old dict format for robustness
        if hasattr(info, "__class__") and "StepInfo" in info.__class__.__name__:
            for key in known_viz_keys:
                if hasattr(info, key):
                    filtered_info[key] = getattr(info, key)
        elif isinstance(info, dict):
            for key, value in info.items():
                if key in known_viz_keys:
                    filtered_info[key] = value

        return filtered_info

    def get_current_run_info(self) -> dict[str, Any]:
        """Get information about the current run.

        Returns:
            Dictionary with current run information
        """
        return self.episode_manager.get_current_run_info()

    # --- Evaluation Summary Support ---
    def _should_generate_evaluation_svg(self) -> bool:
        """Check if evaluation SVG generation is enabled.

        Reuses existing visualization gating; we keep this conservative to avoid
        large output explosions. Defaults to using episode summary setting.
        """
        return self._should_generate_episode_summary()

    def log_evaluation_summary(self, eval_data: dict[str, Any]) -> None:
        """Generate SVGs for evaluation trajectories (if provided).

        Expects eval_data['test_results'] to contain dicts that may include a
        'trajectory' key which is an iterable of step tuples:
            (before_state, action, after_state, info)
        Only the first available trajectory is rendered to limit storage.
        """
        if not self._should_generate_evaluation_svg():
            return
        try:
            test_results = eval_data.get("test_results")
            if not isinstance(test_results, list) or not test_results:
                return
            representative = None
            for res in test_results:
                if isinstance(res, dict) and "trajectory" in res:
                    representative = res
                    break
            if representative is None:
                return
            trajectory = representative.get("trajectory")
            if not trajectory:
                return
            task_id = eval_data.get("task_id", "eval_task")
            # Start a synthetic episode namespace for evaluation visualization
            self.start_episode(episode_num=0)
            # Iterate through limited number of steps to avoid huge dumps
            max_steps = 100
            for idx, step_tuple in enumerate(trajectory):
                if idx >= max_steps:
                    break
                try:
                    before_state, action, after_state, info = step_tuple
                except ValueError:
                    # Fallback if info missing: (before, action, after)
                    if len(step_tuple) == 3:
                        before_state, action, after_state = step_tuple
                        info = {}
                    else:
                        continue
                step_payload = {
                    "step_num": idx,
                    "before_state": before_state,
                    "after_state": after_state,
                    "action": action,
                    "reward": info.get("reward", 0.0)
                    if isinstance(info, dict)
                    else 0.0,
                    "info": info if isinstance(info, dict) else {},
                    "task_id": task_id,
                }
                self.log_step(step_payload)
        except Exception as e:
            logger.warning(f"Evaluation SVG generation failed: {e}")
