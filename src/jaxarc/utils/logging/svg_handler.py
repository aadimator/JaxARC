"""SVG visualization handler for JaxARC experiment logging."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from ..visualization.core import detect_changed_cells
from ..visualization.display import (
    draw_enhanced_episode_summary_svg,
    draw_parsed_task_data_svg,
)
from ..visualization.rl_display import (
    EpisodeConfig,
    EpisodeManager,
    draw_rl_step_svg_enhanced,
    get_operation_display_name,
)

if TYPE_CHECKING:
    from jaxarc.types import Grid


class SVGHandler:
    """Simple SVG visualization handler."""

    def __init__(self, config: Any):
        self.config = config
        episode_config_args = {
            "base_output_dir": getattr(config.storage, "base_output_dir", "outputs"),
            "run_name": getattr(config.storage, "run_name", None),
            "max_episodes_per_run": getattr(
                config.storage,
                "max_episodes_per_run",
                1000,
            ),
            "max_storage_gb": getattr(config.storage, "max_storage_gb", 10.0),
            "cleanup_policy": getattr(config.storage, "cleanup_policy", "size_based"),
        }
        episode_config = EpisodeConfig(**episode_config_args)
        self.episode_manager = EpisodeManager(episode_config)
        self.current_episode_num: int | None = None
        self.current_run_started = False

    def start_run(self, run_name: str | None = None) -> None:
        try:
            self.episode_manager.start_new_run(run_name)
            self.current_run_started = True
            logger.info(f"Started SVG run: {self.episode_manager.current_run_name}")
        except Exception as e:
            logger.error(f"Failed to start run: {e}")

    def start_episode(self, episode_num: int) -> None:
        try:
            if not self.current_run_started:
                self.start_run()
            self.episode_manager.start_new_episode(episode_num)
            self.current_episode_num = episode_num
            logger.debug(f"Started episode {episode_num}")
        except Exception as e:
            logger.error(f"Failed to start episode {episode_num}: {e}")

    def log_task_start(self, task_data: dict[str, Any]) -> None:
        if not self._should_generate_svg("task"):
            return
        try:
            episode_num = task_data.get("episode_num")
            if episode_num is not None and self.current_episode_num != episode_num:
                self.start_episode(episode_num)
            task_object = task_data.get("task_object")
            show_test = task_data.get("show_test", True)
            target_dir = self._get_output_dir()
            if not target_dir:
                return
            if task_object is None:
                self._create_simple_task_svg(task_data, target_dir)
                return
            svg_drawing = draw_parsed_task_data_svg(
                task_object, width=30.0, height=20.0, include_test=show_test
            )
            if target_dir:
                svg_path = target_dir / "task_overview.svg"
                svg_drawing.save_svg(str(svg_path))
                logger.debug(f"Saved task SVG to {svg_path}")
        except Exception as e:
            logger.error(f"Failed to generate task SVG: {e}")

    def log_step(self, step_data: dict[str, Any]) -> None:
        if not self._should_generate_svg("step"):
            return
        try:
            step_num = self._safe_int(step_data.get("step_num", 0))
            episode_num_raw = step_data.get("episode_num")
            if episode_num_raw is not None:
                episode_num = self._safe_int(episode_num_raw)
                if self.current_episode_num != episode_num:
                    self.start_episode(episode_num)
            elif self.current_episode_num is None:
                self.start_episode(0)

            before_state = step_data.get("before_state")
            after_state = step_data.get("after_state")
            action = step_data.get("action")
            reward = self._safe_float(step_data.get("reward", 0.0))
            info = step_data.get("info", {})

            if not all([before_state, after_state, action]):
                logger.warning(f"Missing data for step {step_num} SVG")
                return

            before_grid = self._extract_grid_from_state(before_state)
            after_grid = self._extract_grid_from_state(after_state)

            if not all([before_grid, after_grid]):
                logger.warning(f"Could not extract grids for step {step_num}")
                return

            changed_cells = detect_changed_cells(before_grid, after_grid)
            operation_id = self._extract_operation_id(action)
            operation_name = get_operation_display_name(operation_id, action)
            task_id = step_data.get("task_id", "")
            task_pair_index = self._safe_int(step_data.get("task_pair_index", 0))
            total_task_pairs = self._safe_int(step_data.get("total_task_pairs", 1))
            filtered_info = self._filter_info(info)

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

            svg_path = self.episode_manager.get_step_path(step_num, "svg")
            svg_path.parent.mkdir(parents=True, exist_ok=True)
            with svg_path.open("w", encoding="utf-8") as f:
                f.write(svg_content)
            logger.debug(f"Saved step {step_num} SVG to {svg_path}")
        except Exception as e:
            logger.error(
                f"Failed to generate step {step_data.get('step_num', '?')} SVG: {e}"
            )

    def log_episode_summary(self, summary_data: dict[str, Any]) -> None:
        if not self._should_generate_svg("episode"):
            return
        try:
            episode_num = self._safe_int(summary_data.get("episode_num", 0))
            environment_id = summary_data.get("environment_id")
            if self.current_episode_num != episode_num:
                self.start_episode(episode_num)
            if environment_id is not None:
                self._generate_batched_summary(summary_data, environment_id)
            else:
                self._generate_traditional_summary(summary_data)
        except Exception as e:
            logger.error(f"Failed to generate episode summary: {e}")

    def log_evaluation_summary(self, eval_data: dict[str, Any]) -> None:
        if not self._should_generate_svg("evaluation"):
            return
        try:
            test_results = eval_data.get("test_results", [])
            if not test_results:
                return
            trajectory = None
            for result in test_results:
                if isinstance(result, dict) and "trajectory" in result:
                    trajectory = result["trajectory"]
                    break
            if not trajectory:
                return
            task_id = eval_data.get("task_id", "eval_task")
            self.start_episode(0)
            for idx, step_tuple in enumerate(trajectory[:50]):
                try:
                    if len(step_tuple) == 4:
                        before_state, action, after_state, info = step_tuple
                    elif len(step_tuple) == 3:
                        before_state, action, after_state = step_tuple
                        info = {}
                    else:
                        continue
                    step_data = {
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
                    self.log_step(step_data)
                except Exception as e:
                    logger.warning(f"Failed to process eval step {idx}: {e}")
        except Exception as e:
            logger.warning(f"Evaluation SVG generation failed: {e}")

    def close(self) -> None:
        logger.debug("SVGHandler closed")

    def get_current_run_info(self) -> dict[str, Any]:
        return self.episode_manager.get_current_run_info()

    def _should_generate_svg(self, svg_type: str) -> bool:
        try:
            if not hasattr(self.config, "visualization"):
                return False
            viz_config = self.config.visualization
            if not getattr(viz_config, "enabled", True):
                return False
            if svg_type == "step":
                return getattr(viz_config, "step_visualizations", True)
            if svg_type == "episode":
                return getattr(viz_config, "episode_summaries", True)
            return True
        except Exception:
            return False

    def _get_output_dir(self) -> Path | None:
        target_dir = (
            self.episode_manager.current_episode_dir
            or self.episode_manager.current_run_dir
        )
        if target_dir is None:
            info = self.episode_manager.get_current_run_info()
            run_dir = info.get("run_dir")
            target_dir = Path(run_dir) if run_dir else Path("outputs")
        return Path(target_dir)

    def _create_simple_task_svg(
        self,
        task_data: dict[str, Any],
        target_dir: Path,
    ) -> None:
        task_id = str(task_data.get("task_id", "unknown"))
        train_pairs = task_data.get("num_train_pairs", "?")
        test_pairs = task_data.get("num_test_pairs", "?")
        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="500" height="120">
  <rect width="100%" height="100%" fill="white"/>
  <text x="20" y="28" font-size="18" font-family="Arial" fill="#222">Task Overview</text>
  <text x="20" y="54" font-size="14" font-family="Arial" fill="#111">Task ID: {task_id}</text>
  <text x="20" y="80" font-size="12" font-family="Arial" fill="#111">Train: {train_pairs}  Test: {test_pairs}</text>
</svg>"""
        svg_path = target_dir / "task_overview.svg"
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        with svg_path.open("w", encoding="utf-8") as f:
            f.write(svg_content)

    def _generate_batched_summary(
        self,
        summary_data: dict[str, Any],
        environment_id: int,
    ) -> None:
        try:
            initial_state = summary_data.get("initial_state")
            final_state = summary_data.get("final_state")
            if initial_state is None and final_state is None:
                summary_text = self._create_text_summary(summary_data, environment_id)
                filename = f"episode_summary_env_{environment_id}.txt"
                if self.episode_manager.current_episode_dir:
                    summary_path = self.episode_manager.current_episode_dir / filename
                    with summary_path.open("w", encoding="utf-8") as f:
                        f.write(summary_text)
                return
            svg_content = self._create_batched_svg(summary_data, environment_id)
            filename = f"episode_summary_env_{environment_id}.svg"
            if self.episode_manager.current_episode_dir:
                summary_path = self.episode_manager.current_episode_dir / filename
                with summary_path.open("w", encoding="utf-8") as f:
                    f.write(svg_content)
                logger.debug(f"Saved batched summary to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to generate batched summary: {e}")

    def _generate_traditional_summary(self, summary_data: dict[str, Any]) -> None:
        try:

            class SummaryData:
                def __init__(self, data):
                    self.episode_num = 0
                    self.total_steps = 0
                    self.total_reward = 0.0
                    self.final_similarity = 0.0
                    self.success = False
                    self.task_id = "Unknown"
                    self.reward_progression = []
                    self.similarity_progression = []
                    self.key_moments = []
                    for key, value in data.items():
                        setattr(self, key, value)

            summary_obj = SummaryData(summary_data)
            step_data = summary_data.get("step_data", [])
            svg_content = draw_enhanced_episode_summary_svg(
                summary_data=summary_obj,
                step_data=step_data,
                config=self.config,
            )
            summary_path = self.episode_manager.get_episode_summary_path("svg")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as f:
                f.write(svg_content)
            logger.debug(f"Saved traditional summary to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to generate traditional summary: {e}")

    def _create_batched_svg(
        self,
        summary_data: dict[str, Any],
        environment_id: int,
    ) -> str:
        import drawsvg as draw

        try:
            width, height = 800, 400
            d = draw.Drawing(width, height)
            episode_num = summary_data.get("episode_num", 0)
            task_id = summary_data.get("task_id", "Unknown")
            title = f"Episode {episode_num} - Env {environment_id} - {task_id}"
            d.append(
                draw.Text(
                    title,
                    16,
                    x=width // 2,
                    y=30,
                    text_anchor="middle",
                    font_family="Arial",
                    font_weight="bold",
                )
            )
            stats = [
                f"Reward: {summary_data.get('total_reward', 0):.3f}",
                f"Steps: {summary_data.get('total_steps', 0)}",
                f"Similarity: {summary_data.get('final_similarity', 0):.3f}",
                f"Success: {summary_data.get('success', False)}",
            ]
            for i, stat in enumerate(stats):
                d.append(draw.Text(stat, 12, x=50, y=60 + i * 20, font_family="Arial"))
            return d.as_svg()
        except Exception as e:
            logger.error(f"Failed to create batched SVG: {e}")
            return f"""<svg width="400" height="200"><text x="200" y="100" text-anchor="middle">Error: {e}</text></svg>"""

    def _create_text_summary(
        self,
        summary_data: dict[str, Any],
        environment_id: int,
    ) -> str:
        lines = [
            f"Episode Summary - Environment {environment_id}",
            "=" * 40,
            f"Episode: {summary_data.get('episode_num', 0)}",
            f"Task: {summary_data.get('task_id', 'Unknown')}",
            f"Reward: {summary_data.get('total_reward', 0):.3f}",
            f"Steps: {summary_data.get('total_steps', 0)}",
            f"Similarity: {summary_data.get('final_similarity', 0):.3f}",
            f"Success: {summary_data.get('success', False)}",
        ]
        return "\n".join(lines)

    def _extract_grid_from_state(self, state: Any) -> Grid | None:
        from jaxarc.types import Grid

        try:
            if hasattr(state, "working_grid") and hasattr(state, "working_grid_mask"):
                return Grid(
                    data=np.asarray(state.working_grid),
                    mask=np.asarray(state.working_grid_mask),
                )
            if hasattr(state, "data"):
                return state
            if isinstance(state, (np.ndarray, list)):
                return Grid(data=np.asarray(state), mask=None)
            return None
        except Exception as e:
            logger.error(f"Failed to extract grid: {e}")
            return None

    def _extract_operation_id(self, action: Any) -> int:
        try:
            if hasattr(action, "operation"):
                return self._safe_int(action.operation)
            if isinstance(action, dict):
                return self._safe_int(action.get("operation", 0))
            if isinstance(action, tuple) and len(action) >= 1:
                return self._safe_int(action[0])
            return 0
        except Exception:
            return 0

    def _filter_info(self, info: Any) -> dict[str, Any]:
        viz_keys = {
            "success",
            "similarity",
            "similarity_improvement",
            "step_count",
            "operation_type",
            "episode_mode",
            "current_pair_index",
        }
        filtered = {}
        if isinstance(info, dict):
            for key, value in info.items():
                if key in viz_keys:
                    filtered[key] = value
        else:
            for key in viz_keys:
                if hasattr(info, key):
                    filtered[key] = getattr(info, key)
        return filtered

    def _safe_int(self, value: Any) -> int:
        """Called outside JIT — .item() is safe here."""
        if value is None:
            return 0
        try:
            if hasattr(value, "item"):
                return int(value.item())
            return int(value)
        except Exception:
            return 0

    def _safe_float(self, value: Any) -> float:
        """Called outside JIT — .item() is safe here."""
        try:
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        except Exception:
            return 0.0
