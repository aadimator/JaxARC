"""
Logging handlers for different output formats and services.

Includes file, terminal, SVG, and WandB handlers.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..serialization_utils import serialize_log_step
from ..visualization.display import (
    visualize_grid_rich,
    visualize_parsed_task_data_rich,
    visualize_task_pair_rich,
)
from .logger import to_python_float, to_python_scalar

if TYPE_CHECKING:
    from rich.console import Console


# ============================================================================
# SECTION: File Handler (from file_handler.py)
# ============================================================================


class FileHandler:
    """Synchronous file logging handler for episode data."""

    def __init__(self, config):
        if hasattr(config, "storage") and hasattr(config.storage, "base_output_dir"):
            base_dir = config.storage.base_output_dir
            logs_subdir = getattr(config.storage, "logs_dir", "logs")
            self.output_dir = Path(base_dir) / logs_subdir
        elif hasattr(config, "debug") and hasattr(config.debug, "output_dir"):
            self.output_dir = Path(config.debug.output_dir)
        else:
            self.output_dir = Path("outputs/logs")

        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_episode_data = {}

        logger.info(f"FileHandler initialized with output directory: {self.output_dir}")

    def log_task_start(self, task_data: dict[str, Any]) -> None:
        self.current_episode_data["task_info"] = {
            "task_id": task_data.get("task_id"),
            "episode_num": task_data.get("episode_num"),
            "num_train_pairs": task_data.get("num_train_pairs", 0),
            "num_test_pairs": task_data.get("num_test_pairs", 0),
            "task_stats": task_data.get("task_stats", {}),
        }

        task_object = task_data.get("task_object")
        if task_object is not None:
            try:
                self.current_episode_data["task_info"]["task_structure"] = {
                    "input_grid_shapes": [],
                    "output_grid_shapes": [],
                    "max_colors_used": 0,
                }

                for i in range(getattr(task_object, "num_train_pairs", 0)):
                    if hasattr(task_object, "input_grids_examples"):
                        input_grid = task_object.input_grids_examples[i]
                        self.current_episode_data["task_info"]["task_structure"][
                            "input_grid_shapes"
                        ].append(
                            list(input_grid.shape)
                            if hasattr(input_grid, "shape")
                            else [0, 0]
                        )

                    if hasattr(task_object, "output_grids_examples"):
                        output_grid = task_object.output_grids_examples[i]
                        self.current_episode_data["task_info"]["task_structure"][
                            "output_grid_shapes"
                        ].append(
                            list(output_grid.shape)
                            if hasattr(output_grid, "shape")
                            else [0, 0]
                        )

            except Exception as e:
                logger.warning(f"Failed to serialize task structure: {e}")

        logger.debug(
            f"Logged task start for task {task_data.get('task_id', 'unknown')}"
        )

    def log_step(self, step_data: dict[str, Any]) -> None:
        if "steps" not in self.current_episode_data:
            self.current_episode_data["steps"] = []

        serialized_step = serialize_log_step(step_data)
        self.current_episode_data["steps"].append(serialized_step)

        logger.debug(f"Logged step {step_data.get('step_num', 'unknown')}")

    def log_episode_summary(self, summary_data: dict[str, Any]) -> None:
        episode_num = summary_data.get("episode_num", 0)

        complete_episode = {
            **summary_data,
            **self.current_episode_data,
            "timestamp": time.time(),
            "config_hash": self._get_config_hash(),
        }

        self._save_json(complete_episode, episode_num)
        self._save_pickle(complete_episode, episode_num)

        logger.info(
            f"Saved episode {episode_num}: "
            f"{len(self.current_episode_data.get('steps', []))} steps, "
            f"reward: {summary_data.get('total_reward', 0):.3f}"
        )

        self.current_episode_data = {}

    def log_aggregated_metrics(self, metrics: dict[str, float], step: int) -> None:
        try:
            batch_metrics_file = self.output_dir / "batch_metrics.jsonl"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            log_entry = {"timestamp": time.time(), "step": step, "metrics": metrics}
            with Path(batch_metrics_file).open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")
            logger.debug(f"Logged batch metrics to {batch_metrics_file} at step {step}")
        except Exception as e:
            logger.warning(f"File batch logging failed: {e}")

    def log_evaluation_summary(self, eval_data: dict[str, Any]) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "evaluation_summary.json"
            payload = {
                "timestamp": time.time(),
                "evaluation": eval_data,
                "config_hash": self._get_config_hash(),
            }
            with Path(path).open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.info(f"Saved evaluation summary to {path}")
        except Exception as e:
            logger.warning(f"Failed to write evaluation summary: {e}")

    def close(self) -> None:
        if self.current_episode_data:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                incomplete_path = self.output_dir / "incomplete_episode.json"
                with Path(incomplete_path).open("w", encoding="utf-8") as f:
                    json.dump(self.current_episode_data, f, indent=2, default=str)
                logger.info(f"Saved incomplete episode data to {incomplete_path}")
            except Exception as e:
                logger.error(f"Failed to save incomplete episode data: {e}")
        logger.info("FileHandler shutdown complete")

    def _save_json(self, episode_data: dict[str, Any], episode_num: int) -> None:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{episode_num:04d}_{timestamp_str}.json"
        filepath = self.output_dir / filename
        try:
            with Path(filepath).open("w", encoding="utf-8") as f:
                json.dump(episode_data, f, indent=2, default=str)
            logger.debug(f"Saved JSON episode data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save JSON episode data: {e}")

    def _save_pickle(self, episode_data: dict[str, Any], episode_num: int) -> None:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{episode_num:04d}_{timestamp_str}.pkl"
        filepath = self.output_dir / filename
        try:
            with Path(filepath).open("wb") as f:
                pickle.dump(episode_data, f)
            logger.debug(f"Saved pickle episode data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save pickle episode data: {e}")

    def _get_config_hash(self) -> str:
        try:
            import hashlib

            config_str = str(self.config)
            return hashlib.md5(config_str.encode()).hexdigest()[:8]
        except Exception:
            return "unknown"


# ============================================================================
# SECTION: Rich Terminal Handler (from rich_handler.py)
# ============================================================================


class RichHandler:
    def __init__(self, config: Any):
        from rich.console import Console

        self.config = config
        self.console: Console = Console()

    def _get_debug_level(self) -> str:
        if hasattr(self.config, "environment") and hasattr(
            self.config.environment,
            "debug_level",
        ):
            return self.config.environment.debug_level
        return getattr(self.config, "debug_level", "minimal")

    def log_task_start(self, task_data: dict[str, Any]) -> None:
        if self._get_debug_level() in ["minimal", "verbose"]:
            self._display_task_info(task_data)

    def log_step(self, step_data: dict[str, Any]) -> None:
        if self._get_debug_level() in ["minimal", "verbose"]:
            self._display_step_info(step_data)

    def log_episode_summary(self, summary_data: dict[str, Any]) -> None:
        self._display_episode_summary(summary_data)

    def log_aggregated_metrics(self, metrics: dict[str, float], step: int) -> None:
        from rich.table import Table

        try:
            table = Table(title=f"Batch Metrics - Step {step}", title_style="bold cyan")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="green", justify="right")

            reward_metrics = {
                k: v for k, v in metrics.items() if k.startswith("reward_")
            }
            similarity_metrics = {
                k: v for k, v in metrics.items() if k.startswith("similarity_")
            }
            episode_metrics = {
                k: v for k, v in metrics.items() if k.startswith("episode_length_")
            }
            training_metrics = {
                k: v
                for k, v in metrics.items()
                if k in ["policy_loss", "value_loss", "gradient_norm"]
            }
            other_metrics = {
                k: v
                for k, v in metrics.items()
                if k not in reward_metrics
                and k not in similarity_metrics
                and k not in episode_metrics
                and k not in training_metrics
            }

            categories = [
                ("Rewards", reward_metrics),
                ("Similarity", similarity_metrics),
                ("Episode Length", episode_metrics),
                ("Training", training_metrics),
                ("Other", other_metrics),
            ]
            for category_name, category_metrics in categories:
                if category_metrics:
                    table.add_row(f"[bold]{category_name}[/bold]", "", style="dim")
                    for key, value in category_metrics.items():
                        if isinstance(value, float):
                            if abs(value) < 0.001:
                                formatted_value = f"{value:.6f}"
                            elif abs(value) < 1:
                                formatted_value = f"{value:.4f}"
                            else:
                                formatted_value = f"{value:.3f}"
                        else:
                            formatted_value = str(value)
                        table.add_row(
                            f"  {key.replace('_', ' ').title()}", formatted_value
                        )
            self.console.print(table)
        except Exception as e:
            logger.warning(f"Rich batch logging failed: {e}")

    def log_evaluation_summary(self, eval_data: dict[str, Any]) -> None:
        from rich.panel import Panel
        from rich.table import Table

        try:
            table = Table(title="Evaluation Summary", title_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green", justify="right")
            core = [
                ("task_id", eval_data.get("task_id")),
                ("success_rate", eval_data.get("success_rate")),
                ("average_episode_length", eval_data.get("average_episode_length")),
                ("num_timeouts", eval_data.get("num_timeouts")),
            ]
            for k, v in core:
                if v is not None:
                    if isinstance(v, float):
                        disp = f"{v:.4f}" if abs(v) < 1 else f"{v:.3f}"
                    else:
                        disp = str(v)
                    table.add_row(k.replace("_", " ").title(), disp)
            trs = eval_data.get("test_results")
            if isinstance(trs, list):
                table.add_row("Test Results", str(len(trs)))
            self.console.print(Panel.fit(table, border_style="magenta"))
        except Exception as e:
            logger.warning(f"Rich evaluation summary logging failed: {e}")

    def close(self) -> None:
        return None

    def _display_step_info(self, step_data: dict[str, Any]) -> None:
        step_num = step_data.get("step_num", 0)
        reward = step_data.get("reward", 0.0)
        try:
            r_val = to_python_float(reward)
            reward = float(r_val) if r_val is not None else float(reward)
        except Exception:
            try:
                reward = float(reward)
            except Exception:
                reward = 0.0
        info = step_data.get("info")
        self.console.print(f"\n[bold blue]Step {step_num}[/bold blue]")
        reward_color = "green" if reward > 0 else "red" if reward < 0 else "yellow"
        self.console.print(f"Reward: [{reward_color}]{reward:.3f}[/{reward_color}]")
        before_state = step_data.get("before_state")
        after_state = step_data.get("after_state")

        if (
            before_state
            and hasattr(before_state, "working_grid")
            and after_state
            and hasattr(after_state, "working_grid")
        ):
            visualize_task_pair_rich(
                input_grid=before_state.working_grid,
                output_grid=after_state.working_grid,
                title=f"Step {step_num}",
                console=self.console,
            )
        elif before_state and hasattr(before_state, "working_grid"):
            grid_table = visualize_grid_rich(
                before_state.working_grid,
                title=f"Step {step_num} - Before",
                border_style="input",
            )
            self.console.print(grid_table)

        if info:
            if hasattr(info, "success") and hasattr(info, "similarity_improvement"):
                self.console.print("[bold]Metrics:[/bold]")
                for key in ("success", "similarity_improvement", "similarity"):
                    if hasattr(info, key):
                        v = getattr(info, key)
                        v_parsed = to_python_scalar(v)
                        if isinstance(v_parsed, float):
                            self.console.print(f"  {key}: {v_parsed:.3f}")
                        elif isinstance(v_parsed, (int, bool)):
                            self.console.print(f"  {key}: {v_parsed}")
                        else:
                            self.console.print(f"  {key}: {v_parsed}")
            elif isinstance(info, dict):
                if "metrics" in info and isinstance(info["metrics"], dict):
                    self.console.print("[bold]Metrics:[/bold]")
                    for k, v in info["metrics"].items():
                        v_parsed = to_python_scalar(v)
                        if isinstance(v_parsed, float):
                            self.console.print(f"  {k}: {v_parsed:.3f}")
                        elif isinstance(v_parsed, (int, bool)):
                            self.console.print(f"  {k}: {v_parsed}")
                        else:
                            self.console.print(f"  {k}: {v_parsed}")

        action = step_data.get("action")
        if action is not None:
            self.console.print(f"[bold]Action:[/bold] {action}")

    def _display_task_info(self, task_data: dict[str, Any]) -> None:
        from rich.panel import Panel
        from rich.text import Text

        task_id = task_data.get("task_id", "Unknown")
        episode_num = task_data.get("episode_num", 0)
        num_train_pairs = task_data.get("num_train_pairs", 0)
        num_test_pairs = task_data.get("num_test_pairs", 0)
        task_info = Text(justify="center")
        task_info.append("Training Examples: ", style="bold")
        task_info.append(str(num_train_pairs), style="green")
        task_info.append("  ")
        task_info.append("Test Examples: ", style="bold")
        task_info.append(str(num_test_pairs), style="blue")
        panel = Panel(
            task_info,
            title=f"Episode {episode_num} - Task: {task_id}",
            title_align="left",
            border_style="bright_blue",
            padding=(0, 1),
        )
        self.console.print(panel)
        task_object = task_data.get("task_object")
        show_test = task_data.get("show_test", True)
        if task_object is not None:
            try:
                self.console.print("\n[bold cyan]Task Examples:[/bold cyan]")
                visualize_parsed_task_data_rich(task_object, show_test=show_test)
            except Exception as e:
                logger.debug(f"Could not display task visualization: {e}")
                self.console.print("[dim]Task visualization unavailable[/dim]")

    def _display_episode_summary(self, summary_data: dict[str, Any]) -> None:
        episode_num = summary_data.get("episode_num", 0)
        total_steps = summary_data.get("total_steps", 0)
        total_reward = summary_data.get("total_reward", 0.0)
        final_similarity = summary_data.get("final_similarity", 0.0)
        success = summary_data.get("success", False)
        self.console.print(
            f"\n[bold magenta]Episode {episode_num} Summary[/bold magenta]"
        )
        self.console.rule()
        self.console.print(f"Total Steps: [bold]{total_steps}[/bold]")
        reward_color = (
            "green" if total_reward > 0 else "red" if total_reward < 0 else "yellow"
        )
        self.console.print(
            f"Total Reward: [{reward_color}]{total_reward:.3f}[/{reward_color}]"
        )
        similarity_color = (
            "green"
            if final_similarity > 0.8
            else "yellow"
            if final_similarity > 0.5
            else "red"
        )
        self.console.print(
            f"Final Similarity: [{similarity_color}]{final_similarity:.3f}[/{similarity_color}]"
        )
        success_color = "green" if success else "red"
        success_text = "SUCCESS" if success else "FAILED"
        self.console.print(f"Status: [{success_color}]{success_text}[/{success_color}]")
        task_id = summary_data.get("task_id")
        if task_id:
            self.console.print(f"Task ID: [bold]{task_id}[/bold]")
        self.console.rule()


# ============================================================================
# SECTION: WandB Integration (from wandb_handler.py)
# ============================================================================


class WandbHandler:
    """Simplified Weights & Biases integration handler."""

    def __init__(self, wandb_config):
        self.config = wandb_config
        self.run = None
        self._initialize_wandb()

    def _initialize_wandb(self) -> None:
        if not self.config.enabled:
            logger.info("Wandb integration disabled in config")
            return
        try:
            import wandb

            self._wandb = wandb
            if getattr(self.config, "offline_mode", False):
                os.environ["WANDB_MODE"] = "offline"
            project_name = str(getattr(self.config, "project_name", "jaxarc-test"))
            entity = getattr(self.config, "entity", None)
            if entity is not None:
                entity = str(entity)
            tags = getattr(self.config, "tags", [])
            if hasattr(tags, "__iter__") and not isinstance(tags, str):
                tags = [str(tag) for tag in tags]
            else:
                tags = [str(tags)] if tags else []
            notes = getattr(self.config, "notes", None)
            if notes is not None:
                notes = str(notes)
            group = getattr(self.config, "group", None)
            if group is not None:
                group = str(group)
            job_type = getattr(self.config, "job_type", None)
            if job_type is not None:
                job_type = str(job_type)
            self.run = self._wandb.init(
                project=project_name,
                entity=entity,
                tags=tags,
                notes=notes,
                group=group,
                job_type=job_type,
                save_code=getattr(self.config, "save_code", True),
            )
            if self.run:
                logger.info(f"Wandb run initialized: {self.run.name} ({self.run.id})")
        except ImportError:
            logger.warning("wandb not available - skipping wandb logging")
            self.run = None
        except Exception as e:
            logger.warning(f"wandb initialization failed: {e}")
            self.run = None

    def log_task_start(self, task_data: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            task_metrics = {}
            if "task_id" in task_data:
                task_metrics["task_id"] = task_data["task_id"]
            if "num_train_pairs" in task_data:
                task_metrics["task_num_train_pairs"] = task_data["num_train_pairs"]
            if "num_test_pairs" in task_data:
                task_metrics["task_num_test_pairs"] = task_data["num_test_pairs"]
            task_stats = task_data.get("task_stats", {})
            for key, value in task_stats.items():
                if isinstance(value, (int, float, bool)):
                    task_metrics[f"task_stat_{key}"] = value
            if task_metrics:
                self.run.log(task_metrics)
            logger.debug(
                f"Logged task start to wandb: {task_data.get('task_id', 'unknown')}"
            )
        except Exception as e:
            logger.warning(f"wandb task logging failed: {e}")

    def log_step(self, step_data: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            metrics = {}
            if (
                "info" in step_data
                and isinstance(step_data["info"], dict)
                and "metrics" in step_data["info"]
                and isinstance(step_data["info"]["metrics"], dict)
            ):
                metrics.update(step_data["info"]["metrics"])

            from .logger import to_python_scalar

            if "reward" in step_data:
                r_val = to_python_scalar(step_data["reward"])
                if isinstance(r_val, (int, float, bool)):
                    metrics["reward"] = (
                        float(r_val)
                        if not isinstance(r_val, bool)
                        else float(int(r_val))
                    )
                else:
                    metrics["reward"] = step_data["reward"]
            if "step_num" in step_data:
                s_val = to_python_scalar(step_data["step_num"])
                if isinstance(s_val, (int, bool)) or isinstance(s_val, float):
                    metrics["step"] = int(s_val)
                else:
                    metrics["step"] = step_data["step_num"]
            for key, value in step_data.items():
                if key in {"info", "before_state", "after_state", "action", "step_num"}:
                    continue
                v = to_python_scalar(value)
                if isinstance(v, (int, float, bool)):
                    metrics[key] = (
                        float(v) if not isinstance(v, bool) else float(int(v))
                    )
                elif isinstance(value, (int, float, bool)):
                    metrics[key] = (
                        float(value)
                        if not isinstance(value, bool)
                        else float(int(value))
                    )
            if metrics:
                self.run.log(metrics)
        except Exception as e:
            logger.warning(f"wandb step logging failed: {e}")

    def log_episode_summary(self, summary_data: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            episode_metrics = {}
            standard_keys = [
                "episode_num",
                "total_reward",
                "total_steps",
                "final_similarity",
                "success",
            ]
            for key in standard_keys:
                if key in summary_data:
                    episode_metrics[key] = summary_data[key]
            for key, value in summary_data.items():
                if key not in standard_keys and isinstance(value, (int, float, bool)):
                    episode_metrics[key] = value
            if episode_metrics:
                self.run.log(episode_metrics)
            if (
                "summary_svg_path" in summary_data
                and getattr(self, "_wandb", None) is not None
            ):
                try:
                    self.run.log(
                        {
                            "episode_summary": self._wandb.Image(
                                summary_data["summary_svg_path"]
                            )
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to log summary image: {e}")
        except Exception as e:
            logger.warning(f"wandb episode logging failed: {e}")

    def log_aggregated_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.run is None:
            return
        try:
            normalized: dict[str, float] = {}
            from .logger import to_python_scalar

            for key, value in metrics.items():
                v = to_python_scalar(value)
                if isinstance(v, (bool, int, float)):
                    normalized[key] = (
                        float(v) if not isinstance(v, bool) else float(int(v))
                    )
            if "gradient_norm" in normalized and "grad_norm" not in normalized:
                normalized["grad_norm"] = normalized["gradient_norm"]
            batch_metrics = {f"batch/{key}": value for key, value in normalized.items()}
            self.run.log(batch_metrics, step=step)
            logger.debug(
                f"Logged {len(batch_metrics)} batch metrics to wandb at step {step}"
            )
        except Exception as e:
            logger.warning(f"Wandb batch logging failed: {e}")

    def log_evaluation_summary(self, eval_data: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            summary_metrics = {}
            for k in [
                "task_id",
                "success_rate",
                "average_episode_length",
                "num_timeouts",
            ]:
                if k in eval_data and isinstance(eval_data[k], (int, float)):
                    summary_metrics[f"eval/{k}"] = eval_data[k]
            if "test_results" in eval_data and isinstance(
                eval_data["test_results"],
                list,
            ):
                summary_metrics["eval/num_test_results"] = len(
                    eval_data["test_results"]
                )
            if summary_metrics:
                self.run.log(summary_metrics)
                for k, v in summary_metrics.items():
                    with suppress(Exception):
                        self.run.summary[k] = v
        except Exception as e:
            logger.warning(f"wandb evaluation summary logging failed: {e}")

    def close(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
                logger.info("Wandb run finished successfully")
            except Exception as e:
                logger.warning(f"wandb finish failed: {e}")
            finally:
                self.run = None
