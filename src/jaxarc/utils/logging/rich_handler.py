"""Rich console output handler for JaxARC logging system."""

from __future__ import annotations

from typing import Any

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..visualization.rich_display import (
    visualize_grid_rich,
    visualize_task_pair_rich,
    visualize_parsed_task_data_rich,
)


class RichHandler:
    def __init__(self, config: Any):
        self.config = config
        self.console = Console()

    def _get_debug_level(self) -> str:
        if hasattr(self.config, 'environment') and hasattr(self.config.environment, 'debug_level'):
            return self.config.environment.debug_level
        return getattr(self.config, 'debug_level', 'standard')

    def log_task_start(self, task_data: dict[str, Any]) -> None:
        if self._get_debug_level() in ["standard", "verbose", "research"]:
            self._display_task_info(task_data)

    def log_step(self, step_data: dict[str, Any]) -> None:
        if self._get_debug_level() in ["verbose", "research"]:
            self._display_step_info(step_data)

    def log_episode_summary(self, summary_data: dict[str, Any]) -> None:
        self._display_episode_summary(summary_data)

    def log_aggregated_metrics(self, metrics: dict[str, float], step: int) -> None:
        try:
            table = Table(title=f"Batch Metrics - Step {step}", title_style="bold cyan")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="green", justify="right")

            reward_metrics = {k: v for k, v in metrics.items() if k.startswith('reward_')}
            similarity_metrics = {k: v for k, v in metrics.items() if k.startswith('similarity_')}
            episode_metrics = {k: v for k, v in metrics.items() if k.startswith('episode_length_')}
            training_metrics = {k: v for k, v in metrics.items() if k in ['policy_loss', 'value_loss', 'gradient_norm']}
            other_metrics = {k: v for k, v in metrics.items() if k not in reward_metrics and k not in similarity_metrics and k not in episode_metrics and k not in training_metrics}

            categories = [
                ("Rewards", reward_metrics),
                ("Similarity", similarity_metrics),
                ("Episode Length", episode_metrics),
                ("Training", training_metrics),
                ("Other", other_metrics)
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
                        table.add_row(f"  {key.replace('_',' ').title()}", formatted_value)
            self.console.print(table)
        except Exception as e:
            logger.warning(f"Rich batch logging failed: {e}")

    def log_evaluation_summary(self, eval_data: dict[str, Any]) -> None:
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
                    table.add_row(k.replace('_',' ').title(), disp)
            trs = eval_data.get('test_results')
            if isinstance(trs, list):
                table.add_row("Test Results", str(len(trs)))
            self.console.print(Panel.fit(table, border_style="magenta"))
        except Exception as e:
            logger.warning(f"Rich evaluation summary logging failed: {e}")

    def close(self) -> None:
        return None

    def _display_step_info(self, step_data: dict[str, Any]) -> None:
        step_num = step_data.get('step_num', 0)
        reward = step_data.get('reward', 0.0)
        info = step_data.get('info', {})
        self.console.print(f"\n[bold blue]Step {step_num}[/bold blue]")
        reward_color = "green" if reward > 0 else "red" if reward < 0 else "yellow"
        self.console.print(f"Reward: [{reward_color}]{reward:.3f}[/{reward_color}]")
        before_state = step_data.get('before_state')
        after_state = step_data.get('after_state')
        if before_state is not None and hasattr(before_state, 'grid'):
            if after_state is not None and hasattr(after_state, 'grid'):
                visualize_task_pair_rich(
                    input_grid=before_state.grid,
                    output_grid=after_state.grid,
                    title=f"Step {step_num}",
                    console=self.console
                )
            else:
                grid_table = visualize_grid_rich(before_state.grid, title=f"Step {step_num} - Before", border_style="input")
                self.console.print(grid_table)
        if 'metrics' in info and isinstance(info['metrics'], dict):
            self.console.print("[bold]Metrics:[/bold]")
            for k, v in info['metrics'].items():
                if isinstance(v, (int, float)):
                    self.console.print(f"  {k}: {v:.3f}")
                else:
                    self.console.print(f"  {k}: {v}")
        for key in ('success', 'similarity_improvement'):
            if key in info:
                v = info[key]
                if isinstance(v, (int, float)):
                    self.console.print(f"[bold]{key}:[/bold] {v:.3f}")
                else:
                    self.console.print(f"[bold]{key}:[/bold] {v}")
        action = step_data.get('action')
        if action is not None:
            self.console.print(f"[bold]Action:[/bold] {action}")

    def _display_task_info(self, task_data: dict[str, Any]) -> None:
        task_id = task_data.get('task_id', 'Unknown')
        episode_num = task_data.get('episode_num', 0)
        num_train_pairs = task_data.get('num_train_pairs', 0)
        num_test_pairs = task_data.get('num_test_pairs', 0)
        task_info = Text(justify="center")
        task_info.append("Training Examples: ", style="bold")
        task_info.append(str(num_train_pairs), style="green")
        task_info.append("  ")
        task_info.append("Test Examples: ", style="bold")
        task_info.append(str(num_test_pairs), style="blue")
        panel = Panel(task_info, title=f"Episode {episode_num} - Task: {task_id}", title_align="left", border_style="bright_blue", padding=(0,1))
        self.console.print(panel)
        task_object = task_data.get('task_object')
        show_test = task_data.get('show_test', True)
        if task_object is not None:
            try:
                self.console.print("\n[bold cyan]Task Examples:[/bold cyan]")
                visualize_parsed_task_data_rich(task_object, show_test=show_test)
            except Exception as e:
                logger.debug(f"Could not display task visualization: {e}")
                self.console.print("[dim]Task visualization unavailable[/dim]")

    def _display_episode_summary(self, summary_data: dict[str, Any]) -> None:
        episode_num = summary_data.get('episode_num', 0)
        total_steps = summary_data.get('total_steps', 0)
        total_reward = summary_data.get('total_reward', 0.0)
        final_similarity = summary_data.get('final_similarity', 0.0)
        success = summary_data.get('success', False)
        self.console.print(f"\n[bold magenta]Episode {episode_num} Summary[/bold magenta]")
        self.console.rule()
        self.console.print(f"Total Steps: [bold]{total_steps}[/bold]")
        reward_color = "green" if total_reward > 0 else "red" if total_reward < 0 else "yellow"
        self.console.print(f"Total Reward: [{reward_color}]{total_reward:.3f}[/{reward_color}]")
        similarity_color = "green" if final_similarity > 0.8 else "yellow" if final_similarity > 0.5 else "red"
        self.console.print(f"Final Similarity: [{similarity_color}]{final_similarity:.3f}[/{similarity_color}]")
        success_color = "green" if success else "red"
        success_text = "SUCCESS" if success else "FAILED"
        self.console.print(f"Status: [{success_color}]{success_text}[/{success_color}]")
        task_id = summary_data.get('task_id')
        if task_id:
            self.console.print(f"Task ID: [bold]{task_id}[/bold]")
        self.console.rule()