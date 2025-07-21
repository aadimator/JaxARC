"""Analysis and debugging tools for episode data.

This module provides tools for analyzing episode patterns, identifying failure modes,
and performing comparative analysis across multiple episodes.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import chex
from loguru import logger

from ..logging.structured_logger import EpisodeLogEntry
from .replay_system import EpisodeReplaySystem


@chex.dataclass
class AnalysisConfig:
    """Configuration for episode analysis."""

    output_dir: str = "outputs/analysis"
    generate_plots: bool = True
    plot_format: str = "png"  # "png", "svg", "both"
    include_step_analysis: bool = True
    failure_threshold: float = 0.1  # Similarity threshold for failure
    success_threshold: float = 0.9  # Similarity threshold for success
    max_episodes_per_analysis: int = 1000


@chex.dataclass
class FailureModeAnalysis:
    """Analysis of failure modes in episodes."""

    failure_episodes: List[int]
    common_failure_patterns: Dict[str, int]
    failure_step_distribution: Dict[int, int]
    average_failure_similarity: float
    failure_task_distribution: Dict[str, int]


@chex.dataclass
class PerformanceMetrics:
    """Performance metrics across episodes."""

    total_episodes: int
    success_rate: float
    average_reward: float
    average_similarity: float
    average_steps: float
    reward_std: float
    similarity_std: float
    steps_std: float
    best_episode: int
    worst_episode: int


class EpisodeAnalysisTools:
    """Tools for analyzing and debugging episode data."""

    def __init__(self, replay_system: EpisodeReplaySystem, config: AnalysisConfig):
        """Initialize analysis tools.

        Args:
            replay_system: Replay system for accessing episode data
            config: Configuration for analysis behavior
        """
        self.replay_system = replay_system
        self.config = config

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"EpisodeAnalysisTools initialized with output dir: {self.output_dir}"
        )

    def analyze_performance_metrics(
        self, episode_nums: Optional[List[int]] = None
    ) -> PerformanceMetrics:
        """Analyze performance metrics across episodes.

        Args:
            episode_nums: Specific episodes to analyze, or None for all available

        Returns:
            Performance metrics summary
        """
        if episode_nums is None:
            episode_nums = self.replay_system.list_available_episodes()

        episode_nums = episode_nums[: self.config.max_episodes_per_analysis]
        summaries = self.replay_system.get_episode_summaries(episode_nums)

        if not summaries:
            logger.warning("No episode summaries available for analysis")
            return PerformanceMetrics(
                total_episodes=0,
                success_rate=0.0,
                average_reward=0.0,
                average_similarity=0.0,
                average_steps=0.0,
                reward_std=0.0,
                similarity_std=0.0,
                steps_std=0.0,
                best_episode=-1,
                worst_episode=-1,
            )

        # Extract metrics
        rewards = [s["total_reward"] for s in summaries]
        similarities = [s["final_similarity"] for s in summaries]
        steps = [s["total_steps"] for s in summaries]

        # Calculate success rate
        successes = sum(1 for s in similarities if s >= self.config.success_threshold)
        success_rate = successes / len(summaries)

        # Find best and worst episodes
        best_idx = max(
            range(len(summaries)), key=lambda i: summaries[i]["final_similarity"]
        )
        worst_idx = min(
            range(len(summaries)), key=lambda i: summaries[i]["final_similarity"]
        )

        metrics = PerformanceMetrics(
            total_episodes=len(summaries),
            success_rate=success_rate,
            average_reward=statistics.mean(rewards),
            average_similarity=statistics.mean(similarities),
            average_steps=statistics.mean(steps),
            reward_std=statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            similarity_std=statistics.stdev(similarities)
            if len(similarities) > 1
            else 0.0,
            steps_std=statistics.stdev(steps) if len(steps) > 1 else 0.0,
            best_episode=summaries[best_idx]["episode_num"],
            worst_episode=summaries[worst_idx]["episode_num"],
        )

        # Save metrics
        metrics_path = self.output_dir / "performance_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_episodes": metrics.total_episodes,
                    "success_rate": metrics.success_rate,
                    "average_reward": metrics.average_reward,
                    "average_similarity": metrics.average_similarity,
                    "average_steps": metrics.average_steps,
                    "reward_std": metrics.reward_std,
                    "similarity_std": metrics.similarity_std,
                    "steps_std": metrics.steps_std,
                    "best_episode": metrics.best_episode,
                    "worst_episode": metrics.worst_episode,
                },
                f,
                indent=2,
            )

        logger.info(
            f"Performance analysis complete: {metrics.success_rate:.1%} success rate"
        )
        return metrics

    def analyze_failure_modes(
        self, episode_nums: Optional[List[int]] = None
    ) -> FailureModeAnalysis:
        """Analyze common failure modes in episodes.

        Args:
            episode_nums: Specific episodes to analyze, or None for all available

        Returns:
            Failure mode analysis results
        """
        if episode_nums is None:
            episode_nums = self.replay_system.list_available_episodes()

        episode_nums = episode_nums[: self.config.max_episodes_per_analysis]
        summaries = self.replay_system.get_episode_summaries(episode_nums)

        # Identify failure episodes
        failure_episodes = [
            s["episode_num"]
            for s in summaries
            if s["final_similarity"] < self.config.failure_threshold
        ]

        if not failure_episodes:
            logger.info("No failure episodes found")
            return FailureModeAnalysis(
                failure_episodes=[],
                common_failure_patterns={},
                failure_step_distribution={},
                average_failure_similarity=0.0,
                failure_task_distribution={},
            )

        logger.info(f"Analyzing {len(failure_episodes)} failure episodes")

        # Analyze failure patterns
        failure_patterns = defaultdict(int)
        failure_step_distribution = defaultdict(int)
        failure_similarities = []
        failure_task_distribution = defaultdict(int)

        for episode_num in failure_episodes:
            episode = self.replay_system.load_episode(episode_num)
            if episode is None:
                continue

            failure_similarities.append(episode.final_similarity)
            failure_task_distribution[episode.task_id] += 1

            # Analyze step distribution
            step_range = self._get_step_range(episode.total_steps)
            failure_step_distribution[step_range] += 1

            # Analyze failure patterns from episode metadata and steps
            patterns = self._identify_failure_patterns(episode)
            for pattern in patterns:
                failure_patterns[pattern] += 1

        analysis = FailureModeAnalysis(
            failure_episodes=failure_episodes,
            common_failure_patterns=dict(failure_patterns),
            failure_step_distribution=dict(failure_step_distribution),
            average_failure_similarity=statistics.mean(failure_similarities)
            if failure_similarities
            else 0.0,
            failure_task_distribution=dict(failure_task_distribution),
        )

        # Save analysis
        analysis_path = self.output_dir / "failure_mode_analysis.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "failure_episodes": analysis.failure_episodes,
                    "common_failure_patterns": analysis.common_failure_patterns,
                    "failure_step_distribution": analysis.failure_step_distribution,
                    "average_failure_similarity": analysis.average_failure_similarity,
                    "failure_task_distribution": analysis.failure_task_distribution,
                },
                f,
                indent=2,
            )

        logger.info(
            f"Failure mode analysis complete: {len(failure_episodes)} failures analyzed"
        )
        return analysis

    def _get_step_range(self, steps: int) -> str:
        """Categorize step count into ranges.

        Args:
            steps: Number of steps

        Returns:
            Step range category
        """
        if steps <= 10:
            return "0-10"
        if steps <= 25:
            return "11-25"
        if steps <= 50:
            return "26-50"
        if steps <= 100:
            return "51-100"
        return "100+"

    def _identify_failure_patterns(self, episode: EpisodeLogEntry) -> List[str]:
        """Identify failure patterns in an episode.

        Args:
            episode: Episode to analyze

        Returns:
            List of identified failure patterns
        """
        patterns = []

        # Pattern: Early termination
        if episode.total_steps < 5:
            patterns.append("early_termination")

        # Pattern: No progress (similarity doesn't improve)
        if len(episode.steps) > 1:
            initial_similarity = episode.steps[0].info.get("similarity", 0.0)
            if episode.final_similarity <= initial_similarity + 0.01:
                patterns.append("no_progress")

        # Pattern: Negative reward accumulation
        if episode.total_reward < -1.0:
            patterns.append("negative_reward_spiral")

        # Pattern: Repetitive actions
        if self._has_repetitive_actions(episode):
            patterns.append("repetitive_actions")

        # Pattern: Low final similarity
        if episode.final_similarity < 0.05:
            patterns.append("very_low_similarity")

        return patterns

    def _has_repetitive_actions(self, episode: EpisodeLogEntry) -> bool:
        """Check if episode has repetitive action patterns.

        Args:
            episode: Episode to check

        Returns:
            True if repetitive patterns detected
        """
        if len(episode.steps) < 6:
            return False

        # Check for repeated operation sequences
        operations = [step.action.get("operation", -1) for step in episode.steps]

        # Look for sequences of 3+ identical operations
        consecutive_count = 1
        for i in range(1, len(operations)):
            if operations[i] == operations[i - 1]:
                consecutive_count += 1
                if consecutive_count >= 3:
                    return True
            else:
                consecutive_count = 1

        return False

    def compare_episodes(
        self, episode_nums: List[int], metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple episodes across specified metrics.

        Args:
            episode_nums: Episodes to compare
            metrics: Metrics to compare (uses config default if None)

        Returns:
            Comparison results
        """
        if metrics is None:
            metrics = self.config.comparison_metrics or [
                "reward",
                "similarity",
                "steps",
            ]

        summaries = []
        for episode_num in episode_nums:
            summary = self.replay_system.structured_logger.get_episode_summary(
                episode_num
            )
            if summary:
                summaries.append(summary)

        if not summaries:
            logger.warning("No valid episodes found for comparison")
            return {}

        comparison = {"episodes": [s["episode_num"] for s in summaries], "metrics": {}}

        # Compare each metric
        for metric in metrics:
            if metric == "reward":
                values = [s["total_reward"] for s in summaries]
                comparison["metrics"]["reward"] = {
                    "values": values,
                    "best_episode": summaries[values.index(max(values))]["episode_num"],
                    "worst_episode": summaries[values.index(min(values))][
                        "episode_num"
                    ],
                    "range": max(values) - min(values),
                }
            elif metric == "similarity":
                values = [s["final_similarity"] for s in summaries]
                comparison["metrics"]["similarity"] = {
                    "values": values,
                    "best_episode": summaries[values.index(max(values))]["episode_num"],
                    "worst_episode": summaries[values.index(min(values))][
                        "episode_num"
                    ],
                    "range": max(values) - min(values),
                }
            elif metric == "steps":
                values = [s["total_steps"] for s in summaries]
                comparison["metrics"]["steps"] = {
                    "values": values,
                    "most_steps": summaries[values.index(max(values))]["episode_num"],
                    "least_steps": summaries[values.index(min(values))]["episode_num"],
                    "range": max(values) - min(values),
                }

        # Save comparison
        comparison_path = (
            self.output_dir / f"episode_comparison_{len(episode_nums)}_episodes.json"
        )
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Episode comparison complete for {len(episode_nums)} episodes")
        return comparison

    def generate_step_by_step_analysis(
        self, episode_num: int, focus_on_failures: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Generate detailed step-by-step analysis for debugging.

        Args:
            episode_num: Episode to analyze
            focus_on_failures: Whether to highlight potential failure points

        Returns:
            Step-by-step analysis or None if episode not found
        """
        episode = self.replay_system.load_episode(episode_num)
        if episode is None:
            logger.error(f"Episode {episode_num} not found")
            return None

        logger.info(f"Generating step-by-step analysis for episode {episode_num}")

        analysis = {
            "episode_num": episode_num,
            "total_steps": episode.total_steps,
            "final_similarity": episode.final_similarity,
            "total_reward": episode.total_reward,
            "step_analysis": [],
            "key_moments": [],
            "potential_issues": [],
        }

        prev_similarity = 0.0
        prev_reward = 0.0

        for i, step in enumerate(episode.steps):
            step_similarity = step.info.get("similarity", 0.0)
            similarity_change = step_similarity - prev_similarity
            reward_change = step.reward - prev_reward if i > 0 else step.reward

            step_info = {
                "step_num": step.step_num,
                "operation": step.action.get("operation", "unknown"),
                "reward": step.reward,
                "reward_change": reward_change,
                "similarity": step_similarity,
                "similarity_change": similarity_change,
                "timestamp": step.timestamp,
            }

            # Identify key moments
            if abs(similarity_change) > 0.1:
                analysis["key_moments"].append(
                    {
                        "step": step.step_num,
                        "type": "similarity_jump"
                        if similarity_change > 0
                        else "similarity_drop",
                        "change": similarity_change,
                    }
                )

            if focus_on_failures:
                # Identify potential issues
                if step.reward < -0.5:
                    analysis["potential_issues"].append(
                        {
                            "step": step.step_num,
                            "issue": "large_negative_reward",
                            "value": step.reward,
                        }
                    )

                if similarity_change < -0.05:
                    analysis["potential_issues"].append(
                        {
                            "step": step.step_num,
                            "issue": "similarity_regression",
                            "change": similarity_change,
                        }
                    )

            analysis["step_analysis"].append(step_info)
            prev_similarity = step_similarity
            prev_reward = step.reward

        # Save analysis
        analysis_path = (
            self.output_dir / f"step_analysis_episode_{episode_num:04d}.json"
        )
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Step-by-step analysis saved: {analysis_path}")
        return analysis

    def generate_performance_plots(
        self, episode_nums: Optional[List[int]] = None
    ) -> List[str]:
        """Generate performance visualization plots.

        Args:
            episode_nums: Episodes to include in plots, or None for all

        Returns:
            List of generated plot file paths
        """
        if not self.config.generate_plots:
            return []

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping plot generation")
            return []

        if episode_nums is None:
            episode_nums = self.replay_system.list_available_episodes()

        summaries = self.replay_system.get_episode_summaries(episode_nums)
        if not summaries:
            logger.warning("No episode data available for plotting")
            return []

        plot_paths = []

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Plot 1: Reward vs Similarity scatter
        plt.figure(figsize=(10, 6))
        rewards = [s["total_reward"] for s in summaries]
        similarities = [s["final_similarity"] for s in summaries]

        plt.scatter(similarities, rewards, alpha=0.6)
        plt.xlabel("Final Similarity")
        plt.ylabel("Total Reward")
        plt.title("Episode Performance: Reward vs Similarity")
        plt.grid(True, alpha=0.3)

        plot_path = self.output_dir / f"reward_vs_similarity.{self.config.plot_format}"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths.append(str(plot_path))

        # Plot 2: Performance over time (episode number)
        plt.figure(figsize=(12, 8))

        episode_nums_sorted = [s["episode_num"] for s in summaries]
        rewards_sorted = [s["total_reward"] for s in summaries]
        similarities_sorted = [s["final_similarity"] for s in summaries]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(
            episode_nums_sorted, rewards_sorted, "b-", alpha=0.7, label="Total Reward"
        )
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Performance Over Episodes")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(
            episode_nums_sorted,
            similarities_sorted,
            "r-",
            alpha=0.7,
            label="Final Similarity",
        )
        ax2.set_xlabel("Episode Number")
        ax2.set_ylabel("Final Similarity")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plot_path = self.output_dir / f"performance_over_time.{self.config.plot_format}"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths.append(str(plot_path))

        # Plot 3: Step count distribution
        plt.figure(figsize=(10, 6))
        steps = [s["total_steps"] for s in summaries]

        plt.hist(steps, bins=20, alpha=0.7, edgecolor="black")
        plt.xlabel("Total Steps")
        plt.ylabel("Frequency")
        plt.title("Distribution of Episode Lengths")
        plt.grid(True, alpha=0.3)

        plot_path = self.output_dir / f"step_distribution.{self.config.plot_format}"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths.append(str(plot_path))

        logger.info(f"Generated {len(plot_paths)} performance plots")
        return plot_paths

    def export_analysis_report(
        self, episode_nums: Optional[List[int]] = None, include_plots: bool = True
    ) -> str:
        """Export comprehensive analysis report.

        Args:
            episode_nums: Episodes to include in report, or None for all
            include_plots: Whether to include performance plots

        Returns:
            Path to generated report file
        """
        if episode_nums is None:
            episode_nums = self.replay_system.list_available_episodes()

        logger.info(
            f"Generating comprehensive analysis report for {len(episode_nums)} episodes"
        )

        # Perform all analyses
        performance_metrics = self.analyze_performance_metrics(episode_nums)
        failure_analysis = self.analyze_failure_modes(episode_nums)

        # Generate plots if requested
        plot_paths = []
        if include_plots:
            plot_paths = self.generate_performance_plots(episode_nums)

        # Create comprehensive report
        report = {
            "analysis_timestamp": time.time(),
            "total_episodes_analyzed": len(episode_nums),
            "episode_range": f"{min(episode_nums)}-{max(episode_nums)}"
            if episode_nums
            else "none",
            "performance_metrics": {
                "total_episodes": performance_metrics.total_episodes,
                "success_rate": performance_metrics.success_rate,
                "average_reward": performance_metrics.average_reward,
                "average_similarity": performance_metrics.average_similarity,
                "average_steps": performance_metrics.average_steps,
                "best_episode": performance_metrics.best_episode,
                "worst_episode": performance_metrics.worst_episode,
            },
            "failure_analysis": {
                "total_failures": len(failure_analysis.failure_episodes),
                "failure_rate": len(failure_analysis.failure_episodes)
                / len(episode_nums)
                if episode_nums
                else 0,
                "common_patterns": failure_analysis.common_failure_patterns,
                "average_failure_similarity": failure_analysis.average_failure_similarity,
            },
            "generated_plots": plot_paths,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
            },
        }

        # Save report
        report_path = self.output_dir / "comprehensive_analysis_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Comprehensive analysis report saved: {report_path}")
        return str(report_path)
