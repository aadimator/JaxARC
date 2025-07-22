#!/usr/bin/env python3
"""
Demonstration of episode replay and analysis functionality.

This example shows how to use the replay system and analysis tools to:
1. Load and validate episode data from structured logs
2. Replay episodes with visualization regeneration
3. Analyze performance metrics and failure modes
4. Generate comparative analysis across episodes
5. Export comprehensive analysis reports

Usage:
    pixi run python examples/replay_analysis_demo.py
"""

from __future__ import annotations

from pathlib import Path

from jaxarc.utils.logging.structured_logger import (
    LoggingConfig,
    StructuredLogger,
)
from jaxarc.utils.visualization.analysis_tools import (
    AnalysisConfig,
    EpisodeAnalysisTools,
)
from jaxarc.utils.visualization.replay_system import (
    EpisodeReplaySystem,
    ReplayConfig,
)


def create_sample_episodes(logger: StructuredLogger) -> None:
    """Create sample episode data for demonstration."""
    print("Creating sample episode data...")

    # Episode 1: Successful episode
    logger.start_episode(1, "demo_task_001", "config_hash_1", {"demo": True})

    for step in range(5):
        logger.log_step(
            step_num=step,
            before_state={"type": "ArcEnvState", "step_count": step},
            action={"operation": 10 + step, "selection": [[1, 0], [0, 1]]},
            after_state={"type": "ArcEnvState", "step_count": step + 1},
            reward=0.2 + step * 0.1,
            info={"similarity": 0.1 + step * 0.2},
        )

    logger.end_episode()

    # Episode 2: Failure episode
    logger.start_episode(2, "demo_task_002", "config_hash_2", {"demo": True})

    for step in range(3):
        logger.log_step(
            step_num=step,
            before_state={"type": "ArcEnvState", "step_count": step},
            action={
                "operation": 10,
                "selection": [[0, 1], [1, 0]],
            },  # Repetitive action
            after_state={"type": "ArcEnvState", "step_count": step + 1},
            reward=-0.1 - step * 0.05,
            info={"similarity": 0.05},  # No improvement
        )

    logger.end_episode()

    # Episode 3: Medium performance episode
    logger.start_episode(3, "demo_task_001", "config_hash_1", {"demo": True})

    for step in range(8):
        logger.log_step(
            step_num=step,
            before_state={"type": "ArcEnvState", "step_count": step},
            action={"operation": 15 + (step % 3), "selection": [[1, 1], [0, 0]]},
            after_state={"type": "ArcEnvState", "step_count": step + 1},
            reward=0.05 + (step % 2) * 0.1,
            info={"similarity": 0.3 + step * 0.05},
        )

    logger.end_episode()

    print("Sample episodes created successfully!")


def demonstrate_replay_functionality(replay_system: EpisodeReplaySystem) -> None:
    """Demonstrate episode replay functionality."""
    print("\n" + "=" * 60)
    print("EPISODE REPLAY DEMONSTRATION")
    print("=" * 60)

    # List available episodes
    available_episodes = replay_system.list_available_episodes()
    print(f"Available episodes: {available_episodes}")

    # Get episode summaries
    summaries = replay_system.get_episode_summaries()
    print("\nEpisode summaries:")
    for summary in summaries:
        print(
            f"  Episode {summary['episode_num']}: "
            f"reward={summary['total_reward']:.3f}, "
            f"similarity={summary['final_similarity']:.3f}, "
            f"steps={summary['total_steps']}"
        )

    # Validate and replay each episode
    for episode_num in available_episodes:
        print(f"\n--- Replaying Episode {episode_num} ---")

        # Load episode for validation
        episode = replay_system.load_episode(episode_num)
        if episode is None:
            print(f"Failed to load episode {episode_num}")
            continue

        # Validate episode integrity
        validation = replay_system.validate_episode_integrity(episode)
        print(f"Validation result: {'VALID' if validation.is_valid else 'INVALID'}")
        if validation.errors:
            print(f"Errors: {validation.errors}")
        if validation.warnings:
            print(f"Warnings: {validation.warnings}")

        # Replay episode
        replay_result = replay_system.replay_episode(
            episode_num,
            validate=True,
            regenerate_visualizations=False,  # Skip visualization for demo
        )

        if replay_result:
            print(f"Replay successful: {replay_result['total_steps']} steps processed")
            print(f"Duration: {replay_result['duration']:.2f} seconds")
        else:
            print("Replay failed")

    # Demonstrate episode filtering
    print("\n--- Episode Filtering ---")

    # Find successful episodes
    successful_episodes = replay_system.find_episodes_by_criteria(min_similarity=0.8)
    print(f"Successful episodes (similarity >= 0.8): {successful_episodes}")

    # Find failed episodes
    failed_episodes = replay_system.find_episodes_by_criteria(max_similarity=0.2)
    print(f"Failed episodes (similarity <= 0.2): {failed_episodes}")

    # Find episodes by task
    task_episodes = replay_system.find_episodes_by_criteria(task_id="demo_task_001")
    print(f"Episodes for demo_task_001: {task_episodes}")


def demonstrate_analysis_functionality(analysis_tools: EpisodeAnalysisTools) -> None:
    """Demonstrate episode analysis functionality."""
    print("\n" + "=" * 60)
    print("EPISODE ANALYSIS DEMONSTRATION")
    print("=" * 60)

    # Analyze performance metrics
    print("--- Performance Metrics Analysis ---")
    metrics = analysis_tools.analyze_performance_metrics()

    print(f"Total episodes analyzed: {metrics.total_episodes}")
    print(f"Success rate: {metrics.success_rate:.1%}")
    print(f"Average reward: {metrics.average_reward:.3f} ± {metrics.reward_std:.3f}")
    print(
        f"Average similarity: {metrics.average_similarity:.3f} ± {metrics.similarity_std:.3f}"
    )
    print(f"Average steps: {metrics.average_steps:.1f} ± {metrics.steps_std:.1f}")
    print(f"Best episode: {metrics.best_episode}")
    print(f"Worst episode: {metrics.worst_episode}")

    # Analyze failure modes
    print("\n--- Failure Mode Analysis ---")
    failure_analysis = analysis_tools.analyze_failure_modes()

    print(f"Total failure episodes: {len(failure_analysis.failure_episodes)}")
    print(f"Failure episodes: {failure_analysis.failure_episodes}")
    print(
        f"Average failure similarity: {failure_analysis.average_failure_similarity:.3f}"
    )

    if failure_analysis.common_failure_patterns:
        print("Common failure patterns:")
        for pattern, count in failure_analysis.common_failure_patterns.items():
            print(f"  {pattern}: {count} episodes")

    if failure_analysis.failure_step_distribution:
        print("Failure step distribution:")
        for step_range, count in failure_analysis.failure_step_distribution.items():
            print(f"  {step_range} steps: {count} episodes")

    if failure_analysis.failure_task_distribution:
        print("Failure task distribution:")
        for task_id, count in failure_analysis.failure_task_distribution.items():
            print(f"  {task_id}: {count} failures")

    # Compare episodes
    print("\n--- Episode Comparison ---")
    available_episodes = analysis_tools.replay_system.list_available_episodes()
    if len(available_episodes) >= 2:
        comparison = analysis_tools.compare_episodes(
            available_episodes[:3],  # Compare first 3 episodes
            metrics=["reward", "similarity", "steps"],
        )

        print(f"Comparing episodes: {comparison['episodes']}")

        for metric, data in comparison["metrics"].items():
            print(f"\n{metric.capitalize()} comparison:")
            print(f"  Values: {[f'{v:.3f}' for v in data['values']]}")
            if "best_episode" in data:
                print(f"  Best: Episode {data['best_episode']}")
                print(f"  Worst: Episode {data['worst_episode']}")
            print(f"  Range: {data['range']:.3f}")

    # Generate step-by-step analysis for one episode
    print("\n--- Step-by-Step Analysis ---")
    if available_episodes:
        episode_num = available_episodes[0]
        step_analysis = analysis_tools.generate_step_by_step_analysis(episode_num)

        if step_analysis:
            print(f"Detailed analysis for Episode {episode_num}:")
            print(f"  Total steps: {step_analysis['total_steps']}")
            print(f"  Final similarity: {step_analysis['final_similarity']:.3f}")
            print(f"  Total reward: {step_analysis['total_reward']:.3f}")

            if step_analysis["key_moments"]:
                print("  Key moments:")
                for moment in step_analysis["key_moments"]:
                    print(
                        f"    Step {moment['step']}: {moment['type']} ({moment['change']:+.3f})"
                    )

            if step_analysis["potential_issues"]:
                print("  Potential issues:")
                for issue in step_analysis["potential_issues"]:
                    print(f"    Step {issue['step']}: {issue['issue']}")

    # Export comprehensive report
    print("\n--- Comprehensive Report Export ---")
    report_path = analysis_tools.export_analysis_report(
        episode_nums=available_episodes,
        include_plots=False,  # Skip plots for demo
    )
    print(f"Comprehensive analysis report saved to: {report_path}")


def main():
    """Main demonstration function."""
    print("Episode Replay and Analysis System Demo")
    print("=" * 60)

    # Setup temporary directories for demo
    demo_dir = Path("outputs/demo_replay_analysis")
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Initialize structured logger
    logging_config = LoggingConfig(
        output_dir=str(demo_dir / "logs"),
        structured_logging=True,
        include_full_states=False,  # Use minimal logging for demo
        async_logging=False,  # Synchronous for demo simplicity
    )

    structured_logger = StructuredLogger(logging_config)

    try:
        # Create sample episode data
        create_sample_episodes(structured_logger)

        # Initialize replay system
        replay_config = ReplayConfig(
            output_dir=str(demo_dir / "replay"),
            validate_integrity=True,
            regenerate_visualizations=False,  # Skip for demo
            max_episodes_to_load=10,
        )

        replay_system = EpisodeReplaySystem(structured_logger, replay_config)

        # Initialize analysis tools
        analysis_config = AnalysisConfig(
            output_dir=str(demo_dir / "analysis"),
            generate_plots=False,  # Skip plots for demo
            failure_threshold=0.2,
            success_threshold=0.8,
        )

        analysis_tools = EpisodeAnalysisTools(replay_system, analysis_config)

        # Demonstrate functionality
        demonstrate_replay_functionality(replay_system)
        demonstrate_analysis_functionality(analysis_tools)

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Demo files saved to: {demo_dir}")
        print("Check the following directories:")
        print(f"  - Logs: {demo_dir / 'logs'}")
        print(f"  - Replay: {demo_dir / 'replay'}")
        print(f"  - Analysis: {demo_dir / 'analysis'}")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        raise
    finally:
        # Cleanup
        structured_logger.shutdown()


if __name__ == "__main__":
    main()
