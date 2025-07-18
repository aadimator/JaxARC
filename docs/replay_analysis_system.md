# Episode Replay and Analysis System

The JaxARC episode replay and analysis system provides comprehensive tools for analyzing training episodes, identifying failure modes, and debugging agent behavior.

## Overview

The system consists of two main components:

1. **Episode Replay System** (`EpisodeReplaySystem`): Loads and replays episodes from structured logs
2. **Analysis Tools** (`EpisodeAnalysisTools`): Provides statistical analysis and debugging capabilities

## Quick Start

```python
from jaxarc.utils.logging.structured_logger import StructuredLogger, LoggingConfig
from jaxarc.utils.visualization.replay_system import EpisodeReplaySystem, ReplayConfig
from jaxarc.utils.visualization.analysis_tools import EpisodeAnalysisTools, AnalysisConfig

# Initialize structured logger (should be done during training)
logging_config = LoggingConfig(
    output_dir="outputs/logs",
    include_full_states=False  # Set to True for full state reconstruction
)
structured_logger = StructuredLogger(logging_config)

# Initialize replay system
replay_config = ReplayConfig(
    output_dir="outputs/replay",
    validate_integrity=True,
    regenerate_visualizations=True
)
replay_system = EpisodeReplaySystem(structured_logger, replay_config)

# Initialize analysis tools
analysis_config = AnalysisConfig(
    output_dir="outputs/analysis",
    generate_plots=True,
    failure_threshold=0.1,
    success_threshold=0.9
)
analysis_tools = EpisodeAnalysisTools(replay_system, analysis_config)
```

## Episode Replay Features

### Loading Episodes

```python
# List available episodes
episodes = replay_system.list_available_episodes()
print(f"Available episodes: {episodes}")

# Load specific episode
episode = replay_system.load_episode(episode_num=1)

# Get episode summaries
summaries = replay_system.get_episode_summaries([1, 2, 3])
```

### Episode Validation

```python
# Validate episode integrity
validation = replay_system.validate_episode_integrity(episode)
if not validation.is_valid:
    print(f"Validation errors: {validation.errors}")
    print(f"Warnings: {validation.warnings}")
```

### Episode Filtering

```python
# Find episodes by criteria
successful_episodes = replay_system.find_episodes_by_criteria(
    min_similarity=0.8,
    min_reward=0.5
)

failed_episodes = replay_system.find_episodes_by_criteria(
    max_similarity=0.2,
    task_id="specific_task"
)
```

### Full Episode Replay

```python
# Replay episode with validation and visualization regeneration
replay_result = replay_system.replay_episode(
    episode_num=1,
    validate=True,
    regenerate_visualizations=True
)

if replay_result:
    print(f"Replay successful: {replay_result['total_steps']} steps")
    print(f"Visualizations: {len(replay_result['visualization_paths'])}")
```

## Analysis Features

### Performance Metrics

```python
# Analyze overall performance
metrics = analysis_tools.analyze_performance_metrics()

print(f"Success rate: {metrics.success_rate:.1%}")
print(f"Average reward: {metrics.average_reward:.3f}")
print(f"Average similarity: {metrics.average_similarity:.3f}")
print(f"Best episode: {metrics.best_episode}")
```

### Failure Mode Analysis

```python
# Analyze failure patterns
failure_analysis = analysis_tools.analyze_failure_modes()

print(f"Total failures: {len(failure_analysis.failure_episodes)}")
print(f"Common patterns: {failure_analysis.common_failure_patterns}")
print(f"Failure distribution: {failure_analysis.failure_step_distribution}")
```

### Episode Comparison

```python
# Compare specific episodes
comparison = analysis_tools.compare_episodes(
    episode_nums=[1, 2, 3],
    metrics=['reward', 'similarity', 'steps']
)

for metric, data in comparison['metrics'].items():
    print(f"{metric}: best={data['best_episode']}, range={data['range']}")
```

### Step-by-Step Analysis

```python
# Detailed step analysis for debugging
step_analysis = analysis_tools.generate_step_by_step_analysis(
    episode_num=1,
    focus_on_failures=True
)

print(f"Key moments: {step_analysis['key_moments']}")
print(f"Potential issues: {step_analysis['potential_issues']}")
```

### Comprehensive Reports

```python
# Generate complete analysis report
report_path = analysis_tools.export_analysis_report(
    episode_nums=None,  # All episodes
    include_plots=True
)
print(f"Report saved to: {report_path}")
```

## Configuration Options

### ReplayConfig

- `validate_integrity`: Validate episode data integrity
- `regenerate_visualizations`: Regenerate step visualizations
- `output_dir`: Directory for replay outputs
- `max_episodes_to_load`: Limit on episodes to process
- `comparison_metrics`: Metrics for episode comparison

### AnalysisConfig

- `generate_plots`: Create performance plots
- `plot_format`: Output format for plots ("png", "svg", "both")
- `include_step_analysis`: Include detailed step analysis
- `failure_threshold`: Similarity threshold for failure classification
- `success_threshold`: Similarity threshold for success classification
- `max_episodes_per_analysis`: Limit on episodes per analysis

## Output Structure

The system creates organized output directories:

```
outputs/
├── logs/                    # Structured episode logs
│   ├── episode_0001/
│   │   └── episode_0001_timestamp.json.gz
│   └── episode_0002/
│       └── episode_0002_timestamp.json.gz
├── replay/                  # Episode replay outputs
│   ├── episode_0001_replay/
│   │   ├── replay_summary.json
│   │   └── step_*.svg       # Regenerated visualizations
│   └── episode_0002_replay/
└── analysis/                # Analysis results
    ├── performance_metrics.json
    ├── failure_mode_analysis.json
    ├── episode_comparison_*.json
    ├── step_analysis_episode_*.json
    ├── comprehensive_analysis_report.json
    └── *.png                # Performance plots
```

## Integration with Training

The replay system integrates seamlessly with the structured logging system:

```python
# During training
structured_logger.start_episode(episode_num, task_id, config_hash)

for step in training_loop:
    structured_logger.log_step(
        step_num=step,
        before_state=state,
        action=action,
        after_state=new_state,
        reward=reward,
        info=info,
        visualization_path=viz_path  # Optional
    )

structured_logger.end_episode(summary_visualization_path)

# After training - analysis
replay_system = EpisodeReplaySystem(structured_logger, replay_config)
analysis_tools = EpisodeAnalysisTools(replay_system, analysis_config)

# Analyze all episodes
metrics = analysis_tools.analyze_performance_metrics()
failure_analysis = analysis_tools.analyze_failure_modes()
report_path = analysis_tools.export_analysis_report()
```

## Best Practices

1. **Enable Full State Logging**: Set `include_full_states=True` in `LoggingConfig` for complete replay capability
2. **Regular Analysis**: Run analysis after training batches to identify issues early
3. **Filter Episodes**: Use filtering to focus analysis on specific scenarios
4. **Validate Data**: Always validate episode integrity before analysis
5. **Export Reports**: Generate comprehensive reports for documentation and sharing
6. **Monitor Failure Modes**: Track common failure patterns to guide training improvements

## Example Usage

See `examples/replay_analysis_demo.py` for a complete working example demonstrating all features of the replay and analysis system.