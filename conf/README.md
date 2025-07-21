# JaxARC Configuration Guide

This directory contains Hydra configuration files for JaxARC. The configuration
is organized into logical groups that correspond to the Python configuration
classes.

## Configuration Groups

### Dataset (`dataset/`)

- `arc_agi_1.yaml` - ARC-AGI-1 dataset configuration
- `arc_agi_2.yaml` - ARC-AGI-2 dataset configuration
- `concept_arc.yaml` - ConceptARC dataset configuration
- `mini_arc.yaml` - MiniARC dataset configuration

### Environment (`environment/`)

- `training.yaml` - Training environment (permissive, shorter episodes)
- `evaluation.yaml` - Evaluation environment (strict, longer episodes)

### Action (`action/`)

- `standard.yaml` - Balanced action set (fill, flood fill, clipboard, grid ops)
- `full.yaml` - Complete action set (all 35 operations)
- `raw.yaml` - Minimal action set (fill colors, resize, submit only)

### Reward (`reward/`)

- `standard.yaml` - Balanced reward structure
- `training.yaml` - Training-optimized rewards
- `evaluation.yaml` - Evaluation-focused rewards

### Logging (`logging/`)

- `basic.yaml` - Basic logging with moderate detail
- `full.yaml` - Comprehensive logging for research

### Storage (`storage/`)

- `development.yaml` - Development-friendly storage (small limits, quick
  cleanup)
- `production.yaml` - Production-safe storage (balanced limits, regular cleanup)
- `research.yaml` - Research storage (large limits, manual cleanup)

### Visualization (`visualization/`)

- `off.yaml` - No visualization (maximum performance)
- `minimal.yaml` - Episode summaries only
- `standard.yaml` - Balanced visualization for development
- `full.yaml` - Complete visualization for research/debugging

### WandB (`wandb/`)

- `disabled.yaml` - WandB integration disabled
- `basic.yaml` - Basic WandB logging
- `research.yaml` - Full WandB integration for research

## Usage Examples

### Basic Development Setup

```bash
python your_script.py
# Uses defaults: training environment, standard actions, local logging, standard visualization
```

### Research Configuration

```bash
python your_script.py \
  environment=evaluation \
  logging=full \
  storage=research \
  visualization=full \
  wandb=research
```

### Production Configuration

```bash
python your_script.py \
  environment=evaluation \
  logging=basic \
  storage=production \
  visualization=minimal \
  wandb=disabled
```

### Custom Overrides

```bash
python your_script.py \
  environment.debug_level=verbose \
  action.selection_format=bbox \
  logging.log_operations=true
```

## Configuration Structure

Each configuration group corresponds to a Python configuration class:

- `environment/` → `EnvironmentConfig`
- `dataset/` → `DatasetConfig`
- `action/` → `ActionConfig`
- `reward/` → `RewardConfig`
- `logging/` → `LoggingConfig`
- `storage/` → `StorageConfig`
- `visualization/` → `VisualizationConfig`
- `wandb/` → `WandbConfig`

The main `JaxArcConfig` class combines all these configurations into a unified,
typed configuration object that supports YAML serialization and validation.
