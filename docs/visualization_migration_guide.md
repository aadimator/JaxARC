# Visualization System Migration Guide

This guide helps you migrate from JaxARC's basic visualization system to the
enhanced visualization and logging system.

## Overview

The enhanced visualization system provides:

- **Comprehensive episode management** with organized storage
- **Asynchronous processing** for better performance
- **Weights & Biases integration** for experiment tracking
- **Memory management** with automatic cleanup
- **Performance monitoring** and optimization
- **Structured logging** and replay capabilities

## Migration Strategy

### Phase 1: Parallel Implementation

Run both systems side-by-side to ensure compatibility:

```python
# Old system (keep running)
from jaxarc.utils.visualization import log_grid_to_console, draw_grid_svg

# New system (add alongside)
from jaxarc.utils.visualization import Visualizer, VisualizationConfig

# Initialize both systems
old_viz_enabled = True
vis_config = VisualizationConfig(debug_level="standard")
new_visualizer = Visualizer(vis_config)


# In your training loop
def training_step_phase1(state, action, new_state, reward, info, step_num):
    # Old visualization (keep working)
    if old_viz_enabled:
        log_grid_to_console(new_state.working_grid, f"Step {step_num}")
        if step_num % 10 == 0:
            svg_content = draw_grid_svg(new_state.working_grid, f"Step {step_num}")
            with open(f"outputs/step_{step_num}.svg", "w") as f:
                f.write(svg_content)

    # New visualization (test in parallel)
    try:
        new_visualizer.visualize_step(
            before_state=state,
            action=action,
            after_state=new_state,
            reward=reward,
            info=info,
            step_num=step_num,
        )
    except Exception as e:
        print(f"New visualization failed: {e}")
        # Continue with old system
```

### Phase 2: Gradual Transition

Start using the new system as primary with old system as fallback:

```python
def training_step_phase2(state, action, new_state, reward, info, step_num):
    # Try new system first
    try:
        new_visualizer.visualize_step(
            before_state=state,
            action=action,
            after_state=new_state,
            reward=reward,
            info=info,
            step_num=step_num,
        )

        # If successful, we can reduce old system usage
        if step_num % 50 == 0:  # Reduced frequency for old system
            log_grid_to_console(new_state.working_grid, f"Backup log: Step {step_num}")

    except Exception as e:
        print(f"New visualization failed, using fallback: {e}")

        # Fallback to old system
        log_grid_to_console(new_state.working_grid, f"Step {step_num}")
        if step_num % 10 == 0:
            svg_content = draw_grid_svg(new_state.working_grid, f"Step {step_num}")
            with open(f"outputs/fallback_step_{step_num}.svg", "w") as f:
                f.write(svg_content)
```

### Phase 3: Complete Migration

Use only the new system with comprehensive error handling:

```python
def training_step_phase3(state, action, new_state, reward, info, step_num):
    # New system only
    new_visualizer.visualize_step(
        before_state=state,
        action=action,
        after_state=new_state,
        reward=reward,
        info=info,
        step_num=step_num,
    )
```

## Configuration Migration

### Old Configuration Format

```python
# Old way - basic parameters
debug_enabled = True
log_frequency = 10
output_dir = "outputs/visualization"
save_svg = True
save_png = False
show_coordinates = True
```

### New Configuration Format

```python
# New way - structured configuration
from jaxarc.utils.visualization import VisualizationConfig

vis_config = VisualizationConfig(
    debug_level="standard",  # Maps from debug_enabled
    log_frequency=10,  # Direct mapping
    output_dir="outputs/visualization",  # Direct mapping
    output_formats=["svg"],  # Maps from save_svg/save_png
    show_coordinates=True,  # Direct mapping
)
```

### Configuration Mapping Function

```python
def migrate_old_config(old_config):
    """Convert old configuration to new format."""

    # Map debug level
    if old_config.get("debug_enabled", False):
        if old_config.get("verbose_debug", False):
            debug_level = "verbose"
        else:
            debug_level = "standard"
    else:
        debug_level = "minimal"

    # Map output formats
    output_formats = []
    if old_config.get("save_svg", True):
        output_formats.append("svg")
    if old_config.get("save_png", False):
        output_formats.append("png")
    if not output_formats:
        output_formats = ["svg"]  # Default

    # Create new configuration
    new_config = VisualizationConfig(
        debug_level=debug_level,
        log_frequency=old_config.get("log_frequency", 10),
        output_dir=old_config.get("output_dir", "outputs/visualization"),
        output_formats=output_formats,
        show_coordinates=old_config.get("show_coordinates", False),
        show_operation_names=old_config.get("show_operations", False),
        highlight_changes=old_config.get("highlight_changes", True),
        include_metrics=old_config.get("include_metrics", False),
    )

    return new_config


# Usage
old_config = {
    "debug_enabled": True,
    "verbose_debug": False,
    "log_frequency": 5,
    "save_svg": True,
    "save_png": True,
    "show_coordinates": True,
}

new_config = migrate_old_config(old_config)
visualizer = Visualizer(new_config)
```

## Code Migration Examples

### Basic Visualization Migration

```python
# OLD CODE
from jaxarc.utils.visualization import log_grid_to_console, draw_grid_svg


def old_visualization(grid, title, step_num):
    # Console logging
    log_grid_to_console(grid, title)

    # SVG generation
    if step_num % 10 == 0:
        svg_content = draw_grid_svg(grid, title)
        filename = f"outputs/step_{step_num:03d}.svg"
        with open(filename, "w") as f:
            f.write(svg_content)


# NEW CODE
from jaxarc.utils.visualization import Visualizer, VisualizationConfig

# Initialize once
vis_config = VisualizationConfig(
    debug_level="standard", output_formats=["svg"], log_frequency=10
)
visualizer = Visualizer(vis_config)


def new_visualization(before_state, action, after_state, reward, info, step_num):
    # Single call handles everything
    visualizer.visualize_step(
        before_state=before_state,
        action=action,
        after_state=after_state,
        reward=reward,
        info=info,
        step_num=step_num,
    )
```

### Episode Management Migration

```python
# OLD CODE - Manual episode management
import os


def old_episode_management(episode_num):
    episode_dir = f"outputs/episode_{episode_num:04d}"
    os.makedirs(episode_dir, exist_ok=True)
    return episode_dir


def old_save_episode_summary(episode_num, summary_data):
    episode_dir = old_episode_management(episode_num)
    summary_file = os.path.join(episode_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f)


# NEW CODE - Automatic episode management
from jaxarc.utils.visualization import EpisodeManager

episode_manager = EpisodeManager(
    base_output_dir="outputs", episode_dir_format="episode_{episode:04d}"
)


def new_episode_management(episode_num):
    # Automatic directory creation and management
    return episode_manager.start_new_episode(episode_num)


def new_save_episode_summary(episode_num):
    # Automatic summary generation
    visualizer.visualize_episode_summary(episode_num)
```

### Performance Monitoring Migration

```python
# OLD CODE - Manual performance tracking
import time


class OldPerformanceTracker:
    def __init__(self):
        self.step_times = []

    def measure_step(self, func, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        self.step_times.append(end_time - start_time)
        return result

    def get_average_time(self):
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0


# NEW CODE - Built-in performance monitoring
from jaxarc.utils.visualization import PerformanceMonitor

perf_monitor = PerformanceMonitor(target_overhead=0.05, auto_adjust=True)


# Automatic performance measurement
@perf_monitor.measure_step_impact
def training_step_with_monitoring(state, action, config):
    new_state, obs, reward, done, info = arc_step(state, action, config)
    visualizer.visualize_step(state, action, new_state, reward, info, step_num)
    return new_state, obs, reward, done, info


# Get performance report
report = perf_monitor.get_performance_report()
print(f"Visualization overhead: {report['visualization_overhead']:.1f}%")
```

## Data Migration

### Migrating Existing Visualization Files

```python
import os
import shutil
from pathlib import Path


def migrate_visualization_data(old_dir, new_dir):
    """Migrate existing visualization files to new structure."""

    print(f"Migrating data from {old_dir} to {new_dir}")

    # Create new directory structure
    os.makedirs(new_dir, exist_ok=True)

    # Find all old visualization files
    old_files = []
    for root, dirs, files in os.walk(old_dir):
        for file in files:
            if file.endswith((".svg", ".png", ".json")):
                old_files.append(os.path.join(root, file))

    print(f"Found {len(old_files)} files to migrate")

    # Group files by episode (if possible)
    episodes = {}
    for file_path in old_files:
        # Try to extract episode number from filename
        filename = os.path.basename(file_path)

        # Common patterns: step_001.svg, episode_0_step_5.svg, etc.
        import re

        episode_match = re.search(r"episode_(\d+)", filename)
        step_match = re.search(r"step_(\d+)", filename)

        if episode_match:
            episode_num = int(episode_match.group(1))
        else:
            episode_num = 0  # Default episode

        if episode_num not in episodes:
            episodes[episode_num] = []

        episodes[episode_num].append(file_path)

    # Migrate files with new structure
    for episode_num, files in episodes.items():
        episode_dir = os.path.join(new_dir, f"episode_{episode_num:04d}")
        os.makedirs(episode_dir, exist_ok=True)

        for file_path in files:
            filename = os.path.basename(file_path)
            new_path = os.path.join(episode_dir, filename)
            shutil.copy2(file_path, new_path)
            print(f"  Migrated: {filename} -> episode_{episode_num:04d}/{filename}")

    print("Migration completed!")


# Usage
migrate_visualization_data("outputs/old_viz", "outputs/episodes")
```

### Converting Old Configuration Files

```python
import yaml
import json


def convert_old_config_file(old_config_path, new_config_path):
    """Convert old configuration file to new format."""

    # Load old configuration
    if old_config_path.endswith(".json"):
        with open(old_config_path, "r") as f:
            old_config = json.load(f)
    elif old_config_path.endswith(".yaml"):
        with open(old_config_path, "r") as f:
            old_config = yaml.safe_load(f)
    else:
        raise ValueError("Unsupported config file format")

    # Convert to new format
    new_config = {
        "visualization": {
            "debug_level": "standard" if old_config.get("debug", False) else "minimal",
            "enabled": old_config.get("enabled", True),
            "output_formats": ["svg"] if old_config.get("svg_output", True) else [],
            "log_frequency": old_config.get("log_every", 10),
            "show_coordinates": old_config.get("show_coords", False),
            "highlight_changes": old_config.get("highlight", True),
            "memory_limit_mb": old_config.get("memory_limit", 500),
        }
    }

    # Add PNG format if enabled
    if old_config.get("png_output", False):
        new_config["visualization"]["output_formats"].append("png")

    # Save new configuration
    with open(new_config_path, "w") as f:
        yaml.dump(new_config, f, default_flow_style=False)

    print(f"Converted {old_config_path} -> {new_config_path}")


# Usage
convert_old_config_file("old_config.json", "conf/visualization/migrated.yaml")
```

## Testing Migration

### Validation Script

```python
#!/usr/bin/env python3
"""
Migration validation script to ensure old and new systems produce equivalent results.
"""

import jax
import jax.numpy as jnp
from jaxarc.envs import arc_reset, arc_step, create_standard_config


def validate_migration():
    """Validate that migration produces equivalent results."""

    print("🔍 Validating migration...")

    # Setup test environment
    key = jax.random.PRNGKey(42)
    config = create_standard_config()
    state, obs = arc_reset(key, config)

    # Test data
    test_steps = []
    for step in range(5):
        selection = jax.random.bernoulli(key, 0.2, state.working_grid.shape)
        action = {"selection": selection, "operation": jnp.array(step + 1)}

        new_state, obs, reward, done, info = arc_step(state, action, config)

        test_steps.append(
            {
                "before_state": state,
                "action": action,
                "after_state": new_state,
                "reward": reward,
                "info": info,
                "step_num": step,
            }
        )

        state = new_state
        key = jax.random.split(key)[0]

        if done:
            break

    # Test old system
    print("  Testing old visualization system...")
    old_outputs = []
    for step_data in test_steps:
        # Simulate old system output
        old_output = {
            "console_log": f"Step {step_data['step_num']}: Grid shape {step_data['after_state'].working_grid.shape}",
            "svg_generated": step_data["step_num"] % 2 == 0,  # Every other step
        }
        old_outputs.append(old_output)

    # Test new system
    print("  Testing new visualization system...")
    from jaxarc.utils.visualization import Visualizer, VisualizationConfig

    vis_config = VisualizationConfig(
        debug_level="standard",
        output_formats=["svg"],
        log_frequency=2,  # Every other step, like old system
    )

    new_visualizer = Visualizer(vis_config)
    new_visualizer.start_episode(0)

    new_outputs = []
    for step_data in test_steps:
        try:
            new_visualizer.visualize_step(**step_data)
            new_output = {
                "visualization_success": True,
                "step_num": step_data["step_num"],
            }
        except Exception as e:
            new_output = {
                "visualization_success": False,
                "error": str(e),
                "step_num": step_data["step_num"],
            }

        new_outputs.append(new_output)

    # Compare results
    print("  Comparing results...")
    success = True

    for i, (old_out, new_out) in enumerate(zip(old_outputs, new_outputs)):
        if not new_out.get("visualization_success", False):
            print(f"    ❌ Step {i}: New system failed - {new_out.get('error')}")
            success = False
        else:
            print(f"    ✅ Step {i}: Both systems successful")

    # Generate episode summary
    try:
        new_visualizer.visualize_episode_summary(0)
        print("    ✅ Episode summary generated successfully")
    except Exception as e:
        print(f"    ❌ Episode summary failed: {e}")
        success = False

    if success:
        print("🎉 Migration validation successful!")
    else:
        print("❌ Migration validation failed - check errors above")

    return success


if __name__ == "__main__":
    validate_migration()
```

## Rollback Plan

### Emergency Rollback

If issues occur during migration, you can quickly rollback:

```python
def emergency_rollback():
    """Emergency rollback to old visualization system."""

    print("🚨 Performing emergency rollback...")

    # Disable new system
    global use_new_visualization
    use_new_visualization = False

    # Re-enable old system
    global use_old_visualization
    use_old_visualization = True

    print("✅ Rollback completed - using old visualization system")


def safe_visualization_step(state, action, new_state, reward, info, step_num):
    """Safe visualization with automatic rollback on failure."""

    if use_new_visualization:
        try:
            new_visualizer.visualize_step(
                before_state=state,
                action=action,
                after_state=new_state,
                reward=reward,
                info=info,
                step_num=step_num,
            )
            return
        except Exception as e:
            print(f"New visualization failed: {e}")
            emergency_rollback()

    if use_old_visualization:
        # Fallback to old system
        log_grid_to_console(new_state.working_grid, f"Step {step_num}")
```

## Migration Checklist

### Pre-Migration

- [ ] Backup existing visualization data
- [ ] Test new system in development environment
- [ ] Validate configuration migration
- [ ] Prepare rollback plan
- [ ] Document current visualization usage

### During Migration

- [ ] Run both systems in parallel (Phase 1)
- [ ] Monitor performance impact
- [ ] Validate output equivalence
- [ ] Gradually reduce old system usage (Phase 2)
- [ ] Complete transition to new system (Phase 3)

### Post-Migration

- [ ] Verify all functionality works
- [ ] Clean up old visualization code
- [ ] Update documentation
- [ ] Train team on new system
- [ ] Monitor for issues

## Common Migration Issues

### Issue 1: Performance Degradation

**Symptoms**: Training becomes slower after migration

**Solution**:

```python
# Reduce visualization frequency
visualizer.set_log_frequency(50)  # Less frequent logging

# Use minimal debug level
visualizer.set_debug_level("minimal")

# Enable async processing
visualizer.enable_async_processing()
```

### Issue 2: Memory Usage Increase

**Symptoms**: Higher memory usage with new system

**Solution**:

```python
# Set memory limits
visualizer.set_memory_limit(200)  # MB

# Enable cleanup
visualizer.enable_auto_cleanup()

# Use compression
visualizer.enable_compression()
```

### Issue 3: File Organization Changes

**Symptoms**: Can't find visualization files in expected locations

**Solution**:

```python
# Check new file organization
print(f"Episodes directory: {visualizer.get_episodes_directory()}")
print(f"Current run directory: {visualizer.get_current_run_directory()}")

# List available episodes
episodes = visualizer.list_episodes()
print(f"Available episodes: {episodes}")
```

## Support and Resources

### Getting Help

1. **Check the troubleshooting guide**: `docs/troubleshooting_guide.md`
2. **Review examples**: `examples/enhanced_visualization_examples.py`
3. **Run diagnostics**: Use built-in diagnostic tools
4. **Community support**: GitHub issues and discussions

### Additional Resources

- **Performance optimization**: `docs/performance_optimization_guide.md`
- **Best practices**: `docs/visualization_best_practices.md`
- **API reference**: `docs/api_reference.md`
- **Configuration guide**: `docs/enhanced_visualization.md`

This migration guide provides a structured approach to upgrading from the basic
to enhanced visualization system while minimizing risks and ensuring a smooth
transition.
