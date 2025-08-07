# Troubleshooting Guide

This guide covers common issues with JaxARC's enhanced visualization and logging
system, along with solutions and debugging techniques.

## Quick Diagnostics

### System Health Check

```python
from jaxarc.utils.visualization import run_system_diagnostics

# Run comprehensive system check
diagnostics = run_system_diagnostics()

print("🔍 System Diagnostics:")
print(f"JAX version: {diagnostics['jax_version']}")
print(f"Memory available: {diagnostics['memory_gb']:.1f} GB")
print(f"Disk space: {diagnostics['disk_space_gb']:.1f} GB")
print(
    f"Visualization system: {'✅ OK' if diagnostics['viz_system_ok'] else '❌ Error'}"
)
print(f"Async logging: {'✅ OK' if diagnostics['async_logging_ok'] else '❌ Error'}")
print(f"Wandb connection: {'✅ OK' if diagnostics['wandb_ok'] else '❌ Error'}")
```

### Quick Fix Commands

```python
# Reset visualization system
visualizer.reset_system()

# Clear all caches
visualizer.clear_all_caches()

# Force cleanup memory
visualizer.force_memory_cleanup()

# Restart async workers
visualizer.restart_async_workers()

# Test basic functionality
visualizer.run_self_test()
```

## Common Issues and Solutions

### 1. Performance Issues

#### High Visualization Overhead

**Symptoms:**

- Training significantly slower with visualization enabled
- High CPU usage during visualization
- Memory usage growing rapidly

**Diagnosis:**

```python
# Check performance impact
perf_report = visualizer.get_performance_report()
print(f"Overhead: {perf_report['visualization_overhead']:.1f}%")

if perf_report["visualization_overhead"] > 10:
    print("⚠️  High visualization overhead detected")
```

**Solutions:**

```python
# 1. Reduce debug level
visualizer.set_debug_level("minimal")

# 2. Decrease logging frequency
visualizer.set_log_frequency(50)  # Log every 50 steps

# 3. Enable async processing
visualizer.enable_async_processing()

# 4. Reduce image quality
visualizer.set_image_quality("medium")

# 5. Limit output formats
visualizer.set_output_formats(["svg"])  # Only SVG

# 6. Enable compression
visualizer.enable_compression()
```

#### Slow JAX Compilation

**Symptoms:**

- Long delays when starting training
- "Compiling..." messages taking too long
- JIT compilation errors with visualization

**Diagnosis:**

```python
# Check if visualization breaks JIT
@jax.jit
def test_jit_compatibility(state, action, config):
    new_state, obs, reward, done, info = arc_step(state, action, config)
    # This should work without issues
    return new_state, reward


# Test compilation
try:
    result = test_jit_compatibility(state, action, config)
    print("✅ JIT compilation working")
except Exception as e:
    print(f"❌ JIT compilation failed: {e}")
```

**Solutions:**

```python
# 1. Use JAX debug callbacks properly
@jax.jit
def step_with_proper_callback(state, action, config):
    new_state, obs, reward, done, info = arc_step(state, action, config)

    # Proper debug callback usage
    jax.debug.callback(
        visualizer.jax_callback_function, state, action, new_state, reward
    )

    return new_state, obs, reward, done, info


# 2. Mark visualization config as static
@jax.jit
def step_with_static_config(state, action, env_config, viz_config):
    # Implementation with static viz_config
    pass


jit_step = jax.jit(step_with_static_config, static_argnums=(3,))


# 3. Separate visualization from core computation
@jax.jit
def core_step(state, action, config):
    return arc_step(state, action, config)


# Visualize outside JIT
new_state, obs, reward, done, info = core_step(state, action, config)
visualizer.visualize_step(state, action, new_state, reward, info, step_num)
```

### 2. Memory Issues

#### Memory Leaks

**Symptoms:**

- Memory usage continuously growing
- Out of memory errors during long training runs
- System becoming unresponsive

**Diagnosis:**

```python
import psutil
import gc


def diagnose_memory_leak():
    """Diagnose memory leak issues."""

    # Check system memory
    memory = psutil.virtual_memory()
    print(f"System memory: {memory.percent}% used")

    # Check visualization memory
    viz_memory = visualizer.get_memory_usage()
    print(f"Visualization memory: {viz_memory['current_mb']:.1f} MB")

    # Check for unreleased objects
    gc.collect()
    objects_before = len(gc.get_objects())

    # Run a few visualization steps
    for i in range(10):
        visualizer.visualize_step(state, action, new_state, reward, info, i)

    gc.collect()
    objects_after = len(gc.get_objects())

    print(f"Objects created: {objects_after - objects_before}")

    # Check for circular references
    if gc.garbage:
        print(f"⚠️  Circular references detected: {len(gc.garbage)}")


diagnose_memory_leak()
```

**Solutions:**

```python
# 1. Enable automatic cleanup
visualizer.enable_auto_cleanup(
    cleanup_frequency=100,  # Every 100 steps
    memory_threshold=0.8,  # At 80% memory usage
)

# 2. Set memory limits
visualizer.set_memory_limit(500)  # 500 MB limit

# 3. Use context managers
with visualizer.episode_context(episode_num=0):
    # Visualization code here
    # Automatic cleanup when exiting context
    pass

# 4. Manual cleanup
if step % 100 == 0:
    visualizer.cleanup_old_data()
    gc.collect()

# 5. Disable caching for long runs
visualizer.disable_caching()
```

#### Out of Disk Space

**Symptoms:**

- "No space left on device" errors
- Visualization files not being saved
- System warnings about disk space

**Diagnosis:**

```python
import shutil


def check_disk_space():
    """Check available disk space."""

    # Check output directory space
    output_dir = visualizer.get_output_directory()
    total, used, free = shutil.disk_usage(output_dir)

    print(f"Disk space in {output_dir}:")
    print(f"Total: {total // (1024**3)} GB")
    print(f"Used: {used // (1024**3)} GB")
    print(f"Free: {free // (1024**3)} GB")

    # Check visualization storage usage
    viz_storage = visualizer.get_storage_stats()
    print(f"Visualization storage: {viz_storage['used_gb']:.1f} GB")

    if free < 1024**3:  # Less than 1 GB free
        print("⚠️  Low disk space!")


check_disk_space()
```

**Solutions:**

```python
# 1. Enable automatic cleanup
visualizer.enable_storage_cleanup(
    max_storage_gb=5.0, cleanup_policy="oldest_first"  # Limit to 5 GB
)

# 2. Use compression
visualizer.enable_compression(level=6)

# 3. Reduce image quality
visualizer.set_image_quality("low")

# 4. Clean up old episodes
visualizer.cleanup_episodes(keep_recent=10)

# 5. Change output directory
visualizer.set_output_directory("/path/to/larger/disk")

# 6. Use temporary storage
import tempfile

temp_dir = tempfile.mkdtemp()
visualizer.set_output_directory(temp_dir)
```

### 3. Async Processing Issues

#### Async Workers Not Starting

**Symptoms:**

- Visualization queue growing without processing
- No background processing happening
- Async logger errors

**Diagnosis:**

```python
# Check async worker status
async_status = visualizer.get_async_status()
print(f"Workers running: {async_status['workers_running']}")
print(f"Queue size: {async_status['queue_size']}")
print(f"Processed items: {async_status['processed_count']}")

if async_status["workers_running"] == 0:
    print("❌ No async workers running")
```

**Solutions:**

```python
# 1. Restart async workers
visualizer.restart_async_workers()

# 2. Check thread limits
import threading

print(f"Active threads: {threading.active_count()}")

# 3. Reduce worker count if too many threads
visualizer.set_async_workers(1)  # Use single worker

# 4. Use process-based workers instead of threads
visualizer.enable_process_workers()

# 5. Disable async processing if problematic
visualizer.disable_async_processing()
```

#### Queue Overflow

**Symptoms:**

- "Queue full" warnings
- Visualization data being dropped
- Slow processing of visualization queue

**Diagnosis:**

```python
queue_stats = visualizer.get_queue_stats()
print(f"Queue size: {queue_stats['current_size']}")
print(f"Max size: {queue_stats['max_size']}")
print(f"Drop count: {queue_stats['dropped_items']}")

if queue_stats["current_size"] > queue_stats["max_size"] * 0.8:
    print("⚠️  Queue nearly full")
```

**Solutions:**

```python
# 1. Increase queue size
visualizer.set_queue_size(2000)

# 2. Add more workers
visualizer.set_async_workers(4)

# 3. Increase batch size
visualizer.set_batch_size(50)

# 4. Reduce logging frequency
visualizer.set_log_frequency(20)

# 5. Enable priority queuing
visualizer.enable_priority_queue()
```

### 4. Wandb Integration Issues

#### Authentication Failures

**Symptoms:**

- "Authentication failed" errors
- Unable to log to wandb
- API key errors

**Diagnosis:**

```python
import wandb

# Test wandb authentication
try:
    wandb.login()
    print("✅ Wandb authentication successful")
except Exception as e:
    print(f"❌ Wandb authentication failed: {e}")

# Check API key
import os

api_key = os.environ.get("WANDB_API_KEY")
if api_key:
    print(f"API key found: {api_key[:8]}...")
else:
    print("❌ No API key found")
```

**Solutions:**

```python
# 1. Set API key explicitly
import os

os.environ["WANDB_API_KEY"] = "your_api_key_here"

# 2. Login manually
import wandb

wandb.login(key="your_api_key_here")

# 3. Use offline mode
visualizer.set_wandb_offline_mode(True)

# 4. Disable wandb temporarily
visualizer.disable_wandb()

# 5. Check network connectivity
import requests

try:
    response = requests.get("https://api.wandb.ai/health", timeout=5)
    print(f"Wandb API status: {response.status_code}")
except Exception as e:
    print(f"Network issue: {e}")
```

#### Sync Issues

**Symptoms:**

- Offline runs not syncing
- Data not appearing in wandb dashboard
- Sync errors

**Diagnosis:**

```python
# Check offline runs
offline_runs = visualizer.get_offline_runs()
print(f"Offline runs: {len(offline_runs)}")

for run in offline_runs:
    print(f"Run: {run['id']}, Status: {run['status']}")
```

**Solutions:**

```python
# 1. Manual sync
visualizer.sync_offline_runs()

# 2. Sync specific run
visualizer.sync_run("run_id_here")

# 3. Check sync status
sync_status = visualizer.get_sync_status()
print(f"Sync status: {sync_status}")

# 4. Force sync with retry
visualizer.force_sync_with_retry(max_retries=3)

# 5. Clear sync cache
visualizer.clear_sync_cache()
```

### 5. File System Issues

#### Permission Errors

**Symptoms:**

- "Permission denied" errors
- Unable to create output directories
- File access errors

**Diagnosis:**

```python
import os
import stat


def check_permissions(path):
    """Check file system permissions."""

    try:
        # Check if path exists
        if os.path.exists(path):
            # Check permissions
            st = os.stat(path)
            permissions = stat.filemode(st.st_mode)
            print(f"Path: {path}")
            print(f"Permissions: {permissions}")
            print(f"Owner: {st.st_uid}")

            # Test write access
            test_file = os.path.join(path, "test_write.tmp")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                print("✅ Write access OK")
            except Exception as e:
                print(f"❌ Write access failed: {e}")
        else:
            print(f"❌ Path does not exist: {path}")

    except Exception as e:
        print(f"❌ Permission check failed: {e}")


# Check output directory permissions
output_dir = visualizer.get_output_directory()
check_permissions(output_dir)
```

**Solutions:**

```python
# 1. Change output directory
import tempfile

temp_dir = tempfile.mkdtemp()
visualizer.set_output_directory(temp_dir)

# 2. Fix permissions (Unix/Linux)
import os

output_dir = visualizer.get_output_directory()
os.chmod(output_dir, 0o755)

# 3. Use user home directory
import os

home_dir = os.path.expanduser("~/jaxarc_output")
os.makedirs(home_dir, exist_ok=True)
visualizer.set_output_directory(home_dir)

# 4. Run with different user (if possible)
# sudo chown -R $USER:$USER /path/to/output/dir

# 5. Use relative paths
visualizer.set_output_directory("./outputs")
```

#### File Corruption

**Symptoms:**

- Corrupted visualization files
- Unable to load saved episodes
- Incomplete file writes

**Diagnosis:**

```python
def check_file_integrity():
    """Check integrity of visualization files."""

    output_dir = visualizer.get_output_directory()

    for root, dirs, files in os.walk(output_dir):
        for file in files:
            filepath = os.path.join(root, file)

            try:
                # Check file size
                size = os.path.getsize(filepath)
                if size == 0:
                    print(f"❌ Empty file: {filepath}")

                # Try to read file
                if filepath.endswith(".svg"):
                    with open(filepath, "r") as f:
                        content = f.read()
                        if not content.strip():
                            print(f"❌ Empty SVG: {filepath}")
                        elif not content.startswith("<svg"):
                            print(f"❌ Invalid SVG: {filepath}")

            except Exception as e:
                print(f"❌ Corrupted file {filepath}: {e}")


check_file_integrity()
```

**Solutions:**

```python
# 1. Enable atomic writes
visualizer.enable_atomic_writes()

# 2. Add file validation
visualizer.enable_file_validation()

# 3. Use backup writes
visualizer.enable_backup_writes()

# 4. Increase write timeout
visualizer.set_write_timeout(30)  # 30 seconds

# 5. Clean up corrupted files
visualizer.cleanup_corrupted_files()
```

### 6. Batched Logging Issues

#### Batched Logging Not Working

**Symptoms:**
- No aggregated metrics appearing in logs
- Batch data not being processed
- Missing wandb metrics in batched training

**Diagnosis:**
```python
# Check configuration
print(f"Batched logging enabled: {config.logging.batched_logging_enabled}")
print(f"Log frequency: {config.logging.log_frequency}")
print(f"Sampling enabled: {config.logging.sampling_enabled}")

# Verify batch data format
batch_data = {
    "update_step": 0,  # Required
    "episode_returns": jnp.array([1.0, 2.0, 3.0]),  # Required
    "episode_lengths": jnp.array([50, 45, 60]),  # Required
    "similarity_scores": jnp.array([0.8, 0.6, 0.9]),  # Required
    "success_mask": jnp.array([True, False, True]),  # Required
    "policy_loss": 1.5,  # Required scalar
    "value_loss": 0.8,  # Required scalar
    "gradient_norm": 2.1,  # Required scalar
}

# Test logging
logger.log_batch_step(batch_data)
```

**Solutions:**
1. **Enable batched logging**:
   ```python
   config.logging.batched_logging_enabled = True
   ```

2. **Check update step alignment**:
   ```python
   # Ensure update_step % log_frequency == 0 for logging to occur
   assert update_step % config.logging.log_frequency == 0
   ```

3. **Verify data types**:
   ```python
   # Ensure arrays are JAX arrays
   episode_returns = jnp.asarray(episode_returns)
   ```

#### Performance Issues with Batched Logging

**Symptoms:**
- Training loop slowdown when batched logging is enabled
- High memory usage during logging
- JAX compilation issues with logging callbacks

**Diagnosis:**
```python
import time

# Measure logging overhead
start_time = time.time()
logger.log_batch_step(batch_data)
logging_time = time.time() - start_time

print(f"Logging overhead: {logging_time:.3f}s")

# Check memory usage
import psutil
process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {memory_mb:.1f} MB")
```

**Solutions:**
1. **Reduce logging frequency**:
   ```python
   config.logging.log_frequency = 100  # Instead of 10
   ```

2. **Disable expensive features**:
   ```python
   config.logging.include_full_states = False
   config.logging.log_operations = False
   config.logging.log_grid_changes = False
   ```

3. **Reduce sampling**:
   ```python
   config.logging.num_samples = 1  # Instead of 5
   config.logging.sample_frequency = 500  # Instead of 50
   ```

4. **Enable compression**:
   ```python
   config.logging.compression = True
   ```

#### Missing Metrics in Aggregation

**Symptoms:**
- Some expected metrics not appearing in aggregated output
- Inconsistent metric availability
- Metrics showing as NaN or zero

**Diagnosis:**
```python
# Check metric configuration
print("Enabled metrics:")
print(f"  Rewards: {config.logging.log_aggregated_rewards}")
print(f"  Similarity: {config.logging.log_aggregated_similarity}")
print(f"  Loss metrics: {config.logging.log_loss_metrics}")
print(f"  Gradient norms: {config.logging.log_gradient_norms}")
print(f"  Episode lengths: {config.logging.log_episode_lengths}")
print(f"  Success rates: {config.logging.log_success_rates}")

# Check data validity
print(f"Episode returns shape: {episode_returns.shape}")
print(f"Episode returns range: [{episode_returns.min():.3f}, {episode_returns.max():.3f}]")
print(f"Any NaN values: {jnp.isnan(episode_returns).any()}")
```

**Solutions:**
1. **Enable required metrics**:
   ```python
   config.logging.log_aggregated_rewards = True
   config.logging.log_loss_metrics = True
   ```

2. **Check data validity**:
   ```python
   # Remove NaN values
   episode_returns = jnp.where(jnp.isnan(episode_returns), 0.0, episode_returns)
   
   # Ensure non-empty arrays
   assert episode_returns.size > 0, "Empty episode returns array"
   ```

3. **Verify data types**:
   ```python
   # Ensure correct scalar types
   policy_loss = float(policy_loss)
   value_loss = float(value_loss)
   ```

### 7. Configuration Issues

#### Invalid Configuration

**Symptoms:**

- Configuration validation errors
- Unexpected behavior
- Missing required settings

**Diagnosis:**

```python
from jaxarc.utils.visualization import validate_configuration

# Validate current configuration
try:
    validate_configuration(visualizer.config)
    print("✅ Configuration valid")
except Exception as e:
    print(f"❌ Configuration invalid: {e}")

# Check specific settings
config_issues = visualizer.check_configuration_issues()
for issue in config_issues:
    print(f"⚠️  {issue}")
```

**Solutions:**

```python
# 1. Reset to default configuration
visualizer.reset_to_default_config()

# 2. Fix specific issues
if "debug_level" in config_issues:
    visualizer.set_debug_level("standard")

if "output_formats" in config_issues:
    visualizer.set_output_formats(["svg"])

# 3. Use configuration templates
from jaxarc.utils.visualization import get_config_template

template = get_config_template("training")
visualizer.apply_config(template)


# 4. Validate before applying
def safe_config_update(visualizer, new_config):
    try:
        validate_configuration(new_config)
        visualizer.apply_config(new_config)
        print("✅ Configuration updated")
    except Exception as e:
        print(f"❌ Configuration update failed: {e}")


safe_config_update(visualizer, new_config)
```

#### Hydra Configuration Issues

**Symptoms:**

- Hydra config loading errors
- Override failures
- Missing configuration files

**Diagnosis:**

```python
import hydra
from omegaconf import OmegaConf

# Check if config file exists
config_path = "conf/visualization/debug_standard.yaml"
if os.path.exists(config_path):
    print(f"✅ Config file found: {config_path}")

    # Try to load config
    try:
        cfg = OmegaConf.load(config_path)
        print("✅ Config loaded successfully")
        print(f"Config keys: {list(cfg.keys())}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
else:
    print(f"❌ Config file not found: {config_path}")
```

**Solutions:**

```python
# 1. Create missing config files
from jaxarc.utils.visualization import create_default_configs

create_default_configs("conf/visualization/")

# 2. Fix YAML syntax
# Check for indentation, quotes, and structure issues

# 3. Use programmatic config
config_dict = {"debug_level": "standard", "output_formats": ["svg"], "enabled": True}
cfg = OmegaConf.create(config_dict)
visualizer = create_visualizer_from_config(cfg)


# 4. Validate Hydra overrides
@hydra.main(config_path="conf", config_name="config", version_base=None)
def test_config(cfg):
    print("Config loaded successfully")
    print(OmegaConf.to_yaml(cfg))


# 5. Use absolute paths
config_path = os.path.abspath("conf/visualization/debug_standard.yaml")
```

## Debugging Techniques

### Enable Debug Logging

```python
import logging

# Enable debug logging for visualization system
logging.getLogger("jaxarc.utils.visualization").setLevel(logging.DEBUG)

# Enable debug logging for specific components
logging.getLogger("jaxarc.utils.visualization.async_logger").setLevel(logging.DEBUG)
logging.getLogger("jaxarc.utils.visualization.wandb_sync").setLevel(logging.DEBUG)

# Create debug handler
debug_handler = logging.StreamHandler()
debug_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
debug_handler.setFormatter(formatter)

# Add handler to loggers
logging.getLogger("jaxarc.utils.visualization").addHandler(debug_handler)
```

### Trace Function Calls

```python
import functools
import time


def trace_calls(func):
    """Decorator to trace function calls."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        print(f"🔍 Calling {func.__name__}")

        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            print(f"✅ {func.__name__} completed in {end_time - start_time:.3f}s")
            return result
        except Exception as e:
            end_time = time.perf_counter()
            print(f"❌ {func.__name__} failed in {end_time - start_time:.3f}s: {e}")
            raise

    return wrapper


# Apply to visualization methods
visualizer.visualize_step = trace_calls(visualizer.visualize_step)
visualizer.visualize_episode_summary = trace_calls(visualizer.visualize_episode_summary)
```

### Memory Debugging

```python
import tracemalloc
import linecache


def display_top_memory_usage(snapshot, key_type="lineno", limit=10):
    """Display top memory usage."""

    top_stats = snapshot.statistics(key_type)

    print(f"Top {limit} lines:")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback.format()[-1]
        print(f"#{index}: {frame}")
        print(f"    {stat.size / 1024 / 1024:.1f} MB")
        print()


# Start memory tracing
tracemalloc.start()

# Run visualization code
# ... your code here ...

# Take snapshot and display
snapshot = tracemalloc.take_snapshot()
display_top_memory_usage(snapshot)
```

### Performance Profiling

```python
import cProfile
import pstats


def profile_visualization(func, *args, **kwargs):
    """Profile visualization function."""

    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()

    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)  # Top 20 functions

    return result


# Profile visualization step
result = profile_visualization(
    visualizer.visualize_step, state, action, new_state, reward, info, step_num
)
```

## Recovery Procedures

### System Recovery

```python
def recover_visualization_system():
    """Recover from visualization system failure."""

    print("🔧 Starting system recovery...")

    try:
        # 1. Stop all async workers
        visualizer.stop_async_workers()
        print("✅ Stopped async workers")

        # 2. Clear all queues
        visualizer.clear_all_queues()
        print("✅ Cleared queues")

        # 3. Clean up memory
        visualizer.force_memory_cleanup()
        print("✅ Cleaned up memory")

        # 4. Reset configuration
        visualizer.reset_to_safe_config()
        print("✅ Reset configuration")

        # 5. Restart async workers
        visualizer.restart_async_workers()
        print("✅ Restarted async workers")

        # 6. Test basic functionality
        if visualizer.run_self_test():
            print("✅ System recovery successful")
            return True
        else:
            print("❌ System recovery failed")
            return False

    except Exception as e:
        print(f"❌ Recovery failed: {e}")
        return False


# Use recovery procedure
if not recover_visualization_system():
    print("Manual intervention required")
```

### Data Recovery

```python
def recover_corrupted_data():
    """Recover from corrupted visualization data."""

    print("🔧 Starting data recovery...")

    # 1. Backup current data
    backup_dir = visualizer.create_backup()
    print(f"✅ Created backup: {backup_dir}")

    # 2. Scan for corrupted files
    corrupted_files = visualizer.scan_for_corruption()
    print(f"Found {len(corrupted_files)} corrupted files")

    # 3. Attempt to repair files
    repaired_count = 0
    for file_path in corrupted_files:
        if visualizer.attempt_file_repair(file_path):
            repaired_count += 1

    print(f"✅ Repaired {repaired_count}/{len(corrupted_files)} files")

    # 4. Remove unrecoverable files
    removed_count = visualizer.remove_corrupted_files()
    print(f"✅ Removed {removed_count} unrecoverable files")

    # 5. Rebuild indices
    visualizer.rebuild_episode_indices()
    print("✅ Rebuilt episode indices")

    return repaired_count, removed_count


# Use data recovery
repaired, removed = recover_corrupted_data()
```

## Prevention Strategies

### Monitoring Setup

```python
def setup_monitoring():
    """Set up monitoring to prevent issues."""

    # 1. Performance monitoring
    visualizer.enable_performance_monitoring(
        check_interval=100, alert_threshold=0.1  # Every 100 steps  # 10% overhead
    )

    # 2. Memory monitoring
    visualizer.enable_memory_monitoring(
        check_interval=50, alert_threshold=0.8  # Every 50 steps  # 80% memory usage
    )

    # 3. Disk space monitoring
    visualizer.enable_disk_monitoring(
        check_interval=1000, alert_threshold=0.9  # Every 1000 steps  # 90% disk usage
    )

    # 4. Queue monitoring
    visualizer.enable_queue_monitoring(
        check_interval=10, alert_threshold=0.8  # Every 10 steps  # 80% queue full
    )

    print("✅ Monitoring enabled")


setup_monitoring()
```

### Automated Recovery

```python
def setup_automated_recovery():
    """Set up automated recovery procedures."""

    # 1. Auto-cleanup on memory pressure
    visualizer.enable_auto_cleanup(memory_threshold=0.8, cleanup_frequency=100)

    # 2. Auto-restart workers on failure
    visualizer.enable_worker_auto_restart(failure_threshold=3, restart_delay=5.0)

    # 3. Auto-sync offline data
    visualizer.enable_auto_sync(sync_interval=300, max_retries=3)  # Every 5 minutes

    # 4. Auto-backup critical data
    visualizer.enable_auto_backup(
        backup_interval=1000, keep_backups=5  # Every 1000 steps
    )

    print("✅ Automated recovery enabled")


setup_automated_recovery()
```

## Getting Help

### Collecting Debug Information

```python
def collect_debug_info():
    """Collect comprehensive debug information."""

    debug_info = {
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "jax_version": jax.__version__,
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_count": multiprocessing.cpu_count(),
        },
        "visualization": {
            "config": visualizer.get_config_dict(),
            "status": visualizer.get_system_status(),
            "performance": visualizer.get_performance_stats(),
            "memory": visualizer.get_memory_stats(),
            "errors": visualizer.get_recent_errors(),
        },
        "environment": {
            "output_dir": visualizer.get_output_directory(),
            "disk_space": shutil.disk_usage(visualizer.get_output_directory()),
            "permissions": check_permissions(visualizer.get_output_directory()),
        },
    }

    # Save debug info
    debug_file = "debug_info.json"
    with open(debug_file, "w") as f:
        json.dump(debug_info, f, indent=2, default=str)

    print(f"✅ Debug info saved to {debug_file}")
    return debug_info


# Collect debug information
debug_info = collect_debug_info()
```

### Reporting Issues

When reporting issues, please include:

1. **System Information**: OS, Python version, JAX version
2. **Configuration**: Your visualization configuration
3. **Error Messages**: Complete error messages and stack traces
4. **Debug Information**: Output from `collect_debug_info()`
5. **Reproduction Steps**: Minimal code to reproduce the issue
6. **Expected vs Actual Behavior**: What you expected vs what happened

### Community Resources

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check the latest documentation
- **Examples**: Review working examples in the `examples/` directory
- **Tests**: Look at test cases for usage patterns

This troubleshooting guide covers the most common issues you might encounter
with JaxARC's enhanced visualization system. For issues not covered here, please
refer to the community resources or file a detailed bug report.
