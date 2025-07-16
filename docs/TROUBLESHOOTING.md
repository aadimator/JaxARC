# Troubleshooting Guide

This guide helps you resolve common issues with JaxARC, focusing on GitHub dataset downloads, parser configuration, and environment setup.

## 🚀 Quick Diagnostics

Run this diagnostic script to quickly identify common issues:

```python
#!/usr/bin/env python3
"""Quick diagnostic script for JaxARC setup issues."""

import json
import subprocess
from pathlib import Path
from jaxarc.parsers import ArcAgiParser
from omegaconf import DictConfig

def quick_diagnosis():
    """Run quick diagnostics for common JaxARC issues."""
    
    print("🔍 JaxARC Quick Diagnostics")
    print("=" * 40)
    
    # Check system requirements
    print("\n1. System Requirements")
    print("-" * 20)
    
    # Check Git
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Git: {result.stdout.strip()}")
        else:
            print("❌ Git: Command failed")
    except FileNotFoundError:
        print("❌ Git: Not installed")
        print("   Fix: brew install git (macOS) or apt install git (Linux)")
    
    # Check Python
    import sys
    print(f"✅ Python: {sys.version.split()[0]}")
    
    # Check datasets
    print("\n2. Dataset Availability")
    print("-" * 25)
    
    datasets = {
        "ARC-AGI-1": "data/raw/ARC-AGI-1/data",
        "ARC-AGI-2": "data/raw/ARC-AGI-2/data", 
        "ConceptARC": "data/raw/ConceptARC/corpus",
        "MiniARC": "data/raw/MiniARC/data/MiniARC"
    }
    
    missing_datasets = []
    for name, path_str in datasets.items():
        path = Path(path_str)
        if path.exists():
            if name.startswith("ARC-AGI"):
                train_files = list((path / "training").glob("*.json"))
                eval_files = list((path / "evaluation").glob("*.json"))
                print(f"✅ {name}: {len(train_files)} train, {len(eval_files)} eval")
            else:
                files = list(path.glob("**/*.json"))
                print(f"✅ {name}: {len(files)} tasks")
        else:
            print(f"❌ {name}: Missing")
            missing_datasets.append(name.lower().replace('-', '-'))
    
    # Check parser functionality
    print("\n3. Parser Test")
    print("-" * 15)
    
    if Path("data/raw/ARC-AGI-1/data/training").exists():
        try:
            config = DictConfig({
                "training": {"path": "data/raw/ARC-AGI-1/data/training"},
                "evaluation": {"path": "data/raw/ARC-AGI-1/data/evaluation"},
                "grid": {"max_grid_height": 30, "max_grid_width": 30},
                "max_train_pairs": 10, "max_test_pairs": 3,
            })
            
            parser = ArcAgiParser(config)
            task_ids = parser.get_available_task_ids()
            print(f"✅ Parser: {len(task_ids)} tasks loaded")
            
        except Exception as e:
            print(f"❌ Parser: {e}")
    else:
        print("⏭️  Parser: Skipped (no ARC-AGI-1 dataset)")
    
    # Recommendations
    print("\n4. Recommendations")
    print("-" * 18)
    
    if missing_datasets:
        print("📥 Download missing datasets:")
        for dataset in missing_datasets:
            print(f"   python scripts/download_dataset.py {dataset}")
        print("   Or download all: python scripts/download_dataset.py all")
    
    print("📚 For detailed help:")
    print("   - GitHub issues: Common download problems")
    print("   - Parser issues: Configuration format problems") 
    print("   - Performance: Memory and speed optimization")
    print("   - Migration: Kaggle to GitHub format")

if __name__ == "__main__":
    quick_diagnosis()
```

Save this as `diagnose.py` and run: `python diagnose.py`

## 📥 GitHub Download Issues

### Issue 1: Repository Access Errors

**Symptoms:**
- "Repository not found"
- "Permission denied" 
- "Could not resolve hostname"

**Solutions:**

```bash
# Test internet connectivity
ping github.com

# Test repository access
curl -I https://github.com/fchollet/ARC-AGI
curl -I https://github.com/arcprize/ARC-AGI-2

# Check if behind corporate firewall
git config --global http.proxy http://proxy.company.com:8080
git config --global https.proxy https://proxy.company.com:8080

# Try with different protocol
git config --global url."https://".insteadOf git://
```

### Issue 2: Git Not Found

**Symptoms:**
- "git: command not found"
- "'git' is not recognized"

**Solutions:**

```bash
# macOS
brew install git

# Ubuntu/Debian  
sudo apt update && sudo apt install git

# CentOS/RHEL
sudo yum install git

# Windows (use Git for Windows or WSL)
# Download from: https://git-scm.com/download/win

# Verify installation
git --version
```

### Issue 3: Disk Space Issues

**Symptoms:**
- "No space left on device"
- Download stops unexpectedly

**Solutions:**

```bash
# Check available space
df -h

# Clean up old downloads
rm -rf data/raw/old-datasets/

# Use custom output directory
python scripts/download_dataset.py arc-agi-1 --output /path/with/space

# Clean package caches
pip cache purge
conda clean --all  # if using conda
```

### Issue 4: Network Timeouts

**Symptoms:**
- "Connection timed out"
- "Transfer closed with outstanding read data remaining"

**Solutions:**

```bash
# Increase git timeout settings
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 300
git config --global http.postBuffer 524288000

# Use different DNS
echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolv.conf

# Retry with verbose output
python scripts/download_dataset.py arc-agi-1 --verbose

# Manual clone as fallback
git clone https://github.com/fchollet/ARC-AGI.git data/raw/ARC-AGI-1
```

### Issue 5: SSL Certificate Problems

**Symptoms:**
- "SSL certificate problem"
- "certificate verify failed"

**Solutions:**

```bash
# Update certificates (macOS)
brew install ca-certificates && brew upgrade ca-certificates

# Update certificates (Ubuntu)
sudo apt update && sudo apt install ca-certificates

# Temporary workaround (not recommended for production)
git config --global http.sslVerify false

# Use system certificate store
git config --global http.sslCAInfo /etc/ssl/certs/ca-certificates.crt
```

### Issue 6: Directory Already Exists

**Symptoms:**
- "Directory already exists"
- "fatal: destination path exists"

**Solutions:**

```bash
# Use --force flag to overwrite
python scripts/download_dataset.py arc-agi-1 --force

# Manual cleanup
rm -rf data/raw/ARC-AGI-1/
python scripts/download_dataset.py arc-agi-1

# Backup existing data first
mv data/raw/ARC-AGI-1 data/raw/ARC-AGI-1.backup
python scripts/download_dataset.py arc-agi-1
```

## 🔧 Parser Configuration Issues

### Issue 1: Legacy Kaggle Format Detected

**Symptoms:**
```
RuntimeError: Legacy Kaggle format detected. Please update configuration 
to use GitHub format with 'path' instead of 'challenges'/'solutions'
```

**Solution:**

```python
# OLD (Kaggle format - no longer supported)
config = DictConfig({
    "training": {
        "challenges": "data/raw/arc-prize-2024/arc-agi_training_challenges.json",
        "solutions": "data/raw/arc-prize-2024/arc-agi_training_solutions.json"
    },
    "evaluation": {
        "challenges": "data/raw/arc-prize-2024/arc-agi_evaluation_challenges.json", 
        "solutions": "data/raw/arc-prize-2024/arc-agi_evaluation_solutions.json"
    }
})

# NEW (GitHub format)
config = DictConfig({
    "training": {"path": "data/raw/ARC-AGI-1/data/training"},
    "evaluation": {"path": "data/raw/ARC-AGI-1/data/evaluation"},
    "grid": {"max_grid_height": 30, "max_grid_width": 30},
    "max_train_pairs": 10,
    "max_test_pairs": 3
})
```

### Issue 2: No Data Path Specified

**Symptoms:**
```
RuntimeError: No data path specified in configuration
```

**Solution:**

```yaml
# Add required path fields to your configuration
training:
  path: "data/raw/ARC-AGI-1/data/training"
evaluation:
  path: "data/raw/ARC-AGI-1/data/evaluation"
```

### Issue 3: Data Directory Not Found

**Symptoms:**
```
RuntimeError: Data directory not found: /path/to/data
```

**Solutions:**

```bash
# Check if dataset was downloaded
ls -la data/raw/ARC-AGI-1/data/training/

# Download if missing
python scripts/download_dataset.py arc-agi-1

# Verify correct path in configuration
# Path should point to directory containing .json files
```

### Issue 4: No JSON Files Found

**Symptoms:**
```
RuntimeError: No JSON files found in /path/to/directory
```

**Solutions:**

```bash
# Check directory contents
ls -la data/raw/ARC-AGI-1/data/training/

# Should see files like: 007bbfb7.json, 00d62c1b.json, etc.

# If empty, re-download
python scripts/download_dataset.py arc-agi-1 --force

# Check for permission issues
chmod -R 755 data/raw/ARC-AGI-1/
```

### Issue 5: Task Loading Errors

**Symptoms:**
- "Invalid JSON format"
- "Task validation failed"
- "Malformed task data"

**Solutions:**

```python
# Test individual task file
import json
from pathlib import Path

task_file = Path("data/raw/ARC-AGI-1/data/training/007bbfb7.json")
try:
    with task_file.open() as f:
        data = json.load(f)
    print("✅ JSON is valid")
    print(f"Keys: {list(data.keys())}")  # Should see 'train' and 'test'
except json.JSONDecodeError as e:
    print(f"❌ Invalid JSON: {e}")
    # Re-download to fix corruption
    # python scripts/download_dataset.py arc-agi-1 --force
```

## 🚀 Performance Issues

### Issue 1: Slow Task Loading

**Symptoms:**
- Parser takes long time to initialize
- High memory usage during loading

**Solutions:**

```python
# Use lazy loading - only load tasks when needed
parser = ArcAgiParser(config)
task = parser.get_random_task(key)  # Loads single task

# Pre-load frequently used tasks
task_ids = parser.get_available_task_ids()[:10]  # First 10 tasks
for task_id in task_ids:
    parser.get_task_by_id(task_id)  # Cache these tasks

# Clear cache periodically for long-running processes
parser._cached_tasks.clear()
```

### Issue 2: Memory Issues with Large Datasets

**Symptoms:**
- Out of memory errors
- System becomes unresponsive

**Solutions:**

```python
# Use MiniARC for development (36x less memory)
from jaxarc.parsers import MiniArcParser

miniarc_config = DictConfig({
    "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
    "grid": {"max_grid_height": 5, "max_grid_width": 5},
    "max_train_pairs": 3, "max_test_pairs": 1
})

parser = MiniArcParser(miniarc_config)  # Much faster and lighter

# Or limit dataset size
config["max_tasks"] = 100  # Only load first 100 tasks
```

### Issue 3: JAX Compilation Issues

**Symptoms:**
- "Non-hashable static arguments"
- JIT compilation fails

**Solutions:**

```python
# Ensure configs are frozen dataclasses
from jaxarc.envs import create_standard_config

config = create_standard_config()  # Already frozen
assert config.__dataclass_params__.frozen  # Should be True

# Mark config as static for JIT
@jax.jit
def step_fn(state, action, config):
    return arc_step(state, action, config)

# Or use static_argnums
step_fn = jax.jit(arc_step, static_argnums=(2,))
```

## 🔄 Migration Issues

### Issue 1: Updating from Kaggle Format

**Problem:** Existing code uses old Kaggle-based dataset format.

**Solution:** Follow the [Migration Guide](KAGGLE_TO_GITHUB_MIGRATION.md):

1. **Update download commands:**
   ```bash
   # OLD
   kaggle competitions download -c arc-prize-2024
   
   # NEW  
   python scripts/download_dataset.py arc-agi-1
   ```

2. **Update configuration format:**
   ```python
   # OLD
   config = {"training": {"challenges": "...", "solutions": "..."}}
   
   # NEW
   config = {"training": {"path": "data/raw/ARC-AGI-1/data/training"}}
   ```

3. **Remove Kaggle dependencies:**
   ```bash
   pip uninstall kaggle  # If no longer needed
   ```

### Issue 2: Class-Based to Config-Based API

**Problem:** Using old class-based environment API.

**Solution:**

```python
# OLD (Class-based)
from jaxarc.envs import ArcEnvironment

env = ArcEnvironment(config)
state, obs = env.reset(key)
new_state, obs, reward, done, info = env.step(state, action)

# NEW (Config-based)
from jaxarc.envs import arc_reset, arc_step, create_standard_config

config = create_standard_config()
state, obs = arc_reset(key, config)
new_state, obs, reward, done, info = arc_step(state, action, config)
```

## 🛠️ Environment Setup Issues

### Issue 1: Import Errors

**Symptoms:**
- "ModuleNotFoundError: No module named 'jaxarc'"
- "ImportError: cannot import name 'ArcAgiParser'"

**Solutions:**

```bash
# Install in development mode
pip install -e .

# Or install from PyPI
pip install jaxarc

# Check installation
python -c "import jaxarc; print(jaxarc.__version__)"

# Verify parser imports
python -c "from jaxarc.parsers import ArcAgiParser; print('✅ Import successful')"
```

### Issue 2: JAX Installation Issues

**Symptoms:**
- JAX not found or incompatible version
- CUDA/GPU issues

**Solutions:**

```bash
# Install JAX (CPU)
pip install jax jaxlib

# Install JAX (GPU) - CUDA 12
pip install jax[cuda12]

# Install JAX (GPU) - CUDA 11  
pip install jax[cuda11_local] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify JAX installation
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
```

### Issue 3: Dependency Conflicts

**Symptoms:**
- Version conflicts between packages
- "pip check" shows dependency issues

**Solutions:**

```bash
# Use pixi (recommended)
pixi shell
pixi install

# Or create clean environment
python -m venv jaxarc_env
source jaxarc_env/bin/activate  # Linux/macOS
# jaxarc_env\Scripts\activate  # Windows
pip install jaxarc

# Check for conflicts
pip check
```

## 🧪 Testing and Validation

### Validation Script

```python
#!/usr/bin/env python3
"""Comprehensive validation script for JaxARC setup."""

import jax
import jax.numpy as jnp
from pathlib import Path
from jaxarc.parsers import ArcAgiParser
from jaxarc.envs import arc_reset, arc_step, create_standard_config
from omegaconf import DictConfig

def validate_complete_setup():
    """Validate complete JaxARC setup including datasets, parsers, and environment."""
    
    print("🧪 Complete JaxARC Validation")
    print("=" * 50)
    
    success_count = 0
    total_tests = 6
    
    # Test 1: Dataset availability
    print("\n1. Testing Dataset Availability")
    print("-" * 30)
    
    arc_agi_1_path = Path("data/raw/ARC-AGI-1/data/training")
    if arc_agi_1_path.exists():
        task_files = list(arc_agi_1_path.glob("*.json"))
        if len(task_files) > 0:
            print(f"✅ ARC-AGI-1: {len(task_files)} training tasks")
            success_count += 1
        else:
            print("❌ ARC-AGI-1: No task files found")
    else:
        print("❌ ARC-AGI-1: Dataset not found")
        print("   Run: python scripts/download_dataset.py arc-agi-1")
    
    # Test 2: Parser functionality
    print("\n2. Testing Parser Functionality")
    print("-" * 32)
    
    if arc_agi_1_path.exists():
        try:
            config = DictConfig({
                "training": {"path": "data/raw/ARC-AGI-1/data/training"},
                "evaluation": {"path": "data/raw/ARC-AGI-1/data/evaluation"},
                "grid": {"max_grid_height": 30, "max_grid_width": 30},
                "max_train_pairs": 10, "max_test_pairs": 3,
            })
            
            parser = ArcAgiParser(config)
            task_ids = parser.get_available_task_ids()
            
            if len(task_ids) > 0:
                print(f"✅ Parser: {len(task_ids)} tasks loaded")
                success_count += 1
            else:
                print("❌ Parser: No tasks loaded")
                
        except Exception as e:
            print(f"❌ Parser: {e}")
    else:
        print("⏭️  Parser: Skipped (no dataset)")
    
    # Test 3: Task loading
    print("\n3. Testing Task Loading")
    print("-" * 25)
    
    if arc_agi_1_path.exists():
        try:
            key = jax.random.PRNGKey(42)
            task = parser.get_random_task(key)
            
            print(f"✅ Task Loading: {task.num_train_pairs} train pairs, {task.num_test_pairs} test pairs")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Task Loading: {e}")
    else:
        print("⏭️  Task Loading: Skipped (no dataset)")
    
    # Test 4: Environment creation
    print("\n4. Testing Environment Creation")
    print("-" * 33)
    
    try:
        env_config = create_standard_config(max_episode_steps=10)
        print("✅ Environment Config: Created successfully")
        success_count += 1
        
    except Exception as e:
        print(f"❌ Environment Config: {e}")
    
    # Test 5: Environment reset
    print("\n5. Testing Environment Reset")
    print("-" * 30)
    
    try:
        key = jax.random.PRNGKey(42)
        if 'task' in locals():
            state, obs = arc_reset(key, env_config, task_data=task)
            print(f"✅ Environment Reset: State shape {state.working_grid.shape}")
            success_count += 1
        else:
            print("⏭️  Environment Reset: Skipped (no task)")
            
    except Exception as e:
        print(f"❌ Environment Reset: {e}")
    
    # Test 6: Environment step
    print("\n6. Testing Environment Step")
    print("-" * 29)
    
    try:
        if 'state' in locals():
            action = {
                "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
                "operation": jnp.array(1, dtype=jnp.int32)
            }
            
            new_state, obs, reward, done, info = arc_step(state, action, env_config)
            print(f"✅ Environment Step: Reward {reward:.3f}, Done {done}")
            success_count += 1
        else:
            print("⏭️  Environment Step: Skipped (no state)")
            
    except Exception as e:
        print(f"❌ Environment Step: {e}")
    
    # Summary
    print(f"\n📊 Validation Summary")
    print("=" * 20)
    print(f"Passed: {success_count}/{total_tests} tests")
    
    if success_count == total_tests:
        print("🎉 All tests passed! JaxARC is ready to use.")
    elif success_count >= total_tests - 1:
        print("⚠️  Almost ready! Check failed tests above.")
    else:
        print("❌ Setup incomplete. Please resolve issues above.")
        print("\n📚 For help:")
        print("   - Check troubleshooting guide: docs/TROUBLESHOOTING.md")
        print("   - Run diagnostics: python diagnose.py")
        print("   - Open issue: https://github.com/aadimator/JaxARC/issues")
    
    return success_count == total_tests

if __name__ == "__main__":
    validate_complete_setup()
```

## 🆘 Getting Help

### Before Opening an Issue

1. **Run diagnostics:** `python diagnose.py`
2. **Check this troubleshooting guide**
3. **Review the [Migration Guide](KAGGLE_TO_GITHUB_MIGRATION.md)**
4. **Try the validation script above**

### When Opening an Issue

Include this information:

```bash
# System information
uname -a  # Linux/macOS
# or
systeminfo  # Windows

# Python and package versions
python --version
pip list | grep -E "(jax|jaxarc)"

# Git version
git --version

# Error output
python your_failing_script.py 2>&1 | tee error.log

# Diagnostic output
python diagnose.py > diagnosis.txt
```

### Community Resources

- **GitHub Issues:** [Report bugs](https://github.com/aadimator/JaxARC/issues)
- **GitHub Discussions:** [Ask questions](https://github.com/aadimator/JaxARC/discussions)
- **Documentation:** [Read the docs](https://JaxARC.readthedocs.io)
- **Examples:** [Working code samples](../examples/)

### Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Git not found | `brew install git` (macOS) or `apt install git` (Linux) |
| Dataset missing | `python scripts/download_dataset.py <dataset-name>` |
| Legacy config | Update to use `path` instead of `challenges`/`solutions` |
| Import errors | `pip install -e .` or `pip install jaxarc` |
| Memory issues | Use MiniARC: `from jaxarc.parsers import MiniArcParser` |
| Network timeout | `git config --global http.lowSpeedTime 300` |
| Disk space | `df -h` and clean up or use `--output /other/path` |
| SSL issues | Update certificates or use `http.sslVerify false` temporarily |

---

**Remember:** Most issues are resolved by ensuring datasets are downloaded and using the correct GitHub configuration format. When in doubt, start with `python scripts/download_dataset.py all` and use the diagnostic script.