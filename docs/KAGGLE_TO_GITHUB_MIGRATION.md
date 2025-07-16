# Kaggle to GitHub Migration Guide

This guide helps you migrate from Kaggle-based ARC-AGI dataset downloads to the new GitHub-based system. The migration eliminates external dependencies and provides a more streamlined, reliable download experience.

## 🔄 Overview of Changes

### What Changed

- **Data Source**: ARC-AGI datasets now downloaded from GitHub repositories instead of Kaggle
- **No External Dependencies**: Removed `kaggle` package requirement and CLI tool dependency
- **Unified Download System**: All datasets (ARC-AGI-1, ARC-AGI-2, ConceptARC, MiniARC) use GitHub
- **Individual Task Files**: GitHub repositories store tasks as individual JSON files instead of combined files
- **Simplified Authentication**: No need for Kaggle API credentials or account setup

### What's Improved

- **Reliability**: Direct GitHub repository cloning is more stable than Kaggle API
- **Speed**: Faster downloads without API rate limits
- **Consistency**: Same download mechanism for all datasets
- **Maintenance**: Easier to maintain without external CLI dependencies

## 📋 Migration Checklist

- [ ] Remove Kaggle CLI and credentials (if no longer needed)
- [ ] Update download commands to use new GitHub-based script
- [ ] Verify dataset configurations point to GitHub format
- [ ] Test parser functionality with GitHub data format
- [ ] Update any custom scripts that referenced Kaggle paths
- [ ] Remove `kaggle` package from your environment (if not needed elsewhere)

## 🔄 Side-by-Side Comparison

### Dataset Download

#### Old Way (Kaggle-Based)

```bash
# Required Kaggle CLI installation and API credentials
pip install kaggle
kaggle config set -n username -v your_username
kaggle config set -n key -v your_api_key

# Download commands
kaggle competitions download -c arc-prize-2024
kaggle competitions download -c arc-prize-2025

# Manual extraction and organization required
unzip arc-prize-2024.zip
unzip arc-prize-2025.zip
```

#### New Way (GitHub-Based)

```bash
# No external dependencies or credentials required
# Direct repository cloning with automatic organization

# Download individual datasets
python scripts/download_dataset.py arc-agi-1
python scripts/download_dataset.py arc-agi-2

# Or download all at once
python scripts/download_dataset.py all
```

### Data Format

#### Old Format (Kaggle)

```text
data/raw/arc-prize-2024/
├── arc-agi_training_challenges.json    # Combined file
├── arc-agi_training_solutions.json     # Combined file
├── arc-agi_evaluation_challenges.json  # Combined file
└── arc-agi_evaluation_solutions.json   # Combined file
```

#### New Format (GitHub)

```text
data/raw/ARC-AGI-1/
└── data/
    ├── training/
    │   ├── 007bbfb7.json              # Individual task files
    │   ├── 00d62c1b.json
    │   └── ... (400 training tasks)
    └── evaluation/
        ├── 00576224.json
        ├── 009d5c81.json
        └── ... (400 evaluation tasks)
```

### Configuration Files

#### Old Configuration

```yaml
# conf/dataset/arc_agi_1.yaml (Kaggle format)
training:
  challenges: "data/raw/arc-prize-2024/arc-agi_training_challenges.json"
  solutions: "data/raw/arc-prize-2024/arc-agi_training_solutions.json"
evaluation:
  challenges: "data/raw/arc-prize-2024/arc-agi_evaluation_challenges.json"
  solutions: "data/raw/arc-prize-2024/arc-agi_evaluation_solutions.json"
```

#### New Configuration

```yaml
# conf/dataset/arc_agi_1.yaml (GitHub format)
training:
  path: "data/raw/ARC-AGI-1/data/training"
evaluation:
  path: "data/raw/ARC-AGI-1/data/evaluation"
```

## 🚀 Step-by-Step Migration

### Step 1: Remove Kaggle Dependencies (Optional)

If you no longer need Kaggle for other projects:

```bash
# Remove Kaggle CLI (if installed globally)
pip uninstall kaggle

# Remove Kaggle credentials (optional)
rm -rf ~/.kaggle/
```

### Step 2: Download Datasets Using New System

```bash
# Download ARC-AGI datasets from GitHub
python scripts/download_dataset.py arc-agi-1
python scripts/download_dataset.py arc-agi-2

# Verify download success
ls -la data/raw/ARC-AGI-1/data/training/
ls -la data/raw/ARC-AGI-2/data/training/
```

### Step 3: Update Configuration (Automatic)

The new parser automatically detects and handles GitHub format. If you have old Kaggle configurations, you'll see helpful error messages:

```python
# Old Kaggle config will show:
# RuntimeError: Legacy Kaggle format detected. Please update configuration 
# to use GitHub format with 'path' instead of 'challenges'/'solutions'
```

### Step 4: Test Parser Functionality

```python
from jaxarc.parsers import ArcAgiParser
from omegaconf import DictConfig

# Test with new GitHub format
config = DictConfig({
    "training": {"path": "data/raw/ARC-AGI-1/data/training"},
    "evaluation": {"path": "data/raw/ARC-AGI-1/data/evaluation"},
    "grid": {"max_grid_height": 30, "max_grid_width": 30},
    "max_train_pairs": 10,
    "max_test_pairs": 3,
})

parser = ArcAgiParser(config)
print(f"Successfully loaded {len(parser.get_available_task_ids())} tasks")
```

### Step 5: Clean Up Old Data (Optional)

```bash
# Remove old Kaggle data if no longer needed
rm -rf data/raw/arc-prize-2024/
rm -rf data/raw/arc-prize-2025/
```

## 🔧 Common Migration Issues

### Issue 1: "Legacy Kaggle format detected"

**Problem**: Using old configuration format with `challenges` and `solutions` keys.

**Solution**: Update configuration to use `path` key:

```yaml
# OLD (Kaggle format)
training:
  challenges: "path/to/challenges.json"
  solutions: "path/to/solutions.json"

# NEW (GitHub format)
training:
  path: "data/raw/ARC-AGI-1/data/training"
```

### Issue 2: "No data path specified"

**Problem**: Configuration missing required `path` field.

**Solution**: Add path to GitHub repository data directory:

```yaml
training:
  path: "data/raw/ARC-AGI-1/data/training"
evaluation:
  path: "data/raw/ARC-AGI-1/data/evaluation"
```

### Issue 3: "Task files not found"

**Problem**: Incorrect path or dataset not downloaded.

**Solution**: Verify download and path:

```bash
# Re-download dataset
python scripts/download_dataset.py arc-agi-1

# Check file structure
ls -la data/raw/ARC-AGI-1/data/training/
```

### Issue 4: Performance differences

**Problem**: Concerned about performance with individual files vs combined files.

**Solution**: The new format is actually more efficient:

- **Faster Loading**: Only loads needed tasks instead of entire dataset
- **Better Memory Usage**: Reduced memory footprint
- **Parallel Processing**: Can process tasks in parallel more easily

## 📊 Benefits of GitHub Migration

### Reliability Improvements

| Aspect | Kaggle | GitHub |
|--------|--------|--------|
| **Authentication** | API credentials required | No authentication needed |
| **Rate Limits** | API rate limits apply | No rate limits |
| **Availability** | Depends on Kaggle API | Direct repository access |
| **Maintenance** | External dependency | Self-contained |

### Performance Improvements

| Metric | Kaggle Format | GitHub Format |
|--------|---------------|---------------|
| **Download Speed** | Limited by API | Full bandwidth |
| **Loading Time** | Load entire dataset | Load only needed tasks |
| **Memory Usage** | High (combined files) | Lower (individual files) |
| **Parallel Processing** | Limited | Excellent |

### Development Experience

- **Simpler Setup**: No external CLI tools or credentials
- **Better Debugging**: Individual task files easier to inspect
- **Version Control**: Can track individual task changes
- **Offline Development**: Works without internet after initial download

## 🧪 Validation Script

Use this script to validate your migration:

```python
#!/usr/bin/env python3
"""Validation script for Kaggle to GitHub migration."""

import jax
from pathlib import Path
from jaxarc.parsers import ArcAgiParser
from omegaconf import DictConfig

def validate_migration():
    """Validate successful migration from Kaggle to GitHub format."""
    
    # Check if GitHub data exists
    github_paths = [
        "data/raw/ARC-AGI-1/data/training",
        "data/raw/ARC-AGI-1/data/evaluation",
        "data/raw/ARC-AGI-2/data/training", 
        "data/raw/ARC-AGI-2/data/evaluation"
    ]
    
    for path_str in github_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"❌ Missing: {path}")
            print(f"   Run: python scripts/download_dataset.py {path.parts[2].lower()}")
            continue
            
        task_files = list(path.glob("*.json"))
        print(f"✅ Found {len(task_files)} tasks in {path}")
    
    # Test parser functionality
    try:
        config = DictConfig({
            "training": {"path": "data/raw/ARC-AGI-1/data/training"},
            "evaluation": {"path": "data/raw/ARC-AGI-1/data/evaluation"},
            "grid": {"max_grid_height": 30, "max_grid_width": 30},
            "max_train_pairs": 10,
            "max_test_pairs": 3,
        })
        
        parser = ArcAgiParser(config)
        task_ids = parser.get_available_task_ids()
        
        # Test loading a random task
        if task_ids:
            key = jax.random.PRNGKey(42)
            task = parser.get_random_task(key)
            print(f"✅ Successfully loaded random task with {task.num_train_pairs} training pairs")
        
        print(f"✅ Parser validation successful: {len(task_ids)} tasks available")
        
    except Exception as e:
        print(f"❌ Parser validation failed: {e}")
        return False
    
    print("\n🎉 Migration validation completed successfully!")
    return True

if __name__ == "__main__":
    validate_migration()
```

## 📚 Additional Resources

- [Parser Usage Guide](parser_usage.md) - Updated for GitHub format
- [Data Format Documentation](data_format.md) - GitHub repository structure
- [Download Script Documentation](../scripts/download_dataset.py) - New unified download system
- [Configuration Examples](../conf/dataset/) - Updated configuration files

## 🤝 Getting Help

If you encounter issues during migration:

1. Check the [common issues section](#common-migration-issues) above
2. Run the [validation script](#validation-script) to identify problems
3. Review the [examples](../examples/) for reference implementations
4. Open an issue on [GitHub](https://github.com/aadimator/JaxARC/issues)
5. Start a discussion on [GitHub Discussions](https://github.com/aadimator/JaxARC/discussions)

## 🎯 Migration Summary

The Kaggle to GitHub migration provides:

- ✅ **Simplified Setup**: No external dependencies or credentials
- ✅ **Better Reliability**: Direct repository access without API limits
- ✅ **Improved Performance**: Faster downloads and more efficient loading
- ✅ **Unified System**: Same download mechanism for all datasets
- ✅ **Better Development Experience**: Individual task files and offline support

The migration is designed to be seamless with automatic format detection and helpful error messages to guide you through any necessary updates.