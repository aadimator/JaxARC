# API Reference

This document provides a comprehensive reference for the JaxARC API, including
all parser classes, configuration utilities, and core functionality.

## Parser Classes

### ArcAgiParser

General parser for ARC-AGI datasets from GitHub repositories. Supports both ARC-AGI-1 (2024) and ARC-AGI-2 (2025) datasets with individual JSON task files.

```python
from jaxarc.parsers import ArcAgiParser


class ArcAgiParser(ArcDataParserBase):
    """Parser for ARC-AGI datasets from GitHub repositories.
    
    Handles individual JSON task files instead of combined Kaggle format.
    Automatically detects and validates GitHub repository structure.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize parser with GitHub format configuration.
        
        Args:
            cfg: Configuration with 'path' fields pointing to directories
                 containing individual JSON task files.
                 
        Raises:
            RuntimeError: If legacy Kaggle format detected or paths invalid.
        """

    def get_available_task_ids(self) -> list[str]:
        """Get list of all available task IDs.
        
        Task IDs are derived from JSON filenames (without .json extension).
        """

    def get_task_by_id(self, task_id: str) -> JaxArcTask:
        """Get specific task by ID.
        
        Args:
            task_id: Task identifier (JSON filename without extension)
            
        Returns:
            JaxArcTask with preprocessed data and static shapes
        """

    def get_random_task(self, key: chex.PRNGKey) -> JaxArcTask:
        """Get a random task from the dataset.
        
        Args:
            key: JAX PRNG key for random selection
            
        Returns:
            Randomly selected JaxArcTask
        """

    def load_task_file(self, task_file_path: str) -> dict:
        """Load raw task data from individual JSON file.
        
        Args:
            task_file_path: Path to individual task JSON file
            
        Returns:
            Raw task data dictionary with 'train' and 'test' sections
            
        Raises:
            FileNotFoundError: If task file doesn't exist
            ValueError: If JSON is malformed
        """

    def preprocess_task_data(self, raw_task_data: dict, key: chex.PRNGKey) -> JaxArcTask:
        """Convert raw GitHub format task data into JaxArcTask.
        
        Args:
            raw_task_data: Raw task data from JSON file
            key: JAX PRNG key for preprocessing
            
        Returns:
            JaxArcTask with static shapes and proper padding
        """
```

**Supported Datasets:**

- **ARC-AGI-1 (2024)**: 400 training + 400 evaluation tasks from `fchollet/ARC-AGI`
- **ARC-AGI-2 (2025)**: 1000 training + 120 evaluation tasks from `arcprize/ARC-AGI-2`

**GitHub Format Structure:**

```text
data/raw/ARC-AGI-1/
└── data/
    ├── training/
    │   ├── 007bbfb7.json    # Individual task files
    │   ├── 00d62c1b.json
    │   └── ... (400 files)
    └── evaluation/
        ├── 00576224.json
        └── ... (400 files)
```

**Configuration (GitHub Format):**

```yaml
dataset_name: "ARC-AGI-1"
dataset_year: 2024
description: "ARC-AGI-1 dataset from GitHub (fchollet/ARC-AGI)"

default_split: "training"

# GitHub format uses directory paths
data_root: "data/raw/ARC-AGI-1"
training:
  path: "${dataset.data_root}/data/training"
evaluation:
  path: "${dataset.data_root}/data/evaluation"

# Parser configuration
parser:
  _target_: jaxarc.parsers.ArcAgiParser
  description: "ARC-AGI parser for GitHub format datasets"

# Grid and task constraints
grid:
  max_grid_height: 30
  max_grid_width: 30
  min_grid_height: 1
  min_grid_width: 1
  max_colors: 10
  background_color: 0

max_train_pairs: 10
max_test_pairs: 3
```

**Migration from Kaggle Format:**

The parser automatically detects legacy Kaggle format and provides helpful error messages:

```python
# Legacy Kaggle format (no longer supported)
config = DictConfig({
    "training": {
        "challenges": "path/to/challenges.json",  # Old format
        "solutions": "path/to/solutions.json"
    }
})

# Will raise: RuntimeError: Legacy Kaggle format detected. 
# Please update configuration to use GitHub format with 'path' instead of 'challenges'/'solutions'

# New GitHub format
config = DictConfig({
    "training": {"path": "data/raw/ARC-AGI-1/data/training"},
    "evaluation": {"path": "data/raw/ARC-AGI-1/data/evaluation"}
})
```

### ConceptArcParser

Specialized parser for ConceptARC dataset with concept group organization.

```python
from jaxarc.parsers import ConceptArcParser


class ConceptArcParser(ArcDataParserBase):
    """Parser for ConceptARC dataset with concept group organization."""

    def __init__(self, cfg: DictConfig):
        """Initialize parser with ConceptARC configuration."""

    def get_concept_groups(self) -> list[str]:
        """Get list of available concept groups."""

    def get_random_task_from_concept(
        self, concept: str, key: chex.PRNGKey
    ) -> JaxArcTask:
        """Get random task from specific concept group."""

    def get_tasks_in_concept(self, concept: str) -> list[str]:
        """Get all task IDs in a specific concept group."""

    def get_task_by_id(self, task_id: str) -> JaxArcTask:
        """Get specific task by ID (format: 'ConceptGroup/TaskName')."""

    def get_task_metadata(self, task_id: str) -> dict:
        """Get metadata for a specific task."""

    def get_dataset_statistics(self) -> dict:
        """Get comprehensive dataset statistics."""
```

**Concept Groups:**

- **Spatial**: AboveBelow, Center, InsideOutside, TopBottom2D, TopBottom3D
- **Pattern**: Copy, CompleteShape, SameDifferent, Order
- **Object**: ExtractObjects, MoveToBoundary, ExtendToBoundary
- **Property**: FilledNotFilled, Count, CleanUp, HorizontalVertical

**Configuration:**

```yaml
dataset_name: "ConceptARC"
data_root: "data/raw/ConceptARC"
corpus:
  path: "${dataset.data_root}/corpus"
  concept_groups:
    [
      "AboveBelow",
      "Center",
      "CleanUp",
      "CompleteShape",
      "Copy",
      "Count",
      "ExtendToBoundary",
      "ExtractObjects",
      "FilledNotFilled",
      "HorizontalVertical",
      "InsideOutside",
      "MoveToBoundary",
      "Order",
      "SameDifferent",
      "TopBottom2D",
      "TopBottom3D",
    ]
max_train_pairs: 4
max_test_pairs: 3
```

### MiniArcParser

Optimized parser for MiniARC dataset with 5x5 grids and rapid prototyping
capabilities.

```python
from jaxarc.parsers import MiniArcParser


class MiniArcParser(ArcDataParserBase):
    """Parser for MiniARC dataset optimized for 5x5 grids."""

    def __init__(self, cfg: DictConfig):
        """Initialize parser with MiniARC configuration.

        Automatically validates grid constraints and logs warnings
        for suboptimal configurations (e.g., max_grid_height/width > 5).
        """

    def load_task_file(self, task_file_path: str) -> Any:
        """Load raw task data from a JSON file.

        Raises:
            FileNotFoundError: If task file doesn't exist
            ValueError: If JSON is invalid
        """

    def preprocess_task_data(self, raw_task_data: Any, key: chex.PRNGKey) -> JaxArcTask:
        """Convert raw task data into JaxArcTask structure.

        Validates grid sizes and rejects tasks exceeding 5x5 constraint.
        """

    def get_available_task_ids(self) -> list[str]:
        """Get list of all available task IDs.

        Task IDs are derived from filenames without .json extension.
        """

    def get_task_by_id(self, task_id: str) -> JaxArcTask:
        """Get specific task by ID.

        Args:
            task_id: Task identifier (filename without .json)

        Raises:
            ValueError: If task ID not found in dataset
        """

    def get_dataset_statistics(self) -> dict:
        """Get comprehensive dataset statistics.

        Returns:
            Dictionary containing:
            - total_tasks: Number of loaded tasks
            - optimization: "5x5 grids"
            - max_configured_dimensions: Grid size configuration
            - is_5x5_optimized: Whether optimally configured
            - train_pairs: Min/max/avg training pairs
            - test_pairs: Min/max/avg test pairs
            - grid_dimensions: Actual grid size statistics
        """

    def get_random_task(self, key: chex.PRNGKey) -> JaxArcTask:
        """Get random task optimized for 5x5 processing.

        Raises:
            RuntimeError: If no tasks available in dataset
        """

    # Private validation methods (used internally)
    def _validate_grid_size(self, grid: list, context: str) -> None:
        """Validate grid doesn't exceed 5x5 constraint."""

    def _validate_grid_colors(self, grid: chex.Array) -> None:
        """Validate grid colors are within valid range."""

    def _validate_task_structure(self, task_data: dict, task_id: str) -> None:
        """Validate task has required structure (train/test sections)."""
```

**Performance Benefits:**

- **Memory**: 36x less memory per grid (25 vs 900 cells)
- **Speed**: 10-50x faster processing and training
- **Batch Size**: Support for larger batch sizes
- **Development**: Seconds to minutes vs hours for iteration cycles

**Configuration:**

```yaml
dataset_name: "MiniARC"
data_root: "data/raw/MiniARC"
tasks:
  path: "${dataset.data_root}/data/MiniARC"
grid:
  max_grid_height: 5
  max_grid_width: 5
max_train_pairs: 3
max_test_pairs: 1
optimization:
  enable_5x5_optimizations: true
  fast_processing: true
  reduced_memory_usage: true
```

## Configuration Factory Functions

### Dataset-Specific Configurations

```python
from jaxarc.envs.factory import create_conceptarc_config, create_miniarc_config


def create_conceptarc_config(
    max_episode_steps: int = 150,
    task_split: str = "corpus",
    reward_on_submit_only: bool = True,
    success_bonus: float = 20.0,
    step_penalty: float = -0.01,
    **kwargs
) -> ArcEnvConfig:
    """Create ConceptARC-optimized environment configuration."""


def create_miniarc_config(
    max_episode_steps: int = 50,
    task_split: str = "tasks",
    reward_on_submit_only: bool = False,
    success_bonus: float = 5.0,
    step_penalty: float = -0.001,
    **kwargs
) -> ArcEnvConfig:
    """Create MiniARC-optimized environment configuration."""
```

### Configuration Utilities

```python
from jaxarc.utils.config import create_conceptarc_config, create_miniarc_config


def create_conceptarc_config(
    max_episode_steps: int = 100,
    task_split: str = "corpus",
    success_bonus: float = 15.0,
    **kwargs
) -> DictConfig:
    """Create ConceptARC configuration for parser usage."""


def create_miniarc_config(
    max_episode_steps: int = 80,
    task_split: str = "training",
    success_bonus: float = 5.0,
    **kwargs
) -> DictConfig:
    """Create MiniARC configuration for parser usage."""
```

## Core Data Types

### JaxArcTask

Main data structure for ARC tasks, compatible with all parsers.

```python
from jaxarc.types import JaxArcTask


@chex.dataclass
class JaxArcTask:
    """JAX-compatible ARC task representation."""

    input_grids_examples: chex.Array  # Shape: (max_train_pairs, H, W)
    input_masks_examples: chex.Array  # Shape: (max_train_pairs, H, W)
    output_grids_examples: chex.Array  # Shape: (max_train_pairs, H, W)
    output_masks_examples: chex.Array  # Shape: (max_train_pairs, H, W)
    num_train_pairs: int

    test_input_grids: chex.Array  # Shape: (max_test_pairs, H, W)
    test_input_masks: chex.Array  # Shape: (max_test_pairs, H, W)
    true_test_output_grids: chex.Array  # Shape: (max_test_pairs, H, W)
    true_test_output_masks: chex.Array  # Shape: (max_test_pairs, H, W)
    num_test_pairs: int

    task_index: chex.Array  # Unique task identifier
```

### Grid

2D color grid representation.

```python
from jaxarc.types import Grid

Grid = chex.Array  # Shape: (height, width), dtype: int32
```

## Environment Integration

### Functional API

```python
from jaxarc.envs import arc_reset, arc_step


def arc_reset(
    key: chex.PRNGKey, config: ArcEnvConfig, task_data: JaxArcTask | None = None
) -> tuple[ArcEnvState, chex.Array]:
    """Reset environment with optional task data."""


def arc_step(
    state: ArcEnvState, action: dict[str, chex.Array], config: ArcEnvConfig
) -> tuple[ArcEnvState, chex.Array, float, bool, dict]:
    """Step environment with action."""
```

### Class-Based API

```python
from jaxarc.envs import ArcEnvironment


class ArcEnvironment:
    """JAX-compatible ARC environment."""

    def __init__(self, config: ArcEnvConfig):
        """Initialize environment with configuration."""

    def reset(
        self, key: chex.PRNGKey, task_data: JaxArcTask | None = None
    ) -> tuple[ArcEnvState, chex.Array]:
        """Reset environment."""

    def step(
        self, action: dict[str, chex.Array]
    ) -> tuple[ArcEnvState, chex.Array, float, dict]:
        """Step environment."""
```

## Usage Examples

### ConceptARC Example

```python
import jax
from jaxarc.parsers import ConceptArcParser
from jaxarc.envs import create_conceptarc_config, ArcEnvironment
from omegaconf import DictConfig

# Create parser configuration
parser_config = DictConfig(
    {
        "corpus": {"path": "data/raw/ConceptARC/corpus"},
        "grid": {"max_grid_height": 30, "max_grid_width": 30},
        "max_train_pairs": 4,
        "max_test_pairs": 3,
    }
)

# Initialize parser
parser = ConceptArcParser(parser_config)

# Explore concept groups
concepts = parser.get_concept_groups()
print(f"Available concepts: {concepts}")

# Get task from specific concept
key = jax.random.PRNGKey(42)
task = parser.get_random_task_from_concept("Center", key)

# Create environment configuration
env_config = create_conceptarc_config(max_episode_steps=150, success_bonus=20.0)

# Run environment
env = ArcEnvironment(env_config)
state, obs = env.reset(key, task_data=task)
```

### MiniARC Example

```python
import jax
from jaxarc.parsers import MiniArcParser
from jaxarc.envs import create_miniarc_config, ArcEnvironment
from omegaconf import DictConfig

# Create parser configuration
parser_config = DictConfig(
    {
        "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
        "grid": {"max_grid_height": 5, "max_grid_width": 5},
        "max_train_pairs": 3,
        "max_test_pairs": 1,
    }
)

# Initialize parser
parser = MiniArcParser(parser_config)

# Get random task (5x5 optimized)
key = jax.random.PRNGKey(42)
task = parser.get_random_task(key)

# Create optimized environment configuration
env_config = create_miniarc_config(
    max_episode_steps=50, success_bonus=5.0  # Shorter for rapid iteration
)

# Run environment (10-50x faster than standard ARC)
env = ArcEnvironment(env_config)
state, obs = env.reset(key, task_data=task)
```

## Performance Considerations

### ConceptARC

- **Grid Size**: Up to 30×30 (same as standard ARC)
- **Memory Usage**: Standard ARC memory requirements
- **Use Case**: Systematic evaluation and concept-based analysis
- **Optimization**: Concept-based task sampling and organization

### MiniARC

- **Grid Size**: Maximum 5×5 (36x fewer cells than standard ARC)
- **Memory Usage**: 36x less memory per grid
- **Processing Speed**: 10-50x faster than standard ARC
- **Batch Size**: Support for much larger batch sizes
- **Use Case**: Rapid prototyping, algorithm development, quick experiments

### JAX Compatibility

All parsers and configurations are fully compatible with JAX transformations:

```python
# JIT compilation
@jax.jit
def process_task(task, config):
    return some_processing(task, config)


# Vectorization
batch_process = jax.vmap(process_task, in_axes=(0, None))

# Parallel processing
parallel_process = jax.pmap(process_task, in_axes=(0, None))
```

## Error Handling

### Common Exceptions

- **`FileNotFoundError`**: Dataset files not found
- **`ValueError`**: Invalid configuration or task parameters
- **`KeyError`**: Missing concept group or task ID
- **`ValidationError`**: Configuration validation failures
- **`RuntimeError`**: Legacy Kaggle format detected or GitHub download issues

### Best Practices

1. **Validate configurations** before use
2. **Handle missing datasets** gracefully
3. **Use appropriate grid sizes** for each dataset type
4. **Check task availability** before accessing specific tasks
5. **Monitor memory usage** with large batch sizes

## Troubleshooting GitHub Downloads

### Common Download Issues

#### Issue 1: "Repository not found" or "Permission denied"

**Problem**: Git clone fails with repository access errors.

**Solutions:**
```bash
# Check internet connectivity
ping github.com

# Verify repository URLs are accessible
curl -I https://github.com/fchollet/ARC-AGI
curl -I https://github.com/arcprize/ARC-AGI-2

# Try manual clone to test
git clone https://github.com/fchollet/ARC-AGI.git /tmp/test-clone
```

#### Issue 2: "No space left on device"

**Problem**: Insufficient disk space for dataset download.

**Solutions:**
```bash
# Check available disk space
df -h

# Clean up old datasets if needed
rm -rf data/raw/old-datasets/

# Use custom output directory with more space
python scripts/download_dataset.py arc-agi-1 --output /path/with/more/space
```

#### Issue 3: "Git command not found"

**Problem**: Git is not installed or not in PATH.

**Solutions:**
```bash
# Install git (macOS)
brew install git

# Install git (Ubuntu/Debian)
sudo apt-get install git

# Install git (CentOS/RHEL)
sudo yum install git

# Verify installation
git --version
```

#### Issue 4: "SSL certificate problem"

**Problem**: SSL/TLS certificate verification issues.

**Solutions:**
```bash
# Update certificates (macOS)
brew install ca-certificates

# Update certificates (Ubuntu)
sudo apt-get update && sudo apt-get install ca-certificates

# Temporary workaround (not recommended for production)
git config --global http.sslVerify false
```

#### Issue 5: "Directory already exists"

**Problem**: Target directory exists and conflicts with download.

**Solutions:**
```bash
# Use --force flag to overwrite
python scripts/download_dataset.py arc-agi-1 --force

# Or manually remove existing directory
rm -rf data/raw/ARC-AGI-1/
python scripts/download_dataset.py arc-agi-1
```

#### Issue 6: "Network timeout" or "Connection reset"

**Problem**: Network connectivity issues during download.

**Solutions:**
```bash
# Increase git timeout
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 300

# Use different DNS servers
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

# Try download with verbose output
python scripts/download_dataset.py arc-agi-1 --verbose
```

### Parser Issues with GitHub Format

#### Issue 1: "Legacy Kaggle format detected"

**Problem**: Configuration still uses old Kaggle format.

**Solution:**
```python
# OLD (Kaggle format - no longer supported)
config = DictConfig({
    "training": {
        "challenges": "path/to/challenges.json",
        "solutions": "path/to/solutions.json"
    }
})

# NEW (GitHub format)
config = DictConfig({
    "training": {"path": "data/raw/ARC-AGI-1/data/training"},
    "evaluation": {"path": "data/raw/ARC-AGI-1/data/evaluation"}
})
```

#### Issue 2: "No data path specified in configuration"

**Problem**: Missing required `path` field in configuration.

**Solution:**
```yaml
# Add path fields to configuration
training:
  path: "data/raw/ARC-AGI-1/data/training"
evaluation:
  path: "data/raw/ARC-AGI-1/data/evaluation"
```

#### Issue 3: "Data directory not found"

**Problem**: Dataset not downloaded or incorrect path.

**Solutions:**
```bash
# Verify dataset was downloaded
ls -la data/raw/ARC-AGI-1/data/training/

# Re-download if missing
python scripts/download_dataset.py arc-agi-1

# Check configuration paths match actual structure
```

#### Issue 4: "No JSON files found in directory"

**Problem**: Directory exists but contains no task files.

**Solutions:**
```bash
# Check directory contents
ls -la data/raw/ARC-AGI-1/data/training/

# Verify download completed successfully
python scripts/download_dataset.py arc-agi-1 --force

# Check for hidden files or permission issues
ls -la data/raw/ARC-AGI-1/data/training/.*
```

### Performance Issues

#### Issue 1: Slow task loading

**Problem**: Individual JSON files load slower than expected.

**Solutions:**
```python
# Use task caching
parser = ArcAgiParser(config)
# Tasks are automatically cached after first load

# Pre-load frequently used tasks
task_ids = parser.get_available_task_ids()
for task_id in task_ids[:10]:  # Pre-load first 10 tasks
    parser.get_task_by_id(task_id)
```

#### Issue 2: High memory usage

**Problem**: Loading large datasets consumes too much memory.

**Solutions:**
```python
# Use lazy loading - only load tasks when needed
task = parser.get_random_task(key)  # Loads single task

# Clear cache periodically for long-running processes
parser._cached_tasks.clear()  # Clear internal cache

# Use MiniARC for development/testing
from jaxarc.parsers import MiniArcParser
mini_parser = MiniArcParser(miniarc_config)  # 36x less memory
```

### Validation and Debugging

#### Diagnostic Script

```python
#!/usr/bin/env python3
"""Diagnostic script for GitHub download and parser issues."""

import json
from pathlib import Path
from jaxarc.parsers import ArcAgiParser
from omegaconf import DictConfig

def diagnose_github_setup():
    """Diagnose common GitHub download and parser issues."""
    
    print("🔍 JaxARC GitHub Setup Diagnostics")
    print("=" * 50)
    
    # Check dataset directories
    datasets = {
        "ARC-AGI-1": "data/raw/ARC-AGI-1/data",
        "ARC-AGI-2": "data/raw/ARC-AGI-2/data",
        "ConceptARC": "data/raw/ConceptARC/corpus",
        "MiniARC": "data/raw/MiniARC/data/MiniARC"
    }
    
    for name, path_str in datasets.items():
        path = Path(path_str)
        if path.exists():
            if name.startswith("ARC-AGI"):
                train_files = list((path / "training").glob("*.json"))
                eval_files = list((path / "evaluation").glob("*.json"))
                print(f"✅ {name}: {len(train_files)} train, {len(eval_files)} eval tasks")
            else:
                files = list(path.glob("**/*.json"))
                print(f"✅ {name}: {len(files)} task files")
        else:
            print(f"❌ {name}: Not found at {path}")
            print(f"   Run: python scripts/download_dataset.py {name.lower().replace('-', '-')}")
    
    # Test parser functionality
    print("\n🧪 Testing Parser Functionality")
    print("-" * 30)
    
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
        
        if task_ids:
            # Test loading a task
            sample_task = parser.get_task_by_id(task_ids[0])
            print(f"✅ Parser test successful: {len(task_ids)} tasks available")
            print(f"   Sample task: {sample_task.num_train_pairs} train pairs")
        else:
            print("❌ No tasks found - check dataset download")
            
    except Exception as e:
        print(f"❌ Parser test failed: {e}")
        print("   Check configuration and dataset paths")
    
    # Check git installation
    print("\n🔧 System Requirements")
    print("-" * 20)
    
    import subprocess
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Git: {result.stdout.strip()}")
        else:
            print("❌ Git: Command failed")
    except FileNotFoundError:
        print("❌ Git: Not installed or not in PATH")
        print("   Install with: brew install git (macOS) or apt-get install git (Linux)")
    
    print("\n🎯 Recommendations")
    print("-" * 15)
    print("1. Ensure all datasets are downloaded: python scripts/download_dataset.py all")
    print("2. Use --force flag to re-download if issues persist")
    print("3. Check disk space: df -h")
    print("4. Verify network connectivity: ping github.com")
    print("5. See full troubleshooting guide in docs/api_reference.md")

if __name__ == "__main__":
    diagnose_github_setup()
```

### Getting Help

If issues persist after trying these solutions:

1. **Run the diagnostic script** above to identify specific problems
2. **Check the [Migration Guide](KAGGLE_TO_GITHUB_MIGRATION.md)** for detailed migration steps
3. **Review [examples](../examples/)** for working code patterns
4. **Open an issue** on [GitHub](https://github.com/aadimator/JaxARC/issues) with:
   - Error messages and stack traces
   - Output from diagnostic script
   - System information (OS, Python version, git version)
   - Steps to reproduce the issue
5. **Start a discussion** on [GitHub Discussions](https://github.com/aadimator/JaxARC/discussions) for general questions

## Migration Guide

### From ArcAgi1Parser to ArcAgiParser

```python
# Old way
from jaxarc.parsers import ArcAgi1Parser

parser = ArcAgi1Parser()

# New way
from jaxarc.parsers import ArcAgiParser

parser = ArcAgiParser(config)
```

### Adding ConceptARC Support

```python
# Add ConceptARC to existing code
from jaxarc.parsers import ConceptArcParser
from jaxarc.envs import create_conceptarc_config

# Replace standard parser with ConceptARC
parser = ConceptArcParser(conceptarc_config)
env_config = create_conceptarc_config()
```

### Adding MiniARC for Rapid Prototyping

```python
# Add MiniARC for faster development
from jaxarc.parsers import MiniArcParser
from jaxarc.envs import create_miniarc_config

# Use MiniARC for rapid iteration
parser = MiniArcParser(miniarc_config)
env_config = create_miniarc_config(max_episode_steps=50)
```
