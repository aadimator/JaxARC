# Design Document

## Overview

This design migrates ARC-AGI dataset downloading from Kaggle to GitHub repositories and consolidates all dataset downloading under a unified API. The migration addresses the need to remove Kaggle dependencies and standardize dataset access across all supported datasets. The key challenge is handling the different data formats between Kaggle (combined JSON files) and GitHub (individual JSON task files).

## Architecture

### Data Format Differences

**Kaggle Format (Current):**
- Combined JSON files: `arc-agi_training_challenges.json`, `arc-agi_training_solutions.json`
- Structure: `{"task_id": {"train": [...], "test": [...]}}`
- Challenges and solutions stored separately
- Solutions contain only test outputs as arrays

**GitHub Format (Target):**
- Individual JSON files per task: `007bbfb7.json`, `00d62c1b.json`, etc.
- Structure: `{"train": [...], "test": [...]}` (direct task content)
- Complete task data in single file including test outputs when available
- No separate challenges/solutions files

### Repository Structure Analysis

**ARC-AGI-1 (fchollet/ARC-AGI):**
- Repository: `https://github.com/fchollet/ARC-AGI`
- Structure: `data/training/` and `data/evaluation/` directories
- 400 training tasks, 400 evaluation tasks
- Individual JSON files per task

**ARC-AGI-2 (arcprize/ARC-AGI-2):**
- Repository: `https://github.com/arcprize/ARC-AGI-2`
- Structure: `data/training/` and `data/evaluation/` directories
- 1000 training tasks, 120 evaluation tasks
- Individual JSON files per task
- Additional `.txt` files listing task IDs

## Components and Interfaces

### Enhanced DatasetDownloader

```python
class DatasetDownloader:
    """Unified dataset downloader supporting GitHub repositories."""
    
    def download_arc_agi_1(self, target_dir: Optional[Path] = None) -> Path:
        """Download ARC-AGI-1 from GitHub repository."""
        repo_url = "https://github.com/fchollet/ARC-AGI.git"
        repo_name = "ARC-AGI-1"
        return self._clone_repository(repo_url, target_dir, repo_name)
    
    def download_arc_agi_2(self, target_dir: Optional[Path] = None) -> Path:
        """Download ARC-AGI-2 from GitHub repository."""
        repo_url = "https://github.com/arcprize/ARC-AGI-2.git"
        repo_name = "ARC-AGI-2"
        return self._clone_repository(repo_url, target_dir, repo_name)
```

### Updated ArcAgiParser

The existing `ArcAgiParser` needs to be updated to handle GitHub format:

```python
class ArcAgiParser(ArcDataParserBase):
    """Parser for ARC-AGI datasets from GitHub repositories."""
    
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._task_ids: list[str] = []
        self._cached_tasks: dict[str, dict] = {}
        self._load_and_cache_tasks()
    
    def _load_and_cache_tasks(self) -> None:
        """Load and cache tasks from GitHub format (individual JSON files)."""
        try:
            default_split = self.cfg.get("default_split", "training")
            split_config = self.cfg.get(default_split, {})
            
            # GitHub format uses directory paths
            data_dir_path = split_config.get("path")
            if not data_dir_path:
                raise RuntimeError("No data path specified in configuration")
            
            data_dir = here(data_dir_path)
            if not data_dir.exists() or not data_dir.is_dir():
                raise RuntimeError(f"Data directory not found: {data_dir}")
            
            # Load individual JSON files
            json_files = list(data_dir.glob("*.json"))
            if not json_files:
                raise RuntimeError(f"No JSON files found in {data_dir}")
            
            self._cached_tasks = {}
            for json_file in json_files:
                task_id = json_file.stem  # filename without extension
                with json_file.open("r", encoding="utf-8") as f:
                    task_data = json.load(f)
                self._cached_tasks[task_id] = task_data
            
            self._task_ids = list(self._cached_tasks.keys())
            logger.info(f"Loaded {len(self._cached_tasks)} tasks from GitHub format")
            
        except Exception as e:
            logger.error(f"Error loading and caching tasks: {e}")
            raise
```

### Updated Configuration Files

**ARC-AGI-1 Configuration (`conf/dataset/arc_agi_1.yaml`):**

```yaml
# ARC-AGI-1 Dataset Configuration (GitHub Format)
dataset_name: "ARC-AGI-1"
dataset_year: 2024
description: "ARC-AGI-1 dataset from GitHub (fchollet/ARC-AGI)"

default_split: "training"

# Data Paths - GitHub format uses directories
data_root: "data/raw/ARC-AGI-1"
training:
  path: "${dataset.data_root}/data/training"
evaluation:
  path: "${dataset.data_root}/data/evaluation"

# Parser Configuration
parser:
  _target_: jaxarc.parsers.ArcAgiParser
  description: "ARC-AGI parser for GitHub format datasets"

# Grid Configuration
grid:
  max_grid_height: 30
  max_grid_width: 30
  min_grid_height: 1
  min_grid_width: 1
  max_colors: 10
  background_color: 0

# Task Configuration
max_train_pairs: 10
max_test_pairs: 3
```

**ARC-AGI-2 Configuration (`conf/dataset/arc_agi_2.yaml`):**

```yaml
# ARC-AGI-2 Dataset Configuration (GitHub Format)
dataset_name: "ARC-AGI-2"
dataset_year: 2025
description: "ARC-AGI-2 dataset from GitHub (arcprize/ARC-AGI-2)"

default_split: "training"

# Data Paths - GitHub format uses directories
data_root: "data/raw/ARC-AGI-2"
training:
  path: "${dataset.data_root}/data/training"
evaluation:
  path: "${dataset.data_root}/data/evaluation"

# Parser Configuration
parser:
  _target_: jaxarc.parsers.ArcAgiParser
  description: "ARC-AGI parser for GitHub format datasets"

# Grid Configuration
grid:
  max_grid_height: 30
  max_grid_width: 30
  min_grid_height: 1
  min_grid_width: 1
  max_colors: 10
  background_color: 0

# Task Configuration
max_train_pairs: 10
max_test_pairs: 4  # ARC-AGI-2 can have up to 2 test pairs typically
```

### Streamlined CLI Interface

**Enhanced Download Script (`scripts/download_dataset.py`):**

```python
app = typer.Typer(
    help="Download ARC datasets from GitHub repositories",
    epilog="""
Examples:
  # Download specific datasets
  python scripts/download_dataset.py arc-agi-1
  python scripts/download_dataset.py arc-agi-2
  python scripts/download_dataset.py conceptarc
  python scripts/download_dataset.py miniarc

  # Download all datasets
  python scripts/download_dataset.py all

  # Download with custom options
  python scripts/download_dataset.py arc-agi-1 --output /custom/path --force
    """,
    rich_markup_mode="rich",
)

@app.command()
def arc_agi_1(
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Output directory (default: configured raw data path)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download ARC-AGI-1 dataset from GitHub (fchollet/ARC-AGI)."""
    
@app.command()
def arc_agi_2(
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Output directory (default: configured raw data path)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download ARC-AGI-2 dataset from GitHub (arcprize/ARC-AGI-2)."""

@app.command()
def conceptarc(
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Output directory (default: configured raw data path)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download ConceptARC dataset from GitHub."""

@app.command()
def miniarc(
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Output directory (default: configured raw data path)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download MiniARC dataset from GitHub."""

@app.command()
def all(
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Output directory (default: configured raw data path)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download all ARC datasets from GitHub."""
```

## Data Models

### No Changes to JaxArcTask

The existing `JaxArcTask` structure remains unchanged as the GitHub format provides the same task data structure after parsing.

## Error Handling

### Download Error Handling

- **Repository Access**: Handle GitHub repository access issues
- **Network Connectivity**: Robust network error handling
- **Disk Space**: Check available space before cloning
- **Permissions**: Validate write permissions
- **Partial Downloads**: Recovery from interrupted downloads

### Parser Error Handling

- **Missing Files**: Clear error messages when JSON files are missing
- **Invalid JSON**: Handle malformed JSON files gracefully
- **Directory Structure**: Validate expected directory structure
- **Task Validation**: Ensure task data meets expected format

## Testing Strategy

### Unit Tests

- **GitHub Format Parsing**: Test individual JSON file loading
- **Configuration Loading**: Test new configuration structures
- **Error Handling**: Test various failure scenarios
- **Task Validation**: Ensure parsed tasks match expected format

### Integration Tests

- **End-to-End Download**: Test complete download and parsing workflow
- **CLI Interface**: Test streamlined command interface
- **Dataset Validation**: Verify downloaded datasets have correct structure

## Implementation Phases

### Phase 1: Enhanced Parser
- Update `ArcAgiParser` to support GitHub format
- Replace Kaggle format loading with individual JSON file loading
- Update task preprocessing for new format

### Phase 2: Download System Enhancement
- Add GitHub repository support to `DatasetDownloader`
- Implement ARC-AGI-1 and ARC-AGI-2 download methods
- Add validation for GitHub dataset structure

### Phase 3: Configuration Updates
- Update configuration files for GitHub format
- Replace file paths with directory paths
- Update existing examples to use GitHub format

### Phase 4: CLI Streamlining
- Simplify download script interface
- Replace Kaggle commands with GitHub commands
- Add comprehensive help and examples

### Phase 5: Cleanup and Documentation
- Remove Kaggle dependencies from requirements
- Update documentation and migration guides
- Remove Kaggle-specific code

## Performance Considerations

### Optimization
- **Caching**: Efficient caching for individual file loading
- **Lazy Loading**: Load tasks on demand for large datasets
- **Memory Management**: Optimize memory usage for GitHub format
- **File I/O**: Minimize file system operations during task loading

## Security Considerations

### Repository Validation
- **URL Validation**: Validate GitHub repository URLs
- **Content Verification**: Verify downloaded content integrity
- **Safe Cloning**: Use safe git cloning practices

### Dependency Removal
- **Kaggle CLI**: Remove dependency on Kaggle CLI and credentials
- **Simplified Auth**: No authentication required for public GitHub repositories
- **Reduced Attack Surface**: Fewer external dependencies and tools