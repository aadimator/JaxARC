# Design Document

## Overview

This design adds support for ConceptARC and MiniARC datasets to the JaxARC
project through dedicated parsers, configuration files, and download utilities.
The implementation follows the existing architecture patterns while
accommodating the unique characteristics of each dataset.

## Architecture

### Dataset Characteristics

**ConceptARC:**

- 16 concept groups (AboveBelow, Center, CleanUp, etc.) with 10 tasks each
- 1-4 demonstration pairs per task, 3 test inputs per task
- Same JSON format as original ARC dataset
- Standard ARC grid sizes (up to 30x30)
- Organized in hierarchical directory structure:
  `corpus/{ConceptGroup}/{TaskName}.json`
- Available via GitHub repository: `https://github.com/victorvikram/ConceptARC`

**MiniARC:**

- 400+ individual task files in single directory
- Standard ARC JSON format with train/test structure
- Optimized for 5x5 grids (smaller than standard ARC)
- Descriptive filenames indicating task purpose
- Available via GitHub repository: `https://github.com/KSB21ST/MINI-ARC`

### Parser Design

#### ConceptArcParser

```python
class ConceptArcParser(ArcDataParserBase):
    """Parser for ConceptARC dataset with concept group organization."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._concept_groups: dict[str, list[str]] = {}
        self._task_metadata: dict[str, dict] = {}
        self._load_and_cache_tasks()

    def _load_concept_groups(self) -> None:
        """Load tasks organized by concept groups."""

    def get_random_task_from_concept(
        self, concept: str, key: chex.PRNGKey
    ) -> JaxArcTask:
        """Get random task from specific concept group."""

    def get_concept_groups(self) -> list[str]:
        """Get list of available concept groups."""
```

#### MiniArcParser

```python
class MiniArcParser(ArcDataParserBase):
    """Parser for MiniARC dataset optimized for 5x5 grids."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._validate_grid_constraints()
        self._load_and_cache_tasks()

    def _validate_grid_constraints(self) -> None:
        """Ensure configuration is optimized for 5x5 grids."""
        if self.max_grid_height > 5 or self.max_grid_width > 5:
            logger.warning("MiniARC is optimized for 5x5 grids")
```

### Configuration System

#### ConceptARC Configuration (`conf/dataset/concept_arc.yaml`)

```yaml
# ConceptARC Dataset Configuration
dataset_name: "ConceptARC"
dataset_year: 2023
description:
  "ConceptARC dataset organized around 16 concept groups for systematic
  evaluation"

default_split: "corpus"

# Data Paths
data_root: "data/raw/ConceptARC"
corpus:
  path: "${dataset.data_root}/corpus"
  concept_groups:
    - "AboveBelow"
    - "Center"
    - "CleanUp"
    - "CompleteShape"
    - "Copy"
    - "Count"
    - "ExtendToBoundary"
    - "ExtractObjects"
    - "FilledNotFilled"
    - "HorizontalVertical"
    - "InsideOutside"
    - "MoveToBoundary"
    - "Order"
    - "SameDifferent"
    - "TopBottom2D"
    - "TopBottom3D"

# Parser Configuration
parser:
  _target_: jaxarc.parsers.ConceptArcParser
  description: "ConceptARC parser with concept group organization"

# Grid Configuration - Standard ARC dimensions
grid:
  max_grid_height: 30
  max_grid_width: 30
  min_grid_height: 1
  min_grid_width: 1
  max_colors: 10
  background_color: 0

# Task Configuration
max_train_pairs: 4 # ConceptARC has 1-4 demonstration pairs
max_test_pairs: 3 # ConceptARC has 3 test inputs per task
```

#### MiniARC Configuration (`conf/dataset/mini_arc.yaml`)

```yaml
# MiniARC Dataset Configuration
dataset_name: "MiniARC"
dataset_year: 2022
description: "MiniARC dataset with 5x5 grids for rapid prototyping and testing"

default_split: "tasks"

# Data Paths
data_root: "data/raw/MiniARC"
tasks:
  path: "${dataset.data_root}/data/MiniARC"

# Parser Configuration
parser:
  _target_: jaxarc.parsers.MiniArcParser
  description: "MiniARC parser optimized for 5x5 grids"

# Grid Configuration - Optimized for 5x5
grid:
  max_grid_height: 5
  max_grid_width: 5
  min_grid_height: 1
  min_grid_width: 1
  max_colors: 10
  background_color: 0

# Task Configuration - Smaller for faster processing
max_train_pairs: 3 # Typical for MiniARC tasks
max_test_pairs: 1 # Usually 1 test pair per task
```

### Download System

#### Enhanced Download Script

```python
class DatasetDownloader:
    """Unified dataset downloader supporting multiple sources."""

    def download_conceptarc(self, output_dir: Path) -> None:
        """Download ConceptARC from GitHub repository."""
        repo_url = "https://github.com/victorvikram/ConceptARC.git"
        self._clone_repository(repo_url, output_dir / "ConceptARC")

    def download_miniarc(self, output_dir: Path) -> None:
        """Download MiniARC from GitHub repository."""
        repo_url = "https://github.com/KSB21ST/MINI-ARC.git"
        self._clone_repository(repo_url, output_dir / "MiniARC")

    def _clone_repository(self, repo_url: str, target_dir: Path) -> None:
        """Clone Git repository with error handling."""
```

#### CLI Integration

```python
# Enhanced download_dataset.py
@typer.command()
def download_conceptarc(
    output_dir: Path = typer.Option(None, help="Output directory")
) -> None:
    """Download ConceptARC dataset."""


@typer.command()
def download_miniarc(
    output_dir: Path = typer.Option(None, help="Output directory")
) -> None:
    """Download MiniARC dataset."""
```

## Components and Interfaces

### Parser Interface Extensions

Both new parsers extend `ArcDataParserBase` and implement:

- `load_task_file()`: Handle JSON file loading with dataset-specific paths
- `preprocess_task_data()`: Convert to `JaxArcTask` with appropriate padding
- `get_random_task()`: Random task selection with dataset-specific logic

### Configuration Factory Functions

```python
def create_conceptarc_config() -> DictConfig:
    """Create ConceptARC configuration with standard settings."""


def create_miniarc_config() -> DictConfig:
    """Create MiniARC configuration optimized for 5x5 grids."""
```

### Task Metadata Handling

ConceptARC parser includes concept group metadata:

```python
@dataclass
class ConceptArcTaskMetadata:
    concept_group: str
    task_name: str
    file_path: str
    num_demonstrations: int
    num_test_inputs: int
```

## Data Models

### Enhanced JaxArcTask

No changes needed - existing `JaxArcTask` structure accommodates both datasets
through appropriate padding and masking.

### Configuration Dataclasses

```python
@chex.dataclass
class ConceptArcConfig:
    concept_groups: list[str]
    corpus_path: str


@chex.dataclass
class MiniArcConfig:
    tasks_path: str
    optimize_for_5x5: bool = True
```

## Error Handling

### Dataset-Specific Validation

- **ConceptARC**: Validate concept group directory structure, check for missing
  concept groups
- **MiniARC**: Validate 5x5 grid constraints, warn if configuration exceeds
  optimal dimensions
- **Both**: Handle missing files gracefully, provide clear error messages for
  malformed JSON

### Download Error Handling

- Network connectivity issues
- Git repository access problems
- Disk space and permissions
- Partial download recovery

## Testing Strategy

### Unit Tests

- Parser functionality for both datasets
- Configuration validation
- Task loading and preprocessing
- Error handling scenarios

### Integration Tests

- End-to-end dataset loading
- Configuration factory functions
- Download functionality (mocked)
- JAX compatibility validation

### Dataset-Specific Tests

- ConceptARC concept group organization
- MiniARC 5x5 grid optimization
- Task metadata extraction
- Random sampling from concept groups

## Implementation Phases

### Phase 1: Core Parsers

- Implement `ConceptArcParser` and `MiniArcParser`
- Create configuration files
- Add parser exports to `__init__.py`

### Phase 2: Download System

- Enhance download script with Git cloning
- Add CLI commands for new datasets
- Implement error handling and validation

### Phase 3: Configuration Integration

- Create factory functions
- Add configuration validation
- Update existing examples

### Phase 4: Testing and Documentation

- Comprehensive test suite
- Usage examples
- Documentation updates

## Compatibility Considerations

### Backward Compatibility

- Existing ARC-AGI parsers remain unchanged
- No breaking changes to core interfaces
- Configuration system maintains existing patterns

### JAX Compatibility

- All parsers maintain static array shapes
- Proper padding and masking for efficient JIT compilation
- Consistent with existing `JaxArcTask` structure

### Performance Optimization

- MiniARC parser optimized for smaller grids
- ConceptARC parser includes concept-based sampling
- Efficient caching for repeated access
