# Testing Guide

JaxARC includes comprehensive test coverage for all parser implementations, with
particular emphasis on the MiniARC parser's unique 5x5 grid constraints and
performance optimizations.

## Test Coverage Overview

### MiniARC Parser Tests (`tests/parsers/test_mini_arc.py`)

The MiniARC parser has the most comprehensive test suite with 15+ test methods
covering:

#### Core Functionality Tests

- **Initialization with optimal configuration**: Validates proper setup with 5x5
  grid constraints
- **Initialization with suboptimal configuration**: Tests warning system for
  non-optimal settings
- **Task loading from flat directory structure**: Ensures proper file discovery
  and loading
- **Random task selection**: Validates random sampling functionality
- **Task retrieval by ID**: Tests specific task access by filename-based IDs

#### Validation and Error Handling Tests

- **5x5 grid constraint validation**: Ensures oversized grids are rejected
  during loading
- **Grid size validation methods**: Tests helper methods for dimension checking
- **Grid color validation**: Validates color values are within acceptable range
- **Task structure validation**: Ensures required train/test sections are
  present
- **Missing configuration handling**: Tests graceful handling of missing paths
- **Empty directory handling**: Validates behavior with no available tasks
- **Invalid JSON file handling**: Tests error handling for malformed files

#### Performance and Optimization Tests

- **Performance optimizations**: Validates 5x5-specific optimizations are active
- **Task preprocessing**: Tests conversion to JAX-compatible data structures
- **Dataset statistics**: Validates comprehensive statistics generation
- **Memory efficiency**: Ensures optimal array shapes (5x5 vs 30x30)

#### Integration Tests

- **File loading and caching**: Tests task loading and caching mechanisms
- **Configuration validation**: Ensures proper configuration parameter checking
- **Error message quality**: Validates informative error messages for debugging

## Running Tests

### Full Test Suite

```bash
# Run all tests with coverage
pixi run -e test test

# Run tests with verbose output
pixi run -e test pytest -v

# Run tests with coverage report
pixi run -e test pytest --cov=src/jaxarc --cov-report=html
```

### Parser-Specific Tests

```bash
# Run all parser tests
pixi run -e test pytest tests/parsers/

# Run MiniARC parser tests specifically
pixi run -e test pytest tests/parsers/test_mini_arc.py

# Run ConceptARC parser tests
pixi run -e test pytest tests/parsers/test_concept_arc.py

# Run ARC-AGI parser tests
pixi run -e test pytest tests/parsers/test_arc_agi.py
```

### Individual Test Methods

```bash
# Run specific test method
pixi run -e test pytest tests/parsers/test_mini_arc.py::TestMiniArcParser::test_miniarc_parser_5x5_grid_constraint_validation

# Run tests matching pattern
pixi run -e test pytest -k "miniarc_parser_performance"

# Run tests with specific markers
pixi run -e test pytest -m "integration"
```

## Test Structure and Patterns

### Fixture-Based Testing

The MiniARC tests use comprehensive fixtures for consistent test data:

```python
@pytest.fixture
def sample_miniarc_task():
    """Sample MiniARC task data with 5x5 grids."""
    return {
        "train": [
            {
                "input": [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], ...],
                "output": [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], ...]
            }
        ],
        "test": [{"input": [[0, 3, 0, 3], [3, 0, 3, 0], ...]}]
    }

@pytest.fixture
def miniarc_config():
    """Sample MiniARC configuration optimized for 5x5 grids."""
    return DictConfig({
        "grid": {
            "max_grid_height": 5,
            "max_grid_width": 5,
            "max_colors": 10,
        },
        "max_train_pairs": 3,
        "max_test_pairs": 1,
        "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
    })
```

### Mock-Based Testing

Tests use mocking to isolate functionality and avoid filesystem dependencies:

```python
def test_miniarc_parser_initialization_optimal_config(self, miniarc_config, mock_miniarc_directory):
    """Test MiniArcParser initialization with optimal 5x5 configuration."""
    with patch("jaxarc.parsers.mini_arc.here") as mock_here:
        mock_here.return_value = Path(miniarc_config.tasks.path)
        parser = MiniArcParser(miniarc_config)
        # Test assertions...
```

### Validation Testing

Comprehensive validation testing ensures robust error handling:

```python
def test_miniarc_parser_5x5_grid_constraint_validation(self, miniarc_config):
    """Test 5x5 grid constraint validation during task loading."""
    # Create oversized task that should be rejected
    oversized_task = {
        "train": [{
            "input": [[0, 1, 0, 1, 0, 1], ...],  # 6x6 grid (exceeds 5x5)
            "output": [[1, 0, 1, 0, 1, 0], ...]
        }],
        "test": [{"input": [[0, 2, 0, 2, 0, 2]]}]  # 1x6 grid
    }

    # Verify task is rejected with appropriate error logging
    with patch("jaxarc.parsers.mini_arc.logger") as mock_logger:
        parser = MiniArcParser(miniarc_config)
        mock_logger.error.assert_called()
        # Verify specific error messages...
```

## Test Data Management

### Temporary Test Data

Tests create temporary directories and files to avoid dependencies on external
data:

```python
@pytest.fixture
def mock_miniarc_directory(sample_miniarc_task):
    """Create a mock MiniARC directory structure with flat file organization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tasks_dir = Path(temp_dir) / "data" / "MiniARC"
        tasks_dir.mkdir(parents=True)

        # Create sample task files
        for task_file in ["pattern_reversal_001.json", "color_swap_002.json"]:
            task_path = tasks_dir / task_file
            with task_path.open("w") as f:
                json.dump(sample_miniarc_task, f)

        yield temp_dir
```

### Edge Case Testing

Tests cover various edge cases and error conditions:

```python
def test_miniarc_parser_error_handling_oversized_grids(self, miniarc_config):
    """Test error handling for oversized grids."""
    test_cases = [
        {
            "name": "height_violation.json",
            "task": {"train": [{"input": [[0], [1], [0], [1], [0], [1]]}]}  # 6x1
        },
        {
            "name": "width_violation.json",
            "task": {"train": [{"input": [[0, 1, 0, 1, 0, 1]]}]}  # 1x6
        },
        {
            "name": "both_violation.json",
            "task": {"train": [{"input": 6x6_grid}]}  # 6x6
        }
    ]
    # Test each violation type...
```

## Performance Testing

### Optimization Validation

Tests verify that MiniARC optimizations are properly applied:

```python
def test_miniarc_parser_performance_optimizations(self, miniarc_config, mock_miniarc_directory):
    """Test performance optimizations for 5x5 grids."""
    parser = MiniArcParser(miniarc_config)

    # Verify parser is configured for optimal 5x5 performance
    assert parser.max_grid_height == 5
    assert parser.max_grid_width == 5

    # Verify optimized array shapes
    task = parser.get_random_task(key)
    expected_shape = (miniarc_config.max_train_pairs, 5, 5)
    assert task.input_grids_examples.shape == expected_shape

    # Verify dataset statistics show optimization
    stats = parser.get_dataset_statistics()
    assert stats["optimization"] == "5x5 grids"
    assert stats["is_5x5_optimized"] is True
```

### Memory Efficiency Testing

Tests validate memory-efficient data structures:

```python
def test_miniarc_parser_task_preprocessing(self, miniarc_config, sample_miniarc_task):
    """Test task data preprocessing with MiniARC-specific optimizations."""
    jax_task = parser.preprocess_task_data((task_id, sample_miniarc_task), key)

    # Check optimized array shapes (5x5 instead of 30x30)
    assert jax_task.input_grids_examples.shape == (3, 5, 5)
    assert jax_task.test_input_grids.shape == (1, 5, 5)

    # Verify proper data types and masking
    assert jax_task.input_grids_examples.dtype == jnp.int32
    assert jax_task.input_masks_examples.dtype == jnp.bool_
```

## Test Configuration

### pytest Configuration

The project uses pytest with specific configuration for JAX compatibility:

```ini
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src/jaxarc",
    "--cov-report=term-missing",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

### JAX Testing Considerations

Tests account for JAX-specific requirements:

```python
import jax
import jax.numpy as jnp
import chex

def test_jax_compatibility():
    """Test JAX transformations work correctly."""

    @jax.jit
    def process_task(task):
        return task.input_grids_examples.sum()

    # Test JIT compilation works
    result = process_task(task)
    assert isinstance(result, jnp.ndarray)

    # Test array shapes and types
    chex.assert_shape(task.input_grids_examples, (3, 5, 5))
    chex.assert_type(task.input_grids_examples, jnp.int32)
```

## Continuous Integration

### GitHub Actions

Tests run automatically on all pull requests and commits:

```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: |
    pixi run -e test pytest --cov=src/jaxarc --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Coverage Requirements

The project maintains high test coverage standards:

- **Target**: 95%+ overall coverage
- **MiniARC Parser**: 100% line coverage
- **Critical paths**: 100% branch coverage
- **Error handling**: All exception paths tested

## Writing New Tests

### Test Naming Convention

Follow consistent naming patterns:

```python
def test_{component}_{functionality}_{scenario}():
    """Test {component} {functionality} with {scenario}."""
    pass

# Examples:
def test_miniarc_parser_initialization_optimal_config():
def test_miniarc_parser_5x5_grid_constraint_validation():
def test_miniarc_parser_error_handling_oversized_grids():
```

### Test Structure

Use the Arrange-Act-Assert pattern:

```python
def test_example():
    """Test example functionality."""
    # Arrange
    config = create_test_config()
    parser = MiniArcParser(config)

    # Act
    result = parser.get_random_task(key)

    # Assert
    assert isinstance(result, JaxArcTask)
    assert result.num_train_pairs > 0
```

### Fixture Usage

Leverage existing fixtures and create new ones as needed:

```python
def test_with_fixtures(miniarc_config, sample_miniarc_task, mock_miniarc_directory):
    """Test using multiple fixtures."""
    # Test implementation using provided fixtures
    pass
```

## Debugging Tests

### Running Individual Tests

```bash
# Run single test with output
pixi run -e test pytest tests/parsers/test_mini_arc.py::TestMiniArcParser::test_miniarc_parser_initialization_optimal_config -v -s

# Run with debugger
pixi run -e test pytest --pdb tests/parsers/test_mini_arc.py::TestMiniArcParser::test_specific_method

# Run with print statements visible
pixi run -e test pytest -s tests/parsers/test_mini_arc.py
```

### Test Output Analysis

```bash
# Generate detailed coverage report
pixi run -e test pytest --cov=src/jaxarc --cov-report=html
open htmlcov/index.html

# Show missing coverage
pixi run -e test pytest --cov=src/jaxarc --cov-report=term-missing

# Generate coverage for specific module
pixi run -e test pytest --cov=src/jaxarc/parsers/mini_arc --cov-report=term-missing tests/parsers/test_mini_arc.py
```

## Best Practices

### Test Independence

- Each test should be independent and not rely on other tests
- Use fixtures for shared setup rather than class-level setup
- Clean up resources in teardown or use context managers

### Comprehensive Coverage

- Test both success and failure paths
- Include edge cases and boundary conditions
- Validate error messages and exception types
- Test configuration validation thoroughly

### Performance Considerations

- Mock external dependencies (filesystem, network)
- Use temporary directories for file operations
- Keep test data small but representative
- Measure and validate performance improvements

### Documentation

- Write clear docstrings for test methods
- Include comments explaining complex test logic
- Document expected behavior and edge cases
- Keep test names descriptive and consistent
