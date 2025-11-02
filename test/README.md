# JaxARC Test Suite

This directory contains comprehensive unit tests for the JaxARC codebase,
focusing on correctness, reliability, and JAX compatibility.

## Structure

```
test/
├── conftest.py              # Pytest configuration and shared fixtures
├── test_utils.py            # JAX-specific testing utilities
├── test_types.py            # Core data structures tests
├── test_state.py            # State management tests
├── configs/                 # Configuration system tests
├── envs/                    # Environment tests
├── parsers/                 # Parser tests
├── utils/                   # Utility tests
├── integration/             # Integration tests
└── fixtures/                # Test data and fixtures
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pixi run -e test pytest

# Run with coverage
pixi run -e test pytest --cov=src/jaxarc --cov-report=html

# Run specific test categories
pixi run -e test pytest test/configs/
pixi run -e test pytest test/integration/

# Run tests with markers
pixi run -e test pytest -m "not slow"
pixi run -e test pytest -m "jax_transform"
```

### Test Categories

- **Unit Tests**: Individual functions and classes in isolation
- **JAX Transformation Tests**: Functions under jit, vmap, and pmap
- **Integration Tests**: Multi-component workflows and end-to-end scenarios
- **Property-Based Tests**: Edge cases and robustness using Hypothesis

### Test Markers

- `@pytest.mark.slow`: Marks tests as slow (can be skipped with `-m "not slow"`)
- `@pytest.mark.integration`: Marks integration tests
- `@pytest.mark.jax_transform`: Marks tests that verify JAX transformations

## Testing Utilities

The `test_utils.py` module provides JAX-specific testing utilities:

- `assert_jax_compatible()`: Verify functions work with JAX transformations
- `assert_arrays_equal()`: Compare JAX arrays with appropriate tolerance
- `assert_pytree_structure()`: Validate PyTree structure
- `create_mock_grid()`: Create test Grid objects
- `assert_grid_valid()`: Validate Grid objects

## Fixtures

Common fixtures are defined in `conftest.py`:

- `prng_key`: JAX PRNG key for reproducible tests
- `sample_grid`: Standard test grid with known properties
- `sample_task`: Complete JaxArcTask for testing
- `default_config`: Valid JaxArcConfig for testing

## Coverage Goals

- **Line coverage**: 90%+ for core modules
- **Branch coverage**: 85%+ for conditional logic
- **Function coverage**: 100% for public APIs

## JAX-Specific Testing

All tests are designed to work with JAX's functional programming model:

- Pure functions are tested under JIT compilation
- Array operations are validated with `chex` assertions
- PyTree structures are verified for consistency
- Static shapes are maintained throughout transformations
