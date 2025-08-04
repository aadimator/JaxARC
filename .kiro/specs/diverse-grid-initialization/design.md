# Design Document

## Overview

This design implements diverse working grid initialization strategies for the JaxARC environment to enhance training diversity in batched scenarios. The solution extends the existing configuration and functional API to support multiple initialization modes while maintaining backward compatibility and JAX performance characteristics.

## Architecture

### Configuration Extension

The design extends the existing `ArcEnvConfig` to include initialization parameters:

```python
@equinox.Module
class GridInitializationConfig:
    mode: str = "demo"  # "demo", "permutation", "empty", "random", "mixed"
    demo_weight: float = 0.25
    permutation_weight: float = 0.25  
    empty_weight: float = 0.25
    random_weight: float = 0.25
    permutation_types: List[str] = ["rotate", "reflect", "color_remap"]
    random_density: float = 0.3  # For random mode
    enable_fallback: bool = True  # Fallback to demo if other modes fail
```

This configuration will be integrated into the existing `ArcEnvConfig` structure to maintain the single source of truth principle.

### Functional API Enhancement

The core `arc_reset` function will be enhanced to support diverse initialization:

```python
def arc_reset(
    task: JaxArcTask,
    config: ArcEnvConfig,
    key: PRNGKey,
    batch_size: Optional[int] = None
) -> ArcEnvState:
    # Enhanced to handle diverse initialization based on config.grid_initialization
```

## Components and Interfaces

### 1. Grid Initialization Engine

**Core Interface:**
```python
def initialize_working_grids(
    task: JaxArcTask,
    config: GridInitializationConfig,
    key: PRNGKey,
    batch_size: int
) -> Grid:
    """Initialize working grids based on configuration strategy."""
```

**Implementation Strategy:**
- Use JAX's `jax.random.choice` for mode selection based on weights
- Vectorize initialization across batch dimensions using `jax.vmap`
- Maintain static shapes for JIT compatibility

### 2. Initialization Mode Handlers

**Demo Mode Handler:**
```python
def _init_demo_grids(task: JaxArcTask, key: PRNGKey, batch_size: int) -> Grid:
    """Initialize grids from demo input examples (current behavior)."""
```

**Permutation Mode Handler:**
```python
def _init_permutation_grids(
    task: JaxArcTask, 
    config: GridInitializationConfig,
    key: PRNGKey, 
    batch_size: int
) -> Grid:
    """Initialize grids with permuted versions of demo inputs."""
```

**Empty Mode Handler:**
```python
def _init_empty_grids(task: JaxArcTask, batch_size: int) -> Grid:
    """Initialize completely empty grids (all zeros)."""
```

**Random Mode Handler:**
```python
def _init_random_grids(
    task: JaxArcTask,
    config: GridInitializationConfig, 
    key: PRNGKey,
    batch_size: int
) -> Grid:
    """Initialize grids with random patterns."""
```

### 3. Grid Transformation Utilities

**Permutation Operations:**
```python
def apply_grid_permutations(
    grid: Grid,
    permutation_types: List[str],
    key: PRNGKey
) -> Grid:
    """Apply various transformations to create grid variations."""
```

**Supported Transformations:**
- **Rotation**: 90°, 180°, 270° rotations
- **Reflection**: Horizontal and vertical flips
- **Color Remapping**: Systematic color palette changes while preserving structure
- **Translation**: Shifting patterns within grid bounds (with wrapping or padding)

### 4. Random Pattern Generation

**Pattern Generators:**
```python
def generate_random_patterns(
    shape: Tuple[int, int],
    density: float,
    key: PRNGKey,
    batch_size: int
) -> Grid:
    """Generate diverse random patterns with specified density."""
```

**Pattern Types:**
- **Sparse patterns**: Low density with isolated elements
- **Dense patterns**: Higher density with connected regions
- **Structured patterns**: Simple geometric shapes and lines
- **Noise patterns**: Completely random color assignments

## Data Models

### Enhanced Configuration Structure

```python
@equinox.Module
class ArcEnvConfig:
    # Existing fields...
    grid_initialization: GridInitializationConfig = GridInitializationConfig()
    
    # Backward compatibility
    def __post_init__(self):
        # Ensure demo mode if no initialization config provided
        if self.grid_initialization.mode == "demo":
            # Maintain current behavior
            pass
```

### Batch Mode Selection

```python
@equinox.Module  
class BatchInitializationState:
    """Tracks initialization modes used across batch for debugging/analysis."""
    modes: jnp.ndarray  # Shape: (batch_size,) with mode indices
    demo_indices: jnp.ndarray  # Which demo was used for demo/permutation modes
    permutation_types: jnp.ndarray  # Which permutation was applied
```

## Error Handling

### Validation Strategy

1. **Configuration Validation**: Ensure weights sum to 1.0 and all modes are valid
2. **Fallback Mechanisms**: If permutation/random generation fails, fall back to demo mode
3. **Shape Consistency**: Ensure all initialization modes produce grids with consistent shapes
4. **Color Validation**: Verify all generated grids use valid ARC colors (0-9)

### Error Recovery

```python
def _safe_initialize_grid(
    task: JaxArcTask,
    config: GridInitializationConfig,
    key: PRNGKey,
    mode: str
) -> Tuple[Grid, bool]:
    """Initialize grid with error recovery, returns (grid, success_flag)."""
```

## Testing Strategy

### Unit Tests

1. **Configuration Tests**: Validate configuration parsing and defaults
2. **Mode Handler Tests**: Test each initialization mode independently
3. **Permutation Tests**: Verify transformations preserve grid validity
4. **Random Generation Tests**: Ensure random patterns meet constraints
5. **Batch Processing Tests**: Verify correct distribution of modes across batches

### Integration Tests

1. **Functional API Tests**: Test enhanced `arc_reset` with various configurations
2. **JAX Compatibility Tests**: Verify JIT compilation and vectorization work correctly
3. **Performance Tests**: Ensure initialization doesn't significantly impact step rate
4. **Backward Compatibility Tests**: Verify existing code continues to work

### Property-Based Tests

1. **Grid Validity**: All generated grids have valid shapes and colors
2. **Determinism**: Same PRNG key produces identical results
3. **Distribution**: Mode selection follows specified probability weights
4. **Permutation Preservation**: Transformations maintain structural relationships

## Performance Considerations

### JAX Optimization

- **Vectorization**: Use `jax.vmap` for batch processing of initialization
- **Pre-compilation**: JIT compile initialization functions for performance
- **Memory Efficiency**: Avoid creating unnecessary intermediate arrays
- **Static Shapes**: Maintain consistent array shapes for optimal JIT performance

### Caching Strategy

- **Permutation Cache**: Pre-compute common transformations for reuse
- **Pattern Templates**: Cache random pattern generators for efficiency
- **Configuration Validation**: Cache validated configurations to avoid repeated checks

### Scalability

The design supports scaling to thousands of batch elements by:
- Using efficient JAX random number generation
- Vectorizing all operations across batch dimensions
- Minimizing memory allocation during initialization
- Leveraging JAX's automatic parallelization capabilities