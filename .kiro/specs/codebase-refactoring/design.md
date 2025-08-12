# Design Document

## Overview

This design document outlines a comprehensive refactoring strategy for the
JaxARC codebase to eliminate code duplication, reduce complexity, and modernize
the architecture using Equinox and JAXTyping. The refactoring will maintain
backward compatibility while significantly improving code maintainability, type
safety, and JAX integration.

## Architecture

### Core Principles

1. **Single Source of Truth**: Eliminate all code duplication by centralizing
   shared functionality
2. **Modern JAX Patterns**: Leverage Equinox and JAXTyping for better JAX
   integration and type safety
3. **Hydra-First Configuration**: Simplify configuration management by fully
   leveraging Hydra's capabilities
4. **Clean Separation of Concerns**: Ensure each module has a single, clear
   responsibility
5. **Functional Core**: Maintain pure functional patterns for JAX compatibility

### High-Level Architecture Changes

```
src/jaxarc/
├── types.py                    # Centralized type definitions with JAXTyping
├── state.py                    # Equinox-based state management (NEW)
├── envs/
│   ├── environment.py          # Simplified environment class
│   ├── functional.py           # Streamlined functional API
│   ├── actions.py              # Clean action handler system
│   ├── config.py               # Hydra-integrated configuration
│   └── grid_operations.py      # Grid operations (unchanged)
├── parsers/
│   ├── base_parser.py          # DRY base implementation
│   ├── arc_agi.py              # Minimal specific implementation
│   └── concept_arc.py          # Minimal specific implementation
└── utils/
    ├── equinox_utils.py        # Equinox integration utilities (NEW)
    └── jax_types.py            # JAXTyping type definitions (NEW)
```

## Components and Interfaces

### 1. Centralized Type System with JAXTyping

**File: `src/jaxarc/utils/jax_types.py`**

```python
from jaxtyping import Array, Float, Int, Bool
from typing import TypeAlias

# Grid type aliases with precise shape annotations
GridArray: TypeAlias = Int[Array, "height width"]
MaskArray: TypeAlias = Bool[Array, "height width"]
SelectionArray: TypeAlias = Float[Array, "height width"]
SimilarityScore: TypeAlias = Float[Array, ""]

# Batch types for vectorized operations
BatchGridArray: TypeAlias = Int[Array, "batch height width"]
BatchMaskArray: TypeAlias = Bool[Array, "batch height width"]

# Action types
PointCoords: TypeAlias = Int[Array, "2"]  # [row, col]
BboxCoords: TypeAlias = Int[Array, "4"]   # [r1, c1, r2, c2]
OperationId: TypeAlias = Int[Array, ""]
```

**File: `src/jaxarc/types.py` (Updated)**

```python
# Remove ArcEnvState (moved to state.py)
# Keep Grid, JaxArcTask, ARCLEAction with JAXTyping annotations
from jaxarc.utils.jax_types import GridArray, MaskArray

@chex.dataclass
class Grid:
    data: GridArray
    mask: MaskArray
```

### 2. Equinox-Based State Management

**File: `src/jaxarc/state.py` (NEW)**

```python
import equinox as eqx
from jaxarc.utils.jax_types import GridArray, MaskArray, SimilarityScore

class ArcEnvState(eqx.Module):
    """Equinox-based ARC environment state with automatic PyTree registration."""

    # Core ARC state
    task_data: JaxArcTask
    working_grid: GridArray
    working_grid_mask: MaskArray
    target_grid: GridArray

    # Episode management
    step_count: int
    episode_done: bool
    current_example_idx: int

    # Grid operations
    selected: MaskArray
    clipboard: GridArray
    similarity_score: SimilarityScore

    def __check_init__(self):
        """Equinox validation method."""
        # JAXTyping will handle shape validation automatically
        pass
```

### 3. Streamlined Action Handling

**File: `src/jaxarc/envs/functional.py` (Simplified)**

```python
def arc_step(
    state: ArcEnvState,
    action: ActionType,
    config: ConfigType,
) -> Tuple[ArcEnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """Simplified arc_step that delegates to action handlers."""
    typed_config = _ensure_config(config)

    # Get the correct handler based on config
    handler = get_action_handler(typed_config.action.selection_format)

    # Extract action data based on format
    if typed_config.action.selection_format == "point":
        action_data = jnp.array(action["point"])
    elif typed_config.action.selection_format == "bbox":
        action_data = jnp.array(action["bbox"])
    else:  # mask
        action_data = action["mask"].flatten()

    # Handler creates standardized selection mask
    selection_mask = handler(action_data, state.working_grid_mask)

    # Create standardized action
    standard_action = {
        "selection": selection_mask,
        "operation": action["operation"]
    }

    # Update state and execute operation
    state = state.replace(selected=standard_action["selection"])
    new_state = execute_grid_operation(state, standard_action["operation"])

    # Calculate reward, done, observation (unchanged logic)
    # ...
```

### 4. DRY Parser Architecture

**File: `src/jaxarc/parsers/base_parser.py` (Enhanced)**

```python
class ArcDataParserBase:
    """Base parser with all common functionality."""

    def _process_training_pairs(self, task_data: dict) -> tuple:
        """Common training pair processing logic."""
        # Move all shared logic here from specific parsers
        pass

    def _pad_and_create_masks(self, grids: list) -> tuple:
        """Common padding and mask creation logic."""
        # Move all shared logic here from specific parsers
        pass

    def _validate_grid_colors(self, grid: jnp.ndarray) -> bool:
        """Common grid color validation logic."""
        # Move all shared logic here from specific parsers
        pass
```

**File: `src/jaxarc/parsers/arc_agi.py` (Simplified)**

```python
class ArcAgiParser(ArcDataParserBase):
    """Minimal ARC-AGI specific implementation."""

    def _load_and_cache_tasks(self) -> None:
        """ARC-AGI specific loading logic only."""
        # Only dataset-specific logic here
        # Call super()._process_training_pairs() for common functionality
        pass
```

### 5. Hydra-First Configuration

**Eliminate Factory Functions**: Remove most factory functions from `factory.py`
and rely on Hydra composition.

**Enhanced Hydra Structure**:

```yaml
# conf/presets/minimal.yaml (replaces create_raw_config)
defaults:
  - base_config
  - action: minimal
  - reward: basic

# conf/presets/standard.yaml (replaces create_standard_config)
defaults:
  - base_config
  - action: standard
  - reward: standard
```

**Usage**:

```python
# Instead of create_raw_config()
@hydra.main(config_path="conf", config_name="presets/minimal")
def main(cfg: DictConfig):
    config = ArcEnvConfig.from_hydra(cfg)
```

### 6. Equinox Integration Utilities

**File: `src/jaxarc/utils/equinox_utils.py` (NEW)**

```python
import equinox as eqx
from typing import TypeVar, Callable

T = TypeVar('T', bound=eqx.Module)

def tree_map_with_path(fn: Callable, tree: T) -> T:
    """Enhanced tree mapping with path information."""
    pass

def validate_state_shapes(state: ArcEnvState) -> bool:
    """Validate state using Equinox patterns."""
    pass

def create_state_diff(old_state: ArcEnvState, new_state: ArcEnvState) -> dict:
    """Create diff between states for debugging."""
    pass
```

## Data Models

### State Management with Equinox

```python
# Before (chex dataclass with duplication)
@chex.dataclass
class ArcEnvState:  # Defined in multiple files
    working_grid: jnp.ndarray
    # ... validation in __post_init__

# After (Equinox module, single definition)
class ArcEnvState(eqx.Module):
    working_grid: GridArray  # JAXTyping provides shape validation
    # ... automatic PyTree registration and validation
```

### Type Safety with JAXTyping

```python
# Before (generic arrays)
def compute_similarity(grid1: jnp.ndarray, grid2: jnp.ndarray) -> jnp.ndarray:

# After (precise type annotations)
def compute_similarity(
    grid1: GridArray,
    grid2: GridArray
) -> SimilarityScore:
```

### Configuration Simplification

```python
# Before (verbose factory functions)
def create_standard_config(max_steps=100, penalty=-0.01, ...):
    reward_config = RewardConfig(...)
    grid_config = GridConfig(...)
    # ... 50+ lines of boilerplate

# After (Hydra composition)
# conf/presets/standard.yaml handles composition
config = ArcEnvConfig.from_hydra(hydra_cfg)
```

## Error Handling

### JAXTyping Runtime Validation

```python
from jaxtyping import jaxtyped
from beartype import beartype

@jaxtyped
@beartype
def process_grid(grid: GridArray) -> MaskArray:
    """Automatic runtime shape and type validation."""
    pass
```

### Equinox Error Messages

```python
# Equinox provides better error messages for PyTree operations
try:
    new_state = state.replace(working_grid=invalid_grid)
except ValueError as e:
    # Clear error message about shape mismatches
    logger.error(f"State update failed: {e}")
```

### Configuration Validation

```python
# Enhanced Hydra validation with structured configs
@dataclass
class ActionConfigSchema:
    selection_format: Literal["point", "bbox", "mask"]
    num_operations: int = field(validator=lambda x: x > 0)

# Automatic validation at config creation time
```

## Testing Strategy

### 1. Migration Testing

- **Backward Compatibility Tests**: Ensure all existing functionality works
- **Performance Benchmarks**: Verify no performance regression with Equinox
- **Type Safety Tests**: Validate JAXTyping annotations catch errors

### 2. Integration Testing

- **Hydra Configuration Tests**: Verify all config combinations work
- **Action Handler Tests**: Ensure simplified action handling maintains
  functionality
- **Parser Tests**: Validate DRY parser refactoring doesn't break data loading

### 3. JAX Transformation Tests

- **JIT Compilation**: Ensure all functions remain JIT-compatible
- **Vectorization**: Test vmap compatibility with new state structure
- **Gradient Computation**: Verify autodiff works with Equinox modules

### 4. Property-Based Testing

```python
from hypothesis import given, strategies as st
from jaxtyping import Array

@given(grid=st.arrays(dtype=jnp.int32, shape=(30, 30)))
def test_grid_operations_preserve_shape(grid: GridArray):
    """Property test for shape preservation."""
    result = some_grid_operation(grid)
    assert result.shape == grid.shape
```

## Implementation Phases

### Phase 1: Foundation (Requirements 1-2)

- Move ArcEnvState to centralized location
- Add JAXTyping annotations to core types
- Simplify arc_step function to use action handlers

### Phase 2: Parser Consolidation (Requirement 3)

- Move common methods to base parser
- Update specific parsers to use inheritance
- Add comprehensive parser tests

### Phase 3: Equinox Integration (Requirement 5)

- Convert ArcEnvState to Equinox module
- Add Equinox utilities
- Migrate state management patterns

### Phase 4: Configuration Simplification (Requirement 4)

- Create Hydra preset configurations
- Remove redundant factory functions
- Enhance structured config validation

### Phase 5: Type Safety Enhancement (Requirement 6)

- Add comprehensive JAXTyping annotations
- Implement runtime type checking
- Add property-based tests

### Phase 6: Code Organization (Requirement 7-8)

- Reorganize modules for clarity
- Add comprehensive documentation
- Final cleanup and optimization

## Migration Strategy

### Backward Compatibility

1. **Deprecation Warnings**: Add warnings for old patterns
2. **Adapter Functions**: Provide compatibility shims
3. **Gradual Migration**: Allow mixed old/new patterns during transition

### Testing During Migration

1. **Dual Implementation**: Run old and new implementations in parallel
2. **Property Testing**: Ensure behavioral equivalence
3. **Performance Monitoring**: Track performance throughout migration

### Documentation Updates

1. **Migration Guide**: Step-by-step migration instructions
2. **API Documentation**: Updated with new patterns
3. **Examples**: Modernized example code

## Benefits

### Code Quality

- **50% reduction** in code duplication
- **Improved type safety** with JAXTyping runtime validation
- **Better error messages** from Equinox and structured configs

### Developer Experience

- **Simplified configuration** through Hydra presets
- **Cleaner action handling** with single responsibility handlers
- **Modern JAX patterns** following ecosystem best practices

### Performance

- **Better JIT compilation** with Equinox PyTrees
- **Optimized transformations** with proper type annotations
- **Reduced memory overhead** from eliminating duplicate code

### Maintainability

- **Single source of truth** for all shared functionality
- **Clear module boundaries** with well-defined responsibilities
- **Comprehensive test coverage** with property-based testing
