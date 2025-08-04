# Design Document

## Overview

This design addresses the systematic refactoring of the JaxARC codebase based on a thorough analysis of current pain points. The main issues identified are:

1. **Configuration Pattern Inconsistency**: Config classes have `from_hydra` methods but parsers take raw `DictConfig` objects
2. **No Duplicate Functions Found**: Contrary to initial assumptions, there are no `arc_step_enhanced` or similar duplicates
3. **Parser Initialization Inconsistency**: `MiniArcParser(hydra_config.dataset)` takes raw Hydra config instead of typed config
4. **Visualization System Complexity**: Multiple overlapping visualization components with unclear boundaries
5. **Missing Task Visualization**: No way to visualize the actual task being solved at episode start
6. **Long Functions**: Some functions in `functional.py` exceed 50 lines and need decomposition

## Architecture

### Current Pain Points Analysis

**Configuration System Issues:**
```python
# Current inconsistent pattern:
config = JaxArcConfig.from_hydra(hydra_config)  # ✓ Good
parser = MiniArcParser(hydra_config.dataset)   # ✗ Takes raw DictConfig

# Should be:
config = JaxArcConfig.from_hydra(hydra_config)  # ✓ Good  
parser = MiniArcParser(config.dataset)         # ✓ Takes typed config
```

**Visualization System Issues:**
- `EnhancedVisualizer` exists but no task visualization at episode start
- `StepVisualizationData` lacks task context (task_id, pair_index)
- Multiple visualization configs (`VisualizationConfig` in different modules)
- Complex initialization with optional dependencies

**Function Length Issues:**
- `arc_reset` in `functional.py` is 200+ lines
- `arc_step` in `functional.py` is 300+ lines  
- Multiple helper functions that could be extracted

## Components and Interfaces

### 1. Configuration System Standardization

**Problem**: Parsers take raw `DictConfig` instead of typed configs

**Solution**: Update all component constructors to use typed configs

```python
# Current problematic pattern
class MiniArcParser:
    def __init__(self, cfg: DictConfig) -> None:  # ✗ Raw DictConfig
        self.cfg = cfg

# Fixed pattern  
class MiniArcParser:
    def __init__(self, config: DatasetConfig) -> None:  # ✓ Typed config
        self.config = config
        
    @classmethod
    def from_hydra(cls, hydra_config: DictConfig) -> MiniArcParser:
        """Alternative constructor for Hydra compatibility."""
        dataset_config = DatasetConfig.from_hydra(hydra_config)
        return cls(dataset_config)
```

**Components to Update:**
- `MiniArcParser.__init__(cfg: DictConfig)` → `MiniArcParser.__init__(config: DatasetConfig)`
- `ArcAgiParser.__init__(cfg: DictConfig)` → `ArcAgiParser.__init__(config: DatasetConfig)`
- `ConceptArcParser.__init__(cfg: DictConfig)` → `ConceptArcParser.__init__(config: DatasetConfig)`

### 2. Function Decomposition Strategy

**Problem**: Long functions in `functional.py` are hard to maintain

**Solution**: Extract focused helper functions while maintaining JAX compliance

```python
# Current: arc_reset is 200+ lines
def arc_reset(key, config, task_data=None, episode_mode=0, initial_pair_idx=None):
    # 200+ lines of mixed logic
    pass

# Refactored: Extract focused functions
def arc_reset(key, config, task_data=None, episode_mode=0, initial_pair_idx=None):
    typed_config = _ensure_config(config)
    task_data = _get_or_create_task_data(task_data, typed_config)
    selected_pair_idx = _select_initial_pair(key, task_data, episode_mode, initial_pair_idx)
    initial_grid, target_grid, initial_mask = _initialize_grids(task_data, selected_pair_idx, episode_mode, typed_config)
    state = _create_initial_state(task_data, initial_grid, target_grid, initial_mask, selected_pair_idx, episode_mode)
    observation = create_observation(state, typed_config)
    return state, observation

def _get_or_create_task_data(task_data, config):
    """Get task data or create demo task - focused helper function."""
    # 20-30 lines focused on task data logic
    
def _select_initial_pair(key, task_data, episode_mode, initial_pair_idx):
    """Select initial pair based on mode and configuration - focused helper function."""
    # 20-30 lines focused on pair selection logic
    
def _initialize_grids(task_data, selected_pair_idx, episode_mode, config):
    """Initialize grids based on episode mode - focused helper function."""
    # 20-30 lines focused on grid initialization logic
```

### 3. Visualization System Consolidation

**Problem**: No task visualization at episode start, missing task context in steps, "Enhanced" naming suggests there's a basic version

**Solution**: Consolidate to single `Visualizer` class, add task visualization capabilities

```python
# StepVisualizationData with task context
@chex.dataclass
class StepVisualizationData:
    # Existing fields
    step_num: int
    before_grid: Grid
    after_grid: Grid
    action: Dict[str, Any]
    reward: float
    info: Dict[str, Any]
    
    # New fields for task context
    task_id: str = ""
    task_pair_index: int = 0
    total_pairs: int = 1
    
    # Optional existing fields
    selection_mask: Optional[jnp.ndarray] = None
    changed_cells: Optional[jnp.ndarray] = None
    operation_name: str = ""
    timestamp: float = field(default_factory=time.time)

# New TaskVisualizationData for episode start
@chex.dataclass  
class TaskVisualizationData:
    task_id: str
    task_pairs: List[Any]  # List of (input, output) pairs
    current_pair_index: int
    episode_mode: str  # "train" or "test"
    metadata: Dict[str, Any] = field(default_factory=dict)

# Visualizer methods (renamed from EnhancedVisualizer)
class Visualizer:
    def start_episode_with_task(self, episode_num: int, task: JaxArcTask, task_id: str = "") -> None:
        """Start episode and create task visualization first."""
        self.start_episode(episode_num, task_id)
        
        # Create task visualization
        task_viz_data = TaskVisualizationData(
            task_id=task_id,
            task_pairs=self._extract_task_pairs(task),
            current_pair_index=0,
            episode_mode="train",  # or "test"
        )
        self._create_task_visualization(task_viz_data)
    
    def _create_task_visualization(self, task_data: TaskVisualizationData) -> Optional[Path]:
        """Create SVG visualization of the complete task."""
        # Use existing draw_parsed_task_data_svg functionality
        pass
```

### 4. Code Organization Improvements

**Problem**: Visualization system has overlapping responsibilities

**Solution**: Consolidate visualization components and clarify boundaries

```python
# Current structure (confusing):
src/jaxarc/utils/visualization/
├── core.py                    # Basic visualization functions
├── enhanced_visualizer.py     # Main visualizer class
├── episode_manager.py         # Episode management
├── wandb_integration.py       # Wandb logging
└── config_*.py               # Multiple config modules

# Improved structure (clearer):
src/jaxarc/utils/visualization/
├── core.py                    # Basic grid/task visualization functions
├── visualizer.py              # Main Visualizer (renamed from enhanced_visualizer.py, replaces any basic version)
├── episode_manager.py         # Episode management (unchanged)
├── integrations/              # External integrations
│   ├── wandb.py              # Wandb integration (renamed)
│   └── __init__.py
└── config.py                 # Single config module (consolidated)
```

## Data Models

### Enhanced Configuration Pattern

**All parsers will follow this pattern:**

```python
class BaseParser:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        
    @classmethod  
    def from_hydra(cls, hydra_config: DictConfig) -> Self:
        """Alternative constructor for Hydra compatibility."""
        dataset_config = DatasetConfig.from_hydra(hydra_config)
        return cls(dataset_config)

# Usage patterns:
# Direct typed config usage (preferred):
dataset_config = DatasetConfig.from_hydra(hydra_config.dataset)
parser = MiniArcParser(dataset_config)

# Hydra compatibility (for migration):
parser = MiniArcParser.from_hydra(hydra_config.dataset)
```

### Enhanced Visualization Data Models

```python
@chex.dataclass
class TaskVisualizationData:
    """Data for visualizing the complete task at episode start."""
    task_id: str
    demonstration_pairs: List[Tuple[Grid, Grid]]  # (input, output) pairs
    test_pairs: List[Grid]  # test inputs (outputs hidden)
    current_pair_index: int
    episode_mode: Literal["train", "test"]
    metadata: Dict[str, Any] = field(default_factory=dict)

@chex.dataclass  
class StepVisualizationData:
    """Step data with task context (replaces any basic version)."""
    # Core step data (unchanged)
    step_num: int
    before_grid: Grid
    after_grid: Grid
    action: Dict[str, Any]
    reward: float
    info: Dict[str, Any]
    
    # Task context (new)
    task_id: str
    task_pair_index: int
    total_task_pairs: int
    
    # Optional fields (unchanged)
    selection_mask: Optional[jnp.ndarray] = None
    changed_cells: Optional[jnp.ndarray] = None
    operation_name: str = ""
    timestamp: float = field(default_factory=time.time)
```

## Implementation Strategy

### Phase 1: Parser Configuration Standardization (High Priority)

**Specific Changes:**
1. Update `MiniArcParser.__init__(cfg: DictConfig)` → `MiniArcParser.__init__(config: DatasetConfig)`
2. Update `ArcAgiParser.__init__(cfg: DictConfig)` → `ArcAgiParser.__init__(config: DatasetConfig)`  
3. Update `ConceptArcParser.__init__(cfg: DictConfig)` → `ConceptArcParser.__init__(config: DatasetConfig)`
4. Add `@classmethod from_hydra()` methods to all parsers for backward compatibility
5. Update usage in `notebooks/miniarc_rl_loop.py`: `parser = MiniArcParser(config.dataset)`

**Files to Modify:**
- `src/jaxarc/parsers/mini_arc.py`
- `src/jaxarc/parsers/arc_agi.py`  
- `src/jaxarc/parsers/concept_arc.py`
- `notebooks/miniarc_rl_loop.py`
- Any other files using parsers

### Phase 2: Function Decomposition (Medium Priority)

**Specific Changes:**
1. Break down `arc_reset()` in `src/jaxarc/envs/functional.py` (200+ lines)
2. Break down `arc_step()` in `src/jaxarc/envs/functional.py` (300+ lines)
3. Extract helper functions while maintaining JAX compliance
4. Ensure all extracted functions are pure and JIT-compatible

**Target Functions:**
- `_get_or_create_task_data()` - 20-30 lines
- `_select_initial_pair()` - 20-30 lines  
- `_initialize_grids()` - 20-30 lines
- `_create_initial_state()` - 20-30 lines
- `_process_action()` - 30-40 lines
- `_update_state()` - 30-40 lines
- `_calculate_reward_and_done()` - 20-30 lines

### Phase 3: Enhanced Visualization (Medium Priority)

**Specific Changes:**
1. Add `task_id` and `task_pair_index` fields to `StepVisualizationData`
2. Create `TaskVisualizationData` dataclass
3. Add `start_episode_with_task()` method to `Visualizer`
4. Add `_create_task_visualization()` method using existing SVG functions
5. Update `notebooks/miniarc_rl_loop.py` to use task visualization

**Files to Modify:**
- `src/jaxarc/utils/visualization/enhanced_visualizer.py` (rename to `visualizer.py`)
- `notebooks/miniarc_rl_loop.py`
- Update all imports from `EnhancedVisualizer` to `Visualizer`

### Phase 4: Code Organization (Low Priority)

**Specific Changes:**
1. Rename `enhanced_visualizer.py` → `visualizer.py` and `EnhancedVisualizer` → `Visualizer`
2. Move wandb integration to `integrations/wandb.py`
3. Consolidate visualization configs into single `config.py`
4. Update imports across codebase

## Testing Strategy

### Configuration Testing
- Test parser initialization with typed configs
- Test `from_hydra()` methods work correctly
- Test backward compatibility during transition
- Test configuration validation

### Function Decomposition Testing  
- Test that decomposed functions maintain same behavior
- Test JAX compliance of all extracted functions
- Test performance characteristics are preserved
- Test error handling in decomposed functions

### Visualization Testing
- Test task visualization generation
- Test step visualization with task context
- Test episode flow with task visualization
- Test visualization output quality

## Migration Path

### Backward Compatibility Strategy
```python
# During transition, support both patterns:

# Old pattern (deprecated but working)
parser = MiniArcParser(hydra_config.dataset)

# New pattern (preferred)  
parser = MiniArcParser(DatasetConfig.from_hydra(hydra_config.dataset))

# Alternative new pattern (convenience)
parser = MiniArcParser.from_hydra(hydra_config.dataset)
```

### Gradual Migration Steps
1. Add typed config constructors alongside existing ones
2. Add deprecation warnings to old constructors  
3. Update examples and documentation to use new patterns
4. Remove old constructors after migration period
5. Update all internal usage to new patterns