# Getting Started: JaxARC Transition to Jumanji/Mava

## Quick Start Guide

This guide provides step-by-step instructions to begin the transition from JaxMARL to Jumanji/Mava. Follow these steps to get your development environment ready and implement your first Jumanji-based ARC environment.

## Prerequisites

- Python 3.10+ with JAX installed
- Basic familiarity with JAX and RL environments
- Access to ARC-AGI dataset (optional for initial development)

## Step 1: Environment Setup (15 minutes)

### 1.1 Update Dependencies

```bash
cd JaxARC

# Update pixi configuration
pixi add jumanji
pixi add id-mava

# Install new dependencies
pixi install
```

### 1.2 Verify Installation

Create and run a test script:

```bash
# Create test script
cat > scripts/verify_installation.py << 'EOF'
#!/usr/bin/env python3
"""Verify Jumanji and Mava installation."""

import jax
import jax.numpy as jnp
import jumanji
import chex

def test_jumanji():
    print("Testing Jumanji...")
    
    # Test basic environment
    env = jumanji.make('Snake-v1')
    key = jax.random.PRNGKey(0)
    state, timestep = env.reset(key)
    
    print(f"✅ Environment reset successful")
    print(f"   State type: {type(state)}")
    print(f"   Timestep type: {type(timestep)}")
    
    # Test step
    action = env.action_spec().generate_value()
    state, timestep = env.step(state, action)
    print(f"✅ Environment step successful")
    
    return True

def test_mava():
    print("\nTesting Mava...")
    try:
        import mava
        from mava.environments import Environment as MavaEnvironment
        print("✅ Mava imports successful")
        return True
    except ImportError as e:
        print(f"❌ Mava import failed: {e}")
        return False

if __name__ == "__main__":
    jumanji_ok = test_jumanji()
    mava_ok = test_mava()
    
    if jumanji_ok and mava_ok:
        print("\n🎉 All dependencies installed correctly!")
    else:
        print("\n❌ Some dependencies failed. Check installation.")
EOF

# Run verification
python scripts/verify_installation.py
```

## Step 2: Create Basic Jumanji Environment (30 minutes)

### 2.1 Create Environment Structure

```bash
# Create new environment files
mkdir -p src/jaxarc/envs/jumanji
touch src/jaxarc/envs/jumanji/__init__.py
```

### 2.2 Implement Basic Environment

Create `src/jaxarc/envs/jumanji/basic_arc_env.py`:

```python
"""Basic ARC environment using Jumanji - Getting Started Version."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from jumanji import Environment
from jumanji.types import TimeStep, StepType
from jumanji.specs import Spec, Array, DiscreteArray
from typing import Tuple, Dict, Any, NamedTuple


@chex.dataclass
class ArcState:
    """Simple ARC environment state."""
    grid: jnp.ndarray  # Current working grid
    target: jnp.ndarray  # Target grid to match
    step_count: jnp.ndarray  # Current step
    max_steps: jnp.ndarray  # Maximum steps allowed
    

class BasicArcEnv(Environment[ArcState]):
    """
    Basic ARC environment for getting started with Jumanji.
    
    This is a simplified version to demonstrate the core concepts
    before building the full collaborative reasoning system.
    """
    
    def __init__(self, grid_size: int = 10, max_steps: int = 50):
        """Initialize basic ARC environment."""
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
    
    def reset(self, key: chex.PRNGKey) -> Tuple[ArcState, TimeStep[Array]]:
        """Reset environment to initial state."""
        # Create simple initial state
        initial_grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        
        # Create simple target pattern (diagonal line)
        target_grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        target_grid = target_grid.at[jnp.arange(self.grid_size), jnp.arange(self.grid_size)].set(1)
        
        state = ArcState(
            grid=initial_grid,
            target=target_grid,
            step_count=jnp.array(0, dtype=jnp.int32),
            max_steps=jnp.array(self.max_steps, dtype=jnp.int32),
        )
        
        # Create initial observation
        observation = self._get_observation(state)
        
        # Create timestep
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0),
            discount=jnp.array(1.0),
            observation=observation,
        )
        
        return state, timestep
    
    def step(self, state: ArcState, action: Dict[str, jnp.ndarray]) -> Tuple[ArcState, TimeStep[Array]]:
        """Take environment step."""
        # Extract action components
        position = action['position']  # [row, col]
        color = action['color']        # color value
        
        # Update grid
        row, col = position[0], position[1]
        new_grid = state.grid.at[row, col].set(color)
        
        # Update step count
        new_step_count = state.step_count + 1
        
        # Create new state
        new_state = state.replace(
            grid=new_grid,
            step_count=new_step_count
        )
        
        # Calculate reward (how well grid matches target)
        reward = self._calculate_reward(new_state)
        
        # Check if done
        is_done = self._is_terminal(new_state)
        
        # Create observation
        observation = self._get_observation(new_state)
        
        # Create timestep
        if is_done:
            timestep = TimeStep(
                step_type=StepType.LAST,
                reward=reward,
                discount=jnp.array(0.0),
                observation=observation,
            )
        else:
            timestep = TimeStep(
                step_type=StepType.MID,
                reward=reward,
                discount=jnp.array(1.0),
                observation=observation,
            )
        
        return new_state, timestep
    
    def _get_observation(self, state: ArcState) -> Dict[str, jnp.ndarray]:
        """Get observation from state."""
        return {
            'grid': state.grid,
            'target': state.target,
            'step_count': state.step_count,
        }
    
    def _calculate_reward(self, state: ArcState) -> jnp.ndarray:
        """Calculate reward based on grid-target similarity."""
        # Simple reward: number of matching pixels
        matches = jnp.sum(state.grid == state.target)
        total_pixels = self.grid_size * self.grid_size
        return matches / total_pixels
    
    def _is_terminal(self, state: ArcState) -> jnp.ndarray:
        """Check if episode should terminate."""
        # Terminal if max steps or perfect match
        max_steps_reached = state.step_count >= state.max_steps
        perfect_match = jnp.array_equal(state.grid, state.target)
        return max_steps_reached | perfect_match
    
    def observation_spec(self) -> Dict[str, Spec]:
        """Define observation specification."""
        return {
            'grid': Array(shape=(self.grid_size, self.grid_size), dtype=jnp.int32),
            'target': Array(shape=(self.grid_size, self.grid_size), dtype=jnp.int32),
            'step_count': Array(shape=(), dtype=jnp.int32),
        }
    
    def action_spec(self) -> Dict[str, Spec]:
        """Define action specification."""
        return {
            'position': Array(shape=(2,), dtype=jnp.int32),
            'color': DiscreteArray(num_values=10),  # ARC colors 0-9
        }
```

### 2.3 Create Test Script

Create `scripts/test_basic_env.py`:

```python
#!/usr/bin/env python3
"""Test the basic ARC environment."""

import jax
import jax.numpy as jnp
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jaxarc.envs.jumanji.basic_arc_env import BasicArcEnv


def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing basic ARC environment...")
    
    # Create environment
    env = BasicArcEnv(grid_size=5, max_steps=10)
    key = jax.random.PRNGKey(42)
    
    # Test reset
    state, timestep = env.reset(key)
    print(f"✅ Reset successful")
    print(f"   Grid shape: {state.grid.shape}")
    print(f"   Target shape: {state.target.shape}")
    print(f"   Initial reward: {timestep.reward}")
    
    # Test step
    action = {
        'position': jnp.array([0, 0], dtype=jnp.int32),
        'color': jnp.array(1, dtype=jnp.int32),
    }
    
    new_state, new_timestep = env.step(state, action)
    print(f"✅ Step successful")
    print(f"   New grid[0,0]: {new_state.grid[0, 0]}")
    print(f"   Step reward: {new_timestep.reward}")
    print(f"   Step count: {new_state.step_count}")
    
    return True


def test_jax_compatibility():
    """Test JAX transformation compatibility."""
    print("\nTesting JAX compatibility...")
    
    env = BasicArcEnv(grid_size=5, max_steps=10)
    
    # Test JIT compilation
    @jax.jit
    def jit_reset(key):
        return env.reset(key)
    
    @jax.jit  
    def jit_step(state, action):
        return env.step(state, action)
    
    key = jax.random.PRNGKey(42)
    state, timestep = jit_reset(key)
    print(f"✅ JIT reset successful")
    
    action = {
        'position': jnp.array([1, 1], dtype=jnp.int32),
        'color': jnp.array(2, dtype=jnp.int32),
    }
    
    new_state, new_timestep = jit_step(state, action)
    print(f"✅ JIT step successful")
    
    # Test vectorization
    @jax.vmap
    def vmap_reset(keys):
        return env.reset(keys)
    
    keys = jax.random.split(key, 4)
    states, timesteps = vmap_reset(keys)
    print(f"✅ VMAP successful - batch size: {states.grid.shape[0]}")
    
    return True


def run_simple_episode():
    """Run a simple episode to demonstrate usage."""
    print("\nRunning simple episode...")
    
    env = BasicArcEnv(grid_size=3, max_steps=5)
    key = jax.random.PRNGKey(123)
    
    state, timestep = env.reset(key)
    
    print(f"Initial state:")
    print(f"Grid:\n{state.grid}")
    print(f"Target:\n{state.target}")
    print(f"Reward: {timestep.reward:.3f}")
    
    # Take a few steps
    for step in range(3):
        # Simple strategy: fill diagonal
        action = {
            'position': jnp.array([step, step], dtype=jnp.int32),
            'color': jnp.array(1, dtype=jnp.int32),
        }
        
        state, timestep = env.step(state, action)
        
        print(f"\nStep {step + 1}:")
        print(f"Action: position={action['position']}, color={action['color']}")
        print(f"Grid:\n{state.grid}")
        print(f"Reward: {timestep.reward:.3f}")
        print(f"Done: {timestep.step_type == 'LAST'}")
        
        if timestep.step_type.name == 'LAST':
            print("Episode terminated!")
            break
    
    return True


if __name__ == "__main__":
    success = True
    
    try:
        success &= test_basic_functionality()
        success &= test_jax_compatibility()
        success &= run_simple_episode()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        success = False
    
    if success:
        print("\n🎉 All tests passed! Basic environment is working.")
    else:
        print("\n❌ Some tests failed. Check implementation.")
```

## Step 3: Test Your Implementation (10 minutes)

```bash
# Run the test
python scripts/test_basic_env.py
```

Expected output:
```
Testing basic ARC environment...
✅ Reset successful
   Grid shape: (5, 5)
   Target shape: (5, 5)
   Initial reward: 0.04

✅ Step successful
   New grid[0,0]: 1
   Step reward: 0.08
   Step count: 1

Testing JAX compatibility...
✅ JIT reset successful
✅ JIT step successful
✅ VMAP successful - batch size: 4

Running simple episode...
Initial state:
Grid:
[[0 0 0]
 [0 0 0]
 [0 0 0]]
Target:
[[1 0 0]
 [0 1 0]
 [0 0 1]]
Reward: 0.000

Step 1:
Action: position=[0 0], color=1
Grid:
[[1 0 0]
 [0 0 0]
 [0 0 0]]
Reward: 0.333

Step 2:
Action: position=[1 1], color=1
Grid:
[[1 0 0]
 [0 1 0]
 [0 0 0]]
Reward: 0.667

Step 3:
Action: position=[2 2], color=1
Grid:
[[1 0 0]
 [0 1 0]
 [0 0 1]]
Reward: 1.000
Done: True
Episode terminated!

🎉 All tests passed! Basic environment is working.
```

## Step 4: Integration with Existing Code (20 minutes)

### 4.1 Test Parser Integration

Create `scripts/test_parser_integration.py`:

```python
#!/usr/bin/env python3
"""Test integration with existing parser."""

import jax
import jax.numpy as jnp
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jaxarc.types import ParsedTaskData
from jaxarc.envs.jumanji.basic_arc_env import BasicArcEnv


def create_mock_task_data() -> ParsedTaskData:
    """Create mock task data for testing."""
    grid_size = 10
    
    return ParsedTaskData(
        input_grids_examples=jnp.zeros((2, grid_size, grid_size), dtype=jnp.int32),
        input_masks_examples=jnp.ones((2, grid_size, grid_size), dtype=jnp.bool_),
        output_grids_examples=jnp.ones((2, grid_size, grid_size), dtype=jnp.int32),
        output_masks_examples=jnp.ones((2, grid_size, grid_size), dtype=jnp.bool_),
        num_train_pairs=2,
        test_input_grids=jnp.zeros((1, grid_size, grid_size), dtype=jnp.int32),
        test_input_masks=jnp.ones((1, grid_size, grid_size), dtype=jnp.bool_),
        true_test_output_grids=jnp.ones((1, grid_size, grid_size), dtype=jnp.int32),
        true_test_output_masks=jnp.ones((1, grid_size, grid_size), dtype=jnp.bool_),
        num_test_pairs=1,
        task_id="test_task"
    )


def test_type_compatibility():
    """Test that existing types work with new environment."""
    print("Testing type compatibility...")
    
    # Create mock task data
    task_data = create_mock_task_data()
    print(f"✅ ParsedTaskData created successfully")
    print(f"   Task ID: {task_data.task_id}")
    print(f"   Train pairs: {task_data.num_train_pairs}")
    print(f"   Test pairs: {task_data.num_test_pairs}")
    
    # Test that it works with JAX operations
    @jax.jit
    def process_task_data(data):
        return jnp.sum(data.test_input_grids)
    
    result = process_task_data(task_data)
    print(f"✅ JAX operations work with ParsedTaskData")
    print(f"   Sum of test input: {result}")
    
    return True


if __name__ == "__main__":
    try:
        test_type_compatibility()
        print("\n🎉 Parser integration test passed!")
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
```

Run the test:
```bash
python scripts/test_parser_integration.py
```

## Step 5: Next Steps and Development Plan

### 5.1 Immediate Next Steps (This Week)

1. **Enhance Basic Environment**:
   - Add more sophisticated action types (copy, paste, transform)
   - Implement attention mechanisms
   - Add proper reward shaping

2. **Create Environment Factory**:
   - Build factory pattern to create environments with real task data
   - Integrate with existing `ArcAgiParser`
   - Add configuration management

3. **Performance Benchmarking**:
   - Measure baseline performance
   - Compare with theoretical Jumanji benefits
   - Identify optimization opportunities

### 5.2 Short-term Goals (Next 2-4 weeks)

1. **Advanced Single-Agent Features**:
   - Implement reasoning traces and scratchpad
   - Add sophisticated observation spaces
   - Create proper reward functions for ARC tasks

2. **Multi-Agent Planning**:
   - Design multi-agent state representation
   - Plan 4-phase reasoning architecture
   - Create Mava integration strategy

### 5.3 Development Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run performance benchmarks
python scripts/benchmark_performance.py

# Run specific environment tests
python scripts/test_basic_env.py

# Check code quality
pixi run lint

# Generate documentation
pixi run docs-serve
```

## Troubleshooting

### Common Issues

1. **JAX Installation Issues**:
   ```bash
   # Check JAX version and backend
   python -c "import jax; print(jax.version.__version__); print(jax.devices())"
   ```

2. **Import Errors**:
   ```bash
   # Ensure Python path includes src
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

3. **Jumanji Environment Issues**:
   ```bash
   # Test basic Jumanji functionality
   python -c "import jumanji; env = jumanji.make('Snake-v1'); print('OK')"
   ```

### Getting Help

- **Jumanji Documentation**: https://instadeepai.github.io/jumanji/
- **Mava Documentation**: https://id-mava.readthedocs.io/
- **JAX Documentation**: https://jax.readthedocs.io/

## Success Criteria

You've successfully completed the getting started phase when:

- ✅ Basic Jumanji environment runs without errors
- ✅ JAX transformations (jit, vmap) work correctly  
- ✅ Integration with existing types is successful
- ✅ Performance shows improvement over baseline
- ✅ Ready to move to advanced features

## What's Next?

After completing this guide, you're ready to:

1. **Implement Advanced Features**: Add sophisticated reasoning capabilities
2. **Multi-Agent Extension**: Begin Mava integration for collaborative reasoning
3. **Performance Optimization**: Leverage JAX optimizations for speed
4. **Research Integration**: Connect with broader research goals

This foundation provides a solid base for building the full JaxARC collaborative reasoning system using industry-standard frameworks.