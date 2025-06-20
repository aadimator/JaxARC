# JaxARC Base Environment Implementation

This directory contains the foundational components for ARC Multi-Agent
Reinforcement Learning environments that integrate with the JaxMARL framework.

## Overview

The base environment implementation provides:

- **`ArcEnvState`**: A comprehensive JAX-compatible state dataclass that
  captures all aspects of ARC task solving in a multi-agent collaborative
  setting
- **`ArcMarlEnvBase`**: An abstract base class that extends JaxMARL's
  `MultiAgentEnv` with ARC-specific functionality

## Design Principles

### 1. JAX Compatibility First

- All state components use JAX arrays with static shapes
- Full support for `jax.jit`, `jax.vmap`, and other transformations
- Immutable state updates using `dataclass.replace()`
- Proper pytree structure for efficient tree operations

### 2. JaxMARL Integration

- Extends `jaxmarl.environments.multi_agent_env.MultiAgentEnv`
- Follows JaxMARL conventions for observations, actions, and state management
- Compatible with JaxMARL training loops and utilities
- Lazy imports to avoid dependency issues during development

### 3. ARC-Specific Design

- Four-phase collaboration workflow: Ideation → Proposal → Voting → Commit
- Comprehensive hypothesis tracking and consensus mechanisms
- Grid manipulation with proper masking for variable-sized grids
- Solution verification and reward calculation

### 4. Modularity and Extensibility

- Abstract base class allows for multiple concrete implementations
- Configurable parameters for different experimental setups
- Clear separation of concerns between state management and environment logic

## Core Components

### ArcEnvState

A comprehensive state dataclass that tracks:

```python
@chex.dataclass
class ArcEnvState:
    # JaxMARL compatibility
    done: chex.Array  # Environment termination status
    step: int  # Current step number

    # ARC task state
    task_data: ParsedTaskData  # Current ARC task being solved
    current_test_case: jnp.ndarray  # Which test case is being worked on
    phase: jnp.ndarray  # Current collaboration phase (0-3)

    # Grid manipulation
    current_grid: jnp.ndarray  # Working grid being modified
    current_grid_mask: jnp.ndarray  # Valid cell mask
    target_grid: jnp.ndarray  # Ground truth solution
    target_grid_mask: jnp.ndarray  # Target valid cell mask

    # Agent collaboration
    agent_hypotheses: jnp.ndarray  # Agent-generated hypotheses
    hypothesis_votes: jnp.ndarray  # Vote counts per hypothesis
    consensus_threshold: jnp.ndarray  # Required votes for consensus
    active_agents: jnp.ndarray  # Which agents are participating

    # Timing and control
    phase_step: jnp.ndarray  # Steps within current phase
    max_phase_steps: jnp.ndarray  # Phase step limit
    episode_step: jnp.ndarray  # Overall episode steps
    max_episode_steps: jnp.ndarray  # Episode step limit

    # Performance tracking
    cumulative_rewards: jnp.ndarray  # Agent reward tracking
    solution_found: jnp.ndarray  # Whether solution is found
    last_action_valid: jnp.ndarray  # Action validity tracking
```

### ArcMarlEnvBase

An abstract base class that defines the interface for ARC environments:

#### Required Abstract Methods

- `reset()`: Initialize environment with new ARC task
- `step_env()`: Execute environment step with agent actions
- `get_obs()`: Generate observations for all agents
- `_load_task_data()`: Load ARC task data
- `_process_hypotheses()`: Handle agent hypothesis generation
- `_update_consensus()`: Update voting and consensus state
- `_apply_grid_transformation()`: Apply transformations to grid
- `_calculate_rewards()`: Compute agent rewards

#### Provided Helper Methods

- `_advance_phase()`: Move to next collaboration phase
- `_check_phase_completion()`: Determine if phase should end
- `_check_solution_correctness()`: Verify grid matches target
- `_is_terminal()`: Check termination conditions
- `_get_default_action_space()`: Standard action space structure
- `_get_default_observation_space()`: Standard observation space structure

## Four-Phase Collaboration Workflow

### Phase 0: Ideation

Agents privately observe the task and generate initial ideas without
communication.

### Phase 1: Proposal

Agents propose hypotheses for solving the current test case.

### Phase 2: Voting

Agents evaluate and vote on proposed hypotheses to build consensus.

### Phase 3: Commit

Apply the consensus hypothesis to modify the grid and check for solution.

## JAX Compatibility Features

### Static Shapes

All arrays are pre-allocated with maximum dimensions:

- Grids padded to `max_grid_size`
- Agent arrays sized to `num_agents`
- Hypothesis arrays sized to `max_hypotheses_per_agent`

### Immutable Updates

State changes use immutable updates:

```python
new_state = state.replace(phase=next_phase, phase_step=jnp.array(0, dtype=jnp.int32))
```

### Tree Operations

Supports JAX tree operations:

```python
# Works with jax.tree.map, jax.tree.leaves, etc.
modified_state = jax.tree.map(transform_fn, state)
```

### JIT Compilation

All methods designed for JIT compilation:

```python
@jax.jit
def step_env(self, key, state, actions):
    # Implementation here
    pass
```

## Usage Example

```python
from jaxarc.base import ArcMarlEnvBase, ArcEnvState


class MyArcEnv(ArcMarlEnvBase):
    def __init__(self, num_agents=3):
        super().__init__(
            num_agents=num_agents, max_grid_size=(30, 30), max_hypotheses_per_agent=5
        )

    def reset(self, key):
        # Load task and create initial state
        pass

    def step_env(self, key, state, actions):
        # Process actions and update state
        pass

    # Implement other abstract methods...


# Usage
env = MyArcEnv(num_agents=3)
key = jax.random.PRNGKey(42)
obs, state = env.reset(key)
```

## Testing

Comprehensive tests are provided:

- `test_arc_state_only.py`: Tests for `ArcEnvState` dataclass
- `test_base_env.py`: Tests for `ArcMarlEnvBase` (requires JaxMARL)

Run tests with:

```bash
pixi run -e test test tests/base/test_arc_state_only.py
```

## Configuration

The base environment accepts configuration parameters:

```python
env = ArcMarlEnvBase(
    num_agents=4,  # Number of collaborative agents
    max_grid_size=(30, 30),  # Maximum grid dimensions
    max_hypotheses_per_agent=5,  # Hypothesis capacity per agent
    hypothesis_dim=64,  # Hypothesis representation size
    consensus_threshold=3,  # Required votes (default: majority)
    max_phase_steps=10,  # Steps per collaboration phase
    max_episode_steps=100,  # Total episode step limit
)
```

## Next Steps

To create a concrete environment implementation:

1. **Extend ArcMarlEnvBase**: Implement all abstract methods
2. **Define Action Space**: Specify valid agent actions for each phase
3. **Implement Observation Generation**: Create observations from state
4. **Add Task Loading**: Connect to ARC dataset parsers
5. **Implement Hypothesis Processing**: Define hypothesis representation and
   voting
6. **Add Grid Transformations**: Implement grid manipulation operations
7. **Design Reward Function**: Define collaboration and solution rewards

## Dependencies

- JAX/JAXlib for array operations and transformations
- Chex for type checking and validation
- JaxMARL for multi-agent environment interface (lazy loaded)

## Integration Points

This base implementation is designed to work with:

- **JaxARC parsers**: For loading ARC task data
- **JaxARC types**: Uses `ParsedTaskData` and other core types
- **JaxMARL framework**: For training and evaluation
- **JAX ecosystem**: For high-performance computation
