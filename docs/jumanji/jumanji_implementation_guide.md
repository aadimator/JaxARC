# JaxARC Jumanji Implementation Guide

## Overview

This guide provides step-by-step technical instructions for implementing the JaxARC environment using Jumanji for single-agent and Mava for multi-agent scenarios. This guide complements the transition analysis and focuses on practical implementation details.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Single-Agent Jumanji Implementation](#single-agent-jumanji-implementation)
3. [Multi-Agent Mava Extension](#multi-agent-mava-extension)
4. [Testing and Validation](#testing-and-validation)
5. [Performance Optimization](#performance-optimization)
6. [Integration Patterns](#integration-patterns)

## Environment Setup

### 1. Update Dependencies

**Update `pyproject.toml`:**

```toml
[tool.pixi.dependencies]
# Core JAX ecosystem
python = "3.12.*"
jax = { extras = ["cuda12"] }
chex = ">=0.1.86,<0.2"

# Jumanji and Mava
jumanji = ">=1.1.0,<2"
id-mava = ">=0.2.0,<0.3"

# Existing dependencies
tqdm = ">=4.67.1,<5"
typer = ">=0.16.0,<0.17"
pyprojroot = ">=0.3.0,<0.4"
loguru = ">=0.7.3,<0.8"
hydra-core = ">=1.3.2,<2"
rich = ">=14.0.0,<15"
drawsvg = ">=2.4.0,<3"
kaggle = ">=1.6.17,<2"

# Remove JaxMARL if present
# jaxmarl = ">=0.0.3,<0.1"  # Remove this line
```

**Install dependencies:**

```bash
cd JaxARC
pixi install
```

### 2. Verify Installation

**Test script (`scripts/test_jumanji_install.py`):**

```python
#!/usr/bin/env python3
"""Test Jumanji and Mava installation."""

import jax
import jax.numpy as jnp
import jumanji
import mava
import chex

def test_jumanji():
    """Test basic Jumanji functionality."""
    print("Testing Jumanji...")
    
    # Test environment creation
    env = jumanji.make('Snake-v1')
    key = jax.random.PRNGKey(0)
    
    # Test reset
    state, timestep = env.reset(key)
    print(f"Initial state type: {type(state)}")
    print(f"Initial timestep: {timestep}")
    
    # Test step
    action = env.action_spec().generate_value()
    state, timestep = env.step(state, action)
    print(f"After step timestep: {timestep}")
    
    print("✅ Jumanji working correctly!")

def test_mava():
    """Test basic Mava functionality."""
    print("Testing Mava...")
    
    # Test imports
    from mava.environments import Environment as MavaEnvironment
    from mava.systems import IPPO
    
    print("✅ Mava imports working correctly!")

if __name__ == "__main__":
    test_jumanji()
    test_mava()
    print("🎉 All tests passed!")
```

## Single-Agent Jumanji Implementation

### 1. Core State Definition

**`src/jaxarc/envs/states.py`:**

```python
"""Jumanji-compatible state definitions for ARC environments."""

from __future__ import annotations

import chex
import jax.numpy as jnp
from jumanji.types import State
from typing import Any, Dict

from jaxarc.types import ParsedTaskData


@chex.dataclass
class ArcSingleAgentState(State):
    """
    State for single-agent ARC environment.
    
    This state encompasses all information needed for a single agent
    to reason about and solve ARC tasks.
    """
    
    # Task information
    task_data: ParsedTaskData
    current_test_index: jnp.ndarray  # Which test case we're solving
    
    # Current working grid
    working_grid: jnp.ndarray  # Shape: (max_grid_h, max_grid_w)
    working_mask: jnp.ndarray  # Boolean mask for valid cells
    
    # Agent reasoning state
    scratchpad: jnp.ndarray  # Private reasoning space
    attention_map: jnp.ndarray  # Where agent is focusing
    reasoning_trace: jnp.ndarray  # Step-by-step reasoning history
    
    # Episode management
    step_count: jnp.ndarray  # Current step number
    max_steps: jnp.ndarray   # Maximum steps allowed
    is_done: jnp.ndarray     # Episode termination flag
    
    # Performance tracking
    last_reward: jnp.ndarray  # Most recent reward
    cumulative_reward: jnp.ndarray  # Total episode reward
    
    # Metadata
    episode_id: jnp.ndarray  # Episode identifier
    
    def __post_init__(self) -> None:
        """Validate state structure."""
        # Validate scalar fields
        chex.assert_rank(self.current_test_index, 0)
        chex.assert_rank(self.step_count, 0)
        chex.assert_rank(self.max_steps, 0)
        chex.assert_rank(self.is_done, 0)
        chex.assert_rank(self.last_reward, 0)
        chex.assert_rank(self.cumulative_reward, 0)
        chex.assert_rank(self.episode_id, 0)
        
        # Validate grid fields
        chex.assert_rank(self.working_grid, 2)
        chex.assert_rank(self.working_mask, 2)
        chex.assert_shape(self.working_mask, self.working_grid.shape)
        
        # Validate reasoning fields
        chex.assert_rank(self.scratchpad, 3)  # (reasoning_steps, grid_h, grid_w)
        chex.assert_rank(self.attention_map, 2)  # (grid_h, grid_w)
        chex.assert_rank(self.reasoning_trace, 2)  # (max_steps, trace_dim)


@chex.dataclass  
class ArcMultiAgentState(State):
    """
    State for multi-agent ARC environment.
    
    Extends single-agent state with collaborative reasoning components.
    """
    
    # Base single-agent components
    task_data: ParsedTaskData
    current_test_index: jnp.ndarray
    working_grid: jnp.ndarray
    working_mask: jnp.ndarray
    
    # Multi-agent specific
    agent_scratchpads: Dict[str, jnp.ndarray]  # Per-agent private reasoning
    shared_hypotheses: jnp.ndarray  # Shared hypothesis space
    voting_state: jnp.ndarray  # Current voting information
    consensus_grid: jnp.ndarray  # Consensus working grid
    
    # Reasoning phase management
    current_phase: jnp.ndarray  # 0=scratchpad, 1=hypothesis, 2=voting, 3=consensus
    phase_timer: jnp.ndarray    # Time remaining in current phase
    phase_transitions: jnp.ndarray  # History of phase transitions
    
    # Collaboration metrics
    agreement_scores: jnp.ndarray  # Agent agreement measurements
    participation_flags: jnp.ndarray  # Which agents are active
    
    # Episode management (shared with single-agent)
    step_count: jnp.ndarray
    max_steps: jnp.ndarray
    is_done: jnp.ndarray
    last_reward: jnp.ndarray
    cumulative_reward: jnp.ndarray
    episode_id: jnp.ndarray
```

### 2. Action and Observation Specifications

**`src/jaxarc/envs/specs.py`:**

```python
"""Environment specifications for ARC environments."""

from __future__ import annotations

import jax.numpy as jnp
from jumanji.specs import Spec, Array, DiscreteArray, BoundedArray
from typing import Dict, Any

# Constants
MAX_GRID_SIZE = 32
NUM_ARC_COLORS = 10
MAX_REASONING_STEPS = 16
MAX_EPISODE_STEPS = 100


def single_agent_observation_spec() -> Dict[str, Spec]:
    """Observation specification for single-agent ARC environment."""
    return {
        # Current working grid
        'working_grid': Array(
            shape=(MAX_GRID_SIZE, MAX_GRID_SIZE), 
            dtype=jnp.int32,
            name='working_grid'
        ),
        'working_mask': Array(
            shape=(MAX_GRID_SIZE, MAX_GRID_SIZE), 
            dtype=jnp.bool_,
            name='working_mask'
        ),
        
        # Task examples
        'train_inputs': Array(
            shape=(4, MAX_GRID_SIZE, MAX_GRID_SIZE), 
            dtype=jnp.int32,
            name='train_inputs'
        ),
        'train_outputs': Array(
            shape=(4, MAX_GRID_SIZE, MAX_GRID_SIZE), 
            dtype=jnp.int32,
            name='train_outputs'
        ),
        'train_masks': Array(
            shape=(4, MAX_GRID_SIZE, MAX_GRID_SIZE), 
            dtype=jnp.bool_,
            name='train_masks'
        ),
        
        # Current test input
        'test_input': Array(
            shape=(MAX_GRID_SIZE, MAX_GRID_SIZE), 
            dtype=jnp.int32,
            name='test_input'
        ),
        'test_mask': Array(
            shape=(MAX_GRID_SIZE, MAX_GRID_SIZE), 
            dtype=jnp.bool_,
            name='test_mask'
        ),
        
        # Agent state information
        'attention_map': Array(
            shape=(MAX_GRID_SIZE, MAX_GRID_SIZE), 
            dtype=jnp.float32,
            name='attention_map'
        ),
        'reasoning_summary': Array(
            shape=(MAX_REASONING_STEPS,), 
            dtype=jnp.float32,
            name='reasoning_summary'
        ),
        
        # Episode information
        'step_count': DiscreteArray(
            num_values=MAX_EPISODE_STEPS,
            name='step_count'
        ),
        'test_index': DiscreteArray(
            num_values=4,  # Max test cases per task
            name='test_index'
        ),
    }


def single_agent_action_spec() -> Dict[str, Spec]:
    """Action specification for single-agent ARC environment."""
    return {
        # Primary action type
        'action_type': DiscreteArray(
            num_values=6,  # paint, select, copy, paste, transform, submit
            name='action_type'
        ),
        
        # Position specification
        'position': BoundedArray(
            shape=(2,),
            minimum=0,
            maximum=MAX_GRID_SIZE - 1,
            dtype=jnp.int32,
            name='position'
        ),
        
        # Color specification
        'color': DiscreteArray(
            num_values=NUM_ARC_COLORS,
            name='color'
        ),
        
        # Region specification (for selection/transformation)
        'region_size': BoundedArray(
            shape=(2,),
            minimum=1,
            maximum=MAX_GRID_SIZE,
            dtype=jnp.int32,
            name='region_size'
        ),
        
        # Transformation type (rotation, reflection, etc.)
        'transform_type': DiscreteArray(
            num_values=8,  # identity, rot90, rot180, rot270, fliph, flipv, etc.
            name='transform_type'
        ),
        
        # Attention update
        'attention_position': BoundedArray(
            shape=(2,),
            minimum=0,
            maximum=MAX_GRID_SIZE - 1,
            dtype=jnp.int32,
            name='attention_position'
        ),
        'attention_radius': BoundedArray(
            shape=(),
            minimum=1,
            maximum=MAX_GRID_SIZE // 2,
            dtype=jnp.int32,
            name='attention_radius'
        ),
    }


def multi_agent_observation_spec(num_agents: int) -> Dict[str, Spec]:
    """Observation specification for multi-agent ARC environment."""
    base_spec = single_agent_observation_spec()
    
    # Add multi-agent specific observations
    multi_agent_spec = {
        **base_spec,
        
        # Shared state
        'consensus_grid': Array(
            shape=(MAX_GRID_SIZE, MAX_GRID_SIZE), 
            dtype=jnp.int32,
            name='consensus_grid'
        ),
        'shared_hypotheses': Array(
            shape=(num_agents, 64),  # Encoded hypothesis representations
            dtype=jnp.float32,
            name='shared_hypotheses'
        ),
        
        # Phase information
        'current_phase': DiscreteArray(
            num_values=4,  # scratchpad, hypothesis, voting, consensus
            name='current_phase'
        ),
        'phase_timer': BoundedArray(
            shape=(),
            minimum=0,
            maximum=50,  # Max steps per phase
            dtype=jnp.int32,
            name='phase_timer'
        ),
        
        # Collaboration information
        'agreement_scores': Array(
            shape=(num_agents,),
            dtype=jnp.float32,
            name='agreement_scores'
        ),
        'agent_participation': Array(
            shape=(num_agents,),
            dtype=jnp.bool_,
            name='agent_participation'
        ),
    }
    
    return multi_agent_spec


def multi_agent_action_spec(num_agents: int) -> Dict[str, Spec]:
    """Action specification for multi-agent ARC environment."""
    base_spec = single_agent_action_spec()
    
    # Add multi-agent specific actions
    multi_agent_spec = {
        **base_spec,
        
        # Voting actions
        'vote_target': DiscreteArray(
            num_values=num_agents,  # Which agent's hypothesis to vote for
            name='vote_target'
        ),
        'vote_confidence': BoundedArray(
            shape=(),
            minimum=0.0,
            maximum=1.0,
            dtype=jnp.float32,
            name='vote_confidence'
        ),
        
        # Communication actions
        'share_hypothesis': DiscreteArray(
            num_values=2,  # share or don't share
            name='share_hypothesis'
        ),
        'hypothesis_data': Array(
            shape=(64,),  # Encoded hypothesis
            dtype=jnp.float32,
            name='hypothesis_data'
        ),
    }
    
    return multi_agent_spec
```

### 3. Core Single-Agent Environment

**`src/jaxarc/envs/arc_single_agent.py`:**

```python
"""Single-agent ARC environment implementation using Jumanji."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from jax import random
from jumanji import Environment
from jumanji.types import TimeStep, StepType, restart, transition, termination
from typing import Tuple, Dict, Any, Optional

from jaxarc.types import ParsedTaskData
from jaxarc.envs.states import ArcSingleAgentState
from jaxarc.envs.specs import (
    single_agent_observation_spec, 
    single_agent_action_spec,
    MAX_GRID_SIZE,
    MAX_EPISODE_STEPS
)


class ArcSingleAgentEnv(Environment[ArcSingleAgentState]):
    """
    Single-agent ARC environment using Jumanji framework.
    
    This environment allows a single agent to reason about and solve
    ARC tasks through a sequence of grid manipulation actions.
    """
    
    def __init__(
        self,
        max_steps: int = MAX_EPISODE_STEPS,
        grid_size: int = MAX_GRID_SIZE,
        reward_config: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the ARC single-agent environment.
        
        Args:
            max_steps: Maximum number of steps per episode
            grid_size: Maximum grid size for padding
            reward_config: Reward function configuration
        """
        super().__init__()
        
        self.max_steps = max_steps
        self.grid_size = grid_size
        
        # Default reward configuration
        self.reward_config = reward_config or {
            'correct_pixel': 1.0,
            'incorrect_pixel': -0.1,
            'no_change': -0.01,
            'task_completion': 100.0,
            'invalid_action': -1.0,
        }
    
    def reset(self, key: chex.PRNGKey) -> Tuple[ArcSingleAgentState, TimeStep]:
        """
        Reset the environment to initial state.
        
        Args:
            key: JAX random key
            
        Returns:
            Tuple of (initial_state, initial_timestep)
        """
        # This is a placeholder - in practice, you'd load task_data
        # from your parser or pass it in somehow
        dummy_task_data = self._create_dummy_task_data()
        
        # Initialize state
        initial_state = ArcSingleAgentState(
            task_data=dummy_task_data,
            current_test_index=jnp.array(0, dtype=jnp.int32),
            working_grid=jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32),
            working_mask=jnp.ones((self.grid_size, self.grid_size), dtype=jnp.bool_),
            scratchpad=jnp.zeros((16, self.grid_size, self.grid_size), dtype=jnp.float32),
            attention_map=jnp.ones((self.grid_size, self.grid_size), dtype=jnp.float32),
            reasoning_trace=jnp.zeros((self.max_steps, 64), dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            max_steps=jnp.array(self.max_steps, dtype=jnp.int32),
            is_done=jnp.array(False, dtype=jnp.bool_),
            last_reward=jnp.array(0.0, dtype=jnp.float32),
            cumulative_reward=jnp.array(0.0, dtype=jnp.float32),
            episode_id=random.randint(key, (), 0, 1000000),
        )
        
        # Create initial timestep
        observation = self._get_observation(initial_state)
        timestep = restart(observation=observation)
        
        return initial_state, timestep
    
    def step(
        self, 
        state: ArcSingleAgentState, 
        action: Dict[str, jnp.ndarray]
    ) -> Tuple[ArcSingleAgentState, TimeStep]:
        """
        Take a step in the environment.
        
        Args:
            state: Current environment state
            action: Action dictionary matching action spec
            
        Returns:
            Tuple of (new_state, timestep)
        """
        # Process the action
        new_state = self._process_action(state, action)
        
        # Update step count
        new_state = new_state.replace(
            step_count=state.step_count + 1
        )
        
        # Calculate reward
        reward = self._calculate_reward(state, new_state, action)
        new_state = new_state.replace(
            last_reward=reward,
            cumulative_reward=state.cumulative_reward + reward
        )
        
        # Check termination
        is_terminal = self._is_terminal(new_state)
        new_state = new_state.replace(is_done=is_terminal)
        
        # Create timestep
        observation = self._get_observation(new_state)
        
        if is_terminal:
            timestep = termination(
                reward=reward,
                observation=observation
            )
        else:
            timestep = transition(
                reward=reward,
                observation=observation
            )
        
        return new_state, timestep
    
    def _process_action(
        self, 
        state: ArcSingleAgentState, 
        action: Dict[str, jnp.ndarray]
    ) -> ArcSingleAgentState:
        """Process an agent action and update the state."""
        action_type = action['action_type']
        position = action['position']
        color = action['color']
        
        # Get current working grid
        working_grid = state.working_grid
        
        # Process different action types
        new_grid = jax.lax.switch(
            action_type,
            [
                lambda: self._action_paint(working_grid, position, color),
                lambda: self._action_select(working_grid, position, action['region_size']),
                lambda: self._action_copy(working_grid, position, action['region_size']),
                lambda: self._action_paste(working_grid, position),
                lambda: self._action_transform(working_grid, position, action['transform_type']),
                lambda: working_grid,  # submit action doesn't change grid
            ]
        )
        
        # Update attention map
        new_attention = self._update_attention(
            state.attention_map, 
            action['attention_position'], 
            action['attention_radius']
        )
        
        return state.replace(
            working_grid=new_grid,
            attention_map=new_attention
        )
    
    def _action_paint(
        self, 
        grid: jnp.ndarray, 
        position: jnp.ndarray, 
        color: jnp.ndarray
    ) -> jnp.ndarray:
        """Paint a single pixel."""
        row, col = position[0], position[1]
        
        # Bounds checking
        valid_position = (
            (row >= 0) & (row < self.grid_size) & 
            (col >= 0) & (col < self.grid_size)
        )
        
        # Update grid conditionally
        new_grid = jnp.where(
            valid_position,
            grid.at[row, col].set(color),
            grid
        )
        
        return new_grid
    
    def _action_select(
        self, 
        grid: jnp.ndarray, 
        position: jnp.ndarray, 
        region_size: jnp.ndarray
    ) -> jnp.ndarray:
        """Select a region (for now, just return the grid unchanged)."""
        # TODO: Implement region selection logic
        return grid
    
    def _action_copy(
        self, 
        grid: jnp.ndarray, 
        position: jnp.ndarray, 
        region_size: jnp.ndarray
    ) -> jnp.ndarray:
        """Copy a region."""
        # TODO: Implement copy logic
        return grid
    
    def _action_paste(
        self, 
        grid: jnp.ndarray, 
        position: jnp.ndarray
    ) -> jnp.ndarray:
        """Paste previously copied region."""
        # TODO: Implement paste logic
        return grid
    
    def _action_transform(
        self, 
        grid: jnp.ndarray, 
        position: jnp.ndarray, 
        transform_type: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply transformation to a region."""
        # TODO: Implement transformation logic
        return grid
    
    def _update_attention(
        self, 
        attention_map: jnp.ndarray, 
        position: jnp.ndarray, 
        radius: jnp.ndarray
    ) -> jnp.ndarray:
        """Update agent attention map."""
        row, col = position[0], position[1]
        
        # Create attention update
        y, x = jnp.ogrid[:self.grid_size, :self.grid_size]
        distance = jnp.sqrt((y - row)**2 + (x - col)**2)
        attention_update = jnp.exp(-distance / radius)
        
        # Exponential moving average update
        alpha = 0.1
        new_attention = alpha * attention_update + (1 - alpha) * attention_map
        
        return new_attention
    
    def _calculate_reward(
        self, 
        old_state: ArcSingleAgentState, 
        new_state: ArcSingleAgentState, 
        action: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Calculate reward for the transition."""
        # Get target grid (ground truth)
        target_grid = new_state.task_data.true_test_output_grids[
            new_state.current_test_index
        ]
        target_mask = new_state.task_data.true_test_output_masks[
            new_state.current_test_index
        ]
        
        # Calculate pixel-wise accuracy
        working_grid = new_state.working_grid
        correct_pixels = jnp.sum(
            (working_grid == target_grid) & target_mask
        )
        total_pixels = jnp.sum(target_mask)
        
        # Pixel accuracy reward
        pixel_accuracy = correct_pixels / jnp.maximum(total_pixels, 1)
        pixel_reward = pixel_accuracy * self.reward_config['correct_pixel']
        
        # Check if task is complete
        task_complete = jnp.all(
            (working_grid == target_grid) | ~target_mask
        )
        completion_reward = jnp.where(
            task_complete,
            self.reward_config['task_completion'],
            0.0
        )
        
        # Small negative reward for each step (encourage efficiency)
        step_penalty = self.reward_config['no_change']
        
        total_reward = pixel_reward + completion_reward + step_penalty
        
        return total_reward
    
    def _is_terminal(self, state: ArcSingleAgentState) -> jnp.ndarray:
        """Check if episode should terminate."""
        # Terminal if max steps reached
        max_steps_reached = state.step_count >= state.max_steps
        
        # Terminal if task is solved
        target_grid = state.task_data.true_test_output_grids[
            state.current_test_index
        ]
        target_mask = state.task_data.true_test_output_masks[
            state.current_test_index
        ]
        
        task_solved = jnp.all(
            (state.working_grid == target_grid) | ~target_mask
        )
        
        return max_steps_reached | task_solved
    
    def _get_observation(self, state: ArcSingleAgentState) -> Dict[str, jnp.ndarray]:
        """Get observation from current state."""
        return {
            'working_grid': state.working_grid,
            'working_mask': state.working_mask,
            'train_inputs': state.task_data.input_grids_examples,
            'train_outputs': state.task_data.output_grids_examples,
            'train_masks': state.task_data.input_masks_examples,
            'test_input': state.task_data.test_input_grids[state.current_test_index],
            'test_mask': state.task_data.test_input_masks[state.current_test_index],
            'attention_map': state.attention_map,
            'reasoning_summary': state.reasoning_trace[state.step_count],
            'step_count': state.step_count,
            'test_index': state.current_test_index,
        }
    
    def _create_dummy_task_data(self) -> ParsedTaskData:
        """Create dummy task data for testing."""
        # This is just for testing - replace with actual task loading
        return ParsedTaskData(
            input_grids_examples=jnp.zeros((4, self.grid_size, self.grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((4, self.grid_size, self.grid_size), dtype=jnp.bool_),
            output_grids_examples=jnp.zeros((4, self.grid_size, self.grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((4, self.grid_size, self.grid_size), dtype=jnp.bool_),
            num_train_pairs=4,
            test_input_grids=jnp.zeros((2, self.grid_size, self.grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((2, self.grid_size, self.grid_size), dtype=jnp.bool_),
            true_test_output_grids=jnp.ones((2, self.grid_size, self.grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((2, self.grid_size, self.grid_size), dtype=jnp.bool_),
            num_test_pairs=2,
            task_id="dummy_task"
        )
    
    def observation_spec(self) -> Dict[str, Any]:
        """Return observation specification."""
        return single_agent_observation_spec()
    
    def action_spec(self) -> Dict[str, Any]:
        """Return action specification."""
        return single_agent_action_spec()
```

### 4. Integration with Existing Parser

**`src/jaxarc/envs/env_factory.py`:**

```python
"""Factory functions for creating ARC environments with parsed data."""

from __future__ import annotations

import chex
from typing import Dict, Any, Optional

from jaxarc.parsers.arc_agi_parser import ArcAgiParser
from jaxarc.envs.arc_single_agent import ArcSingleAgentEnv
from jaxarc.types import ParsedTaskData


class ArcEnvFactory:
    """Factory for creating ARC environments with task data."""
    
    def __init__(self, parser: ArcAgiParser):
        """Initialize with a parser instance."""
        self.parser = parser
    
    def create_single_agent_env(
        self, 
        task_path: Optional[str] = None,
        max_steps: int = 100,
        **env_kwargs
    ) -> tuple[ArcSingleAgentEnv, ParsedTaskData]:
        """
        Create a single-agent environment with task data.
        
        Args:
            task_path: Path to specific task, or None for random task
            max_steps: Maximum steps per episode
            **env_kwargs: Additional environment configuration
            
        Returns:
            Tuple of (environment, task_data)
        """
        # Load task data
        if task_path is None:
            task_data = self.parser.get_random_task()
        else:
            task_data = self.parser.load_and_parse_task(task_path)
        
        # Create environment
        env = ArcSingleAgentEnv(
            max_steps=max_steps,
            **env_kwargs
        )
        
        return env, task_data
    
    def create_multi_agent_env(
        self,
        num_agents: int,
        task_path: Optional[str] = None,
        max_steps: int = 100,
        **env_kwargs
    ) -> tuple[Any, ParsedTaskData]:  # Import multi-agent env when implemented
        """
        Create a multi-agent environment with task data.
        
        Args:
            num_agents: Number of agents
            task_path: Path to specific task, or None for random task
            max_steps: Maximum steps per episode
            **env_kwargs: Additional environment configuration
            
        Returns:
            Tuple of (environment, task_data)
        """
        # Load task data
        if task_path is None:
            task_data = self.parser.get_random_task()
        else:
            task_data = self.parser.load_and_parse_task(task_path)
        
        # TODO: Implement multi-agent environment creation
        raise NotImplementedError("Multi-agent environment not yet implemented")


def make_arc_env(
    env_type: str = "single",
    num_agents: int = 1,
    data_path: str = "data/arc-agi",
    task_path: Optional[str] = None,
    **env_kwargs
) -> tuple[Any, ParsedTaskData]:
    """
    Convenience function to create ARC environments.
    
    Args:
        env_type: "single" or "multi"
        num_agents: Number of agents (for multi-agent)
        data_path: Path to ARC data directory
        task_path: Specific task path or None for random
        **env_kwargs: Environment configuration
        
    Returns:
        Tuple of (environment, task_data)
    """
    # Create parser
    parser = ArcAgiParser(data_path)
    factory = ArcEnvFactory(parser)
    
    if env_type == "single":
        return factory.create_single_agent_env(task_path=task_path, **env_kwargs)
    elif env_type == "multi":
        return factory.create_multi_agent_env(
            num_agents=num_agents, 
            task_path=task_path, 
            **env_kwargs
        )
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
```

## Multi-Agent Mava Extension

### 1. Multi-Agent Environment Implementation

**`src/jaxarc/envs/arc_multi_agent.py`:**

```python
"""Multi-agent ARC environment implementation using Mava."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from jax import random
from mava.environments import Environment as MavaEnvironment
from mava.types import State, TimeStep
from typing import Tuple, Dict, Any, Optional

from jaxarc.types import ParsedTaskData
from jaxarc.envs.states import ArcMultiAgentState
from jaxarc.envs.specs import multi_agent_observation_spec, multi_agent_action_spec
from jaxarc.envs.arc_single_agent import ArcSingleAgentEnv


class ArcMultiAgentEnv(MavaEnvironment):
    """
    Multi-agent ARC environment with 4-phase collaborative reasoning.
    
    Phases:
    0. Scratchpad: Private agent reasoning
    1. Hypothesis: Public proposal sharing  
    2. Voting: Collaborative decision making
    3. Consensus: Apply agreed-upon changes
    """
    
    def __init__(
        self,
        num_agents: int = 4,
        max_steps: int = 200,
        phase_lengths: Dict[int, int] = None,
        collaboration_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize multi-agent ARC environment.
        
        Args:
            num_agents: Number of collaborating agents
            max_steps: Maximum total steps per episode
            phase_lengths: Steps per reasoning phase
            collaboration_config: Configuration for collaboration mechanics
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.max_steps = max_steps
        
        # Default phase lengths
        self.phase_lengths = phase_lengths or {
            0: 20,  # Scratchpad phase
            1: 10,  # Hypothesis phase
            2: 10,  # Voting phase
            3: 5,   # Consensus phase
        }
        
        # Collaboration configuration
        self.collaboration_config = collaboration_config or {
            'voting_threshold': 0.6,
            'consensus_method': 'weighted_average',
            'min_participation': 0.5,
        }
        
        # Agent identifiers
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
    
    def reset(self, key: chex.PRNGKey) -> Tuple[ArcMultiAgentState, TimeStep]:
        """Reset environment to initial multi-agent state."""
        # Initialize base state (similar to single-agent)
        dummy_task_data = self._create_dummy_task_data()
        
        # Initialize multi-agent specific state
        initial_state = ArcMultiAgentState(
            # Base components
            task_data=dummy_task_data,
            current_test_index=jnp.array(0, dtype=jnp.int32),
            working_grid=jnp.zeros((32, 32), dtype=jnp.int32),
            working_mask=jnp.ones((32, 32), dtype=jnp.bool_),
            
            # Multi-agent components
            agent_scratchpads={
                agent_id: jnp.zeros((16, 32, 32), dtype=jnp.float32)
                for agent_id in self.agent_ids
            },
            shared_hypotheses=jnp.zeros((self.num_agents, 64), dtype=jnp.float32),
            voting_state=jnp.zeros((self.num_agents, self.num_agents), dtype=jnp.float32),
            consensus_grid=jnp.zeros((32, 32), dtype=jnp.int32),
            
            # Phase management
            current_phase=jnp.array(0, dtype=jnp.int32),
            phase_timer=jnp.array(self.phase_lengths[0], dtype=jnp.int32),
            phase_transitions=jnp.zeros((100,), dtype=jnp.int32),
            
            # Collaboration metrics
            agreement_scores=jnp.zeros((self.num_agents,), dtype=jnp.float32),
            participation_flags=jnp.ones((self.num_agents,), dtype=jnp.bool_),
            
            # Episode management
            step_count=jnp.array(0, dtype=jnp.int32),
            max_steps=jnp.array(self.max_steps, dtype=jnp.int32),
            is_done=jnp.array(False, dtype=jnp.bool_),
            last_reward=jnp.array(0.0, dtype=jnp.float32),
            cumulative_reward=jnp.array(0.0, dtype=jnp.float32),
            episode_id=random.randint(key, (), 0, 1000000),
        )
        
        # Create initial observations for all agents
        observations = self._get_observations(initial_state)
        
        # Create initial timestep
        timestep = TimeStep(
            observation=observations,
            reward={agent_id: 0.0 for agent_id in self.agent_ids},
            discount={agent_id: 1.0 for agent_id in self.agent_ids},
            step_type={agent_id: 'FIRST' for agent_id in self.agent_ids},
        )
        
        return initial_state, timestep
    
    def step(
        self,
        state: ArcMultiAgentState,
        actions: Dict[str, Dict[str, jnp.ndarray]]
    ) -> Tuple[ArcMultiAgentState, TimeStep]:
        """Take a step with multi-agent actions."""
        # Route to phase-specific processing
        new_state = jax.lax.switch(
            state.current_phase,
            [
                lambda: self._process_scratchpad_phase(state, actions),
                lambda: self._process_hypothesis_phase(state, actions),
                lambda: self._process_voting_phase(state, actions),
                lambda: self._process_consensus_phase(state, actions),
            ]
        )
        
        # Update step count and phase timer
        new_state = self._update_phase_timer(new_state)
        
        # Calculate rewards
        rewards = self._calculate_multi_agent_rewards(state, new_state, actions)
        
        # Check termination
        is_terminal = self._is_terminal(new_state)
        new_state = new_state.replace(is_done=is_terminal)
        
        # Create observations
        observations = self._get_observations(new_state)
        
        # Create timestep
        timestep = TimeStep(
            observation=observations,
            reward=rewards,
            discount={agent_id: 0.0 if is_terminal else 1.0 for agent_id in self.agent_ids},
            step_type={agent_id: 'LAST' if is_terminal else 'MID' for agent_id in self.agent_ids},
        )
        
        return new_state, timestep
    
    def _process_scratchpad_phase(
        self,
        state: ArcMultiAgentState,
        actions: Dict[str, Dict[str, jnp.ndarray]]
    ) -> ArcMultiAgentState:
        """Process actions during scratchpad (private reasoning) phase."""
        new_scratchpads = {}
        
        for agent_id in self.agent_ids:
            if agent_id in actions:
                agent_action = actions[agent_id]
                # Update agent's private scratchpad
                new_scratchpad = self._update_scratchpad(
                    state.agent_scratchpads[agent_id],
                    agent_action
                )
                new_scratchpads[agent_id] = new_scratchpad
            else:
                new_scratchpads[agent_id] = state.agent_scratchpads[agent_id]
        
        return state.replace(agent_scratchpads=new_scratchpads)
    
    def _process_hypothesis_phase(
        self,
        state: ArcMultiAgentState,
        actions: Dict[str, Dict[str, jnp.ndarray]]
    ) -> ArcMultiAgentState:
        """Process actions during hypothesis proposal phase."""
        new_hypotheses = state.shared_hypotheses
        
        for i, agent_id in enumerate(self.agent_ids):
            if agent_id in actions:
                agent_action = actions[agent_id]
                # Check if agent wants to share hypothesis
                if agent_action.get('share_hypothesis', 0) == 1:
                    hypothesis_data = agent_action.get('hypothesis_data')
                    new_hypotheses = new_hypotheses.at[i].set(hypothesis_data)
        
        return state.replace(shared_hypotheses=new_hypotheses)
    
    def _process_voting_phase(
        self,
        state: ArcMultiAgentState,
        actions: Dict[str, Dict[str, jnp.ndarray]]
    ) -> ArcMultiAgentState:
        """Process actions during voting phase."""
        new_voting_state = state.voting_state
        
        for i, agent_id in enumerate(self.agent_ids):
            if agent_id in actions:
                agent_action = actions[agent_id]
                vote_target = agent_action.get('vote_target', i)  # Default to self-vote
                vote_confidence = agent_action.get('vote_confidence', 0.0)
                
                # Update voting matrix
                new_voting_state = new_voting_state.at[i, vote_target].set(vote_confidence)
        
        return state.replace(voting_state=new_voting_state)
    
    def _process_consensus_phase(
        self,
        state: ArcMultiAgentState,
        actions: Dict[str, Dict[str, jnp.ndarray]]
    ) -> ArcMultiAgentState:
        """Process consensus resolution and apply changes."""
        # Determine winning hypothesis based on votes
        vote_totals = jnp.sum(state.voting_state, axis=0)
        winning_agent = jnp.argmax(vote_totals)
        
        # Apply winning hypothesis to consensus grid
        winning_hypothesis = state.shared_hypotheses[winning_agent]
        new_consensus_grid = self._apply_hypothesis(
            state.consensus_grid,
            winning_hypothesis
        )
        
        # Calculate agreement scores
        agreement_scores = self._calculate_agreement_scores(state.voting_state)
        
        return state.replace(
            consensus_grid=new_consensus_grid,
            agreement_scores=agreement_scores,
            working_grid=new_consensus_grid  # Update main working grid
        )
    
    def _update_phase_timer(self, state: ArcMultiAgentState) -> ArcMultiAgentState:
        """Update phase timer and transition phases if needed."""
        new_timer = state.phase_timer - 1
        new_step_count = state.step_count + 1
        
        # Check if phase should transition
        should_transition = new_timer <= 0
        
        # Calculate next phase
        next_phase = (state.current_phase + 1) % 4
        next_timer = jnp.where(
            should_transition,
            self.phase_lengths.get(next_phase, 10),
            new_timer
        )
        next_phase = jnp.where(should_transition, next_phase, state.current_phase)
        
        return state.replace(
            phase_timer=next_timer,
            current_phase=next_phase,
            step_count=new_step_count
        )
    
    def _get_observations(self, state: ArcMultiAgentState) -> Dict[str, Dict[str, jnp.ndarray]]:
        """Get observations for all agents."""
        base_obs = {
            'working_grid': state.working_grid,
            'working_mask': state.working_mask,
            'train_inputs': state.task_data.input_grids_examples,
            'train_outputs': state.task_data.output_grids_examples,
            'train_masks': state.task_data.input_masks_examples,
            'test_input': state.task_data.test_input_grids[state.current_test_index],
            'test_mask': state.task_data.test_input_masks[state.current_test_index],
            'consensus_grid': state.consensus_grid,
            'shared_hypotheses': state.shared_hypotheses,
            'current_phase': state.current_phase,
            'phase_timer': state.phase_timer,
            'agreement_scores': state.agreement_scores,
            'agent_participation': state.participation_flags,
            'step_count': state.step_count,
            'test_index': state.current_test_index,
        }
        
        # Add agent-specific observations
        observations = {}
        for i, agent_id in enumerate(self.agent_ids):
            agent_obs = base_obs.copy()
            agent_obs.update({
                'agent_scratchpad': state.agent_scratchpads[agent_id],
                'agent_id': jnp.array(i, dtype=jnp.int32),
                'voting_state': state.voting_state[i],  # This agent's votes
            })
            observations[agent_id] = agent_obs
        
        return observations
    
    def observation_spec(self) -> Dict[str, Any]:
        """Return multi-agent observation specification."""
        return multi_agent_observation_spec(self.num_agents)
    
    def action_spec(self) -> Dict[str, Any]:
        """Return multi-agent action specification."""
        return multi_agent_action_spec(self.num_agents)
```

## Testing and Validation

### 1. Unit Tests

**`tests/test_jumanji_environment.py`:**

```python
"""Tests for Jumanji-based ARC environments."""

import pytest
import jax
import jax.numpy as jnp
from jaxarc.envs.arc_single_agent import ArcSingleAgentEnv
from jaxarc.envs.env_factory import make_arc_env


class TestSingleAgentEnv:
    """Test suite for single-agent ARC environment."""
    
    def test_environment_creation(self):
        """Test basic environment creation."""
        env = ArcSingleAgentEnv()
        assert env is not None
        assert env.max_steps > 0
    
    def test_reset_functionality(self):
        """Test environment reset."""
        env = ArcSingleAgentEnv()
        key = jax.random.PRNGKey(42)
        
        state, timestep = env.reset(key)
        
        # Check state validity
        assert state.step_count == 0
        assert not state.is_done
        assert state.working_grid.shape == (32, 32)
        
        # Check timestep validity
        assert timestep.observation is not None
        assert 'working_grid' in timestep.observation
        assert timestep.step_type.name == 'FIRST'
    
    def test_step_functionality(self):
        """Test environment step."""
        env = ArcSingleAgentEnv()
        key = jax.random.PRNGKey(42)
        
        state, timestep = env.reset(key)
        
        # Create a simple action
        action = {
            'action_type': jnp.array(0, dtype=jnp.int32),  # Paint action
            'position': jnp.array([5, 5], dtype=jnp.int32),
            'color': jnp.array(1, dtype=jnp.int32),
            'region_size': jnp.array([1, 1], dtype=jnp.int32),
            'transform_type': jnp.array(0, dtype=jnp.int32),
            'attention_position': jnp.array([5, 5], dtype=jnp.int32),
            'attention_radius': jnp.array(2, dtype=jnp.int32),
        }
        
        new_state, new_timestep = env.step(state, action)
        
        # Check state updates
        assert new_state.step_count == 1
        assert new_state.working_grid[5, 5] == 1  # Pixel should be painted
        
        # Check timestep
        assert new_timestep.reward is not None
        assert new_timestep.observation is not None
    
    def test_jax_transformations(self):
        """Test that environment works with JAX transformations."""
        env = ArcSingleAgentEnv()
        
        # Test JIT compilation
        @jax.jit
        def reset_step(key):
            state, timestep = env.reset(key)
            action = {
                'action_type': jnp.array(0, dtype=jnp.int32),
                'position': jnp.array([0, 0], dtype=jnp.int32),
                'color': jnp.array(1, dtype=jnp.int32),
                'region_size': jnp.array([1, 1], dtype=jnp.int32),
                'transform_type': jnp.array(0, dtype=jnp.int32),
                'attention_position': jnp.array([0, 0], dtype=jnp.int32),
                'attention_radius': jnp.array(1, dtype=jnp.int32),
            }
            new_state, new_timestep = env.step(state, action)
            return new_state, new_timestep
        
        key = jax.random.PRNGKey(42)
        state, timestep = reset_step(key)
        
        assert state.step_count == 1
        assert timestep.observation is not None
    
    def test_vectorization(self):
        """Test environment vectorization."""
        env = ArcSingleAgentEnv()
        
        # Test vmap over multiple episodes
        @jax.vmap
        def vectorized_reset(keys):
            return env.reset(keys)
        
        keys = jax.random.split(jax.random.PRNGKey(42), 4)
        states, timesteps = vectorized_reset(keys)
        
        assert states.step_count.shape == (4,)
        assert states.working_grid.shape == (4, 32, 32)


class TestEnvironmentFactory:
    """Test suite for environment factory."""
    
    def test_single_agent_creation(self):
        """Test single-agent environment creation via factory."""
        # This test requires actual ARC data, so we'll mock it
        # In practice, you'd use real data paths
        pass  # TODO: Implement with mock data
    
    def test_multi_agent_creation(self):
        """Test multi-agent environment creation via factory."""
        # TODO: Implement when multi-agent env is complete
        pass


if __name__ == "__main__":
    pytest.main([__file__])
```

### 2. Integration Tests

**`tests/test_integration.py`:**

```python
"""Integration tests for complete ARC pipeline."""

import pytest
import jax
import jax.numpy as jnp
from jaxarc.envs.arc_single_agent import ArcSingleAgentEnv
from jaxarc.parsers.arc_agi_parser import ArcAgiParser


class TestIntegration:
    """Integration tests for parser + environment."""
    
    @pytest.mark.skipif(
        True,  # Skip by default since it requires data
        reason="Requires ARC-AGI dataset"
    )
    def test_parser_environment_integration(self):
        """Test that parser output works with environment."""
        # Create parser
        parser = ArcAgiParser("data/arc-agi")
        
        # Load a task
        task_data = parser.get_random_task()
        
        # Create environment
        env = ArcSingleAgentEnv()
        
        # Test that environment can handle real task data
        # This would require modifying environment to accept task_data
        # in reset method
        pass
    
    def test_performance_benchmark(self):
        """Basic performance benchmark."""
        env = ArcSingleAgentEnv()
        key = jax.random.PRNGKey(42)
        
        # Compile functions
        @jax.jit
        def run_episode(key):
            state, timestep = env.reset(key)
            
            def step_fn(carry, _):
                state, timestep = carry
                action = {
                    'action_type': jnp.array(0, dtype=jnp.int32),
                    'position': jnp.array([0, 0], dtype=jnp.int32),
                    'color': jnp.array(1, dtype=jnp.int32),
                    'region_size': jnp.array([1, 1], dtype=jnp.int32),
                    'transform_type': jnp.array(0, dtype=jnp.int32),
                    'attention_position': jnp.array([0, 0], dtype=jnp.int32),
                    'attention_radius': jnp.array(1, dtype=jnp.int32),
                }
                new_state, new_timestep = env.step(state, action)
                return (new_state, new_timestep), new_timestep.reward
            
            (final_state, final_timestep), rewards = jax.lax.scan(
                step_fn, (state, timestep), None, length=10
            )
            return final_state, rewards
        
        # Warm up
        _ = run_episode(key)
        
        # Time execution
        import time
        start_time = time.time()
        
        # Run multiple episodes
        keys = jax.random.split(key, 100)
        results = jax.vmap(run_episode)(keys)
        
        end_time = time.time()
        
        print(f"100 episodes of 10 steps each: {end_time - start_time:.3f}s")
        print(f"Steps per second: {1000 / (end_time - start_time):.0f}")


if __name__ == "__main__":
    pytest.main([__file__])
```

## Performance Optimization

### 1. JAX Optimization Patterns

**`src/jaxarc/envs/optimizations.py`:**

```python
"""Performance optimizations for ARC environments."""

import jax
import jax.numpy as jnp
import chex
from functools import partial


# Efficient grid operations
@jax.jit
def efficient_grid_update(grid: jnp.ndarray, position: jnp.ndarray, color: jnp.ndarray) -> jnp.ndarray:
    """Optimized grid update operation."""
    row, col = position[0], position[1]
    return grid.at[row, col].set(color)


@jax.jit  
def batch_grid_updates(grid: jnp.ndarray, positions: jnp.ndarray, colors: jnp.ndarray) -> jnp.ndarray:
    """Batch multiple grid updates efficiently."""
    def update_fn(g, pos_color):
        pos, color = pos_color
        return efficient_grid_update(g, pos, color)
    
    return jax.lax.foldl(update_fn, grid, (positions, colors))


# Memory-efficient attention computation
@jax.jit
def compute_attention_map(
    current_attention: jnp.ndarray,
    focus_position: jnp.ndarray,
    radius: jnp.ndarray,
    decay_rate: float = 0.1
) -> jnp.ndarray:
    """Compute attention map with memory efficiency."""
    h, w = current_attention.shape
    y, x = jnp.ogrid[:h, :w]
    
    # Compute distance from focus point
    focus_y, focus_x = focus_position[0], focus_position[1]
    distance = jnp.sqrt((y - focus_y)**2 + (x - focus_x)**2)
    
    # Compute attention update
    attention_update = jnp.exp(-distance / jnp.maximum(radius, 1.0))
    
    # Exponential moving average
    new_attention = decay_rate * attention_update + (1 - decay_rate) * current_attention
    
    return new_attention


# Vectorized reward computation
@jax.vmap
def vectorized_pixel_accuracy(predicted: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Vectorized pixel accuracy computation."""
    correct = jnp.sum((predicted == target) & mask)
    total = jnp.sum(mask)
    return correct / jnp.maximum(total, 1.0)


# Efficient state transitions
@partial(jax.jit, static_argnums=(0,))
def efficient_phase_transition(phase_lengths: dict, current_phase: jnp.ndarray, timer: jnp.ndarray):
    """Efficiently handle phase transitions."""
    should_transition = timer <= 0
    next_phase = (current_phase + 1) % 4
    
    # Use static_argnums to make phase_lengths compile-time constant
    next_timer_default = 10  # Default fallback
    next_timer = jnp.where(
        should_transition,
        next_timer_default,  # In practice, lookup from phase_lengths
        timer
    )
    
    return jnp.where(should_transition, next_phase, current_phase), next_timer


# Memory-efficient hypothesis encoding
@jax.jit
def encode_hypothesis(grid_state: jnp.ndarray, attention_map: jnp.ndarray) -> jnp.ndarray:
    """Encode grid state and attention into compact hypothesis representation."""
    # Simple encoding: concatenate downsampled versions
    grid_features = jax.image.resize(
        grid_state[..., None], 
        (8, 8, 1), 
        method='nearest'
    ).reshape(-1)
    
    attention_features = jax.image.resize(
        attention_map[..., None], 
        (4, 4, 1), 
        method='linear'
    ).reshape(-1)
    
    # Combine features
    hypothesis = jnp.concatenate([
        grid_features.astype(jnp.float32),
        attention_features
    ])
    
    # Pad or truncate to fixed size
    target_size = 64
    if len(hypothesis) < target_size:
        hypothesis = jnp.pad(hypothesis, (0, target_size - len(hypothesis)))
    else:
        hypothesis = hypothesis[:target_size]
    
    return hypothesis


# Batch processing utilities
def create_batched_env_step(env_step_fn):
    """Create batched version of environment step function."""
    @jax.vmap
    def batched_step(states, actions):
        return env_step_fn(states, actions)
    
    return batched_step


def create_parallel_env_reset(env_reset_fn):
    """Create parallel version of environment reset function."""
    @jax.vmap  
    def parallel_reset(keys):
        return env_reset_fn(keys)
    
    return parallel_reset
```

### 2. Benchmarking Suite

**`scripts/benchmark_performance.py`:**

```python
#!/usr/bin/env python3
"""Performance benchmarking for ARC environments."""

import jax
import jax.numpy as jnp
import time
from typing import Dict, Any
import matplotlib.pyplot as plt

from jaxarc.envs.arc_single_agent import ArcSingleAgentEnv


def benchmark_single_episode(env: ArcSingleAgentEnv, num_steps: int = 50) -> Dict[str, float]:
    """Benchmark a single episode performance."""
    
    @jax.jit
    def run_episode(key):
        state, timestep = env.reset(key)
        
        def step_fn(carry, _):
            state, timestep = carry
            # Random action
            action = {
                'action_type': jnp.array(0, dtype=jnp.int32),
                'position': jnp.array([0, 0], dtype=jnp.int32),
                'color': jnp.array(1, dtype=jnp.int32),
                'region_size': jnp.array([1, 1], dtype=jnp.int32),
                'transform_type': jnp.array(0, dtype=jnp.int32),
                'attention_position': jnp.array([0, 0], dtype=jnp.int32),
                'attention_radius': jnp.array(1, dtype=jnp.int32),