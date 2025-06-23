"""
ARCLE Environment - JAX-compatible implementation using proper base classes.

This module implements the ARCLE (Abstraction and Reasoning Challenge Learning Environment)
using JAX and integrates with the JaxMARL framework through the ArcMarlEnvBase.

Key Features:
- Single-agent ARCLE implementation (extensible to multi-agent)
- Selection mask + operation ID action space
- Task loading from ParsedTaskData
- JAX-compatible operations and state updates
- JIT-compiled for high performance
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.spaces import Box, Dict, Discrete

from ..base.base_env import ArcEnvState, ArcMarlEnvBase
from ..types import ParsedTaskData, ARCLEState
from .arcle_operations import execute_arcle_operation, compute_grid_similarity


class ARCLEEnvironment(ArcMarlEnvBase):
    """
    Single-agent ARCLE environment with JaxMARL compatibility.

    This environment implements the ARCLE design where an agent selects a region
    of the grid and applies an operation to it. The goal is to transform the
    input grid to match the target output grid.
    """

    def __init__(
        self,
        config: dict | None = None,
        num_agents: int = 1,
        max_grid_size: tuple[int, int] = (30, 30),
        max_episode_steps: int = 100,
        **kwargs,
    ):
        """
        Initialize the ARCLE environment.

        Args:
            config: Configuration dictionary with environment parameters
            num_agents: Number of agents (defaults to 1 for single-agent ARCLE)
            max_grid_size: Maximum grid dimensions (height, width)
            max_episode_steps: Maximum steps per episode
            **kwargs: Additional arguments for parent class
        """
        # Load configuration
        self.config = config or {}

        # ARCLE-specific configuration
        reward_config = self.config.get("reward", {})
        self.reward_on_submit_only = reward_config.get("reward_on_submit_only", True)
        self.similarity_threshold = reward_config.get("similarity_threshold", 0.95)
        self.success_bonus = reward_config.get("success_bonus", 1.0)
        self.step_penalty = reward_config.get("step_penalty", 0.0)

        # Initialize parent
        super().__init__(
            num_agents=num_agents,
            max_grid_size=max_grid_size,
            max_episode_steps=max_episode_steps,
            config=config,
            **kwargs,
        )

        # ARCLE is typically single-agent
        if num_agents != 1:
            import warnings
            warnings.warn(f"ARCLE is designed for single-agent use, got {num_agents} agents", UserWarning)

    def _setup_spaces(self) -> None:
        """Set up ARCLE-specific action and observation spaces."""
        h, w = self.max_grid_size

        # ARCLE action space: selection mask + operation ID
        action_space = Dict({
            'selection': Box(0.0, 1.0, (h, w), dtype=jnp.float32),  # Continuous selection mask
            'operation': Discrete(35)  # ARCLE operations 0-34
        })

        # Observation space: flattened grids + metadata
        grid_size = h * w
        obs_size = (
            grid_size * 4 +  # grid, input_grid, target_grid, clipboard
            10  # metadata (step, similarity, etc.)
        )
        obs_space = Box(0.0, 10.0, (obs_size,), dtype=jnp.float32)

        self.action_spaces = dict.fromkeys(self.agents, action_space)
        self.observation_spaces = dict.fromkeys(self.agents, obs_space)

    def reset(self, key: chex.PRNGKey, task_data: ParsedTaskData | None = None) -> tuple[dict[str, chex.Array], ArcEnvState]:
        """
        Reset the environment with a new ARC task.

        Args:
            key: JAX random key for initialization
            task_data: Optional parsed task data. If None, creates dummy data.

        Returns:
            Tuple of (observations, initial_state)
        """
        # Create or use provided task data
        if task_data is None:
            key, task_key = jax.random.split(key)
            task_data = self._create_dummy_task_data(task_key)

        # Initialize state
        state = self._initialize_state(key, task_data)

        # Get initial observations
        observations = self.get_obs(state)

        return observations, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: ArcEnvState,
        actions: dict[str, chex.Array],
    ) -> tuple[
        dict[str, chex.Array],
        ArcEnvState,
        dict[str, float],
        dict[str, bool],
        dict[str, Any],
    ]:
        """
        Execute one environment step.

        Args:
            key: JAX random key
            state: Current environment state
            actions: Actions from all agents

        Returns:
            Tuple of (observations, next_state, rewards, dones, info)
        """
        # Cast state to ARCLEState for ARCLE-specific operations
        arcle_state = state  # type: ARCLEState

        # Get agent action (single agent for now)
        agent_id = self.agents[0]
        action = actions[agent_id]

        # Convert continuous selection to boolean mask
        # Handle both dict and array-based action formats
        if isinstance(action, dict):
            selection_probs = action['selection']
            operation_id = action['operation']
        else:
            # Fallback: assume action is a single array/tensor
            # For now, create dummy values - this should be updated based on actual action format
            h, w = self.max_grid_size
            selection_probs = jnp.ones((h, w)) * 0.1  # Low selection probability
            operation_id = jnp.array(0, dtype=jnp.int32)  # Default to fill color 0

        selection_mask = selection_probs > 0.5  # Threshold for binary selection

        # Execute the ARCLE operation
        new_state = self._execute_action(arcle_state, selection_mask, operation_id)

        # Calculate reward
        reward = self._calculate_reward(arcle_state, new_state, operation_id)

        # Check if episode is done
        done = self._is_episode_done(new_state)

        # Update step counter and done status
        from dataclasses import replace
        new_state = replace(
            new_state,
            done=jnp.array(done),
            step=state.step + 1,
            terminated=jnp.array(done)
        )

        # Get observations
        observations = self.get_obs(new_state)

        # Create empty info dict for JAX compatibility
        # (State information can be accessed directly from new_state)
        info = {}

        return (
            observations,
            new_state,
            {agent_id: reward},
            {agent_id: done, "__all__": done},
            {agent_id: info}
        )

    def get_obs(self, state: ArcEnvState) -> dict[str, chex.Array]:
        """
        Generate observations for all agents from the current state.

        Args:
            state: Current environment state

        Returns:
            Dictionary mapping agent IDs to observations
        """
        # Cast to ARCLEState to access ARCLE-specific fields
        arcle_state = state  # type: ARCLEState
        h, w = self.max_grid_size

        # Flatten grid components
        grid_flat = arcle_state.grid.flatten().astype(jnp.float32)
        input_grid_flat = arcle_state.input_grid.flatten().astype(jnp.float32)
        target_grid_flat = arcle_state.target_grid.flatten().astype(jnp.float32)
        clipboard_flat = arcle_state.clipboard.flatten().astype(jnp.float32)

        # Metadata - convert all values to float32 for JAX compatibility
        metadata = jnp.array([
            jnp.array(arcle_state.step, dtype=jnp.float32),  # step is Python int
            arcle_state.similarity_score.astype(jnp.float32),
            arcle_state.step_count.astype(jnp.float32),
            arcle_state.active_train_pair_idx.astype(jnp.float32),
            arcle_state.terminated.astype(jnp.float32),
            0.0, 0.0, 0.0, 0.0, 0.0  # padding for future use
        ], dtype=jnp.float32)

        # Concatenate all observation components
        obs = jnp.concatenate([
            grid_flat,
            input_grid_flat,
            target_grid_flat,
            clipboard_flat,
            metadata
        ])

        # Create observation dict for all agents
        return dict.fromkeys(self.agents, obs)

    def _initialize_state(self, key: chex.PRNGKey, task_data: ParsedTaskData) -> ARCLEState:
        """Initialize the ARCLE environment state from task data."""
        h, w = self.max_grid_size

        # Get first training pair as initial state
        input_grid = task_data.input_grids_examples[0]
        target_grid = task_data.output_grids_examples[0]

        # Calculate initial similarity
        initial_similarity = compute_grid_similarity(input_grid, target_grid)

        # Initialize grid dimensions (use full grid size for simplicity)
        grid_dim = jnp.array([h, w], dtype=jnp.int32)
        target_dim = jnp.array([h, w], dtype=jnp.int32)
        max_grid_dim = jnp.array([h, w], dtype=jnp.int32)

        # Initialize base state fields
        empty_program = jnp.zeros((self.max_program_length, self.max_action_params), dtype=jnp.int32)
        active_agents = jnp.ones(self.num_agents, dtype=jnp.bool_)
        cumulative_rewards = jnp.zeros(self.num_agents, dtype=jnp.float32)

        return ARCLEState(
            # Base state fields (ArcEnvState)
            done=jnp.array(False, dtype=jnp.bool_),
            step=0,
            task_data=task_data,
            active_train_pair_idx=jnp.array(0, dtype=jnp.int32),
            working_grid=input_grid,
            working_grid_mask=task_data.input_masks_examples[0],
            program=empty_program,
            program_length=jnp.array(0, dtype=jnp.int32),
            active_agents=active_agents,
            cumulative_rewards=cumulative_rewards,

            # ARCLE-specific fields
            grid=input_grid,  # Start with input grid
            input_grid=input_grid,
            target_grid=target_grid,
            selected=jnp.zeros((h, w), dtype=jnp.bool_),
            clipboard=jnp.zeros((h, w), dtype=jnp.int32),

            # Grid metadata
            grid_dim=grid_dim,
            target_dim=target_dim,
            max_grid_dim=max_grid_dim,

            # Episode state
            step_count=jnp.array(0, dtype=jnp.int32),
            terminated=jnp.array(False, dtype=jnp.bool_),
            similarity_score=jnp.array(initial_similarity, dtype=jnp.float32),
        )

    def _execute_action(self, state: ARCLEState, selection_mask: chex.Array, operation: chex.Array) -> ARCLEState:
        """Execute an ARCLE action (selection + operation) on the current state."""
        from dataclasses import replace

        # Create state with updated selection mask
        state_with_selection = replace(state, selected=selection_mask)

        # Execute the operation using the ARCLE operations module
        new_state = execute_arcle_operation(state_with_selection, operation)

        # Update step count and working grid (sync with grid)
        new_step_count = state.step_count + 1

        # Update similarity score
        similarity = compute_grid_similarity(new_state.grid, new_state.target_grid)

        # Update state with new values
        final_state = replace(
            new_state,
            working_grid=new_state.grid,  # Sync working grid
            step_count=new_step_count,
            similarity_score=similarity
        )

        return final_state

    def _calculate_reward(self, old_state: ARCLEState, new_state: ARCLEState, operation: chex.Array) -> float:
        """Calculate reward for the current step."""
        # Base reward is 0
        reward = 0.0

        if self.reward_on_submit_only:
            # Only give reward on submit operation (operation 34)
            is_submit = operation == 34
            submit_reward = jax.lax.cond(
                new_state.similarity_score >= self.similarity_threshold,
                lambda: self.success_bonus,
                lambda: new_state.similarity_score - old_state.similarity_score
            )
            reward = jax.lax.cond(
                is_submit,
                lambda: submit_reward,
                lambda: -self.step_penalty
            )
        else:
            # Give reward based on similarity improvement
            similarity_improvement = new_state.similarity_score - old_state.similarity_score
            reward = similarity_improvement - self.step_penalty

            # Bonus for reaching threshold
            reward = jax.lax.cond(
                new_state.similarity_score >= self.similarity_threshold,
                lambda: reward + self.success_bonus,
                lambda: reward
            )

        return reward

    def _is_episode_done(self, state: ARCLEState) -> bool:
        """Check if the episode should terminate."""
        # Terminate if max steps reached
        max_steps_reached = state.step >= self.max_episode_steps

        # Terminate if similarity threshold reached (task solved)
        task_solved = state.similarity_score >= self.similarity_threshold

        # Terminate if explicitly terminated
        explicitly_terminated = state.terminated

        return max_steps_reached | task_solved | explicitly_terminated

    def reset_with_task(self, key: chex.PRNGKey, task_data: ParsedTaskData) -> tuple[dict[str, chex.Array], ArcEnvState]:
        """
        Reset the environment with specific task data.

        Args:
            key: JAX random key
            task_data: Task data to use for reset

        Returns:
            Tuple of (observations, initial_state)
        """
        return self.reset(key, task_data)

    @property
    def name(self) -> str:
        """Environment name."""
        return "ARCLE"

    @property
    def agent_classes(self) -> dict[str, str]:
        """Agent class mapping."""
        return dict.fromkeys(self.agents, "arcle_agent")
