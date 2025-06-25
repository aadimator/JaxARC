"""
ARC Environment - JAX-compatible implementation using proper base classes.

This module implements an ARC (Abstraction and Reasoning Corpus) environment
using JAX and integrates with the JaxMARL framework through the ArcMarlEnvBase.

Key Features:
- Single-agent ARC implementation (extensible to multi-agent)
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
from ..spaces.multibinary import MultiBinary
from ..types import ParsedTaskData
from .grid_operations import compute_grid_similarity, execute_grid_operation


@chex.dataclass(kw_only=True)
class ArcEnvironmentState(ArcEnvState):
    """
    State representation for ARC environment.

    This includes all base environment state fields (equivalent to ArcEnvState)
    plus ARC-specific additions for grid manipulation, clipboard operations,
    and episode tracking.

    All arrays use maximum dimensions from config with actual dimensions
    tracked separately for dynamic grid size support.

    Attributes:
        # Base environment state fields (from ArcEnvState)
        done: Boolean indicating if environment is done
        step: Current step number in environment
        task_data: Parsed ARC task containing training/test examples
        active_train_pair_idx: Index of current training pair being worked on
        working_grid: Grid being modified by agents
        working_grid_mask: Mask indicating valid cells in working_grid
        program: Sequence of actions taken so far
        program_length: Current length of the program
        active_agents: Mask indicating which agents are active
        cumulative_rewards: Cumulative rewards for each agent

        # ARCLE-specific fields
        grid: Current working grid being modified by agent (same as working_grid)
        input_grid: Original input grid (immutable reference)
        target_grid: Target output grid for comparison
        selected: Current selection mask (for visualization)
        clipboard: Clipboard data for copy/paste operations
        grid_dim: Actual grid dimensions [height, width]
        target_dim: Target grid dimensions [height, width]
        max_grid_dim: Maximum grid dimensions from config [height, width]
        similarity_score: Current similarity to target grid [0.0, 1.0]
    """

    # ARC-specific fields
    selected: jnp.ndarray  # (max_grid_h, max_grid_w) bool - selection mask
    clipboard: jnp.ndarray  # (max_grid_h, max_grid_w) int32 - clipboard data

    # Grid metadata
    grid_dim: jnp.ndarray  # (2,) int32 - actual [h, w]
    target_dim: jnp.ndarray  # (2,) int32 - target [h, w]
    max_grid_dim: jnp.ndarray  # (2,) int32 - max [h, w] from config

    # Episode state
    similarity_score: jnp.ndarray  # float32 scalar

    @property
    def grid(self) -> chex.Array:
        """Alias for working_grid for script compatibility."""
        return self.working_grid

    @property
    def terminated(self) -> chex.Array:
        """Alias for done for script compatibility."""
        return self.done

    @property
    def input_grid(self) -> chex.Array:
        """Get input grid from task data."""
        return self.task_data.input_grids_examples[self.active_train_pair_idx]

    @property
    def target_grid(self) -> chex.Array:
        """Get target grid from task data."""
        return self.task_data.output_grids_examples[self.active_train_pair_idx]

    def __post_init__(self) -> None:
        """Validate ARCLE state structure and consistency."""
        # Skip validation during JAX transformations
        if not (hasattr(self.done, "ndim") and hasattr(self.done, "shape")):
            return

        try:
            # Validate base fields
            chex.assert_type(self.done, jnp.bool_)
            chex.assert_shape(self.done, ())
            chex.assert_type(self.active_train_pair_idx, jnp.int32)
            chex.assert_shape(self.active_train_pair_idx, ())

            # Validate base grid state
            chex.assert_rank(self.working_grid, 2)
            chex.assert_rank(self.working_grid_mask, 2)
            chex.assert_type(self.working_grid, jnp.integer)
            chex.assert_type(self.working_grid_mask, jnp.bool_)
            chex.assert_shape(self.working_grid_mask, self.working_grid.shape)

            # Validate program state
            chex.assert_rank(self.program, 2)
            chex.assert_type(self.program_length, jnp.int32)
            chex.assert_shape(self.program_length, ())

            # Validate agent arrays
            chex.assert_rank(self.active_agents, 1)
            chex.assert_rank(self.cumulative_rewards, 1)
            chex.assert_type(self.active_agents, jnp.bool_)
            chex.assert_type(self.cumulative_rewards, jnp.float32)
            num_agents = self.active_agents.shape[0]
            chex.assert_shape(self.cumulative_rewards, (num_agents,))

            # Validate ARC-specific grid arrays
            chex.assert_rank(self.selected, 2)
            chex.assert_rank(self.clipboard, 2)

            chex.assert_type(self.selected, jnp.bool_)
            chex.assert_type(self.clipboard, jnp.integer)

            # Validate grid arrays have consistent shapes with working_grid
            grid_shape = self.working_grid.shape
            chex.assert_shape(self.selected, grid_shape)
            chex.assert_shape(self.clipboard, grid_shape)

            # Validate metadata arrays
            chex.assert_rank(self.grid_dim, 1)
            chex.assert_rank(self.target_dim, 1)
            chex.assert_rank(self.max_grid_dim, 1)
            chex.assert_shape(self.grid_dim, (2,))
            chex.assert_shape(self.target_dim, (2,))
            chex.assert_shape(self.max_grid_dim, (2,))

            chex.assert_type(self.grid_dim, jnp.int32)
            chex.assert_type(self.target_dim, jnp.int32)
            chex.assert_type(self.max_grid_dim, jnp.int32)

            # Validate episode state scalars
            chex.assert_type(self.similarity_score, jnp.float32)
            chex.assert_shape(self.similarity_score, ())

        except (AttributeError, TypeError):
            # Skip validation during JAX transformations
            pass


class ArcEnvironment(ArcMarlEnvBase):
    """
    Single-agent ARC environment with JaxMARL compatibility.

    This environment implements an ARC design where an agent selects a region
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
        Initialize the ARC environment.

        Args:
            config: Configuration dictionary with environment parameters
            num_agents: Number of agents (defaults to 1 for single-agent ARC)
            max_grid_size: Maximum grid dimensions (height, width)
            max_episode_steps: Maximum steps per episode
            **kwargs: Additional arguments for parent class
        """
        # Load configuration
        self.config = config or {}

        # Override parameters from config if provided
        if "max_grid_size" in self.config:
            max_grid_size = tuple(self.config["max_grid_size"])
        if "max_episode_steps" in self.config:
            max_episode_steps = self.config["max_episode_steps"]

        # ARC-specific configuration
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

            warnings.warn(
                f"ARCLE is designed for single-agent use, got {num_agents} agents",
                UserWarning,
            )

    def _setup_spaces(self) -> None:
        """Set up ARCLE-specific action and observation spaces."""
        h, w = self.max_grid_size

        # ARCLE action space: selection mask + operation ID
        action_space = Dict(
            {
                "selection": MultiBinary(h * w),
                "operation": Discrete(35),  # ARCLE operations 0-34
            }
        )

        # Observation space: flattened grids + metadata
        grid_size = h * w
        obs_size = (
            grid_size * 4  # grid, input_grid, target_grid, clipboard
            + 10  # metadata (step, similarity, etc.)
        )
        obs_space = Box(0.0, 10.0, shape=(obs_size,), dtype=jnp.float32)

        self.action_spaces = dict.fromkeys(self.agents, action_space)
        self.observation_spaces = dict.fromkeys(self.agents, obs_space)

    def reset(
        self, key: chex.PRNGKey, task_data: ParsedTaskData | None = None
    ) -> tuple[dict[str, chex.Array], ArcEnvironmentState]:
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
        # Cast state to ArcEnvironmentState for ARC-specific operations
        arc_state = state  # type: ArcEnvironmentState

        # Get agent action (single agent for now)
        agent_id = self.agents[0]
        action = actions[agent_id]

        # Convert continuous selection to boolean mask
        # Handle both dict and array-based action formats
        if isinstance(action, dict):
            selection = action["selection"]
            operation_id = action["operation"]
        else:
            # Fallback: assume action is a single array/tensor
            # For now, create dummy values - this should be updated based on actual action format
            h, w = self.max_grid_size
            selection = jnp.zeros((h * w,), dtype=jnp.int32)  # Default to no selection
            operation_id = jnp.array(0, dtype=jnp.int32)  # Default to fill color 0

        selection_mask = selection.reshape(self.max_grid_size).astype(jnp.bool_)

        # Execute the ARC operation
        new_state = self._execute_action(arc_state, selection_mask, operation_id)

        # Calculate reward
        reward = self._calculate_reward(arc_state, new_state, operation_id)

        # Check if episode is done
        done = self._is_episode_done(new_state)

        # Update step counter and done status
        from dataclasses import replace

        new_state = replace(
            new_state,
            done=jnp.array(done),
            step=state.step + 1,
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
            {agent_id: info},
        )

    def get_obs(self, state: ArcEnvState) -> dict[str, chex.Array]:
        """
        Generate observations for all agents from the current state.

        Args:
            state: Current environment state

        Returns:
            Dictionary mapping agent IDs to observations
        """
        # Cast to ArcEnvironmentState to access ARC-specific fields
        arc_state = state  # type: ArcEnvironmentState
        max_h, max_w = self.max_grid_size

        # Pad grids to max dimensions and flatten
        working_grid_padded = self._pad_to_max_dims(arc_state.working_grid)
        grid_flat = working_grid_padded.flatten().astype(jnp.float32)

        input_grid = arc_state.task_data.input_grids_examples[
            arc_state.active_train_pair_idx
        ]
        input_grid_padded = self._pad_to_max_dims(input_grid)
        input_grid_flat = input_grid_padded.flatten().astype(jnp.float32)

        target_grid = arc_state.task_data.output_grids_examples[
            arc_state.active_train_pair_idx
        ]
        target_grid_padded = self._pad_to_max_dims(target_grid)
        target_grid_flat = target_grid_padded.flatten().astype(jnp.float32)

        clipboard_padded = self._pad_to_max_dims(arc_state.clipboard)
        clipboard_flat = clipboard_padded.flatten().astype(jnp.float32)

        # Metadata - convert all values to float32 for JAX compatibility
        metadata = jnp.array(
            [
                jnp.array(arc_state.step, dtype=jnp.float32),  # step is Python int
                arc_state.similarity_score.astype(jnp.float32),
                arc_state.active_train_pair_idx.astype(jnp.float32),
                arc_state.done.astype(jnp.float32),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,  # padding for future use
            ],
            dtype=jnp.float32,
        )

        # Concatenate all observation components
        obs = jnp.concatenate(
            [grid_flat, input_grid_flat, target_grid_flat, clipboard_flat, metadata]
        )

        # Create observation dict for all agents
        return dict.fromkeys(self.agents, obs)

    def _pad_to_max_dims(self, grid: chex.Array) -> chex.Array:
        """
        Pad a grid to max dimensions.

        Args:
            grid: Input grid to pad

        Returns:
            Grid padded to max dimensions
        """
        max_h, max_w = self.max_grid_size
        current_h, current_w = grid.shape

        # If already max size, return as is
        if current_h == max_h and current_w == max_w:
            return grid

        # Pad with zeros to max dimensions
        pad_h = max_h - current_h
        pad_w = max_w - current_w

        return jnp.pad(grid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    def _initialize_state(
        self, key: chex.PRNGKey, task_data: ParsedTaskData
    ) -> ArcEnvironmentState:
        """Initialize the ARC environment state from task data."""
        max_h, max_w = self.max_grid_size

        # Get first training pair as initial state
        input_grid = task_data.input_grids_examples[0]
        target_grid = task_data.output_grids_examples[0]

        # Use actual grid dimensions, not max dimensions
        actual_shape = input_grid.shape
        if len(actual_shape) == 2:
            h, w = actual_shape
        else:
            h, w = 0, 0

        # Calculate initial similarity between input and target
        initial_similarity = compute_grid_similarity(input_grid, target_grid)

        # Initialize grid dimensions (use full grid size for simplicity)
        grid_dim = jnp.array([h, w], dtype=jnp.int32)
        target_dim = jnp.array([h, w], dtype=jnp.int32)
        max_grid_dim = jnp.array([max_h, max_w], dtype=jnp.int32)

        # Initialize base state fields
        empty_program = jnp.zeros(
            (self.max_program_length, self.max_action_params), dtype=jnp.int32
        )
        active_agents = jnp.ones(self.num_agents, dtype=jnp.bool_)
        cumulative_rewards = jnp.zeros(self.num_agents, dtype=jnp.float32)

        return ArcEnvironmentState(
            # Base state fields (ArcEnvState)
            done=jnp.array(False, dtype=jnp.bool_),
            step=0,
            task_data=task_data,
            active_train_pair_idx=jnp.array(0, dtype=jnp.int32),
            working_grid=self._pad_to_max_dims(input_grid),  # Pad to max dimensions
            working_grid_mask=self._pad_to_max_dims(task_data.input_masks_examples[0]),
            program=empty_program,
            program_length=jnp.array(0, dtype=jnp.int32),
            active_agents=active_agents,
            cumulative_rewards=cumulative_rewards,
            # ARC-specific fields
            selected=jnp.zeros((max_h, max_w), dtype=jnp.bool_),
            clipboard=jnp.zeros((max_h, max_w), dtype=jnp.int32),
            # Grid metadata
            grid_dim=grid_dim,
            target_dim=target_dim,
            max_grid_dim=max_grid_dim,
            # Episode state
            similarity_score=jnp.array(initial_similarity, dtype=jnp.float32),
        )

    def _execute_action(
        self, state: ArcEnvironmentState, selection_mask: chex.Array, operation: chex.Array
    ) -> ArcEnvironmentState:
        """Execute an ARC action (selection + operation) on the current state."""
        from dataclasses import replace

        # Create state with updated selection mask
        state_with_selection = replace(state, selected=selection_mask)

        # Execute the operation using the grid operations module
        new_state = execute_grid_operation(state_with_selection, operation)

        return new_state

    def _calculate_reward(
        self, old_state: ArcEnvironmentState, new_state: ArcEnvironmentState, operation: chex.Array
    ) -> float:
        """Calculate reward for the current step."""
        # Base reward is 0
        reward = 0.0

        if self.reward_on_submit_only:
            # Only give reward on submit operation (operation 34)
            is_submit = operation == 34
            submit_reward = jax.lax.cond(
                new_state.similarity_score >= self.similarity_threshold,
                lambda: self.success_bonus,
                lambda: new_state.similarity_score - old_state.similarity_score,
            )
            reward = jax.lax.cond(
                is_submit, lambda: submit_reward, lambda: -self.step_penalty
            )
        else:
            # Give reward based on similarity improvement
            similarity_improvement = (
                new_state.similarity_score - old_state.similarity_score
            )
            reward = similarity_improvement - self.step_penalty

            # Bonus for reaching threshold
            reward = jax.lax.cond(
                new_state.similarity_score >= self.similarity_threshold,
                lambda: reward + self.success_bonus,
                lambda: reward,
            )

        return reward

    def _is_episode_done(self, state: ArcEnvironmentState) -> bool:
        """Check if the episode should terminate."""
        # Terminate if max steps reached
        max_steps_reached = state.step >= self.max_episode_steps

        # Terminate if similarity threshold reached (task solved)
        task_solved = state.similarity_score >= self.similarity_threshold

        # Terminate if explicitly terminated
        explicitly_terminated = state.done

        return max_steps_reached | task_solved | explicitly_terminated

    def reset_with_task(
        self, key: chex.PRNGKey, task_data: ParsedTaskData
    ) -> tuple[dict[str, chex.Array], ArcEnvState]:
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
        return "ARC"

    @property
    def agent_classes(self) -> dict[str, str]:
        """Agent class mapping."""
        return dict.fromkeys(self.agents, "arc_agent")
