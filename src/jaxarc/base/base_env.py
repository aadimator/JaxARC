"""
Abstract base class for ARC Multi-Agent Reinforcement Learning environments.

This module provides the foundational components for creating JAX-compatible
ARC environments that integrate with the JaxMARL framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box

from ..types import ParsedTaskData


@chex.dataclass(kw_only=True)
class ArcEnvState:
    """
    Base state dataclass for ARC Multi-Agent environments.

    This provides the minimal state structure that all ARC environments should have.

    Attributes:
        # JaxMARL required fields
        done: Boolean array indicating if environment is done
        step: Current step number in the environment

        # ARC task state
        task_data: Parsed ARC task containing training/test examples
        active_train_pair_idx: Index of current training pair being worked on

        # Grid state
        working_grid: Grid being modified by agents
        working_grid_mask: Mask indicating valid cells in working_grid

        # Program state
        program: Sequence of actions taken so far
        program_length: Current length of the program

        # Agent state
        active_agents: Mask indicating which agents are active
        cumulative_rewards: Cumulative rewards for each agent
    """

    # JaxMARL required fields
    done: chex.Array
    step: int

    # ARC task state
    task_data: ParsedTaskData
    active_train_pair_idx: jnp.ndarray  # int32 scalar

    # Grid state
    working_grid: jnp.ndarray  # Shape: (max_grid_h, max_grid_w)
    working_grid_mask: jnp.ndarray  # Shape: (max_grid_h, max_grid_w), bool

    # Program state
    program: jnp.ndarray  # Shape: (max_program_length, max_action_params)
    program_length: jnp.ndarray  # int32 scalar

    # Agent state
    active_agents: jnp.ndarray  # Shape: (max_agents,), bool
    cumulative_rewards: jnp.ndarray  # Shape: (max_agents,), float32

    def __post_init__(self) -> None:
        """Validate the ArcEnvState structure and types."""
        # Skip validation during JAX transformations
        if not (hasattr(self.done, "ndim") and hasattr(self.done, "shape")):
            return

        try:
            # Validate base fields
            chex.assert_type(self.done, jnp.bool_)
            chex.assert_shape(self.done, ())
            # step can be int or jax array during transformations
            if not (isinstance(self.step, int) or hasattr(self.step, "dtype")):
                step_type = type(self.step)
                msg = f"step must be int or JAX array, got {step_type}"
                raise TypeError(msg)
            chex.assert_type(self.active_train_pair_idx, jnp.int32)
            chex.assert_shape(self.active_train_pair_idx, ())

            # Validate grid state
            chex.assert_rank(self.working_grid, 2)
            chex.assert_rank(self.working_grid_mask, 2)
            chex.assert_type(self.working_grid, jnp.integer)
            chex.assert_type(self.working_grid_mask, jnp.bool_)

            # Ensure grid and mask shapes match
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

            # Check array consistency
            num_agents = self.active_agents.shape[0]
            chex.assert_shape(self.cumulative_rewards, (num_agents,))

        except (AttributeError, TypeError):
            # Skip validation during JAX transformations
            pass


class ArcMarlEnvBase(MultiAgentEnv, ABC):
    """
    Abstract base class for ARC Multi-Agent environments.

    This class provides the common interface and utilities that all ARC environments
    should implement. It handles basic setup and defines the core methods that
    subclasses must implement.
    """

    def __init__(
        self,
        num_agents: int = 2,
        max_grid_size: tuple[int, int] = (30, 30),
        max_episode_steps: int = 100,
        max_program_length: int = 20,
        max_action_params: int = 8,
        config: dict | None = None,
        **kwargs,
    ):
        """
        Initialize the ARC MARL environment.

        Args:
            num_agents: Number of agents in the environment
            max_grid_size: Maximum grid dimensions (height, width)
            max_episode_steps: Maximum steps per episode
            max_program_length: Maximum length of action programs
            max_action_params: Maximum parameters per action
            config: Additional configuration dictionary
            **kwargs: Additional arguments for parent class
        """
        super().__init__(num_agents=num_agents, **kwargs)

        self.num_agents = num_agents
        self.max_grid_size = max_grid_size
        self.max_episode_steps = max_episode_steps
        self.max_program_length = max_program_length
        self.max_action_params = max_action_params
        self.config = config or {}

        # Create agent list
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Set up action and observation spaces
        self._setup_spaces()

    def _setup_spaces(self) -> None:
        """Set up action and observation spaces for all agents."""
        # Default action space - subclasses can override
        action_space = self._get_default_action_space()
        obs_space = self._get_default_observation_space()

        self.action_spaces = dict.fromkeys(self.agents, action_space)
        self.observation_spaces = dict.fromkeys(self.agents, obs_space)

    def _get_default_action_space(self) -> Box:
        """Get default action space."""
        # Action: [category, type_id, ...params]
        action_dim = 2 + self.max_action_params
        return Box(
            low=0,
            high=30,  # Reasonable upper bound for grid coordinates and colors
            shape=(action_dim,),
            dtype=jnp.int32,
        )

    def _get_default_observation_space(self) -> Box:
        """Get default observation space."""
        # Observation: flattened grids + program + state info
        grid_size = self.max_grid_size[0] * self.max_grid_size[1]
        obs_size = (
            grid_size * 3  # working grid + mask + target
            + self.max_program_length * self.max_action_params  # program
            + 20  # misc state info
        )
        return Box(
            low=0,
            high=10,  # ARC colors are 0-9
            shape=(obs_size,),
            dtype=jnp.float32,
        )

    # Abstract methods that subclasses must implement

    @abstractmethod
    def reset(self, key: chex.PRNGKey) -> tuple[dict[str, chex.Array], ArcEnvState]:
        """
        Reset the environment with a new ARC task.

        Args:
            key: JAX random key for task selection and initialization

        Returns:
            Tuple of (observations, initial_state)
        """

    @abstractmethod
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

    @abstractmethod
    def get_obs(self, state: ArcEnvState) -> dict[str, chex.Array]:
        """
        Generate observations for all agents from the current state.

        Args:
            state: Current environment state

        Returns:
            Dictionary mapping agent IDs to observations
        """

    # Utility methods that subclasses can use or override

    def _calculate_grid_similarity(self, grid1: chex.Array, grid2: chex.Array) -> float:
        """Calculate pixel-wise similarity between two grids."""
        matches = jnp.sum(grid1 == grid2)
        total_pixels = grid1.size
        return matches / total_pixels

    def _is_terminal(self, state: ArcEnvState) -> bool:
        """Check if the episode should terminate."""
        # Terminate if max steps reached
        max_steps_reached = state.step >= self.max_episode_steps

        # Terminate if explicitly done
        explicitly_done = state.done

        return max_steps_reached | explicitly_done

    def _create_dummy_task_data(self, key: chex.PRNGKey) -> ParsedTaskData:
        """Create dummy task data for testing."""
        grid_h, grid_w = self.max_grid_size

        # Create dummy grids with different seeds
        key1, key2 = jax.random.split(key)
        input_grid = jax.random.randint(key1, (2, grid_h, grid_w), 0, 10)
        output_grid = jax.random.randint(key2, (2, grid_h, grid_w), 0, 10)

        return ParsedTaskData(
            input_grids_examples=input_grid,
            input_masks_examples=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
            output_grids_examples=output_grid,
            output_masks_examples=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
            num_train_pairs=2,
            test_input_grids=input_grid[:1],
            test_input_masks=jnp.ones((1, grid_h, grid_w), dtype=jnp.bool_),
            true_test_output_grids=output_grid[:1],
            true_test_output_masks=jnp.ones((1, grid_h, grid_w), dtype=jnp.bool_),
            num_test_pairs=1,
            task_index=jnp.array(-1, dtype=jnp.int32),
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "ArcMarlEnvBase"

    @property
    def agent_classes(self) -> dict[str, str]:
        """Agent class mapping."""
        return dict.fromkeys(self.agents, "base_agent")
