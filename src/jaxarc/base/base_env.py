"""
Abstract base class for ARC Multi-Agent Reinforcement Learning environments.

This module provides the foundational components for creating JAX-compatible
ARC environments that integrate with the JaxMARL framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Dict as DictSpace, Discrete

from ..types import AgentID, ParsedTaskData


@chex.dataclass
class ArcEnvState:
    """
    State dataclass for ARC Multi-Agent environments.

    Extends the basic JaxMARL state with ARC-specific fields for tracking
    task progress, agent collaboration, and grid manipulation.

    Attributes:
        # JaxMARL required fields
        done: Boolean array indicating if environment is done
        step: Current step number in the environment

        # ARC task state
        task_data: Parsed ARC task containing training/test examples
        current_test_case: Index of current test case being solved
        phase: Current collaboration phase (0=ideation, 1=proposal, 2=voting, 3=commit)

        # Grid manipulation state
        current_grid: Working grid being modified by agents
        current_grid_mask: Mask indicating valid cells in current_grid
        target_grid: Ground truth solution for verification
        target_grid_mask: Mask indicating valid cells in target_grid

        # Agent collaboration state
        agent_hypotheses: Array storing agent hypotheses
        hypothesis_votes: Vote counts for each hypothesis
        consensus_threshold: Required votes for consensus
        active_agents: Mask indicating which agents are active

        # Step and timing state
        phase_step: Current step within the current phase
        max_phase_steps: Maximum steps allowed per phase
        episode_step: Overall episode step counter
        max_episode_steps: Maximum steps allowed per episode

        # Reward and performance tracking
        cumulative_rewards: Cumulative rewards for each agent
        solution_found: Boolean indicating if solution has been found
        last_action_valid: Boolean array indicating if last actions were valid
    """

    # JaxMARL required fields
    done: chex.Array
    step: int

    # ARC task state
    task_data: ParsedTaskData
    current_test_case: jnp.ndarray  # int32 scalar
    phase: jnp.ndarray  # int32 scalar (0=ideation, 1=proposal, 2=voting, 3=commit)

    # Grid manipulation state
    current_grid: jnp.ndarray  # Shape: (max_grid_h, max_grid_w)
    current_grid_mask: jnp.ndarray  # Shape: (max_grid_h, max_grid_w), bool
    target_grid: jnp.ndarray  # Shape: (max_grid_h, max_grid_w)
    target_grid_mask: jnp.ndarray  # Shape: (max_grid_h, max_grid_w), bool

    # Agent collaboration state
    agent_hypotheses: jnp.ndarray  # Shape: (max_agents, max_hypotheses, hypothesis_dim)
    hypothesis_votes: jnp.ndarray  # Shape: (max_agents, max_hypotheses)
    consensus_threshold: jnp.ndarray  # int32 scalar
    active_agents: jnp.ndarray  # Shape: (max_agents,), bool

    # Step and timing state
    phase_step: jnp.ndarray  # int32 scalar
    max_phase_steps: jnp.ndarray  # int32 scalar
    episode_step: jnp.ndarray  # int32 scalar
    max_episode_steps: jnp.ndarray  # int32 scalar

    # Reward and performance tracking
    cumulative_rewards: jnp.ndarray  # Shape: (max_agents,), float32
    solution_found: jnp.ndarray  # bool scalar
    last_action_valid: jnp.ndarray  # Shape: (max_agents,), bool

    def __post_init__(self) -> None:
        """Validate the ArcEnvState structure and types."""
        # Check if we're in a JAX transformation context where array fields
        # have been transformed into non-array objects (e.g., during tree operations)
        if not (hasattr(self.current_test_case, 'ndim') and
                hasattr(self.current_test_case, 'shape')):
            # Skip validation during JAX tree operations where fields are transformed
            return

        try:
            # Determine if we're dealing with batched data (for vmap compatibility)
            # Check if current_test_case has a batch dimension
            is_batched = self.current_test_case.ndim > 0
            batch_rank_offset = 1 if is_batched else 0
            scalar_shape = (self.current_test_case.shape[0],) if is_batched else ()

            # Validate JaxMARL required fields
            chex.assert_type(self.done, jnp.bool_)
            chex.assert_type(self.step, int)

            # Validate ARC task state
            chex.assert_type(self.current_test_case, jnp.int32)
            chex.assert_shape(self.current_test_case, scalar_shape)
            chex.assert_type(self.phase, jnp.int32)
            chex.assert_shape(self.phase, scalar_shape)

            # Validate grid state (grids get extra batch dimension if batched)
            expected_grid_rank = 2 + batch_rank_offset
            chex.assert_rank(self.current_grid, expected_grid_rank)
            chex.assert_rank(self.current_grid_mask, expected_grid_rank)
            chex.assert_rank(self.target_grid, expected_grid_rank)
            chex.assert_rank(self.target_grid_mask, expected_grid_rank)
            chex.assert_type(self.current_grid, jnp.integer)
            chex.assert_type(self.target_grid, jnp.integer)
            chex.assert_type(self.current_grid_mask, jnp.bool_)
            chex.assert_type(self.target_grid_mask, jnp.bool_)

            # Ensure grid dimensions match
            chex.assert_shape(self.current_grid_mask, self.current_grid.shape)
            chex.assert_shape(self.target_grid, self.current_grid.shape)
            chex.assert_shape(self.target_grid_mask, self.current_grid.shape)

            # Validate collaboration state (get extra batch dimension if batched)
            expected_hypotheses_rank = 3 + batch_rank_offset
            expected_votes_rank = 2 + batch_rank_offset
            expected_agents_rank = 1 + batch_rank_offset
            chex.assert_rank(self.agent_hypotheses, expected_hypotheses_rank)
            chex.assert_rank(self.hypothesis_votes, expected_votes_rank)
            chex.assert_rank(self.active_agents, expected_agents_rank)
            chex.assert_type(self.active_agents, jnp.bool_)

            # Validate step and timing state
            for field in [self.consensus_threshold, self.phase_step, self.max_phase_steps,
                         self.episode_step, self.max_episode_steps]:
                chex.assert_type(field, jnp.int32)
                chex.assert_shape(field, scalar_shape)

            # Validate reward tracking
            expected_rewards_rank = 1 + batch_rank_offset
            expected_actions_rank = 1 + batch_rank_offset
            chex.assert_rank(self.cumulative_rewards, expected_rewards_rank)
            chex.assert_type(self.cumulative_rewards, jnp.float32)
            chex.assert_type(self.solution_found, jnp.bool_)
            chex.assert_shape(self.solution_found, scalar_shape)
            chex.assert_rank(self.last_action_valid, expected_actions_rank)
            chex.assert_type(self.last_action_valid, jnp.bool_)

        except (AttributeError, TypeError):
            # Skip validation only for specific JAX transformation errors
            # This preserves normal validation while allowing JAX operations
            pass


class ArcMarlEnvBase(MultiAgentEnv, ABC):
    """
    Abstract base class for ARC Multi-Agent Reinforcement Learning environments.

    This class provides the foundational structure for creating collaborative
    AI environments that solve ARC tasks through hypothesis generation,
    consensus building, and grid manipulation.

    The environment follows a structured four-phase approach:
    1. **Ideation Phase**: Agents observe and think privately
    2. **Proposal Phase**: Agents propose hypotheses
    3. **Voting Phase**: Agents vote on hypotheses and build consensus
    4. **Commit Phase**: Apply the consensus solution to the grid
    """

    def __init__(
        self,
        num_agents: int,
        max_grid_size: Tuple[int, int] = (30, 30),
        max_hypotheses_per_agent: int = 5,
        hypothesis_dim: int = 64,
        consensus_threshold: Optional[int] = None,
        max_phase_steps: int = 10,
        max_episode_steps: int = 100,
    ) -> None:
        """
        Initialize the ARC MARL environment base class.

        Args:
            num_agents: Number of collaborative agents
            max_grid_size: Maximum dimensions for grids (height, width)
            max_hypotheses_per_agent: Maximum hypotheses each agent can propose
            hypothesis_dim: Dimensionality of hypothesis representations
            consensus_threshold: Required votes for consensus (defaults to majority)
            max_phase_steps: Maximum steps allowed per collaboration phase
            max_episode_steps: Maximum total steps per episode
        """
        super().__init__(num_agents=num_agents)

        # Store configuration
        self.max_grid_size = max_grid_size
        self.max_hypotheses_per_agent = max_hypotheses_per_agent
        self.hypothesis_dim = hypothesis_dim
        self.consensus_threshold = consensus_threshold or (num_agents // 2 + 1)
        self.max_phase_steps = max_phase_steps
        self.max_episode_steps = max_episode_steps

        # Create agent IDs
        self.agents = [f"agent_{i}" for i in range(num_agents)]

        # Initialize action and observation spaces (to be defined by subclasses)
        self.action_spaces = {}
        self.observation_spaces = {}

    @abstractmethod
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], ArcEnvState]:
        """
        Reset the environment with a new ARC task.

        Args:
            key: JAX random key for reproducible randomness

        Returns:
            observations: Dictionary of initial observations for each agent
            state: Initial environment state
        """
        pass

    @abstractmethod
    def step_env(
        self,
        key: chex.PRNGKey,
        state: ArcEnvState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], ArcEnvState, Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            key: JAX random key
            state: Current environment state
            actions: Dictionary of actions for each agent

        Returns:
            observations: Next observations for each agent
            next_state: Updated environment state
            rewards: Rewards for each agent
            dones: Done flags for each agent
            infos: Additional information dictionary
        """
        pass

    @abstractmethod
    def get_obs(self, state: ArcEnvState) -> Dict[str, chex.Array]:
        """
        Generate observations for all agents from the current state.

        Args:
            state: Current environment state

        Returns:
            observations: Dictionary of observations for each agent
        """
        pass

    # Abstract methods for ARC-specific functionality

    @abstractmethod
    def _load_task_data(self, key: chex.PRNGKey) -> ParsedTaskData:
        """
        Load and return a parsed ARC task.

        Args:
            key: JAX random key for task selection

        Returns:
            Parsed task data ready for environment use
        """
        pass

    @abstractmethod
    def _process_hypotheses(
        self,
        key: chex.PRNGKey,
        state: ArcEnvState,
        actions: Dict[str, chex.Array],
    ) -> ArcEnvState:
        """
        Process agent hypotheses and update the collaboration state.

        Args:
            key: JAX random key
            state: Current environment state
            actions: Agent actions containing hypothesis data

        Returns:
            Updated environment state with new hypotheses
        """
        pass

    @abstractmethod
    def _update_consensus(
        self,
        key: chex.PRNGKey,
        state: ArcEnvState,
        actions: Dict[str, chex.Array],
    ) -> ArcEnvState:
        """
        Update consensus tracking based on agent votes.

        Args:
            key: JAX random key
            state: Current environment state
            actions: Agent actions containing vote data

        Returns:
            Updated environment state with new consensus information
        """
        pass

    @abstractmethod
    def _apply_grid_transformation(
        self,
        key: chex.PRNGKey,
        state: ArcEnvState,
        transformation_data: chex.Array,
    ) -> ArcEnvState:
        """
        Apply a transformation to the current grid based on consensus.

        Args:
            key: JAX random key
            state: Current environment state
            transformation_data: Data describing the transformation to apply

        Returns:
            Updated environment state with modified grid
        """
        pass

    @abstractmethod
    def _calculate_rewards(
        self,
        key: chex.PRNGKey,
        prev_state: ArcEnvState,
        next_state: ArcEnvState,
        actions: Dict[str, chex.Array],
    ) -> Dict[str, float]:
        """
        Calculate rewards for all agents based on state transition.

        Args:
            key: JAX random key
            prev_state: Previous environment state
            next_state: Next environment state
            actions: Actions taken by agents

        Returns:
            Dictionary of rewards for each agent
        """
        pass

    # Helper methods that can be overridden by subclasses

    def _advance_phase(self, state: ArcEnvState) -> ArcEnvState:
        """
        Advance to the next collaboration phase.

        Args:
            state: Current environment state

        Returns:
            Updated state with advanced phase
        """
        next_phase = (state.phase + 1) % 4
        return state.replace(
            phase=next_phase,
            phase_step=jnp.array(0, dtype=jnp.int32),
        )

    def _check_phase_completion(self, state: ArcEnvState) -> jnp.ndarray:
        """
        Check if the current phase should be completed.

        Args:
            state: Current environment state

        Returns:
            Boolean indicating if phase should advance
        """
        step_limit_reached = state.phase_step >= state.max_phase_steps

        # Phase-specific completion conditions can be added by subclasses
        return step_limit_reached

    def _check_solution_correctness(self, state: ArcEnvState) -> jnp.ndarray:
        """
        Check if the current grid matches the target solution.

        Args:
            state: Current environment state

        Returns:
            Boolean indicating if solution is correct
        """
        # Only compare valid cells (where both masks are True)
        valid_mask = state.current_grid_mask & state.target_grid_mask

        # Check if grids match at all valid positions
        matches = jnp.where(
            valid_mask,
            state.current_grid == state.target_grid,
            True  # Invalid positions are considered matching
        )

        return jnp.all(matches)

    def _is_terminal(self, state: ArcEnvState) -> jnp.ndarray:
        """
        Check if the environment should terminate.

        Args:
            state: Current environment state

        Returns:
            Boolean indicating if environment should terminate
        """
        solution_found = self._check_solution_correctness(state)
        step_limit_reached = state.episode_step >= state.max_episode_steps

        return solution_found | step_limit_reached

    def _get_default_action_space(self) -> DictSpace:
        """
        Get a default action space structure for ARC environments.

        Returns:
            Default action space with common action types
        """
        return DictSpace({
            "action_type": Discrete(8),  # Different action types
            "grid_x": Discrete(self.max_grid_size[1]),  # Grid x coordinate
            "grid_y": Discrete(self.max_grid_size[0]),  # Grid y coordinate
            "color": Discrete(10),  # ARC color (0-9)
            "hypothesis_id": Discrete(self.max_hypotheses_per_agent),
            "vote": Discrete(2),  # Binary vote (0=no, 1=yes)
        })

    def _get_default_observation_space(self) -> DictSpace:
        """
        Get a default observation space structure for ARC environments.

        Returns:
            Default observation space with common observation components
        """
        return DictSpace({
            "current_grid": Box(
                low=0, high=9,
                shape=self.max_grid_size,
                dtype=jnp.int32
            ),
            "target_grid": Box(
                low=0, high=9,
                shape=self.max_grid_size,
                dtype=jnp.int32
            ),
            "grid_mask": Box(
                low=0, high=1,
                shape=self.max_grid_size,
                dtype=jnp.bool_
            ),
            "phase": Box(low=0, high=3, shape=(), dtype=jnp.int32),
            "phase_step": Box(low=0, high=self.max_phase_steps, shape=(), dtype=jnp.int32),
            "agent_hypotheses": Box(
                low=-1.0, high=1.0,
                shape=(self.num_agents, self.max_hypotheses_per_agent, self.hypothesis_dim),
                dtype=jnp.float32
            ),
            "hypothesis_votes": Box(
                low=0, high=self.num_agents,
                shape=(self.num_agents, self.max_hypotheses_per_agent),
                dtype=jnp.int32
            ),
        })

    @property
    def name(self) -> str:
        """Environment name."""
        return "ArcMarlEnv-Base"

    @property
    def agent_classes(self) -> dict:
        """
        Returns agent class information for JaxMARL.

        Returns:
            Dictionary mapping agent names to their classes
        """
        return {agent: "ArcAgent" for agent in self.agents}
