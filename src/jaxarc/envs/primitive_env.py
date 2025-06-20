"""
Primitive ARC Multi-Agent Environment Implementation.

This module implements a JAX-compatible multi-agent environment for solving ARC tasks
using primitive operations. Agents collaborate to manipulate grids through actions like
drawing pixels, lines, flood filling, and copy-paste operations.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.spaces import Box, Discrete

from ..base.base_env import ArcMarlEnvBase, ArcEnvState
from ..types import (
    ActionCategory,
    AgentID,
    ControlType,
    ParsedTaskData,
    PrimitiveType,
)


class MultiAgentPrimitiveArcEnv(ArcMarlEnvBase):
    """
    Multi-agent primitive ARC environment.

    This environment allows multiple agents to collaborate on solving ARC tasks
    by taking primitive actions on a shared working grid. Agents can draw pixels, lines,
    flood fill regions, and copy-paste rectangles.
    """

    def __init__(
        self,
        num_agents: int = 2,
        config: dict | None = None,
        **kwargs,
    ):
        """
        Initialize the primitive ARC environment.

        Args:
            num_agents: Number of collaborative agents
            config: Environment configuration dictionary (from Hydra)
            **kwargs: Additional arguments passed to base class
        """
        # Set up configuration with defaults
        self.config = config or {}
        max_grid_size = tuple(self.config.get("max_grid_size", [30, 30]))
        max_episode_steps = self.config.get("max_episode_steps", 100)
        max_program_length = self.config.get("max_program_length", 20)
        max_action_params = self.config.get("max_action_params", 8)

        # Validate configuration
        if num_agents > self.config.get("max_num_agents", 4):
            raise ValueError(
                f"num_agents ({num_agents}) exceeds max_num_agents ({self.config.get('max_num_agents', 4)})"
            )

        super().__init__(
            num_agents=num_agents,
            max_grid_size=max_grid_size,
            max_episode_steps=max_episode_steps,
            max_program_length=max_program_length,
            max_action_params=max_action_params,
            config=self.config,
            **kwargs,
        )

    def _setup_spaces(self) -> None:
        """Set up action and observation spaces for all agents."""
        # Action space: [category, primitive_type, control_type, ...params]
        action_dim = 3 + self.max_action_params
        action_space = Box(
            low=0,
            high=max(len(PrimitiveType), len(ControlType), 30),  # Grid size upper bound
            shape=(action_dim,),
            dtype=jnp.int32,
        )

        # Observation space: flattened working grid + task examples + program state
        grid_size = self.max_grid_size[0] * self.max_grid_size[1]
        obs_size = (
            grid_size  # working_grid
            + grid_size  # working_grid_mask
            + grid_size * 4  # train examples (2 pairs x 2 grids each)
            + self.max_program_length * self.max_action_params  # program history
            + 10  # misc state info
        )

        obs_space = Box(
            low=0,
            high=10,  # ARC colors are 0-9
            shape=(obs_size,),
            dtype=jnp.float32,
        )

        self.action_spaces = {agent: action_space for agent in self.agents}
        self.observation_spaces = {agent: obs_space for agent in self.agents}

    def reset(self, key: chex.PRNGKey) -> tuple[dict[str, chex.Array], ArcEnvState]:
        """
        Reset the environment to initial state.

        Args:
            key: JAX PRNG key for randomization

        Returns:
            Tuple of (observations, initial_state)
        """
        key, task_key, init_key = jax.random.split(key, 3)

        # Load task data (placeholder - would normally load from dataset)
        task_data = self._create_dummy_task_data(task_key)

        # Initialize environment state
        state = self._initialize_state(init_key, task_data)

        # Get initial observations
        observations = self.get_obs(state)

        return observations, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: PrimitiveArcEnvState,
        actions: dict[str, chex.Array],
    ) -> tuple[
        dict[str, chex.Array],  # observations
        ArcEnvState,   # next_state
        dict[str, float],       # rewards
        dict[str, bool],        # dones
        dict[str, Any],         # info
    ]:
        """
        Execute one environment step.

        Args:
            key: JAX PRNG key
            state: Current environment state
            actions: Actions from all agents

        Returns:
            Tuple of (observations, next_state, rewards, dones, info)
        """
        key, step_key, reward_key = jax.random.split(key, 3)

        # Process actions for all agents
        next_state = self._process_actions(step_key, state, actions)

        # Calculate rewards
        rewards = self._calculate_rewards(reward_key, state, next_state, actions)

        # Check if episode is done
        done = self._is_terminal(next_state)
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        # Get observations for next state
        observations = self.get_obs(next_state)

        # Compile info dictionary
        info = self._get_info(state, next_state, actions, rewards)

        return observations, next_state, rewards, dones, info

    def get_obs(self, state: ArcEnvState) -> dict[str, chex.Array]:
        """
        Get observations for all agents.

        Args:
            state: Current environment state

        Returns:
            Dictionary mapping agent IDs to observations
        """
        # Flatten working grid and mask
        working_grid_flat = state.working_grid.flatten()
        working_mask_flat = state.working_grid_mask.flatten().astype(jnp.float32)

        # Get current target from task data (first training pair output)
        target_grid_flat = state.task_data.output_grids_examples[state.active_train_pair_idx].flatten()

        # Task examples (first 2 training pairs, simplified)
        task_examples = jnp.concatenate([
            state.task_data.input_grids_examples[:2].flatten(),
            state.task_data.output_grids_examples[:2].flatten()
        ])

        # Program history
        program_flat = state.program.flatten()

        # Misc state info
        misc_info = jnp.array([
            jnp.float32(state.step),
            state.active_train_pair_idx.astype(jnp.float32),
            state.program_length.astype(jnp.float32),
            jnp.sum(state.active_agents).astype(jnp.float32),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
        ])

        # Combine all observation components
        obs_components = [
            working_grid_flat.astype(jnp.float32),
            working_mask_flat,
            target_grid_flat.astype(jnp.float32),
            task_examples.astype(jnp.float32),
            program_flat.astype(jnp.float32),
            misc_info,
        ]

        obs = jnp.concatenate(obs_components)

        # Pad or truncate to expected size
        expected_size = self.observation_spaces[self.agents[0]].shape[0]
        if obs.shape[0] < expected_size:
            obs = jnp.pad(obs, (0, expected_size - obs.shape[0]))
        else:
            obs = obs[:expected_size]

        return {agent: obs for agent in self.agents}

    def _create_dummy_task_data(self, key: chex.PRNGKey) -> ParsedTaskData:
        """Create dummy task data for testing (placeholder implementation)."""
        grid_h, grid_w = self.max_grid_size

        # Create dummy grids with different seeds so input != output
        key1, key2 = jax.random.split(key)
        input_grid = jax.random.randint(key1, (2, grid_h, grid_w), 0, 10)  # 2 training pairs
        output_grid = jax.random.randint(key2, (2, grid_h, grid_w), 0, 10)

        return ParsedTaskData(
            input_grids_examples=input_grid,
            input_masks_examples=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
            output_grids_examples=output_grid,
            output_masks_examples=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
            num_train_pairs=2,
            test_input_grids=input_grid[:1],  # Use first as test
            test_input_masks=jnp.ones((1, grid_h, grid_w), dtype=jnp.bool_),
            true_test_output_grids=output_grid[:1],
            true_test_output_masks=jnp.ones((1, grid_h, grid_w), dtype=jnp.bool_),
            num_test_pairs=1,
            task_id=None,
        )

    def _initialize_state(
        self, key: chex.PRNGKey, task_data: ParsedTaskData
    ) -> ArcEnvState:
        """Initialize the environment state."""
        # Initialize working grid with first training input
        working_grid = task_data.input_grids_examples[0]
        working_grid_mask = task_data.input_masks_examples[0]

        # Initialize program
        program = jnp.zeros((self.max_program_length, self.max_action_params), dtype=jnp.int32)

        # Initialize agent states
        active_agents = jnp.ones(self.num_agents, dtype=jnp.bool_)
        cumulative_rewards = jnp.zeros(self.num_agents, dtype=jnp.float32)

        return ArcEnvState(
            done=jnp.array(False),
            step=0,
            task_data=task_data,
            active_train_pair_idx=jnp.array(0, dtype=jnp.int32),
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            program=program,
            program_length=jnp.array(0, dtype=jnp.int32),
            active_agents=active_agents,
            cumulative_rewards=cumulative_rewards,
        )

    def _process_actions(
        self,
        key: chex.PRNGKey,
        state: ArcEnvState,
        actions: dict[str, chex.Array],
    ) -> ArcEnvState:
        """
        Process actions from all agents sequentially.

        Args:
            key: JAX PRNG key
            state: Current state
            actions: Actions from all agents

        Returns:
            Updated state after processing actions
        """
        # Process actions sequentially for deterministic behavior
        new_state = state

        for agent_id in sorted(actions.keys()):
            action = actions[agent_id]
            new_state = self._process_single_action(key, new_state, agent_id, action)

        # Update global state
        new_state = new_state.replace(
            step=state.step + 1,
        )

        return new_state

    def _process_single_action(
        self,
        key: chex.PRNGKey,
        state: ArcEnvState,
        agent_id: str,
        action: chex.Array,
    ) -> ArcEnvState:
        """Process a single agent's action."""
        # Parse action
        category = action[0]
        primitive_type = action[1]
        control_type = action[2]
        params = action[3:]

        # Process based on category using JAX-compatible control flow
        return jax.lax.cond(
            category == ActionCategory.PRIMITIVE,
            lambda: self._process_primitive_action(key, state, primitive_type, params),
            lambda: jax.lax.cond(
                category == ActionCategory.CONTROL,
                lambda: self._process_control_action(key, state, control_type, params),
                lambda: state  # Invalid action - return state unchanged
            )
        )

    def _process_primitive_action(
        self,
        key: chex.PRNGKey,
        state: ArcEnvState,
        primitive_type: int,
        params: chex.Array,
    ) -> ArcEnvState:
        """Process a primitive action (placeholder - will implement actual primitives later)."""
        # For now, just add to program without modifying grid
        # TODO: Implement actual primitive operations

        # Add action to program if there's space
        new_program_length = jnp.minimum(
            state.program_length + 1,
            self.max_program_length
        )

        # Create program entry
        program_entry = jnp.concatenate([
            jnp.array([ActionCategory.PRIMITIVE, primitive_type, 0], dtype=jnp.int32),
            params[:self.max_action_params-3].astype(jnp.int32)
        ])

        # Pad to full size
        if program_entry.shape[0] < self.max_action_params:
            program_entry = jnp.pad(
                program_entry,
                (0, self.max_action_params - program_entry.shape[0])
            )
        else:
            program_entry = program_entry[:self.max_action_params]

        # Update program
        new_program = state.program.at[state.program_length].set(program_entry)

        return state.replace(
            program=new_program,
            program_length=new_program_length,
        )

    def _process_control_action(
        self,
        key: chex.PRNGKey,
        state: ArcEnvState,
        control_type: int,
        params: chex.Array,
    ) -> ArcEnvState:
        """Process a control action."""
        return jax.lax.cond(
            control_type == ControlType.RESET,
            lambda: state.replace(
                working_grid=state.task_data.input_grids_examples[state.active_train_pair_idx],
                working_grid_mask=state.task_data.input_masks_examples[state.active_train_pair_idx],
                program=jnp.zeros_like(state.program),
                program_length=jnp.array(0, dtype=jnp.int32),
            ),
            lambda: jax.lax.cond(
                control_type == ControlType.SUBMIT,
                lambda: state.replace(done=jnp.array(True)),
                lambda: state  # Unknown control action
            )
        )

    def _calculate_rewards(
        self,
        key: chex.PRNGKey,
        prev_state: PrimitiveArcEnvState,
        next_state: PrimitiveArcEnvState,
        actions: dict[str, chex.Array],
    ) -> dict[str, float]:
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
        # Get current target
        target_grid = next_state.task_data.output_grids_examples[next_state.active_train_pair_idx]

        # Calculate progress reward
        prev_similarity = self._calculate_grid_similarity(prev_state.working_grid, target_grid)
        new_similarity = self._calculate_grid_similarity(next_state.working_grid, target_grid)
        progress_reward = (new_similarity - prev_similarity) * self.config.get("reward", {}).get("progress_weight", 1.0)

        # Step penalty
        step_penalty = self.config.get("reward", {}).get("step_penalty", -0.01)

        # Success bonus if submitted and correct
        success_bonus = jax.lax.cond(
            next_state.done & (new_similarity > 0.95),  # Nearly perfect
            lambda: self.config.get("reward", {}).get("success_bonus", 10.0),
            lambda: 0.0
        )

        total_reward = progress_reward + step_penalty + success_bonus

        return {agent: total_reward for agent in self.agents}

    def _calculate_grid_similarity(self, grid1: chex.Array, grid2: chex.Array) -> float:
        """Calculate similarity between two grids."""
        matches = jnp.sum(grid1 == grid2)
        total_pixels = grid1.size
        return matches / total_pixels

    def _is_terminal(self, state: ArcEnvState) -> bool:
        """Check if the episode should terminate."""
        # Terminate if max steps reached
        max_steps_reached = state.step >= self.max_episode_steps

        # Terminate if explicitly done (e.g., submitted)
        explicitly_done = state.done

        return max_steps_reached | explicitly_done

    def _get_info(
        self,
        prev_state: ArcEnvState,
        next_state: ArcEnvState,
        actions: dict[str, chex.Array],
        rewards: dict[str, float],
    ) -> dict[str, Any]:
        """Compile info dictionary for debugging and analysis."""
        target_grid = next_state.task_data.output_grids_examples[next_state.active_train_pair_idx]

        return {
            "step": next_state.step,
            "active_train_pair": next_state.active_train_pair_idx,
            "program_length": next_state.program_length,
            "grid_similarity": self._calculate_grid_similarity(next_state.working_grid, target_grid),
            "active_agents": jnp.sum(next_state.active_agents),
            "episode_done": self._is_terminal(next_state),
        }

    @property
    def name(self) -> str:
        """Environment name."""
        return "MultiAgentPrimitiveArcEnv"

    @property
    def agent_classes(self) -> dict[str, str]:
        """Agent class mapping."""
        return {agent: "primitive_agent" for agent in self.agents}


# --- JAX Optimization Patterns ---

@jax.jit
def batched_env_step(
    keys: chex.Array,
    states: ArcEnvState,
    actions: dict[str, chex.Array],
    env: MultiAgentPrimitiveArcEnv,
) -> tuple[dict[str, chex.Array], ArcEnvState, dict[str, chex.Array], dict[str, chex.Array]]:
    """Batched environment step for parallel execution."""
    # Use vmap to vectorize over batch dimension
    return jax.vmap(env.step_env)(keys, states, actions)


def parallel_program_execution(
    programs: chex.Array, initial_grids: chex.Array
) -> chex.Array:
    """Execute multiple programs in parallel (placeholder)."""
    # TODO: Implement parallel program execution
    return initial_grids


def memory_efficient_primitive_application(
    primitive_type: int, params: chex.Array, grid: chex.Array
) -> chex.Array:
    """Memory-efficient primitive operation application (placeholder)."""
    # TODO: Implement memory-efficient primitive operations
    return grid


def batch_grid_similarity(grids1: chex.Array, grids2: chex.Array) -> chex.Array:
    """Calculate similarity between batches of grids."""
    def single_similarity(grid1: chex.Array, grid2: chex.Array) -> float:
        matches = jnp.sum(grid1 == grid2)
        total_pixels = grid1.size
        return matches / total_pixels

    return jax.vmap(single_similarity)(grids1, grids2)


# --- Configuration Loading ---

def load_config(config_path: str | None = None) -> dict:
    """Load environment configuration from Hydra config."""
    if config_path is None:
        # Return default config
        return {
            "max_grid_size": [30, 30],
            "max_num_agents": 4,
            "max_episode_steps": 100,
            "max_program_length": 20,
            "max_action_params": 8,
            "reward": {
                "progress_weight": 1.0,
                "step_penalty": -0.01,
                "success_bonus": 10.0,
            }
        }
    else:
        # TODO: Implement actual Hydra config loading
        raise NotImplementedError("Config loading from file not yet implemented")
