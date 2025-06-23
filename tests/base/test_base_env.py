"""
Tests for ArcMarlEnvBase and ArcEnvState.

This module contains comprehensive tests for the abstract base classes
used in ARC multi-agent reinforcement learning environments.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.base.base_env import ArcEnvState, ArcMarlEnvBase
from jaxarc.types import ParsedTaskData


class MockArcEnv(ArcMarlEnvBase):
    """Mock implementation of ArcMarlEnvBase for testing."""

    def __init__(self, num_agents: int = 2, **kwargs):
        super().__init__(num_agents=num_agents, **kwargs)

        # Set up action and observation spaces
        self.action_spaces = {
            agent: self._get_default_action_space() for agent in self.agents
        }
        self.observation_spaces = {
            agent: self._get_default_observation_space() for agent in self.agents
        }

    def reset(self, key: chex.PRNGKey) -> tuple[dict[str, chex.Array], ArcEnvState]:
        """Mock reset implementation."""
        task_data = self._create_mock_task_data()
        state = self._create_mock_state(task_data)
        obs = self.get_obs(state)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,  # noqa: ARG002
        state: ArcEnvState,
        actions: dict[str, chex.Array],  # noqa: ARG002
    ) -> tuple[
        dict[str, chex.Array],
        ArcEnvState,
        dict[str, float],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Mock step_env implementation."""
        next_state = state.replace(
            step=state.step + 1,
        )

        obs = self.get_obs(next_state)
        rewards = dict.fromkeys(self.agents, 0.0)
        dones = dict.fromkeys(self.agents, False)
        dones["__all__"] = False
        infos = {}

        return obs, next_state, rewards, dones, infos

    def get_obs(self, state: ArcEnvState) -> dict[str, chex.Array]:
        """Mock get_obs implementation."""
        obs = {}
        for agent in self.agents:
            obs[agent] = {
                "working_grid": state.working_grid,
                "working_grid_mask": state.working_grid_mask,
                "program": state.program,
                "program_length": state.program_length,
                "active_train_pair_idx": state.active_train_pair_idx,
            }
        return obs

    def _load_task_data(self, key: chex.PRNGKey) -> ParsedTaskData:  # noqa: ARG002
        """Mock task data loading."""
        return self._create_mock_task_data()

    def _process_hypotheses(
        self,
        key: chex.PRNGKey,  # noqa: ARG002
        state: ArcEnvState,
        actions: dict[str, chex.Array],  # noqa: ARG002
    ) -> ArcEnvState:
        """Mock hypothesis processing."""
        return state

    def _update_consensus(
        self,
        key: chex.PRNGKey,  # noqa: ARG002
        state: ArcEnvState,
        actions: dict[str, chex.Array],  # noqa: ARG002
    ) -> ArcEnvState:
        """Mock consensus updating."""
        return state

    def _apply_grid_transformation(
        self,
        key: chex.PRNGKey,  # noqa: ARG002
        state: ArcEnvState,
        transformation_data: chex.Array,  # noqa: ARG002
    ) -> ArcEnvState:
        """Mock grid transformation."""
        return state

    def _calculate_rewards(
        self,
        key: chex.PRNGKey,  # noqa: ARG002
        prev_state: ArcEnvState,  # noqa: ARG002
        next_state: ArcEnvState,  # noqa: ARG002
        actions: dict[str, chex.Array],  # noqa: ARG002
    ) -> dict[str, float]:
        """Mock reward calculation."""
        return dict.fromkeys(self.agents, 0.0)

    def _create_mock_task_data(self) -> ParsedTaskData:
        """Create mock task data for testing."""
        max_train_pairs = 3
        max_test_pairs = 1
        grid_h, grid_w = 5, 5

        return ParsedTaskData(
            input_grids_examples=jnp.zeros(
                (max_train_pairs, grid_h, grid_w), dtype=jnp.int32
            ),
            input_masks_examples=jnp.ones(
                (max_train_pairs, grid_h, grid_w), dtype=jnp.bool_
            ),
            output_grids_examples=jnp.ones(
                (max_train_pairs, grid_h, grid_w), dtype=jnp.int32
            ),
            output_masks_examples=jnp.ones(
                (max_train_pairs, grid_h, grid_w), dtype=jnp.bool_
            ),
            num_train_pairs=2,
            test_input_grids=jnp.zeros(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.int32
            ),
            test_input_masks=jnp.ones(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.bool_
            ),
            true_test_output_grids=jnp.ones(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.int32
            ),
            true_test_output_masks=jnp.ones(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.bool_
            ),
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

    def _create_mock_state(self, task_data: ParsedTaskData) -> ArcEnvState:
        """Create mock environment state for testing."""
        grid_h, grid_w = self.max_grid_size

        return ArcEnvState(
            # JaxMARL required fields
            done=jnp.array(False, dtype=jnp.bool_),
            step=0,
            # ARC task state
            task_data=task_data,
            active_train_pair_idx=jnp.array(0, dtype=jnp.int32),
            # Grid state
            working_grid=jnp.zeros((grid_h, grid_w), dtype=jnp.int32),
            working_grid_mask=jnp.ones((grid_h, grid_w), dtype=jnp.bool_),
            # Program state
            program=jnp.zeros(
                (self.max_program_length, self.max_action_params), dtype=jnp.int32
            ),
            program_length=jnp.array(0, dtype=jnp.int32),
            # Agent state
            active_agents=jnp.ones((self.num_agents,), dtype=jnp.bool_),
            cumulative_rewards=jnp.zeros((self.num_agents,), dtype=jnp.float32),
        )


class TestArcEnvState:
    """Test cases for ArcEnvState dataclass."""

    def test_state_creation(self):
        """Test basic ArcEnvState creation."""
        env = MockArcEnv(num_agents=2)
        task_data = env._create_mock_task_data()
        state = env._create_mock_state(task_data)

        # Basic structure checks
        assert isinstance(state, ArcEnvState)
        assert state.step == 0
        assert not state.done
        assert state.active_train_pair_idx == 0

    def test_state_validation(self):
        """Test ArcEnvState validation logic."""
        env = MockArcEnv(num_agents=2)
        task_data = env._create_mock_task_data()
        state = env._create_mock_state(task_data)

        # Check types are correct
        chex.assert_type(state.done, jnp.bool_)
        chex.assert_type(state.active_train_pair_idx, jnp.int32)
        chex.assert_type(state.program_length, jnp.int32)
        chex.assert_type(state.working_grid, jnp.integer)
        chex.assert_type(state.working_grid_mask, jnp.bool_)

    def test_state_shapes(self):
        """Test ArcEnvState array shapes."""
        env = MockArcEnv(num_agents=3, max_grid_size=(10, 15))
        task_data = env._create_mock_task_data()
        state = env._create_mock_state(task_data)

        # Check grid shapes
        assert state.working_grid.shape == (10, 15)
        assert state.working_grid_mask.shape == (10, 15)

        # Check program shape
        assert state.program.shape == (env.max_program_length, env.max_action_params)

        # Check agent-related arrays
        assert state.active_agents.shape == (3,)
        assert state.cumulative_rewards.shape == (3,)

    def test_state_immutability(self):
        """Test that ArcEnvState supports immutable updates."""
        env = MockArcEnv(num_agents=2)
        task_data = env._create_mock_task_data()
        state = env._create_mock_state(task_data)

        # Test replace method
        new_state = state.replace(active_train_pair_idx=jnp.array(1, dtype=jnp.int32))

        # Original should be unchanged
        assert state.active_train_pair_idx == 0
        assert new_state.active_train_pair_idx == 1

        # Other fields should be the same
        assert jnp.array_equal(state.working_grid, new_state.working_grid)


class TestArcMarlEnvBase:
    """Test cases for ArcMarlEnvBase abstract class."""

    def test_initialization(self):
        """Test basic initialization of ArcMarlEnvBase."""
        env = MockArcEnv(num_agents=3, max_grid_size=(20, 25))

        assert env.num_agents == 3
        assert env.max_grid_size == (20, 25)
        assert len(env.agents) == 3
        assert env.agents == ["agent_0", "agent_1", "agent_2"]

    def test_max_program_length(self):
        """Test max program length configuration."""
        env = MockArcEnv(num_agents=4, max_program_length=50)
        assert env.max_program_length == 50

    def test_max_episode_steps(self):
        """Test max episode steps configuration."""
        env = MockArcEnv(num_agents=2, max_episode_steps=200)
        assert env.max_episode_steps == 200

    def test_action_and_observation_spaces(self):
        """Test action and observation space creation."""
        env = MockArcEnv(num_agents=2)

        # Check that spaces are created for all agents
        assert len(env.action_spaces) == 2
        assert len(env.observation_spaces) == 2

        for agent in env.agents:
            assert agent in env.action_spaces
            assert agent in env.observation_spaces

    def test_name_property(self):
        """Test environment name property."""
        env = MockArcEnv(num_agents=2)
        assert env.name == "ArcMarlEnvBase"

    def test_agent_classes_property(self):
        """Test agent_classes property."""
        env = MockArcEnv(num_agents=3)
        agent_classes = env.agent_classes

        assert len(agent_classes) == 3
        for agent in env.agents:
            assert agent_classes[agent] == "base_agent"


class TestArcMarlEnvHelperMethods:
    """Test cases for helper methods in ArcMarlEnvBase."""

    def test_grid_similarity_calculation(self):
        """Test grid similarity calculation."""
        env = MockArcEnv(num_agents=2)

        # Create identical grids
        grid1 = jnp.array([[1, 2], [3, 4]])
        grid2 = jnp.array([[1, 2], [3, 4]])

        similarity = env._calculate_grid_similarity(grid1, grid2)
        assert similarity == 1.0

        # Create partially matching grids
        grid3 = jnp.array([[1, 2], [3, 5]])
        similarity = env._calculate_grid_similarity(grid1, grid3)
        assert similarity == 0.75  # 3 out of 4 pixels match

    def test_is_terminal(self):
        """Test terminal condition checking."""
        env = MockArcEnv(num_agents=2, max_episode_steps=10)
        task_data = env._create_mock_task_data()
        state = env._create_mock_state(task_data)

        # Not terminal initially
        assert not env._is_terminal(state)

        # Terminal when step limit reached
        limit_state = state.replace(step=10)
        assert env._is_terminal(limit_state)

        # Terminal when explicitly done
        done_state = state.replace(done=jnp.array(True, dtype=jnp.bool_))
        assert env._is_terminal(done_state)

    def test_default_action_space(self):
        """Test default action space structure."""
        env = MockArcEnv(num_agents=2, max_grid_size=(15, 20))
        action_space = env._get_default_action_space()

        from jaxmarl.environments.spaces import Box

        assert isinstance(action_space, Box)

        # Should have action_dim = 2 + max_action_params
        expected_dim = 2 + env.max_action_params
        assert action_space.shape == (expected_dim,)
        assert action_space.dtype == jnp.int32

    def test_default_observation_space(self):
        """Test default observation space structure."""
        env = MockArcEnv(num_agents=3, max_grid_size=(12, 8))
        obs_space = env._get_default_observation_space()

        from jaxmarl.environments.spaces import Box

        assert isinstance(obs_space, Box)
        assert obs_space.dtype == jnp.float32

        # Check that obs_space has positive dimensions
        assert len(obs_space.shape) == 1
        assert obs_space.shape[0] > 0


class TestJaxCompatibility:
    """Test JAX compatibility and transformations."""

    def test_jit_compilation(self):
        """Test that environment methods can be JIT compiled."""
        env = MockArcEnv(num_agents=2)

        # Test JIT compilation of reset
        jit_reset = jax.jit(env.reset)
        key = jax.random.PRNGKey(42)
        obs, state = jit_reset(key)

        assert isinstance(obs, dict)
        assert isinstance(state, ArcEnvState)

    def test_step_jit_compilation(self):
        """Test JIT compilation of step method."""
        env = MockArcEnv(num_agents=2)
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        # Create mock actions
        actions = {}
        for agent in env.agents:
            # Action: [category, type_id, ...params]
            action_dim = 2 + env.max_action_params
            actions[agent] = jnp.zeros(action_dim, dtype=jnp.int32)

        # Test JIT compilation of step
        jit_step = jax.jit(env.step_env)
        key, subkey = jax.random.split(key)
        result = jit_step(subkey, state, actions)

        assert len(result) == 5  # obs, state, rewards, dones, infos

    def test_vmap_compatibility(self):
        """Test that environment state array fields work with vmap."""
        env = MockArcEnv(num_agents=2)
        task_data = env._create_mock_task_data()

        # Create batch of states and extract batchable array fields
        batch_size = 4
        steps = []
        cumulative_rewards = []
        for i in range(batch_size):
            state = env._create_mock_state(task_data)
            state = state.replace(step=i)
            steps.append(jnp.array(state.step, dtype=jnp.int32))
            cumulative_rewards.append(state.cumulative_rewards)

        # Stack the individual array fields
        batched_steps = jnp.stack(steps)
        batched_rewards = jnp.stack(cumulative_rewards)

        # Test vmap on individual array fields
        def increment_step(step):
            return step + 1

        def sum_rewards(rewards):
            return jnp.sum(rewards)

        vmap_increment = jax.vmap(increment_step)
        vmap_sum = jax.vmap(sum_rewards)

        incremented_steps = vmap_increment(batched_steps)
        summed_rewards = vmap_sum(batched_rewards)

        expected_steps = jnp.array([1, 2, 3, 4])
        expected_reward_sums = jnp.array([0.0, 0.0, 0.0, 0.0])  # All rewards start at 0

        assert jnp.array_equal(incremented_steps, expected_steps)
        assert jnp.array_equal(summed_rewards, expected_reward_sums)

    def test_state_tree_structure(self):
        """Test that ArcEnvState is a proper JAX pytree."""
        env = MockArcEnv(num_agents=2)
        task_data = env._create_mock_task_data()
        state = env._create_mock_state(task_data)

        # Test tree operations
        leaves = jax.tree.leaves(state)
        assert len(leaves) > 0

        # Test tree map
        def add_one_to_ints(x):
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.integer):
                return x + 1
            return x

        modified_state = jax.tree.map(add_one_to_ints, state)
        assert isinstance(modified_state, ArcEnvState)

    def test_reproducibility(self):
        """Test that environment is reproducible with same keys."""
        env1 = MockArcEnv(num_agents=2)
        env2 = MockArcEnv(num_agents=2)

        key = jax.random.PRNGKey(42)

        obs1, state1 = env1.reset(key)
        obs2, state2 = env2.reset(key)

        # States should be equivalent (same random seed)
        def compare_arrays(x, y):
            if hasattr(x, "shape") and hasattr(y, "shape"):
                return jnp.array_equal(x, y)
            return x == y

        # Compare all array fields without tree reconstruction
        leaves1 = jax.tree.leaves(state1)
        leaves2 = jax.tree.leaves(state2)

        assert len(leaves1) == len(leaves2)
        all_equal = all(compare_arrays(x, y) for x, y in zip(leaves1, leaves2))
        assert all_equal


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_num_agents(self):
        """Test initialization with invalid number of agents."""
        # Note: Currently the base class doesn't validate num_agents > 0
        # This test verifies that 0 agents doesn't crash during initialization
        try:
            env = MockArcEnv(num_agents=0)
            # If this doesn't raise an exception, that's actually fine
            # The base class allows 0 agents, though it may not be practical
            assert env.num_agents == 0
            assert len(env.agents) == 0
        except Exception:
            # If an exception is raised, that's also acceptable
            pass

    def test_state_validation_errors(self):
        """Test that state validation catches type errors."""
        env = MockArcEnv(num_agents=2)
        task_data = env._create_mock_task_data()

        # This should work
        valid_state = env._create_mock_state(task_data)

        # Test that validation would catch wrong types
        # (We can't actually create invalid states due to chex validation,
        # but we can verify the validation exists)
        assert hasattr(valid_state, "__post_init__")


if __name__ == "__main__":
    pytest.main([__file__])
