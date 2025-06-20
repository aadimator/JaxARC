"""
Tests for the primitive ARC environment.

This module contains tests for the MultiAgentPrimitiveArcEnv class and related
functionality to ensure proper JAX compatibility and multi-agent behavior.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import chex
from typing import Dict

from jaxarc.envs.primitive_env import MultiAgentPrimitiveArcEnv
from jaxarc.base.base_env import ArcEnvState


class TestMultiAgentPrimitiveArcEnv:
    """Test suite for MultiAgentPrimitiveArcEnv."""

    @pytest.fixture
    def config(self) -> dict:
        """Create a test environment configuration."""
        return {
            "max_grid_size": [10, 10],
            "max_num_agents": 2,
            "max_episode_steps": 20,
            "max_program_length": 5,
            "max_action_params": 4,
            "reward": {
                "progress_weight": 1.0,
                "step_penalty": -0.01,
                "success_bonus": 5.0,
            }
        }

    @pytest.fixture
    def env(self, config: dict) -> MultiAgentPrimitiveArcEnv:
        """Create a test environment instance."""
        return MultiAgentPrimitiveArcEnv(
            num_agents=2,
            config=config,
        )

    @pytest.fixture
    def prng_key(self) -> chex.PRNGKey:
        """Create a test PRNG key."""
        return jax.random.PRNGKey(42)

    def test_environment_initialization(self, env: MultiAgentPrimitiveArcEnv):
        """Test that environment initializes correctly."""
        assert env.num_agents == 2
        assert len(env.agents) == 2
        assert env.max_grid_size == (10, 10)
        assert env.config["reward"]["progress_weight"] == 1.0

    def test_action_space_setup(self, env: MultiAgentPrimitiveArcEnv):
        """Test that action spaces are set up correctly."""
        assert len(env.action_spaces) == env.num_agents

        for agent_id, action_space in env.action_spaces.items():
            assert hasattr(action_space, 'shape')
            assert hasattr(action_space, 'dtype')
            assert action_space.dtype == jnp.int32
            # Action space should be [category, primitive_type, control_type, ...params]
            expected_dim = 3 + env.max_action_params
            assert action_space.shape == (expected_dim,)

    def test_observation_space_setup(self, env: MultiAgentPrimitiveArcEnv):
        """Test that observation spaces are set up correctly."""
        assert len(env.observation_spaces) == env.num_agents

        for agent_id, obs_space in env.observation_spaces.items():
            assert hasattr(obs_space, 'shape')
            assert hasattr(obs_space, 'dtype')
            assert obs_space.dtype == jnp.float32
            assert len(obs_space.shape) == 1  # Flattened observation

    def test_reset_functionality(self, env: MultiAgentPrimitiveArcEnv, prng_key: chex.PRNGKey):
        """Test environment reset functionality."""
        observations, state = env.reset(prng_key)

        # Check observations
        assert isinstance(observations, dict)
        assert len(observations) == env.num_agents

        for agent_id in env.agents:
            assert agent_id in observations
            obs = observations[agent_id]
            expected_shape = env.observation_spaces[agent_id].shape
            assert obs.shape == expected_shape
            assert obs.dtype == jnp.float32

        # Check state
        assert isinstance(state, ArcEnvState)
        chex.assert_type(state.done, jnp.bool_)
        assert isinstance(state.step, int)
        assert state.step == 0
        assert not state.done

    def test_step_functionality(self, env: MultiAgentPrimitiveArcEnv, prng_key: chex.PRNGKey):
        """Test environment step functionality."""
        # Reset environment first
        key1, key2 = jax.random.split(prng_key)
        observations, state = env.reset(key1)

        # Create dummy actions
        actions = {}
        for agent_id in env.agents:
            action_shape = env.action_spaces[agent_id].shape
            actions[agent_id] = jnp.zeros(action_shape, dtype=jnp.int32)

        # Take a step
        next_obs, next_state, rewards, dones, info = env.step_env(key2, state, actions)

        # Check outputs
        assert isinstance(next_obs, dict)
        assert isinstance(next_state, ArcEnvState)
        assert isinstance(rewards, dict)
        assert isinstance(dones, dict)
        assert isinstance(info, dict)

        # Check state progression
        assert next_state.step == state.step + 1

        # Check rewards structure
        assert len(rewards) == env.num_agents
        for agent_id in env.agents:
            assert agent_id in rewards
            # Rewards can be JAX arrays or Python floats
            assert hasattr(rewards[agent_id], 'dtype') or isinstance(rewards[agent_id], float)

        # Check dones structure
        assert len(dones) == env.num_agents + 1  # +1 for "__all__"
        assert "__all__" in dones

    def test_jax_compatibility(self, env: MultiAgentPrimitiveArcEnv, prng_key: chex.PRNGKey):
        """Test that environment functions work with JAX transformations."""
        # Test that reset can be JIT compiled
        jitted_reset = jax.jit(env.reset)
        observations, state = jitted_reset(prng_key)

        assert isinstance(observations, dict)
        assert isinstance(state, ArcEnvState)

        # Test that step can be JIT compiled
        actions = {}
        for agent_id in env.agents:
            action_shape = env.action_spaces[agent_id].shape
            actions[agent_id] = jnp.zeros(action_shape, dtype=jnp.int32)

        jitted_step = jax.jit(env.step_env)
        key1, key2 = jax.random.split(prng_key)
        _, state = env.reset(key1)

        next_obs, next_state, rewards, dones, info = jitted_step(key2, state, actions)

        assert isinstance(next_obs, dict)
        assert isinstance(next_state, ArcEnvState)

    def test_grid_similarity_calculation(self, env: MultiAgentPrimitiveArcEnv):
        """Test grid similarity calculation."""
        # Test identical grids
        grid1 = jnp.ones((5, 5), dtype=jnp.int32)
        grid2 = jnp.ones((5, 5), dtype=jnp.int32)
        similarity = env._calculate_grid_similarity(grid1, grid2)
        assert similarity == 1.0

        # Test completely different grids
        grid3 = jnp.zeros((5, 5), dtype=jnp.int32)
        similarity = env._calculate_grid_similarity(grid1, grid3)
        assert similarity == 0.0

        # Test partially similar grids
        grid4 = jnp.ones((5, 5), dtype=jnp.int32)
        grid4 = grid4.at[0, 0].set(0)  # Change one pixel
        similarity = env._calculate_grid_similarity(grid1, grid4)
        expected = 24.0 / 25.0  # 24 matching pixels out of 25
        assert abs(similarity - expected) < 1e-6

    def test_terminal_conditions(self, env: MultiAgentPrimitiveArcEnv, prng_key: chex.PRNGKey):
        """Test terminal condition checking."""
        observations, state = env.reset(prng_key)

        # Test that environment is not initially terminal
        assert not env._is_terminal(state)

        # Test max steps termination
        max_steps_state = state.replace(step=env.max_episode_steps)
        assert env._is_terminal(max_steps_state)

    def test_reward_calculation(self, env: MultiAgentPrimitiveArcEnv, prng_key: chex.PRNGKey):
        """Test reward calculation."""
        key1, key2 = jax.random.split(prng_key)
        observations, state = env.reset(key1)

        # Create dummy actions
        actions = {}
        for agent_id in env.agents:
            action_shape = env.action_spaces[agent_id].shape
            actions[agent_id] = jnp.zeros(action_shape, dtype=jnp.int32)

        # Calculate rewards
        next_state = state.replace(step=state.step + 1)
        rewards = env._calculate_rewards(key2, state, next_state, actions)

        assert isinstance(rewards, dict)
        assert len(rewards) == env.num_agents

        for agent_id in env.agents:
            assert agent_id in rewards
            # Rewards can be JAX arrays or Python floats
            assert hasattr(rewards[agent_id], 'dtype') or isinstance(rewards[agent_id], float)
            # Should include step penalty at minimum
            reward_val = float(rewards[agent_id]) if hasattr(rewards[agent_id], 'dtype') else rewards[agent_id]
            assert reward_val <= 1.0  # Should be reasonable (including progress reward)


class TestConfig:
    """Test suite for configuration handling."""

    def test_default_config_loading(self):
        """Test loading default configuration."""
        from jaxarc.envs.primitive_env import load_config

        config = load_config()

        assert config["max_grid_size"] == [30, 30]
        assert config["max_num_agents"] == 4
        assert config["max_episode_steps"] == 100
        assert "reward" in config
        assert config["reward"]["progress_weight"] == 1.0

    def test_config_with_environment(self):
        """Test that environment accepts config properly."""
        config = {
            "max_grid_size": [8, 8],
            "max_num_agents": 3,
            "max_episode_steps": 50,
            "max_program_length": 10,
            "max_action_params": 6,
        }

        env = MultiAgentPrimitiveArcEnv(num_agents=2, config=config)
        assert env.max_grid_size == (8, 8)
        assert env.max_episode_steps == 50
        assert env.max_program_length == 10
        assert env.max_action_params == 6
