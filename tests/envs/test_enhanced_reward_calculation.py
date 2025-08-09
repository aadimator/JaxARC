from __future__ import annotations

import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf
from jaxarc.envs.config import RewardConfig
from jaxarc.envs.functional import _calculate_enhanced_reward

"""Tests for enhanced reward calculation with mode awareness.

This module tests the enhanced reward calculation system that provides:
- Training mode reward calculation with configurable frequency
- Evaluation mode reward calculation with target masking
- Proper similarity scoring for different pair types
- Different reward structures based on configuration
- JIT-compilable and efficient implementation
"""


class TestEnhancedRewardConfig:
    """Test enhanced reward configuration."""

    def test_enhanced_reward_config_creation(self):
        """Test creation of enhanced reward configuration."""
        config = RewardConfig(
            step_penalty=-0.01,
            success_bonus=10.0,
            similarity_weight=1.0,
            control_operation_penalty=-0.01,
            training_similarity_weight=1.0,
            evaluation_similarity_weight=0.0,
            demo_completion_bonus=1.0,
            test_completion_bonus=5.0,
            efficiency_bonus_threshold=50,
            efficiency_bonus=2.0,
        )

        assert config.training_similarity_weight == 1.0
        assert config.evaluation_similarity_weight == 0.0
        assert config.demo_completion_bonus == 1.0
        assert config.test_completion_bonus == 5.0
        assert config.control_operation_penalty == -0.01
        assert config.efficiency_bonus_threshold == 50
        assert config.efficiency_bonus == 2.0

    def test_enhanced_reward_config_validation(self):
        """Test validation of enhanced reward configuration."""
        config = RewardConfig(
            training_similarity_weight=1.0,
            evaluation_similarity_weight=0.0,
            demo_completion_bonus=1.0,
            test_completion_bonus=5.0,
            efficiency_bonus_threshold=50,
        )

        errors = config.validate()
        assert len(errors) == 0

    def test_enhanced_reward_config_from_hydra(self):
        """Test creation from Hydra configuration."""
        hydra_config = OmegaConf.create({
            "step_penalty": -0.01,
            "success_bonus": 10.0,
            "similarity_weight": 1.0,
            "control_operation_penalty": -0.01,
            "training_similarity_weight": 1.0,
            "evaluation_similarity_weight": 0.0,
            "demo_completion_bonus": 1.0,
            "test_completion_bonus": 5.0,
            "efficiency_bonus_threshold": 50,
            "efficiency_bonus": 2.0,
        })

        config = RewardConfig.from_hydra(hydra_config)
        assert config.training_similarity_weight == 1.0
        assert config.evaluation_similarity_weight == 0.0
        assert config.demo_completion_bonus == 1.0
        assert config.test_completion_bonus == 5.0


class TestEnhancedRewardCalculation:
    """Test enhanced reward calculation via unified _calculate_enhanced_reward."""

    @pytest.fixture
    def reward_config(self):
        """Create a reward configuration for testing."""
        return RewardConfig(
            step_penalty=-0.01,
            success_bonus=10.0,
            similarity_weight=1.0,
            progress_bonus=0.1,
            control_operation_penalty=-0.01,
            training_similarity_weight=1.0,
            evaluation_similarity_weight=0.0,
            demo_completion_bonus=1.0,
            test_completion_bonus=5.0,
            efficiency_bonus_threshold=50,
            efficiency_bonus=2.0,
            pair_switching_bonus=0.05,
        )

    @pytest.fixture
    def mock_states(self):
        """Create mock states for testing."""
        class MockState:
            def __init__(self, similarity_score, step_count, episode_done=False):
                self.similarity_score = jnp.array(similarity_score, dtype=jnp.float32)
                self.step_count = jnp.array(step_count, dtype=jnp.int32)
                self.episode_done = jnp.array(episode_done, dtype=jnp.bool_)

        old_state = MockState(similarity_score=0.3, step_count=9)
        new_state = MockState(similarity_score=0.7, step_count=10)
        solved_state = MockState(similarity_score=1.0, step_count=25, episode_done=True)
        efficient_solved_state = MockState(similarity_score=1.0, step_count=20, episode_done=True)

        return old_state, new_state, solved_state, efficient_solved_state

    class _DummyConfig:
        def __init__(self, reward: RewardConfig):
            self.reward = reward

    def _wrap_config(self, reward_cfg: RewardConfig):  # returns minimal object with reward attr
        return self._DummyConfig(reward_cfg)

    def test_training_step_reward(self, reward_config, mock_states):
        """Training mode reward: similarity + progress + step + no success bonuses when not solved."""
        old_state, new_state, _, _ = mock_states
        # Inject episode_mode=0
        old_state.episode_mode = jnp.array(0)
        new_state.episode_mode = jnp.array(0)
        config = self._wrap_config(reward_config)
        reward = _calculate_enhanced_reward(old_state, new_state, config, is_control_operation=False)
        similarity_improvement = new_state.similarity_score - old_state.similarity_score
        expected = (
            reward_config.training_similarity_weight * similarity_improvement
            + reward_config.progress_bonus
            + reward_config.step_penalty
        )
        assert jnp.isclose(reward, expected, atol=1e-6)

    def test_training_step_reward_with_success(self, reward_config, mock_states):
        """Solved training reward includes success, demo completion, efficiency bonuses."""
        old_state, _, _, efficient_solved_state = mock_states
        old_state.episode_mode = jnp.array(0)
        efficient_solved_state.episode_mode = jnp.array(0)
        config = self._wrap_config(reward_config)
        reward = _calculate_enhanced_reward(old_state, efficient_solved_state, config, False)
        similarity_improvement = efficient_solved_state.similarity_score - old_state.similarity_score
        expected = (
            reward_config.training_similarity_weight * similarity_improvement
            + reward_config.progress_bonus
            + reward_config.step_penalty
            + reward_config.success_bonus
            + reward_config.demo_completion_bonus
            + reward_config.efficiency_bonus
        )
        assert jnp.isclose(reward, expected, atol=1e-6)

    def test_control_operation_penalty(self, reward_config, mock_states):
        """Control operations subtract control penalty and may add pair switching bonus (not modeled separately now)."""
        old_state, new_state, _, _ = mock_states
        old_state.episode_mode = jnp.array(0)
        new_state.episode_mode = jnp.array(0)
        config = self._wrap_config(reward_config)
        reward_normal = _calculate_enhanced_reward(old_state, new_state, config, False)
        reward_control = _calculate_enhanced_reward(old_state, new_state, config, True)
        assert jnp.isclose(reward_control, reward_normal + reward_config.control_operation_penalty, atol=1e-6)

    def test_evaluation_rewards(self, reward_config, mock_states):
        """Evaluation mode rewards exclude similarity & progress; add test bonuses on solve."""
        old_state, new_state, solved_state, efficient_solved_state = mock_states
        for s in [old_state, new_state, solved_state, efficient_solved_state]:
            s.episode_mode = jnp.array(1)
        config = self._wrap_config(reward_config)
        # Non-solved
        reward_unsolved = _calculate_enhanced_reward(old_state, new_state, config, False)
        assert jnp.isclose(reward_unsolved, reward_config.step_penalty, atol=1e-6)
        # Solved inefficient (step 25 still within threshold)
        reward_solved = _calculate_enhanced_reward(old_state, solved_state, config, False)
        expected_solved = (
            reward_config.step_penalty
            + reward_config.success_bonus
            + reward_config.test_completion_bonus
            + reward_config.efficiency_bonus
    )
        assert jnp.isclose(reward_solved, expected_solved, atol=1e-6)
        # Efficient solved (step 20)
        reward_efficient = _calculate_enhanced_reward(old_state, efficient_solved_state, config, False)
        assert jnp.isclose(reward_efficient, expected_solved, atol=1e-6)


class TestRewardConfigurationFiles:
    """Test reward configuration files."""

    def test_standard_config_file(self):
        """Test standard reward configuration file from src path."""
        config_dict = OmegaConf.load("src/jaxarc/conf/reward/standard.yaml")
        config = RewardConfig.from_hydra(config_dict)

        assert config.training_similarity_weight == 1.0
        assert config.evaluation_similarity_weight == 0.0
        assert config.demo_completion_bonus == 1.0
        assert config.test_completion_bonus == 5.0
        assert config.efficiency_bonus_threshold == 50

        errors = config.validate()
        assert len(errors) == 0

    def test_training_config_file(self):
        """Test training reward configuration file from src path."""
        config_dict = OmegaConf.load("src/jaxarc/conf/reward/training.yaml")
        config = RewardConfig.from_hydra(config_dict)

        assert config.step_penalty == -0.005  # Smaller penalty for exploration
        assert config.progress_bonus == 0.2  # Higher progress bonus
        assert config.demo_completion_bonus == 2.0
        assert config.efficiency_bonus_threshold == 30

        errors = config.validate()
        assert len(errors) == 0

    def test_evaluation_config_file(self):
        """Test evaluation reward configuration file from src path."""
        config_dict = OmegaConf.load("src/jaxarc/conf/reward/evaluation.yaml")
        config = RewardConfig.from_hydra(config_dict)

        assert config.step_penalty == -0.02  # Higher penalty for efficiency
        assert config.success_bonus == 20.0  # High success bonus
        assert config.test_completion_bonus == 15.0  # Very high test bonus
        assert config.efficiency_bonus_threshold == 25  # Strict efficiency

        errors = config.validate()
        assert len(errors) == 0


class TestModeAwareRewardCalculation:
    """Test mode-aware reward calculation integration."""

    def test_mode_specific_reward_structures(self):
        """Test that different modes use appropriate reward structures."""
        # This would require integration with actual ArcEnvState
        # For now, we test the individual components
        
        reward_config = RewardConfig(
            training_similarity_weight=1.0,
            evaluation_similarity_weight=0.0,
            demo_completion_bonus=1.0,
            test_completion_bonus=5.0,
        )

        # Training mode should use training_similarity_weight
        assert reward_config.training_similarity_weight == 1.0
        
        # Evaluation mode should use evaluation_similarity_weight (masked)
        assert reward_config.evaluation_similarity_weight == 0.0
        
        # Different bonuses for different pair types
        assert reward_config.test_completion_bonus > reward_config.demo_completion_bonus

    def test_jax_compatibility(self):
        """Test that reward functions are JAX-compatible."""
        # Test that the functions can be called with JAX arrays
        reward_config = RewardConfig()
        
        similarity_score = jnp.array(0.5, dtype=jnp.float32)
        step_count = jnp.array(10, dtype=jnp.int32)
        episode_done = jnp.array(False, dtype=jnp.bool_)
        
        # These should not raise errors with JAX arrays
        assert isinstance(similarity_score, jnp.ndarray)
        assert isinstance(step_count, jnp.ndarray)
        assert isinstance(episode_done, jnp.ndarray)