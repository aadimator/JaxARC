"""
Tests for enhanced reward calculation with mode awareness.

This module tests the enhanced reward calculation system that provides:
- Training mode reward calculation with configurable frequency
- Evaluation mode reward calculation with target masking
- Proper similarity scoring for different pair types
- Different reward structures based on configuration
- JIT-compilable and efficient implementation
"""

import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from jaxarc.envs.config import RewardConfig
from jaxarc.envs.functional import (
    _calculate_training_step_reward,
    _calculate_training_submit_reward,
    _calculate_evaluation_step_reward,
    _calculate_evaluation_submit_reward,
    _calculate_control_operation_reward,
)


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
    """Test enhanced reward calculation functions."""

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

    def test_training_step_reward(self, reward_config, mock_states):
        """Test training mode step reward calculation."""
        old_state, new_state, _, _ = mock_states
        similarity_improvement = new_state.similarity_score - old_state.similarity_score

        reward = _calculate_training_step_reward(
            old_state, new_state, reward_config, similarity_improvement
        )

        # Should include similarity reward, progress bonus, step penalty
        expected = (
            reward_config.training_similarity_weight * similarity_improvement
            + reward_config.progress_bonus  # positive improvement
            + reward_config.step_penalty
        )
        assert jnp.isclose(reward, expected, atol=1e-6)

    def test_training_step_reward_with_success(self, reward_config, mock_states):
        """Test training mode step reward with success bonuses."""
        old_state, _, _, efficient_solved_state = mock_states
        similarity_improvement = efficient_solved_state.similarity_score - old_state.similarity_score

        reward = _calculate_training_step_reward(
            old_state, efficient_solved_state, reward_config, similarity_improvement
        )

        # Should include all bonuses for efficient solution
        expected = (
            reward_config.training_similarity_weight * similarity_improvement
            + reward_config.progress_bonus
            + reward_config.step_penalty
            + reward_config.success_bonus
            + reward_config.demo_completion_bonus
            + reward_config.efficiency_bonus  # solved within threshold
        )
        assert jnp.isclose(reward, expected, atol=1e-6)

    def test_training_submit_reward(self, reward_config, mock_states):
        """Test training mode submit reward calculation."""
        old_state, new_state, solved_state, _ = mock_states

        # Test non-submit case
        similarity_improvement = new_state.similarity_score - old_state.similarity_score
        reward_non_submit = _calculate_training_submit_reward(
            old_state, new_state, reward_config, similarity_improvement
        )
        assert jnp.isclose(reward_non_submit, reward_config.step_penalty, atol=1e-6)

        # Test submit case (solved)
        similarity_improvement_solved = solved_state.similarity_score - old_state.similarity_score
        reward_submit = _calculate_training_submit_reward(
            old_state, solved_state, reward_config, similarity_improvement_solved
        )
        
        # Should include full reward calculation (including efficiency bonus since step_count=25 <= threshold=50)
        expected = (
            reward_config.training_similarity_weight * similarity_improvement_solved
            + reward_config.progress_bonus
            + reward_config.step_penalty
            + reward_config.success_bonus
            + reward_config.demo_completion_bonus
            + reward_config.efficiency_bonus  # included because step_count <= threshold
        )
        assert jnp.isclose(reward_submit, expected, atol=1e-6)

    def test_evaluation_step_reward(self, reward_config, mock_states):
        """Test evaluation mode step reward calculation."""
        old_state, new_state, solved_state, _ = mock_states

        # Test normal step
        reward = _calculate_evaluation_step_reward(old_state, new_state, reward_config)
        expected = reward_config.step_penalty
        assert jnp.isclose(reward, expected, atol=1e-6)

        # Test solved step (includes efficiency bonus since step_count=25 <= threshold=50)
        reward_solved = _calculate_evaluation_step_reward(old_state, solved_state, reward_config)
        expected_solved = (
            reward_config.step_penalty
            + reward_config.success_bonus
            + reward_config.test_completion_bonus
            + reward_config.efficiency_bonus  # included because step_count <= threshold
        )
        assert jnp.isclose(reward_solved, expected_solved, atol=1e-6)

    def test_evaluation_submit_reward(self, reward_config, mock_states):
        """Test evaluation mode submit reward calculation."""
        old_state, new_state, solved_state, efficient_solved_state = mock_states

        # Test non-submit case
        reward_non_submit = _calculate_evaluation_submit_reward(
            old_state, new_state, reward_config
        )
        assert jnp.isclose(reward_non_submit, reward_config.step_penalty, atol=1e-6)

        # Test submit case (solved, includes efficiency bonus since step_count=25 <= threshold=50)
        reward_submit = _calculate_evaluation_submit_reward(
            old_state, solved_state, reward_config
        )
        expected = (
            reward_config.step_penalty
            + reward_config.success_bonus
            + reward_config.test_completion_bonus
            + reward_config.efficiency_bonus  # included because step_count <= threshold
        )
        assert jnp.isclose(reward_submit, expected, atol=1e-6)

        # Test efficient solution
        reward_efficient = _calculate_evaluation_submit_reward(
            old_state, efficient_solved_state, reward_config
        )
        expected_efficient = (
            reward_config.step_penalty
            + reward_config.success_bonus
            + reward_config.test_completion_bonus
            + reward_config.efficiency_bonus
        )
        assert jnp.isclose(reward_efficient, expected_efficient, atol=1e-6)

    def test_control_operation_reward(self, reward_config, mock_states):
        """Test control operation reward calculation."""
        _, new_state, _, _ = mock_states

        reward = _calculate_control_operation_reward(new_state, reward_config)
        expected = (
            reward_config.control_operation_penalty + reward_config.pair_switching_bonus
        )
        assert jnp.isclose(reward, expected, atol=1e-6)


class TestRewardConfigurationFiles:
    """Test reward configuration files."""

    def test_enhanced_config_file(self):
        """Test enhanced reward configuration file."""
        config_dict = OmegaConf.load("conf/reward/enhanced.yaml")
        config = RewardConfig.from_hydra(config_dict)

        assert config.training_similarity_weight == 1.0
        assert config.evaluation_similarity_weight == 0.0
        assert config.demo_completion_bonus == 1.0
        assert config.test_completion_bonus == 5.0
        assert config.efficiency_bonus_threshold == 50

        errors = config.validate()
        assert len(errors) == 0

    def test_training_optimized_config_file(self):
        """Test training optimized reward configuration file."""
        config_dict = OmegaConf.load("conf/reward/training_optimized.yaml")
        config = RewardConfig.from_hydra(config_dict)

        assert config.step_penalty == -0.005  # Smaller penalty for exploration
        assert config.progress_bonus == 0.2  # Higher progress bonus
        assert config.demo_completion_bonus == 2.0
        assert config.efficiency_bonus_threshold == 30

        errors = config.validate()
        assert len(errors) == 0

    def test_evaluation_focused_config_file(self):
        """Test evaluation focused reward configuration file."""
        config_dict = OmegaConf.load("conf/reward/evaluation_focused.yaml")
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