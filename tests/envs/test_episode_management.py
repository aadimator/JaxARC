"""
Unit tests for episode management system.

This module tests the ArcEpisodeManager class and ArcEpisodeConfig for managing
demonstration and test pair selection, episode lifecycle, and non-parametric
pair control operations.

Test Coverage:
- ArcEpisodeManager pair selection strategies (sequential, random)
- Episode termination criteria and continuation logic
- Mode switching between training and evaluation
- Configuration validation and error handling
- Pair switching control operations (next/prev/first_unsolved)
- Context-aware operation validation
- JAX compatibility of all episode management operations

Requirements Coverage: 2.1, 2.2, 2.3, 5.1, 5.2, 5.3, 5.4
"""

import pytest
import jax
import jax.numpy as jnp
import chex
import equinox as eqx

from jaxarc.envs.episode_manager import ArcEpisodeManager, ArcEpisodeConfig
from jaxarc.state import ArcEnvState, create_arc_env_state
from jaxarc.types import JaxArcTask, ARCLEOperationType
from jaxarc.utils import jax_types


class TestArcEpisodeConfig:
    """Test ArcEpisodeConfig validation and configuration options."""

    def test_default_config_values(self):
        """Test default ArcEpisodeConfig values."""
        config = ArcEpisodeConfig()
        
        # Test mode settings
        assert config.episode_mode == "train"
        
        # Test multi-demonstration settings
        assert config.demo_selection_strategy == "random"
        assert config.allow_demo_switching is True
        assert config.require_all_demos_solved is False
        
        # Test test evaluation settings
        assert config.test_selection_strategy == "sequential"
        assert config.allow_test_switching is False
        assert config.require_all_tests_solved is True
        
        # Test termination criteria
        assert config.terminate_on_first_success is False
        assert config.max_pairs_per_episode == 4
        assert config.success_threshold == 1.0
        
        # Test reward settings
        assert config.training_reward_frequency == "step"
        assert config.evaluation_reward_frequency == "submit"

    def test_config_validation_valid_cases(self):
        """Test validation with valid configuration values."""
        # Test valid training configuration
        train_config = ArcEpisodeConfig(
            episode_mode="train",
            demo_selection_strategy="sequential",
            allow_demo_switching=True,
            max_pairs_per_episode=3,
            success_threshold=0.95,
            training_reward_frequency="submit"
        )
        
        errors = train_config.validate()
        assert len(errors) == 0, f"Valid config should not have errors: {errors}"
        
        # Test valid test configuration
        test_config = ArcEpisodeConfig(
            episode_mode="test",
            test_selection_strategy="random",
            allow_test_switching=True,
            require_all_tests_solved=False,
            max_pairs_per_episode=2,
            success_threshold=1.0
        )
        
        errors = test_config.validate()
        assert len(errors) == 0, f"Valid config should not have errors: {errors}"

    def test_config_validation_invalid_mode(self):
        """Test validation with invalid episode mode."""
        config = ArcEpisodeConfig(episode_mode="invalid")
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("episode_mode must be 'train' or 'test'" in error for error in errors)

    def test_config_validation_invalid_strategies(self):
        """Test validation with invalid selection strategies."""
        # Invalid demo strategy
        config1 = ArcEpisodeConfig(demo_selection_strategy="invalid")
        errors1 = config1.validate()
        assert len(errors1) > 0
        assert any("demo_selection_strategy must be one of" in error for error in errors1)
        
        # Invalid test strategy
        config2 = ArcEpisodeConfig(test_selection_strategy="invalid")
        errors2 = config2.validate()
        assert len(errors2) > 0
        assert any("test_selection_strategy must be one of" in error for error in errors2)

    def test_config_validation_invalid_numeric_fields(self):
        """Test validation with invalid numeric field values."""
        # Invalid max_pairs_per_episode
        config1 = ArcEpisodeConfig(max_pairs_per_episode=0)
        errors1 = config1.validate()
        assert len(errors1) > 0
        assert any("max_pairs_per_episode must be a positive integer" in error for error in errors1)
        
        config2 = ArcEpisodeConfig(max_pairs_per_episode=-1)
        errors2 = config2.validate()
        assert len(errors2) > 0
        
        # Invalid success_threshold
        config3 = ArcEpisodeConfig(success_threshold=-0.1)
        errors3 = config3.validate()
        assert len(errors3) > 0
        assert any("success_threshold must be a float in [0.0, 1.0]" in error for error in errors3)
        
        config4 = ArcEpisodeConfig(success_threshold=1.5)
        errors4 = config4.validate()
        assert len(errors4) > 0

    def test_config_validation_invalid_reward_frequencies(self):
        """Test validation with invalid reward frequency settings."""
        # Invalid training reward frequency
        config1 = ArcEpisodeConfig(training_reward_frequency="invalid")
        errors1 = config1.validate()
        assert len(errors1) > 0
        assert any("training_reward_frequency must be one of" in error for error in errors1)
        
        # Invalid evaluation reward frequency
        config2 = ArcEpisodeConfig(evaluation_reward_frequency="invalid")
        errors2 = config2.validate()
        assert len(errors2) > 0
        assert any("evaluation_reward_frequency must be one of" in error for error in errors2)

    def test_config_from_hydra(self):
        """Test creating config from Hydra dictionary."""
        hydra_cfg = {
            "episode_mode": "test",
            "demo_selection_strategy": "sequential",
            "allow_demo_switching": False,
            "max_pairs_per_episode": 2,
            "success_threshold": 0.9,
            "training_reward_frequency": "submit"
        }
        
        config = ArcEpisodeConfig.from_hydra(hydra_cfg)
        
        assert config.episode_mode == "test"
        assert config.demo_selection_strategy == "sequential"
        assert config.allow_demo_switching is False
        assert config.max_pairs_per_episode == 2
        assert config.success_threshold == 0.9
        assert config.training_reward_frequency == "submit"
        
        # Test with missing keys (should use defaults)
        minimal_cfg = {"episode_mode": "train"}
        minimal_config = ArcEpisodeConfig.from_hydra(minimal_cfg)
        assert minimal_config.episode_mode == "train"
        assert minimal_config.demo_selection_strategy == "random"  # Default


class TestArcEpisodeManagerPairSelection:
    """Test ArcEpisodeManager pair selection strategies."""

    @pytest.fixture
    def sample_task_data(self) -> JaxArcTask:
        """Create sample task data with multiple pairs."""
        max_pairs = 4
        grid_size = 8
        
        return JaxArcTask(
            input_grids_examples=jnp.zeros((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=bool),
            output_grids_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=bool),
            num_train_pairs=3,  # 3 available demo pairs
            test_input_grids=jnp.zeros((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((max_pairs, grid_size, grid_size), dtype=bool),
            true_test_output_grids=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((max_pairs, grid_size, grid_size), dtype=bool),
            num_test_pairs=2,  # 2 available test pairs
            task_index=jnp.array(42, dtype=jnp.int32),
        )

    def test_select_initial_demo_pair_sequential(self, sample_task_data: JaxArcTask):
        """Test sequential demo pair selection."""
        key = jax.random.PRNGKey(42)
        config = ArcEpisodeConfig(
            episode_mode="train",
            demo_selection_strategy="sequential"
        )
        
        pair_idx, success = ArcEpisodeManager.select_initial_pair(key, sample_task_data, config)
        
        # Should select first available pair (index 0)
        assert success
        assert pair_idx == 0
        
        # Test JAX types
        chex.assert_type(pair_idx, jnp.integer)
        chex.assert_type(success, jnp.bool_)

    def test_select_initial_demo_pair_random(self, sample_task_data: JaxArcTask):
        """Test random demo pair selection."""
        config = ArcEpisodeConfig(
            episode_mode="train",
            demo_selection_strategy="random"
        )
        
        # Test multiple random selections
        selected_indices = []
        for i in range(10):
            key = jax.random.PRNGKey(i)
            pair_idx, success = ArcEpisodeManager.select_initial_pair(key, sample_task_data, config)
            
            assert success
            assert 0 <= pair_idx < sample_task_data.num_train_pairs
            selected_indices.append(int(pair_idx))
        
        # Should have some variation in random selection
        unique_selections = len(set(selected_indices))
        assert unique_selections > 1, "Random selection should produce different results"

    def test_select_initial_test_pair_sequential(self, sample_task_data: JaxArcTask):
        """Test sequential test pair selection."""
        key = jax.random.PRNGKey(42)
        config = ArcEpisodeConfig(
            episode_mode="test",
            test_selection_strategy="sequential"
        )
        
        pair_idx, success = ArcEpisodeManager.select_initial_pair(key, sample_task_data, config)
        
        # Should select first available test pair (index 0)
        assert success
        assert pair_idx == 0

    def test_select_initial_test_pair_random(self, sample_task_data: JaxArcTask):
        """Test random test pair selection."""
        config = ArcEpisodeConfig(
            episode_mode="test",
            test_selection_strategy="random"
        )
        
        # Test multiple random selections
        selected_indices = []
        for i in range(10):
            key = jax.random.PRNGKey(i)
            pair_idx, success = ArcEpisodeManager.select_initial_pair(key, sample_task_data, config)
            
            assert success
            assert 0 <= pair_idx < sample_task_data.num_test_pairs
            selected_indices.append(int(pair_idx))
        
        # Should have some variation in random selection
        unique_selections = len(set(selected_indices))
        assert unique_selections > 1, "Random selection should produce different results"

    def test_select_initial_pair_no_available_pairs(self):
        """Test pair selection when no pairs are available."""
        # Create task with no available pairs
        grid_size = 5
        empty_task = JaxArcTask(
            input_grids_examples=jnp.zeros((2, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((2, grid_size, grid_size), dtype=bool),
            output_grids_examples=jnp.ones((2, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((2, grid_size, grid_size), dtype=bool),
            num_train_pairs=0,  # No demo pairs
            test_input_grids=jnp.zeros((2, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((2, grid_size, grid_size), dtype=bool),
            true_test_output_grids=jnp.ones((2, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((2, grid_size, grid_size), dtype=bool),
            num_test_pairs=0,  # No test pairs
            task_index=jnp.array(0, dtype=jnp.int32),
        )
        
        key = jax.random.PRNGKey(42)
        
        # Test demo selection with no available pairs
        demo_config = ArcEpisodeConfig(episode_mode="train")
        demo_idx, demo_success = ArcEpisodeManager.select_initial_pair(key, empty_task, demo_config)
        assert not demo_success
        assert demo_idx == -1
        
        # Test test selection with no available pairs
        test_config = ArcEpisodeConfig(episode_mode="test")
        test_idx, test_success = ArcEpisodeManager.select_initial_pair(key, empty_task, test_config)
        assert not test_success
        assert test_idx == -1

    def test_pair_selection_jax_compatibility(self, sample_task_data: JaxArcTask):
        """Test JAX compatibility of pair selection methods."""
        config = ArcEpisodeConfig(episode_mode="train", demo_selection_strategy="random")
        
        @jax.jit
        def select_pair_jit(key):
            return ArcEpisodeManager.select_initial_pair(key, sample_task_data, config)
        
        # Should compile and execute without errors
        key = jax.random.PRNGKey(42)
        pair_idx, success = select_pair_jit(key)
        
        assert success
        assert 0 <= pair_idx < sample_task_data.num_train_pairs


class TestArcEpisodeManagerTermination:
    """Test episode termination criteria and continuation logic."""

    @pytest.fixture
    def sample_state(self) -> ArcEnvState:
        """Create sample state for termination testing."""
        grid_size = 6
        max_train_pairs = 4
        max_test_pairs = 3
        
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((max_train_pairs, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=bool),
            output_grids_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=bool),
            num_train_pairs=3,
            test_input_grids=jnp.zeros((max_test_pairs, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=bool),
            true_test_output_grids=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=bool),
            num_test_pairs=2,
            task_index=jnp.array(123, dtype=jnp.int32),
        )
        
        working_grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        working_grid_mask = jnp.ones((grid_size, grid_size), dtype=bool)
        target_grid = jnp.ones((grid_size, grid_size), dtype=jnp.int32)
        
        state = create_arc_env_state(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            max_train_pairs=max_train_pairs,
            max_test_pairs=max_test_pairs,
            episode_mode=jax_types.EPISODE_MODE_TRAIN,
        )
        
        # Fix available pairs to match task data
        correct_demo_pairs = task_data.get_available_demo_pairs()
        correct_test_pairs = task_data.get_available_test_pairs()
        
        return eqx.tree_at(
            lambda s: (s.available_demo_pairs, s.available_test_pairs),
            state,
            (correct_demo_pairs, correct_test_pairs)
        )

    def test_should_continue_episode_basic(self, sample_state: ArcEnvState):
        """Test basic episode continuation logic."""
        config = ArcEpisodeConfig()
        
        # Episode should continue by default
        should_continue = ArcEpisodeManager.should_continue_episode(sample_state, config)
        assert should_continue
        
        # Episode should not continue if marked as done
        done_state = eqx.tree_at(lambda s: s.episode_done, sample_state, jnp.array(True))
        should_continue_done = ArcEpisodeManager.should_continue_episode(done_state, config)
        assert not should_continue_done

    def test_should_continue_terminate_on_first_success(self, sample_state: ArcEnvState):
        """Test termination on first success."""
        config = ArcEpisodeConfig(
            terminate_on_first_success=True,
            success_threshold=0.9
        )
        
        # Should continue with low similarity score
        low_score_state = eqx.tree_at(
            lambda s: s.similarity_score,
            sample_state,
            jnp.array(0.5)
        )
        should_continue = ArcEpisodeManager.should_continue_episode(low_score_state, config)
        assert should_continue
        
        # Should terminate with high similarity score
        high_score_state = eqx.tree_at(
            lambda s: s.similarity_score,
            sample_state,
            jnp.array(0.95)
        )
        should_continue = ArcEpisodeManager.should_continue_episode(high_score_state, config)
        assert not should_continue

    def test_should_continue_require_all_demos_solved(self, sample_state: ArcEnvState):
        """Test termination when all demos must be solved."""
        config = ArcEpisodeConfig(require_all_demos_solved=True)
        
        # Should continue with some demos unsolved
        partial_completion = jnp.array([True, False, True, False], dtype=bool)  # 2 of 4 completed
        partial_state = eqx.tree_at(
            lambda s: s.demo_completion_status,
            sample_state,
            partial_completion
        )
        should_continue = ArcEpisodeManager.should_continue_episode(partial_state, config)
        assert should_continue
        
        # Should terminate when all available demos are solved
        # Available demos: [True, True, True, False] (3 available)
        # Completed demos: [True, True, True, False] (all available completed)
        all_available_completed = jnp.array([True, True, True, False], dtype=bool)
        available_demos = jnp.array([True, True, True, False], dtype=bool)
        
        complete_state = eqx.tree_at(
            lambda s: (s.demo_completion_status, s.available_demo_pairs),
            sample_state,
            (all_available_completed, available_demos)
        )
        should_continue = ArcEpisodeManager.should_continue_episode(complete_state, config)
        assert not should_continue

    def test_should_continue_require_all_tests_solved(self, sample_state: ArcEnvState):
        """Test termination when all tests must be solved in test mode."""
        # Switch to test mode
        test_state = eqx.tree_at(
            lambda s: s.episode_mode,
            sample_state,
            jnp.array(jax_types.EPISODE_MODE_TEST)
        )
        
        config = ArcEpisodeConfig(
            episode_mode="test",
            require_all_tests_solved=True
        )
        
        # Should continue with some tests unsolved
        partial_completion = jnp.array([True, False, False], dtype=bool)  # 1 of 3 completed
        partial_state = eqx.tree_at(
            lambda s: s.test_completion_status,
            test_state,
            partial_completion
        )
        should_continue = ArcEpisodeManager.should_continue_episode(partial_state, config)
        assert should_continue
        
        # Should terminate when all available tests are solved
        # Available tests: [True, True, False] (2 available)
        # Completed tests: [True, True, False] (all available completed)
        all_available_completed = jnp.array([True, True, False], dtype=bool)
        available_tests = jnp.array([True, True, False], dtype=bool)
        
        complete_state = eqx.tree_at(
            lambda s: (s.test_completion_status, s.available_test_pairs),
            partial_state,
            (all_available_completed, available_tests)
        )
        should_continue = ArcEpisodeManager.should_continue_episode(complete_state, config)
        assert not should_continue

    def test_should_continue_max_pairs_limit(self, sample_state: ArcEnvState):
        """Test termination based on maximum pairs per episode."""
        config = ArcEpisodeConfig(max_pairs_per_episode=2)
        
        # Should continue with fewer completed pairs
        one_completed = jnp.array([True, False, False, False], dtype=bool)
        partial_state = eqx.tree_at(
            lambda s: s.demo_completion_status,
            sample_state,
            one_completed
        )
        should_continue = ArcEpisodeManager.should_continue_episode(partial_state, config)
        assert should_continue
        
        # Should terminate when max pairs reached
        two_completed = jnp.array([True, True, False, False], dtype=bool)
        max_state = eqx.tree_at(
            lambda s: s.demo_completion_status,
            sample_state,
            two_completed
        )
        should_continue = ArcEpisodeManager.should_continue_episode(max_state, config)
        assert not should_continue

    def test_should_continue_jax_compatibility(self, sample_state: ArcEnvState):
        """Test JAX compatibility of episode continuation logic."""
        config = ArcEpisodeConfig()
        
        @jax.jit
        def should_continue_jit(state):
            return ArcEpisodeManager.should_continue_episode(state, config)
        
        # Should compile and execute without errors
        result = should_continue_jit(sample_state)
        assert result
        
        # Test with different states
        done_state = eqx.tree_at(lambda s: s.episode_done, sample_state, jnp.array(True))
        result_done = should_continue_jit(done_state)
        assert not result_done


class TestArcEpisodeManagerPairSwitching:
    """Test pair switching control operations."""

    @pytest.fixture
    def sample_state(self) -> ArcEnvState:
        """Create sample state for pair switching testing."""
        grid_size = 5
        max_train_pairs = 4
        max_test_pairs = 3
        
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((max_train_pairs, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=bool),
            output_grids_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=bool),
            num_train_pairs=3,
            test_input_grids=jnp.zeros((max_test_pairs, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=bool),
            true_test_output_grids=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=bool),
            num_test_pairs=2,
            task_index=jnp.array(456, dtype=jnp.int32),
        )
        
        working_grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        working_grid_mask = jnp.ones((grid_size, grid_size), dtype=bool)
        target_grid = jnp.ones((grid_size, grid_size), dtype=jnp.int32)
        
        state = create_arc_env_state(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            max_train_pairs=max_train_pairs,
            max_test_pairs=max_test_pairs,
            episode_mode=jax_types.EPISODE_MODE_TRAIN,
            current_example_idx=0,  # Start at first pair
        )
        
        # Fix available pairs to match task data
        correct_demo_pairs = task_data.get_available_demo_pairs()
        correct_test_pairs = task_data.get_available_test_pairs()
        
        return eqx.tree_at(
            lambda s: (s.available_demo_pairs, s.available_test_pairs),
            state,
            (correct_demo_pairs, correct_test_pairs)
        )

    def test_switch_to_next_demo_pair(self, sample_state: ArcEnvState):
        """Test switching to next demo pair."""
        config = ArcEpisodeConfig(allow_demo_switching=True)
        
        # Switch from pair 0 to next available pair
        new_state = ArcEpisodeManager.execute_pair_control_operation(
            sample_state,
            ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR,
            config
        )
        
        # Should switch to next pair (index 1)
        assert new_state.current_example_idx == 1
        
        # Test circular switching (from last pair back to first)
        last_pair_state = eqx.tree_at(
            lambda s: s.current_example_idx,
            sample_state,
            jnp.array(2)  # Last available pair (index 2)
        )
        
        circular_state = ArcEpisodeManager.execute_pair_control_operation(
            last_pair_state,
            ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR,
            config
        )
        
        # Should wrap around to first available pair (index 0)
        # Note: With 3 available pairs (0,1,2), from index 2 next should be 0
        assert circular_state.current_example_idx == 0

    def test_switch_to_prev_demo_pair(self, sample_state: ArcEnvState):
        """Test switching to previous demo pair."""
        config = ArcEpisodeConfig(allow_demo_switching=True)
        
        # Start from pair 1 and switch to previous
        middle_pair_state = eqx.tree_at(
            lambda s: s.current_example_idx,
            sample_state,
            jnp.array(1)
        )
        
        new_state = ArcEpisodeManager.execute_pair_control_operation(
            middle_pair_state,
            ARCLEOperationType.SWITCH_TO_PREV_DEMO_PAIR,
            config
        )
        
        # Should switch to previous pair (index 0)
        assert new_state.current_example_idx == 0
        
        # Test circular switching (from first pair back to last)
        circular_state = ArcEpisodeManager.execute_pair_control_operation(
            sample_state,  # Starting at index 0
            ARCLEOperationType.SWITCH_TO_PREV_DEMO_PAIR,
            config
        )
        
        # Should wrap around to last available pair (index 2)
        # Note: With 3 available pairs (0,1,2), from index 0 prev should be 2
        assert circular_state.current_example_idx == 2

    def test_switch_to_first_unsolved_demo(self, sample_state: ArcEnvState):
        """Test switching to first unsolved demo pair."""
        config = ArcEpisodeConfig(allow_demo_switching=True)
        
        # Mark first two pairs as completed
        completed_status = jnp.array([True, True, False, False], dtype=bool)
        partial_state = eqx.tree_at(
            lambda s: s.demo_completion_status,
            sample_state,
            completed_status
        )
        
        new_state = ArcEpisodeManager.execute_pair_control_operation(
            partial_state,
            ARCLEOperationType.SWITCH_TO_FIRST_UNSOLVED_DEMO,
            config
        )
        
        # Should switch to first unsolved pair (index 2)
        assert new_state.current_example_idx == 2
        
        # Test when all pairs are solved - should stay at current
        all_completed = jnp.array([True, True, True, False], dtype=bool)  # Only 3 available
        all_solved_state = eqx.tree_at(
            lambda s: s.demo_completion_status,
            sample_state,
            all_completed
        )
        
        no_change_state = ArcEpisodeManager.execute_pair_control_operation(
            all_solved_state,
            ARCLEOperationType.SWITCH_TO_FIRST_UNSOLVED_DEMO,
            config
        )
        
        # Should stay at current pair when no unsolved pairs
        assert no_change_state.current_example_idx == all_solved_state.current_example_idx

    def test_switch_to_next_test_pair(self, sample_state: ArcEnvState):
        """Test switching to next test pair."""
        # Switch to test mode
        test_state = eqx.tree_at(
            lambda s: s.episode_mode,
            sample_state,
            jnp.array(jax_types.EPISODE_MODE_TEST)
        )
        
        config = ArcEpisodeConfig(
            episode_mode="test",
            allow_test_switching=True
        )
        
        # Switch from pair 0 to next available pair
        new_state = ArcEpisodeManager.execute_pair_control_operation(
            test_state,
            ARCLEOperationType.SWITCH_TO_NEXT_TEST_PAIR,
            config
        )
        
        # Should switch to next test pair (index 1)
        assert new_state.current_example_idx == 1

    def test_switch_to_prev_test_pair(self, sample_state: ArcEnvState):
        """Test switching to previous test pair."""
        # Switch to test mode and start at pair 1
        test_state = eqx.tree_at(
            lambda s: (s.episode_mode, s.current_example_idx),
            sample_state,
            (jnp.array(jax_types.EPISODE_MODE_TEST), jnp.array(1))
        )
        
        config = ArcEpisodeConfig(
            episode_mode="test",
            allow_test_switching=True
        )
        
        new_state = ArcEpisodeManager.execute_pair_control_operation(
            test_state,
            ARCLEOperationType.SWITCH_TO_PREV_TEST_PAIR,
            config
        )
        
        # Should switch to previous test pair (index 0)
        assert new_state.current_example_idx == 0

    def test_switch_to_first_unsolved_test(self, sample_state: ArcEnvState):
        """Test switching to first unsolved test pair."""
        # Switch to test mode
        test_state = eqx.tree_at(
            lambda s: s.episode_mode,
            sample_state,
            jnp.array(jax_types.EPISODE_MODE_TEST)
        )
        
        config = ArcEpisodeConfig(
            episode_mode="test",
            allow_test_switching=True
        )
        
        # Mark first test pair as completed
        completed_status = jnp.array([True, False, False], dtype=bool)
        partial_state = eqx.tree_at(
            lambda s: s.test_completion_status,
            test_state,
            completed_status
        )
        
        new_state = ArcEpisodeManager.execute_pair_control_operation(
            partial_state,
            ARCLEOperationType.SWITCH_TO_FIRST_UNSOLVED_TEST,
            config
        )
        
        # Should switch to first unsolved test pair (index 1)
        assert new_state.current_example_idx == 1

    def test_reset_current_pair(self, sample_state: ArcEnvState):
        """Test resetting current pair to initial state."""
        config = ArcEpisodeConfig()
        
        # Modify state to simulate some work done
        modified_state = eqx.tree_at(
            lambda s: (s.similarity_score, s.step_count),
            sample_state,
            (jnp.array(0.5), jnp.array(10))
        )
        
        reset_state = ArcEpisodeManager.execute_pair_control_operation(
            modified_state,
            ARCLEOperationType.RESET_CURRENT_PAIR,
            config
        )
        
        # Should reset similarity score but preserve step count and other episode state
        assert reset_state.similarity_score == 0.0
        assert reset_state.step_count == 10  # Step count should be preserved
        assert reset_state.current_example_idx == modified_state.current_example_idx

    def test_pair_switching_disabled(self, sample_state: ArcEnvState):
        """Test pair switching when disabled in configuration."""
        config = ArcEpisodeConfig(allow_demo_switching=False)
        
        # Attempt to switch when disabled
        new_state = ArcEpisodeManager.execute_pair_control_operation(
            sample_state,
            ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR,
            config
        )
        
        # Should remain at same pair
        assert new_state.current_example_idx == sample_state.current_example_idx

    def test_pair_switching_wrong_mode(self, sample_state: ArcEnvState):
        """Test pair switching operations in wrong mode."""
        config = ArcEpisodeConfig(allow_demo_switching=True)
        
        # Switch to test mode
        test_state = eqx.tree_at(
            lambda s: s.episode_mode,
            sample_state,
            jnp.array(jax_types.EPISODE_MODE_TEST)
        )
        
        # Attempt demo operation in test mode
        new_state = ArcEpisodeManager.execute_pair_control_operation(
            test_state,
            ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR,
            config
        )
        
        # Should remain unchanged
        assert new_state.current_example_idx == test_state.current_example_idx

    def test_pair_switching_jax_compatibility(self, sample_state: ArcEnvState):
        """Test JAX compatibility of pair switching operations."""
        config = ArcEpisodeConfig(allow_demo_switching=True)
        
        @jax.jit
        def switch_pair_jit(state, operation_id):
            return ArcEpisodeManager.execute_pair_control_operation(state, operation_id, config)
        
        # Should compile and execute without errors
        new_state = switch_pair_jit(sample_state, ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR)
        assert new_state.current_example_idx == 1


class TestArcEpisodeManagerValidation:
    """Test operation validation and error handling."""

    @pytest.fixture
    def sample_state(self) -> ArcEnvState:
        """Create sample state for validation testing."""
        grid_size = 4
        
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((3, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((3, grid_size, grid_size), dtype=bool),
            output_grids_examples=jnp.ones((3, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((3, grid_size, grid_size), dtype=bool),
            num_train_pairs=2,
            test_input_grids=jnp.zeros((2, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((2, grid_size, grid_size), dtype=bool),
            true_test_output_grids=jnp.ones((2, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((2, grid_size, grid_size), dtype=bool),
            num_test_pairs=2,
            task_index=jnp.array(789, dtype=jnp.int32),
        )
        
        working_grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        working_grid_mask = jnp.ones((grid_size, grid_size), dtype=bool)
        target_grid = jnp.ones((grid_size, grid_size), dtype=jnp.int32)
        
        state = create_arc_env_state(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            max_train_pairs=3,
            max_test_pairs=2,
            episode_mode=jax_types.EPISODE_MODE_TRAIN,
        )
        
        # Fix available pairs to match task data
        correct_demo_pairs = task_data.get_available_demo_pairs()
        correct_test_pairs = task_data.get_available_test_pairs()
        
        return eqx.tree_at(
            lambda s: (s.available_demo_pairs, s.available_test_pairs),
            state,
            (correct_demo_pairs, correct_test_pairs)
        )

    def test_validate_demo_operations_in_train_mode(self, sample_state: ArcEnvState):
        """Test validation of demo operations in training mode."""
        config = ArcEpisodeConfig(allow_demo_switching=True)
        
        # Should be valid in training mode
        is_valid, error = ArcEpisodeManager.validate_pair_control_operation(
            sample_state,
            ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR,
            config
        )
        
        assert is_valid
        assert error is None

    def test_validate_demo_operations_in_test_mode(self, sample_state: ArcEnvState):
        """Test validation of demo operations in test mode."""
        # Switch to test mode
        test_state = eqx.tree_at(
            lambda s: s.episode_mode,
            sample_state,
            jnp.array(jax_types.EPISODE_MODE_TEST)
        )
        
        config = ArcEpisodeConfig(allow_demo_switching=True)
        
        # Should be invalid in test mode
        is_valid, error = ArcEpisodeManager.validate_pair_control_operation(
            test_state,
            ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR,
            config
        )
        
        assert not is_valid
        # Note: Error messages are not available in JAX-compatible validation

    def test_validate_test_operations_in_test_mode(self, sample_state: ArcEnvState):
        """Test validation of test operations in test mode."""
        # Switch to test mode
        test_state = eqx.tree_at(
            lambda s: s.episode_mode,
            sample_state,
            jnp.array(jax_types.EPISODE_MODE_TEST)
        )
        
        config = ArcEpisodeConfig(
            episode_mode="test",
            allow_test_switching=True
        )
        
        # Should be valid in test mode
        is_valid, error = ArcEpisodeManager.validate_pair_control_operation(
            test_state,
            ARCLEOperationType.SWITCH_TO_NEXT_TEST_PAIR,
            config
        )
        
        assert is_valid
        assert error is None

    def test_validate_test_operations_in_train_mode(self, sample_state: ArcEnvState):
        """Test validation of test operations in training mode."""
        config = ArcEpisodeConfig(allow_test_switching=True)
        
        # Should be invalid in training mode
        is_valid, error = ArcEpisodeManager.validate_pair_control_operation(
            sample_state,
            ARCLEOperationType.SWITCH_TO_NEXT_TEST_PAIR,
            config
        )
        
        assert not is_valid
        # Note: Error messages are not available in JAX-compatible validation

    def test_validate_switching_disabled(self, sample_state: ArcEnvState):
        """Test validation when switching is disabled."""
        config = ArcEpisodeConfig(allow_demo_switching=False)
        
        # Should be invalid when switching is disabled
        is_valid, error = ArcEpisodeManager.validate_pair_control_operation(
            sample_state,
            ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR,
            config
        )
        
        assert not is_valid
        # Note: Error messages are not available in JAX-compatible validation

    def test_validate_insufficient_pairs(self, sample_state: ArcEnvState):
        """Test validation when insufficient pairs are available."""
        # Create state with only one available demo pair
        single_pair_available = jnp.array([True, False, False], dtype=bool)
        single_pair_state = eqx.tree_at(
            lambda s: s.available_demo_pairs,
            sample_state,
            single_pair_available
        )
        
        config = ArcEpisodeConfig(allow_demo_switching=True)
        
        # Should be invalid with only one pair
        is_valid, error = ArcEpisodeManager.validate_pair_control_operation(
            single_pair_state,
            ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR,
            config
        )
        
        assert not is_valid
        # Note: Error messages are not available in JAX-compatible validation

    def test_validate_reset_operation(self, sample_state: ArcEnvState):
        """Test validation of reset operation."""
        config = ArcEpisodeConfig()
        
        # Reset should generally be valid
        is_valid, error = ArcEpisodeManager.validate_pair_control_operation(
            sample_state,
            ARCLEOperationType.RESET_CURRENT_PAIR,
            config
        )
        
        assert is_valid
        assert error is None

    def test_validate_unknown_operation(self, sample_state: ArcEnvState):
        """Test validation of unknown operation."""
        config = ArcEpisodeConfig()
        
        # Should be invalid for unknown operation
        is_valid, error = ArcEpisodeManager.validate_pair_control_operation(
            sample_state,
            999,  # Unknown operation ID
            config
        )
        
        assert not is_valid
        # Note: Error messages are not available in JAX-compatible validation

    def test_validation_jax_compatibility(self, sample_state: ArcEnvState):
        """Test JAX compatibility of validation functions."""
        config = ArcEpisodeConfig(allow_demo_switching=True)
        
        @jax.jit
        def validate_operation_jit(state, operation_id):
            is_valid, _ = ArcEpisodeManager.validate_pair_control_operation(
                state, operation_id, config
            )
            return is_valid
        
        # Should compile and execute without errors
        result = validate_operation_jit(sample_state, ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR)
        assert result


class TestArcEpisodeManagerIntegration:
    """Test integration scenarios and edge cases."""

    @pytest.fixture
    def complex_state(self) -> ArcEnvState:
        """Create complex state for integration testing."""
        grid_size = 6
        max_train_pairs = 5
        max_test_pairs = 4
        
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((max_train_pairs, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=bool),
            output_grids_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=bool),
            num_train_pairs=4,
            test_input_grids=jnp.zeros((max_test_pairs, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=bool),
            true_test_output_grids=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=bool),
            num_test_pairs=3,
            task_index=jnp.array(999, dtype=jnp.int32),
        )
        
        working_grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        working_grid_mask = jnp.ones((grid_size, grid_size), dtype=bool)
        target_grid = jnp.ones((grid_size, grid_size), dtype=jnp.int32)
        
        state = create_arc_env_state(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            max_train_pairs=max_train_pairs,
            max_test_pairs=max_test_pairs,
            episode_mode=jax_types.EPISODE_MODE_TRAIN,
            current_example_idx=1,  # Start at middle pair
        )
        
        # Fix available pairs to match task data
        correct_demo_pairs = task_data.get_available_demo_pairs()
        correct_test_pairs = task_data.get_available_test_pairs()
        
        return eqx.tree_at(
            lambda s: (s.available_demo_pairs, s.available_test_pairs),
            state,
            (correct_demo_pairs, correct_test_pairs)
        )

    def test_complete_episode_workflow(self, complex_state: ArcEnvState):
        """Test complete episode workflow with multiple operations."""
        config = ArcEpisodeConfig(
            allow_demo_switching=True,
            terminate_on_first_success=False,
            max_pairs_per_episode=3
        )
        
        current_state = complex_state
        
        # 1. Switch to next pair
        current_state = ArcEpisodeManager.execute_pair_control_operation(
            current_state,
            ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR,
            config
        )
        assert current_state.current_example_idx == 2
        
        # 2. Mark current pair as completed
        completed_status = current_state.demo_completion_status.at[2].set(True)
        current_state = eqx.tree_at(
            lambda s: s.demo_completion_status,
            current_state,
            completed_status
        )
        
        # 3. Switch to first unsolved
        current_state = ArcEpisodeManager.execute_pair_control_operation(
            current_state,
            ARCLEOperationType.SWITCH_TO_FIRST_UNSOLVED_DEMO,
            config
        )
        assert current_state.current_example_idx == 0  # First unsolved pair
        
        # 4. Check episode continuation
        should_continue = ArcEpisodeManager.should_continue_episode(current_state, config)
        assert should_continue  # Should continue with only 1 completed pair

    def test_mode_switching_workflow(self, complex_state: ArcEnvState):
        """Test workflow involving mode switching."""
        # Start in training mode
        train_config = ArcEpisodeConfig(
            episode_mode="train",
            allow_demo_switching=True
        )
        
        # Work on demo pairs
        demo_state = ArcEpisodeManager.execute_pair_control_operation(
            complex_state,
            ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR,
            train_config
        )
        assert demo_state.current_example_idx == 2
        
        # Switch to test mode
        test_state = eqx.tree_at(
            lambda s: s.episode_mode,
            demo_state,
            jnp.array(jax_types.EPISODE_MODE_TEST)
        )
        
        test_config = ArcEpisodeConfig(
            episode_mode="test",
            allow_test_switching=True
        )
        
        # Work on test pairs
        final_state = ArcEpisodeManager.execute_pair_control_operation(
            test_state,
            ARCLEOperationType.SWITCH_TO_NEXT_TEST_PAIR,
            test_config
        )
        
        # Should switch to next test pair (from current index)
        assert final_state.current_example_idx != test_state.current_example_idx

    def test_edge_case_single_pair_available(self):
        """Test edge case with only single pair available."""
        grid_size = 4
        
        # Create task with only one demo pair
        single_pair_task = JaxArcTask(
            input_grids_examples=jnp.zeros((2, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((2, grid_size, grid_size), dtype=bool),
            output_grids_examples=jnp.ones((2, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((2, grid_size, grid_size), dtype=bool),
            num_train_pairs=1,  # Only one demo pair
            test_input_grids=jnp.zeros((1, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, grid_size, grid_size), dtype=bool),
            true_test_output_grids=jnp.ones((1, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, grid_size, grid_size), dtype=bool),
            num_test_pairs=1,  # Only one test pair
            task_index=jnp.array(0, dtype=jnp.int32),
        )
        
        working_grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        working_grid_mask = jnp.ones((grid_size, grid_size), dtype=bool)
        target_grid = jnp.ones((grid_size, grid_size), dtype=jnp.int32)
        
        single_state = create_arc_env_state(
            task_data=single_pair_task,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            max_train_pairs=2,
            max_test_pairs=1,
        )
        
        # Fix available pairs to match task data
        correct_demo_pairs = single_pair_task.get_available_demo_pairs()
        correct_test_pairs = single_pair_task.get_available_test_pairs()
        
        single_state = eqx.tree_at(
            lambda s: (s.available_demo_pairs, s.available_test_pairs),
            single_state,
            (correct_demo_pairs, correct_test_pairs)
        )
        
        config = ArcEpisodeConfig(allow_demo_switching=True)
        
        # Switching should not change anything with single pair
        new_state = ArcEpisodeManager.execute_pair_control_operation(
            single_state,
            ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR,
            config
        )
        
        # Should remain at same pair
        assert new_state.current_example_idx == single_state.current_example_idx

    def test_comprehensive_jax_transformations(self, complex_state: ArcEnvState):
        """Test comprehensive JAX transformations on episode management."""
        config = ArcEpisodeConfig(allow_demo_switching=True)
        
        # Test JIT compilation of all major functions
        @jax.jit
        def episode_management_pipeline(state, key):
            # Select initial pair
            pair_idx, success = ArcEpisodeManager.select_initial_pair(
                key, state.task_data, config
            )
            
            # Check continuation
            should_continue = ArcEpisodeManager.should_continue_episode(state, config)
            
            # Execute pair operation
            new_state = ArcEpisodeManager.execute_pair_control_operation(
                state, ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR, config
            )
            
            # Validate operation
            is_valid, _ = ArcEpisodeManager.validate_pair_control_operation(
                state, ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR, config
            )
            
            return {
                'pair_idx': pair_idx,
                'success': success,
                'should_continue': should_continue,
                'new_current_idx': new_state.current_example_idx,
                'is_valid': is_valid
            }
        
        # Should compile and execute without errors
        key = jax.random.PRNGKey(42)
        result = episode_management_pipeline(complex_state, key)
        
        # Verify results
        assert result['success']
        assert result['should_continue']
        assert result['new_current_idx'] == 2  # Next pair from index 1
        assert result['is_valid']


if __name__ == "__main__":
    pytest.main([__file__])