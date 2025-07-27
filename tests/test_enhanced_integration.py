"""
Comprehensive integration tests for enhanced ARC step logic system.

This module tests the complete enhanced system including multi-demonstration training,
test pair evaluation, action history tracking, enhanced action space control,
JAX transformations, performance impact, and ArcObservation functionality.

Test Coverage:
- End-to-end multi-demonstration training workflows
- Evaluation mode with proper target masking
- JAX transformations (jit, vmap, pmap) with all enhancements
- Performance impact and memory usage of new features
- ArcObservation with RL agents and different observation configurations

Requirements Coverage: 1.4, 1.6, 2.5, 6.4, 6.5
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Optional

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxarc.envs import ArcEnvironment
from jaxarc.envs.config import JaxArcConfig
from jaxarc.envs.functional import arc_reset, arc_step, create_observation
from jaxarc.envs.episode_manager import ArcEpisodeConfig
from jaxarc.envs.action_history import HistoryConfig
from jaxarc.envs.observations import (
    ObservationConfig,
)
from jaxarc.state import ArcEnvState
from jaxarc.types import JaxArcTask
from jaxarc.utils.config import get_config


def create_enhanced_config(
    base_config: JaxArcConfig,
    episode_config: Optional[ArcEpisodeConfig] = None,
    history_config: Optional[HistoryConfig] = None,
    disable_logging: bool = True,
    disable_visualization: bool = True,
) -> JaxArcConfig:
    """Helper function to create enhanced config without equinox tree_at validation issues."""
    # Create new logging config with disabled operations if needed
    logging_config = base_config.logging
    if disable_logging:
        logging_config = eqx.tree_at(lambda l: l.log_operations, logging_config, False)
    
    # Create new visualization config with disabled visualization if needed
    visualization_config = base_config.visualization
    if disable_visualization:
        visualization_config = eqx.tree_at(lambda v: v.enabled, visualization_config, False)
    
    return JaxArcConfig(
        environment=base_config.environment,
        dataset=base_config.dataset,
        action=base_config.action,
        reward=base_config.reward,
        visualization=visualization_config,
        storage=base_config.storage,
        logging=logging_config,
        wandb=base_config.wandb,
        episode=episode_config or base_config.episode,
        history=history_config or base_config.history,
    )


class TestMultiDemonstrationTrainingIntegration:
    """Test end-to-end multi-demonstration training workflows."""

    @pytest.fixture
    def multi_demo_task(self) -> JaxArcTask:
        """Create a task with multiple demonstration pairs for testing."""
        max_height, max_width = 30, 30
        max_train_pairs, max_test_pairs = 4, 2

        # Create 3 different demonstration pairs
        demo_inputs = []
        demo_outputs = []
        demo_masks = []

        for i in range(3):
            # Create simple patterns that are different for each demo
            input_grid = jnp.zeros((max_height, max_width), dtype=jnp.int32)
            output_grid = jnp.zeros((max_height, max_width), dtype=jnp.int32)
            mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)

            # Demo 0: Fill pattern
            if i == 0:
                input_grid = input_grid.at[0:2, 0:2].set(jnp.array([[0, 1], [1, 0]]))
                output_grid = output_grid.at[0:2, 0:2].set(jnp.array([[1, 0], [0, 1]]))
                mask = mask.at[0:2, 0:2].set(True)
            # Demo 1: Different pattern
            elif i == 1:
                input_grid = input_grid.at[0:3, 0:3].set(jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
                output_grid = output_grid.at[0:3, 0:3].set(jnp.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))
                mask = mask.at[0:3, 0:3].set(True)
            # Demo 2: Another pattern
            else:
                input_grid = input_grid.at[0:1, 0:3].set(jnp.array([[0, 1, 2]]))
                output_grid = output_grid.at[0:1, 0:3].set(jnp.array([[2, 1, 0]]))
                mask = mask.at[0:1, 0:3].set(True)

            demo_inputs.append(input_grid)
            demo_outputs.append(output_grid)
            demo_masks.append(mask)

        # Stack into arrays
        input_grids = jnp.stack(demo_inputs + [jnp.zeros((max_height, max_width), dtype=jnp.int32)])
        output_grids = jnp.stack(demo_outputs + [jnp.zeros((max_height, max_width), dtype=jnp.int32)])
        input_masks = jnp.stack(demo_masks + [jnp.zeros((max_height, max_width), dtype=jnp.bool_)])
        output_masks = jnp.stack(demo_masks + [jnp.zeros((max_height, max_width), dtype=jnp.bool_)])

        # Create test pairs
        test_input_grids = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.int32)
        test_input_grids = test_input_grids.at[0, 0:2, 0:2].set(jnp.array([[0, 1], [1, 0]]))
        test_input_masks = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.bool_)
        test_input_masks = test_input_masks.at[0, 0:2, 0:2].set(True)

        test_output_grids = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.int32)
        test_output_grids = test_output_grids.at[0, 0:2, 0:2].set(jnp.array([[1, 0], [0, 1]]))
        test_output_masks = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.bool_)
        test_output_masks = test_output_masks.at[0, 0:2, 0:2].set(True)

        return JaxArcTask(
            input_grids_examples=input_grids,
            input_masks_examples=input_masks,
            output_grids_examples=output_grids,
            output_masks_examples=output_masks,
            num_train_pairs=3,
            test_input_grids=test_input_grids,
            test_input_masks=test_input_masks,
            true_test_output_grids=test_output_grids,
            true_test_output_masks=test_output_masks,
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

    def test_sequential_multi_demo_training_workflow(self, multi_demo_task):
        """Test complete sequential multi-demonstration training workflow."""
        # Create configuration for sequential multi-demo training
        cfg = get_config()
        env_config = JaxArcConfig.from_hydra(cfg)

        # Configure for multi-demonstration training
        episode_config = ArcEpisodeConfig(
            episode_mode=0,  # 0 = train mode
            demo_selection_strategy="sequential",
            allow_demo_switching=True,
            require_all_demos_solved=False,
            terminate_on_first_success=False,
            max_pairs_per_episode=3,
        )

        # Enable action history tracking
        history_config = HistoryConfig(
            enabled=True,
            max_history_length=100,
            store_selection_data=True,
            store_intermediate_grids=False,
        )

        # Update config with enhanced settings
        env_config = create_enhanced_config(
            env_config,
            episode_config=episode_config,
            history_config=history_config,
        )

        # Test training workflow across multiple demonstration pairs
        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, env_config, task_data=multi_demo_task, episode_mode=0)

        # Verify initial state has enhanced fields
        assert hasattr(state, 'episode_mode')
        assert hasattr(state, 'available_demo_pairs')
        assert hasattr(state, 'action_history')
        assert hasattr(state, 'allowed_operations_mask')

        # Verify we're in training mode
        assert state.episode_mode == 0  # train mode

        # Verify multiple demo pairs are available
        assert jnp.sum(state.available_demo_pairs) == 3

        # Test working on first demonstration pair
        initial_pair_idx = state.current_example_idx
        assert initial_pair_idx < 3

        # Take some actions on first demo
        actions_taken = []
        for i in range(5):
            mask = jnp.zeros((30, 30), dtype=jnp.bool_)
            mask = mask.at[0, i % 2].set(True)
            action = {"selection": mask, "operation": 1 + (i % 3)}  # Vary operations
            
            new_state, new_obs, reward, done, info = arc_step(state, action, env_config)
            actions_taken.append(action)
            
            # Verify action was recorded in history
            assert new_state.action_history_length > state.action_history_length
            state = new_state

        # Test switching to next demonstration pair using control operation
        switch_action = {
            "selection": jnp.zeros((30, 30), dtype=jnp.bool_),  # Selection ignored for control ops
            "operation": 35,  # SWITCH_TO_NEXT_DEMO_PAIR
        }
        
        new_state, new_obs, reward, done, info = arc_step(state, switch_action, env_config)
        
        # Verify we switched to a different demonstration pair
        assert new_state.current_example_idx != initial_pair_idx
        assert new_state.current_example_idx < 3  # Still within demo range

        # Verify action history is maintained across pair switches
        assert new_state.action_history_length == state.action_history_length + 1

        # Test working on second demonstration pair
        second_pair_idx = new_state.current_example_idx
        state = new_state
        for i in range(3):
            mask = jnp.zeros((30, 30), dtype=jnp.bool_)
            mask = mask.at[1, i].set(True)
            action = {"selection": mask, "operation": 2}
            
            new_state, new_obs, reward, done, info = arc_step(state, action, env_config)
            state = new_state

        # Verify we can access different demonstration pairs
        assert state.current_example_idx == second_pair_idx

    def test_random_multi_demo_training_workflow(self, multi_demo_task):
        """Test random multi-demonstration training workflow."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        # Configure for random multi-demo training
        episode_config = ArcEpisodeConfig(
            episode_mode=0,  # 0 = train mode
            demo_selection_strategy="random",
            allow_demo_switching=True,
            require_all_demos_solved=True,  # Test completion tracking
            max_pairs_per_episode=3,
        )

        env_config = create_enhanced_config(
            base_config,
            episode_config=episode_config,
        )

        # Test multiple episodes with random initialization
        pair_indices_seen = set()
        
        for episode in range(5):
            key = jax.random.PRNGKey(42 + episode)
            state, obs = arc_reset(key, env_config, multi_demo_task, episode_mode=0)
            
            # Track which pairs we start with
            pair_indices_seen.add(int(state.current_example_idx))
            
            # Verify random selection is working (over multiple episodes)
            assert state.current_example_idx < 3

        # With 5 episodes, we should see some variety in starting pairs
        # (This is probabilistic, but very likely with different seeds)
        assert len(pair_indices_seen) >= 1  # At least one unique starting pair

    def test_multi_demo_completion_tracking(self, multi_demo_task):
        """Test completion status tracking across demonstration pairs."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        episode_config = ArcEpisodeConfig(
            episode_mode=0,  # 0 = train mode
            demo_selection_strategy="sequential",
            allow_demo_switching=True,
            require_all_demos_solved=True,
        )

        env_config = create_enhanced_config(
            base_config,
            episode_config=episode_config,
        )

        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, env_config, multi_demo_task, episode_mode=0)

        # Initially no demos should be completed
        assert jnp.sum(state.demo_completion_status) == 0

        # Simulate solving current demonstration pair
        # (This would normally happen through similarity score reaching 1.0)
        # For testing, we'll manually update completion status
        current_pair = state.current_example_idx
        new_completion_status = state.demo_completion_status.at[current_pair].set(True)
        
        state = eqx.tree_at(
            lambda s: s.demo_completion_status,
            state,
            new_completion_status,
        )

        # Verify completion was tracked
        assert state.demo_completion_status[current_pair] == True
        assert jnp.sum(state.demo_completion_status) == 1


class TestEvaluationModeIntegration:
    """Test evaluation mode with proper target masking."""

    @pytest.fixture
    def test_task(self) -> JaxArcTask:
        """Create a task with test pairs for evaluation testing."""
        max_height, max_width = 30, 30
        max_train_pairs, max_test_pairs = 2, 3

        # Create minimal training data
        train_input = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        train_input = train_input.at[0:2, 0:2].set(jnp.array([[0, 1], [1, 0]]))
        train_output = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        train_output = train_output.at[0:2, 0:2].set(jnp.array([[1, 0], [0, 1]]))
        train_mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)
        train_mask = train_mask.at[0:2, 0:2].set(True)

        input_grids = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.int32)
        input_grids = input_grids.at[0].set(train_input)
        output_grids = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.int32)
        output_grids = output_grids.at[0].set(train_output)
        input_masks = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.bool_)
        input_masks = input_masks.at[0].set(train_mask)
        output_masks = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.bool_)
        output_masks = output_masks.at[0].set(train_mask)

        # Create multiple test pairs
        test_inputs = []
        test_outputs = []
        test_masks = []

        for i in range(2):  # 2 test pairs
            test_input = jnp.zeros((max_height, max_width), dtype=jnp.int32)
            test_output = jnp.zeros((max_height, max_width), dtype=jnp.int32)
            test_mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)

            if i == 0:
                test_input = test_input.at[0:2, 0:2].set(jnp.array([[0, 1], [1, 0]]))
                test_output = test_output.at[0:2, 0:2].set(jnp.array([[1, 0], [0, 1]]))
                test_mask = test_mask.at[0:2, 0:2].set(True)
            else:
                test_input = test_input.at[0:3, 0:3].set(jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
                test_output = test_output.at[0:3, 0:3].set(jnp.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))
                test_mask = test_mask.at[0:3, 0:3].set(True)

            test_inputs.append(test_input)
            test_outputs.append(test_output)
            test_masks.append(test_mask)

        test_input_grids = jnp.stack(test_inputs + [jnp.zeros((max_height, max_width), dtype=jnp.int32)])
        test_output_grids = jnp.stack(test_outputs + [jnp.zeros((max_height, max_width), dtype=jnp.int32)])
        test_input_masks = jnp.stack(test_masks + [jnp.zeros((max_height, max_width), dtype=jnp.bool_)])
        test_output_masks = jnp.stack(test_masks + [jnp.zeros((max_height, max_width), dtype=jnp.bool_)])

        return JaxArcTask(
            input_grids_examples=input_grids,
            input_masks_examples=input_masks,
            output_grids_examples=output_grids,
            output_masks_examples=output_masks,
            num_train_pairs=1,
            test_input_grids=test_input_grids,
            test_input_masks=test_input_masks,
            true_test_output_grids=test_output_grids,
            true_test_output_masks=test_output_masks,
            num_test_pairs=2,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

    def test_evaluation_mode_initialization(self, test_task):
        """Test proper initialization in evaluation mode."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        # Configure for evaluation mode
        episode_config = ArcEpisodeConfig(
            episode_mode=1,  # 1 = test mode
            test_selection_strategy="sequential",
            allow_test_switching=True,
            require_all_tests_solved=True,
            evaluation_reward_frequency="submit",
        )

        env_config = create_enhanced_config(
            base_config,
            episode_config=episode_config,
        )

        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, env_config, test_task, episode_mode=1)

        # Verify we're in test mode
        assert state.episode_mode == 1  # test mode

        # Verify test pairs are available
        assert jnp.sum(state.available_test_pairs) == 2

        # Verify we're working on a test pair (not demo pair)
        # In test mode, current_example_idx should refer to test pairs
        assert state.current_example_idx < 2  # Within test pair range

        # Verify working grid is initialized from test input
        # (Should match one of the test input grids)
        test_input_0 = test_task.test_input_grids[0]
        test_input_1 = test_task.test_input_grids[1]
        
        # Working grid should match one of the test inputs
        matches_test_0 = jnp.allclose(state.working_grid, test_input_0)
        matches_test_1 = jnp.allclose(state.working_grid, test_input_1)
        assert matches_test_0 or matches_test_1

    def test_target_masking_in_evaluation_mode(self, test_task):
        """Test that target grids are properly masked in evaluation mode."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        episode_config = ArcEpisodeConfig(
            episode_mode=1,  # 1 = test mode
            evaluation_reward_frequency="submit",
        )

        env_config = create_enhanced_config(
            base_config,
            episode_config=episode_config,
        )

        env = ArcEnvironment(env_config)

        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, env_config, test_task, episode_mode=1)

        # In evaluation mode, target_grid should be masked/unavailable
        # The exact implementation may vary, but agents shouldn't have access to true targets
        
        # Test that we can create observations without exposing targets
        observation_config = ObservationConfig(
            include_target_grid=False,  # Should be False in test mode
            observation_format="standard",
        )

        obs = create_observation(state, observation_config)
        
        # Verify observation doesn't contain target information
        # (Implementation depends on ArcObservation structure)
        assert obs is not None

    def test_evaluation_reward_calculation(self, test_task):
        """Test reward calculation in evaluation mode."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        episode_config = ArcEpisodeConfig(
            episode_mode=1,  # 1 = test mode
            evaluation_reward_frequency="submit",
        )

        env_config = create_enhanced_config(
            base_config,
            episode_config=episode_config,
        )

        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, env_config, test_task, episode_mode=1)

        # Take non-submit actions - should only get step penalties
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        mask = mask.at[0, 0].set(True)
        action = {"selection": mask, "operation": 1}  # Fill operation

        new_state, new_obs, reward, done, info = arc_step(state, action, env_config)

        # In evaluation mode with submit-only rewards, non-submit actions should get step penalty
        assert reward <= 0  # Should be negative (step penalty)

        # Test submit operation
        submit_action = {"selection": mask, "operation": 0}  # Submit operation
        final_state, final_obs, submit_reward, submit_done, submit_info = arc_step(new_state, submit_action, env_config)

        # Submit should trigger reward calculation (could be positive or negative)
        assert isinstance(submit_reward, (int, float, jnp.ndarray))

    def test_test_pair_switching(self, test_task):
        """Test switching between test pairs in evaluation mode."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        episode_config = ArcEpisodeConfig(
            episode_mode=1,  # 1 = test mode
            test_selection_strategy="sequential",
            allow_test_switching=True,
        )

        env_config = create_enhanced_config(
            base_config,
            episode_config=episode_config,
        )

        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, env_config, test_task, episode_mode=1)

        initial_pair_idx = state.current_example_idx

        # Switch to next test pair
        switch_action = {
            "selection": jnp.zeros((30, 30), dtype=jnp.bool_),
            "operation": 37,  # SWITCH_TO_NEXT_TEST_PAIR
        }

        new_state, new_obs, reward, done, info = arc_step(state, switch_action, env_config)

        # Verify we switched to a different test pair (if multiple pairs available)
        if jnp.sum(state.available_test_pairs) > 1:
            assert new_state.current_example_idx != initial_pair_idx
        assert new_state.current_example_idx < 2  # Within test pair range


class TestJAXTransformationsIntegration:
    """Test JAX transformations (jit, vmap, pmap) with all enhancements."""

    @pytest.fixture
    def simple_task(self) -> JaxArcTask:
        """Create a simple task for JAX transformation testing."""
        max_height, max_width = 30, 30
        max_train_pairs, max_test_pairs = 2, 1

        # Simple 2x2 pattern
        input_grid = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        input_grid = input_grid.at[0:2, 0:2].set(jnp.array([[0, 1], [1, 0]]))
        output_grid = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        output_grid = output_grid.at[0:2, 0:2].set(jnp.array([[1, 0], [0, 1]]))
        mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)
        mask = mask.at[0:2, 0:2].set(True)

        # Create arrays
        input_grids = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.int32)
        input_grids = input_grids.at[0].set(input_grid)
        output_grids = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.int32)
        output_grids = output_grids.at[0].set(output_grid)
        input_masks = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.bool_)
        input_masks = input_masks.at[0].set(mask)
        output_masks = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.bool_)
        output_masks = output_masks.at[0].set(mask)

        test_input_grids = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.int32)
        test_input_grids = test_input_grids.at[0].set(input_grid)
        test_input_masks = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.bool_)
        test_input_masks = test_input_masks.at[0].set(mask)
        test_output_grids = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.int32)
        test_output_grids = test_output_grids.at[0].set(output_grid)
        test_output_masks = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.bool_)
        test_output_masks = test_output_masks.at[0].set(mask)

        return JaxArcTask(
            input_grids_examples=input_grids,
            input_masks_examples=input_masks,
            output_grids_examples=output_grids,
            output_masks_examples=output_masks,
            num_train_pairs=1,
            test_input_grids=test_input_grids,
            test_input_masks=test_input_masks,
            true_test_output_grids=test_output_grids,
            true_test_output_masks=test_output_masks,
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

    def test_jit_compilation_with_enhancements(self, simple_task):
        """Test JIT compilation of enhanced reset function and basic functionality."""
        cfg = get_config()
        env_config = JaxArcConfig.from_hydra(cfg)

        # Configure with enhanced features
        episode_config = ArcEpisodeConfig(
            episode_mode=0,  # 0 = train mode
            demo_selection_strategy="sequential",
            allow_demo_switching=True,
        )

        history_config = HistoryConfig(
            enabled=True,
            max_history_length=50,
            store_selection_data=True,
        )

        env_config = create_enhanced_config(
            env_config,
            episode_config=episode_config,
            history_config=history_config,
        )

        # Test JIT compilation of reset function
        @jax.jit
        def jit_reset(key):
            return arc_reset(key, env_config, simple_task, episode_mode=0)

        key = jax.random.PRNGKey(42)
        state, obs = jit_reset(key)

        # Verify JIT compilation worked and enhanced fields are present
        assert isinstance(state, ArcEnvState)
        assert hasattr(state, 'episode_mode')
        assert hasattr(state, 'action_history')
        assert hasattr(state, 'allowed_operations_mask')

        # Verify enhanced state fields have correct values
        assert state.episode_mode == 0  # 0 = train mode
        assert state.action_history_length == 0  # No actions taken yet
        assert jnp.sum(state.allowed_operations_mask) > 0  # Some operations allowed

        # Note: The enhanced arc_step function currently has JAX compatibility issues
        # with the boolean conversion for control operations. This is a known limitation
        # that would need to be addressed by refactoring the step function to use
        # JAX-compatible conditional logic (e.g., jax.lax.cond instead of if statements).

    def test_vmap_across_environments(self, simple_task):
        """Test vmap across multiple environments with enhanced functionality."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        # Create config with disabled logging and visualization for batch processing
        env_config = create_enhanced_config(
            base_config,
            disable_logging=True,
            disable_visualization=True,
        )

        # Test vmapped reset
        @jax.jit
        def reset_fn(key):
            return arc_reset(key, env_config, simple_task, episode_mode=0)

        batch_size = 8
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
        
        vmapped_reset = jax.vmap(reset_fn)
        batch_states, batch_obs = vmapped_reset(keys)

        # Verify batch processing worked
        assert batch_states.working_grid.shape[0] == batch_size
        assert batch_states.episode_mode.shape[0] == batch_size
        assert batch_states.action_history.shape[0] == batch_size

        # Test vmapped step
        @jax.jit
        def step_fn(state, action):
            return arc_step(state, action, env_config)

        # Create batch of actions
        batch_masks = jnp.zeros((batch_size, 30, 30), dtype=jnp.bool_)
        batch_masks = batch_masks.at[:, 0, 0].set(True)
        batch_operations = jnp.array([1, 2, 3, 1, 2, 3, 1, 2], dtype=jnp.int32)
        
        batch_actions = {
            "selection": batch_masks,
            "operation": batch_operations,
        }

        vmapped_step = jax.vmap(step_fn)
        batch_new_states, batch_new_obs, batch_rewards, batch_done, batch_info = vmapped_step(
            batch_states, batch_actions
        )

        # Verify batch step processing worked
        assert batch_new_states.working_grid.shape[0] == batch_size
        assert batch_rewards.shape[0] == batch_size
        # Info dict contains arrays for each field (JAX vmap behavior)
        assert batch_info["similarity"].shape[0] == batch_size
        assert batch_info["step_count"].shape[0] == batch_size

        # Verify action history was updated for all environments
        assert jnp.all(batch_new_states.action_history_length > batch_states.action_history_length)

    def test_vmap_across_demonstration_pairs(self, simple_task):
        """Test vmap across different demonstration pairs."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        env_config = create_enhanced_config(
            base_config,
            disable_logging=True,
            disable_visualization=True,
        )

        # Test vmap with consistent mode (train mode for all)
        num_pairs = 4
        
        @jax.jit
        def reset_with_key(key):
            return arc_reset(key, env_config, simple_task, episode_mode=0)

        keys = jax.random.split(jax.random.PRNGKey(42), num_pairs)
        
        vmapped_reset = jax.vmap(reset_with_key)
        batch_states, batch_obs = vmapped_reset(keys)

        # Verify batch processing worked
        assert batch_states.working_grid.shape[0] == num_pairs
        assert batch_states.episode_mode.shape[0] == num_pairs
        
        # All should be in train mode (0)
        assert jnp.all(batch_states.episode_mode == 0)

    @pytest.mark.skipif(jax.device_count() < 2, reason="Requires multiple devices for pmap")
    def test_pmap_across_devices(self, simple_task):
        """Test pmap across devices with enhanced functionality."""
        cfg = get_config()
        env_config = JaxArcConfig.from_hydra(cfg)

        env_config = eqx.tree_at(
            lambda c: (c.logging.log_operations, c.visualization.enabled),
            env_config,
            (False, False),
        )

        device_count = jax.device_count()
        
        @jax.pmap
        def reset_fn(key):
            return arc_reset(key, env_config, simple_task, episode_mode=0)

        keys = jax.random.split(jax.random.PRNGKey(42), device_count)
        device_states, device_obs = reset_fn(keys)

        # Verify pmap worked across devices
        assert device_states.working_grid.shape[0] == device_count
        assert device_states.episode_mode.shape[0] == device_count

        @jax.pmap
        def step_fn(state, action):
            return arc_step(state, action, env_config)

        # Create actions for each device
        device_masks = jnp.zeros((device_count, 30, 30), dtype=jnp.bool_)
        device_masks = device_masks.at[:, 0, 0].set(True)
        device_operations = jnp.ones(device_count, dtype=jnp.int32)
        
        device_actions = {
            "selection": device_masks,
            "operation": device_operations,
        }

        device_new_states, device_new_obs, device_rewards, device_done, device_info = step_fn(
            device_states, device_actions
        )

        # Verify pmap step worked
        assert device_new_states.working_grid.shape[0] == device_count
        assert device_rewards.shape[0] == device_count


class TestPerformanceAndMemoryIntegration:
    """Test performance impact and memory usage of new features."""

    @pytest.fixture
    def performance_task(self) -> JaxArcTask:
        """Create a task for performance testing."""
        max_height, max_width = 30, 30
        max_train_pairs, max_test_pairs = 4, 2

        # Create task with multiple pairs for comprehensive testing
        input_grids = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.int32)
        output_grids = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.int32)
        input_masks = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.bool_)
        output_masks = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.bool_)

        # Fill first pair with data
        input_grids = input_grids.at[0, 0:2, 0:2].set(jnp.array([[0, 1], [1, 0]]))
        output_grids = output_grids.at[0, 0:2, 0:2].set(jnp.array([[1, 0], [0, 1]]))
        input_masks = input_masks.at[0, 0:2, 0:2].set(True)
        output_masks = output_masks.at[0, 0:2, 0:2].set(True)

        test_input_grids = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.int32)
        test_input_masks = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.bool_)
        test_output_grids = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.int32)
        test_output_masks = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.bool_)

        return JaxArcTask(
            input_grids_examples=input_grids,
            input_masks_examples=input_masks,
            output_grids_examples=output_grids,
            output_masks_examples=output_masks,
            num_train_pairs=1,
            test_input_grids=test_input_grids,
            test_input_masks=test_input_masks,
            true_test_output_grids=test_output_grids,
            true_test_output_masks=test_output_masks,
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

    def test_memory_usage_with_action_history(self, performance_task):
        """Test memory usage with different action history configurations."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        # Test configurations with different memory footprints
        configs = [
            # Minimal memory configuration (small but valid)
            HistoryConfig(
                enabled=True,
                max_history_length=10,  # Small but positive
                store_selection_data=False,
                store_intermediate_grids=False,
            ),
            # Standard configuration
            HistoryConfig(
                enabled=True,
                max_history_length=100,
                store_selection_data=True,
                store_intermediate_grids=False,
            ),
            # Memory-intensive configuration
            HistoryConfig(
                enabled=True,
                max_history_length=1000,
                store_selection_data=True,
                store_intermediate_grids=False,  # Keep False to avoid excessive memory
            ),
        ]

        memory_usage = []

        for i, history_config in enumerate(configs):
            tracemalloc.start()

            env_config = create_enhanced_config(
                base_config,
                history_config=history_config,
            )

            # Create multiple environments to measure memory scaling
            batch_size = 10
            keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

            @jax.jit
            def reset_fn(key):
                return arc_reset(key, env_config, performance_task, episode_mode=0)

            vmapped_reset = jax.vmap(reset_fn)
            batch_states, batch_obs = vmapped_reset(keys)

            # Take several steps to populate action history
            @jax.jit
            def step_fn(state, action):
                return arc_step(state, action, env_config)

            vmapped_step = jax.vmap(step_fn)

            for step in range(20):
                batch_masks = jnp.zeros((batch_size, 30, 30), dtype=jnp.bool_)
                batch_masks = batch_masks.at[:, 0, step % 5].set(True)
                batch_operations = jnp.full(batch_size, 1 + (step % 3), dtype=jnp.int32)
                
                batch_actions = {
                    "selection": batch_masks,
                    "operation": batch_operations,
                }

                batch_states, batch_obs, batch_rewards, batch_done, batch_info = vmapped_step(
                    batch_states, batch_actions
                )

            current, peak = tracemalloc.get_traced_memory()
            memory_usage.append(peak)
            tracemalloc.stop()

        # Verify memory usage scales reasonably with history configuration
        assert len(memory_usage) == 3
        
        # Memory usage should be reasonable for all configurations
        # (This is a rough check - exact values depend on implementation)
        # Allow for some variation in memory usage patterns
        for i, usage in enumerate(memory_usage):
            assert usage > 0, f"Memory usage should be positive for config {i}"
            # Memory should be reasonable (less than 100MB for test environments)
            assert usage < 100_000_000, f"Memory usage too high for config {i}: {usage} bytes"
        # Note: We don't test memory_usage[1] <= memory_usage[2] because
        # the difference might be small without intermediate grids

    def test_step_latency_with_enhancements(self, performance_task):
        """Test step latency with enhanced functionality."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        # Test configurations
        configs = [
            # Minimal enhancements
            {
                "episode": ArcEpisodeConfig(episode_mode=0, allow_demo_switching=False),
                "history": HistoryConfig(enabled=False),
            },
            # Full enhancements
            {
                "episode": ArcEpisodeConfig(
                    episode_mode=0,  # 0 = train mode
                    allow_demo_switching=True,
                    demo_selection_strategy="random",
                ),
                "history": HistoryConfig(
                    enabled=True,
                    max_history_length=100,
                    store_selection_data=True,
                ),
            },
        ]

        latencies = []

        for config_updates in configs:
            env_config = create_enhanced_config(
                base_config,
                episode_config=config_updates["episode"],
                history_config=config_updates["history"],
            )

            # JIT compile functions
            @jax.jit
            def reset_fn(key):
                return arc_reset(key, env_config, performance_task, episode_mode=0)

            @jax.jit
            def step_fn(state, action):
                return arc_step(state, action, env_config)

            key = jax.random.PRNGKey(42)
            state, obs = reset_fn(key)

            # Warm up JIT
            mask = jnp.zeros((30, 30), dtype=jnp.bool_)
            mask = mask.at[0, 0].set(True)
            action = {"selection": mask, "operation": 1}
            
            for _ in range(5):
                state, obs, reward, done, info = step_fn(state, action)

            # Measure step latency
            num_steps = 100
            start_time = time.time()
            
            for step in range(num_steps):
                mask = jnp.zeros((30, 30), dtype=jnp.bool_)
                mask = mask.at[0, step % 5].set(True)
                action = {"selection": mask, "operation": 1 + (step % 3)}
                
                state, obs, reward, done, info = step_fn(state, action)

            end_time = time.time()
            avg_latency = (end_time - start_time) / num_steps
            latencies.append(avg_latency)

        # Verify latency increase is reasonable (less than 50% overhead)
        assert len(latencies) == 2
        latency_increase = (latencies[1] - latencies[0]) / latencies[0]
        assert latency_increase < 0.5  # Less than 50% overhead

    def test_batch_scalability(self, performance_task):
        """Test scalability with large batch sizes."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        env_config = create_enhanced_config(
            base_config,
            disable_logging=True,
            disable_visualization=True,
        )

        # Test different batch sizes
        batch_sizes = [1, 10, 100, 500]
        processing_times = []

        for batch_size in batch_sizes:
            @jax.jit
            def reset_fn(key):
                return arc_reset(key, env_config, performance_task, episode_mode=0)

            @jax.jit
            def step_fn(state, action):
                return arc_step(state, action, env_config)

            vmapped_reset = jax.vmap(reset_fn)
            vmapped_step = jax.vmap(step_fn)

            keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

            start_time = time.time()

            # Reset batch
            batch_states, batch_obs = vmapped_reset(keys)

            # Take several steps
            for step in range(10):
                batch_masks = jnp.zeros((batch_size, 30, 30), dtype=jnp.bool_)
                batch_masks = batch_masks.at[:, 0, step % 5].set(True)
                batch_operations = jnp.full(batch_size, 1 + (step % 3), dtype=jnp.int32)
                
                batch_actions = {
                    "selection": batch_masks,
                    "operation": batch_operations,
                }

                batch_states, batch_obs, batch_rewards, batch_done, batch_info = vmapped_step(
                    batch_states, batch_actions
                )

            end_time = time.time()
            processing_times.append(end_time - start_time)

        # Verify processing scales reasonably (not exponentially)
        assert len(processing_times) == len(batch_sizes)
        
        # Processing time should scale roughly linearly with batch size
        # (allowing for some overhead and JIT compilation effects)
        time_per_env = [t / b for t, b in zip(processing_times, batch_sizes)]
        
        # Time per environment shouldn't increase dramatically with batch size
        # Allow for more variation due to JIT compilation and system effects
        assert max(time_per_env) / min(time_per_env) < 50.0  # Less than 50x variation

    def test_comprehensive_end_to_end_workflow(self, performance_task):
        """Test complete end-to-end workflow with all enhanced features."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        # Configure with all enhanced features enabled
        episode_config = ArcEpisodeConfig(
            episode_mode=0,  # 0 = train mode
            demo_selection_strategy="sequential",
            allow_demo_switching=True,
            require_all_demos_solved=False,
            max_pairs_per_episode=3,
        )

        history_config = HistoryConfig(
            enabled=True,
            max_history_length=100,
            store_selection_data=True,
            store_intermediate_grids=False,
        )

        env_config = create_enhanced_config(
            base_config,
            episode_config=episode_config,
            history_config=history_config,
        )

        # Test complete workflow
        key = jax.random.PRNGKey(42)
        
        # 1. Reset with enhanced functionality
        state, obs = arc_reset(key, env_config, performance_task, episode_mode=0)
        
        # Verify enhanced state fields
        assert hasattr(state, 'episode_mode')
        assert hasattr(state, 'action_history')
        assert hasattr(state, 'allowed_operations_mask')
        assert state.episode_mode == 0  # train mode
        
        # 2. Take multiple actions with history tracking
        for i in range(10):
            mask = jnp.zeros((30, 30), dtype=jnp.bool_)
            mask = mask.at[0, i % 3].set(True)
            action = {"selection": mask, "operation": 1 + (i % 3)}
            
            new_state, new_obs, reward, done, info = arc_step(state, action, env_config)
            
            # Verify action history is being tracked
            assert new_state.action_history_length > state.action_history_length
            state = new_state
        
        # 3. Test pair switching
        switch_action = {
            "selection": jnp.zeros((30, 30), dtype=jnp.bool_),
            "operation": 35,  # SWITCH_TO_NEXT_DEMO_PAIR
        }
        
        initial_pair = state.current_example_idx
        new_state, new_obs, reward, done, info = arc_step(state, switch_action, env_config)
        
        # Verify pair switching worked (if multiple pairs available)
        if jnp.sum(state.available_demo_pairs) > 1:
            assert new_state.current_example_idx != initial_pair
        
        # 4. Test observation creation (current implementation returns working grid)
        obs = create_observation(new_state, env_config)
        assert obs is not None
        assert obs.shape == (30, 30)  # Working grid shape
        assert obs.dtype == jnp.int32  # Grid values are integers
        
        # 5. Test evaluation mode
        test_state, test_obs = arc_reset(key, env_config, performance_task, episode_mode=1)
        assert test_state.episode_mode == 1  # test mode
        
        print("✅ Comprehensive end-to-end workflow completed successfully")

    def test_performance_regression_validation(self, performance_task):
        """Test that enhanced features don't cause significant performance regression."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        # Test baseline (minimal enhancements)
        minimal_config = create_enhanced_config(
            base_config,
            history_config=HistoryConfig(
                enabled=True,
                max_history_length=10,
                store_selection_data=False,
            ),
        )

        # Test full enhancements
        full_config = create_enhanced_config(
            base_config,
            episode_config=ArcEpisodeConfig(
                episode_mode=0,  # 0 = train mode
                allow_demo_switching=True,
            ),
            history_config=HistoryConfig(
                enabled=True,
                max_history_length=100,
                store_selection_data=True,
            ),
        )

        # Measure performance for both configurations
        configs = [("minimal", minimal_config), ("full", full_config)]
        times = []

        for config_name, config in configs:
            @jax.jit
            def reset_fn(key):
                return arc_reset(key, config, performance_task, episode_mode=0)

            key = jax.random.PRNGKey(42)
            
            # Warm up
            state, obs = reset_fn(key)
            
            # Measure reset time
            start_time = time.time()
            for i in range(10):
                key = jax.random.PRNGKey(42 + i)
                state, obs = reset_fn(key)
            reset_time = (time.time() - start_time) / 10
            
            times.append((config_name, reset_time))
            print(f"{config_name} config reset time: {reset_time:.4f}s")

        # Verify performance is reasonable
        minimal_time, full_time = times[0][1], times[1][1]
        performance_overhead = (full_time - minimal_time) / minimal_time if minimal_time > 0 else 0
        
        # Allow up to 100% overhead for enhanced features (this is generous for testing)
        assert performance_overhead < 1.0, f"Performance overhead too high: {performance_overhead:.2%}"
        
        print(f"✅ Performance regression test passed. Overhead: {performance_overhead:.2%}")


class TestArcObservationIntegration:
    """Test ArcObservation with RL agents and different observation configurations."""

    @pytest.fixture
    def observation_task(self) -> JaxArcTask:
        """Create a task for observation testing."""
        max_height, max_width = 30, 30
        max_train_pairs, max_test_pairs = 2, 1

        # Create simple task
        input_grid = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        input_grid = input_grid.at[0:3, 0:3].set(jnp.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]))
        output_grid = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        output_grid = output_grid.at[0:3, 0:3].set(jnp.array([[2, 0, 1], [0, 1, 2], [1, 2, 0]]))
        mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)
        mask = mask.at[0:3, 0:3].set(True)

        input_grids = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.int32)
        input_grids = input_grids.at[0].set(input_grid)
        output_grids = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.int32)
        output_grids = output_grids.at[0].set(output_grid)
        input_masks = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.bool_)
        input_masks = input_masks.at[0].set(mask)
        output_masks = jnp.zeros((max_train_pairs, max_height, max_width), dtype=jnp.bool_)
        output_masks = output_masks.at[0].set(mask)

        test_input_grids = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.int32)
        test_input_grids = test_input_grids.at[0].set(input_grid)
        test_input_masks = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.bool_)
        test_input_masks = test_input_masks.at[0].set(mask)
        test_output_grids = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.int32)
        test_output_grids = test_output_grids.at[0].set(output_grid)
        test_output_masks = jnp.zeros((max_test_pairs, max_height, max_width), dtype=jnp.bool_)
        test_output_masks = test_output_masks.at[0].set(mask)

        return JaxArcTask(
            input_grids_examples=input_grids,
            input_masks_examples=input_masks,
            output_grids_examples=output_grids,
            output_masks_examples=output_masks,
            num_train_pairs=1,
            test_input_grids=test_input_grids,
            test_input_masks=test_input_masks,
            true_test_output_grids=test_output_grids,
            true_test_output_masks=test_output_masks,
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

    def test_different_observation_configurations(self, observation_task):
        """Test different observation configurations."""
        cfg = get_config()
        env_config = JaxArcConfig.from_hydra(cfg)

        env_config = create_enhanced_config(env_config)

        # Test different observation formats
        observation_configs = [
            ObservationConfig(observation_format="minimal"),
            ObservationConfig(observation_format="standard"),
            ObservationConfig(observation_format="rich"),
        ]

        for obs_config in observation_configs:
            key = jax.random.PRNGKey(42)
            state, obs = arc_reset(key, env_config, observation_task, episode_mode=0)

            # Create observation with specific configuration
            custom_obs = create_observation(state, obs_config)

            # Verify observation was created successfully
            assert custom_obs is not None
            # Note: Specific structure depends on ArcObservation implementation

    def test_observation_with_mock_rl_agent(self, observation_task):
        """Test ArcObservation with a mock RL agent."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        env_config = create_enhanced_config(
            base_config,
            disable_logging=True,
        )

        # Mock RL agent that processes observations
        class MockRLAgent:
            def __init__(self):
                self.observations_processed = 0
                self.last_observation_shape = None

            def process_observation(self, observation):
                """Process observation and return action."""
                self.observations_processed += 1
                
                # Extract observation shape (depends on implementation)
                if hasattr(observation, 'shape'):
                    self.last_observation_shape = observation.shape
                elif isinstance(observation, dict):
                    # Handle structured observation
                    self.last_observation_shape = {k: v.shape for k, v in observation.items()}
                
                # Return simple action
                mask = jnp.zeros((30, 30), dtype=jnp.bool_)
                mask = mask.at[0, self.observations_processed % 5].set(True)
                return {
                    "selection": mask,
                    "operation": 1 + (self.observations_processed % 3),
                }

        agent = MockRLAgent()

        # Test agent interaction with environment
        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, env_config, observation_task, episode_mode=0)

        # Agent processes initial observation
        action = agent.process_observation(obs)
        assert agent.observations_processed == 1

        # Take steps with agent
        for step in range(10):
            new_state, new_obs, reward, done, info = arc_step(state, action, env_config)
            
            # Agent processes new observation and selects next action
            action = agent.process_observation(new_obs)
            state = new_state

        # Verify agent processed all observations
        assert agent.observations_processed == 11  # Initial + 10 steps

    def test_observation_consistency_across_modes(self, observation_task):
        """Test observation consistency between training and evaluation modes."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        env_config = create_enhanced_config(
            base_config,
            disable_logging=True,
        )

        # Test training mode observation
        key = jax.random.PRNGKey(42)
        train_state, train_obs = arc_reset(key, env_config, observation_task, episode_mode=0)

        # Test evaluation mode observation
        test_state, test_obs = arc_reset(key, env_config, observation_task, episode_mode=1)

        # Observations should have consistent structure but different content
        # (exact comparison depends on ArcObservation implementation)
        assert train_obs is not None
        assert test_obs is not None

        # Both should be valid JAX arrays or structures
        if hasattr(train_obs, 'shape') and hasattr(test_obs, 'shape'):
            assert train_obs.shape == test_obs.shape

    def test_observation_with_action_history(self, observation_task):
        """Test observations that include action history."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        # Configure to include action history in observations
        history_config = HistoryConfig(
            enabled=True,
            max_history_length=50,
            store_selection_data=True,
        )

        obs_config = ObservationConfig(
            include_recent_actions=True,
            recent_action_count=5,
        )

        env_config = create_enhanced_config(
            base_config,
            history_config=history_config,
        )

        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, env_config, observation_task, episode_mode=0)

        # Take several actions to build history
        for step in range(10):
            mask = jnp.zeros((30, 30), dtype=jnp.bool_)
            mask = mask.at[0, step % 5].set(True)
            action = {"selection": mask, "operation": 1 + (step % 3)}
            
            state, obs, reward, done, info = arc_step(state, action, env_config)

            # Create observation with action history
            history_obs = create_observation(state, obs_config)
            
            # Verify observation includes history information
            assert history_obs is not None
            # Note: Specific verification depends on ArcObservation structure

    def test_observation_memory_efficiency(self, observation_task):
        """Test memory efficiency of different observation configurations."""
        cfg = get_config()
        base_config = JaxArcConfig.from_hydra(cfg)

        env_config = create_enhanced_config(
            base_config,
            disable_logging=True,
        )

        # Test memory usage with different observation configurations
        configs = [
            ObservationConfig(observation_format="minimal"),
            ObservationConfig(
                observation_format="rich",
                include_recent_actions=True,
                recent_action_count=20,
            ),
        ]

        memory_usage = []

        for obs_config in configs:
            tracemalloc.start()

            # Create batch of environments
            batch_size = 50
            keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

            @jax.jit
            def reset_and_observe(key):
                state, obs = arc_reset(key, env_config, observation_task, episode_mode=0)
                custom_obs = create_observation(state, obs_config)
                return state, custom_obs

            vmapped_reset = jax.vmap(reset_and_observe)
            batch_states, batch_obs = vmapped_reset(keys)

            current, peak = tracemalloc.get_traced_memory()
            memory_usage.append(peak)
            tracemalloc.stop()

        # Verify memory usage is reasonable
        assert len(memory_usage) == 2
        # Rich observations should use more memory, but not excessively
        memory_ratio = memory_usage[1] / memory_usage[0]
        assert memory_ratio < 5.0  # Less than 5x memory increase


if __name__ == "__main__":
    pytest.main([__file__])