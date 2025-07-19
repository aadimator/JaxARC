#!/usr/bin/env python3
"""
Visualization Workflows Examples

This script demonstrates common workflows and best practices for using
JaxARC's enhanced visualization system in different scenarios.
"""

import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig, OmegaConf
import time
import os
from pathlib import Path

from jaxarc.envs import arc_reset, arc_step, create_standard_config
from jaxarc.utils.visualization import (
    EnhancedVisualizer,
    create_visualizer_from_config,
    VisualizationConfig,
    EpisodeManager,
    AsyncLogger,
    WandbIntegration
)


def workflow_1_development_debugging():
    """Workflow 1: Development and debugging workflow."""
    
    print("=" * 60)
    print("Workflow 1: Development and Debugging")
    print("=" * 60)
    
    # Load development configuration
    config_path = "conf/visualization/development.yaml"
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        visualizer = create_visualizer_from_config(cfg)
    else:
        # Fallback to programmatic config
        vis_config = VisualizationConfig(
            debug_level="verbose",
            output_formats=["svg", "png"],
            show_operation_names=True,
            highlight_changes=True,
            log_frequency=5
        )
        visualizer = EnhancedVisualizer(vis_config)
    
    print("🔧 Development workflow started")
    print("Features: Verbose logging, rich visualizations, frequent saves")
    
    # Setup environment
    key = jax.random.PRNGKey(42)
    config = create_standard_config()
    
    # Development workflow: Test specific scenarios
    test_scenarios = [
        {"name": "Fill operations", "operations": [1, 2, 3, 4, 5]},
        {"name": "Flood fill", "operations": [10, 11, 12, 13, 14]},
        {"name": "Movement", "operations": [20, 21, 22, 23]}
    ]
    
    for scenario_idx, scenario in enumerate(test_scenarios):
        print(f"\n🧪 Testing scenario: {scenario['name']}")
        
        state, obs = arc_reset(key, config)
        visualizer.start_episode(scenario_idx)
        
        for step, operation in enumerate(scenario['operations']):
            # Create test action
            selection = jax.random.bernoulli(key, 0.2, state.working_grid.shape)
            action = {"selection": selection, "operation": jnp.array(operation)}
            
            print(f"  Step {step}: Testing operation {operation}")
            
            # Environment step
            new_state, obs, reward, done, info = arc_step(state, action, config)
            
            # Detailed visualization for development
            visualizer.visualize_step(
                before_state=state,
                action=action,
                after_state=new_state,
                reward=reward,
                info=info,
                step_num=step
            )
            
            # Development-specific checks
            if reward < 0:
                print(f"    ⚠️  Negative reward: {reward}")
            
            if done and step < len(scenario['operations']) - 1:
                print(f"    ⚠️  Early termination at step {step}")
                break
            
            state = new_state
            key = jax.random.split(key)[0]
        
        # Generate detailed episode summary
        visualizer.visualize_episode_summary(scenario_idx)
        print(f"  ✅ Scenario '{scenario['name']}' completed")
    
    print("\n🔧 Development workflow completed")
    print("Check outputs/development/ for detailed visualizations")


def workflow_2_research_experiment():
    """Workflow 2: Research experiment workflow."""
    
    print("\n" + "=" * 60)
    print("Workflow 2: Research Experiment")
    print("=" * 60)
    
    # Load research configuration
    config_path = "conf/visualization/research.yaml"
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        visualizer = create_visualizer_from_config(cfg)
    else:
        # Fallback configuration
        vis_config = VisualizationConfig(
            debug_level="full",
            output_formats=["svg", "png"],
            enable_comparisons=True,
            save_intermediate_states=True,
            log_frequency=1
        )
        
        episode_manager = EpisodeManager(
            base_output_dir="outputs/research",
            run_name="research_experiment_v1"
        )
        
        wandb_config = {
            "enabled": True,
            "project_name": "jaxarc-research-demo",
            "tags": ["research", "experiment"],
            "offline_mode": True
        }
        wandb_integration = WandbIntegration(wandb_config)
        
        visualizer = EnhancedVisualizer(
            vis_config=vis_config,
            episode_manager=episode_manager,
            wandb_integration=wandb_integration
        )
    
    print("🔬 Research experiment workflow started")
    print("Features: Comprehensive logging, wandb integration, detailed analysis")
    
    # Initialize experiment
    experiment_config = {
        "algorithm": "Baseline Random",
        "num_episodes": 3,
        "max_steps_per_episode": 20,
        "exploration_strategy": "uniform_random"
    }
    
    if hasattr(visualizer, 'wandb_integration') and visualizer.wandb_integration:
        visualizer.wandb_integration.initialize_run(
            experiment_config=experiment_config,
            run_name="research_workflow_demo"
        )
    
    # Research experiment: Compare different strategies
    strategies = [
        {"name": "Conservative", "selection_prob": 0.1, "operation_range": (1, 5)},
        {"name": "Moderate", "selection_prob": 0.3, "operation_range": (1, 10)},
        {"name": "Aggressive", "selection_prob": 0.5, "operation_range": (1, 15)}
    ]
    
    key = jax.random.PRNGKey(123)
    config = create_standard_config()
    
    strategy_results = []
    
    for strategy_idx, strategy in enumerate(strategies):
        print(f"\n🔬 Testing strategy: {strategy['name']}")
        
        state, obs = arc_reset(key, config)
        visualizer.start_episode(strategy_idx)
        
        episode_reward = 0.0
        strategy_metrics = {
            "name": strategy["name"],
            "total_reward": 0.0,
            "steps_taken": 0,
            "actions_taken": 0
        }
        
        for step in range(20):
            # Apply strategy
            selection_prob = strategy["selection_prob"]
            op_min, op_max = strategy["operation_range"]
            
            selection = jax.random.bernoulli(key, selection_prob, state.working_grid.shape)
            operation = jax.random.randint(key, (), op_min, op_max + 1)
            action = {"selection": selection, "operation": operation}
            
            # Environment step
            new_state, obs, reward, done, info = arc_step(state, action, config)
            episode_reward += reward
            strategy_metrics["actions_taken"] += 1
            
            # Research-level visualization
            visualizer.visualize_step(
                before_state=state,
                action=action,
                after_state=new_state,
                reward=reward,
                info=info,
                step_num=step
            )
            
            # Log to wandb if available
            if hasattr(visualizer, 'wandb_integration') and visualizer.wandb_integration:
                visualizer.wandb_integration.log_step(
                    step_num=strategy_idx * 20 + step,
                    metrics={
                        "step_reward": reward,
                        "cumulative_reward": episode_reward,
                        "strategy": strategy["name"],
                        "selection_density": jnp.sum(selection) / selection.size
                    }
                )
            
            state = new_state
            key = jax.random.split(key)[0]
            
            if done:
                break
        
        # Record strategy results
        strategy_metrics["total_reward"] = episode_reward
        strategy_metrics["steps_taken"] = step + 1
        strategy_results.append(strategy_metrics)
        
        # Generate episode summary
        visualizer.visualize_episode_summary(strategy_idx)
        
        # Log episode summary to wandb
        if hasattr(visualizer, 'wandb_integration') and visualizer.wandb_integration:
            visualizer.wandb_integration.log_episode_summary(
                episode_num=strategy_idx,
                summary_data=strategy_metrics
            )
        
        print(f"  Strategy results: Reward={episode_reward:.3f}, Steps={step + 1}")
    
    # Research analysis
    print(f"\n📊 Research Results Summary:")
    best_strategy = max(strategy_results, key=lambda x: x["total_reward"])
    
    for result in strategy_results:
        print(f"  {result['name']}: Reward={result['total_reward']:.3f}, "
              f"Steps={result['steps_taken']}, Actions={result['actions_taken']}")
    
    print(f"  🏆 Best strategy: {best_strategy['name']}")
    
    # Finish wandb run
    if hasattr(visualizer, 'wandb_integration') and visualizer.wandb_integration:
        visualizer.wandb_integration.finish_run()
    
    print("🔬 Research experiment workflow completed")


def workflow_3_production_monitoring():
    """Workflow 3: Production monitoring workflow."""
    
    print("\n" + "=" * 60)
    print("Workflow 3: Production Monitoring")
    print("=" * 60)
    
    # Load production configuration
    config_path = "conf/visualization/production.yaml"
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        visualizer = create_visualizer_from_config(cfg)
    else:
        # Fallback production config
        vis_config = VisualizationConfig(
            debug_level="minimal",
            output_formats=["svg"],
            log_frequency=50,
            memory_limit_mb=200
        )
        
        wandb_config = {
            "enabled": True,
            "project_name": "jaxarc-production-monitoring",
            "tags": ["production", "monitoring"],
            "log_frequency": 25
        }
        wandb_integration = WandbIntegration(wandb_config)
        
        visualizer = EnhancedVisualizer(
            vis_config=vis_config,
            wandb_integration=wandb_integration
        )
    
    print("🏭 Production monitoring workflow started")
    print("Features: Minimal overhead, essential metrics, automated monitoring")
    
    # Production monitoring setup
    if hasattr(visualizer, 'wandb_integration') and visualizer.wandb_integration:
        monitoring_config = {
            "environment": "production",
            "monitoring_level": "essential",
            "alert_thresholds": {
                "low_reward": -1.0,
                "high_steps": 100,
                "memory_usage": 0.8
            }
        }
        
        visualizer.wandb_integration.initialize_run(
            experiment_config=monitoring_config,
            run_name="production_monitoring"
        )
    
    # Simulate production training with monitoring
    key = jax.random.PRNGKey(456)
    config = create_standard_config()
    
    # Production metrics tracking
    production_metrics = {
        "episodes_completed": 0,
        "total_steps": 0,
        "average_reward": 0.0,
        "performance_issues": 0,
        "memory_warnings": 0
    }
    
    print("🏭 Running production simulation...")
    
    for episode in range(10):  # Simulate 10 episodes
        state, obs = arc_reset(key, config)
        
        # Only visualize some episodes in production
        if episode % 5 == 0:  # Every 5th episode
            visualizer.start_episode(episode)
            visualize_episode = True
        else:
            visualize_episode = False
        
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(30):  # Max 30 steps per episode
            # Simple production action
            selection = jax.random.bernoulli(key, 0.15, state.working_grid.shape)
            operation = jax.random.randint(key, (), 1, 8)
            action = {"selection": selection, "operation": operation}
            
            # Environment step
            new_state, obs, reward, done, info = arc_step(state, action, config)
            episode_reward += reward
            episode_steps += 1
            production_metrics["total_steps"] += 1
            
            # Minimal visualization in production
            if visualize_episode and step % 10 == 0:  # Every 10th step
                visualizer.visualize_step(
                    before_state=state,
                    action=action,
                    after_state=new_state,
                    reward=reward,
                    info=info,
                    step_num=step
                )
            
            # Production monitoring
            if reward < -0.5:
                production_metrics["performance_issues"] += 1
                print(f"  ⚠️  Performance issue in episode {episode}, step {step}")
            
            # Monitor memory usage (simulated)
            if hasattr(visualizer, 'get_memory_stats'):
                memory_stats = visualizer.get_memory_stats()
                if memory_stats.get('usage_ratio', 0) > 0.8:
                    production_metrics["memory_warnings"] += 1
                    print(f"  ⚠️  Memory warning in episode {episode}")
            
            state = new_state
            key = jax.random.split(key)[0]
            
            if done:
                break
        
        # Update production metrics
        production_metrics["episodes_completed"] += 1
        production_metrics["average_reward"] = (
            (production_metrics["average_reward"] * (episode) + episode_reward) / 
            (episode + 1)
        )
        
        # Log to production monitoring
        if hasattr(visualizer, 'wandb_integration') and visualizer.wandb_integration:
            visualizer.wandb_integration.log_step(
                step_num=episode,
                metrics={
                    "episode_reward": episode_reward,
                    "episode_steps": episode_steps,
                    "average_reward": production_metrics["average_reward"],
                    "performance_issues": production_metrics["performance_issues"],
                    "memory_warnings": production_metrics["memory_warnings"]
                }
            )
        
        # Generate episode summary for monitored episodes
        if visualize_episode:
            visualizer.visualize_episode_summary(episode)
        
        # Production status update
        if episode % 3 == 0:
            print(f"  📊 Episode {episode}: Avg Reward={production_metrics['average_reward']:.3f}, "
                  f"Issues={production_metrics['performance_issues']}")
    
    # Production summary
    print(f"\n🏭 Production Monitoring Summary:")
    print(f"  Episodes completed: {production_metrics['episodes_completed']}")
    print(f"  Total steps: {production_metrics['total_steps']}")
    print(f"  Average reward: {production_metrics['average_reward']:.3f}")
    print(f"  Performance issues: {production_metrics['performance_issues']}")
    print(f"  Memory warnings: {production_metrics['memory_warnings']}")
    
    # Finish monitoring
    if hasattr(visualizer, 'wandb_integration') and visualizer.wandb_integration:
        visualizer.wandb_integration.finish_run()
    
    print("🏭 Production monitoring workflow completed")


def workflow_4_performance_benchmarking():
    """Workflow 4: Performance benchmarking workflow."""
    
    print("\n" + "=" * 60)
    print("Workflow 4: Performance Benchmarking")
    print("=" * 60)
    
    # Load benchmark configuration
    config_path = "conf/visualization/benchmark.yaml"
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        # For benchmarking, we might want to enable minimal visualization
        cfg.enabled = True
        cfg.debug_level = "minimal"
        visualizer = create_visualizer_from_config(cfg)
    else:
        # Minimal benchmark config
        vis_config = VisualizationConfig(
            debug_level="minimal",
            output_formats=["svg"],
            log_frequency=0,  # No logging during benchmark
            memory_limit_mb=50
        )
        visualizer = EnhancedVisualizer(vis_config)
    
    print("⚡ Performance benchmarking workflow started")
    print("Features: Minimal overhead, performance measurement, comparison")
    
    # Benchmark different configurations
    benchmark_configs = [
        {"name": "No Visualization", "enabled": False},
        {"name": "Minimal Visualization", "debug_level": "minimal", "log_frequency": 100},
        {"name": "Standard Visualization", "debug_level": "standard", "log_frequency": 10},
    ]
    
    key = jax.random.PRNGKey(789)
    config = create_standard_config()
    
    benchmark_results = []
    
    for bench_idx, bench_config in enumerate(benchmark_configs):
        print(f"\n⚡ Benchmarking: {bench_config['name']}")
        
        # Configure visualizer for this benchmark
        if bench_config["name"] == "No Visualization":
            # Disable visualization completely
            visualizer.disable()
        else:
            visualizer.enable()
            if "debug_level" in bench_config:
                visualizer.set_debug_level(bench_config["debug_level"])
            if "log_frequency" in bench_config:
                visualizer.set_log_frequency(bench_config["log_frequency"])
        
        # Warm-up runs
        print("  🔥 Warming up...")
        for _ in range(3):
            state, obs = arc_reset(key, config)
            for _ in range(5):
                selection = jax.random.bernoulli(key, 0.1, state.working_grid.shape)
                action = {"selection": selection, "operation": jnp.array(1)}
                state, obs, reward, done, info = arc_step(state, action, config)
                key = jax.random.split(key)[0]
                if done:
                    break
        
        # Actual benchmark
        print("  ⏱️  Running benchmark...")
        start_time = time.perf_counter()
        
        total_steps = 0
        for episode in range(5):  # 5 episodes for benchmark
            state, obs = arc_reset(key, config)
            
            if visualizer.enabled:
                visualizer.start_episode(episode)
            
            for step in range(20):  # 20 steps per episode
                selection = jax.random.bernoulli(key, 0.1, state.working_grid.shape)
                operation = jax.random.randint(key, (), 1, 6)
                action = {"selection": selection, "operation": operation}
                
                new_state, obs, reward, done, info = arc_step(state, action, config)
                
                # Visualization (if enabled)
                if visualizer.enabled:
                    visualizer.visualize_step(
                        before_state=state,
                        action=action,
                        after_state=new_state,
                        reward=reward,
                        info=info,
                        step_num=step
                    )
                
                state = new_state
                key = jax.random.split(key)[0]
                total_steps += 1
                
                if done:
                    break
            
            if visualizer.enabled:
                visualizer.visualize_episode_summary(episode)
        
        end_time = time.perf_counter()
        
        # Calculate benchmark results
        total_time = end_time - start_time
        steps_per_second = total_steps / total_time
        
        result = {
            "name": bench_config["name"],
            "total_time": total_time,
            "total_steps": total_steps,
            "steps_per_second": steps_per_second
        }
        
        benchmark_results.append(result)
        
        print(f"  📊 Results: {total_time:.3f}s total, {steps_per_second:.1f} steps/sec")
    
    # Benchmark comparison
    print(f"\n⚡ Benchmark Results Comparison:")
    baseline = benchmark_results[0]  # No visualization baseline
    
    for result in benchmark_results:
        if result["name"] == baseline["name"]:
            print(f"  {result['name']}: {result['steps_per_second']:.1f} steps/sec (baseline)")
        else:
            overhead = ((baseline["steps_per_second"] - result["steps_per_second"]) / 
                       baseline["steps_per_second"]) * 100
            print(f"  {result['name']}: {result['steps_per_second']:.1f} steps/sec "
                  f"({overhead:.1f}% overhead)")
    
    print("⚡ Performance benchmarking workflow completed")


def workflow_5_error_debugging():
    """Workflow 5: Error debugging and troubleshooting workflow."""
    
    print("\n" + "=" * 60)
    print("Workflow 5: Error Debugging and Troubleshooting")
    print("=" * 60)
    
    # Load debugging configuration
    config_path = "conf/visualization/debugging.yaml"
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        visualizer = create_visualizer_from_config(cfg)
    else:
        # Fallback debugging config
        vis_config = VisualizationConfig(
            debug_level="full",
            output_formats=["svg", "png"],
            show_coordinates=True,
            show_operation_names=True,
            log_frequency=1,
            color_scheme="high_contrast"
        )
        visualizer = EnhancedVisualizer(vis_config)
    
    print("🐛 Error debugging workflow started")
    print("Features: Maximum detail, error tracking, step-by-step analysis")
    
    # Setup environment
    key = jax.random.PRNGKey(999)
    config = create_standard_config()
    
    # Simulate debugging scenarios
    debug_scenarios = [
        {
            "name": "Invalid Action Sequence",
            "actions": [
                {"selection": jnp.array([[True, True], [True, True]]), "operation": jnp.array(50)},  # Invalid op
                {"selection": jnp.array([[False, False], [False, False]]), "operation": jnp.array(1)},  # Empty selection
            ]
        },
        {
            "name": "Edge Case Operations",
            "actions": [
                {"selection": jnp.array([[True, False], [False, True]]), "operation": jnp.array(0)},  # Background color
                {"selection": jnp.array([[True, True], [False, False]]), "operation": jnp.array(34)},  # Submit
            ]
        }
    ]
    
    for scenario_idx, scenario in enumerate(debug_scenarios):
        print(f"\n🐛 Debugging scenario: {scenario['name']}")
        
        state, obs = arc_reset(key, config)
        visualizer.start_episode(scenario_idx)
        
        for step, action in enumerate(scenario['actions']):
            print(f"  🔍 Debug step {step}:")
            print(f"    Action: operation={action['operation']}, "
                  f"selection_sum={jnp.sum(action['selection'])}")
            
            try:
                # Environment step with error handling
                new_state, obs, reward, done, info = arc_step(state, action, config)
                
                print(f"    ✅ Step successful: reward={reward:.3f}, done={done}")
                
                # Detailed debugging visualization
                visualizer.visualize_step(
                    before_state=state,
                    action=action,
                    after_state=new_state,
                    reward=reward,
                    info=info,
                    step_num=step
                )
                
                # Check for warning conditions
                if reward < -0.1:
                    print(f"    ⚠️  Low reward detected: {reward}")
                
                if jnp.array_equal(state.working_grid, new_state.working_grid):
                    print(f"    ⚠️  No state change detected")
                
                state = new_state
                
            except Exception as e:
                print(f"    ❌ Error occurred: {e}")
                
                # Log error state for debugging
                error_info = {
                    "error": str(e),
                    "step": step,
                    "action": action,
                    "state_shape": state.working_grid.shape
                }
                
                # Continue with next action
                continue
            
            if done:
                print(f"    🏁 Episode terminated at step {step}")
                break
        
        # Generate debugging episode summary
        visualizer.visualize_episode_summary(scenario_idx)
        print(f"  🐛 Debugging scenario '{scenario['name']}' completed")
    
    # System diagnostics
    print(f"\n🔧 Running system diagnostics...")
    
    # Check visualizer status
    if hasattr(visualizer, 'get_system_status'):
        status = visualizer.get_system_status()
        print(f"  System status: {status}")
    
    # Check memory usage
    if hasattr(visualizer, 'get_memory_stats'):
        memory_stats = visualizer.get_memory_stats()
        print(f"  Memory usage: {memory_stats.get('current_mb', 0):.1f} MB")
    
    # Check performance
    if hasattr(visualizer, 'get_performance_stats'):
        perf_stats = visualizer.get_performance_stats()
        print(f"  Performance overhead: {perf_stats.get('overhead_percent', 0):.1f}%")
    
    print("🐛 Error debugging workflow completed")
    print("Check outputs/debugging/ for detailed error analysis")


def main():
    """Run all visualization workflow examples."""
    
    print("🚀 JaxARC Visualization Workflows Examples")
    print("This script demonstrates common workflows for different use cases.")
    
    try:
        # Run all workflow examples
        workflow_1_development_debugging()
        workflow_2_research_experiment()
        workflow_3_production_monitoring()
        workflow_4_performance_benchmarking()
        workflow_5_error_debugging()
        
        print("\n" + "=" * 60)
        print("🎉 All Visualization Workflows Completed!")
        print("=" * 60)
        print("\nWorkflow outputs:")
        print("- Development: outputs/development/")
        print("- Research: outputs/research/")
        print("- Production: outputs/production/")
        print("- Benchmark: outputs/benchmark/")
        print("- Debugging: outputs/debugging/")
        print("\nEach workflow demonstrates different configuration strategies")
        print("and best practices for specific use cases.")
        
    except Exception as e:
        print(f"\n❌ Error running workflows: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()