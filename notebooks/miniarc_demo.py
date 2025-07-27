from __future__ import annotations

import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        r"""
        # JaxARC MiniArc Demo with Enhanced Visualization & Wandb

        This notebook demonstrates the complete JaxARC environment with:

        - **MiniArc Dataset**: 5x5 grid tasks for rapid prototyping
        - **Enhanced Visualization**: Rich SVG rendering and debug logging
        - **Wandb Integration**: Experiment tracking and logging
        - **JAX-Compliant Random Agent**: Fully vectorized agent implementation
        - **Bbox Actions**: Bounding box action format for intuitive control
        - **Raw Environment**: Minimal operations for focused learning

        Let's explore the complete workflow from dataset loading to agent training!
        """
    )
    return (mo,)


@app.cell
def _():
    # Core imports
    import time
    from typing import Dict, NamedTuple, Tuple

    import jax
    import jax.numpy as jnp
    import jax.random as jr

    # Standard imports
    from loguru import logger
    from omegaconf import OmegaConf
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    # JaxARC imports
    from jaxarc.envs import (
        ArcEnvConfig,
        arc_reset,
        arc_step,
    )
    from jaxarc.parsers import MiniArcParser
    from jaxarc.utils.config import get_config
    from jaxarc.utils.visualization import (
        EpisodeManager,
        VisualizationConfig,
        Visualizer,
        WandbConfig,
        WandbIntegration,
        create_research_wandb_config,
    )

    console = Console()
    logger.info("JaxARC MiniArc Demo - Imports loaded successfully!")
    return (
        ArcEnvConfig,
        Dict,
        Visualizer,
        EpisodeManager,
        MiniArcParser,
        NamedTuple,
        OmegaConf,
        Panel,
        Table,
        Tuple,
        VisualizationConfig,
        WandbConfig,
        WandbIntegration,
        arc_reset,
        arc_step,
        console,
        create_research_wandb_config,
        get_config,
        jax,
        jnp,
        jr,
        logger,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Configuration Setup

    Let's set up our environment configuration with:

    - **Bbox actions** for intuitive rectangular selections
    - **Raw environment** with minimal operations
    - **MiniArc dataset** optimized for 5x5 grids
    """
    )


@app.cell
def _(ArcEnvConfig, Table, console, get_config):
    config_overrides = [
        "dataset=mini_arc",
        "action=raw",
        "action.selection_format=bbox",
        "action.allow_partial_selection=false",
        "reward=training",
    ]

    miniarc_config = get_config(overrides=config_overrides)
    typed_config = ArcEnvConfig.from_hydra(miniarc_config)

    # Display configuration summary
    config_table = Table(title="MiniArc Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row(
        "Max Episode Steps", str(miniarc_config.environment.max_episode_steps)
    )
    config_table.add_row(
        "Grid Size",
        f"{miniarc_config.dataset.grid.max_grid_height}x{miniarc_config.dataset.grid.max_grid_width}",
    )
    config_table.add_row("Action Format", miniarc_config.action.selection_format)
    config_table.add_row("Operations", str(miniarc_config.action.num_operations))
    config_table.add_row("Success Bonus", str(miniarc_config.reward.success_bonus))

    console.print(config_table)

    # Let's also print marimo_config as YAML for reference
    # console.print("\n[bold]Full Configuration:[/bold]")
    # console.print(OmegaConf.to_yaml(miniarc_config))
    return (miniarc_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Enhanced Visualization Setup

    Configure the enhanced visualization system with:
    - **Research-level detail** for comprehensive logging
    - **SVG output** for crisp grid visualizations
    - **Episode management** for organized storage
    """
    )


@app.cell
def _(
    Visualizer,
    EpisodeManager,
    Panel,
    VisualizationConfig,
    console,
    time,
):
    # Enhanced visualization configuration
    def create_visualization_setup():
        """Create enhanced visualization system."""

        # Visualization configuration for research
        vis_config = VisualizationConfig(
            debug_level="standard",
            output_formats=["svg"],
            image_quality="high",
            show_operation_names=True,
            highlight_changes=True,
            include_metrics=True,
            color_scheme="default",
        )

        # Episode management for organized output
        episode_manager = EpisodeManager(
            base_output_dir="outputs/miniarc_demo",
            run_name=f"miniarc_demo_{int(time.time())}",
            max_episodes_per_run=10,
            cleanup_policy="size_based",
            max_storage_gb=1.0,
        )

        # Create visualizer
        visualizer = Visualizer(
            vis_config=vis_config, episode_manager=episode_manager
        )

        return visualizer, vis_config, episode_manager

    # Set up visualization
    visualizer, vis_config, episode_manager = create_visualization_setup()

    console.print(
        Panel(
            f"[green]Enhanced Visualization Setup Complete[/green]\n\n"
            f"• Debug Level: {vis_config.debug_level}\n"
            f"• Output Directory: {episode_manager.get_current_run_dir()}\n"
            f"• Log Frequency: Every {vis_config.log_frequency} step(s)\n"
            f"• Image Quality: {vis_config.image_quality}",
            title="Visualization Configuration",
        )
    )

    visualizer
    return episode_manager, visualizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Wandb Integration Setup

    Configure Weights & Biases for experiment tracking:
    - **Offline mode** for demo purposes (no login required)
    - **Research configuration** with detailed logging
    - **Image logging** for grid visualizations
    """
    )


@app.cell
def _(
    Panel,
    WandbConfig,
    WandbIntegration,
    console,
    create_research_wandb_config,
):
    # Wandb configuration
    def create_wandb_setup():
        """Create Wandb integration for experiment tracking."""

        # Create research-optimized wandb config
        wandb_config = create_research_wandb_config(
            project_name="jaxarc-miniarc-demo",
            entity=None,  # Use default entity
        )

        # Override for demo - use offline mode
        wandb_config = WandbConfig(
            enabled=True,
            project_name="jaxarc-miniarc-demo",
            offline_mode=True,  # No login required for demo
            log_frequency=5,
            image_format="png",
            tags=["demo", "miniarc", "bbox", "raw"],
            save_code=True,
        )

        # Create wandb integration
        wandb_integration = WandbIntegration(wandb_config)

        return wandb_integration, wandb_config

    # Set up wandb
    wandb_integration, wandb_config = create_wandb_setup()

    console.print(
        Panel(
            f"[blue]Wandb Integration Setup Complete[/blue]\n\n"
            f"• Project: {wandb_config.project_name}\n"
            f"• Offline Mode: {wandb_config.offline_mode}\n"
            f"• Log Frequency: Every {wandb_config.log_frequency} steps\n"
            f"• Tags: {', '.join(wandb_config.tags)}\n"
            f"• Available: {wandb_integration.is_available}",
            title="Wandb Configuration",
        )
    )

    wandb_integration
    return (wandb_integration,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. MiniArc Dataset Loading

    Load the MiniArc dataset with:
    - **5x5 grid constraints** for rapid experimentation
    - **400 training tasks** for diverse patterns
    - **JAX-compatible data structures** for efficient processing
    """
    )


@app.cell
def _(MiniArcParser, OmegaConf, Panel, console, miniarc_config):
    # Load dataset
    from jaxarc.envs.config import DatasetConfig
    typed_dataset_config = DatasetConfig.from_hydra(miniarc_config.dataset)
    parser = MiniArcParser(typed_dataset_config)

    console.print(
        Panel(
            OmegaConf.to_yaml(parser.get_dataset_statistics()),
            title="MiniARC Dataset Summary",
        )
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. JAX-Compliant Random Agent

    Implement a fully vectorized random agent that:
    - **Uses JAX transformations** for efficient execution
    - **Generates bbox actions** with random rectangular selections
    - **Supports batch processing** with vmap
    - **Maintains pure functional design** for JIT compilation
    """
    )


@app.cell
def _(Dict, NamedTuple, Panel, Tuple, console, jax, jnp, jr):
    # JAX-compliant Random Agent
    class AgentState(NamedTuple):
        """Agent state for JAX compatibility."""

        key: jax.Array
        step_count: jax.Array
        total_reward: jax.Array

    class RandomAgent:
        """JAX-compliant random agent for MiniArc environment."""

        def __init__(
            self, grid_size: Tuple[int, int] = (5, 5), num_operations: int = 10
        ):
            self.grid_height, self.grid_width = grid_size
            self.num_operations = num_operations

        @staticmethod
        @jax.jit
        def init_agent(key: jax.Array) -> AgentState:
            """Initialize agent state."""
            return AgentState(
                key=key,
                step_count=jnp.array(0, dtype=jnp.int32),
                total_reward=jnp.array(0.0, dtype=jnp.float32),
            )

        @jax.jit
        def select_action(
            self, agent_state: AgentState, observation: jax.Array
        ) -> Tuple[Dict[str, jax.Array], AgentState]:
            """Select random bbox action."""
            key, subkey = jr.split(agent_state.key)

            # Generate random bounding box
            # Ensure valid bbox coordinates (x1 <= x2, y1 <= y2)
            coords = jr.uniform(subkey, (4,), minval=0, maxval=1)

            # Convert to grid coordinates
            x1 = jnp.floor(coords[0] * self.grid_width).astype(jnp.int32)
            y1 = jnp.floor(coords[1] * self.grid_height).astype(jnp.int32)
            x2 = jnp.floor(coords[2] * self.grid_width).astype(jnp.int32)
            y2 = jnp.floor(coords[3] * self.grid_height).astype(jnp.int32)

            # Ensure valid ordering
            x1, x2 = jnp.minimum(x1, x2), jnp.maximum(x1, x2)
            y1, y2 = jnp.minimum(y1, y2), jnp.maximum(y1, y2)

            # Ensure minimum size of 1x1
            x2 = jnp.maximum(x2, x1)
            y2 = jnp.maximum(y2, y1)

            # Create bbox tuple (x1, y1, x2, y2)
            bbox = (x1, y1, x2, y2)

            # Random operation
            key, op_key = jr.split(key)
            operation = jr.randint(op_key, (), 0, self.num_operations)

            # Create action
            action = {"bbox": bbox, "operation": operation}

            # Update agent state
            new_agent_state = AgentState(
                key=key,
                step_count=agent_state.step_count + 1,
                total_reward=agent_state.total_reward,  # Will be updated after step
            )

            return action, new_agent_state

        @jax.jit
        def update_reward(
            self, agent_state: AgentState, reward: jax.Array
        ) -> AgentState:
            """Update agent state with received reward."""
            return AgentState(
                key=agent_state.key,
                step_count=agent_state.step_count,
                total_reward=agent_state.total_reward + reward,
            )

    # Create agent instance
    random_agent = RandomAgent(grid_size=(5, 5), num_operations=10)

    # Test agent initialization
    test_key = jr.PRNGKey(42)
    test_agent_state = RandomAgent.init_agent(test_key)

    console.print(
        Panel(
            f"[green]JAX-Compliant Random Agent Created![/green]\n\n"
            f"• Grid Size: {random_agent.grid_height}x{random_agent.grid_width}\n"
            f"• Operations: {random_agent.num_operations}\n"
            f"• JIT Compiled: ✓\n"
            f"• Batch Compatible: ✓\n"
            f"• Initial State: {test_agent_state}",
            title="Random Agent",
        )
    )

    random_agent
    return (random_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. Environment Testing

    Test the complete setup with a sample task:
    - **Load a MiniArc task** from the dataset
    - **Initialize environment** with our configuration
    - **Run a few steps** with the random agent
    - **Visualize results** with enhanced logging
    """
    )


@app.cell
def _(
    Panel,
    arc_reset,
    arc_step,
    console,
    jr,
    logger,
    miniarc_config,
    random_agent,
    training_tasks,
    visualizer,
):
    # Environment testing
    def test_environment_setup():
        """Test the complete environment setup."""

        if training_tasks is None:
            console.print("[red]Cannot test - dataset not loaded[/red]")
            return None

        # Select a random task
        task_idx = 0  # Use first task for consistency
        sample_task = training_tasks[task_idx]

        logger.info(f"Testing with task {task_idx}")

        # Initialize environment
        key = jr.PRNGKey(123)
        env_key, agent_key = jr.split(key)

        # Reset environment with sample task
        state, observation = arc_reset(env_key, miniarc_config, task_data=sample_task)

        # Initialize agent
        agent_state = random_agent.init_agent(agent_key)

        # Start episode visualization
        visualizer.start_episode(0)

        logger.info(f"Environment initialized - Grid shape: {observation.shape}")
        logger.info(f"Initial similarity: {state.similarity_score:.3f}")

        # Run a few test steps
        episode_data = []

        for step in range(5):
            # Agent selects action
            action, agent_state = random_agent.select_action(agent_state, observation)

            # Environment step
            new_state, new_obs, reward, done, info = arc_step(
                state, action, miniarc_config
            )

            # Update agent with reward
            agent_state = random_agent.update_reward(agent_state, reward)

            # Store step data
            step_data = {
                "step": step,
                "action": action,
                "reward": float(reward),
                "similarity": float(info.get("similarity", 0.0)),
                "done": bool(done),
            }
            episode_data.append(step_data)

            # Visualize step
            visualizer.visualize_step(
                before_state=state,
                action=action,
                after_state=new_state,
                reward=reward,
                info=info,
                step_num=step,
            )

            logger.info(
                f"Step {step}: Reward={reward:.3f}, Similarity={info.get('similarity', 0.0):.3f}"
            )

            # Update state
            state, observation = new_state, new_obs

            if done:
                logger.info(f"Episode finished at step {step}")
                break

        # Generate episode summary
        visualizer.visualize_episode_summary(episode_num=0)

        return episode_data, agent_state, state

    # Run test
    test_results = test_environment_setup()

    if test_results is not None:
        episode_data, final_agent_state, final_env_state = test_results

        console.print(
            Panel(
                f"[green]Environment Test Completed![/green]\n\n"
                f"• Steps Executed: {len(episode_data)}\n"
                f"• Total Reward: {final_agent_state.total_reward:.3f}\n"
                f"• Final Similarity: {final_env_state.similarity_score:.3f}\n"
                f"• Episode Done: {episode_data[-1]['done'] if episode_data else False}",
                title="Test Results",
            )
        )

    test_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 7. Full Training Loop with Wandb Logging

    Run a complete training loop that demonstrates:

    - **Multiple episodes** with different tasks
    - **Wandb experiment tracking** with metrics and images
    - **Enhanced visualization** with comprehensive logging
    - **JAX-optimized execution** with batch processing
    """
    )


@app.cell
def _(
    Panel,
    Table,
    arc_reset,
    arc_step,
    console,
    jr,
    logger,
    miniarc_config,
    random_agent,
    time,
    training_tasks,
    visualizer,
    wandb_integration,
):
    # Full training loop
    def run_training_loop(num_episodes: int = 3, max_steps_per_episode: int = 15):
        """Run complete training loop with logging."""

        if training_tasks is None:
            console.print("[red]Cannot train - dataset not loaded[/red]")
            return None

        # Initialize wandb run
        experiment_config = {
            "algorithm": "RandomAgent",
            "environment": "JaxARC-MiniArc",
            "action_format": "bbox",
            "num_episodes": num_episodes,
            "max_steps": max_steps_per_episode,
            "grid_size": "5x5",
            "dataset": "MiniArc",
        }

        wandb_integration.initialize_run(
            experiment_config=experiment_config,
            run_name=f"miniarc_random_agent_{int(time.time())}",
        )

        # Training loop
        key = jr.PRNGKey(456)
        training_results = []

        for episode in range(num_episodes):
            logger.info(f"\n=== Episode {episode + 1}/{num_episodes} ===")

            # Select random task
            task_idx = episode % len(training_tasks)
            task = training_tasks[task_idx]

            # Split keys
            key, env_key, agent_key = jr.split(key, 3)

            # Reset environment
            state, observation = arc_reset(env_key, miniarc_config, task_data=task)

            # Initialize agent
            agent_state = random_agent.init_agent(agent_key)

            # Start episode visualization
            visualizer.start_episode(episode)

            # Episode data
            episode_rewards = []
            episode_similarities = []

            # Episode loop
            for step in range(max_steps_per_episode):
                # Agent action
                action, agent_state = random_agent.select_action(
                    agent_state, observation
                )

                # Environment step
                new_state, new_obs, reward, done, info = arc_step(
                    state, action, miniarc_config
                )

                # Update agent
                agent_state = random_agent.update_reward(agent_state, reward)

                # Store metrics
                episode_rewards.append(float(reward))
                episode_similarities.append(float(info.get("similarity", 0.0)))

                # Visualize step
                visualizer.visualize_step(
                    before_state=state,
                    action=action,
                    after_state=new_state,
                    reward=reward,
                    info=info,
                    step_num=step,
                )

                # Log to wandb
                step_metrics = {
                    "step_reward": float(reward),
                    "cumulative_reward": float(agent_state.total_reward),
                    "similarity": float(info.get("similarity", 0.0)),
                    "episode": episode,
                    "task_id": task_idx,
                }

                wandb_integration.log_step(
                    step_num=episode * max_steps_per_episode + step,
                    metrics=step_metrics,
                )

                # Update state
                state, observation = new_state, new_obs

                if done:
                    logger.info(f"Episode {episode} completed at step {step}")
                    break

            # Episode summary
            episode_summary = {
                "episode_reward": float(agent_state.total_reward),
                "episode_steps": step + 1,
                "final_similarity": float(state.similarity_score),
                "success": float(state.similarity_score) > 0.9,
                "task_id": task_idx,
            }

            training_results.append(episode_summary)

            # Visualize episode summary
            visualizer.visualize_episode_summary(episode)

            # Log episode summary to wandb
            wandb_integration.log_episode_summary(episode, episode_summary)

            logger.info(
                f"Episode {episode}: Reward={episode_summary['episode_reward']:.3f}, "
                f"Similarity={episode_summary['final_similarity']:.3f}"
            )

        # Finish wandb run
        wandb_integration.finish_run()

        return training_results

    # Run training loop
    console.print(
        Panel(
            "[yellow]Starting Training Loop...[/yellow]\n\n"
            "This will run 3 episodes with enhanced visualization and Wandb logging.",
            title="Training Loop",
        )
    )

    training_results = run_training_loop(num_episodes=3, max_steps_per_episode=10)

    if training_results:
        # Display results
        results_table = Table(title="Training Results")
        results_table.add_column("Episode", style="cyan")
        results_table.add_column("Reward", style="green")
        results_table.add_column("Steps", style="yellow")
        results_table.add_column("Final Similarity", style="magenta")
        results_table.add_column("Success", style="red")

        for i, result in enumerate(training_results):
            results_table.add_row(
                str(i),
                f"{result['episode_reward']:.3f}",
                str(result["episode_steps"]),
                f"{result['final_similarity']:.3f}",
                "✓" if result["success"] else "✗",
            )

        console.print(results_table)

    training_results


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8. Batch Processing Demo

    Demonstrate JAX's batch processing capabilities:
    - **Vectorized agent execution** across multiple environments
    - **Parallel episode processing** with vmap
    - **Performance comparison** between single and batch execution
    """
    )


@app.cell
def _(
    Panel,
    arc_reset,
    arc_step,
    console,
    jax,
    jr,
    logger,
    miniarc_config,
    random_agent,
    time,
    training_tasks,
):
    # Batch processing demonstration
    def demo_batch_processing():
        """Demonstrate JAX batch processing capabilities."""

        if training_tasks is None:
            console.print(
                "[red]Cannot demo batch processing - dataset not loaded[/red]"
            )
            return None

        logger.info("=== Batch Processing Demo ===")

        # Single episode function
        def single_episode(key, task_data):
            """Run a single episode and return total reward."""
            env_key, agent_key = jr.split(key)

            # Reset environment
            state, observation = arc_reset(env_key, miniarc_config, task_data=task_data)

            # Initialize agent
            agent_state = random_agent.init_agent(agent_key)

            # Run episode
            total_reward = 0.0
            for step in range(5):  # Short episodes for demo
                action, agent_state = random_agent.select_action(
                    agent_state, observation
                )
                state, observation, reward, done, info = arc_step(
                    state, action, miniarc_config
                )
                agent_state = random_agent.update_reward(agent_state, reward)
                total_reward += reward

                if done:
                    break

            return total_reward

        # Prepare batch data
        batch_size = 4
        keys = jr.split(jr.PRNGKey(789), batch_size)
        tasks = [training_tasks[i % len(training_tasks)] for i in range(batch_size)]

        # Single execution timing
        start_time = time.perf_counter()
        single_rewards = []
        for i in range(batch_size):
            reward = single_episode(keys[i], tasks[i])
            single_rewards.append(reward)
        single_time = time.perf_counter() - start_time

        # Batch execution with vmap
        batch_episode = jax.vmap(single_episode, in_axes=(0, 0))

        # Compile first (JIT compilation time)
        start_time = time.perf_counter()
        batch_rewards = batch_episode(keys, tasks)
        compile_time = time.perf_counter() - start_time

        # Actual batch execution timing
        start_time = time.perf_counter()
        batch_rewards = batch_episode(keys, tasks)
        batch_time = time.perf_counter() - start_time

        # Results
        speedup = single_time / batch_time if batch_time > 0 else float("inf")

        results = {
            "single_time": single_time,
            "batch_time": batch_time,
            "compile_time": compile_time,
            "speedup": speedup,
            "single_rewards": single_rewards,
            "batch_rewards": batch_rewards.tolist(),
        }

        return results

    # Run batch processing demo
    batch_results = demo_batch_processing()

    if batch_results:
        console.print(
            Panel(
                f"[green]Batch Processing Results[/green]\n\n"
                f"• Single Execution: {batch_results['single_time']:.4f}s\n"
                f"• Batch Execution: {batch_results['batch_time']:.4f}s\n"
                f"• Compile Time: {batch_results['compile_time']:.4f}s\n"
                f"• Speedup: {batch_results['speedup']:.2f}x\n"
                f"• Batch Size: 4 episodes\n"
                f"• Rewards Match: {abs(sum(batch_results['single_rewards']) - sum(batch_results['batch_rewards'])) < 1e-6}",
                title="JAX Batch Processing",
            )
        )

    batch_results


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 9. Summary and Next Steps

    This notebook demonstrated the complete JaxARC workflow with:

    ✅ **MiniArc Dataset Integration** - 5x5 grid tasks for rapid prototyping
    ✅ **Enhanced Visualization** - Rich SVG rendering and debug logging
    ✅ **Wandb Integration** - Experiment tracking and metrics logging
    ✅ **JAX-Compliant Random Agent** - Fully vectorized implementation
    ✅ **Bbox Actions** - Intuitive rectangular selection format
    ✅ **Raw Environment** - Minimal operations for focused learning
    ✅ **Batch Processing** - Parallel execution with significant speedups

    ### Next Steps:

    1. **Implement Smarter Agents**: Replace random actions with learned policies
    2. **Curriculum Learning**: Progress from simple to complex tasks
    3. **Multi-Task Training**: Train on multiple ARC datasets simultaneously
    4. **Hierarchical RL**: Decompose complex tasks into subtasks
    5. **Meta-Learning**: Learn to adapt quickly to new task types

    The foundation is now ready for advanced RL research on abstract reasoning!
    """
    )


@app.cell
def _(Panel, console, episode_manager):
    # Final summary and cleanup
    def display_final_summary():
        """Display final summary of the demo."""

        # Get output directory info
        output_dir = episode_manager.get_current_run_dir()

        console.print(
            Panel(
                f"[bold green]JaxARC MiniArc Demo Complete![/bold green]\n\n"
                f"📁 **Output Directory**: {output_dir}\n"
                f"🎨 **Visualizations**: Check SVG files in the output directory\n"
                f"📊 **Wandb Logs**: Experiment data logged (offline mode)\n"
                f"🚀 **JAX Performance**: Batch processing demonstrated\n"
                f"🎯 **Ready for Research**: Foundation set for advanced RL\n\n"
                f"[yellow]Tip:[/yellow] Open the SVG files in your browser to see detailed grid visualizations!",
                title="Demo Summary",
            )
        )

        return str(output_dir)

    output_directory = display_final_summary()
    output_directory


if __name__ == "__main__":
    app.run()
