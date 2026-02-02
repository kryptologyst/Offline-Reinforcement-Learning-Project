"""Streamlit demo for offline RL project."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from pathlib import Path
import json

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.buffers.dataset import DatasetGenerator, OfflineDataset
from src.algorithms.offline_rl import BehaviorCloning, BCQ, CQL, IQL
from src.train.trainer import OfflineRLTrainer
from src.eval.evaluator import OfflineRLEvaluator


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Offline Reinforcement Learning Demo",
        page_icon="🤖",
        layout="wide",
    )
    
    st.title("🤖 Offline Reinforcement Learning Demo")
    st.markdown("""
    This demo showcases advanced offline RL algorithms including Behavior Cloning (BC), 
    Batch Constrained Q-learning (BCQ), Conservative Q-Learning (CQL), and Implicit Q-Learning (IQL).
    
    **⚠️ DISCLAIMER: This is a research/educational demo. NOT for production control of real systems.**
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Environment",
        ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"],
        index=0,
    )
    
    # Algorithm selection
    algorithm_name = st.sidebar.selectbox(
        "Algorithm",
        ["bc", "bcq", "cql", "iql"],
        index=0,
    )
    
    # Dataset configuration
    st.sidebar.subheader("Dataset Configuration")
    num_episodes = st.sidebar.slider("Number of Episodes", 100, 2000, 1000)
    behavior_policy = st.sidebar.selectbox(
        "Behavior Policy",
        ["random", "expert", "epsilon_greedy"],
        index=0,
    )
    
    # Training configuration
    st.sidebar.subheader("Training Configuration")
    num_epochs = st.sidebar.slider("Training Epochs", 100, 2000, 500)
    batch_size = st.sidebar.slider("Batch Size", 32, 512, 256)
    learning_rate = st.sidebar.slider("Learning Rate", 1e-5, 1e-2, 3e-4, format="%.2e")
    
    # Create config
    config = Config()
    config.env.name = env_name
    config.algorithm.name = algorithm_name
    config.dataset.num_episodes = num_episodes
    config.dataset.behavior_policy = behavior_policy
    config.training.num_epochs = num_epochs
    config.algorithm.batch_size = batch_size
    config.algorithm.learning_rate = learning_rate
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Training", "Evaluation", "Visualization"])
    
    with tab1:
        show_dataset_tab(config)
    
    with tab2:
        show_training_tab(config)
    
    with tab3:
        show_evaluation_tab(config)
    
    with tab4:
        show_visualization_tab(config)


def show_dataset_tab(config: Config):
    """Show dataset generation and analysis tab."""
    st.header("📊 Dataset Generation & Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Generate Dataset")
        
        if st.button("Generate New Dataset", type="primary"):
            with st.spinner("Generating dataset..."):
                generator = DatasetGenerator(config.env.name, seed=42)
                
                if config.dataset.behavior_policy == "random":
                    dataset = generator.generate_random_dataset(
                        num_episodes=config.dataset.num_episodes,
                        save_path=config.dataset.save_path,
                    )
                elif config.dataset.behavior_policy == "expert":
                    dataset = generator.generate_expert_dataset(
                        num_episodes=config.dataset.num_episodes,
                        save_path=config.dataset.save_path,
                    )
                elif config.dataset.behavior_policy == "epsilon_greedy":
                    dataset = generator.generate_epsilon_greedy_dataset(
                        num_episodes=config.dataset.num_episodes,
                        epsilon=config.dataset.epsilon,
                        save_path=config.dataset.save_path,
                    )
                
                st.success(f"Dataset generated with {len(dataset)} samples!")
                st.session_state.dataset = dataset
    
    with col2:
        st.subheader("Load Existing Dataset")
        
        if st.button("Load Dataset"):
            if Path(config.dataset.save_path).exists():
                dataset = OfflineDataset.load(config.dataset.save_path)
                st.success(f"Dataset loaded with {len(dataset)} samples!")
                st.session_state.dataset = dataset
            else:
                st.error("No existing dataset found. Please generate one first.")
    
    # Dataset analysis
    if "dataset" in st.session_state:
        dataset = st.session_state.dataset
        
        st.subheader("Dataset Statistics")
        
        # Basic stats
        stats = dataset.get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(dataset))
        with col2:
            st.metric("Mean Reward", f"{stats['rewards']['mean']:.2f}")
        with col3:
            st.metric("Reward Std", f"{stats['rewards']['std']:.2f}")
        
        # Reward distribution
        st.subheader("Reward Distribution")
        fig = px.histogram(
            x=dataset.rewards,
            nbins=50,
            title="Reward Distribution",
            labels={"x": "Reward", "y": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Action distribution
        st.subheader("Action Distribution")
        action_counts = pd.Series(dataset.actions).value_counts().sort_index()
        fig = px.bar(
            x=action_counts.index,
            y=action_counts.values,
            title="Action Distribution",
            labels={"x": "Action", "y": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)


def show_training_tab(config: Config):
    """Show training tab."""
    st.header("🏋️ Training")
    
    if "dataset" not in st.session_state:
        st.warning("Please generate or load a dataset first.")
        return
    
    dataset = st.session_state.dataset
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Configuration")
        st.json({
            "Algorithm": config.algorithm.name,
            "Environment": config.env.name,
            "Dataset Size": len(dataset),
            "Epochs": config.training.num_epochs,
            "Batch Size": config.algorithm.batch_size,
            "Learning Rate": config.algorithm.learning_rate,
        })
    
    with col2:
        st.subheader("Start Training")
        
        if st.button("Start Training", type="primary"):
            with st.spinner("Training in progress..."):
                # Create trainer
                trainer = OfflineRLTrainer(config)
                
                # Train
                results = trainer.train(dataset)
                
                st.session_state.training_results = results
                st.session_state.trained_algorithm = trainer.algorithm
                
                st.success("Training completed!")
    
    # Training progress
    if "training_results" in st.session_state:
        results = st.session_state.training_results
        
        st.subheader("Training Progress")
        
        # Plot training metrics
        training_metrics = results["training_metrics"]
        
        if training_metrics:
            # Convert to DataFrame for plotting
            df = pd.DataFrame(training_metrics)
            
            # Plot training losses
            fig = make_subplots(
                rows=len(df.columns),
                cols=1,
                subplot_titles=list(df.columns),
                vertical_spacing=0.05,
            )
            
            for i, col in enumerate(df.columns):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(df))),
                        y=df[col],
                        mode="lines",
                        name=col,
                    ),
                    row=i + 1,
                    col=1,
                )
            
            fig.update_layout(
                height=200 * len(df.columns),
                title="Training Metrics",
                showlegend=False,
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Plot evaluation metrics
        if "eval_metrics" in results and results["eval_metrics"]:
            eval_df = pd.DataFrame(results["eval_metrics"])
            
            fig = px.line(
                eval_df,
                x=list(range(len(eval_df))),
                y="mean_return",
                title="Evaluation Returns",
                labels={"x": "Evaluation Step", "y": "Mean Return"},
            )
            st.plotly_chart(fig, use_container_width=True)


def show_evaluation_tab(config: Config):
    """Show evaluation tab."""
    st.header("📈 Evaluation")
    
    if "trained_algorithm" not in st.session_state:
        st.warning("Please train a model first.")
        return
    
    algorithm = st.session_state.trained_algorithm
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evaluation Configuration")
        num_eval_episodes = st.slider("Evaluation Episodes", 5, 50, 10)
        config.evaluation.num_eval_episodes = num_eval_episodes
    
    with col2:
        st.subheader("Run Evaluation")
        
        if st.button("Evaluate Model", type="primary"):
            with st.spinner("Evaluating model..."):
                evaluator = OfflineRLEvaluator(config)
                eval_results = evaluator.evaluate(algorithm)
                evaluator.close()
                
                st.session_state.eval_results = eval_results
                st.success("Evaluation completed!")
    
    # Evaluation results
    if "eval_results" in st.session_state:
        results = st.session_state.eval_results
        
        st.subheader("Evaluation Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Return", f"{results['mean_return']:.2f}")
        with col2:
            st.metric("Return Std", f"{results['return_std']:.2f}")
        with col3:
            st.metric("Success Rate", f"{results['success_rate']:.2f}")
        with col4:
            st.metric("Mean Episode Length", f"{results['mean_episode_length']:.1f}")
        
        # Confidence interval
        st.info(f"95% Confidence Interval: [{results['return_ci_low']:.2f}, {results['return_ci_high']:.2f}]")


def show_visualization_tab(config: Config):
    """Show visualization tab."""
    st.header("🎨 Visualization")
    
    if "trained_algorithm" not in st.session_state:
        st.warning("Please train a model first.")
        return
    
    algorithm = st.session_state.trained_algorithm
    
    st.subheader("Policy Visualization")
    
    # Create environment for visualization
    import gymnasium as gym
    env = gym.make(config.env.name, render_mode="rgb_array")
    
    # Run episode and collect data
    obs, _ = env.reset(seed=42)
    episode_data = {
        "observations": [obs],
        "actions": [],
        "rewards": [],
        "timesteps": [0],
    }
    
    total_reward = 0
    timestep = 0
    
    for _ in range(200):  # Max episode length
        action = algorithm.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        episode_data["actions"].append(action)
        episode_data["rewards"].append(reward)
        episode_data["observations"].append(obs)
        episode_data["timesteps"].append(timestep + 1)
        
        total_reward += reward
        timestep += 1
        
        if done:
            break
    
    env.close()
    
    # Plot episode trajectory
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Actions", "Rewards", "Observations", "Cumulative Reward"],
        vertical_spacing=0.1,
    )
    
    # Actions
    fig.add_trace(
        go.Scatter(
            x=episode_data["timesteps"][:-1],
            y=episode_data["actions"],
            mode="lines+markers",
            name="Actions",
        ),
        row=1,
        col=1,
    )
    
    # Rewards
    fig.add_trace(
        go.Scatter(
            x=episode_data["timesteps"][:-1],
            y=episode_data["rewards"],
            mode="lines+markers",
            name="Rewards",
        ),
        row=1,
        col=2,
    )
    
    # Observations (first dimension)
    obs_array = np.array(episode_data["observations"])
    fig.add_trace(
        go.Scatter(
            x=episode_data["timesteps"],
            y=obs_array[:, 0],
            mode="lines",
            name="Obs[0]",
        ),
        row=2,
        col=1,
    )
    
    # Cumulative reward
    cumulative_rewards = np.cumsum(episode_data["rewards"])
    fig.add_trace(
        go.Scatter(
            x=episode_data["timesteps"][:-1],
            y=cumulative_rewards,
            mode="lines",
            name="Cumulative Reward",
        ),
        row=2,
        col=2,
    )
    
    fig.update_layout(
        height=600,
        title=f"Episode Trajectory (Total Reward: {total_reward:.2f})",
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Episode summary
    st.subheader("Episode Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reward", f"{total_reward:.2f}")
    with col2:
        st.metric("Episode Length", timestep)
    with col3:
        st.metric("Mean Reward", f"{np.mean(episode_data['rewards']):.2f}")


if __name__ == "__main__":
    main()
