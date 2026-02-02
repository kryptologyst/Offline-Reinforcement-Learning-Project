"""Dataset generation and management for offline RL."""

import h5py
import numpy as np
import torch
import gymnasium as gym
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pickle

from ..utils.utils import set_seed, create_env


class OfflineDataset:
    """Offline dataset for reinforcement learning."""
    
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        infos: Optional[List[Dict]] = None,
    ):
        """Initialize offline dataset."""
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.dones = dones
        self.infos = infos or []
        
        self.size = len(observations)
        assert len(observations) == len(actions) == len(rewards) == len(next_observations) == len(dones)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.size
    
    def __getitem__(self, idx: Union[int, slice]) -> Dict[str, np.ndarray]:
        """Get item(s) from dataset."""
        if isinstance(idx, int):
            return {
                "observations": self.observations[idx],
                "actions": self.actions[idx],
                "rewards": self.rewards[idx],
                "next_observations": self.next_observations[idx],
                "dones": self.dones[idx],
            }
        else:
            return {
                "observations": self.observations[idx],
                "actions": self.actions[idx],
                "rewards": self.rewards[idx],
                "next_observations": self.next_observations[idx],
                "dones": self.dones[idx],
            }
    
    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a random batch from the dataset."""
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return self[indices]
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get dataset statistics."""
        return {
            "observations": {
                "mean": np.mean(self.observations, axis=0),
                "std": np.std(self.observations, axis=0),
                "min": np.min(self.observations, axis=0),
                "max": np.max(self.observations, axis=0),
            },
            "actions": {
                "mean": np.mean(self.actions, axis=0),
                "std": np.std(self.actions, axis=0),
                "min": np.min(self.actions, axis=0),
                "max": np.max(self.actions, axis=0),
            },
            "rewards": {
                "mean": np.mean(self.rewards),
                "std": np.std(self.rewards),
                "min": np.min(self.rewards),
                "max": np.max(self.rewards),
            },
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save dataset to HDF5 file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(path, "w") as f:
            f.create_dataset("observations", data=self.observations)
            f.create_dataset("actions", data=self.actions)
            f.create_dataset("rewards", data=self.rewards)
            f.create_dataset("next_observations", data=self.next_observations)
            f.create_dataset("dones", data=self.dones)
            
            if self.infos:
                f.create_dataset("infos", data=pickle.dumps(self.infos))
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "OfflineDataset":
        """Load dataset from HDF5 file."""
        path = Path(path)
        
        with h5py.File(path, "r") as f:
            observations = f["observations"][:]
            actions = f["actions"][:]
            rewards = f["rewards"][:]
            next_observations = f["next_observations"][:]
            dones = f["dones"][:]
            
            infos = None
            if "infos" in f:
                infos = pickle.loads(f["infos"][:])
        
        return cls(observations, actions, rewards, next_observations, dones, infos)


class DatasetGenerator:
    """Generate offline datasets from environment interactions."""
    
    def __init__(self, env_name: str, seed: Optional[int] = None):
        """Initialize dataset generator."""
        self.env_name = env_name
        self.seed = seed
        if seed is not None:
            set_seed(seed)
    
    def generate_random_dataset(
        self,
        num_episodes: int,
        max_episode_steps: int = 500,
        save_path: Optional[str] = None,
    ) -> OfflineDataset:
        """Generate dataset using random policy."""
        env = create_env(self.env_name, seed=self.seed)
        
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        infos = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            step = 0
            
            while not done and step < max_episode_steps:
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_obs)
                dones.append(done)
                infos.append(info)
                
                obs = next_obs
                step += 1
        
        env.close()
        
        dataset = OfflineDataset(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_observations=np.array(next_observations),
            dones=np.array(dones),
            infos=infos,
        )
        
        if save_path:
            dataset.save(save_path)
        
        return dataset
    
    def generate_expert_dataset(
        self,
        num_episodes: int,
        max_episode_steps: int = 500,
        save_path: Optional[str] = None,
    ) -> OfflineDataset:
        """Generate dataset using expert policy (simple heuristic for CartPole)."""
        env = create_env(self.env_name, seed=self.seed)
        
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        infos = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            step = 0
            
            while not done and step < max_episode_steps:
                # Simple expert policy for CartPole: move in direction of pole
                if self.env_name == "CartPole-v1":
                    action = 1 if obs[2] > 0 else 0  # Move right if pole leans right
                else:
                    action = env.action_space.sample()  # Fallback to random
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_obs)
                dones.append(done)
                infos.append(info)
                
                obs = next_obs
                step += 1
        
        env.close()
        
        dataset = OfflineDataset(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_observations=np.array(next_observations),
            dones=np.array(dones),
            infos=infos,
        )
        
        if save_path:
            dataset.save(save_path)
        
        return dataset
    
    def generate_epsilon_greedy_dataset(
        self,
        num_episodes: int,
        epsilon: float = 0.1,
        max_episode_steps: int = 500,
        save_path: Optional[str] = None,
    ) -> OfflineDataset:
        """Generate dataset using epsilon-greedy policy."""
        env = create_env(self.env_name, seed=self.seed)
        
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        infos = []
        
        # Simple Q-table for epsilon-greedy policy
        q_table = {}
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            step = 0
            
            while not done and step < max_episode_steps:
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    # Use Q-table to select best action
                    state_key = tuple(np.round(obs, 2))
                    if state_key not in q_table:
                        q_table[state_key] = np.zeros(env.action_space.n)
                    action = np.argmax(q_table[state_key])
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update Q-table
                if not done:
                    next_state_key = tuple(np.round(next_obs, 2))
                    if next_state_key not in q_table:
                        q_table[next_state_key] = np.zeros(env.action_space.n)
                    
                    current_q = q_table[state_key][action]
                    next_q = np.max(q_table[next_state_key])
                    q_table[state_key][action] = current_q + 0.1 * (reward + 0.99 * next_q - current_q)
                
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_obs)
                dones.append(done)
                infos.append(info)
                
                obs = next_obs
                step += 1
        
        env.close()
        
        dataset = OfflineDataset(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_observations=np.array(next_observations),
            dones=np.array(dones),
            infos=infos,
        )
        
        if save_path:
            dataset.save(save_path)
        
        return dataset
