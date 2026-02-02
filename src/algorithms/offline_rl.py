"""Advanced offline RL algorithms implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

from ..utils.utils import create_mlp, soft_update, hard_update, get_device
from ..buffers.dataset import OfflineDataset


class BaseOfflineRLAlgorithm(ABC):
    """Base class for offline RL algorithms."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        device: str = "auto",
    ):
        """Initialize base offline RL algorithm."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = get_device(device)
        
        # Networks will be defined in subclasses
        self.actor = None
        self.critic = None
        self.target_critic = None
        
        # Optimizers
        self.actor_optimizer = None
        self.critic_optimizer = None
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        pass
    
    @abstractmethod
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select action given state."""
        pass
    
    def save(self, path: str) -> None:
        """Save model."""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])


class BehaviorCloning(BaseOfflineRLAlgorithm):
    """Behavior Cloning algorithm."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        device: str = "auto",
    ):
        """Initialize Behavior Cloning."""
        super().__init__(state_dim, action_dim, hidden_dims, learning_rate, device)
        
        # Policy network
        self.actor = create_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation="relu",
        ).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train behavior cloning policy."""
        states = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        
        # Forward pass
        predicted_actions = self.actor(states)
        
        # Compute loss (MSE for continuous actions, CrossEntropy for discrete)
        if self.action_dim == 1:  # Continuous action
            loss = F.mse_loss(predicted_actions, actions.float())
        else:  # Discrete action
            loss = F.cross_entropy(predicted_actions, actions.long())
        
        # Backward pass
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        return {"bc_loss": loss.item()}
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select action using behavior cloning policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits = self.actor(state_tensor)
            
            if self.action_dim == 1:  # Continuous action
                action = action_logits.cpu().numpy()[0]
            else:  # Discrete action
                if deterministic:
                    action = torch.argmax(action_logits, dim=1).cpu().numpy()[0]
                else:
                    action_probs = F.softmax(action_logits, dim=1)
                    action = torch.multinomial(action_probs, 1).cpu().numpy()[0]
        
        return action


class BCQ(BaseOfflineRLAlgorithm):
    """Batch Constrained Q-learning algorithm."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        threshold: float = 0.3,
        device: str = "auto",
    ):
        """Initialize BCQ."""
        super().__init__(state_dim, action_dim, hidden_dims, learning_rate, device)
        self.threshold = threshold
        
        # Generator network (VAE)
        self.generator = create_mlp(
            input_dim=state_dim + action_dim,
            output_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            activation="relu",
        ).to(self.device)
        
        # Critic networks
        self.critic = create_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation="relu",
        ).to(self.device)
        
        self.target_critic = create_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation="relu",
        ).to(self.device)
        
        # Actor network
        self.actor = create_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation="relu",
        ).to(self.device)
        
        # Optimizers
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        
        # Initialize target network
        hard_update(self.target_critic, self.critic)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train BCQ."""
        states = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device)
        
        # Train generator (VAE)
        generator_loss = self._train_generator(states, actions)
        
        # Train critic
        critic_loss = self._train_critic(states, actions, rewards, next_states, dones)
        
        # Train actor
        actor_loss = self._train_actor(states)
        
        # Update target network
        soft_update(self.target_critic, self.critic, tau=0.005)
        
        return {
            "generator_loss": generator_loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }
    
    def _train_generator(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """Train generator (VAE) network."""
        # Simplified VAE training - in practice, you'd implement proper VAE
        state_action = torch.cat([states, actions], dim=1)
        reconstructed = self.generator(state_action)
        
        loss = F.mse_loss(reconstructed, state_action)
        
        self.generator_optimizer.zero_grad()
        loss.backward()
        self.generator_optimizer.step()
        
        return loss.item()
    
    def _train_critic(self, states: torch.Tensor, actions: torch.Tensor, 
                     rewards: torch.Tensor, next_states: torch.Tensor, 
                     dones: torch.Tensor) -> float:
        """Train critic network."""
        with torch.no_grad():
            # Generate actions for next states
            next_actions = self.actor(next_states)
            next_state_action = torch.cat([next_states, next_actions], dim=1)
            target_q = self.target_critic(next_state_action)
            target_q = rewards + 0.99 * target_q * (1 - dones)
        
        current_state_action = torch.cat([states, actions], dim=1)
        current_q = self.critic(current_state_action)
        
        loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        return loss.item()
    
    def _train_actor(self, states: torch.Tensor) -> float:
        """Train actor network."""
        # Generate actions
        actions = self.actor(states)
        state_action = torch.cat([states, actions], dim=1)
        
        # Compute Q-values
        q_values = self.critic(state_action)
        
        # Maximize Q-values
        loss = -q_values.mean()
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        return loss.item()
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select action using BCQ policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Generate action using actor
            action = self.actor(state_tensor)
            
            # Apply threshold constraint (simplified)
            if not deterministic:
                noise = torch.randn_like(action) * 0.1
                action = action + noise
            
            return action.cpu().numpy()[0]


class CQL(BaseOfflineRLAlgorithm):
    """Conservative Q-Learning algorithm."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        alpha: float = 1.0,
        device: str = "auto",
    ):
        """Initialize CQL."""
        super().__init__(state_dim, action_dim, hidden_dims, learning_rate, device)
        self.alpha = alpha
        
        # Critic networks
        self.critic = create_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation="relu",
        ).to(self.device)
        
        self.target_critic = create_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation="relu",
        ).to(self.device)
        
        # Actor network
        self.actor = create_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation="relu",
        ).to(self.device)
        
        # Optimizers
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        
        # Initialize target network
        hard_update(self.target_critic, self.critic)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train CQL."""
        states = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device)
        
        # Train critic with conservative penalty
        critic_loss = self._train_critic(states, actions, rewards, next_states, dones)
        
        # Train actor
        actor_loss = self._train_actor(states)
        
        # Update target network
        soft_update(self.target_critic, self.critic, tau=0.005)
        
        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }
    
    def _train_critic(self, states: torch.Tensor, actions: torch.Tensor, 
                     rewards: torch.Tensor, next_states: torch.Tensor, 
                     dones: torch.Tensor) -> float:
        """Train critic with conservative penalty."""
        # Standard Q-learning loss
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_state_action = torch.cat([next_states, next_actions], dim=1)
            target_q = self.target_critic(next_state_action)
            target_q = rewards + 0.99 * target_q * (1 - dones)
        
        current_state_action = torch.cat([states, actions], dim=1)
        current_q = self.critic(current_state_action)
        
        q_loss = F.mse_loss(current_q, target_q)
        
        # Conservative penalty
        # Sample random actions
        random_actions = torch.rand_like(actions)
        random_state_action = torch.cat([states, random_actions], dim=1)
        random_q = self.critic(random_state_action)
        
        # Conservative penalty: minimize Q-values for random actions
        conservative_loss = self.alpha * (random_q.mean() - current_q.mean())
        
        total_loss = q_loss + conservative_loss
        
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.critic_optimizer.step()
        
        return total_loss.item()
    
    def _train_actor(self, states: torch.Tensor) -> float:
        """Train actor network."""
        actions = self.actor(states)
        state_action = torch.cat([states, actions], dim=1)
        q_values = self.critic(state_action)
        
        loss = -q_values.mean()
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        return loss.item()
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select action using CQL policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor)
            
            if not deterministic:
                noise = torch.randn_like(action) * 0.1
                action = action + noise
            
            return action.cpu().numpy()[0]


class IQL(BaseOfflineRLAlgorithm):
    """Implicit Q-Learning algorithm."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        temperature: float = 3.0,
        expectile: float = 0.8,
        device: str = "auto",
    ):
        """Initialize IQL."""
        super().__init__(state_dim, action_dim, hidden_dims, learning_rate, device)
        self.temperature = temperature
        self.expectile = expectile
        
        # Value network
        self.value_net = create_mlp(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation="relu",
        ).to(self.device)
        
        # Q-network
        self.q_net = create_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation="relu",
        ).to(self.device)
        
        # Actor network
        self.actor = create_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation="relu",
        ).to(self.device)
        
        # Optimizers
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train IQL."""
        states = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device)
        
        # Train value network
        value_loss = self._train_value(states, actions, rewards, next_states, dones)
        
        # Train Q-network
        q_loss = self._train_q(states, actions, rewards, next_states, dones)
        
        # Train actor
        actor_loss = self._train_actor(states)
        
        return {
            "value_loss": value_loss,
            "q_loss": q_loss,
            "actor_loss": actor_loss,
        }
    
    def _train_value(self, states: torch.Tensor, actions: torch.Tensor, 
                    rewards: torch.Tensor, next_states: torch.Tensor, 
                    dones: torch.Tensor) -> float:
        """Train value network using expectile regression."""
        with torch.no_grad():
            next_values = self.value_net(next_states)
            target_values = rewards + 0.99 * next_values * (1 - dones)
        
        current_values = self.value_net(states)
        
        # Expectile regression loss
        diff = target_values - current_values
        loss = torch.where(
            diff > 0,
            self.expectile * diff ** 2,
            (1 - self.expectile) * diff ** 2
        ).mean()
        
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
        return loss.item()
    
    def _train_q(self, states: torch.Tensor, actions: torch.Tensor, 
                 rewards: torch.Tensor, next_states: torch.Tensor, 
                 dones: torch.Tensor) -> float:
        """Train Q-network."""
        with torch.no_grad():
            next_values = self.value_net(next_states)
            target_q = rewards + 0.99 * next_values * (1 - dones)
        
        state_action = torch.cat([states, actions], dim=1)
        current_q = self.q_net(state_action)
        
        loss = F.mse_loss(current_q, target_q)
        
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        
        return loss.item()
    
    def _train_actor(self, states: torch.Tensor) -> float:
        """Train actor using advantage-weighted regression."""
        actions = self.actor(states)
        state_action = torch.cat([states, actions], dim=1)
        
        # Compute advantages
        values = self.value_net(states)
        q_values = self.q_net(state_action)
        advantages = q_values - values
        
        # Advantage-weighted regression
        weights = torch.exp(advantages / self.temperature)
        weights = weights / weights.mean()  # Normalize weights
        
        # Compute loss with weights
        loss = (weights * F.mse_loss(actions, actions, reduction='none')).mean()
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        return loss.item()
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select action using IQL policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor)
            
            if not deterministic:
                noise = torch.randn_like(action) * 0.1
                action = action + noise
            
            return action.cpu().numpy()[0]
