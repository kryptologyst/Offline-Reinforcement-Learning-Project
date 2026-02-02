"""Utility functions for offline RL experiments."""

import random
import numpy as np
import torch
import gymnasium as gym
from typing import Any, Dict, List, Optional, Tuple, Union


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def create_env(env_name: str, seed: Optional[int] = None, **kwargs) -> gym.Env:
    """Create and configure environment."""
    env = gym.make(env_name, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env


def normalize_rewards(rewards: np.ndarray, method: str = "z_score") -> np.ndarray:
    """Normalize rewards using specified method."""
    if method == "z_score":
        return (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
    elif method == "min_max":
        return (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards) + 1e-8)
    elif method == "none":
        return rewards
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted returns from rewards."""
    returns = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)
    return returns


def compute_advantages(
    rewards: List[float], 
    values: List[float], 
    gamma: float = 0.99, 
    lam: float = 0.95
) -> List[float]:
    """Compute GAE advantages."""
    advantages = []
    advantage = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        advantage = delta + gamma * lam * advantage
        advantages.insert(0, advantage)
    
    return advantages


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Soft update target network parameters."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """Hard update target network parameters."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
    dropout: float = 0.0,
    output_activation: Optional[str] = None,
) -> torch.nn.Module:
    """Create a multi-layer perceptron."""
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]
    
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        
        if i < len(dims) - 2:  # Don't add activation after last layer
            if activation == "relu":
                layers.append(torch.nn.ReLU())
            elif activation == "tanh":
                layers.append(torch.nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(torch.nn.LeakyReLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
    
    if output_activation == "tanh":
        layers.append(torch.nn.Tanh())
    elif output_activation == "sigmoid":
        layers.append(torch.nn.Sigmoid())
    
    return torch.nn.Sequential(*layers)


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute confidence interval for data."""
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    
    # Use t-distribution for small samples
    if n < 30:
        from scipy import stats
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_val * std / np.sqrt(n)
    else:
        # Use normal approximation for large samples
        z_val = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        margin = z_val * std / np.sqrt(n)
    
    return mean, mean - margin, mean + margin


def log_dict_to_tensorboard(writer, metrics: Dict[str, float], step: int) -> None:
    """Log dictionary of metrics to tensorboard."""
    for key, value in metrics.items():
        writer.add_scalar(key, value, step)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> Tuple[int, Dict[str, float]]:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["metrics"]
