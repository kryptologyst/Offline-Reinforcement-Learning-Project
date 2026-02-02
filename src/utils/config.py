"""Configuration management for offline RL experiments."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import yaml
from pathlib import Path


@dataclass
class EnvironmentConfig:
    """Configuration for environment setup."""
    name: str = "CartPole-v1"
    max_episode_steps: int = 500
    render_mode: Optional[str] = None
    seed: Optional[int] = None


@dataclass
class DatasetConfig:
    """Configuration for offline dataset generation."""
    num_episodes: int = 1000
    behavior_policy: str = "random"  # random, expert, epsilon_greedy
    epsilon: float = 0.1
    save_path: str = "data/offline_dataset.h5"
    load_existing: bool = False


@dataclass
class AlgorithmConfig:
    """Configuration for offline RL algorithms."""
    name: str = "bcq"  # bcq, cql, iql, bc
    learning_rate: float = 3e-4
    batch_size: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    dropout: float = 0.0
    
    # Algorithm-specific parameters
    bcq_threshold: float = 0.3
    cql_alpha: float = 1.0
    iql_temperature: float = 3.0
    iql_expectile: float = 0.8


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    num_epochs: int = 1000
    eval_frequency: int = 50
    save_frequency: int = 100
    log_frequency: int = 10
    device: str = "auto"  # auto, cpu, cuda, mps
    seed: int = 42
    deterministic: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    num_eval_episodes: int = 10
    eval_seed: int = 123
    compute_ope: bool = True
    ope_methods: List[str] = field(default_factory=lambda: ["ips", "dr", "snips"])


@dataclass
class Config:
    """Main configuration class."""
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "env": self.env.__dict__,
            "dataset": self.dataset.__dict__,
            "algorithm": self.algorithm.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(
            env=EnvironmentConfig(**config_dict.get("env", {})),
            dataset=DatasetConfig(**config_dict.get("dataset", {})),
            algorithm=AlgorithmConfig(**config_dict.get("algorithm", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        """Load config from YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
