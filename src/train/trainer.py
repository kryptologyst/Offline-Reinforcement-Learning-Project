"""Training module for offline RL algorithms."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
from pathlib import Path

from ..utils.utils import set_seed, get_device, save_checkpoint, log_dict_to_tensorboard
from ..utils.config import Config
from ..buffers.dataset import OfflineDataset
from ..algorithms.offline_rl import BaseOfflineRLAlgorithm, BehaviorCloning, BCQ, CQL, IQL


class OfflineRLTrainer:
    """Trainer for offline RL algorithms."""
    
    def __init__(self, config: Config):
        """Initialize trainer."""
        self.config = config
        
        # Set device
        self.device = get_device(config.training.device)
        
        # Set seed
        if config.training.deterministic:
            set_seed(config.training.seed)
        
        # Initialize algorithm
        self.algorithm = self._create_algorithm()
        
        # Initialize logging
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Training metrics
        self.training_metrics = []
        self.eval_metrics = []
    
    def _create_algorithm(self) -> BaseOfflineRLAlgorithm:
        """Create algorithm based on config."""
        algorithm_name = self.config.algorithm.name.lower()
        
        if algorithm_name == "bc":
            return BehaviorCloning(
                state_dim=self.config.env.name == "CartPole-v1" and 4 or 8,  # Simplified
                action_dim=self.config.env.name == "CartPole-v1" and 2 or 4,  # Simplified
                hidden_dims=self.config.algorithm.hidden_dims,
                learning_rate=self.config.algorithm.learning_rate,
                device=self.config.training.device,
            )
        elif algorithm_name == "bcq":
            return BCQ(
                state_dim=self.config.env.name == "CartPole-v1" and 4 or 8,
                action_dim=self.config.env.name == "CartPole-v1" and 2 or 4,
                hidden_dims=self.config.algorithm.hidden_dims,
                learning_rate=self.config.algorithm.learning_rate,
                threshold=self.config.algorithm.bcq_threshold,
                device=self.config.training.device,
            )
        elif algorithm_name == "cql":
            return CQL(
                state_dim=self.config.env.name == "CartPole-v1" and 4 or 8,
                action_dim=self.config.env.name == "CartPole-v1" and 2 or 4,
                hidden_dims=self.config.algorithm.hidden_dims,
                learning_rate=self.config.algorithm.learning_rate,
                alpha=self.config.algorithm.cql_alpha,
                device=self.config.training.device,
            )
        elif algorithm_name == "iql":
            return IQL(
                state_dim=self.config.env.name == "CartPole-v1" and 4 or 8,
                action_dim=self.config.env.name == "CartPole-v1" and 2 or 4,
                hidden_dims=self.config.algorithm.hidden_dims,
                learning_rate=self.config.algorithm.learning_rate,
                temperature=self.config.algorithm.iql_temperature,
                expectile=self.config.algorithm.iql_expectile,
                device=self.config.training.device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def train(self, dataset: OfflineDataset) -> Dict[str, List[float]]:
        """Train the algorithm on offline dataset."""
        print(f"Training {self.config.algorithm.name} on {len(dataset)} samples")
        
        # Create data loader
        data_loader = self._create_data_loader(dataset)
        
        # Training loop
        for epoch in tqdm(range(self.config.training.num_epochs), desc="Training"):
            epoch_metrics = self._train_epoch(data_loader)
            self.training_metrics.append(epoch_metrics)
            
            # Logging
            if epoch % self.config.training.log_frequency == 0:
                self._log_metrics(epoch_metrics, epoch)
            
            # Evaluation
            if epoch % self.config.training.eval_frequency == 0:
                eval_metrics = self._evaluate()
                self.eval_metrics.append(eval_metrics)
                print(f"Epoch {epoch}: Eval Return = {eval_metrics['mean_return']:.2f}")
            
            # Save checkpoint
            if epoch % self.config.training.save_frequency == 0:
                self._save_checkpoint(epoch)
        
        return {
            "training_metrics": self.training_metrics,
            "eval_metrics": self.eval_metrics,
        }
    
    def _create_data_loader(self, dataset: OfflineDataset) -> DataLoader:
        """Create data loader from dataset."""
        # Convert to tensors
        observations = torch.FloatTensor(dataset.observations)
        actions = torch.LongTensor(dataset.actions) if dataset.actions.dtype == np.int64 else torch.FloatTensor(dataset.actions)
        rewards = torch.FloatTensor(dataset.rewards)
        next_observations = torch.FloatTensor(dataset.next_observations)
        dones = torch.BoolTensor(dataset.dones)
        
        # Create dataset
        tensor_dataset = TensorDataset(observations, actions, rewards, next_observations, dones)
        
        # Create data loader
        data_loader = DataLoader(
            tensor_dataset,
            batch_size=self.config.algorithm.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for reproducibility
        )
        
        return data_loader
    
    def _train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {}
        
        for batch in data_loader:
            # Convert batch to dictionary
            batch_dict = {
                "observations": batch[0],
                "actions": batch[1],
                "rewards": batch[2],
                "next_observations": batch[3],
                "dones": batch[4],
            }
            
            # Train step
            step_metrics = self.algorithm.train_step(batch_dict)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
        
        # Average metrics over epoch
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate the trained policy."""
        from ..eval.evaluator import OfflineRLEvaluator
        
        evaluator = OfflineRLEvaluator(self.config)
        eval_results = evaluator.evaluate(self.algorithm)
        
        return eval_results
    
    def _log_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """Log metrics to console and tensorboard."""
        # Console logging
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {epoch}: {metric_str}")
        
        # Tensorboard logging (if available)
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(self.log_dir)
            log_dict_to_tensorboard(writer, metrics, epoch)
            writer.close()
        except ImportError:
            pass
    
    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{self.config.algorithm.name}_epoch_{epoch}.pt"
        self.algorithm.save(str(checkpoint_path))
        
        # Save config
        config_path = checkpoint_dir / f"{self.config.algorithm.name}_config.yaml"
        self.config.save(config_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        self.algorithm.load(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")


def train_offline_rl(config_path: Optional[str] = None, **kwargs) -> Dict[str, List[float]]:
    """Main training function."""
    # Load config
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config()
    
    # Update config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Generate or load dataset
    from ..buffers.dataset import DatasetGenerator, OfflineDataset
    
    if config.dataset.load_existing and Path(config.dataset.save_path).exists():
        print(f"Loading existing dataset from {config.dataset.save_path}")
        dataset = OfflineDataset.load(config.dataset.save_path)
    else:
        print("Generating new dataset...")
        generator = DatasetGenerator(config.env.name, seed=config.training.seed)
        
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
        else:
            raise ValueError(f"Unknown behavior policy: {config.dataset.behavior_policy}")
    
    # Print dataset statistics
    stats = dataset.get_stats()
    print(f"Dataset size: {len(dataset)}")
    print(f"Reward stats: mean={stats['rewards']['mean']:.2f}, std={stats['rewards']['std']:.2f}")
    
    # Train
    trainer = OfflineRLTrainer(config)
    results = trainer.train(dataset)
    
    return results
