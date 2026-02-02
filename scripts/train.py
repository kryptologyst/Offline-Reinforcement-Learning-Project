#!/usr/bin/env python3
"""Main training script for offline RL project."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.train.trainer import train_offline_rl
from src.utils.config import Config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train offline RL algorithms")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--algorithm", type=str, default="bcq", help="Algorithm name")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--behavior-policy", type=str, default="random", help="Behavior policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Create config
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Update config with command line arguments
    config.env.name = args.env
    config.algorithm.name = args.algorithm
    config.dataset.num_episodes = args.episodes
    config.training.num_epochs = args.epochs
    config.algorithm.batch_size = args.batch_size
    config.algorithm.learning_rate = args.lr
    config.dataset.behavior_policy = args.behavior_policy
    config.training.seed = args.seed
    config.training.device = args.device
    
    # Print configuration
    print("Configuration:")
    print(f"  Environment: {config.env.name}")
    print(f"  Algorithm: {config.algorithm.name}")
    print(f"  Episodes: {config.dataset.num_episodes}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch Size: {config.algorithm.batch_size}")
    print(f"  Learning Rate: {config.algorithm.learning_rate}")
    print(f"  Behavior Policy: {config.dataset.behavior_policy}")
    print(f"  Seed: {config.training.seed}")
    print(f"  Device: {config.training.device}")
    print()
    
    # Train
    try:
        results = train_offline_rl(config=config)
        print("Training completed successfully!")
        
        # Print final results
        if results["eval_metrics"]:
            final_eval = results["eval_metrics"][-1]
            print(f"Final evaluation return: {final_eval['mean_return']:.2f}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
