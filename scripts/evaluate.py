#!/usr/bin/env python3
"""Evaluation script for offline RL project."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.eval.evaluator import evaluate_offline_rl, compare_algorithms
from src.algorithms.offline_rl import BehaviorCloning, BCQ, CQL, IQL
from src.utils.config import Config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate offline RL algorithms")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--algorithm", type=str, default="bcq", help="Algorithm name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--compare", action="store_true", help="Compare multiple algorithms")
    
    args = parser.parse_args()
    
    # Create config
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Update config with command line arguments
    config.env.name = args.env
    config.algorithm.name = args.algorithm
    config.evaluation.num_eval_episodes = args.episodes
    config.evaluation.eval_seed = args.seed
    
    if args.compare:
        # Compare multiple algorithms
        algorithms = {}
        
        # Create algorithms (simplified for demo)
        state_dim = 4 if args.env == "CartPole-v1" else 8
        action_dim = 2 if args.env == "CartPole-v1" else 4
        
        algorithms["BC"] = BehaviorCloning(state_dim, action_dim)
        algorithms["BCQ"] = BCQ(state_dim, action_dim)
        algorithms["CQL"] = CQL(state_dim, action_dim)
        algorithms["IQL"] = IQL(state_dim, action_dim)
        
        # Load checkpoints if provided
        if args.checkpoint:
            checkpoint_dir = Path(args.checkpoint)
            for name, algorithm in algorithms.items():
                checkpoint_path = checkpoint_dir / f"{name.lower()}_final.pt"
                if checkpoint_path.exists():
                    algorithm.load(str(checkpoint_path))
                    print(f"Loaded checkpoint for {name}")
        
        # Compare algorithms
        results = compare_algorithms(algorithms, config)
        
    else:
        # Evaluate single algorithm
        state_dim = 4 if args.env == "CartPole-v1" else 8
        action_dim = 2 if args.env == "CartPole-v1" else 4
        
        if args.algorithm.lower() == "bc":
            algorithm = BehaviorCloning(state_dim, action_dim)
        elif args.algorithm.lower() == "bcq":
            algorithm = BCQ(state_dim, action_dim)
        elif args.algorithm.lower() == "cql":
            algorithm = CQL(state_dim, action_dim)
        elif args.algorithm.lower() == "iql":
            algorithm = IQL(state_dim, action_dim)
        else:
            raise ValueError(f"Unknown algorithm: {args.algorithm}")
        
        # Load checkpoint if provided
        if args.checkpoint:
            algorithm.load(args.checkpoint)
            print(f"Loaded checkpoint from {args.checkpoint}")
        
        # Evaluate algorithm
        results = evaluate_offline_rl(algorithm, config)
        
        # Print results
        print(f"\nEvaluation Results for {args.algorithm.upper()}:")
        print(f"  Mean Return: {results['mean_return']:.2f}")
        print(f"  95% CI: [{results['return_ci_low']:.2f}, {results['return_ci_high']:.2f}]")
        print(f"  Success Rate: {results['success_rate']:.2f}")
        print(f"  Mean Episode Length: {results['mean_episode_length']:.1f}")


if __name__ == "__main__":
    main()
