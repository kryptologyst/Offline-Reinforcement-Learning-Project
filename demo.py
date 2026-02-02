#!/usr/bin/env python3
"""Demonstration script for the offline RL project."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import Config
from src.buffers.dataset import DatasetGenerator
from src.algorithms.offline_rl import BehaviorCloning, BCQ, CQL, IQL
from src.train.trainer import OfflineRLTrainer
from src.eval.evaluator import OfflineRLEvaluator


def main():
    """Run a complete demonstration of the offline RL project."""
    print("🤖 Offline Reinforcement Learning Project Demo")
    print("=" * 50)
    
    # Create configuration
    config = Config()
    config.env.name = "CartPole-v1"
    config.dataset.num_episodes = 500  # Smaller for demo
    config.training.num_epochs = 100   # Smaller for demo
    config.training.eval_frequency = 25
    config.evaluation.num_eval_episodes = 5
    
    print(f"Environment: {config.env.name}")
    print(f"Dataset episodes: {config.dataset.num_episodes}")
    print(f"Training epochs: {config.training.num_epochs}")
    print()
    
    # Generate dataset
    print("📊 Generating offline dataset...")
    generator = DatasetGenerator(config.env.name, seed=config.training.seed)
    dataset = generator.generate_random_dataset(
        num_episodes=config.dataset.num_episodes,
        save_path=config.dataset.save_path,
    )
    print(f"Generated dataset with {len(dataset)} samples")
    
    # Print dataset statistics
    stats = dataset.get_stats()
    print(f"Reward statistics: mean={stats['rewards']['mean']:.2f}, std={stats['rewards']['std']:.2f}")
    print()
    
    # Test different algorithms
    algorithms = {
        "BC": BehaviorCloning(state_dim=4, action_dim=2),
        "BCQ": BCQ(state_dim=4, action_dim=2),
        "CQL": CQL(state_dim=4, action_dim=2),
        "IQL": IQL(state_dim=4, action_dim=2),
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"🏋️ Training {name}...")
        
        # Update config for this algorithm
        config.algorithm.name = name.lower()
        
        # Create trainer
        trainer = OfflineRLTrainer(config)
        trainer.algorithm = algorithm
        
        # Train (simplified for demo)
        data_loader = trainer._create_data_loader(dataset)
        
        for epoch in range(min(50, config.training.num_epochs)):  # Limit for demo
            epoch_metrics = trainer._train_epoch(data_loader)
            
            if epoch % 25 == 0:
                print(f"  Epoch {epoch}: {list(epoch_metrics.keys())[0]} = {list(epoch_metrics.values())[0]:.4f}")
        
        # Evaluate
        print(f"📈 Evaluating {name}...")
        evaluator = OfflineRLEvaluator(config)
        eval_results = evaluator.evaluate(algorithm)
        evaluator.close()
        
        results[name] = eval_results
        print(f"  Mean Return: {eval_results['mean_return']:.2f}")
        print(f"  Success Rate: {eval_results['success_rate']:.2f}")
        print()
    
    # Print comparison
    print("🏆 Algorithm Comparison")
    print("=" * 50)
    print(f"{'Algorithm':<10} {'Mean Return':<12} {'Success Rate':<12}")
    print("-" * 50)
    
    for name, metrics in results.items():
        print(f"{name:<10} {metrics['mean_return']:<12.2f} {metrics['success_rate']:<12.2f}")
    
    print()
    print("✅ Demo completed successfully!")
    print()
    print("Next steps:")
    print("1. Run 'streamlit run demo/app.py' for interactive demo")
    print("2. Run 'python scripts/train.py --help' for training options")
    print("3. Run 'python scripts/evaluate.py --help' for evaluation options")
    print("4. Check README.md for detailed usage instructions")


if __name__ == "__main__":
    main()
