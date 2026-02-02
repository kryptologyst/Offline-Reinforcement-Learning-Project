"""Evaluation module for offline RL algorithms."""

import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..utils.utils import set_seed, create_env, compute_confidence_interval
from ..utils.config import Config
from ..algorithms.offline_rl import BaseOfflineRLAlgorithm


class OfflineRLEvaluator:
    """Evaluator for offline RL algorithms."""
    
    def __init__(self, config: Config):
        """Initialize evaluator."""
        self.config = config
        
        # Create evaluation environment
        self.eval_env = create_env(
            config.env.name,
            seed=config.evaluation.eval_seed,
            render_mode=config.env.render_mode,
        )
        
        # Set seed for evaluation
        if config.training.deterministic:
            set_seed(config.evaluation.eval_seed)
    
    def evaluate(self, algorithm: BaseOfflineRLAlgorithm) -> Dict[str, float]:
        """Evaluate the algorithm."""
        print(f"Evaluating {algorithm.__class__.__name__}...")
        
        # Run evaluation episodes
        returns = []
        episode_lengths = []
        
        for episode in range(self.config.evaluation.num_eval_episodes):
            return_val, episode_length = self._run_episode(algorithm, episode)
            returns.append(return_val)
            episode_lengths.append(episode_length)
        
        # Compute metrics
        mean_return, ci_low, ci_high = compute_confidence_interval(np.array(returns))
        mean_length = np.mean(episode_lengths)
        
        metrics = {
            "mean_return": mean_return,
            "return_ci_low": ci_low,
            "return_ci_high": ci_high,
            "return_std": np.std(returns),
            "mean_episode_length": mean_length,
            "success_rate": self._compute_success_rate(returns),
        }
        
        # Compute OPE estimates if requested
        if self.config.evaluation.compute_ope:
            ope_metrics = self._compute_ope_estimates(algorithm)
            metrics.update(ope_metrics)
        
        return metrics
    
    def _run_episode(self, algorithm: BaseOfflineRLAlgorithm, episode: int) -> Tuple[float, int]:
        """Run a single evaluation episode."""
        obs, _ = self.eval_env.reset(seed=self.config.evaluation.eval_seed + episode)
        total_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action = algorithm.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.eval_env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            episode_length += 1
            
            # Safety check for infinite episodes
            if episode_length > 1000:
                break
        
        return total_reward, episode_length
    
    def _compute_success_rate(self, returns: List[float]) -> float:
        """Compute success rate based on returns."""
        # Define success threshold based on environment
        if self.config.env.name == "CartPole-v1":
            success_threshold = 195.0  # CartPole solved threshold
        else:
            success_threshold = np.mean(returns) + np.std(returns)  # Heuristic
        
        successful_episodes = sum(1 for r in returns if r >= success_threshold)
        return successful_episodes / len(returns)
    
    def _compute_ope_estimates(self, algorithm: BaseOfflineRLAlgorithm) -> Dict[str, float]:
        """Compute Offline Policy Evaluation estimates."""
        # Load dataset for OPE
        from ..buffers.dataset import OfflineDataset
        
        if Path(self.config.dataset.save_path).exists():
            dataset = OfflineDataset.load(self.config.dataset.save_path)
        else:
            print("Warning: Dataset not found for OPE. Skipping OPE estimates.")
            return {}
        
        ope_metrics = {}
        
        for method in self.config.evaluation.ope_methods:
            if method == "ips":
                ope_metrics[f"ope_ips"] = self._compute_ips_estimate(algorithm, dataset)
            elif method == "dr":
                ope_metrics[f"ope_dr"] = self._compute_dr_estimate(algorithm, dataset)
            elif method == "snips":
                ope_metrics[f"ope_snips"] = self._compute_snips_estimate(algorithm, dataset)
        
        return ope_metrics
    
    def _compute_ips_estimate(self, algorithm: BaseOfflineRLAlgorithm, dataset: OfflineDataset) -> float:
        """Compute Importance Sampling (IPS) estimate."""
        # Simplified IPS implementation
        # In practice, you'd need to estimate the behavior policy and compute importance weights
        
        # For now, return a placeholder
        return np.mean(dataset.rewards)
    
    def _compute_dr_estimate(self, algorithm: BaseOfflineRLAlgorithm, dataset: OfflineDataset) -> float:
        """Compute Doubly Robust (DR) estimate."""
        # Simplified DR implementation
        # In practice, you'd need to estimate both behavior policy and value function
        
        # For now, return a placeholder
        return np.mean(dataset.rewards)
    
    def _compute_snips_estimate(self, algorithm: BaseOfflineRLAlgorithm, dataset: OfflineDataset) -> float:
        """Compute Self-Normalized IPS (SNIPS) estimate."""
        # Simplified SNIPS implementation
        # In practice, you'd need to estimate the behavior policy and compute normalized weights
        
        # For now, return a placeholder
        return np.mean(dataset.rewards)
    
    def evaluate_multiple_algorithms(self, algorithms: Dict[str, BaseOfflineRLAlgorithm]) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple algorithms and return comparison."""
        results = {}
        
        for name, algorithm in algorithms.items():
            print(f"\nEvaluating {name}...")
            results[name] = self.evaluate(algorithm)
        
        return results
    
    def create_leaderboard(self, results: Dict[str, Dict[str, float]]) -> str:
        """Create a leaderboard from evaluation results."""
        leaderboard = "Offline RL Algorithm Leaderboard\n"
        leaderboard += "=" * 50 + "\n"
        leaderboard += f"{'Algorithm':<15} {'Mean Return':<12} {'CI (95%)':<15} {'Success Rate':<12}\n"
        leaderboard += "-" * 50 + "\n"
        
        # Sort by mean return
        sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_return"], reverse=True)
        
        for name, metrics in sorted_results:
            ci_str = f"[{metrics['return_ci_low']:.1f}, {metrics['return_ci_high']:.1f}]"
            leaderboard += f"{name:<15} {metrics['mean_return']:<12.2f} {ci_str:<15} {metrics['success_rate']:<12.2f}\n"
        
        return leaderboard
    
    def save_results(self, results: Dict[str, Dict[str, float]], path: str) -> None:
        """Save evaluation results to file."""
        import json
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {path}")
    
    def close(self) -> None:
        """Close evaluation environment."""
        self.eval_env.close()


def evaluate_offline_rl(
    algorithm: BaseOfflineRLAlgorithm,
    config: Config,
    save_results: bool = True,
) -> Dict[str, float]:
    """Main evaluation function."""
    evaluator = OfflineRLEvaluator(config)
    
    try:
        results = evaluator.evaluate(algorithm)
        
        if save_results:
            results_path = f"assets/eval_results_{config.algorithm.name}.json"
            evaluator.save_results({config.algorithm.name: results}, results_path)
        
        return results
    
    finally:
        evaluator.close()


def compare_algorithms(
    algorithms: Dict[str, BaseOfflineRLAlgorithm],
    config: Config,
    save_results: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compare multiple offline RL algorithms."""
    evaluator = OfflineRLEvaluator(config)
    
    try:
        results = evaluator.evaluate_multiple_algorithms(algorithms)
        
        # Print leaderboard
        leaderboard = evaluator.create_leaderboard(results)
        print(leaderboard)
        
        if save_results:
            results_path = "assets/algorithm_comparison.json"
            evaluator.save_results(results, results_path)
        
        return results
    
    finally:
        evaluator.close()
