# Offline Reinforcement Learning Project

A comprehensive implementation of advanced offline reinforcement learning algorithms including Behavior Cloning (BC), Batch Constrained Q-learning (BCQ), Conservative Q-Learning (CQL), and Implicit Q-Learning (IQL).

## ⚠️ DISCLAIMER

**This project is for research and educational purposes only. It is NOT intended for production control of real-world systems. Use at your own risk.**

## Features

- **Advanced Offline RL Algorithms**: BC, BCQ, CQL, IQL implementations
- **Comprehensive Evaluation**: Returns, confidence intervals, offline policy evaluation
- **Interactive Demo**: Streamlit-based visualization and experimentation
- **Reproducible Research**: Deterministic seeding, structured configs, comprehensive logging
- **Modern Tech Stack**: PyTorch 2.x, Gymnasium, structured data handling

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Offline-Reinforcement-Learning-Project.git
cd Offline-Reinforcement-Learning-Project

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Train BCQ on CartPole with random data
python scripts/train.py --algorithm bcq --env CartPole-v1 --episodes 1000 --epochs 500

# Evaluate trained model
python scripts/evaluate.py --algorithm bcq --checkpoint checkpoints/bcq_final.pt

# Compare all algorithms
python scripts/evaluate.py --compare --episodes 20
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── algorithms/         # Offline RL algorithms
│   ├── buffers/           # Dataset management
│   ├── eval/             # Evaluation utilities
│   ├── train/            # Training utilities
│   └── utils/            # Configuration and utilities
├── configs/              # Configuration files
├── scripts/              # Training and evaluation scripts
├── demo/                 # Streamlit demo
├── tests/                # Unit tests
├── assets/               # Results and visualizations
└── data/                 # Datasets
```

## Algorithms

### Behavior Cloning (BC)
Simple supervised learning approach that mimics the behavior policy from offline data.

### Batch Constrained Q-learning (BCQ)
Prevents distribution shift by constraining actions to the support of the behavior policy using a generative model.

### Conservative Q-Learning (CQL)
Learns conservative Q-functions by penalizing Q-values for out-of-distribution actions.

### Implicit Q-Learning (IQL)
Uses expectile regression to learn value functions and advantage-weighted regression for policy learning.

## Configuration

The project uses YAML configuration files for easy experimentation:

```yaml
env:
  name: "CartPole-v1"
  max_episode_steps: 500

dataset:
  num_episodes: 1000
  behavior_policy: "random"  # random, expert, epsilon_greedy

algorithm:
  name: "bcq"
  learning_rate: 0.0003
  batch_size: 256
  hidden_dims: [256, 256]

training:
  num_epochs: 500
  eval_frequency: 50
  device: "auto"
  seed: 42
```

## Evaluation Metrics

- **Returns**: Average return ± 95% confidence interval
- **Success Rate**: Percentage of episodes meeting success threshold
- **Offline Policy Evaluation**: IPS, DR, SNIPS estimates
- **Sample Efficiency**: Training progress and convergence

## Environment Support

Currently supports:
- CartPole-v1
- MountainCar-v0
- Acrobot-v1

## Dataset Generation

The project includes utilities to generate offline datasets using different behavior policies:

- **Random**: Uniform random action selection
- **Expert**: Simple heuristic policies (e.g., CartPole balance)
- **Epsilon-greedy**: Q-learning with exploration

## Safety and Reproducibility

- **Deterministic Seeding**: All random sources are seeded for reproducibility
- **Device Fallback**: Automatic CUDA → MPS → CPU device selection
- **Safety Disclaimers**: Clear warnings about research-only usage
- **Comprehensive Logging**: TensorBoard integration for monitoring

## Development

### Code Quality

```bash
# Format code
black src/ scripts/ demo/

# Lint code
ruff src/ scripts/ demo/

# Run tests
pytest tests/
```

### Adding New Algorithms

1. Inherit from `BaseOfflineRLAlgorithm`
2. Implement `train_step()` and `select_action()` methods
3. Add algorithm-specific configuration parameters
4. Update trainer to support new algorithm

### Adding New Environments

1. Ensure environment follows Gymnasium interface
2. Update state/action dimensions in trainer
3. Add environment-specific success thresholds in evaluator

## Results

Expected performance on CartPole-v1 with random behavior policy:

| Algorithm | Mean Return | Success Rate |
|-----------|------------|--------------|
| BC        | ~150-200   | ~60-80%      |
| BCQ       | ~180-220   | ~70-90%      |
| CQL       | ~160-210   | ~65-85%      |
| IQL       | ~170-215   | ~70-88%      |

*Results may vary based on dataset quality and hyperparameters.*

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{offline_rl_project,
  title={Offline Reinforcement Learning: Advanced Algorithms and Evaluation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Offline-Reinforcement-Learning-Project}
}
```

## Acknowledgments

- OpenAI Gymnasium for environment interfaces
- PyTorch team for the deep learning framework
- The offline RL research community for algorithm implementations
- Streamlit for the interactive demo framework
# Offline-Reinforcement-Learning-Project
