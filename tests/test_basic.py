"""Basic tests for offline RL project."""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import Config
from src.utils.utils import set_seed, get_device, create_mlp
from src.buffers.dataset import OfflineDataset
from src.algorithms.offline_rl import BehaviorCloning, BCQ, CQL, IQL


class TestConfig:
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test config creation."""
        config = Config()
        assert config.env.name == "CartPole-v1"
        assert config.algorithm.name == "bcq"
        assert config.training.seed == 42
    
    def test_config_save_load(self):
        """Test config save and load."""
        config = Config()
        config.env.name = "TestEnv"
        
        # Save config
        config.save("test_config.yaml")
        
        # Load config
        loaded_config = Config.load("test_config.yaml")
        
        assert loaded_config.env.name == "TestEnv"
        
        # Cleanup
        Path("test_config.yaml").unlink()


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        assert np.random.get_state()[1][0] == 42
        assert torch.initial_seed() == 42
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_create_mlp(self):
        """Test MLP creation."""
        mlp = create_mlp(4, 2, [64, 64])
        
        # Test forward pass
        x = torch.randn(10, 4)
        y = mlp(x)
        
        assert y.shape == (10, 2)


class TestDataset:
    """Test dataset functionality."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        observations = np.random.randn(100, 4)
        actions = np.random.randint(0, 2, 100)
        rewards = np.random.randn(100)
        next_observations = np.random.randn(100, 4)
        dones = np.random.choice([True, False], 100)
        
        dataset = OfflineDataset(
            observations, actions, rewards, next_observations, dones
        )
        
        assert len(dataset) == 100
        assert dataset.observations.shape == (100, 4)
        assert dataset.actions.shape == (100,)
    
    def test_dataset_sampling(self):
        """Test dataset sampling."""
        observations = np.random.randn(100, 4)
        actions = np.random.randint(0, 2, 100)
        rewards = np.random.randn(100)
        next_observations = np.random.randn(100, 4)
        dones = np.random.choice([True, False], 100)
        
        dataset = OfflineDataset(
            observations, actions, rewards, next_observations, dones
        )
        
        batch = dataset.sample_batch(32)
        assert len(batch["observations"]) == 32


class TestAlgorithms:
    """Test offline RL algorithms."""
    
    def test_behavior_cloning(self):
        """Test behavior cloning."""
        bc = BehaviorCloning(state_dim=4, action_dim=2)
        
        # Test action selection
        state = np.random.randn(4)
        action = bc.select_action(state)
        assert isinstance(action, np.ndarray)
    
    def test_bcq(self):
        """Test BCQ."""
        bcq = BCQ(state_dim=4, action_dim=2)
        
        # Test action selection
        state = np.random.randn(4)
        action = bcq.select_action(state)
        assert isinstance(action, np.ndarray)
    
    def test_cql(self):
        """Test CQL."""
        cql = CQL(state_dim=4, action_dim=2)
        
        # Test action selection
        state = np.random.randn(4)
        action = cql.select_action(state)
        assert isinstance(action, np.ndarray)
    
    def test_iql(self):
        """Test IQL."""
        iql = IQL(state_dim=4, action_dim=2)
        
        # Test action selection
        state = np.random.randn(4)
        action = iql.select_action(state)
        assert isinstance(action, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__])
