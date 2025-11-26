"""
Replay Buffer for Experience Storage

Supports both online RL (collecting experience during training)
and offline RL (loading pre-collected datasets).
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple


class ReplayBuffer:
    """Experience replay buffer for RL algorithms.
    
    Stores transitions (s, a, r, s', done) and supports:
    - Random sampling for off-policy RL
    - Loading from pre-collected datasets
    - Efficient numpy-based storage
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        capacity: Maximum number of transitions to store
        device: Device to move sampled batches to
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.device = device
        
        # Pre-allocate memory
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add a single transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ):
        """Add a batch of transitions to the buffer.
        
        Args:
            states: Batch of states, shape (batch_size, state_dim)
            actions: Batch of actions, shape (batch_size, action_dim)
            rewards: Batch of rewards, shape (batch_size,) or (batch_size, 1)
            next_states: Batch of next states, shape (batch_size, state_dim)
            dones: Batch of done flags, shape (batch_size,) or (batch_size, 1)
        """
        batch_size = states.shape[0]
        
        # Ensure rewards and dones have correct shape
        rewards = rewards.reshape(-1, 1) if rewards.ndim == 1 else rewards
        dones = dones.reshape(-1, 1) if dones.ndim == 1 else dones
        
        # Handle wrap-around if batch exceeds capacity
        if self.ptr + batch_size <= self.capacity:
            self.states[self.ptr:self.ptr + batch_size] = states
            self.actions[self.ptr:self.ptr + batch_size] = actions
            self.rewards[self.ptr:self.ptr + batch_size] = rewards
            self.next_states[self.ptr:self.ptr + batch_size] = next_states
            self.dones[self.ptr:self.ptr + batch_size] = dones
        else:
            # Split the batch
            first_part = self.capacity - self.ptr
            second_part = batch_size - first_part
            
            self.states[self.ptr:] = states[:first_part]
            self.actions[self.ptr:] = actions[:first_part]
            self.rewards[self.ptr:] = rewards[:first_part]
            self.next_states[self.ptr:] = next_states[:first_part]
            self.dones[self.ptr:] = dones[:first_part]
            
            self.states[:second_part] = states[first_part:]
            self.actions[:second_part] = actions[first_part:]
            self.rewards[:second_part] = rewards[first_part:]
            self.next_states[:second_part] = next_states[first_part:]
            self.dones[:second_part] = dones[first_part:]
        
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary with keys: states, actions, rewards, next_states, dones
            Each value is a tensor on the specified device
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "states": torch.FloatTensor(self.states[indices]).to(self.device),
            "actions": torch.FloatTensor(self.actions[indices]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[indices]).to(self.device),
            "next_states": torch.FloatTensor(self.next_states[indices]).to(self.device),
            "dones": torch.FloatTensor(self.dones[indices]).to(self.device)
        }
    
    def sample_all(self) -> Dict[str, torch.Tensor]:
        """Return all data in the buffer as tensors.
        
        Returns:
            Dictionary with all transitions
        """
        return {
            "states": torch.FloatTensor(self.states[:self.size]).to(self.device),
            "actions": torch.FloatTensor(self.actions[:self.size]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[:self.size]).to(self.device),
            "next_states": torch.FloatTensor(self.next_states[:self.size]).to(self.device),
            "dones": torch.FloatTensor(self.dones[:self.size]).to(self.device)
        }
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]):
        """Load a pre-collected dataset into the buffer.
        
        Args:
            dataset: Dictionary with keys: states, actions, rewards, next_states, dones
        """
        self.add_batch(
            states=dataset["states"],
            actions=dataset["actions"],
            rewards=dataset["rewards"],
            next_states=dataset["next_states"],
            dones=dataset["dones"]
        )
        
    def __len__(self) -> int:
        return self.size
    
    def is_empty(self) -> bool:
        return self.size == 0
    
    def is_full(self) -> bool:
        return self.size >= self.capacity


class RolloutBuffer:
    """Rollout buffer for on-policy algorithms like PPO/DPPO.
    
    Stores complete trajectories with additional information
    needed for policy gradient computation (log probs, advantages, etc.)
    
    Args:
        capacity: Maximum number of steps to store
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        device: Device for tensors
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: str = "cpu"
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.reset()
        
    def reset(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None
        
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add a step to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """Compute returns and GAE advantages.
        
        Args:
            last_value: Value estimate for the last state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [False])
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t + 1]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            
        returns = advantages + values[:-1]
        
        self.advantages = advantages
        self.returns = returns
        
    def get_batches(
        self, 
        batch_size: int,
        shuffle: bool = True
    ):
        """Generate batches for training.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            
        Yields:
            Dictionary with batch data
        """
        # Convert to arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        log_probs = np.array(self.log_probs)
        advantages = self.advantages
        returns = self.returns
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        n_samples = len(self.states)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
            
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield {
                "states": torch.FloatTensor(states[batch_indices]).to(self.device),
                "actions": torch.FloatTensor(actions[batch_indices]).to(self.device),
                "old_log_probs": torch.FloatTensor(log_probs[batch_indices]).to(self.device),
                "advantages": torch.FloatTensor(advantages[batch_indices]).to(self.device),
                "returns": torch.FloatTensor(returns[batch_indices]).to(self.device)
            }
            
    def __len__(self) -> int:
        return len(self.states)
