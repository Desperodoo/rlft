"""
Base Flow Matching Policy

Abstract base class for flow matching policies.
Flow matching learns a velocity field v(x, t | s) that transports
noise distribution to action distribution.

References:
- Flow Matching for Generative Modeling (Lipman et al., 2022)
- HRI-EU/flow_matching: https://github.com/HRI-EU/flow_matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from ...common.networks import FlowVelocityPredictor


class FlowMatchingPolicyBase(ABC):
    """Abstract base class for flow matching policies.
    
    Flow matching policies learn a velocity field that defines an ODE:
    dx/dt = v(x, t | s)
    
    Integrating this ODE from t=0 (noise) to t=1 (action) generates actions.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        device: Device for computation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256, 256],
        learning_rate: float = 1e-4,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Velocity field network
        self.velocity_net = FlowVelocityPredictor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.velocity_net.parameters(),
            lr=learning_rate
        )
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step.
        
        Args:
            batch: Dictionary with 'states' and 'actions' tensors
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def sample_action(self, state: np.ndarray) -> np.ndarray:
        """Sample an action given a state.
        
        Args:
            state: Current state
            
        Returns:
            Sampled action
        """
        pass
    
    def _euler_integrate(
        self,
        state: torch.Tensor,
        num_steps: int = 10,
        clip_action: bool = True
    ) -> torch.Tensor:
        """Integrate the ODE using Euler method.
        
        dx/dt = v(x, t | s)  from t=0 to t=1
        
        Args:
            state: Conditioning state
            num_steps: Number of integration steps
            clip_action: Whether to clip final action to [-1, 1]
            
        Returns:
            Final action
        """
        batch_size = state.shape[0]
        dt = 1.0 / num_steps
        
        # Start from Gaussian noise
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            v = self.velocity_net(state, x, t)
            x = x + v * dt
            
        if clip_action:
            x = torch.clamp(x, -1.0, 1.0)
            
        return x
    
    def _rk4_integrate(
        self,
        state: torch.Tensor,
        num_steps: int = 10,
        clip_action: bool = True
    ) -> torch.Tensor:
        """Integrate the ODE using 4th-order Runge-Kutta.
        
        More accurate than Euler but slower.
        
        Args:
            state: Conditioning state
            num_steps: Number of integration steps
            clip_action: Whether to clip final action
            
        Returns:
            Final action
        """
        batch_size = state.shape[0]
        dt = 1.0 / num_steps
        
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for i in range(num_steps):
            t = i * dt
            t_tensor = torch.full((batch_size,), t, device=self.device)
            
            k1 = self.velocity_net(state, x, t_tensor)
            
            t_mid = torch.full((batch_size,), t + dt/2, device=self.device)
            k2 = self.velocity_net(state, x + k1 * dt/2, t_mid)
            k3 = self.velocity_net(state, x + k2 * dt/2, t_mid)
            
            t_end = torch.full((batch_size,), t + dt, device=self.device)
            k4 = self.velocity_net(state, x + k3 * dt, t_end)
            
            x = x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
            
        if clip_action:
            x = torch.clamp(x, -1.0, 1.0)
            
        return x
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "velocity_net": self.velocity_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.velocity_net.load_state_dict(checkpoint["velocity_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
