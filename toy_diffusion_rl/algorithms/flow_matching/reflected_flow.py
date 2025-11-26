"""
Reflected Flow Policy

Flow matching with boundary reflection for bounded action spaces.
When the flow trajectory hits action bounds, it reflects back into
the valid region, maintaining the density.

References:
- Reflected Flow Matching (inspired by rectified flow ideas)
- rectified-flow-pytorch: https://github.com/lucidrains/rectified-flow-pytorch
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from .base_flow import FlowMatchingPolicyBase


class ReflectedFlowPolicy(FlowMatchingPolicyBase):
    """Reflected Flow Policy for bounded action spaces.
    
    Extends vanilla flow matching with boundary reflection:
    - During training, actions that would go outside [-1, 1] are reflected
    - This helps maintain proper density when learning bounded actions
    
    Two modes of reflection are supported:
    1. Hard reflection: Immediate bounce back at boundary
    2. Soft reflection: Smooth penalty pushing trajectory back
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        num_inference_steps: Number of ODE integration steps
        reflection_mode: 'hard' or 'soft'
        action_bounds: Action bounds (default: [-1, 1])
        device: Device for computation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256, 256],
        learning_rate: float = 1e-4,
        num_inference_steps: int = 10,
        reflection_mode: str = "hard",
        action_bounds: tuple = (-1.0, 1.0),
        device: str = "cpu"
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            device=device
        )
        
        self.num_inference_steps = num_inference_steps
        self.reflection_mode = reflection_mode
        self.action_low = action_bounds[0]
        self.action_high = action_bounds[1]
        
    def _reflect_trajectory(
        self, 
        x: torch.Tensor, 
        v: torch.Tensor, 
        dt: float
    ) -> torch.Tensor:
        """Apply reflection when trajectory exits bounds.
        
        Implements elastic reflection: when x + v*dt would exceed bounds,
        the excess is reflected back into the valid region.
        
        Args:
            x: Current position
            v: Velocity
            dt: Time step
            
        Returns:
            New position with reflection applied
        """
        x_new = x + v * dt
        
        if self.reflection_mode == "hard":
            # Hard reflection: bounce back
            # Track how many times we cross each boundary
            while True:
                # Check lower bound
                below_low = x_new < self.action_low
                if below_low.any():
                    excess = self.action_low - x_new
                    x_new = torch.where(below_low, self.action_low + excess, x_new)
                    
                # Check upper bound  
                above_high = x_new > self.action_high
                if above_high.any():
                    excess = x_new - self.action_high
                    x_new = torch.where(above_high, self.action_high - excess, x_new)
                    
                # Check if all within bounds
                if not (below_low.any() or above_high.any()):
                    break
                    
                # Safety: prevent infinite loops
                if (x_new < self.action_low - 1).any() or (x_new > self.action_high + 1).any():
                    x_new = torch.clamp(x_new, self.action_low, self.action_high)
                    break
        else:
            # Soft reflection: just clamp (with gradient)
            x_new = torch.clamp(x_new, self.action_low, self.action_high)
            
        return x_new
    
    def _compute_reflected_target(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute target velocity with reflection awareness.
        
        The target velocity is adjusted to account for reflections
        that would occur along the linear interpolation path.
        
        Args:
            x_0: Initial noise
            x_1: Target action (already bounded)
            t: Current timestep
            
        Returns:
            Target velocity that accounts for reflection
        """
        # For simplicity, we use a modified path that stays in bounds
        # Compute linear interpolation
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        # If x_t is outside bounds, project to boundary
        x_t_reflected = torch.clamp(x_t, self.action_low, self.action_high)
        
        # Target velocity: direction to final action
        # Adjusted for remaining time
        remaining_t = 1 - t_expand + 1e-6
        target_v = (x_1 - x_t_reflected) / remaining_t
        
        return target_v, x_t_reflected
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train step with reflection-aware flow matching.
        
        Args:
            batch: Dictionary with 'states' and 'actions' tensors
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"]
        actions = batch["actions"]  # Already bounded to [-1, 1]
        batch_size = states.shape[0]
        
        # Sample noise (can be larger than action bounds)
        x_0 = torch.randn_like(actions)
        
        # Sample timesteps
        t = torch.rand(batch_size, device=self.device)
        
        # Compute interpolation with reflection consideration
        target_v, x_t = self._compute_reflected_target(x_0, actions, t)
        
        # Predict velocity
        predicted_v = self.velocity_net(states, x_t, t)
        
        # Loss
        loss = F.mse_loss(predicted_v, target_v)
        
        # Add boundary regularization loss
        # Encourage velocities that point inward at boundaries
        boundary_mask = (x_t <= self.action_low + 0.1) | (x_t >= self.action_high - 0.1)
        if boundary_mask.any():
            # At lower bound, velocity should be positive
            # At upper bound, velocity should be negative
            lower_mask = x_t <= self.action_low + 0.1
            upper_mask = x_t >= self.action_high - 0.1
            
            boundary_loss = 0.0
            if lower_mask.any():
                # Penalize negative velocities at lower bound
                boundary_loss += F.relu(-predicted_v[lower_mask]).mean() * 0.1
            if upper_mask.any():
                # Penalize positive velocities at upper bound
                boundary_loss += F.relu(predicted_v[upper_mask]).mean() * 0.1
                
            loss = loss + boundary_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.velocity_net.parameters(), 1.0)
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    @torch.no_grad()
    def sample_action(
        self,
        state: np.ndarray,
        num_steps: Optional[int] = None
    ) -> np.ndarray:
        """Sample action with reflected integration.
        
        Args:
            state: Current state
            num_steps: Number of integration steps
            
        Returns:
            Sampled action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        self.velocity_net.eval()
        num_steps = num_steps or self.num_inference_steps
        
        batch_size = state.shape[0]
        dt = 1.0 / num_steps
        
        # Start from noise
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            v = self.velocity_net(state, x, t)
            
            # Apply reflection during integration
            x = self._reflect_trajectory(x, v, dt)
            
        # Final clamp for safety
        x = torch.clamp(x, self.action_low, self.action_high)
        
        return x.cpu().numpy().squeeze()
