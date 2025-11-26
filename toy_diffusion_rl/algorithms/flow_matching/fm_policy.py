"""
Vanilla Flow Matching Policy

Standard flow matching policy that learns a velocity field using
the Conditional Flow Matching objective.

References:
- Flow Matching for Generative Modeling (Lipman et al., 2022)
- Optimal Transport Flow Matching
- HRI-EU/flow_matching
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from .base_flow import FlowMatchingPolicyBase


class FlowMatchingPolicy(FlowMatchingPolicyBase):
    """Vanilla Flow Matching Policy.
    
    Uses the Conditional Flow Matching (CFM) objective to learn
    a velocity field that transports Gaussian noise to the action distribution.
    
    The training objective is:
    L = E_{t, x_0, x_1} [||v_theta(x_t, t | s) - (x_1 - x_0)||^2]
    
    where x_t = (1-t) * x_0 + t * x_1 is the linear interpolation.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        num_inference_steps: Number of ODE integration steps
        sigma_min: Minimum noise scale (for stability)
        device: Device for computation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256, 256],
        learning_rate: float = 1e-4,
        num_inference_steps: int = 10,
        sigma_min: float = 0.001,
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
        self.sigma_min = sigma_min
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step using CFM objective.
        
        Conditional Flow Matching loss:
        L = E_{t, x_0, x_1} [||v_theta(x_t, t | s) - u_t(x_t | x_1)||^2]
        
        For linear interpolation: u_t = x_1 - x_0
        
        Args:
            batch: Dictionary with 'states' and 'actions' tensors
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"]  # Conditioning state s
        actions = batch["actions"]  # Target action x_1
        batch_size = states.shape[0]
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(actions)
        
        # Sample random timesteps t ~ U(0, 1)
        t = torch.rand(batch_size, device=self.device)
        
        # Compute interpolation x_t = (1-t) * x_0 + t * x_1
        # Add small sigma_min for numerical stability
        t_expand = t.unsqueeze(-1)
        sigma_t = 1 - (1 - self.sigma_min) * t_expand
        x_t = sigma_t * x_0 + t_expand * actions
        
        # Target velocity: dx/dt = x_1 - x_0 (for optimal transport path)
        # With sigma_min adjustment
        target_velocity = actions - (1 - self.sigma_min) * x_0
        
        # Predict velocity
        predicted_velocity = self.velocity_net(states, x_t, t)
        
        # MSE loss
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
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
        num_steps: Optional[int] = None,
        use_rk4: bool = False
    ) -> np.ndarray:
        """Sample an action by integrating the learned ODE.
        
        Args:
            state: Current state
            num_steps: Number of integration steps
            use_rk4: Use RK4 integrator (more accurate but slower)
            
        Returns:
            Sampled action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        self.velocity_net.eval()
        num_steps = num_steps or self.num_inference_steps
        
        if use_rk4:
            action = self._rk4_integrate(state, num_steps)
        else:
            action = self._euler_integrate(state, num_steps)
            
        return action.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def sample_action_stochastic(
        self,
        state: np.ndarray,
        num_steps: Optional[int] = None,
        temperature: float = 0.1
    ) -> np.ndarray:
        """Sample action with stochastic noise injection.
        
        Adds small Gaussian noise at each integration step for exploration.
        
        Args:
            state: Current state
            num_steps: Number of integration steps
            temperature: Scale of injected noise
            
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
        
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            v = self.velocity_net(state, x, t)
            
            # Add stochastic noise (decreasing with t)
            noise_scale = temperature * (1 - i / num_steps)
            noise = noise_scale * torch.randn_like(x)
            
            x = x + v * dt + noise * np.sqrt(dt)
            
        x = torch.clamp(x, -1.0, 1.0)
        return x.cpu().numpy().squeeze()
