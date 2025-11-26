"""
Consistency Flow Policy

Implements consistency-style training for flow matching,
enabling single-step or few-step action generation.

References:
- Consistency Models (Song et al., 2023)
- ManiFlow: https://github.com/allenai/maniflow
- rectified-flow-pytorch: https://github.com/lucidrains/rectified-flow-pytorch

Key Insight from ManiFlow:
- The model predicts AVERAGE VELOCITY v_avg = (x_1 - x_t) / (1 - t)
- At inference, we use: x_1 = x_t + (1 - t) * v_avg
- This naturally handles single-step generation when t=0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import copy

from .base_flow import FlowMatchingPolicyBase
from ...common.networks import FlowVelocityPredictor


class ConsistencyFlowPolicy(FlowMatchingPolicyBase):
    """Consistency Flow Policy for fast action generation (ManiFlow-style).
    
    Following ManiFlow's approach:
    1. Train with mixed Flow + Consistency objectives
    2. Model predicts average velocity v_avg = (x_1 - x_t) / (1-t)
    3. Use EMA teacher for consistency targets
    4. Single-step inference: x_1 = x_0 + 1.0 * v_avg(x_0, t=0)
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space  
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        flow_batch_ratio: Fraction of batch for flow loss (default 0.5)
        consistency_batch_ratio: Fraction of batch for consistency loss (default 0.5)
        use_ema_teacher: Use EMA model as teacher
        ema_decay: Decay rate for EMA
        num_inference_steps: Number of ODE steps for multi-step inference
        device: Device for computation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256, 256],
        learning_rate: float = 1e-4,
        consistency_weight: float = 1.0,  # kept for compatibility
        flow_batch_ratio: float = 0.5,
        consistency_batch_ratio: float = 0.5,
        use_ema_teacher: bool = True,
        ema_decay: float = 0.999,
        num_inference_steps: int = 10,
        denoise_timesteps: int = 10,  # discretization for t sampling
        device: str = "cpu"
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            device=device
        )
        
        self.flow_batch_ratio = flow_batch_ratio
        self.consistency_batch_ratio = consistency_batch_ratio
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self.num_inference_steps = num_inference_steps
        self.denoise_timesteps = denoise_timesteps
        
        # Single network that predicts average velocity
        # v_avg = (x_1 - x_t) / (1 - t), so x_1 = x_t + (1-t) * v_avg
        self.optimizer = torch.optim.Adam(
            self.velocity_net.parameters(),
            lr=learning_rate
        )
        
        # EMA teacher for consistency training
        if use_ema_teacher:
            self.ema_velocity_net = copy.deepcopy(self.velocity_net)
            for p in self.ema_velocity_net.parameters():
                p.requires_grad = False
                
    def _update_ema(self):
        """Update EMA parameters."""
        if not self.use_ema_teacher:
            return
            
        with torch.no_grad():
            for ema_p, p in zip(
                self.ema_velocity_net.parameters(),
                self.velocity_net.parameters()
            ):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1-self.ema_decay)
    
    def _linear_interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation: x_t = (1-t)*x_0 + t*x_1"""
        t_expand = t.view(-1, 1) if t.dim() == 1 else t
        return (1 - t_expand) * x_0 + t_expand * x_1
    
    def _get_flow_targets(
        self,
        actions: torch.Tensor,
        states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get flow matching targets.
        
        Standard flow matching: predict instantaneous velocity v = x_1 - x_0
        This is constant along the linear interpolation path.
        """
        batch_size = actions.shape[0]
        
        # Sample noise (starting point)
        x_0 = torch.randn_like(actions)
        
        # Sample timestep uniformly
        t = torch.rand(batch_size, device=self.device)
        
        # Interpolate
        x_t = self._linear_interpolate(x_0, actions, t)
        
        # Target: instantaneous velocity (constant for linear flow)
        v_target = actions - x_0
        
        return {
            'x_t': x_t,
            't': t,
            'v_target': v_target,
            'x_0': x_0,
            'x_1': actions
        }
    
    def _get_consistency_targets(
        self,
        actions: torch.Tensor,
        states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get consistency training targets (ManiFlow-style).
        
        Standard flow matching predicts v = x_1 - x_0 (constant velocity).
        For consistency, we want the model to learn to predict the same x_1
        regardless of the timestep t.
        
        Method:
        1. At t_next, use EMA teacher to predict velocity v
        2. Compute predicted x_1 = x_t_next + (1 - t_next) * v
        3. Target velocity at t is: (pred_x_1 - x_t) / (1 - t) 
           But since v should be constant, we can use: pred_x_1 - x_0
        """
        batch_size = actions.shape[0]
        
        # Sample noise
        x_0 = torch.randn_like(actions)
        
        # Sample t uniformly, but keep away from boundaries
        t = 0.05 + torch.rand(batch_size, device=self.device) * 0.9  # t in [0.05, 0.95]
        
        # Sample delta_t 
        delta_t = torch.rand(batch_size, device=self.device) * (0.99 - t)
        delta_t = torch.clamp(delta_t, min=0.02)
        
        t_next = torch.clamp(t + delta_t, max=0.99)
        
        # Interpolate to get x_t and x_t_next
        x_t = self._linear_interpolate(x_0, actions, t)
        x_t_next = self._linear_interpolate(x_0, actions, t_next)
        
        # Use EMA teacher to predict velocity at t_next
        with torch.no_grad():
            net = self.ema_velocity_net if self.use_ema_teacher else self.velocity_net
            v_teacher = net(states, x_t_next, t_next)
            
            # Estimate x_1 using teacher's prediction: x_1 = x_t_next + (1 - t_next) * v
            t_next_expand = t_next.unsqueeze(-1)
            pred_x1 = x_t_next + (1 - t_next_expand) * v_teacher
        
        # Target velocity at t: should point to the same pred_x1
        # v_target = pred_x1 - x_0 (since for constant velocity, v = x_1 - x_0)
        v_target = pred_x1 - x_0
        
        return {
            'x_t': x_t,
            't': t,
            'v_target': v_target,
            'target_t': delta_t,
            'x_0': x_0,
            'x_1': actions
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train with combined flow matching and consistency loss (ManiFlow-style).
        
        Args:
            batch: Dictionary with 'states' and 'actions'
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"]
        actions = batch["actions"]
        batch_size = states.shape[0]
        
        # Split batch for flow and consistency training
        flow_size = int(batch_size * self.flow_batch_ratio)
        cons_size = int(batch_size * self.consistency_batch_ratio)
        
        total_loss = 0.0
        flow_loss_val = 0.0
        cons_loss_val = 0.0
        
        # ============ Flow Matching Loss ============
        if flow_size > 0:
            flow_states = states[:flow_size]
            flow_actions = actions[:flow_size]
            
            flow_targets = self._get_flow_targets(flow_actions, flow_states)
            
            # Predict velocity
            v_pred = self.velocity_net(flow_states, flow_targets['x_t'], flow_targets['t'])
            
            flow_loss = F.mse_loss(v_pred, flow_targets['v_target'])
            total_loss = total_loss + flow_loss
            flow_loss_val = flow_loss.item()
        
        # ============ Consistency Loss ============
        if cons_size > 0:
            cons_states = states[flow_size:flow_size+cons_size]
            cons_actions = actions[flow_size:flow_size+cons_size]
            
            cons_targets = self._get_consistency_targets(cons_actions, cons_states)
            
            # Predict average velocity
            v_pred = self.velocity_net(cons_states, cons_targets['x_t'], cons_targets['t'])
            
            cons_loss = F.mse_loss(v_pred, cons_targets['v_target'])
            total_loss = total_loss + cons_loss
            cons_loss_val = cons_loss.item()
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.velocity_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update EMA
        self._update_ema()
        
        return {
            "loss": total_loss.item(),
            "flow_loss": flow_loss_val,
            "consistency_loss": cons_loss_val,
            "direct_loss": 0.0  # Not used in ManiFlow-style
        }
    
    @torch.no_grad()
    def sample_action(
        self,
        state: np.ndarray,
        num_steps: Optional[int] = None,
        use_consistency: bool = True
    ) -> np.ndarray:
        """Sample action using ODE integration.
        
        The model predicts velocity v = x_1 - x_0 (constant for linear flow).
        ODE: dx/dt = v, so x_t = x_0 + t * v
        
        For single-step at t=0:
        x_1 = x_0 + 1.0 * v = x_0 + v
        
        For multi-step:
        x_{t+dt} = x_t + dt * v
        
        Args:
            state: Current state
            num_steps: Number of ODE steps (default: self.num_inference_steps)
            use_consistency: If True, use EMA model
            
        Returns:
            Sampled action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        batch_size = state.shape[0]
        num_steps = num_steps or self.num_inference_steps
        
        # Select which network to use
        net = self.ema_velocity_net if (use_consistency and self.use_ema_teacher) else self.velocity_net
        net.eval()
        
        # Start from noise
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # ODE integration with Euler method
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            
            # Predict velocity (constant for linear flow, but may vary for learned flow)
            v = net(state, x, t)
            x = x + v * dt
        
        action = torch.clamp(x, -1.0, 1.0)
        return action.cpu().numpy().squeeze()
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "velocity_net": self.velocity_net.state_dict(),
            "consistency_net": self.consistency_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.use_ema_teacher:
            checkpoint["ema_velocity_net"] = self.ema_velocity_net.state_dict()
            checkpoint["ema_consistency_net"] = self.ema_consistency_net.state_dict()
        torch.save(checkpoint, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.velocity_net.load_state_dict(checkpoint["velocity_net"])
        self.consistency_net.load_state_dict(checkpoint["consistency_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.use_ema_teacher and "ema_velocity_net" in checkpoint:
            self.ema_velocity_net.load_state_dict(checkpoint["ema_velocity_net"])
            self.ema_consistency_net.load_state_dict(checkpoint["ema_consistency_net"])
