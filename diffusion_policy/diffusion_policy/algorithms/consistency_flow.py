"""
Consistency Flow Policy with self-consistency loss.
Migrated from toy_diffusion_rl to diffusion_policy framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal
import copy

from .networks import VelocityUNet1D, soft_update


class ConsistencyFlowAgent(nn.Module):
    """
    Consistency Flow Matching agent.
    Combines flow matching with consistency loss for faster inference.
    Uses EMA teacher for stable consistency targets.
    """
    
    def __init__(
        self,
        velocity_net: VelocityUNet1D,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        num_flow_steps: int = 10,
        flow_weight: float = 1.0,
        consistency_weight: float = 1.0,
        ema_decay: float = 0.999,
        consistency_delta: float = 0.01,  # Small time step for consistency
        device: str = "cuda",
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.velocity_net_ema = copy.deepcopy(velocity_net)
        
        # Freeze EMA model
        for param in self.velocity_net_ema.parameters():
            param.requires_grad = False
            
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.num_flow_steps = num_flow_steps
        self.flow_weight = flow_weight
        self.consistency_weight = consistency_weight
        self.ema_decay = ema_decay
        self.consistency_delta = consistency_delta  # Not used anymore, kept for API compatibility
        self.device = device
        
        # Consistency loss hyperparameters (aligned with best practices)
        self.t_min = 0.05  # Avoid boundary instability at t≈0
        self.t_max = 0.95  # Avoid boundary instability at t≈1
        self.delta_min = 0.02  # Minimum delta for consistency
        self.delta_max = 0.15  # Maximum delta (avoid large teacher error)
        self.teacher_steps = 2  # Multi-step teacher for accurate targets
        
    def update_ema(self):
        """Update EMA velocity network.
        
        Formula: θ_ema = ema_decay * θ_ema + (1 - ema_decay) * θ
        Equivalent to: soft_update(ema, source, tau=1-ema_decay)
        """
        soft_update(self.velocity_net_ema, self.velocity_net, 1 - self.ema_decay)
    
    def _get_consistency_targets(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        obs_features: torch.Tensor,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """
        Compute consistency target by integrating from t to 1 using EMA model.
        
        Args:
            x_t: Current state at time t
            t: Current time
            obs_features: Observation conditioning
            num_steps: Number of integration steps to take
        """
        x = x_t.clone()
        current_t = t.clone()
        
        remaining_time = 1.0 - current_t
        dt = remaining_time / num_steps
        
        for _ in range(num_steps):
            v = self.velocity_net_ema(x, current_t, obs_features)
            # Expand dt for broadcasting
            dt_expand = dt.view(-1, 1, 1)
            x = x + v * dt_expand
            current_t = current_t + dt
            
        return x
    
    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mixed flow matching + consistency loss.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim]
            actions: Expert actions [B, pred_horizon, action_dim]
        """
        batch_size = actions.shape[0]
        
        # Sample noise
        x_0 = torch.randn_like(actions)
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=actions.device)
        
        # Interpolate
        t_expand = t.view(-1, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * actions
        
        # Flow matching target
        v_target = actions - x_0
        
        # Predict velocity
        v_pred = self.velocity_net(x_t, t, obs_features)
        
        # Flow matching loss
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # Consistency loss: predictions from t and t+delta should reach same endpoint
        # Using velocity space loss for gradient stability
        consistency_loss = torch.tensor(0.0, device=actions.device)
        if self.consistency_weight > 0:
            # Sample t in restricted range [t_min, t_max] to avoid boundary instability
            t_cons = self.t_min + torch.rand(batch_size, device=actions.device) * (self.t_max - self.t_min)
            
            # Random delta in [delta_min, delta_max], clamped to not exceed t_max
            delta_t = self.delta_min + torch.rand(batch_size, device=actions.device) * (self.delta_max - self.delta_min)
            t_plus = torch.clamp(t_cons + delta_t, max=self.t_max)
            
            # Get x at both time points (using same x_0 for consistency)
            t_cons_expand = t_cons.view(-1, 1, 1)
            t_plus_expand = t_plus.view(-1, 1, 1)
            x_t_cons = (1 - t_cons_expand) * x_0 + t_cons_expand * actions
            x_t_plus = (1 - t_plus_expand) * x_0 + t_plus_expand * actions
            
            # Teacher: multi-step integration from t_plus to 1 using EMA network
            with torch.no_grad():
                target_x1 = self._get_consistency_targets(
                    x_t_plus, t_plus, obs_features, num_steps=self.teacher_steps
                )
            
            # Student: predict velocity at t_plus, compute implied x1
            # The target velocity is: v_target = (target_x1 - x_0) 
            # This is in velocity space for gradient stability
            v_target = target_x1 - x_0
            v_pred = self.velocity_net(x_t_plus, t_plus, obs_features)
            
            consistency_loss = F.mse_loss(v_pred, v_target)
        
        total_loss = (
            self.flow_weight * flow_loss + 
            self.consistency_weight * consistency_loss
        )
        
        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "consistency_loss": consistency_loss,
        }
    
    @torch.no_grad()
    def get_action(
        self,
        obs_features: torch.Tensor,
        num_steps: Optional[int] = None,
        integration_method: Literal["euler", "rk4"] = "euler",
    ) -> torch.Tensor:
        """
        Generate action using flow ODE integration.
        Consistency training allows fewer steps at inference.
        
        Args:
            obs_features: Encoded observation [B, obs_horizon, obs_dim]
            num_steps: Override default flow steps (can use fewer due to consistency)
            integration_method: ODE solver to use
            
        Returns:
            actions: Generated action sequence [B, pred_horizon, action_dim]
        """
        self.velocity_net.eval()
        batch_size = obs_features.shape[0]
        
        # Can use fewer steps due to consistency training
        steps = num_steps if num_steps is not None else self.num_flow_steps
        
        # Start from noise
        x = torch.randn(
            batch_size, self.pred_horizon, self.action_dim,
            device=obs_features.device
        )
        
        dt = 1.0 / steps
        
        for i in range(steps):
            t = torch.full((batch_size,), i * dt, device=obs_features.device)
            
            if integration_method == "euler":
                v = self.velocity_net(x, t, obs_features)
                x = x + v * dt
            else:  # RK4
                t_mid = t + 0.5 * dt
                t_end = t + dt
                
                k1 = self.velocity_net(x, t, obs_features)
                k2 = self.velocity_net(x + 0.5 * dt * k1, t_mid, obs_features)
                k3 = self.velocity_net(x + 0.5 * dt * k2, t_mid, obs_features)
                k4 = self.velocity_net(x + dt * k3, t_end, obs_features)
                
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Clamp to action bounds
        x = torch.clamp(x, -1.0, 1.0)
        
        self.velocity_net.train()
        return x
