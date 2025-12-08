"""
Network architectures for Offline RL algorithms

All networks are designed to be compatible with the official diffusion_policy
structure, using PlainConv for visual encoding and ConditionalUnet1D architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple

from ..conditional_unet1d import ConditionalUnet1D
from ..plain_conv import PlainConv


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timestep encoding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class VelocityUNet1D(nn.Module):
    """1D U-Net for velocity field prediction (Flow Matching).
    
    Similar to ConditionalUnet1D but predicts velocity instead of noise.
    Uses FiLM conditioning with observation features.
    
    Args:
        input_dim: Action dimension
        global_cond_dim: Dimension of global conditioning (obs features)
        diffusion_step_embed_dim: Dimension of timestep embedding
        down_dims: Channel dimensions for each downsampling stage
        n_groups: Number of groups for GroupNorm
    """
    
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 64,
        down_dims: List[int] = [64, 128, 256],
        n_groups: int = 8,
    ):
        super().__init__()
        # Reuse ConditionalUnet1D architecture for velocity prediction
        # The only difference is semantic: output is velocity instead of noise
        self.unet = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            n_groups=n_groups,
        )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sample: (B, pred_horizon, action_dim) noisy action sequence
            timestep: (B,) timestep values in [0, 1] for flow matching
            global_cond: (B, global_cond_dim) or (B, obs_horizon, cond_dim) observation features
            
        Returns:
            velocity: (B, pred_horizon, action_dim) predicted velocity field
        """
        # Flatten global_cond if it's 3D (B, obs_horizon, cond_dim) -> (B, obs_horizon * cond_dim)
        if global_cond.dim() == 3:
            global_cond = global_cond.reshape(global_cond.shape[0], -1)
        
        # Convert continuous timestep [0, 1] to integer for U-Net
        # U-Net internally uses sinusoidal embedding, so we scale to [0, 100]
        timestep_int = (timestep * 100).long()
        return self.unet(sample, timestep_int, global_cond=global_cond)


class DoubleQNetwork(nn.Module):
    """Twin Q-Network with 1D U-Net architecture.
    
    Uses the same architecture as the policy network to maintain consistency.
    Takes (action_sequence, obs_features) and outputs Q-values.
    
    Note on Action Chunking and Q-Learning:
    - This Q-network evaluates entire action sequences, not single actions
    - The Bellman equation should ideally use cumulative rewards over the sequence
    - Current implementation uses single-step reward as approximation
    
    Args:
        input_dim: Action dimension
        global_cond_dim: Dimension of global conditioning (obs features)
        pred_horizon: Length of action sequence (default: 16)
        diffusion_step_embed_dim: Dimension for time/dummy embedding
        down_dims: Channel dimensions for each downsampling stage
        n_groups: Number of groups for GroupNorm
    """
    
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        pred_horizon: int = 16,
        diffusion_step_embed_dim: int = 64,
        down_dims: List[int] = [64, 128, 256],
        n_groups: int = 8,
    ):
        super().__init__()
        
        self.pred_horizon = pred_horizon
        self.input_dim = input_dim
        
        # Q1 network - uses U-Net architecture for consistency with policy
        self.q1_net = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            n_groups=n_groups,
        )
        
        # Q2 network
        self.q2_net = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            n_groups=n_groups,
        )
        
        # Output heads to reduce action sequence to scalar Q-value
        # After U-Net: (B, pred_horizon, input_dim) -> (B, 1)
        flatten_dim = pred_horizon * input_dim
        self.q1_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 256),
            nn.Mish(),
            nn.Linear(256, 1),
        )
        
        self.q2_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 256),
            nn.Mish(),
            nn.Linear(256, 1),
        )
    
    def forward(
        self,
        action_seq: torch.Tensor,
        obs_cond: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            action_seq: (B, pred_horizon, action_dim) action sequence
            obs_cond: (B, global_cond_dim) observation features
            timestep: (B,) optional timestep, defaults to 0
            
        Returns:
            q1, q2: (B, 1) Q-values from both networks
        """
        B = action_seq.shape[0]
        if timestep is None:
            timestep = torch.zeros(B, device=action_seq.device, dtype=torch.long)
        
        # Get U-Net features
        q1_features = self.q1_net(action_seq, timestep, global_cond=obs_cond)
        q2_features = self.q2_net(action_seq, timestep, global_cond=obs_cond)
        
        # Reduce to scalar Q-values
        q1 = self.q1_head(q1_features)
        q2 = self.q2_head(q2_features)
        
        return q1, q2
    
    def q1_forward(
        self,
        action_seq: torch.Tensor,
        obs_cond: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Q1 network only."""
        B = action_seq.shape[0]
        if timestep is None:
            timestep = torch.zeros(B, device=action_seq.device, dtype=torch.long)
        
        q1_features = self.q1_net(action_seq, timestep, global_cond=obs_cond)
        q1 = self.q1_head(q1_features)
        
        return q1


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Soft update target network parameters.
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)
