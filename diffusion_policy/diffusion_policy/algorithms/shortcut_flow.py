"""
ShortCut Flow Policy with adaptive step size.
Migrated from toy_diffusion_rl (ReinFlow) to diffusion_policy framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal, Tuple
import copy
import math

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from .networks import soft_update


class ShortCutVelocityUNet1D(nn.Module):
    """
    Velocity network with step size conditioning for ShortCut Flow.
    Extends ConditionalUnet1D to accept both time t and step size d.
    """
    
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.unet = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        
        # Additional embedding for step size d
        self.step_size_embed = nn.Sequential(
            nn.Linear(1, diffusion_step_embed_dim),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
        )
        
        # Combine t and d embeddings
        self.combine_embed = nn.Sequential(
            nn.Linear(diffusion_step_embed_dim * 2, diffusion_step_embed_dim),
            nn.Mish(),
        )
        
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        step_size: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with time and step size conditioning.
        
        Args:
            sample: [B, T, input_dim] action sequence
            timestep: [B] diffusion timestep (0-1)
            step_size: [B] step size d (0-1)
            global_cond: [B, cond_dim] or [B, obs_horizon, cond_dim] conditioning features
        """
        # Get step size embedding (currently unused, but available for future extensions)
        d_embed = self.step_size_embed(step_size.unsqueeze(-1))
        
        # Flatten obs for global conditioning
        if global_cond is not None and global_cond.dim() == 3:
            global_cond = global_cond.reshape(global_cond.shape[0], -1)
        
        # Use timestep as integer (scaled by 100 for embedding, matching VelocityUNet1D)
        timestep_int = (timestep * 100).long()
        
        # ConditionalUnet1D expects input as (B, T, input_dim) and returns same shape
        output = self.unet(sample, timestep_int, global_cond=global_cond)
        
        return output


class ShortCutFlowAgent(nn.Module):
    """
    ShortCut Flow Matching agent with adaptive step sizes.
    Learns to take larger steps when possible, enabling faster inference.
    Based on ReinFlow implementation.
    """
    
    def __init__(
        self,
        velocity_net: ShortCutVelocityUNet1D,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        max_denoising_steps: int = 8,
        self_consistency_k: float = 0.25,  # Fraction of batch for consistency
        flow_weight: float = 1.0,
        shortcut_weight: float = 1.0,
        ema_decay: float = 0.999,
        device: str = "cuda",
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.max_denoising_steps = max_denoising_steps
        self.self_consistency_k = self_consistency_k
        self.flow_weight = flow_weight
        self.shortcut_weight = shortcut_weight
        self.ema_decay = ema_decay
        self.device = device
        
        # EMA velocity network for stable shortcut targets
        self.velocity_net_ema = copy.deepcopy(velocity_net)
        for param in self.velocity_net_ema.parameters():
            param.requires_grad = False
        
        # Log2 of max steps for step size sampling
        self.log_max_steps = int(math.log2(max_denoising_steps))
        
    def _sample_step_size(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample step sizes d from {1/N, 2/N, 4/N, ...} where N is max_steps.
        Uses log-uniform distribution.
        """
        # Sample power of 2: 0, 1, 2, ..., log_max_steps
        powers = torch.randint(0, self.log_max_steps + 1, (batch_size,), device=device)
        # d = 2^power / max_steps
        d = (2.0 ** powers.float()) / self.max_denoising_steps
        return d
    
    def _compute_shortcut_target(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        d: torch.Tensor,
        obs_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute shortcut target: what velocity would take us 2d steps instead of d.
        Uses two small steps with EMA network to compute the target for one large step.
        
        v(x_t, t, 2d) ≈ v(x_t, t, d) + d * v(x_{t+d}, t+d, d)
        """
        with torch.no_grad():
            # First step with step size d (using EMA for stable targets)
            v_1 = self.velocity_net_ema(x_t, t, d, obs_features)
            d_expand = d.view(-1, 1, 1)
            x_t_plus_d = x_t + d_expand * v_1
            
            # Second step from t+d with step size d
            t_plus_d = t + d
            v_2 = self.velocity_net_ema(x_t_plus_d, t_plus_d, d, obs_features)
            
            # Combined velocity for step size 2d
            # v(t, 2d) should equal (v(t, d) + v(t+d, d)) / 2 for 2d total movement
            shortcut_v = v_1 + d_expand * v_2  # This takes us 2d in one step
            
        return shortcut_v
    
    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ShortCut Flow loss: standard flow + self-consistency.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim]
            actions: Expert actions [B, pred_horizon, action_dim]
        """
        batch_size = actions.shape[0]
        device = actions.device
        
        # Sample noise
        x_0 = torch.randn_like(actions)
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=device)
        
        # Sample step sizes
        d = self._sample_step_size(batch_size, device)
        
        # Ensure t + 2d <= 1 for shortcut training
        # Clamp t to leave room for 2d
        max_t = 1.0 - 2 * d
        t = torch.clamp(t * max_t.clamp(min=0.01), min=0.0, max=0.99)
        
        # Interpolate
        t_expand = t.view(-1, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * actions
        
        # Standard flow matching target (for d -> 0)
        v_target = actions - x_0
        
        # Predict velocity with step size d
        v_pred = self.velocity_net(x_t, t, d, obs_features)
        
        # Flow matching loss (for small d, v should match standard CFM)
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # Self-consistency loss for larger step sizes
        shortcut_loss = torch.tensor(0.0, device=device)
        
        if self.shortcut_weight > 0 and self.self_consistency_k > 0:
            # Only compute for subset of batch (efficiency)
            n_consistency = max(1, int(batch_size * self.self_consistency_k))
            idx = torch.randperm(batch_size)[:n_consistency]
            
            x_t_sub = x_t[idx]
            t_sub = t[idx]
            d_sub = d[idx]
            obs_sub = obs_features[idx]
            
            # Double the step size for shortcut
            d_double = 2 * d_sub
            
            # Only train where 2d is valid (not exceeding 1)
            valid_mask = (t_sub + d_double) <= 1.0
            
            if valid_mask.sum() > 0:
                x_t_valid = x_t_sub[valid_mask]
                t_valid = t_sub[valid_mask]
                d_valid = d_sub[valid_mask]
                d_double_valid = d_double[valid_mask]
                obs_valid = obs_sub[valid_mask]
                
                # Shortcut target from two d-steps
                shortcut_target = self._compute_shortcut_target(
                    x_t_valid, t_valid, d_valid, obs_valid
                )
                
                # Predict with 2d step size
                v_pred_2d = self.velocity_net(
                    x_t_valid, t_valid, d_double_valid, obs_valid
                )
                
                shortcut_loss = F.mse_loss(v_pred_2d, shortcut_target)
        
        total_loss = self.flow_weight * flow_loss + self.shortcut_weight * shortcut_loss
        
        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "shortcut_loss": shortcut_loss,
        }
    
    def update_ema(self):
        """Update EMA velocity network.
        
        Formula: θ_ema = ema_decay * θ_ema + (1 - ema_decay) * θ
        Equivalent to: soft_update(ema, source, tau=1-ema_decay)
        """
        soft_update(self.velocity_net_ema, self.velocity_net, 1 - self.ema_decay)
    
    @torch.no_grad()
    def get_action(
        self,
        obs_features: torch.Tensor,
        num_steps: Optional[int] = None,
        adaptive: bool = True,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        Generate action using ShortCut flow with adaptive step sizes.
        
        Args:
            obs_features: Encoded observation [B, obs_horizon, obs_dim]
            num_steps: Override number of steps (uses max possible step size)
            adaptive: If True, use learned adaptive steps; else use uniform steps
            use_ema: Whether to use EMA network for sampling (default: True)
            
        Returns:
            actions: Generated action sequence [B, pred_horizon, action_dim]
        """
        self.velocity_net.eval()
        net = self.velocity_net_ema if use_ema else self.velocity_net
        batch_size = obs_features.shape[0]
        device = obs_features.device
        
        # Start from noise
        x = torch.randn(
            batch_size, self.pred_horizon, self.action_dim,
            device=device
        )
        
        if adaptive and num_steps is None:
            # Use adaptive step sizes (powers of 2)
            # Start with largest possible step size
            t = torch.zeros(batch_size, device=device)
            
            while t[0] < 1.0:
                # Find largest valid step size
                remaining = 1.0 - t[0]
                
                # Use power-of-2 step sizes
                d_val = min(remaining.item(), 1.0 / 2)  # At least 2 steps
                for power in range(self.log_max_steps, -1, -1):
                    candidate = (2.0 ** power) / self.max_denoising_steps
                    if candidate <= remaining:
                        d_val = candidate
                        break
                
                d = torch.full((batch_size,), d_val, device=device)
                
                v = net(x, t, d, obs_features)
                x = x + d.view(-1, 1, 1) * v
                t = t + d
                
                # Safety check
                if d_val < 1e-6:
                    break
        else:
            # Uniform steps
            steps = num_steps if num_steps is not None else self.max_denoising_steps
            dt = 1.0 / steps
            d = torch.full((batch_size,), dt, device=device)
            
            for i in range(steps):
                t = torch.full((batch_size,), i * dt, device=device)
                v = net(x, t, d, obs_features)
                x = x + dt * v
        
        # Clamp to action bounds
        x = torch.clamp(x, -1.0, 1.0)
        
        self.velocity_net.train()
        return x
