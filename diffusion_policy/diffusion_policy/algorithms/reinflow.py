"""
ReinFlow - Flow Matching Policy with Learnable Exploration Noise

Flow matching based diffusion policy with learnable exploration noise
for online RL fine-tuning. The key innovation is NoisyVelocityUNet1D
which wraps the base velocity network and adds learnable noise injection.

For offline RL, this agent can be used for BC pretraining via compute_loss().

References:
- ReinFlow: https://reinflow.github.io/
- Code: https://github.com/ReinFlow/ReinFlow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union
import copy

from ..conditional_unet1d import ConditionalUnet1D
from .networks import VelocityUNet1D, soft_update


class ExploreNoiseNet(nn.Module):
    """Network that predicts exploration noise scale.
    
    Takes timestep embedding and observation embedding,
    outputs per-dimension noise scale.
    """
    
    def __init__(
        self,
        time_embed_dim: int = 64,
        obs_embed_dim: int = 128,
        hidden_dim: int = 128,
        action_dim: int = 1,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(time_embed_dim + obs_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus(),  # Ensure positive output
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Small init for output
        self.net[-2].weight.data *= 0.01
    
    def forward(
        self,
        time_emb: torch.Tensor,
        obs_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            time_emb: (B, time_embed_dim) timestep embedding
            obs_emb: (B, obs_embed_dim) observation embedding
            
        Returns:
            noise_std: (B, action_dim) noise scale per dimension
        """
        x = torch.cat([time_emb, obs_emb], dim=-1)
        # Add small constant to avoid zero noise
        return self.net(x) + 0.01


class NoisyVelocityUNet1D(nn.Module):
    """Velocity network with learnable exploration noise.
    
    Wraps a base VelocityUNet1D and adds a noise prediction network
    for stochastic policy exploration during RL fine-tuning.
    
    Args:
        base_velocity_net: Base VelocityUNet1D for velocity prediction
        obs_dim: Observation dimension (flattened)
        action_dim: Action dimension
        time_embed_dim: Timestep embedding dimension
        obs_embed_dim: Observation embedding dimension
    """
    
    def __init__(
        self,
        base_velocity_net: VelocityUNet1D,
        obs_dim: int,
        action_dim: int,
        time_embed_dim: int = 64,
        obs_embed_dim: int = 128,
    ):
        super().__init__()
        
        self.base_net = base_velocity_net
        self.action_dim = action_dim
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Observation embedding
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, obs_embed_dim),
            nn.LayerNorm(obs_embed_dim),
            nn.Mish(),
            nn.Linear(obs_embed_dim, obs_embed_dim),
        )
        
        # Exploration noise network
        self.explore_noise_net = ExploreNoiseNet(
            time_embed_dim=time_embed_dim,
            obs_embed_dim=obs_embed_dim,
            action_dim=action_dim,
        )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
        sample_noise: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            sample: (B, pred_horizon, action_dim) noisy action sequence
            timestep: (B,) timestep in [0, 1]
            global_cond: (B, obs_dim) observation features
            sample_noise: Whether to sample exploration noise
            
        Returns:
            velocity: (B, pred_horizon, action_dim) predicted velocity
            noise: (B, pred_horizon, action_dim) sampled noise (or None)
            noise_std: (B, action_dim) noise scale
        """
        # Get base velocity prediction
        velocity = self.base_net(sample, timestep, global_cond)
        
        # Compute embeddings for noise prediction
        t_emb = self.time_embed(timestep.unsqueeze(-1))
        obs_emb = self.obs_embed(global_cond)
        
        # Predict noise scale
        noise_std = self.explore_noise_net(t_emb, obs_emb)
        
        # Sample noise if requested
        if sample_noise:
            # noise_std: (B, action_dim) -> expand to (B, pred_horizon, action_dim)
            noise_std_expanded = noise_std.unsqueeze(1).expand(-1, sample.shape[1], -1)
            noise = torch.randn_like(sample) * noise_std_expanded
        else:
            noise = None
        
        return velocity, noise, noise_std


class ReinFlowAgent(nn.Module):
    """ReinFlow Agent for flow matching policy with exploration.
    
    Core features:
    1. VelocityUNet1D for flow matching velocity prediction
    2. ExploreNoiseNet for learnable exploration during RL
    3. EMA for stable inference
    4. PPO-style updates for online RL
    
    For offline RL, use compute_loss() for BC pretraining.
    For online RL, use the PPO methods.
    
    Args:
        velocity_net: VelocityUNet1D for velocity field prediction
        obs_dim: Dimension of flattened observation features
        act_dim: Action dimension
        pred_horizon: Prediction horizon (action chunk length)
        obs_horizon: Observation horizon
        act_horizon: Action execution horizon
        num_flow_steps: Number of flow integration steps
        ema_decay: EMA decay rate
        use_ema: Whether to use EMA for inference
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_ratio: PPO clip ratio
        entropy_coef: Entropy coefficient
        value_coef: Value loss coefficient
    """
    
    def __init__(
        self,
        velocity_net: VelocityUNet1D,
        obs_dim: int,
        act_dim: int,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        act_horizon: int = 8,
        num_flow_steps: int = 10,
        ema_decay: float = 0.999,
        use_ema: bool = True,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.num_flow_steps = num_flow_steps
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Wrap velocity net with noisy version
        self.noisy_velocity_net = NoisyVelocityUNet1D(
            base_velocity_net=velocity_net,
            obs_dim=obs_dim,
            action_dim=act_dim,
        )
        
        # EMA of base velocity network for stable inference
        if use_ema:
            self.velocity_net_ema = copy.deepcopy(velocity_net)
            for param in self.velocity_net_ema.parameters():
                param.requires_grad = False
        else:
            self.velocity_net_ema = None
        
        # Value network for PPO
        from .dppo import ValueNetwork
        self.value_net = ValueNetwork(obs_dim=obs_dim)
        
        # Old policy for PPO ratio
        self.noisy_velocity_net_old = NoisyVelocityUNet1D(
            base_velocity_net=copy.deepcopy(velocity_net),
            obs_dim=obs_dim,
            action_dim=act_dim,
        )
        for param in self.noisy_velocity_net_old.parameters():
            param.requires_grad = False
    
    def update_ema(self):
        """Update EMA of velocity network."""
        if self.velocity_net_ema is not None:
            soft_update(
                self.velocity_net_ema,
                self.noisy_velocity_net.base_net,
                1 - self.ema_decay
            )
    
    def sync_old_policy(self):
        """Sync old policy with current for PPO ratio."""
        self.noisy_velocity_net_old.base_net.load_state_dict(
            self.noisy_velocity_net.base_net.state_dict()
        )
        self.noisy_velocity_net_old.explore_noise_net.load_state_dict(
            self.noisy_velocity_net.explore_noise_net.state_dict()
        )
        self.noisy_velocity_net_old.time_embed.load_state_dict(
            self.noisy_velocity_net.time_embed.state_dict()
        )
        self.noisy_velocity_net_old.obs_embed.load_state_dict(
            self.noisy_velocity_net.obs_embed.state_dict()
        )
    
    def compute_loss(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute flow matching BC loss.
        
        Uses standard conditional flow matching objective.
        
        Args:
            obs_cond: (B, obs_horizon * obs_dim) observation features
            actions: (B, pred_horizon, act_dim) ground truth action sequence
            
        Returns:
            Dictionary with 'loss' and 'bc_loss' keys
        """
        B = actions.shape[0]
        device = actions.device
        
        # Sample random timesteps t ~ U[0, 1]
        t = torch.rand(B, device=device)
        
        # Sample initial noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(actions)
        
        # Interpolate: x_t = (1 - t) * x_0 + t * x_1
        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * actions
        
        # Target velocity: v = x_1 - x_0
        target_v = actions - x_0
        
        # Predict velocity (no noise during BC training)
        pred_v, _, _ = self.noisy_velocity_net(x_t, t, obs_cond, sample_noise=False)
        
        # MSE loss
        loss = F.mse_loss(pred_v, target_v)
        
        return {
            "loss": loss,
            "bc_loss": loss,
        }
    
    @torch.no_grad()
    def get_action(
        self,
        obs_cond: torch.Tensor,
        deterministic: bool = True,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Sample action using flow integration.
        
        For offline BC training evaluation: use deterministic=True (default)
        For online RL exploration: use deterministic=False to inject learned noise
        
        Args:
            obs_cond: (B, obs_horizon * obs_dim) observation features
            deterministic: If True, don't add exploration noise (default for eval)
            use_ema: Whether to use EMA network for base velocity
            
        Returns:
            actions: (B, pred_horizon, act_dim) sampled action sequence
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        # Start from Gaussian noise
        x = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        
        dt = 1.0 / self.num_flow_steps
        
        # Choose velocity network: EMA for stable inference, base_net otherwise
        if use_ema and self.velocity_net_ema is not None:
            velocity_net = self.velocity_net_ema
        else:
            velocity_net = self.noisy_velocity_net.base_net
        
        # Flow integration
        for i in range(self.num_flow_steps):
            t = torch.full((B,), i * dt, device=device)
            
            if deterministic:
                # Deterministic: use base/EMA velocity only, no exploration noise
                velocity = velocity_net(x, t, obs_cond)
                x = x + velocity * dt
            else:
                # Stochastic: use noisy velocity net with learned exploration
                velocity, noise, noise_std = self.noisy_velocity_net(
                    x, t, obs_cond, sample_noise=True
                )
                x = x + velocity * dt
                # Add exploration noise
                if noise is not None:
                    x = x + noise * np.sqrt(dt)
        
        return torch.clamp(x, -1.0, 1.0)
    
    def compute_value(self, obs_cond: torch.Tensor) -> torch.Tensor:
        """Compute state value."""
        return self.value_net(obs_cond)
    
    def compute_action_log_prob(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
        use_old_policy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probability of actions.
        
        Uses local Gaussian approximation based on accumulated noise.
        
        Args:
            obs_cond: (B, obs_dim) observation features
            actions: (B, pred_horizon, act_dim) action sequence
            use_old_policy: Whether to use old policy
            
        Returns:
            log_prob: (B,) log probability
            entropy: (B,) policy entropy
        """
        B = actions.shape[0]
        device = actions.device
        
        net = self.noisy_velocity_net_old if use_old_policy else self.noisy_velocity_net
        
        dt = 1.0 / self.num_flow_steps
        
        # Run deterministic flow to get expected trajectory
        x = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        
        # Accumulate noise variance
        total_noise_var = torch.zeros((B, self.act_dim), device=device)
        
        with torch.set_grad_enabled(not use_old_policy):
            for i in range(self.num_flow_steps):
                t = torch.full((B,), i * dt, device=device)
                velocity, _, noise_std = net(x, t, obs_cond, sample_noise=False)
                
                # Accumulate variance
                noise_std_clamped = torch.clamp(noise_std, min=0.01, max=0.3)
                total_noise_var = total_noise_var + (noise_std_clamped ** 2) * dt
                
                # Deterministic step
                x = x + velocity * dt
        
        mean_action = torch.clamp(x, -1.0, 1.0)
        
        # Total std
        total_std = torch.sqrt(total_noise_var + 1e-6)
        total_std = torch.clamp(total_std, min=0.05, max=0.8)
        
        # Expand std for action sequence: (B, act_dim) -> (B, pred_horizon, act_dim)
        total_std_expanded = total_std.unsqueeze(1).expand(-1, self.pred_horizon, -1)
        
        # Log probability
        diff = actions - mean_action
        log_prob = -0.5 * (diff / total_std_expanded) ** 2 - torch.log(total_std_expanded)
        log_prob = log_prob - 0.5 * np.log(2 * np.pi)
        log_prob = log_prob.sum(dim=[1, 2])
        log_prob = torch.clamp(log_prob, min=-20.0, max=5.0)
        
        # Entropy
        entropy = 0.5 * (1 + np.log(2 * np.pi)) + torch.log(total_std)
        entropy = entropy.sum(dim=-1) * self.pred_horizon
        
        return log_prob, entropy
    
    def compute_ppo_loss(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO loss for online RL.
        
        Args:
            obs_cond: (B, obs_dim) observation features
            actions: (B, pred_horizon, act_dim) action sequence
            old_log_probs: (B,) log probs from old policy
            advantages: (B,) GAE advantages
            returns: (B,) discounted returns
            
        Returns:
            Dictionary with loss components
        """
        # Compute new log probs
        new_log_probs, entropy = self.compute_action_log_prob(
            obs_cond, actions, use_old_policy=False
        )
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = self.value_net(obs_cond).squeeze(-1)
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy.mean(),
        }
    
    def save(self, path: str):
        """Save checkpoint."""
        checkpoint = {
            "noisy_velocity_net": self.noisy_velocity_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "config": {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "pred_horizon": self.pred_horizon,
                "obs_horizon": self.obs_horizon,
                "act_horizon": self.act_horizon,
                "num_flow_steps": self.num_flow_steps,
                "ema_decay": self.ema_decay,
            },
        }
        if self.velocity_net_ema is not None:
            checkpoint["velocity_net_ema"] = self.velocity_net_ema.state_dict()
        torch.save(checkpoint, path)
    
    def load(self, path: str, device: str = "cpu"):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        self.noisy_velocity_net.load_state_dict(checkpoint["noisy_velocity_net"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        if "velocity_net_ema" in checkpoint and self.velocity_net_ema is not None:
            self.velocity_net_ema.load_state_dict(checkpoint["velocity_net_ema"])
        self.sync_old_policy()
    
    def load_pretrained(self, checkpoint_path: str, device: str = "cpu"):
        """Load pretrained flow matching weights.
        
        Loads weights into base velocity network.
        
        Args:
            checkpoint_path: Path to pretrained checkpoint
            device: Device to load to
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Try different key names
        if "velocity_net" in checkpoint:
            weights = checkpoint["velocity_net"]
        elif "model" in checkpoint:
            weights = checkpoint["model"]
        else:
            weights = checkpoint
        
        # Load into base network
        self.noisy_velocity_net.base_net.load_state_dict(weights, strict=False)
        
        if self.velocity_net_ema is not None:
            self.velocity_net_ema.load_state_dict(weights, strict=False)
        
        self.sync_old_policy()
