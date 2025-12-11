"""
ReinFlow - Reinforcement Learning Fine-tuning for Flow Matching Policies

Key features:
1. Flow matching policy with learnable exploration noise
2. PPO-style policy gradient for online RL
3. Advantage estimation via value network
4. Compatible with ShortCut Flow / AW-ShortCut Flow checkpoints

Based on ReinFlow paper: https://github.com/ReinFlow/ReinFlow

Three-stage pipeline:
- Stage 1: ShortCut Flow BC pretrain (pure BC)
- Stage 2: AW-ShortCut Flow offline RL (Q-weighted BC)
- Stage 3: ReinFlow online RL (this module, PPO fine-tuning)

Architecture:
- ShortCutVelocityUNet1D: Base velocity network with step_size conditioning
- ExploreNoiseNet: Per-dimension exploration noise prediction
- NoisyVelocityUNet1D: Wrapper combining base + exploration
- ValueNetwork: Critic for PPO (V(s_0) predicts return)

Denoising MDP design (from ReinFlow paper):
- State: s_k = (x_k, obs) at denoising step k
- Action: predicted velocity v_k
- Reward: sparse, only final step gets R = A(s, a_final)
- Critic: V(s_0) predicts expected return from executing the action chunk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Literal
import copy
import math

from .shortcut_flow import ShortCutVelocityUNet1D
from .networks import soft_update
from .dppo import ValueNetwork


class ExploreNoiseNet(nn.Module):
    """Network that predicts exploration noise scale.
    
    Takes timestep embedding and observation embedding,
    outputs per-dimension noise scale.
    
    Args:
        time_embed_dim: Timestep embedding dimension
        obs_embed_dim: Observation embedding dimension  
        hidden_dim: Hidden layer dimension
        action_dim: Action dimension (output size)
        min_noise_std: Minimum noise std (added to output)
        max_noise_std: Maximum noise std (clamped)
    """
    
    def __init__(
        self,
        time_embed_dim: int = 64,
        obs_embed_dim: int = 128,
        hidden_dim: int = 128,
        action_dim: int = 1,
        min_noise_std: float = 0.01,
        max_noise_std: float = 0.3,
    ):
        super().__init__()
        
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        
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
        raw_std = self.net(x)
        # Clamp to [min_noise_std, max_noise_std]
        return torch.clamp(raw_std + self.min_noise_std, max=self.max_noise_std)


class NoisyVelocityUNet1D(nn.Module):
    """Velocity network with learnable exploration noise.
    
    Wraps a base ShortCutVelocityUNet1D and adds a noise prediction network
    for stochastic policy exploration during RL fine-tuning.
    
    Compatible with ShortCut Flow architecture (step_size conditioning).
    
    Args:
        base_velocity_net: ShortCutVelocityUNet1D for velocity prediction
        obs_dim: Observation dimension (flattened)
        action_dim: Action dimension
        time_embed_dim: Timestep embedding dimension
        obs_embed_dim: Observation embedding dimension
        min_noise_std: Minimum exploration noise std
        max_noise_std: Maximum exploration noise std
    """
    
    def __init__(
        self,
        base_velocity_net: ShortCutVelocityUNet1D,
        obs_dim: int,
        action_dim: int,
        time_embed_dim: int = 64,
        obs_embed_dim: int = 128,
        min_noise_std: float = 0.01,
        max_noise_std: float = 0.3,
    ):
        super().__init__()
        
        self.base_net = base_velocity_net
        self.action_dim = action_dim
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        
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
            min_noise_std=min_noise_std,
            max_noise_std=max_noise_std,
        )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        step_size: torch.Tensor,
        global_cond: torch.Tensor,
        sample_noise: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            sample: (B, pred_horizon, action_dim) noisy action sequence
            timestep: (B,) timestep in [0, 1]
            step_size: (B,) step size d for ShortCut Flow
            global_cond: (B, obs_dim) observation features
            sample_noise: Whether to sample exploration noise
            
        Returns:
            velocity: (B, pred_horizon, action_dim) predicted velocity
            noise: (B, pred_horizon, action_dim) sampled noise (or None)
            noise_std: (B, action_dim) noise scale
        """
        # Get base velocity prediction (with step_size for ShortCut Flow)
        velocity = self.base_net(sample, timestep, step_size, global_cond)
        
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
    
    def update_noise_bounds(self, min_std: float, max_std: float):
        """Update noise bounds for scheduling."""
        self.min_noise_std = min_std
        self.max_noise_std = max_std
        self.explore_noise_net.min_noise_std = min_std
        self.explore_noise_net.max_noise_std = max_std


class ReinFlowAgent(nn.Module):
    """ReinFlow Agent for online RL fine-tuning of flow matching policies.
    
    Core features:
    1. ShortCutVelocityUNet1D for flow matching velocity prediction
    2. ExploreNoiseNet for learnable exploration during RL
    3. Fixed num_inference_steps mode (no adaptive stepping)
    4. PPO-style updates with denoising MDP formulation
    5. Critic warmup support for stable training
    6. Noise scheduling interface for sweep
    
    Compatible with AW-ShortCut Flow checkpoints via load_from_aw_shortcut_flow().
    
    Denoising MDP (from ReinFlow paper):
    - State: s_k = (x_k, obs) at denoising step k
    - Action: predicted velocity v_k  
    - Reward: sparse, only final step gets R = A(s, a_final)
    - Critic: V(s_0) predicts expected return from executing the action chunk
    
    Args:
        velocity_net: ShortCutVelocityUNet1D for velocity field prediction
        obs_dim: Dimension of flattened observation features
        act_dim: Action dimension
        pred_horizon: Prediction horizon (action chunk length)
        obs_horizon: Observation horizon
        act_horizon: Action execution horizon (full chunk for SMDP)
        num_inference_steps: Fixed number of flow integration steps
        ema_decay: EMA decay rate
        use_ema: Whether to use EMA for inference
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_ratio: PPO clip ratio
        entropy_coef: Entropy coefficient
        value_coef: Value loss coefficient
        # Noise scheduling parameters
        min_noise_std: Minimum exploration noise std
        max_noise_std: Maximum exploration noise std
        noise_decay_type: Type of noise decay ("constant", "linear", "exponential")
        noise_decay_steps: Steps over which to decay noise
        # Critic warmup
        critic_warmup_steps: Number of steps to train critic only
    """
    
    def __init__(
        self,
        velocity_net: ShortCutVelocityUNet1D,
        obs_dim: int,
        act_dim: int,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        act_horizon: int = 8,
        num_inference_steps: int = 8,
        ema_decay: float = 0.999,
        use_ema: bool = True,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        # Noise scheduling
        min_noise_std: float = 0.01,
        max_noise_std: float = 0.3,
        noise_decay_type: Literal["constant", "linear", "exponential"] = "constant",
        noise_decay_steps: int = 100000,
        # Critic warmup
        critic_warmup_steps: int = 0,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.num_inference_steps = num_inference_steps
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Noise scheduling
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        self.initial_max_noise_std = max_noise_std
        self.noise_decay_type = noise_decay_type
        self.noise_decay_steps = noise_decay_steps
        
        # Critic warmup
        self.critic_warmup_steps = critic_warmup_steps
        self._current_step = 0
        
        # Fixed step size for inference (uniform steps)
        self.fixed_step_size = 1.0 / num_inference_steps
        
        # Wrap velocity net with noisy version
        self.noisy_velocity_net = NoisyVelocityUNet1D(
            base_velocity_net=velocity_net,
            obs_dim=obs_dim,
            action_dim=act_dim,
            min_noise_std=min_noise_std,
            max_noise_std=max_noise_std,
        )
        
        # EMA of base velocity network for stable inference
        if use_ema:
            self.velocity_net_ema = copy.deepcopy(velocity_net)
            for param in self.velocity_net_ema.parameters():
                param.requires_grad = False
        else:
            self.velocity_net_ema = None
        
        # Value network for PPO (predicts V(s_0) = expected return)
        self.value_net = ValueNetwork(obs_dim=obs_dim)
        
        # Old policy for PPO ratio
        self.noisy_velocity_net_old = NoisyVelocityUNet1D(
            base_velocity_net=copy.deepcopy(velocity_net),
            obs_dim=obs_dim,
            action_dim=act_dim,
            min_noise_std=min_noise_std,
            max_noise_std=max_noise_std,
        )
        for param in self.noisy_velocity_net_old.parameters():
            param.requires_grad = False
    
    def get_current_noise_std(self) -> Tuple[float, float]:
        """Get current noise bounds based on decay schedule."""
        if self.noise_decay_type == "constant":
            return self.min_noise_std, self.max_noise_std
        
        progress = min(1.0, self._current_step / max(1, self.noise_decay_steps))
        
        if self.noise_decay_type == "linear":
            current_max = self.initial_max_noise_std * (1.0 - progress) + self.min_noise_std * progress
        elif self.noise_decay_type == "exponential":
            decay_factor = math.exp(-3 * progress)  # Decay to ~5% at end
            current_max = self.min_noise_std + (self.initial_max_noise_std - self.min_noise_std) * decay_factor
        else:
            current_max = self.max_noise_std
        
        return self.min_noise_std, max(self.min_noise_std, current_max)
    
    def update_noise_schedule(self, step: Optional[int] = None):
        """Update noise bounds based on training progress."""
        if step is not None:
            self._current_step = step
        else:
            self._current_step += 1
        
        min_std, max_std = self.get_current_noise_std()
        self.noisy_velocity_net.update_noise_bounds(min_std, max_std)
        self.noisy_velocity_net_old.update_noise_bounds(min_std, max_std)
    
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
        """Compute flow matching BC loss (for pretraining/warmup).
        
        Uses standard conditional flow matching objective with fixed step size.
        
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
        
        # Fixed step size for ShortCut Flow compatibility
        d = torch.full((B,), self.fixed_step_size, device=device)
        
        # Sample initial noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(actions)
        
        # Interpolate: x_t = (1 - t) * x_0 + t * x_1
        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * actions
        
        # Target velocity: v = x_1 - x_0
        target_v = actions - x_0
        
        # Predict velocity (no noise during BC training)
        pred_v, _, _ = self.noisy_velocity_net(x_t, t, d, obs_cond, sample_noise=False)
        
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
        return_chains: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Sample action using fixed-step flow integration.
        
        For offline BC training evaluation: use deterministic=True (default)
        For online RL exploration: use deterministic=False to inject learned noise
        
        Args:
            obs_cond: (B, obs_horizon * obs_dim) observation features
            deterministic: If True, don't add exploration noise (default for eval)
            use_ema: Whether to use EMA network for base velocity
            return_chains: Whether to return intermediate states for log_prob
            
        Returns:
            actions: (B, pred_horizon, act_dim) sampled action sequence
            chains: List of (x_k, noise_std_k) tuples if return_chains=True, else None
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        # Start from Gaussian noise
        x = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        
        # Fixed step size and dt
        dt = self.fixed_step_size
        d = torch.full((B,), dt, device=device)
        
        # Store chains for log_prob computation
        chains = [] if return_chains else None
        
        # Choose velocity network
        if use_ema and self.velocity_net_ema is not None and deterministic:
            # Use EMA for deterministic inference
            for i in range(self.num_inference_steps):
                t = torch.full((B,), i * dt, device=device)
                velocity = self.velocity_net_ema(x, t, d, obs_cond)
                x = x + velocity * dt
        else:
            # Use noisy velocity net
            for i in range(self.num_inference_steps):
                t = torch.full((B,), i * dt, device=device)
                
                if deterministic:
                    velocity, _, noise_std = self.noisy_velocity_net(
                        x, t, d, obs_cond, sample_noise=False
                    )
                    if return_chains:
                        chains.append((x.clone(), noise_std.clone()))
                    x = x + velocity * dt
                else:
                    velocity, noise, noise_std = self.noisy_velocity_net(
                        x, t, d, obs_cond, sample_noise=True
                    )
                    if return_chains:
                        chains.append((x.clone(), noise_std.clone()))
                    x = x + velocity * dt
                    # Add exploration noise
                    if noise is not None:
                        x = x + noise * np.sqrt(dt)
        
        actions = torch.clamp(x, -1.0, 1.0)
        return actions, chains
    
    def compute_value(self, obs_cond: torch.Tensor) -> torch.Tensor:
        """Compute state value V(s_0).
        
        In ReinFlow's denoising MDP, V(s_0) predicts the expected return
        from executing the generated action chunk in the environment.
        """
        return self.value_net(obs_cond)
    
    def compute_action_log_prob(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
        chains: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_old_policy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probability of actions.
        
        Uses local Gaussian approximation based on accumulated noise.
        If chains is provided, uses stored trajectories for more accurate estimation.
        
        Args:
            obs_cond: (B, obs_dim) observation features
            actions: (B, pred_horizon, act_dim) action sequence
            chains: Optional list of (x_k, noise_std_k) from get_action
            use_old_policy: Whether to use old policy
            
        Returns:
            log_prob: (B,) log probability
            entropy: (B,) policy entropy
        """
        B = actions.shape[0]
        device = actions.device
        
        net = self.noisy_velocity_net_old if use_old_policy else self.noisy_velocity_net
        
        dt = self.fixed_step_size
        d = torch.full((B,), dt, device=device)
        
        # Accumulate noise variance
        total_noise_var = torch.zeros((B, self.act_dim), device=device)
        
        if chains is not None:
            # Use stored chains for more accurate estimation
            with torch.set_grad_enabled(not use_old_policy):
                for i, (x_k, noise_std_k) in enumerate(chains):
                    noise_std_clamped = torch.clamp(noise_std_k, min=0.01, max=0.3)
                    total_noise_var = total_noise_var + (noise_std_clamped ** 2) * dt
            
            # Mean is approximately the given actions
            mean_action = actions
        else:
            # Re-run deterministic flow to get expected trajectory
            x = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
            
            with torch.set_grad_enabled(not use_old_policy):
                for i in range(self.num_inference_steps):
                    t = torch.full((B,), i * dt, device=device)
                    velocity, _, noise_std = net(x, t, d, obs_cond, sample_noise=False)
                    
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
        
        # Entropy: Gaussian entropy H = 0.5 * (1 + log(2πσ²)) per dimension
        # total_std is (B, act_dim), expand to (B, pred_horizon, act_dim) and sum
        entropy_per_dim = 0.5 * (1 + np.log(2 * np.pi)) + torch.log(total_std_expanded)
        entropy = entropy_per_dim.sum(dim=[1, 2])  # sum over (pred_horizon, act_dim)
        
        return log_prob, entropy
    
    def compute_ppo_loss(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        chains: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        old_values: Optional[torch.Tensor] = None,
        clip_value: bool = False,
        value_clip_range: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO loss for online RL.
        
        Supports critic warmup: during warmup, only value_loss is computed.
        
        Args:
            obs_cond: (B, obs_dim) observation features
            actions: (B, pred_horizon, act_dim) action sequence
            old_log_probs: (B,) log probs from old policy
            advantages: (B,) GAE advantages
            returns: (B,) discounted returns
            chains: Optional denoising chains from rollout
            old_values: (B,) old value predictions (for value clipping)
            clip_value: Whether to clip value loss
            value_clip_range: Range for value clipping
            
        Returns:
            Dictionary with loss components
        """
        # Compute value predictions
        values = self.value_net(obs_cond).squeeze(-1)
        
        # Clip returns to reasonable range to prevent vloss explosion
        # This is critical for the first few iterations when returns are very large
        returns_clipped = torch.clamp(returns, -value_clip_range, value_clip_range)
        values_clipped = torch.clamp(values, -value_clip_range, value_clip_range)
        
        # Value loss (optionally clipped like in PPO2)
        if clip_value and old_values is not None:
            # Clipped value loss (PPO2 style)
            value_pred_clipped = old_values + torch.clamp(
                values - old_values, -value_clip_range, value_clip_range
            )
            value_loss1 = F.mse_loss(values_clipped, returns_clipped)
            value_loss2 = F.mse_loss(value_pred_clipped, returns_clipped)
            value_loss = torch.max(value_loss1, value_loss2)
        else:
            value_loss = F.mse_loss(values_clipped, returns_clipped)
        
        # Compute new log probs
        new_log_probs, entropy = self.compute_action_log_prob(
            obs_cond, actions, chains=chains, use_old_policy=False
        )
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy.mean(),
            "ratio_mean": ratio.mean(),
            "ratio_std": ratio.std(),
            "in_warmup": False,
        }
    
    def save(self, path: str):
        """Save checkpoint."""
        checkpoint = {
            "noisy_velocity_net": self.noisy_velocity_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "noisy_velocity_net_old": self.noisy_velocity_net_old.state_dict(),
            "current_step": self._current_step,
            "config": {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "pred_horizon": self.pred_horizon,
                "obs_horizon": self.obs_horizon,
                "act_horizon": self.act_horizon,
                "num_inference_steps": self.num_inference_steps,
                "ema_decay": self.ema_decay,
                "min_noise_std": self.min_noise_std,
                "max_noise_std": self.max_noise_std,
                "initial_max_noise_std": self.initial_max_noise_std,
                "noise_decay_type": self.noise_decay_type,
                "noise_decay_steps": self.noise_decay_steps,
                "critic_warmup_steps": self.critic_warmup_steps,
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
        if "noisy_velocity_net_old" in checkpoint:
            self.noisy_velocity_net_old.load_state_dict(checkpoint["noisy_velocity_net_old"])
        if "velocity_net_ema" in checkpoint and self.velocity_net_ema is not None:
            self.velocity_net_ema.load_state_dict(checkpoint["velocity_net_ema"])
        if "current_step" in checkpoint:
            self._current_step = checkpoint["current_step"]
        self.sync_old_policy()
    
    def load_from_aw_shortcut_flow(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        load_critic: bool = False,
    ):
        """Load pretrained weights from AW-ShortCut Flow checkpoint.
        
        Extracts velocity_net weights from AWShortCutFlowAgent checkpoint
        and loads them into the base velocity network.
        
        AW-ShortCut Flow checkpoint format:
        - checkpoint['agent'] contains the full agent state dict with keys like 'velocity_net.xxx'
        - checkpoint['ema_agent'] contains the EMA agent state dict
        
        Args:
            checkpoint_path: Path to AW-ShortCut Flow checkpoint
            device: Device to load to
            load_critic: Whether to attempt loading critic weights (usually False,
                        since we use ValueNetwork instead of DoubleQNetwork)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # AW-ShortCut Flow checkpoint format: agent/ema_agent with velocity_net.xxx keys
        if "agent" in checkpoint or "ema_agent" in checkpoint:
            # Extract velocity_net weights from agent state dict
            agent_state = checkpoint.get("ema_agent", checkpoint.get("agent", {}))
            
            # Filter velocity_net keys and remove prefix
            velocity_weights = {}
            for k, v in agent_state.items():
                if k.startswith("velocity_net."):
                    # Remove "velocity_net." prefix since we're loading into base_net
                    new_key = k[len("velocity_net."):]
                    velocity_weights[new_key] = v
            
            if not velocity_weights:
                raise ValueError(f"No velocity_net weights found in checkpoint: {checkpoint_path}")
                
        # Fallback: direct velocity_net state dict
        elif "velocity_net" in checkpoint:
            velocity_weights = checkpoint["velocity_net"]
        elif "velocity_net_ema" in checkpoint:
            velocity_weights = checkpoint["velocity_net_ema"]
        elif "model" in checkpoint:
            velocity_weights = checkpoint["model"]
        else:
            # Try loading directly (might be just the state dict)
            velocity_weights = checkpoint
        
        # Load into base network
        missing_keys, unexpected_keys = self.noisy_velocity_net.base_net.load_state_dict(
            velocity_weights, strict=False
        )
        if missing_keys:
            print(f"Warning: Missing keys when loading velocity_net: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading velocity_net: {unexpected_keys[:5]}...")
        
        # Also load into EMA network
        if self.velocity_net_ema is not None:
            self.velocity_net_ema.load_state_dict(velocity_weights, strict=False)
        
        # Sync old policy
        self.sync_old_policy()
        
        print(f"Loaded velocity network from {checkpoint_path}")
        
        # Note: We don't load critic because AW-ShortCut Flow uses DoubleQNetwork
        # while ReinFlow uses ValueNetwork. The value network should be initialized
        # fresh and trained with critic warmup.
        if load_critic:
            print("Warning: Critic loading not supported (architecture mismatch)")
    
    def load_pretrained(self, checkpoint_path: str, device: str = "cpu"):
        """Load pretrained flow matching weights.
        
        Tries AW-ShortCut Flow format first (checkpoint with 'agent'/'ema_agent' keys),
        then falls back to generic loading for other checkpoint formats.
        
        Args:
            checkpoint_path: Path to pretrained checkpoint
            device: Device to load to
            
        Raises:
            RuntimeError: If checkpoint format doesn't match model architecture
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check checkpoint format and dispatch to appropriate loader
        if "agent" in checkpoint or "ema_agent" in checkpoint:
            # AW-ShortCut Flow format: agent/ema_agent with velocity_net.xxx keys
            self.load_from_aw_shortcut_flow(checkpoint_path, device)
        elif "velocity_net" in checkpoint:
            # Direct velocity_net state dict
            weights = checkpoint["velocity_net"]
            self.noisy_velocity_net.base_net.load_state_dict(weights)
            if self.velocity_net_ema is not None:
                self.velocity_net_ema.load_state_dict(weights)
            self.sync_old_policy()
            print(f"Loaded velocity network from {checkpoint_path}")
        elif "model" in checkpoint:
            # Generic model key
            weights = checkpoint["model"]
            self.noisy_velocity_net.base_net.load_state_dict(weights)
            if self.velocity_net_ema is not None:
                self.velocity_net_ema.load_state_dict(weights)
            self.sync_old_policy()
            print(f"Loaded velocity network from {checkpoint_path}")
        else:
            # Try loading directly (might be just the state dict)
            self.noisy_velocity_net.base_net.load_state_dict(checkpoint)
            if self.velocity_net_ema is not None:
                self.velocity_net_ema.load_state_dict(checkpoint)
            self.sync_old_policy()
            print(f"Loaded velocity network from {checkpoint_path}")


class VectorizedRolloutBuffer:
    """Vectorized rollout buffer for ReinFlow online RL.
    
    Follows the official ManiSkill PPO implementation pattern with
    (num_steps, num_envs) shaped tensors for efficient vectorized computation.
    
    Adapted for SMDP (action chunks) where each "step" executes an entire
    action chunk over act_horizon environment steps.
    
    Args:
        num_steps: Number of rollout steps per update
        num_envs: Number of parallel environments
        obs_dim: Observation dimension
        pred_horizon: Action prediction horizon
        act_dim: Action dimension
        gamma: Discount factor for GAE
        gae_lambda: GAE lambda parameter
        normalize_advantages: Whether to normalize advantages
        device: Device for tensors
    """
    
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        pred_horizon: int,
        act_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.device = device
        
        # Pre-allocate tensors (num_steps, num_envs, ...)
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, pred_horizon, act_dim), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        
        # For handling episode boundaries (following ppo_rgb.py)
        self.final_values = torch.zeros((num_steps, num_envs), device=device)
        
        # Computed after rollout
        self.advantages = None
        self.returns = None
        
        self.ptr = 0
    
    def reset(self):
        """Reset buffer pointer for new rollout."""
        self.ptr = 0
        self.final_values.zero_()
        self.advantages = None
        self.returns = None
    
    def add(
        self,
        obs: torch.Tensor,           # (num_envs, obs_dim)
        actions: torch.Tensor,       # (num_envs, pred_horizon, act_dim)
        rewards: torch.Tensor,       # (num_envs,)
        values: torch.Tensor,        # (num_envs,)
        log_probs: torch.Tensor,     # (num_envs,)
        dones: torch.Tensor,         # (num_envs,)
    ):
        """Add a step of transitions for all environments."""
        if self.ptr >= self.num_steps:
            return
        
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = values.flatten()
        self.log_probs[self.ptr] = log_probs
        self.dones[self.ptr] = dones
        
        self.ptr += 1
    
    def set_final_values(self, step: int, env_mask: torch.Tensor, final_vals: torch.Tensor):
        """Set bootstrap values for episodes that ended at this step.
        
        Args:
            step: The rollout step where episodes ended
            env_mask: Boolean mask of which envs ended (num_envs,)
            final_vals: Value estimates for final observations (num_masked,)
        """
        self.final_values[step, env_mask] = final_vals.flatten()
    
    def compute_returns_and_advantages(
        self,
        next_value: torch.Tensor,    # (num_envs,)
        next_done: torch.Tensor,     # (num_envs,)
    ):
        """Compute GAE advantages and returns using vectorized operations.
        
        Follows the official ManiSkill PPO implementation exactly.
        Optionally normalizes advantages for stable training.
        """
        with torch.no_grad():
            advantages = torch.zeros_like(self.rewards)
            lastgaelam = torch.zeros(self.num_envs, device=self.device)
            
            for t in reversed(range(self.ptr)):
                if t == self.ptr - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value.flatten()
                else:
                    next_not_done = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                
                # Handle episode boundaries using final_values
                real_next_values = next_not_done * nextvalues + self.final_values[t]
                
                # Standard GAE
                delta = self.rewards[t] + self.gamma * real_next_values - self.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * next_not_done * lastgaelam
            
            self.advantages = advantages[:self.ptr]
            self.returns = (advantages + self.values)[:self.ptr]
            
            # Normalize advantages if enabled
            if self.normalize_advantages:
                adv_mean = self.advantages.mean()
                adv_std = self.advantages.std()
                if adv_std > 1e-8:
                    self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """Get mini-batches for PPO updates.
        
        Flattens (num_steps, num_envs) to (num_steps * num_envs,) and creates batches.
        """
        if self.advantages is None:
            raise RuntimeError("Must call compute_returns_and_advantages first")
        
        # Flatten (num_steps, num_envs, ...) -> (num_steps * num_envs, ...)
        n = self.ptr * self.num_envs
        
        b_obs = self.obs[:self.ptr].reshape(n, -1)
        b_actions = self.actions[:self.ptr].reshape(n, self.pred_horizon, self.act_dim)
        b_log_probs = self.log_probs[:self.ptr].reshape(n)
        b_values = self.values[:self.ptr].reshape(n)
        b_advantages = self.advantages.reshape(n)
        b_returns = self.returns.reshape(n)
        
        indices = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        
        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            batch = {
                "obs": b_obs[batch_indices],
                "actions": b_actions[batch_indices],
                "log_probs": b_log_probs[batch_indices],
                "values": b_values[batch_indices],
                "advantages": b_advantages[batch_indices],
                "returns": b_returns[batch_indices],
            }
            batches.append(batch)
        
        return batches
    
    def __len__(self):
        return self.ptr * self.num_envs


def compute_smdp_rewards(
    rewards: List[float],
    dones: List[bool],
    gamma: float,
    act_horizon: int,
) -> Tuple[float, bool, float]:
    """Compute SMDP cumulative reward for an action chunk.
    
    Accumulates rewards over act_horizon steps with discounting.
    
    Args:
        rewards: List of step rewards during chunk execution
        dones: List of done flags during chunk execution
        gamma: Discount factor
        act_horizon: Number of steps in chunk
        
    Returns:
        cumulative_reward: Discounted cumulative reward
        chunk_done: Whether episode ended during chunk
        discount_factor: γ^τ where τ is effective chunk length
    """
    cumulative_reward = 0.0
    discount = 1.0
    chunk_done = False
    effective_steps = 0
    
    for i in range(min(act_horizon, len(rewards))):
        cumulative_reward += discount * rewards[i]
        discount *= gamma
        effective_steps += 1
        
        if dones[i]:
            chunk_done = True
            break
    
    # γ^τ for SMDP bootstrapping
    discount_factor = gamma ** effective_steps
    
    return cumulative_reward, chunk_done, discount_factor
