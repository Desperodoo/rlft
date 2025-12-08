"""
Diffusion Double Q Learning Agent

Combines Diffusion Policy with Double Q-Learning for offline RL.
Uses 1D U-Net architecture aligned with the official diffusion_policy implementation.

References:
- Diffusion-QL: https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL
- IDQL: https://github.com/philippe-eecs/IDQL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import copy

from .networks import DoubleQNetwork, soft_update
from ..conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class DiffusionDoubleQAgent(nn.Module):
    """Diffusion Double Q Agent with 1D U-Net architecture.
    
    Combines diffusion-based actor with twin Q-networks for offline RL.
    
    Hyperparameters inherited from toy_diffusion_rl:
    - alpha: 0.01 (Q-value weight, reduced for stability with dense rewards)
    - bc_weight: 1.0
    - reward_scale: 0.1
    - tau: 0.005
    - gamma: 0.99
    
    Args:
        noise_pred_net: ConditionalUnet1D for noise prediction (actor)
        q_network: DoubleQNetwork for Q-value estimation (critic)
        action_dim: Dimension of action space
        obs_horizon: Number of observation frames (default: 2)
        pred_horizon: Length of action sequence to predict (default: 16)
        num_diffusion_iters: Number of diffusion steps (default: 100)
        alpha: Weight for Q-value term (default: 0.01, reduced for dense reward stability)
        bc_weight: Weight for BC loss (default: 1.0)
        gamma: Discount factor (default: 0.99)
        tau: Soft update coefficient (default: 0.005)
        reward_scale: Scale factor for rewards (default: 0.1)
        q_target_clip: Clip range for Q target (default: 100.0)
        device: Device to run on (default: "cuda")
    """
    
    def __init__(
        self,
        noise_pred_net: ConditionalUnet1D,
        q_network: DoubleQNetwork,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        num_diffusion_iters: int = 100,
        alpha: float = 0.01,
        bc_weight: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scale: float = 0.1,
        q_target_clip: float = 100.0,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.noise_pred_net = noise_pred_net
        self.critic = q_network
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.num_diffusion_iters = num_diffusion_iters
        self.alpha = alpha
        self.bc_weight = bc_weight
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.q_target_clip = q_target_clip
        self.device = device
        
        # Target critic
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        # Noise scheduler (aligned with official implementation)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
    
    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs_features: torch.Tensor,
        dones: torch.Tensor,
        # SMDP chunk-level fields (optional, for proper action chunking)
        cumulative_reward: Optional[torch.Tensor] = None,
        chunk_done: Optional[torch.Tensor] = None,
        discount_factor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined actor and critic loss.
        
        Supports both standard TD learning and SMDP formulation for action chunking.
        
        For SMDP (action chunking), the Bellman target is:
        y_t = R_t^(τ) + (1 - d_t^(τ)) * γ^τ * max Q(s_{t+τ}, a')
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            actions: (B, pred_horizon, action_dim) expert action sequence
            rewards: (B,) or (B, 1) single-step rewards (used if cumulative_reward not provided)
            next_obs_features: Next observation features (s_{t+τ} for SMDP)
            dones: (B,) or (B, 1) done flags (used if chunk_done not provided)
            cumulative_reward: (B,) or (B, 1) SMDP cumulative discounted reward R_t^(τ)
            chunk_done: (B,) or (B, 1) SMDP done flag (1 if episode ends within chunk)
            discount_factor: (B,) or (B, 1) SMDP discount γ^τ
            
        Returns:
            Dict with loss components
        """
        # Flatten obs_features if 3D
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
            
        if next_obs_features.dim() == 3:
            next_obs_cond = next_obs_features.reshape(next_obs_features.shape[0], -1)
        else:
            next_obs_cond = next_obs_features
        
        # Compute actor loss
        actor_dict = self._compute_actor_loss(obs_cond, actions)
        
        # Compute critic loss (with SMDP support)
        critic_loss = self._compute_critic_loss(
            obs_cond, next_obs_cond, actions, rewards, dones,
            cumulative_reward=cumulative_reward,
            chunk_done=chunk_done,
            discount_factor=discount_factor,
        )
        
        # Total loss
        total_loss = actor_dict["actor_loss"] + critic_loss
        
        return {
            "loss": total_loss,
            "actor_loss": actor_dict["actor_loss"],
            "bc_loss": actor_dict["bc_loss"],
            "q_policy_loss": actor_dict["q_loss"],
            "critic_loss": critic_loss,
            "q_mean": actor_dict["q_mean"],
        }
    
    def _compute_actor_loss(self, obs_cond: torch.Tensor, action_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute actor loss: BC loss + Q-value maximization."""
        B = action_seq.shape[0]
        device = action_seq.device
        
        # BC Loss: Diffusion training objective
        noise = torch.randn_like(action_seq)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)
        bc_loss = F.mse_loss(noise_pred, noise)
        
        # Q-value maximization loss
        with torch.no_grad():
            generated_actions = self._sample_actions_batch(obs_cond, num_steps=5)
        
        # Compute Q-value of generated actions at t=0
        t_last = torch.zeros(B, device=device, dtype=torch.long)
        noise_for_q = torch.randn_like(action_seq)
        noisy_for_q = self.noise_scheduler.add_noise(generated_actions.detach(), noise_for_q, t_last)
        
        pred_noise_q = self.noise_pred_net(noisy_for_q, t_last, global_cond=obs_cond)
        
        # Reconstruct action from noise prediction
        alpha_bar = self.noise_scheduler.alphas_cumprod[t_last].view(-1, 1, 1)
        actions_for_q = (noisy_for_q - torch.sqrt(1 - alpha_bar) * pred_noise_q) / torch.sqrt(alpha_bar)
        actions_for_q = torch.clamp(actions_for_q, -1.0, 1.0)
        
        # IMPORTANT: We need gradients to flow to actor (noise_pred_net) but NOT to critic
        # The Q-maximization gradient should only update the actor, not the critic
        # We achieve this by temporarily disabling gradients for critic during this forward pass
        for p in self.critic.parameters():
            p.requires_grad = False
        
        q_value = self.critic.q1_forward(actions_for_q, obs_cond)
        
        for p in self.critic.parameters():
            p.requires_grad = True
        
        # Clip Q-values to prevent explosion in actor optimization
        if self.q_target_clip is not None:
            q_value = torch.clamp(q_value, -self.q_target_clip, self.q_target_clip)
        
        q_loss = -q_value.mean()
        
        # Total actor loss
        actor_loss = self.bc_weight * bc_loss + self.alpha * q_loss
        
        return {
            "actor_loss": actor_loss,
            "bc_loss": bc_loss,
            "q_loss": q_loss,
            "q_mean": q_value.mean(),
        }
    
    def _compute_critic_loss(
        self,
        obs_cond: torch.Tensor,
        next_obs_cond: torch.Tensor,
        action_seq: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        # SMDP fields
        cumulative_reward: Optional[torch.Tensor] = None,
        chunk_done: Optional[torch.Tensor] = None,
        discount_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute critic loss using SMDP Bellman equation for action chunking.
        
        SMDP Bellman target:
        y_t = R_t^(τ) + (1 - d_t^(τ)) * γ^τ * min Q(s_{t+τ}, a')
        
        where:
        - R_t^(τ) = Σ_{i=0}^{τ-1} γ^i r_{t+i} (cumulative discounted reward)
        - d_t^(τ) = 1 if episode ends within chunk
        - γ^τ = discount factor over τ steps
        - s_{t+τ} = state after chunk execution
        
        Falls back to standard single-step TD if SMDP fields not provided.
        """
        # Use SMDP fields if provided, otherwise fall back to single-step
        if cumulative_reward is not None:
            r = cumulative_reward
            d = chunk_done if chunk_done is not None else dones
            gamma_tau = discount_factor if discount_factor is not None else torch.full_like(r, self.gamma)
        else:
            # Fall back to single-step TD (legacy behavior)
            r = rewards
            d = dones
            gamma_tau = torch.full_like(r if r.dim() == 1 else r.squeeze(-1), self.gamma)
        
        # Ensure proper shape
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        if d.dim() == 1:
            d = d.unsqueeze(-1)
        if gamma_tau.dim() == 1:
            gamma_tau = gamma_tau.unsqueeze(-1)
        
        # Scale rewards
        scaled_rewards = r * self.reward_scale
        
        with torch.no_grad():
            # Sample next actions using current policy
            next_actions = self._sample_actions_batch(next_obs_cond, num_steps=5)
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_actions, next_obs_cond)
            target_q = torch.min(target_q1, target_q2)
            
            # SMDP TD target: R_t^(τ) + (1 - d_t^(τ)) * γ^τ * Q(s_{t+τ}, a')
            target_q = scaled_rewards + (1 - d) * gamma_tau * target_q
            
            # Clip targets
            if self.q_target_clip is not None:
                target_q = torch.clamp(target_q, -self.q_target_clip, self.q_target_clip)
        
        # Current Q-values
        current_q1, current_q2 = self.critic(action_seq, obs_cond)
        
        # MSE loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        return critic_loss
    
    def _sample_actions_batch(self, obs_cond: torch.Tensor, num_steps: int = 5) -> torch.Tensor:
        """Sample actions using DDPM sampling with reduced steps."""
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        # Use subset of timesteps for faster sampling
        step_indices = list(range(0, self.num_diffusion_iters, self.num_diffusion_iters // num_steps))[::-1]
        
        for t in step_indices:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.noise_pred_net(x, t_batch, global_cond=obs_cond)
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample
        
        return torch.clamp(x, -1.0, 1.0)
    
    def update_target(self):
        """Soft update target critic network."""
        soft_update(self.critic_target, self.critic, self.tau)
    
    @torch.no_grad()
    def get_action(
        self,
        obs_features: torch.Tensor,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample action sequence for evaluation.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            num_inference_steps: Number of diffusion steps (default: all steps)
            
        Returns:
            actions: (B, pred_horizon, action_dim) action sequence
        """
        self.noise_pred_net.eval()
        
        # Flatten obs_features if 3D
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
        
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        # Full DDPM sampling
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        if num_inference_steps is not None:
            step_indices = list(range(0, self.num_diffusion_iters, self.num_diffusion_iters // num_inference_steps))[::-1]
        else:
            step_indices = self.noise_scheduler.timesteps
        
        for k in step_indices:
            noise_pred = self.noise_pred_net(
                sample=x,
                timestep=k,
                global_cond=obs_cond,
            )
            x = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=x,
            ).prev_sample
        
        x = torch.clamp(x, -1.0, 1.0)
        
        self.noise_pred_net.train()
        return x
