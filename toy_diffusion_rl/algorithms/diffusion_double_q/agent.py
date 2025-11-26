"""
Diffusion Double Q Learning Agent

Combines Diffusion Policy with Double Q-Learning for offline RL.
The diffusion actor is trained with both denoising loss and Q-maximization.

References:
- Diffusion-QL: https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL
- IDQL: https://github.com/philippe-eecs/IDQL
- TD3 (Twin Delayed DDPG) for Double Q idea
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import copy

from ...common.networks import DiffusionNoisePredictor, DoubleQNetwork
from ...common.utils import DiffusionHelper, soft_update


class DiffusionDoubleQAgent:
    """Diffusion Policy with Double Q-Learning.
    
    Combines a diffusion-based actor with twin Q-networks:
    - Actor: Diffusion model that generates actions from noise
    - Critics: Two Q-networks (Q1, Q2) for reduced overestimation
    
    Training:
    - Critics trained via Bellman backup using min(Q1, Q2) targets
    - Actor loss = diffusion_loss - alpha * Q-value
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        num_diffusion_steps: Number of diffusion steps
        learning_rate_actor: Learning rate for actor
        learning_rate_critic: Learning rate for critics
        gamma: Discount factor
        tau: Soft update coefficient
        alpha: Weight for Q-value term in actor loss
        device: Device for computation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        num_diffusion_steps: int = 10,
        learning_rate_actor: float = 3e-4,
        learning_rate_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 1.0,
        bc_weight: float = 1.0,
        noise_schedule: str = "linear",
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.bc_weight = bc_weight
        self.device = device
        
        # Actor: Diffusion noise predictor
        self.actor = DiffusionNoisePredictor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims + [256]
        ).to(device)
        
        # Critics: Double Q-networks
        self.critic = DoubleQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        # Target critics
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
            
        # Diffusion helper
        self.diffusion = DiffusionHelper(
            num_diffusion_steps=num_diffusion_steps,
            schedule=noise_schedule,
            device=device
        )
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=learning_rate_critic
        )
        
        # Training step counter
        self.total_steps = 0
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step.
        
        Updates both critic and actor networks.
        
        Args:
            batch: Dictionary with states, actions, rewards, next_states, dones
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        
        # ============ Update Critics ============
        with torch.no_grad():
            # Sample next actions from diffusion policy
            next_actions = self._sample_actions_batch(next_states)
            
            # Compute target Q-value using Double Q
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q
            
        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ============ Update Actor ============
        actor_loss_dict = self._update_actor(states, actions)
        
        # ============ Update Target Networks ============
        soft_update(self.critic_target, self.critic, self.tau)
        
        self.total_steps += 1
        
        return {
            "critic_loss": critic_loss.item(),
            **actor_loss_dict
        }
    
    def _update_actor(
        self, 
        states: torch.Tensor,
        expert_actions: torch.Tensor
    ) -> Dict[str, float]:
        """Update actor with combined diffusion and Q loss.
        
        Actor loss = BC loss (denoising) - alpha * Q-value
        
        Args:
            states: Batch of states
            expert_actions: Expert actions for BC loss
            
        Returns:
            Dictionary of actor losses
        """
        batch_size = states.shape[0]
        
        # ============ Diffusion (BC) Loss ============
        # Sample random timesteps
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(expert_actions)
        
        # Get noisy actions
        noisy_actions = self.diffusion.q_sample(expert_actions, t, noise)
        
        # Predict noise
        t_normalized = t.float() / self.num_diffusion_steps
        predicted_noise = self.actor(states, noisy_actions, t_normalized)
        
        # Denoising loss
        bc_loss = F.mse_loss(predicted_noise, noise)
        
        # ============ Q-Value Maximization Loss ============
        # Generate actions from the current policy
        with torch.no_grad():
            # Get clean actions by denoising
            generated_actions = self._sample_actions_batch(states)
            
        # Make actions differentiable by re-running last step with gradients
        # This is a common trick in diffusion RL
        t_last = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        noise_for_q = torch.randn_like(expert_actions)
        noisy_for_q = self.diffusion.q_sample(generated_actions.detach(), t_last, noise_for_q)
        
        t_normalized_q = t_last.float() / self.num_diffusion_steps
        pred_noise_q = self.actor(states, noisy_for_q, t_normalized_q)
        
        # Approximate clean action
        alpha_bar = self.diffusion.alphas_bar[t_last].view(-1, 1)
        actions_for_q = (noisy_for_q - torch.sqrt(1 - alpha_bar) * pred_noise_q) / torch.sqrt(alpha_bar)
        actions_for_q = torch.clamp(actions_for_q, -1.0, 1.0)
        
        # Q-value (we want to maximize, so negate for loss)
        q_value = self.critic.q1_forward(states, actions_for_q)
        q_loss = -q_value.mean()
        
        # Total actor loss
        actor_loss = self.bc_weight * bc_loss + self.alpha * q_loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return {
            "actor_loss": actor_loss.item(),
            "bc_loss": bc_loss.item(),
            "q_loss": q_loss.item(),
            "q_mean": q_value.mean().item()
        }
    
    def _sample_actions_batch(
        self, 
        states: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """Sample actions for a batch of states using diffusion.
        
        Args:
            states: Batch of states
            num_samples: Number of action samples per state
            
        Returns:
            Sampled actions
        """
        batch_size = states.shape[0]
        
        # Start from pure noise
        if num_samples > 1:
            states = states.repeat_interleave(num_samples, dim=0)
            
        x = torch.randn(states.shape[0], self.action_dim, device=self.device)
        
        # Reverse diffusion (fast version with fewer steps for training)
        steps = min(self.num_diffusion_steps, 5)  # Use fewer steps during training
        step_indices = list(range(0, self.num_diffusion_steps, self.num_diffusion_steps // steps))[::-1]
        
        for t in step_indices:
            t_batch = torch.full((states.shape[0],), t, device=self.device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            predicted_noise = self.actor(states, x, t_normalized)
            x = self.diffusion.p_sample(x, t_batch, predicted_noise, clip_denoised=True)
            
        x = torch.clamp(x, -1.0, 1.0)
        
        if num_samples > 1:
            # Reshape and average or select
            x = x.view(batch_size, num_samples, self.action_dim)
            x = x[:, 0]  # Take first sample
            
        return x
    
    @torch.no_grad()
    def sample_action(
        self, 
        state: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """Sample an action for a single state.
        
        Args:
            state: Current state
            deterministic: If True, use more diffusion steps
            
        Returns:
            Sampled action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        self.actor.eval()
        
        batch_size = state.shape[0]
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Full diffusion for inference
        steps = self.num_diffusion_steps if deterministic else 5
        step_indices = list(range(0, self.num_diffusion_steps, max(1, self.num_diffusion_steps // steps)))[::-1]
        
        for t in step_indices:
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            predicted_noise = self.actor(state, x, t_normalized)
            x = self.diffusion.p_sample(x, t_batch, predicted_noise, clip_denoised=True)
            
        x = torch.clamp(x, -1.0, 1.0)
        
        return x.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def get_q_values(
        self, 
        state: np.ndarray, 
        action: np.ndarray
    ) -> tuple:
        """Get Q-values for state-action pair.
        
        Args:
            state: State
            action: Action
            
        Returns:
            Tuple of (Q1, Q2, min_Q)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(self.device)
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        q1, q2 = self.critic(state, action)
        min_q = torch.min(q1, q2)
        
        return q1.item(), q2.item(), min_q.item()
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_steps": self.total_steps
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_steps = checkpoint["total_steps"]
