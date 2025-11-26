"""
Consistency Policy Q-Learning (CPQL) Agent

Combines Consistency Flow Matching with Q-Learning for efficient offline RL.
Uses a consistency flow model for few-step action generation while
maximizing Q-values.

This implementation is based on:
1. Consistency Flow (from our flow_matching module) for fast action generation
2. Double Q-Learning (like DDQL) for stable Q-value estimation

Key features:
- Flow matching with consistency training for few-step sampling
- Double Q-networks to reduce overestimation
- Combined BC + Q-maximization objective

Reference:
- CPQL: https://github.com/cccedric/cpql
- Diffusion-QL: https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL
- ManiFlow: https://github.com/allenai/maniflow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import copy

from ...common.networks import FlowVelocityPredictor, DoubleQNetwork
from ...common.utils import soft_update


class CPQLAgent:
    """Consistency Policy Q-Learning Agent.
    
    Combines Consistency Flow with Q-Learning:
    1. Flow Policy: Predicts velocity field for ODE-based action generation
    2. Consistency Training: Enforces consistency across timesteps for few-step sampling
    3. Q-Networks: Twin Q-networks for value estimation
    4. Training: Flow loss + Consistency loss + Q-maximization
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        learning_rate_policy: Learning rate for policy
        learning_rate_q: Learning rate for Q-network
        gamma: Discount factor
        tau: Soft update coefficient
        alpha: Temperature for Q-value weighting
        bc_weight: Weight for BC (flow matching) loss
        consistency_weight: Weight for consistency loss
        num_flow_steps: Number of ODE steps for sampling
        device: Device for computation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        learning_rate_policy: float = 3e-4,
        learning_rate_q: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.1,
        bc_weight: float = 1.0,
        consistency_weight: float = 1.0,
        num_flow_steps: int = 5,
        # Legacy parameters for compatibility (ignored)
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_noise_levels: int = 20,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.bc_weight = bc_weight
        self.consistency_weight = consistency_weight
        self.num_flow_steps = num_flow_steps
        self.device = device
        
        # Flow Policy: predicts velocity field v(x_t, t, s)
        self.policy = FlowVelocityPredictor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims + [256]
        ).to(device)
        
        # EMA policy for consistency training target
        self.policy_ema = copy.deepcopy(self.policy)
        for p in self.policy_ema.parameters():
            p.requires_grad = False
            
        # Q-Networks (Double Q)
        self.critic = DoubleQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        # Target Q-networks
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
            
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate_policy
        )
        self.q_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=learning_rate_q
        )
        
        self.total_steps = 0
        self.ema_decay = 0.999
        
    def _update_ema(self):
        """Update EMA policy."""
        with torch.no_grad():
            for ema_p, p in zip(
                self.policy_ema.parameters(),
                self.policy.parameters()
            ):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
    
    def _linear_interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation: x_t = (1-t)*x_0 + t*x_1"""
        t_expand = t.unsqueeze(-1) if t.dim() == 1 else t
        return (1 - t_expand) * x_0 + t_expand * x_1
    
    @torch.no_grad()
    def _sample_actions_batch(
        self,
        states: torch.Tensor,
        num_steps: Optional[int] = None,
        use_ema: bool = True
    ) -> torch.Tensor:
        """Sample actions using ODE integration.
        
        Args:
            states: Batch of states
            num_steps: Number of ODE steps (default: self.num_flow_steps)
            use_ema: Use EMA model
            
        Returns:
            Sampled actions
        """
        batch_size = states.shape[0]
        num_steps = num_steps or self.num_flow_steps
        
        net = self.policy_ema if use_ema else self.policy
        
        # Start from noise
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # ODE integration
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            v = net(states, x, t)
            x = x + v * dt
            
        return torch.clamp(x, -1.0, 1.0)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step.
        
        Updates Q-networks and policy with flow + consistency + Q objectives.
        
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
        batch_size = states.shape[0]
        
        # ============ Update Q-Networks ============
        with torch.no_grad():
            # Sample next actions from policy
            next_actions = self._sample_actions_batch(next_states)
            
            # Target Q-values (Double Q)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * target_q
            
        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Q-loss
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.q_optimizer.step()
        
        # ============ Update Policy ============
        policy_loss_dict = self._update_policy(states, actions)
        
        # ============ Update Targets ============
        soft_update(self.critic_target, self.critic, self.tau)
        self._update_ema()
        
        self.total_steps += 1
        
        return {
            "q_loss": q_loss.item(),
            **policy_loss_dict
        }
    
    def _update_policy(
        self, 
        states: torch.Tensor,
        expert_actions: torch.Tensor
    ) -> Dict[str, float]:
        """Update policy with flow + consistency + Q objectives.
        
        Loss = bc_weight * flow_loss + consistency_weight * cons_loss + alpha * (-Q)
        
        Args:
            states: Batch of states
            expert_actions: Expert actions for BC
            
        Returns:
            Dictionary of policy losses
        """
        batch_size = states.shape[0]
        
        # ============ Flow Matching Loss ============
        # Sample noise (starting point)
        x_0 = torch.randn_like(expert_actions)
        
        # Sample timestep uniformly
        t = torch.rand(batch_size, device=self.device)
        
        # Interpolate to get x_t
        x_t = self._linear_interpolate(x_0, expert_actions, t)
        
        # Target: velocity v = x_1 - x_0
        v_target = expert_actions - x_0
        
        # Predict velocity
        v_pred = self.policy(states, x_t, t)
        
        # Flow matching loss
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # ============ Consistency Loss ============
        # Sample t and delta_t for consistency
        t_cons = 0.05 + torch.rand(batch_size, device=self.device) * 0.9  # t in [0.05, 0.95]
        delta_t = torch.rand(batch_size, device=self.device) * (0.99 - t_cons)
        delta_t = torch.clamp(delta_t, min=0.02)
        t_next = torch.clamp(t_cons + delta_t, max=0.99)
        
        # Interpolate
        x_t_cons = self._linear_interpolate(x_0, expert_actions, t_cons)
        x_t_next = self._linear_interpolate(x_0, expert_actions, t_next)
        
        # Teacher prediction at t_next
        with torch.no_grad():
            v_teacher = self.policy_ema(states, x_t_next, t_next)
            # Estimate x_1 from teacher
            t_next_expand = t_next.unsqueeze(-1)
            pred_x1 = x_t_next + (1 - t_next_expand) * v_teacher
        
        # Target velocity at t_cons
        v_cons_target = pred_x1 - x_0
        
        # Student prediction at t_cons
        v_cons_pred = self.policy(states, x_t_cons, t_cons)
        
        # Consistency loss
        consistency_loss = F.mse_loss(v_cons_pred, v_cons_target)
        
        # ============ Q-Value Maximization ============
        # Generate actions and maximize Q
        with torch.no_grad():
            generated_actions = self._sample_actions_batch(states, use_ema=False)
        
        # Re-sample with gradients for last step
        x_for_q = torch.randn(batch_size, self.action_dim, device=self.device)
        dt = 1.0 / self.num_flow_steps
        
        # Run ODE without gradients until last step
        with torch.no_grad():
            for i in range(self.num_flow_steps - 1):
                t_step = torch.full((batch_size,), i * dt, device=self.device)
                v = self.policy(states, x_for_q, t_step)
                x_for_q = x_for_q + v * dt
        
        # Last step with gradients
        t_last = torch.full((batch_size,), (self.num_flow_steps - 1) * dt, device=self.device)
        v_last = self.policy(states, x_for_q, t_last)
        actions_for_q = x_for_q + v_last * dt
        actions_for_q = torch.clamp(actions_for_q, -1.0, 1.0)
        
        # Q-value (maximize, so negate for loss)
        q_value = self.critic.q1_forward(states, actions_for_q)
        q_loss = -q_value.mean()
        
        # ============ Total Loss ============
        total_loss = (
            self.bc_weight * flow_loss +
            self.consistency_weight * consistency_loss +
            self.alpha * q_loss
        )
        
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        return {
            "policy_loss": total_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "bc_loss": flow_loss.item(),
            "q_policy_loss": q_loss.item(),
            "q_mean": q_value.mean().item()
        }
    
    @torch.no_grad()
    def sample_action(
        self, 
        state: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """Sample action using ODE integration.
        
        Args:
            state: Current state
            deterministic: Use EMA model if True
            
        Returns:
            Sampled action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        batch_size = state.shape[0]
        net = self.policy_ema if deterministic else self.policy
        
        # Start from noise
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # ODE integration
        dt = 1.0 / self.num_flow_steps
        for i in range(self.num_flow_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            v = net(state, x, t)
            x = x + v * dt
            
        action = torch.clamp(x, -1.0, 1.0)
        
        return action.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def get_q_values(
        self, 
        state: np.ndarray, 
        action: np.ndarray
    ) -> tuple:
        """Get Q-values for state-action pair."""
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
            "policy": self.policy.state_dict(),
            "policy_ema": self.policy_ema.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "total_steps": self.total_steps
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy_ema.load_state_dict(checkpoint["policy_ema"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.total_steps = checkpoint["total_steps"]
