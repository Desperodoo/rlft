"""
Consistency Policy Q-Learning (CPQL) Agent

Combines Consistency Models with Q-Learning for efficient offline RL.
Uses a consistency model for single-step action generation while
maximizing Q-values.

Key features:
- Karras noise schedule (sigma_max=80, sigma_min=0.002, rho=7)
- Euler solver for ODE integration in consistency training
- EMA target model for stable consistency loss

Reference:
- CPQL: https://github.com/cccedric/cpql
- Consistency Models (Song et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import copy

from ...common.networks import FlowVelocityPredictor, QNetwork
from ...common.utils import soft_update


class KarrasDenoiser(nn.Module):
    """Karras-style Denoiser for Consistency Models.
    
    Implements the parameterization from Karras et al. (EDM):
    D(x; sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(c_in(sigma) * x; c_noise(sigma))
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        sigma_data: Data standard deviation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256, 256],
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        
        # Base network (predicts the denoised output)
        self.net = FlowVelocityPredictor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        )
        
    def _get_scalings(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get EDM scalings with boundary condition for consistency models.
        
        Following the original CPQL implementation which uses boundary condition
        to ensure f(x, sigma_min) = x (identity at minimum noise level).
        
        Args:
            sigma: Noise level
            
        Returns:
            c_skip, c_out, c_in, c_noise
        """
        sigma_data = self.sigma_data
        
        # Boundary condition version (original CPQL)
        # This ensures the model outputs x when sigma = sigma_min
        c_skip = sigma_data ** 2 / ((sigma - self.sigma_min) ** 2 + sigma_data ** 2)
        c_out = (sigma - self.sigma_min) * sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        c_in = 1.0 / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        
        # Time embedding: normalize log(sigma) to [0, 1] range for the network
        # log(sigma) ranges from log(sigma_min) to log(sigma_max)
        # We normalize this to [0, 1]
        log_sigma = torch.log(sigma.clamp(min=1e-7))
        log_sigma_min = np.log(self.sigma_min)
        log_sigma_max = np.log(self.sigma_max)
        c_noise = (log_sigma - log_sigma_min) / (log_sigma_max - log_sigma_min)
        
        return c_skip.unsqueeze(-1), c_out.unsqueeze(-1), c_in.unsqueeze(-1), c_noise
    
    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of Karras denoiser.
        
        Args:
            x: Noisy sample
            sigma: Noise level
            state: Conditioning state
            
        Returns:
            Denoised sample
        """
        c_skip, c_out, c_in, c_noise = self._get_scalings(sigma)
        
        # Scale input
        x_scaled = c_in * x
        
        # Use c_noise as the time embedding
        F_theta = self.net(state, x_scaled, c_noise)
        
        # Apply skip connection and output scaling
        output = c_skip * x + c_out * F_theta
        
        return output


class ConsistencyPolicyNetwork(nn.Module):
    """Consistency Policy that maps noise directly to actions.
    
    Uses KarrasDenoiser as the backbone and implements consistency
    model parameterization.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256, 256],
        sigma_min: float = 0.002,
        sigma_max: float = 80.0
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # Use Karras denoiser as backbone
        self.denoiser = KarrasDenoiser(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )
        
    def forward(
        self,
        z: torch.Tensor,
        sigma: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of consistency model.
        
        Args:
            z: Noisy sample
            sigma: Noise level (not normalized timestep!)
            state: Conditioning state
            
        Returns:
            Predicted clean action
        """
        return self.denoiser(z, sigma, state)


class CPQLAgent:
    """Consistency Policy Q-Learning Agent.
    
    Key components:
    1. Consistency Policy with Karras noise schedule
    2. Q-Network: Value estimation
    3. Training combines consistency loss with Q-maximization
    4. Euler solver for ODE integration during training
    
    Karras schedule parameters:
    - sigma_max = 80.0
    - sigma_min = 0.002
    - rho = 7 (schedule curvature)
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        learning_rate_policy: Learning rate for policy
        learning_rate_q: Learning rate for Q-network
        gamma: Discount factor
        tau: Soft update coefficient
        alpha: Temperature for Q-value weighting
        consistency_weight: Weight for consistency loss
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        rho: Karras schedule curvature
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
        consistency_weight: float = 1.0,
        bc_weight: float = 1.0,
        num_noise_levels: int = 20,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.consistency_weight = consistency_weight
        self.bc_weight = bc_weight
        self.num_noise_levels = num_noise_levels
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.device = device
        
        # Consistency Policy with Karras denoiser
        self.policy = ConsistencyPolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims + [256],
            sigma_min=sigma_min,
            sigma_max=sigma_max
        ).to(device)
        
        # EMA policy for consistency training target
        self.policy_ema = copy.deepcopy(self.policy)
        for p in self.policy_ema.parameters():
            p.requires_grad = False
            
        # Q-Networks (using double Q)
        self.q1 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        self.q2 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        # Target Q-networks
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
            
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate_policy
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=learning_rate_q
        )
        
        # Setup Karras noise schedule
        self._setup_karras_schedule()
        
        self.total_steps = 0
        self.ema_decay = 0.995  # Following original CPQL
        self.step_start_ema = 1000  # Start EMA updates after this many steps
        self.update_ema_every = 5  # Update EMA every N steps
        
    def _setup_karras_schedule(self):
        """Setup Karras noise schedule.
        
        sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        
        This creates a schedule that is denser near sigma_min.
        """
        rho_inv = 1.0 / self.rho
        sigmas = []
        
        for i in range(self.num_noise_levels):
            sigma = (
                self.sigma_max ** rho_inv + 
                i / (self.num_noise_levels - 1) * 
                (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
            ) ** self.rho
            sigmas.append(sigma)
            
        self.sigmas = torch.tensor(sigmas, device=self.device, dtype=torch.float32)
        
    def _add_noise(
        self, 
        action: torch.Tensor, 
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """Add Gaussian noise to actions with given sigma.
        
        Args:
            action: Clean action
            sigma: Noise level
            
        Returns:
            Noisy action: action + sigma * noise
        """
        noise = torch.randn_like(action)
        return action + sigma.unsqueeze(-1) * noise
    
    def _euler_solver_step(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        state: torch.Tensor,
        policy: nn.Module
    ) -> torch.Tensor:
        """Single Euler solver step for ODE.
        
        dx/dsigma = (x - D(x, sigma)) / sigma
        x_next = x + (sigma_next - sigma) * dx/dsigma
        
        Args:
            x: Current noisy sample
            sigma: Current noise level
            sigma_next: Next noise level
            state: Conditioning state
            policy: Policy network to use
            
        Returns:
            Sample at next noise level
        """
        # Compute denoised estimate
        denoised = policy(x, sigma, state)
        
        # Compute derivative
        d = (x - denoised) / sigma.unsqueeze(-1)
        
        # Euler step
        dt = sigma_next - sigma
        x_next = x + dt.unsqueeze(-1) * d
        
        return x_next
    
    def _update_ema(self):
        """Update EMA policy with controlled timing.
        
        Following original CPQL: only update after step_start_ema steps
        and update every update_ema_every steps.
        """
        if self.total_steps < self.step_start_ema:
            return
            
        if self.total_steps % self.update_ema_every != 0:
            return
            
        with torch.no_grad():
            for ema_p, p in zip(
                self.policy_ema.parameters(),
                self.policy.parameters()
            ):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step.
        
        Updates Q-networks and policy with consistency + Q-value objectives.
        
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
            # Sample next actions from policy (single step at sigma_max)
            z = torch.randn(batch_size, self.action_dim, device=self.device) * self.sigma_max
            sigma = torch.full((batch_size,), self.sigma_max, device=self.device)
            next_actions = self.policy_ema(z, sigma, next_states)
            next_actions = torch.clamp(next_actions, -1.0, 1.0)
            
            # Target Q-values
            target_q1 = self.q1_target(next_states, next_actions)
            target_q2 = self.q2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q
            
        # Current Q-values
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        
        # Q-loss
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
        )
        self.q_optimizer.step()
        
        # ============ Update Policy ============
        policy_loss_dict = self._update_policy(states, actions)
        
        # ============ Update Targets ============
        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)
        self._update_ema()
        
        self.total_steps += 1
        
        return {
            "q_loss": q_loss.item(),
            **policy_loss_dict
        }
    
    def _consistency_and_recon_loss(
        self,
        states: torch.Tensor,
        expert_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute consistency loss and reconstruction loss.
        
        Following original CPQL implementation (karras_diffusion.py L93-154):
        - Sample noise and noise level indices
        - Add noise to expert actions to get x_t
        - Compute denoised output from x_t at sigma_t
        - Use Euler solver with expert_actions as the denoiser to get x_t2
        - Compute consistency loss between denoise(x_t) and denoise_target(x_t2)
        - Compute recon loss as MSE between denoise(x_t) and expert_actions
        
        Args:
            states: Batch of states
            expert_actions: Expert actions (x_start)
            
        Returns:
            Tuple of (consistency_loss, recon_loss)
        """
        batch_size = states.shape[0]
        noise = torch.randn_like(expert_actions)
        
        # Sample noise level indices using Karras schedule
        # t = sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho))
        # t = t^rho
        indices = torch.randint(0, self.num_noise_levels - 1, (batch_size,), device=self.device)
        
        # Compute sigma values for current and next step
        rho_inv = 1.0 / self.rho
        t = (
            self.sigma_max ** rho_inv + 
            indices.float() / (self.num_noise_levels - 1) * 
            (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
        ) ** self.rho
        
        t2 = (
            self.sigma_max ** rho_inv + 
            (indices.float() + 1) / (self.num_noise_levels - 1) * 
            (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
        ) ** self.rho
        
        # Add noise at current level: x_t = x_start + t * noise
        x_t = expert_actions + t.unsqueeze(-1) * noise
        
        # Denoised prediction at current noise level (for training)
        pred_from_t = self.policy(x_t, t, states)
        
        # Reconstruction loss: predict clean action from noisy sample
        recon_loss = ((pred_from_t - expert_actions) ** 2).mean(dim=-1)
        
        # Euler solver step: use expert_actions as the denoiser estimate
        # This is the key difference from our previous implementation!
        # x_t2 = x_t + (t2 - t) * (x_t - x_start) / t
        with torch.no_grad():
            d = (x_t - expert_actions) / t.unsqueeze(-1)
            x_t2 = x_t + (t2 - t).unsqueeze(-1) * d
            x_t2 = x_t2.detach()
            
            # Target model prediction at next noise level
            target_from_t2 = self.policy_ema(x_t2, t2, states)
            target_from_t2 = target_from_t2.detach()
        
        # Consistency loss: predictions at adjacent noise levels should match
        consistency_loss = ((pred_from_t - target_from_t2) ** 2).mean(dim=-1)
        
        # Apply Karras weighting: snr + 1/sigma_data^2
        snr = t ** (-2)
        weights = snr + 1.0 / (self.policy.denoiser.sigma_data ** 2)
        
        # Weight the losses
        weighted_consistency = (consistency_loss * weights).mean()
        weighted_recon = (recon_loss * weights).mean()
        
        return weighted_consistency, weighted_recon
    
    def _consistency_loss(
        self,
        states: torch.Tensor,
        expert_actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute consistency loss using Euler solver (legacy, for compatibility)."""
        consistency_loss, _ = self._consistency_and_recon_loss(states, expert_actions)
        return consistency_loss
    
    def _update_policy(
        self, 
        states: torch.Tensor,
        expert_actions: torch.Tensor
    ) -> Dict[str, float]:
        """Update policy with consistency and Q-value objectives.
        
        Loss = consistency_loss + bc_loss - alpha * Q_value
        
        Args:
            states: Batch of states
            expert_actions: Expert actions for BC
            
        Returns:
            Dictionary of policy losses
        """
        batch_size = states.shape[0]
        
        # ============ Consistency Loss + Reconstruction Loss ============
        # Following original CPQL: sample noise level, add noise to expert actions,
        # then compute both consistency loss (adjacent levels match) and 
        # recon_loss (predict clean action from noisy)
        consistency_loss, recon_loss = self._consistency_and_recon_loss(states, expert_actions)
        
        # BC loss is the reconstruction loss (denoising from random noise levels)
        bc_loss = recon_loss
        
        # ============ Q-Value Maximization ============
        # Generate actions and maximize Q
        z_for_q = torch.randn(batch_size, self.action_dim, device=self.device) * self.sigma_max
        sigma_for_q = torch.full((batch_size,), self.sigma_max, device=self.device)
        generated_actions = self.policy(z_for_q, sigma_for_q, states)
        generated_actions = torch.clamp(generated_actions, -1.0, 1.0)
        
        # Q-values of generated actions
        q1 = self.q1(states, generated_actions)
        q2 = self.q2(states, generated_actions)
        q_value = torch.min(q1, q2)
        
        q_loss = -q_value.mean()
        
        # Total loss
        total_loss = (
            self.consistency_weight * consistency_loss +
            self.bc_weight * bc_loss +
            self.alpha * q_loss
        )
        
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        return {
            "policy_loss": total_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "bc_loss": bc_loss.item(),
            "q_policy_loss": q_loss.item(),
            "q_mean": q_value.mean().item()
        }
    
    @torch.no_grad()
    def sample_action(
        self, 
        state: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """Sample action using single-step consistency generation.
        
        Following original CPQL implementation (sample_onestep):
        - Start from x_T ~ N(0, sigma_max^2)
        - Use sigmas[0] as the noise level for denoising
        
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
        
        # Sample noise at sigma_max (x_T ~ N(0, sigma_max^2))
        x_T = torch.randn(batch_size, self.action_dim, device=self.device) * self.sigma_max
        
        # Use sigmas[0] as the noise level (following original CPQL)
        sigma = torch.full((batch_size,), self.sigmas[0].item(), device=self.device)
        
        # Single-step generation
        if deterministic:
            action = self.policy_ema(x_T, sigma, state)
        else:
            action = self.policy(x_T, sigma, state)
            
        action = torch.clamp(action, -1.0, 1.0)
        
        return action.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def sample_action_multistep(
        self,
        state: np.ndarray,
        num_steps: int = 4
    ) -> np.ndarray:
        """Sample action using multi-step generation with Euler solver.
        
        Uses a subset of the noise schedule for multi-step denoising.
        
        Args:
            state: Current state
            num_steps: Number of denoising steps
            
        Returns:
            Sampled action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        batch_size = state.shape[0]
        
        # Get subset of sigmas for multi-step
        step_indices = np.linspace(0, self.num_noise_levels - 1, num_steps + 1).astype(int)
        
        # Start from noise at sigma_max
        x = torch.randn(batch_size, self.action_dim, device=self.device) * self.sigma_max
        
        for i in range(num_steps):
            sigma = self.sigmas[step_indices[i]].expand(batch_size)
            sigma_next = self.sigmas[step_indices[i + 1]].expand(batch_size)
            
            x = self._euler_solver_step(x, sigma, sigma_next, state, self.policy_ema)
            
        action = torch.clamp(x, -1.0, 1.0)
        
        return action.cpu().numpy().squeeze()
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "policy": self.policy.state_dict(),
            "policy_ema": self.policy_ema.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "rho": self.rho
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy_ema.load_state_dict(checkpoint["policy_ema"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint["q1_target"])
        self.q2_target.load_state_dict(checkpoint["q2_target"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.total_steps = checkpoint["total_steps"]
        # Load Karras parameters if saved
        if "sigma_min" in checkpoint:
            self.sigma_min = checkpoint["sigma_min"]
            self.sigma_max = checkpoint["sigma_max"]
            self.rho = checkpoint["rho"]
            self._setup_karras_schedule()
