"""
Diffusion Policy Agent

Implementation of Diffusion Policy for behavior cloning / offline RL.
Based on: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
Reference: https://github.com/real-stanford/diffusion_policy

This is a simplified version for low-dimensional continuous control tasks.
The policy learns to denoise actions given states using DDPM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

from ...common.networks import DiffusionNoisePredictor
from ...common.utils import DiffusionHelper


class DiffusionPolicyAgent:
    """Diffusion Policy Agent for behavior cloning.
    
    Uses Denoising Diffusion Probabilistic Models (DDPM) to model
    the action distribution conditioned on state. Trained with
    denoising score matching objective.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions for noise predictor
        num_diffusion_steps: Number of diffusion steps
        noise_schedule: Type of noise schedule ('linear' or 'cosine')
        learning_rate: Learning rate for optimizer
        device: Device for computation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256, 256],
        num_diffusion_steps: int = 100,
        noise_schedule: str = "linear",
        learning_rate: float = 1e-4,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device
        
        # Noise prediction network
        self.noise_predictor = DiffusionNoisePredictor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        # Diffusion helper for forward/reverse process
        self.diffusion = DiffusionHelper(
            num_diffusion_steps=num_diffusion_steps,
            schedule=noise_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.noise_predictor.parameters(),
            lr=learning_rate
        )
        
        # For EMA (optional but recommended)
        self.ema_noise_predictor = None
        self.ema_decay = 0.999
        
    def _init_ema(self):
        """Initialize EMA model."""
        import copy
        self.ema_noise_predictor = copy.deepcopy(self.noise_predictor)
        self.ema_noise_predictor.eval()
        
    def _update_ema(self):
        """Update EMA parameters."""
        if self.ema_noise_predictor is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_noise_predictor.parameters(),
                self.noise_predictor.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1 - self.ema_decay
                )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step.
        
        Trains the noise predictor using denoising score matching:
        L = E_{t, x_0, epsilon}[||epsilon - epsilon_theta(x_t, t, s)||^2]
        
        Args:
            batch: Dictionary with 'states' and 'actions' tensors
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"]
        actions = batch["actions"]  # Clean expert actions (x_0)
        batch_size = states.shape[0]
        
        # Sample random timesteps
        t = torch.randint(
            0, self.num_diffusion_steps, (batch_size,), 
            device=self.device
        )
        
        # Sample noise
        noise = torch.randn_like(actions)
        
        # Get noisy actions (forward diffusion)
        noisy_actions = self.diffusion.q_sample(actions, t, noise)
        
        # Normalize timestep to [0, 1]
        t_normalized = t.float() / self.num_diffusion_steps
        
        # Predict noise
        predicted_noise = self.noise_predictor(states, noisy_actions, t_normalized)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.noise_predictor.parameters(), 1.0)
        self.optimizer.step()
        
        # Update EMA
        self._update_ema()
        
        return {"loss": loss.item()}
    
    @torch.no_grad()
    def sample_action(
        self, 
        state: np.ndarray,
        use_ema: bool = True,
        num_inference_steps: Optional[int] = None
    ) -> np.ndarray:
        """Sample an action given a state.
        
        Uses the reverse diffusion process (DDPM sampling) to generate
        an action conditioned on the current state.
        
        Args:
            state: Current state
            use_ema: Whether to use EMA model for sampling
            num_inference_steps: Number of inference steps (None = full)
            
        Returns:
            Sampled action
        """
        # Prepare state tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        batch_size = state.shape[0]
        
        # Choose model
        model = self.ema_noise_predictor if (use_ema and self.ema_noise_predictor) else self.noise_predictor
        model.eval()
        
        # Start from pure noise
        action = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Determine steps
        steps = num_inference_steps or self.num_diffusion_steps
        step_indices = list(range(0, self.num_diffusion_steps, self.num_diffusion_steps // steps))[::-1]
        
        # Reverse diffusion
        for t in step_indices:
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            predicted_noise = model(state, action, t_normalized)
            action = self.diffusion.p_sample(action, t_batch, predicted_noise, clip_denoised=True)
        
        # Clip to action bounds
        action = torch.clamp(action, -1.0, 1.0)
        
        return action.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def sample_action_ddim(
        self,
        state: np.ndarray,
        num_inference_steps: int = 10,
        eta: float = 0.0
    ) -> np.ndarray:
        """Sample action using DDIM (faster sampling).
        
        DDIM allows fewer steps with deterministic sampling (eta=0)
        or stochastic sampling (eta=1).
        
        Args:
            state: Current state
            num_inference_steps: Number of DDIM steps
            eta: Stochasticity parameter (0 = deterministic)
            
        Returns:
            Sampled action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        batch_size = state.shape[0]
        model = self.ema_noise_predictor if self.ema_noise_predictor else self.noise_predictor
        model.eval()
        
        # DDIM sampling schedule
        step_size = self.num_diffusion_steps // num_inference_steps
        timesteps = list(range(0, self.num_diffusion_steps, step_size))[::-1]
        
        action = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            # Predict noise
            predicted_noise = model(state, action, t_normalized)
            
            # Get alpha values
            alpha_bar = self.diffusion.alphas_bar[t]
            alpha_bar_prev = self.diffusion.alphas_bar[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)
            
            # Predict x_0
            pred_x0 = (action - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Direction pointing to x_t
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            
            # Compute x_{t-1}
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * predicted_noise
            noise = sigma * torch.randn_like(action) if i < len(timesteps) - 1 else 0
            
            action = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + noise
        
        action = torch.clamp(action, -1.0, 1.0)
        return action.cpu().numpy().squeeze()
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "noise_predictor": self.noise_predictor.state_dict(),
            "ema_noise_predictor": self.ema_noise_predictor.state_dict() if self.ema_noise_predictor else None,
            "optimizer": self.optimizer.state_dict(),
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.noise_predictor.load_state_dict(checkpoint["noise_predictor"])
        if checkpoint["ema_noise_predictor"]:
            self._init_ema()
            self.ema_noise_predictor.load_state_dict(checkpoint["ema_noise_predictor"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
