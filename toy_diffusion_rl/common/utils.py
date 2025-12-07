"""
Utility Functions for Toy Diffusion RL

Contains helper functions for:
- Random seed setting
- Network parameter updates
- Device management
- Noise schedules for diffusion
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union

# Try to import diffusers for official DDPM scheduler
try:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


def set_seed(seed: int):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_device(device: Optional[str] = None) -> torch.device:
    """Get the device to use for computation.
    
    Args:
        device: Device string ('cpu', 'cuda', 'cuda:0', etc.)
                If None, automatically selects CUDA if available.
                
    Returns:
        torch.device object
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Soft update target network parameters.
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    
    Args:
        target: Target network
        source: Source network
        tau: Interpolation factor (0 < tau <= 1)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update(target: nn.Module, source: nn.Module):
    """Hard update target network parameters.
    
    θ_target = θ_source
    
    Args:
        target: Target network
        source: Source network
    """
    target.load_state_dict(source.state_dict())


# ============== Diffusion Noise Schedules ==============

def linear_schedule(
    t: torch.Tensor,
    beta_start: float = 0.0001,
    beta_end: float = 0.02
) -> torch.Tensor:
    """Linear noise schedule for diffusion.
    
    Args:
        t: Timestep in [0, 1]
        beta_start: Starting noise level
        beta_end: Ending noise level
        
    Returns:
        Beta value at timestep t
    """
    return beta_start + t * (beta_end - beta_start)


def cosine_schedule(
    t: torch.Tensor,
    s: float = 0.008
) -> torch.Tensor:
    """Cosine noise schedule (improved diffusion schedule).
    
    From "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
    
    Args:
        t: Timestep in [0, 1]
        s: Small offset to prevent singularity
        
    Returns:
        Alpha_bar value at timestep t
    """
    return torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2


def get_alpha_beta(
    timesteps: torch.Tensor,
    schedule: str = "linear",
    beta_start: float = 0.0001,
    beta_end: float = 0.02
):
    """Get alpha and beta values for diffusion.
    
    Args:
        timesteps: Discrete timesteps (0 to T-1)
        schedule: 'linear' or 'cosine'
        beta_start: Starting beta for linear schedule
        beta_end: Ending beta for linear schedule
        
    Returns:
        Dictionary with alpha, beta, alpha_bar, etc.
    """
    T = timesteps.shape[0]
    t_normalized = timesteps.float() / T
    
    if schedule == "linear":
        betas = linear_schedule(t_normalized, beta_start, beta_end)
    elif schedule == "cosine":
        alphas_bar = cosine_schedule(t_normalized)
        alphas_bar_prev = torch.cat([torch.tensor([1.0]), alphas_bar[:-1]])
        betas = 1 - alphas_bar / alphas_bar_prev
        betas = torch.clamp(betas, min=1e-8, max=0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar_prev = torch.cat([torch.tensor([1.0]), alphas_bar[:-1]])
    
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_bar": alphas_bar,
        "alphas_bar_prev": alphas_bar_prev,
        "sqrt_alphas_bar": torch.sqrt(alphas_bar),
        "sqrt_one_minus_alphas_bar": torch.sqrt(1 - alphas_bar)
    }


class DiffusionHelper:
    """Helper class for diffusion forward and reverse processes.
    
    Args:
        num_diffusion_steps: Number of diffusion steps
        schedule: Noise schedule type ('linear' or 'cosine')
        beta_start: Starting beta for linear schedule
        beta_end: Ending beta for linear schedule
        device: Device for tensors
    """
    
    def __init__(
        self,
        num_diffusion_steps: int = 100,
        schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu"
    ):
        self.num_steps = num_diffusion_steps
        self.device = device
        
        timesteps = torch.arange(num_diffusion_steps)
        schedule_dict = get_alpha_beta(timesteps, schedule, beta_start, beta_end)
        
        # Store schedule values on device
        for key, value in schedule_dict.items():
            setattr(self, key, value.to(device))
            
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion: add noise to data.
        
        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x_0: Clean data (can be 2D [B, D] or 3D [B, T, D] for action sequences)
            t: Timesteps (batch of integers)
            noise: Pre-sampled noise (optional)
            
        Returns:
            Noisy samples x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Handle both 2D and 3D tensors (action sequences)
        if x_0.dim() == 3:
            # [B, T, D] - action sequence
            sqrt_alpha_bar = self.sqrt_alphas_bar[t].view(-1, 1, 1)
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
        else:
            # [B, D] - single action
            sqrt_alpha_bar = self.sqrt_alphas_bar[t].view(-1, 1)
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[t].view(-1, 1)
        
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """Reverse diffusion: denoise one step.
        
        Args:
            x_t: Noisy samples (can be 2D [B, D] or 3D [B, T, D])
            t: Current timesteps
            predicted_noise: Predicted noise from model
            clip_denoised: Whether to clip to [-1, 1]
            
        Returns:
            Denoised samples x_{t-1}
        """
        # Handle both 2D and 3D tensors (action sequences)
        if x_t.dim() == 3:
            view_shape = (-1, 1, 1)
        else:
            view_shape = (-1, 1)
        
        alpha = self.alphas[t].view(*view_shape)
        alpha_bar = self.alphas_bar[t].view(*view_shape)
        beta = self.betas[t].view(*view_shape)
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Compute mean for p(x_{t-1} | x_t)
        alpha_bar_prev = self.alphas_bar_prev[t].view(*view_shape)
        
        # Posterior mean
        coef1 = beta * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
        coef2 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
        mean = coef1 * x_0_pred + coef2 * x_t
        
        # Posterior variance
        posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        
        # Sample
        noise = torch.randn_like(x_t)
        # No noise at t=0
        nonzero_mask = (t > 0).float().view(*view_shape)
        
        return mean + nonzero_mask * torch.sqrt(posterior_variance) * noise
    
    def ddpm_sample(
        self,
        model,
        state: torch.Tensor,
        action_dim: int,
        clip_action: bool = True
    ) -> torch.Tensor:
        """Full DDPM sampling loop.
        
        Args:
            model: Noise prediction model
            state: Conditioning state
            action_dim: Dimension of action to sample
            clip_action: Whether to clip final action
            
        Returns:
            Sampled action
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Start from pure noise
        x = torch.randn(batch_size, action_dim, device=device)
        
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_steps
            
            predicted_noise = model(state, x, t_normalized)
            x = self.p_sample(x, t_batch, predicted_noise, clip_denoised=clip_action)
            
        return x


def extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape) -> torch.Tensor:
    """Extract values from 1D array at indices and reshape for broadcasting.
    
    Args:
        arr: 1D tensor of values
        timesteps: Tensor of indices
        broadcast_shape: Target shape for broadcasting
        
    Returns:
        Extracted values reshaped for broadcasting
    """
    res = arr[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# ============== Diffusers-based DDPM Scheduler Wrapper ==============

class DDPMSchedulerHelper:
    """DDPM Scheduler wrapper using HuggingFace diffusers.
    
    Uses the official DDPMScheduler with 'squaredcos_cap_v2' beta schedule,
    which is critical for diffusion policy performance according to ManiSkill.
    
    Reference:
        https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/diffusion_policy/train.py
        "beta_schedule='squaredcos_cap_v2' has big impact on performance, try not to change"
    
    Args:
        num_diffusion_steps: Number of diffusion steps (default: 100)
        beta_schedule: Beta schedule type ('squaredcos_cap_v2', 'linear', 'cosine')
        clip_sample: Whether to clip denoised samples to [-1, 1]
        prediction_type: 'epsilon' (predict noise) or 'sample' (predict x0)
        device: Device for tensors
    """
    
    def __init__(
        self,
        num_diffusion_steps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        device: str = "cpu",
    ):
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers package not available. Install with: pip install diffusers"
            )
        
        self.num_steps = num_diffusion_steps
        self.device = device
        self.prediction_type = prediction_type
        
        # Create DDPM scheduler with official config
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
        )
        
        # Pre-compute and cache alpha values on device for training
        self._cache_alpha_values()
    
    def _cache_alpha_values(self):
        """Cache alpha values on device for efficient training."""
        # Get alphas from scheduler
        alphas_cumprod = self.scheduler.alphas_cumprod
        
        self.sqrt_alphas_bar = torch.sqrt(alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_cumprod).to(self.device)
        
        # Also cache for p_sample compatibility
        self.alphas = self.scheduler.alphas.to(self.device)
        self.betas = self.scheduler.betas.to(self.device)
        self.alphas_bar = alphas_cumprod.to(self.device)
        self.alphas_bar_prev = torch.cat([
            torch.tensor([1.0], device=self.device), 
            self.alphas_bar[:-1]
        ])
    
    def to(self, device: Union[str, torch.device]) -> 'DDPMSchedulerHelper':
        """Move scheduler to device."""
        self.device = device if isinstance(device, str) else str(device)
        self._cache_alpha_values()
        return self
    
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion: add noise to data.
        
        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x_0: Clean data (can be 2D [B, D] or 3D [B, T, D] for action sequences)
            t: Timesteps (batch of integers)
            noise: Pre-sampled noise (optional)
            
        Returns:
            Noisy samples x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Handle both 2D and 3D tensors (action sequences)
        if x_0.dim() == 3:
            # [B, T, D] - action sequence
            sqrt_alpha_bar = self.sqrt_alphas_bar[t].view(-1, 1, 1)
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
        else:
            # [B, D] - single action
            sqrt_alpha_bar = self.sqrt_alphas_bar[t].view(-1, 1)
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[t].view(-1, 1)
        
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """Reverse diffusion: denoise one step using scheduler.step().
        
        Args:
            x_t: Noisy samples (can be 2D [B, D] or 3D [B, T, D])
            t: Current timesteps (integer tensor, same value for all samples)
            predicted_noise: Predicted noise from model
            clip_denoised: Whether to clip to [-1, 1]
            
        Returns:
            Denoised samples x_{t-1}
        """
        # Use scheduler.step for proper DDPM update
        # Note: diffusers scheduler expects scalar timestep for step()
        # but we handle batched timesteps here
        
        # For batched operation, we use manual computation
        if x_t.dim() == 3:
            view_shape = (-1, 1, 1)
        else:
            view_shape = (-1, 1)
        
        alpha = self.alphas[t].view(*view_shape)
        alpha_bar = self.alphas_bar[t].view(*view_shape)
        beta = self.betas[t].view(*view_shape)
        alpha_bar_prev = self.alphas_bar_prev[t].view(*view_shape)
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Posterior mean
        coef1 = beta * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
        coef2 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
        mean = coef1 * x_0_pred + coef2 * x_t
        
        # Posterior variance
        posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        
        # Sample
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(*view_shape)
        
        return mean + nonzero_mask * torch.sqrt(posterior_variance) * noise
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step using diffusers scheduler.
        
        This uses the official scheduler.step() for inference.
        
        Args:
            model_output: Predicted noise from model
            timestep: Current timestep (scalar)
            sample: Current noisy sample
            
        Returns:
            Denoised sample for next step
        """
        result = self.scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
        )
        return result.prev_sample
    
    @property
    def timesteps(self):
        """Get timesteps for inference."""
        return self.scheduler.timesteps


def create_diffusion_helper(
    num_diffusion_steps: int = 100,
    schedule: str = "squaredcos_cap_v2",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    device: str = "cpu",
    use_diffusers: bool = True,
) -> Union[DiffusionHelper, 'DDPMSchedulerHelper']:
    """Factory function to create diffusion helper.
    
    Prefers DDPMSchedulerHelper with 'squaredcos_cap_v2' schedule when diffusers
    is available, as this has significant performance impact.
    
    Args:
        num_diffusion_steps: Number of diffusion steps
        schedule: Schedule type. For diffusers: 'squaredcos_cap_v2', 'linear', 'cosine'.
                 For legacy: 'linear', 'cosine'.
        beta_start: Starting beta for legacy linear schedule
        beta_end: Ending beta for legacy linear schedule
        device: Device for tensors
        use_diffusers: Whether to use diffusers scheduler (recommended)
        
    Returns:
        DiffusionHelper or DDPMSchedulerHelper instance
    """
    if use_diffusers and DIFFUSERS_AVAILABLE:
        # Map legacy schedule names to diffusers names
        schedule_map = {
            "linear": "linear",
            "cosine": "cosine",
            "squaredcos_cap_v2": "squaredcos_cap_v2",
        }
        beta_schedule = schedule_map.get(schedule, "squaredcos_cap_v2")
        
        return DDPMSchedulerHelper(
            num_diffusion_steps=num_diffusion_steps,
            beta_schedule=beta_schedule,
            clip_sample=True,
            prediction_type="epsilon",
            device=device,
        )
    else:
        # Fallback to legacy implementation
        return DiffusionHelper(
            num_diffusion_steps=num_diffusion_steps,
            schedule=schedule if schedule in ["linear", "cosine"] else "linear",
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
        )
