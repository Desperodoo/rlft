"""
Diffusion Policy Agent - Unified Version

Implementation of Diffusion Policy for behavior cloning / offline RL.
Based on: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
Reference: https://github.com/real-stanford/diffusion_policy

This unified version supports multiple observation modes:
- "state": State vector only
- "image": Image observation only  
- "state_image": Both state and image (multimodal)

The policy learns to denoise actions given observations using DDPM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
import copy

try:
    from ...common.networks import MLP, TimestepEmbedding, MultiModalNoisePredictor
    from ...common.utils import DiffusionHelper
    from ...common.obs_encoder import ObservationEncoder, create_obs_encoder
except ImportError:
    try:
        from toy_diffusion_rl.common.networks import MLP, TimestepEmbedding, MultiModalNoisePredictor
        from toy_diffusion_rl.common.utils import DiffusionHelper
        from toy_diffusion_rl.common.obs_encoder import ObservationEncoder, create_obs_encoder
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from common.networks import MLP, TimestepEmbedding, MultiModalNoisePredictor
        from common.utils import DiffusionHelper
        from common.obs_encoder import ObservationEncoder, create_obs_encoder


class DiffusionPolicyAgent:
    """Unified Diffusion Policy Agent for behavior cloning.
    
    Uses DDPM to model the action distribution conditioned on observations.
    Supports multiple observation modalities through configurable encoder.
    
    Args:
        action_dim: Dimension of action space
        obs_mode: Observation mode ("state", "image", or "state_image")
        state_dim: Dimension of state space (required for state/state_image)
        image_shape: Image shape as (H, W, C) (required for image/state_image)
        hidden_dims: Hidden layer dimensions for noise predictor
        num_diffusion_steps: Number of diffusion steps
        noise_schedule: Type of noise schedule ('linear' or 'cosine')
        vision_encoder_type: Type of vision encoder ("cnn" or "dinov2")
        vision_output_dim: Output dimension of vision encoder
        freeze_vision_encoder: Whether to freeze vision encoder weights
        learning_rate: Learning rate for optimizer
        beta_start: Starting beta for noise schedule
        beta_end: Ending beta for noise schedule
        device: Device for computation
    """
    
    def __init__(
        self,
        action_dim: int,
        obs_mode: str = "state",
        state_dim: Optional[int] = None,
        image_shape: Optional[Tuple[int, int, int]] = None,
        hidden_dims: List[int] = [256, 256, 256],
        num_diffusion_steps: int = 100,
        noise_schedule: str = "linear",
        vision_encoder_type: str = "cnn",
        vision_output_dim: int = 128,
        freeze_vision_encoder: bool = False,
        learning_rate: float = 1e-4,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu"
    ):
        self.action_dim = action_dim
        self.obs_mode = obs_mode
        self.state_dim = state_dim
        self.image_shape = image_shape
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device
        
        # Create observation encoder
        self.obs_encoder = create_obs_encoder(
            obs_mode=obs_mode,
            state_dim=state_dim,
            image_shape=image_shape,
            vision_encoder_type=vision_encoder_type,
            vision_output_dim=vision_output_dim,
            freeze_vision=freeze_vision_encoder,
        ).to(device)
        
        # Noise prediction network
        self.noise_predictor = MultiModalNoisePredictor(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            obs_encoder=self.obs_encoder,
        ).to(device)
        
        # Diffusion helper
        self.diffusion = DiffusionHelper(
            num_diffusion_steps=num_diffusion_steps,
            schedule=noise_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device
        )
        
        # Collect trainable parameters
        trainable_params = list(self.noise_predictor.parameters())
        
        # Add vision encoder params if trainable
        if obs_mode in ["image", "state_image"] and not freeze_vision_encoder:
            vision_params = [p for p in self.obs_encoder.vision_encoder.parameters() if p.requires_grad]
            trainable_params.extend(vision_params)
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        
        # EMA model
        self.ema_noise_predictor = None
        self.ema_decay = 0.999
    
    def _init_ema(self):
        """Initialize EMA model."""
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
    
    def _parse_observation(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray], torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Parse observation into state and image tensors."""
        state = None
        image = None
        
        if self.obs_mode == "state":
            if isinstance(obs, dict):
                obs = obs.get("state", obs.get("obs", obs))
            if isinstance(obs, np.ndarray):
                state = torch.FloatTensor(obs).to(self.device)
            else:
                state = obs.to(self.device) if isinstance(obs, torch.Tensor) else torch.FloatTensor(obs).to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
        elif self.obs_mode == "image":
            if isinstance(obs, dict):
                obs = obs.get("image", obs)
            if isinstance(obs, np.ndarray):
                if obs.ndim == 3 and obs.shape[-1] in [1, 3, 4]:
                    obs = np.transpose(obs, (2, 0, 1))
                image = torch.FloatTensor(obs).to(self.device)
            else:
                image = obs.to(self.device) if isinstance(obs, torch.Tensor) else torch.FloatTensor(obs).to(self.device)
            if image.max() > 1.0:
                image = image / 255.0
            if image.dim() == 3:
                image = image.unsqueeze(0)
                
        else:  # state_image
            if isinstance(obs, dict):
                state_data = obs.get("state", obs.get("obs"))
                image_data = obs.get("image")
            else:
                raise ValueError("For state_image mode, obs must be a dict")
            
            if isinstance(state_data, np.ndarray):
                state = torch.FloatTensor(state_data).to(self.device)
            else:
                state = state_data.to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            if isinstance(image_data, np.ndarray):
                if image_data.ndim == 3 and image_data.shape[-1] in [1, 3, 4]:
                    image_data = np.transpose(image_data, (2, 0, 1))
                image = torch.FloatTensor(image_data).to(self.device)
            else:
                image = image_data.to(self.device)
            if image.max() > 1.0:
                image = image / 255.0
            if image.dim() == 3:
                image = image.unsqueeze(0)
        
        return state, image
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step.
        
        Trains the noise predictor using denoising score matching.
        
        Args:
            batch: Dictionary with observation and action tensors
            
        Returns:
            Dictionary of training metrics
        """
        # Parse batch
        states = batch.get("states", batch.get("obs", batch.get("state")))
        images = batch.get("images", batch.get("image"))
        actions = batch["actions"] if "actions" in batch else batch["action"]
        
        if states is not None:
            states = states.to(self.device)
        if images is not None:
            images = images.to(self.device)
            if images.dim() == 4 and images.shape[-1] in [1, 3, 4]:
                images = images.permute(0, 3, 1, 2)
            if images.max() > 1.0:
                images = images / 255.0
        actions = actions.to(self.device)
        
        batch_size = actions.shape[0]
        
        # Sample random timesteps
        t = torch.randint(
            0, self.num_diffusion_steps, (batch_size,), 
            device=self.device
        )
        
        # Sample noise
        noise = torch.randn_like(actions)
        
        # Get noisy actions
        noisy_actions = self.diffusion.q_sample(actions, t, noise)
        
        # Normalize timestep
        t_normalized = t.float() / self.num_diffusion_steps
        
        # Predict noise
        predicted_noise = self.noise_predictor(
            noisy_actions, t_normalized, state=states, image=images
        )
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.noise_predictor.parameters(), 1.0)
        self.optimizer.step()
        
        # Update EMA
        self._update_ema()
        
        return {"loss": loss.item()}
    
    @torch.no_grad()
    def sample_action(
        self, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        use_ema: bool = True,
        num_inference_steps: Optional[int] = None
    ) -> np.ndarray:
        """Sample an action given an observation.
        
        Uses the reverse diffusion process (DDPM sampling).
        
        Args:
            obs: Current observation
            use_ema: Whether to use EMA model
            num_inference_steps: Number of inference steps
            
        Returns:
            Sampled action
        """
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        
        # Choose model
        model = self.ema_noise_predictor if (use_ema and self.ema_noise_predictor) else self.noise_predictor
        model.eval()
        
        # Pre-compute observation features
        obs_features = self.obs_encoder(state=state, image=image)
        
        # Start from pure noise
        action = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Determine steps
        steps = num_inference_steps or self.num_diffusion_steps
        step_indices = list(range(0, self.num_diffusion_steps, self.num_diffusion_steps // steps))[::-1]
        
        # Reverse diffusion
        for t in step_indices:
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            predicted_noise = model(action, t_normalized, obs_features=obs_features)
            action = self.diffusion.p_sample(action, t_batch, predicted_noise, clip_denoised=True)
        
        action = torch.clamp(action, -1.0, 1.0)
        return action.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def sample_action_ddim(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        num_inference_steps: int = 10,
        eta: float = 0.0
    ) -> np.ndarray:
        """Sample action using DDIM (faster sampling).
        
        Args:
            obs: Current observation
            num_inference_steps: Number of DDIM steps
            eta: Stochasticity parameter
            
        Returns:
            Sampled action
        """
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        
        model = self.ema_noise_predictor if self.ema_noise_predictor else self.noise_predictor
        model.eval()
        
        # Pre-compute observation features
        obs_features = self.obs_encoder(state=state, image=image)
        
        # DDIM sampling schedule
        step_size = self.num_diffusion_steps // num_inference_steps
        timesteps = list(range(0, self.num_diffusion_steps, step_size))[::-1]
        
        action = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            predicted_noise = model(action, t_normalized, obs_features=obs_features)
            
            alpha_bar = self.diffusion.alphas_bar[t]
            alpha_bar_prev = self.diffusion.alphas_bar[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)
            
            pred_x0 = (action - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * predicted_noise
            noise = sigma * torch.randn_like(action) if i < len(timesteps) - 1 else 0
            
            action = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + noise
        
        action = torch.clamp(action, -1.0, 1.0)
        return action.cpu().numpy().squeeze()
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "noise_predictor": self.noise_predictor.state_dict(),
            "obs_encoder": self.obs_encoder.state_dict(),
            "ema_noise_predictor": self.ema_noise_predictor.state_dict() if self.ema_noise_predictor else None,
            "optimizer": self.optimizer.state_dict(),
            "obs_mode": self.obs_mode,
            "config": {
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "image_shape": self.image_shape,
                "num_diffusion_steps": self.num_diffusion_steps,
            },
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.noise_predictor.load_state_dict(checkpoint["noise_predictor"])
        self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
        if checkpoint.get("ema_noise_predictor"):
            self._init_ema()
            self.ema_noise_predictor.load_state_dict(checkpoint["ema_noise_predictor"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
