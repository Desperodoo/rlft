"""
Unified Flow Matching Policy

Standard flow matching policy with multi-modal observation support.
Uses the Conditional Flow Matching (CFM) objective to learn
a velocity field that transports Gaussian noise to the action distribution.

Supports multiple observation modes:
- "state": State vector only
- "image": Image observation only  
- "state_image": Both state and image (multimodal)

References:
- Flow Matching for Generative Modeling (Lipman et al., 2022)
- Optimal Transport Flow Matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple, Union

try:
    from ...common.networks import MLP, MultiModalVelocityPredictor
    from ...common.obs_encoder import ObservationEncoder, create_obs_encoder
except ImportError:
    try:
        from toy_diffusion_rl.common.networks import MLP, MultiModalVelocityPredictor
        from toy_diffusion_rl.common.obs_encoder import ObservationEncoder, create_obs_encoder
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from common.networks import MLP, MultiModalVelocityPredictor
        from common.obs_encoder import ObservationEncoder, create_obs_encoder


class FlowMatchingPolicy:
    """Unified Flow Matching Policy with multi-modal observation support.
    
    Uses CFM objective to learn a velocity field for action generation.
    Supports state-only, image-only, or state+image observations.
    
    Args:
        action_dim: Dimension of action space
        obs_mode: Observation mode ("state", "image", or "state_image")
        state_dim: Dimension of state space (required for state/state_image)
        image_shape: Image shape as (H, W, C) (required for image/state_image)
        hidden_dims: Hidden layer dimensions
        num_inference_steps: Number of ODE integration steps
        sigma_min: Minimum noise scale for stability
        vision_encoder_type: Type of vision encoder ("cnn" or "dinov2")
        vision_output_dim: Output dimension of vision encoder
        freeze_vision_encoder: Whether to freeze vision encoder
        learning_rate: Learning rate
        device: Device for computation
    """
    
    def __init__(
        self,
        action_dim: int,
        obs_mode: str = "state",
        state_dim: Optional[int] = None,
        image_shape: Optional[Tuple[int, int, int]] = None,
        hidden_dims: List[int] = [256, 256, 256],
        num_inference_steps: int = 10,
        sigma_min: float = 0.001,
        vision_encoder_type: str = "cnn",
        vision_output_dim: int = 128,
        freeze_vision_encoder: bool = False,
        learning_rate: float = 1e-4,
        device: str = "cpu"
    ):
        self.action_dim = action_dim
        self.obs_mode = obs_mode
        self.state_dim = state_dim
        self.image_shape = image_shape
        self.num_inference_steps = num_inference_steps
        self.sigma_min = sigma_min
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
        
        # Velocity network
        self.velocity_net = MultiModalVelocityPredictor(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            obs_encoder=self.obs_encoder,
        ).to(device)
        
        # Collect trainable parameters
        trainable_params = list(self.velocity_net.parameters())
        
        if obs_mode in ["image", "state_image"] and not freeze_vision_encoder:
            vision_params = [p for p in self.obs_encoder.vision_encoder.parameters() if p.requires_grad]
            trainable_params.extend(vision_params)
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    
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
                image = obs.to(self.device)
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
        """Perform one training step using CFM objective.
        
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
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(actions)
        
        # Sample random timesteps t ~ U(0, 1)
        t = torch.rand(batch_size, device=self.device)
        
        # Compute interpolation
        t_expand = t.unsqueeze(-1)
        sigma_t = 1 - (1 - self.sigma_min) * t_expand
        x_t = sigma_t * x_0 + t_expand * actions
        
        # Target velocity
        target_velocity = actions - (1 - self.sigma_min) * x_0
        
        # Predict velocity
        predicted_velocity = self.velocity_net(x_t, t, state=states, image=images)
        
        # MSE loss
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.velocity_net.parameters(), 1.0)
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    @torch.no_grad()
    def sample_action(
        self, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        num_steps: Optional[int] = None,
        use_rk4: bool = False
    ) -> np.ndarray:
        """Sample an action by integrating the learned ODE.
        
        Args:
            obs: Current observation
            num_steps: Number of integration steps
            use_rk4: Use RK4 integrator
            
        Returns:
            Sampled action
        """
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        
        self.velocity_net.eval()
        num_steps = num_steps or self.num_inference_steps
        
        # Pre-compute observation features
        obs_features = self.obs_encoder(state=state, image=image)
        
        if use_rk4:
            action = self._rk4_integrate(obs_features, batch_size, num_steps)
        else:
            action = self._euler_integrate(obs_features, batch_size, num_steps)
        
        return action.cpu().numpy().squeeze()
    
    def _euler_integrate(
        self,
        obs_features: torch.Tensor,
        batch_size: int,
        num_steps: int,
    ) -> torch.Tensor:
        """Integrate using Euler method."""
        dt = 1.0 / num_steps
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            v = self.velocity_net(x, t, obs_features=obs_features)
            x = x + v * dt
        
        return torch.clamp(x, -1.0, 1.0)
    
    def _rk4_integrate(
        self,
        obs_features: torch.Tensor,
        batch_size: int,
        num_steps: int,
    ) -> torch.Tensor:
        """Integrate using RK4 method."""
        dt = 1.0 / num_steps
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for i in range(num_steps):
            t = i * dt
            t_tensor = torch.full((batch_size,), t, device=self.device)
            
            k1 = self.velocity_net(x, t_tensor, obs_features=obs_features)
            
            t_mid = torch.full((batch_size,), t + dt/2, device=self.device)
            k2 = self.velocity_net(x + k1 * dt/2, t_mid, obs_features=obs_features)
            k3 = self.velocity_net(x + k2 * dt/2, t_mid, obs_features=obs_features)
            
            t_end = torch.full((batch_size,), t + dt, device=self.device)
            k4 = self.velocity_net(x + k3 * dt, t_end, obs_features=obs_features)
            
            x = x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        
        return torch.clamp(x, -1.0, 1.0)
    
    @torch.no_grad()
    def sample_action_stochastic(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        num_steps: Optional[int] = None,
        temperature: float = 0.1
    ) -> np.ndarray:
        """Sample action with stochastic noise injection."""
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        
        self.velocity_net.eval()
        num_steps = num_steps or self.num_inference_steps
        
        obs_features = self.obs_encoder(state=state, image=image)
        
        dt = 1.0 / num_steps
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            v = self.velocity_net(x, t, obs_features=obs_features)
            
            noise_scale = temperature * (1 - i / num_steps)
            noise = noise_scale * torch.randn_like(x)
            
            x = x + v * dt + noise * np.sqrt(dt)
        
        x = torch.clamp(x, -1.0, 1.0)
        return x.cpu().numpy().squeeze()
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "velocity_net": self.velocity_net.state_dict(),
            "obs_encoder": self.obs_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_mode": self.obs_mode,
            "config": {
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "image_shape": self.image_shape,
                "num_inference_steps": self.num_inference_steps,
                "sigma_min": self.sigma_min,
            },
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.velocity_net.load_state_dict(checkpoint["velocity_net"])
        self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
