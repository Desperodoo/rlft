"""
Unified Reflected Flow Policy

Flow matching with boundary reflection for bounded action spaces.
When the flow trajectory hits action bounds, it reflects back into
the valid region, maintaining the density.

Supports multiple observation modes:
- "state": State vector only
- "image": Image observation only  
- "state_image": Both state and image (multimodal)

References:
- Reflected Flow Matching (inspired by rectified flow ideas)
- rectified-flow-pytorch: https://github.com/lucidrains/rectified-flow-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
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


class ReflectedFlowPolicy:
    """Unified Reflected Flow Policy with multi-modal observation support.
    
    Extends vanilla flow matching with boundary reflection:
    - During training, actions that would go outside [-1, 1] are reflected
    - This helps maintain proper density when learning bounded actions
    
    Two modes of reflection are supported:
    1. Hard reflection: Immediate bounce back at boundary
    2. Soft reflection: Smooth clamping (with gradient)
    
    Args:
        action_dim: Dimension of action space
        obs_mode: Observation mode ("state", "image", or "state_image")
        state_dim: Dimension of state space (required for state/state_image)
        image_shape: Image shape as (H, W, C) (required for image/state_image)
        hidden_dims: Hidden layer dimensions
        num_inference_steps: Number of ODE integration steps
        reflection_mode: 'hard' or 'soft'
        action_bounds: Action bounds (default: [-1, 1])
        boundary_regularization: Weight for boundary regularization loss
        use_ema: Whether to use EMA for inference
        ema_decay: Decay rate for EMA
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
        reflection_mode: str = "hard",
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        boundary_regularization: float = 0.1,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        vision_encoder_type: str = "cnn",
        vision_output_dim: int = 128,
        freeze_vision_encoder: bool = False,
        learning_rate: float = 3e-4,
        device: str = "cpu"
    ):
        self.action_dim = action_dim
        self.obs_mode = obs_mode
        self.state_dim = state_dim
        self.image_shape = image_shape
        self.num_inference_steps = num_inference_steps
        self.reflection_mode = reflection_mode
        self.action_low = action_bounds[0]
        self.action_high = action_bounds[1]
        self.boundary_regularization = boundary_regularization
        self.use_ema = use_ema
        self.ema_decay = ema_decay
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
        
        # Velocity network (use same as FlowMatching for consistency)
        self.velocity_net = MultiModalVelocityPredictor(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            obs_encoder=self.obs_encoder,
        ).to(device)
        
        # EMA velocity network for stable inference
        if use_ema:
            self.velocity_net_ema = copy.deepcopy(self.velocity_net)
            for p in self.velocity_net_ema.parameters():
                p.requires_grad = False
        else:
            self.velocity_net_ema = None
        
        # Collect trainable parameters
        trainable_params = list(self.velocity_net.parameters())
        
        if obs_mode in ["image", "state_image"] and not freeze_vision_encoder:
            vision_params = [p for p in self.obs_encoder.vision_encoder.parameters() if p.requires_grad]
            trainable_params.extend(vision_params)
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        self.total_steps = 0
    
    def _update_ema(self):
        """Update EMA velocity network."""
        if not self.use_ema or self.velocity_net_ema is None:
            return
        
        with torch.no_grad():
            for ema_p, p in zip(self.velocity_net_ema.parameters(), self.velocity_net.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
    
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
    
    def _reflect_trajectory(
        self, 
        x: torch.Tensor, 
        v: torch.Tensor, 
        dt: float
    ) -> torch.Tensor:
        """Apply reflection when trajectory exits bounds.
        
        Implements elastic reflection: when x + v*dt would exceed bounds,
        the excess is reflected back into the valid region.
        
        Args:
            x: Current position
            v: Velocity
            dt: Time step
            
        Returns:
            New position with reflection applied
        """
        x_new = x + v * dt
        
        if self.reflection_mode == "hard":
            # Hard reflection: bounce back
            # Track how many times we cross each boundary
            max_iterations = 10
            for _ in range(max_iterations):
                # Check lower bound
                below_low = x_new < self.action_low
                if below_low.any():
                    excess = self.action_low - x_new
                    x_new = torch.where(below_low, self.action_low + excess, x_new)
                    
                # Check upper bound  
                above_high = x_new > self.action_high
                if above_high.any():
                    excess = x_new - self.action_high
                    x_new = torch.where(above_high, self.action_high - excess, x_new)
                    
                # Check if all within bounds
                if not (below_low.any() or above_high.any()):
                    break
            
            # Final clamp for safety
            x_new = torch.clamp(x_new, self.action_low, self.action_high)
        else:
            # Soft reflection: just clamp (with gradient)
            x_new = torch.clamp(x_new, self.action_low, self.action_high)
            
        return x_new
    
    def _compute_reflected_target(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute target velocity with reflection awareness.
        
        The target velocity is adjusted to account for reflections
        that would occur along the linear interpolation path.
        
        Args:
            x_0: Initial noise
            x_1: Target action (already bounded)
            t: Current timestep
            
        Returns:
            Tuple of (target velocity, current position x_t)
        """
        # Compute linear interpolation
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        # If x_t is outside bounds, project to boundary
        x_t_reflected = torch.clamp(x_t, self.action_low, self.action_high)
        
        # Target velocity: direction to final action
        # Adjusted for remaining time
        remaining_t = 1 - t_expand + 1e-6
        target_v = (x_1 - x_t_reflected) / remaining_t
        
        return target_v, x_t_reflected
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train step with reflection-aware flow matching.
        
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
        
        # Pre-compute observation features
        obs_features = self.obs_encoder(state=states, image=images)
        
        batch_size = actions.shape[0]
        
        # Sample noise (can be larger than action bounds)
        x_0 = torch.randn_like(actions)
        
        # Sample timesteps
        t = torch.rand(batch_size, device=self.device)
        
        # Compute interpolation with reflection consideration
        target_v, x_t = self._compute_reflected_target(x_0, actions, t)
        
        # Predict velocity
        predicted_v = self.velocity_net(x_t, t, obs_features=obs_features)
        
        # Main loss
        loss = F.mse_loss(predicted_v, target_v)
        
        # Add boundary regularization loss
        # Encourage velocities that point inward at boundaries
        if self.boundary_regularization > 0:
            boundary_margin = 0.1
            lower_mask = x_t <= self.action_low + boundary_margin
            upper_mask = x_t >= self.action_high - boundary_margin
            
            boundary_loss = 0.0
            if lower_mask.any():
                # At lower bound, velocity should be positive
                boundary_loss = boundary_loss + F.relu(-predicted_v[lower_mask]).mean()
            if upper_mask.any():
                # At upper bound, velocity should be negative
                boundary_loss = boundary_loss + F.relu(predicted_v[upper_mask]).mean()
                
            loss = loss + self.boundary_regularization * boundary_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.velocity_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update EMA
        self._update_ema()
        self.total_steps += 1
        
        return {"loss": loss.item()}
    
    @torch.no_grad()
    def sample_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        num_steps: Optional[int] = None,
        deterministic: bool = True
    ) -> np.ndarray:
        """Sample action with reflected integration.
        
        Args:
            obs: Current observation
            num_steps: Number of integration steps
            deterministic: If True, use EMA model for stable inference
            
        Returns:
            Sampled action
        """
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        
        # Use EMA model for deterministic sampling
        net = self.velocity_net_ema if (deterministic and self.use_ema and self.velocity_net_ema is not None) else self.velocity_net
        net.eval()
        num_steps = num_steps or self.num_inference_steps
        
        # Pre-compute observation features
        obs_features = self.obs_encoder(state=state, image=image)
        
        dt = 1.0 / num_steps
        
        # Start from noise
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            v = net(x, t, obs_features=obs_features)
            
            # Apply reflection during integration
            x = self._reflect_trajectory(x, v, dt)
        
        # Final clamp for safety
        x = torch.clamp(x, self.action_low, self.action_high)
        
        return x.cpu().numpy().squeeze()
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "velocity_net": self.velocity_net.state_dict(),
            "obs_encoder": self.obs_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "obs_mode": self.obs_mode,
            "config": {
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "image_shape": self.image_shape,
                "num_inference_steps": self.num_inference_steps,
                "reflection_mode": self.reflection_mode,
                "action_bounds": (self.action_low, self.action_high),
                "use_ema": self.use_ema,
                "ema_decay": self.ema_decay,
            },
        }
        if self.use_ema and self.velocity_net_ema is not None:
            checkpoint["velocity_net_ema"] = self.velocity_net_ema.state_dict()
        torch.save(checkpoint, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.velocity_net.load_state_dict(checkpoint["velocity_net"])
        self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        if self.use_ema and "velocity_net_ema" in checkpoint and self.velocity_net_ema is not None:
            self.velocity_net_ema.load_state_dict(checkpoint["velocity_net_ema"])
