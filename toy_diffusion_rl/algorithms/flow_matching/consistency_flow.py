"""
Unified Consistency Flow Policy

Consistency-style training for flow matching with multi-modal observation support.
Enables single-step or few-step action generation.

Supports multiple observation modes:
- "state": State vector only
- "image": Image observation only  
- "state_image": Both state and image (multimodal)

References:
- Consistency Models (Song et al., 2023)
- ManiFlow: https://github.com/allenai/maniflow
- rectified-flow-pytorch: https://github.com/lucidrains/rectified-flow-pytorch

Key Insight from ManiFlow:
- The model predicts velocity v = x_1 - x_0 (constant for linear flow)
- At inference: x_1 = x_0 + v
- Consistency training ensures model predicts consistent x_1 across timesteps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
import copy

try:
    from ...common.networks import MLP, TimestepEmbedding
    from ...common.obs_encoder import ObservationEncoder, create_obs_encoder
except ImportError:
    try:
        from toy_diffusion_rl.common.networks import MLP, TimestepEmbedding
        from toy_diffusion_rl.common.obs_encoder import ObservationEncoder, create_obs_encoder
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from common.networks import MLP, TimestepEmbedding
        from common.obs_encoder import ObservationEncoder, create_obs_encoder


class MultiModalVelocityPredictorV2(nn.Module):
    """Velocity predictor with multi-modal observation support.
    
    Uses TimestepEmbedding (sinusoidal + MLP) for consistency with
    non-multimodal FlowVelocityPredictor.
    
    Args:
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        time_embed_dim: Dimension of timestep embedding
        obs_encoder: ObservationEncoder for processing observations
        obs_dim: Fallback observation dimension if no encoder
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        time_embed_dim: int = 64,
        obs_encoder: Optional[ObservationEncoder] = None,
        obs_dim: int = 0,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.obs_encoder = obs_encoder
        
        if obs_encoder is not None:
            obs_feature_dim = obs_encoder.output_dim
        else:
            obs_feature_dim = obs_dim
        
        self.obs_feature_dim = obs_feature_dim
        
        # Use TimestepEmbedding for consistency with non-multimodal version
        self.time_embed = TimestepEmbedding(time_embed_dim)
        
        # Velocity prediction network
        input_dim = obs_feature_dim + action_dim + time_embed_dim
        self.net = MLP(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=nn.SiLU(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        obs_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity at position x and time t.
        
        Args:
            x: Current position in flow (B, action_dim)
            t: Current timestep in [0, 1] (B,)
            state: Optional state vector (B, state_dim)
            image: Optional image tensor (B, C, H, W)
            obs_features: Pre-computed observation features
        
        Returns:
            Predicted velocity (B, action_dim)
        """
        if obs_features is not None:
            features = obs_features
        elif self.obs_encoder is not None:
            features = self.obs_encoder(state=state, image=image)
        else:
            features = state
        
        # Use TimestepEmbedding (takes raw t, handles conversion internally)
        t_embed = self.time_embed(t)
        combined = torch.cat([features, x, t_embed], dim=-1)
        return self.net(combined)


class ConsistencyFlowPolicy:
    """Unified Consistency Flow Policy with multi-modal observation support.
    
    Following ManiFlow's approach:
    1. Train with mixed Flow + Consistency objectives
    2. Model predicts velocity v = x_1 - x_0
    3. Use EMA teacher for consistency targets
    4. Single-step inference: x_1 = x_0 + v
    
    Args:
        action_dim: Dimension of action space
        obs_mode: Observation mode ("state", "image", or "state_image")
        state_dim: Dimension of state space (required for state/state_image)
        image_shape: Image shape as (H, W, C) (required for image/state_image)
        hidden_dims: Hidden layer dimensions
        flow_batch_ratio: Fraction of batch for flow loss (default 0.5)
        consistency_batch_ratio: Fraction of batch for consistency loss (default 0.5)
        use_ema_teacher: Use EMA model as teacher
        ema_decay: Decay rate for EMA
        num_inference_steps: Number of ODE steps for multi-step inference
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
        flow_batch_ratio: float = 0.5,
        consistency_batch_ratio: float = 0.5,
        use_ema_teacher: bool = True,
        ema_decay: float = 0.999,
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
        self.flow_batch_ratio = flow_batch_ratio
        self.consistency_batch_ratio = consistency_batch_ratio
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
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
        self.velocity_net = MultiModalVelocityPredictorV2(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            obs_encoder=self.obs_encoder,
        ).to(device)
        
        # EMA teacher for consistency training
        if use_ema_teacher:
            self.ema_velocity_net = copy.deepcopy(self.velocity_net)
            for p in self.ema_velocity_net.parameters():
                p.requires_grad = False
        
        # Collect trainable parameters
        trainable_params = list(self.velocity_net.parameters())
        
        if obs_mode in ["image", "state_image"] and not freeze_vision_encoder:
            vision_params = [p for p in self.obs_encoder.vision_encoder.parameters() if p.requires_grad]
            trainable_params.extend(vision_params)
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    
    def _update_ema(self):
        """Update EMA parameters."""
        if not self.use_ema_teacher:
            return
            
        with torch.no_grad():
            for ema_p, p in zip(
                self.ema_velocity_net.parameters(),
                self.velocity_net.parameters()
            ):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1-self.ema_decay)
    
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
    
    def _linear_interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation: x_t = (1-t)*x_0 + t*x_1"""
        t_expand = t.view(-1, 1) if t.dim() == 1 else t
        return (1 - t_expand) * x_0 + t_expand * x_1
    
    def _get_flow_targets(
        self,
        actions: torch.Tensor,
        obs_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get flow matching targets.
        
        Standard flow matching: predict instantaneous velocity v = x_1 - x_0
        This is constant along the linear interpolation path.
        """
        batch_size = actions.shape[0]
        
        # Sample noise (starting point)
        x_0 = torch.randn_like(actions)
        
        # Sample timestep uniformly
        t = torch.rand(batch_size, device=self.device)
        
        # Interpolate (with sigma_min for stability)
        t_expand = t.unsqueeze(-1)
        sigma_t = 1 - (1 - self.sigma_min) * t_expand
        x_t = sigma_t * x_0 + t_expand * actions
        
        # Target: velocity accounting for sigma_min
        v_target = actions - (1 - self.sigma_min) * x_0
        
        return {
            'x_t': x_t,
            't': t,
            'v_target': v_target,
            'x_0': x_0,
            'x_1': actions
        }
    
    def _get_consistency_targets(
        self,
        actions: torch.Tensor,
        obs_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get consistency training targets (ManiFlow-style).
        
        Standard flow matching predicts v = x_1 - x_0 (constant velocity).
        For consistency, we want the model to learn to predict the same x_1
        regardless of the timestep t.
        
        Method:
        1. At t_next, use EMA teacher to predict velocity v
        2. Compute predicted x_1 = x_t_next + (1 - t_next) * v
        3. Target velocity at t is: pred_x_1 - x_0
        """
        batch_size = actions.shape[0]
        
        # Sample noise
        x_0 = torch.randn_like(actions)
        
        # Sample t uniformly, but keep away from boundaries
        t = 0.05 + torch.rand(batch_size, device=self.device) * 0.9  # t in [0.05, 0.95]
        
        # Sample delta_t 
        delta_t = torch.rand(batch_size, device=self.device) * (0.99 - t)
        delta_t = torch.clamp(delta_t, min=0.02)
        
        t_next = torch.clamp(t + delta_t, max=0.99)
        
        # Interpolate to get x_t and x_t_next
        x_t = self._linear_interpolate(x_0, actions, t)
        x_t_next = self._linear_interpolate(x_0, actions, t_next)
        
        # Use EMA teacher to predict velocity at t_next
        with torch.no_grad():
            net = self.ema_velocity_net if self.use_ema_teacher else self.velocity_net
            v_teacher = net(x_t_next, t_next, obs_features=obs_features)
            
            # Estimate x_1 using teacher's prediction: x_1 = x_t_next + (1 - t_next) * v
            t_next_expand = t_next.unsqueeze(-1)
            pred_x1 = x_t_next + (1 - t_next_expand) * v_teacher
        
        # Target velocity at t: should point to the same pred_x1
        # v_target = pred_x1 - x_0 (since for constant velocity, v = x_1 - x_0)
        v_target = pred_x1 - x_0
        
        return {
            'x_t': x_t,
            't': t,
            'v_target': v_target,
            'x_0': x_0,
            'x_1': actions
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train with combined flow matching and consistency loss.
        
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
        
        # Pre-compute observation features (shared for both losses)
        obs_features = self.obs_encoder(state=states, image=images)
        
        # Split batch for flow and consistency training
        flow_size = int(batch_size * self.flow_batch_ratio)
        cons_size = int(batch_size * self.consistency_batch_ratio)
        
        total_loss = 0.0
        flow_loss_val = 0.0
        cons_loss_val = 0.0
        
        # ============ Flow Matching Loss ============
        if flow_size > 0:
            flow_actions = actions[:flow_size]
            flow_features = obs_features[:flow_size]
            
            flow_targets = self._get_flow_targets(flow_actions, flow_features)
            
            # Predict velocity
            v_pred = self.velocity_net(flow_targets['x_t'], flow_targets['t'], obs_features=flow_features)
            
            flow_loss = F.mse_loss(v_pred, flow_targets['v_target'])
            total_loss = total_loss + flow_loss
            flow_loss_val = flow_loss.item()
        
        # ============ Consistency Loss ============
        if cons_size > 0:
            cons_actions = actions[flow_size:flow_size+cons_size]
            cons_features = obs_features[flow_size:flow_size+cons_size]
            
            cons_targets = self._get_consistency_targets(cons_actions, cons_features)
            
            # Predict velocity
            v_pred = self.velocity_net(cons_targets['x_t'], cons_targets['t'], obs_features=cons_features)
            
            cons_loss = F.mse_loss(v_pred, cons_targets['v_target'])
            total_loss = total_loss + cons_loss
            cons_loss_val = cons_loss.item()
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.velocity_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update EMA
        self._update_ema()
        
        return {
            "loss": total_loss.item(),
            "flow_loss": flow_loss_val,
            "consistency_loss": cons_loss_val,
        }
    
    @torch.no_grad()
    def sample_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        num_steps: Optional[int] = None,
        use_ema: bool = True
    ) -> np.ndarray:
        """Sample action using ODE integration.
        
        Args:
            obs: Current observation
            num_steps: Number of ODE steps (default: self.num_inference_steps)
            use_ema: If True, use EMA model
            
        Returns:
            Sampled action
        """
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        
        num_steps = num_steps or self.num_inference_steps
        
        # Select which network to use
        net = self.ema_velocity_net if (use_ema and self.use_ema_teacher) else self.velocity_net
        net.eval()
        
        # Pre-compute observation features
        obs_features = self.obs_encoder(state=state, image=image)
        
        # Start from noise
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # ODE integration with Euler method
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            v = net(x, t, obs_features=obs_features)
            x = x + v * dt
        
        action = torch.clamp(x, -1.0, 1.0)
        return action.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def sample_action_single_step(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        use_ema: bool = True
    ) -> np.ndarray:
        """Sample action in a single step (consistency model style).
        
        Args:
            obs: Current observation
            use_ema: If True, use EMA model
            
        Returns:
            Sampled action
        """
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        
        net = self.ema_velocity_net if (use_ema and self.use_ema_teacher) else self.velocity_net
        net.eval()
        
        obs_features = self.obs_encoder(state=state, image=image)
        
        # Start from noise
        x_0 = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Single step at t=0: x_1 = x_0 + v(x_0, t=0)
        t = torch.zeros(batch_size, device=self.device)
        v = net(x_0, t, obs_features=obs_features)
        x_1 = x_0 + v
        
        action = torch.clamp(x_1, -1.0, 1.0)
        return action.cpu().numpy().squeeze()
    
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
                "flow_batch_ratio": self.flow_batch_ratio,
                "consistency_batch_ratio": self.consistency_batch_ratio,
                "ema_decay": self.ema_decay,
            },
        }
        if self.use_ema_teacher:
            checkpoint["ema_velocity_net"] = self.ema_velocity_net.state_dict()
        torch.save(checkpoint, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.velocity_net.load_state_dict(checkpoint["velocity_net"])
        self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.use_ema_teacher and "ema_velocity_net" in checkpoint:
            self.ema_velocity_net.load_state_dict(checkpoint["ema_velocity_net"])
