"""
Consistency Flow Policy V2 - Aligned with CPQL Implementation

This version aligns with CPQL's consistency flow implementation:
1. Uses MultiModalVelocityPredictor (simple linear time embedding)
2. Same training data distribution (full batch for both flow + consistency)
3. Same linear interpolation without sigma_min
4. Same inference procedure

This provides a fair BC-only baseline for comparing against CPQL.

References:
- Consistency Models (Song et al., 2023)
- ManiFlow: https://github.com/allenai/maniflow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
import copy

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


class ConsistencyFlowPolicyV2:
    """Consistency Flow Policy V2 - Aligned with CPQL implementation.
    
    Key differences from original ConsistencyFlowPolicy:
    1. Uses MultiModalVelocityPredictor (same as CPQL) with simple linear time embedding
    2. Full batch is used for both flow and consistency losses (shared x_0)
    3. No sigma_min in interpolation (standard linear flow)
    4. Same network architecture as CPQL policy
    
    This provides a fair BC-only baseline for comparing against CPQL.
    
    Args:
        action_dim: Dimension of action space
        obs_mode: Observation mode ("state", "image", or "state_image")
        state_dim: Dimension of state space (required for state/state_image)
        image_shape: Image shape as (H, W, C) (required for image/state_image)
        hidden_dims: Hidden layer dimensions (will append 256 to match CPQL)
        bc_weight: Weight for flow matching (BC) loss
        consistency_weight: Weight for consistency loss
        use_ema_teacher: Use EMA model as teacher for consistency
        ema_decay: Decay rate for EMA
        num_flow_steps: Number of ODE steps for inference
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
        hidden_dims: List[int] = [256, 256],
        bc_weight: float = 1.0,
        consistency_weight: float = 1.0,
        use_ema_teacher: bool = True,
        ema_decay: float = 0.999,
        num_flow_steps: int = 10,
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
        self.bc_weight = bc_weight
        self.consistency_weight = consistency_weight
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self.num_flow_steps = num_flow_steps
        self.device = device
        
        # Create observation encoder (same as CPQL)
        self.obs_encoder = create_obs_encoder(
            obs_mode=obs_mode,
            state_dim=state_dim,
            image_shape=image_shape,
            vision_encoder_type=vision_encoder_type,
            vision_output_dim=vision_output_dim,
            freeze_vision=freeze_vision_encoder,
        ).to(device)
        
        # Velocity network
        self.policy = MultiModalVelocityPredictor(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            obs_encoder=self.obs_encoder,
        ).to(device)
        
        # EMA policy for consistency training (same as CPQL)
        if use_ema_teacher:
            self.policy_ema = copy.deepcopy(self.policy)
            for p in self.policy_ema.parameters():
                p.requires_grad = False
        else:
            self.policy_ema = None
        
        # Collect trainable parameters (same as CPQL)
        policy_params = list(self.policy.parameters())
        
        if obs_mode in ["image", "state_image"] and not freeze_vision_encoder:
            vision_params = [p for p in self.obs_encoder.vision_encoder.parameters() if p.requires_grad]
            policy_params.extend(vision_params)
        
        self.optimizer = torch.optim.Adam(policy_params, lr=learning_rate)
        self.total_steps = 0
    
    def _update_ema(self):
        """Update EMA policy (same as CPQL)."""
        if not self.use_ema_teacher or self.policy_ema is None:
            return
        
        with torch.no_grad():
            for ema_p, p in zip(self.policy_ema.parameters(), self.policy.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
    
    def _linear_interpolate(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation (same as CPQL, no sigma_min)."""
        t_expand = t.unsqueeze(-1) if t.dim() == 1 else t
        return (1 - t_expand) * x_0 + t_expand * x_1
    
    def _parse_observation(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray], torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Parse observation into state and image tensors (same as CPQL)."""
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
        """Train with combined flow matching and consistency loss (aligned with CPQL).
        
        Key alignment with CPQL:
        1. Full batch used for both losses (shared x_0)
        2. Same flow matching formula (no sigma_min)
        3. Same consistency training procedure
        
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
        
        # Pre-compute observation features (same as CPQL)
        obs_features = self.obs_encoder(state=states, image=images)
        
        # ============ Flow Matching Loss (same as CPQL) ============
        # Sample shared x_0 for both losses
        x_0 = torch.randn_like(actions)
        t = torch.rand(batch_size, device=self.device)
        x_t = self._linear_interpolate(x_0, actions, t)
        
        # Target: v = x_1 - x_0 (no sigma_min, same as CPQL)
        v_target = actions - x_0
        
        # Predict velocity
        v_pred = self.policy(x_t, t, obs_features=obs_features)
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # ============ Consistency Loss (same as CPQL) ============
        # Sample t for consistency (avoid boundaries)
        t_cons = 0.05 + torch.rand(batch_size, device=self.device) * 0.9
        delta_t = torch.rand(batch_size, device=self.device) * (0.99 - t_cons)
        delta_t = torch.clamp(delta_t, min=0.02)
        t_next = torch.clamp(t_cons + delta_t, max=0.99)
        
        # Interpolate using shared x_0 (same as CPQL)
        x_t_cons = self._linear_interpolate(x_0, actions, t_cons)
        x_t_next = self._linear_interpolate(x_0, actions, t_next)
        
        # Use EMA teacher to predict velocity at t_next
        with torch.no_grad():
            teacher = self.policy_ema if self.use_ema_teacher else self.policy
            v_teacher = teacher(x_t_next, t_next, obs_features=obs_features)
            t_next_expand = t_next.unsqueeze(-1)
            pred_x1 = x_t_next + (1 - t_next_expand) * v_teacher
        
        # Target velocity at t_cons should point to pred_x1
        v_cons_target = pred_x1 - x_0
        v_cons_pred = self.policy(x_t_cons, t_cons, obs_features=obs_features)
        consistency_loss = F.mse_loss(v_cons_pred, v_cons_target)
        
        # ============ Total Loss (same weighting as CPQL) ============
        total_loss = self.bc_weight * flow_loss + self.consistency_weight * consistency_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        # Update EMA
        self._update_ema()
        
        self.total_steps += 1
        
        return {
            "loss": total_loss.item(),
            "bc_loss": flow_loss.item(),
            "consistency_loss": consistency_loss.item(),
        }
    
    @torch.no_grad()
    def sample_action(
        self, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = True
    ) -> np.ndarray:
        """Sample action using ODE integration (same as CPQL).
        
        Args:
            obs: Current observation
            deterministic: If True, use EMA model (same as CPQL)
            
        Returns:
            Sampled action
        """
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        
        obs_features = self.obs_encoder(state=state, image=image)
        
        # Use EMA for deterministic sampling (same as CPQL)
        net = self.policy_ema if (deterministic and self.use_ema_teacher) else self.policy
        
        # Start from noise
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        dt = 1.0 / self.num_flow_steps
        
        # Euler ODE integration (same as CPQL)
        for i in range(self.num_flow_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            v = net(x, t, obs_features=obs_features)
            x = x + v * dt
        
        action = torch.clamp(x, -1.0, 1.0)
        return action.cpu().numpy().squeeze()
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "policy": self.policy.state_dict(),
            "obs_encoder": self.obs_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "obs_mode": self.obs_mode,
            "config": {
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "image_shape": self.image_shape,
                "num_flow_steps": self.num_flow_steps,
                "bc_weight": self.bc_weight,
                "consistency_weight": self.consistency_weight,
                "ema_decay": self.ema_decay,
            },
        }
        if self.use_ema_teacher and self.policy_ema is not None:
            checkpoint["policy_ema"] = self.policy_ema.state_dict()
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        if self.use_ema_teacher and "policy_ema" in checkpoint and self.policy_ema is not None:
            self.policy_ema.load_state_dict(checkpoint["policy_ema"])
