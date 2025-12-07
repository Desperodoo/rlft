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


class FlowMatchingPolicy:
    """Unified Flow Matching Policy with multi-modal observation support.
    
    Uses CFM objective to learn a velocity field for action generation.
    Supports state-only, image-only, or state+image observations.
    
    Supports Action Chunking (when action_horizon > 1):
    - Predicts a sequence of future actions instead of a single action
    - Uses an action queue to execute actions over multiple steps
    - action_exec_horizon controls how many actions are executed before re-planning
    
    Args:
        action_dim: Dimension of action space
        obs_mode: Observation mode ("state", "image", or "state_image")
        state_dim: Dimension of state space (required for state/state_image)
        image_shape: Image shape as (H, W, C) (required for image/state_image)
        hidden_dims: Hidden layer dimensions
        num_inference_steps: Number of ODE integration steps
        use_ema: Whether to use EMA for inference
        ema_decay: Decay rate for EMA
        vision_encoder_type: Type of vision encoder ("cnn" or "dinov2")
        vision_output_dim: Output dimension of vision encoder
        freeze_vision_encoder: Whether to freeze vision encoder
        learning_rate: Learning rate
        action_horizon: Number of future actions to predict (1 = no chunking)
        action_exec_horizon: Number of actions to execute before re-planning
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
        use_ema: bool = True,
        ema_decay: float = 0.999,
        vision_encoder_type: str = "cnn",
        vision_output_dim: int = 128,
        freeze_vision_encoder: bool = False,
        learning_rate: float = 3e-4,
        action_horizon: int = 1,
        action_exec_horizon: Optional[int] = None,
        device: str = "cpu"
    ):
        self.action_dim = action_dim
        self.obs_mode = obs_mode
        self.state_dim = state_dim
        self.image_shape = image_shape
        self.num_inference_steps = num_inference_steps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.device = device
        
        # Action chunking parameters
        self.action_horizon = action_horizon
        self.action_exec_horizon = action_exec_horizon if action_exec_horizon is not None else action_horizon
        
        # Vectorized action queues for temporal action execution
        # Dict[env_id, List[action]] to support VecEnv parallel evaluation
        self._action_queues: Dict[int, List[np.ndarray]] = {}
        self._num_envs: int = 1
        
        # Create observation encoder
        self.obs_encoder = create_obs_encoder(
            obs_mode=obs_mode,
            state_dim=state_dim,
            image_shape=image_shape,
            vision_encoder_type=vision_encoder_type,
            vision_output_dim=vision_output_dim,
            freeze_vision=freeze_vision_encoder,
        ).to(device)
        
        # Velocity network with action chunking support
        self.velocity_net = MultiModalVelocityPredictor(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            obs_encoder=self.obs_encoder,
            action_horizon=action_horizon,
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
    
    def reset(self, num_envs: Optional[int] = None, env_ids: Optional[List[int]] = None):
        """Reset the action queues for VecEnv support.
        
        Args:
            num_envs: If provided, initialize queues for this many environments.
                     If None and env_ids is None, resets all existing queues.
            env_ids: If provided, only reset queues for these specific env IDs.
                    Useful for auto-reset in VecEnv where only some envs reset.
        """
        if num_envs is not None:
            # Initialize/reset all queues for given number of envs
            self._num_envs = num_envs
            self._action_queues = {i: [] for i in range(num_envs)}
        elif env_ids is not None:
            # Reset only specific env queues (for auto-reset support)
            for env_id in env_ids:
                self._action_queues[env_id] = []
        else:
            # Reset all existing queues
            for env_id in self._action_queues:
                self._action_queues[env_id] = []
    
    def _update_ema(self):
        """Update EMA velocity network."""
        if not self.use_ema or self.velocity_net_ema is None:
            return
        
        with torch.no_grad():
            for ema_p, p in zip(self.velocity_net_ema.parameters(), self.velocity_net.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
    
    def _linear_interpolate(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation (no sigma_min).
        
        Handles both single action and action sequence cases.
        """
        if x_0.dim() == 3:  # (B, T, action_dim)
            t_expand = t.unsqueeze(-1).unsqueeze(-1)
        else:  # (B, action_dim)
            t_expand = t.unsqueeze(-1)
        return (1 - t_expand) * x_0 + t_expand * x_1
    
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
        
        # Compute linear interpolation (no sigma_min)
        x_t = self._linear_interpolate(x_0, actions, t)
        
        # Target velocity: v = x_1 - x_0 (standard flow matching)
        target_velocity = actions - x_0
        
        # Predict velocity
        predicted_velocity = self.velocity_net(x_t, t, state=states, image=images)
        
        # MSE loss
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
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
        use_rk4: bool = False,
        deterministic: bool = True
    ) -> np.ndarray:
        """Sample action(s) by integrating the learned ODE.
        
        With action chunking enabled, uses per-env action queues to avoid
        re-sampling at every step. Supports both single-env and VecEnv.
        
        Args:
            obs: Current observation(s). Can be batched for VecEnv.
            num_steps: Number of integration steps
            use_rk4: Use RK4 integrator
            deterministic: If True, use EMA model for stable inference
            
        Returns:
            Sampled action(s): shape (action_dim,) for single env,
                              or (num_envs, action_dim) for VecEnv
        """
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        is_batched = batch_size > 1
        
        # Initialize queues if not already done
        if len(self._action_queues) == 0:
            self._action_queues = {i: [] for i in range(batch_size)}
            self._num_envs = batch_size
        
        # Check which envs need new actions (queue empty)
        envs_need_sample = [i for i in range(batch_size) if len(self._action_queues.get(i, [])) == 0]
        
        # If all envs have cached actions, just pop and return
        if len(envs_need_sample) == 0:
            actions = np.stack([self._action_queues[i].pop(0) for i in range(batch_size)])
            return actions if is_batched else actions[0]
        
        # Use EMA model for deterministic sampling
        net = self.velocity_net_ema if (deterministic and self.use_ema and self.velocity_net_ema is not None) else self.velocity_net
        net.eval()
        num_steps = num_steps or self.num_inference_steps
        
        # Pre-compute observation features
        obs_features = self.obs_encoder(state=state, image=image)
        
        if use_rk4:
            action = self._rk4_integrate(obs_features, batch_size, num_steps, net)
        else:
            action = self._euler_integrate(obs_features, batch_size, num_steps, net)
        
        action_np = action.cpu().numpy()  # (batch, action_horizon, action_dim) or (batch, action_dim)
        
        # Handle action chunking: fill each env's action queue
        if self.action_horizon > 1:
            for env_id in range(batch_size):
                if env_id not in self._action_queues:
                    self._action_queues[env_id] = []
                # Only fill for envs that needed new actions
                if env_id in envs_need_sample:
                    self._action_queues[env_id] = []  # Clear first
                    for i in range(min(self.action_exec_horizon, self.action_horizon)):
                        self._action_queues[env_id].append(action_np[env_id, i])
            
            # Pop and return first action from each env's queue
            actions = np.stack([self._action_queues[i].pop(0) for i in range(batch_size)])
            return actions if is_batched else actions[0]
        else:
            return action_np if is_batched else action_np[0]
    
    def _euler_integrate(
        self,
        obs_features: torch.Tensor,
        batch_size: int,
        num_steps: int,
        net: nn.Module = None,
    ) -> torch.Tensor:
        """Integrate using Euler method."""
        if net is None:
            net = self.velocity_net
        dt = 1.0 / num_steps
        
        # Start from noise - shape depends on action_horizon
        if self.action_horizon > 1:
            x = torch.randn(batch_size, self.action_horizon, self.action_dim, device=self.device)
        else:
            x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            v = net(x, t, obs_features=obs_features)
            x = x + v * dt
        
        return torch.clamp(x, -1.0, 1.0)
    
    def _rk4_integrate(
        self,
        obs_features: torch.Tensor,
        batch_size: int,
        num_steps: int,
        net: nn.Module = None,
    ) -> torch.Tensor:
        """Integrate using RK4 method."""
        if net is None:
            net = self.velocity_net
        dt = 1.0 / num_steps
        
        # Start from noise - shape depends on action_horizon
        if self.action_horizon > 1:
            x = torch.randn(batch_size, self.action_horizon, self.action_dim, device=self.device)
        else:
            x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for i in range(num_steps):
            t = i * dt
            t_tensor = torch.full((batch_size,), t, device=self.device)
            
            k1 = net(x, t_tensor, obs_features=obs_features)
            
            t_mid = torch.full((batch_size,), t + dt/2, device=self.device)
            k2 = net(x + k1 * dt/2, t_mid, obs_features=obs_features)
            k3 = net(x + k2 * dt/2, t_mid, obs_features=obs_features)
            
            t_end = torch.full((batch_size,), t + dt, device=self.device)
            k4 = net(x + k3 * dt, t_end, obs_features=obs_features)
            
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
            "total_steps": self.total_steps,
            "obs_mode": self.obs_mode,
            "config": {
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "image_shape": self.image_shape,
                "num_inference_steps": self.num_inference_steps,
                "use_ema": self.use_ema,
                "ema_decay": self.ema_decay,
                "action_horizon": self.action_horizon,
                "action_exec_horizon": self.action_exec_horizon,
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
