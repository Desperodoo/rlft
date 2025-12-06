"""
Consistency Policy Q-Learning (CPQL) Agent - Unified Version

Combines Consistency Flow Matching with Q-Learning for efficient offline RL.
Supports multiple observation modes through configurable encoder.

References:
- CPQL: https://github.com/cccedric/cpql
- Diffusion-QL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
import copy

try:
    from ...common.networks import MLP, MultiModalVelocityPredictor, MultiModalDoubleQNetwork
    from ...common.utils import soft_update
    from ...common.obs_encoder import ObservationEncoder, create_obs_encoder
except ImportError:
    try:
        from toy_diffusion_rl.common.networks import MLP, MultiModalVelocityPredictor, MultiModalDoubleQNetwork
        from toy_diffusion_rl.common.utils import soft_update
        from toy_diffusion_rl.common.obs_encoder import ObservationEncoder, create_obs_encoder
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from common.networks import MLP, MultiModalVelocityPredictor, MultiModalDoubleQNetwork
        from common.utils import soft_update
        from common.obs_encoder import ObservationEncoder, create_obs_encoder


class CPQLAgent:
    """Unified CPQL Agent with multi-modal observation support.
    
    Combines Consistency Flow with Q-Learning.
    Supports state-only, image-only, or state+image observations.
    
    Args:
        action_dim: Dimension of action space
        obs_mode: Observation mode ("state", "image", or "state_image")
        state_dim: Dimension of state space
        image_shape: Image shape as (H, W, C)
        hidden_dims: Hidden layer dimensions
        vision_encoder_type: Type of vision encoder
        vision_output_dim: Output dimension of vision encoder
        freeze_vision_encoder: Whether to freeze vision encoder
        learning_rate_policy: Learning rate for policy
        learning_rate_q: Learning rate for Q-network
        gamma: Discount factor
        tau: Soft update coefficient
        alpha: Temperature for Q-value weighting (reduced for stability)
        bc_weight: Weight for BC loss
        consistency_weight: Weight for consistency loss
        num_flow_steps: Number of ODE steps
        reward_scale: Scale factor for rewards (for Q-value stability)
        q_target_clip: Clip range for Q target (None to disable)
        device: Device for computation
    """
    
    def __init__(
        self,
        action_dim: int,
        obs_mode: str = "state",
        state_dim: Optional[int] = None,
        image_shape: Optional[Tuple[int, int, int]] = None,
        hidden_dims: List[int] = [256, 256],
        vision_encoder_type: str = "cnn",
        vision_output_dim: int = 128,
        freeze_vision_encoder: bool = False,
        learning_rate_policy: float = 3e-4,
        learning_rate_q: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.01,  # Reduced from 0.1 for stability
        bc_weight: float = 1.0,
        consistency_weight: float = 1.0,
        num_flow_steps: int = 10,
        reward_scale: float = 0.1,  # Scale down rewards for Q-value stability
        q_target_clip: Optional[float] = 100.0,  # Clip Q targets to prevent explosion
        device: str = "cpu",
        # Legacy parameters
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_noise_levels: int = 20,
    ):
        self.action_dim = action_dim
        self.obs_mode = obs_mode
        self.state_dim = state_dim
        self.image_shape = image_shape
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.bc_weight = bc_weight
        self.consistency_weight = consistency_weight
        self.num_flow_steps = num_flow_steps
        self.reward_scale = reward_scale
        self.q_target_clip = q_target_clip
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
        
        # Flow Policy
        self.policy = MultiModalVelocityPredictor(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            obs_encoder=self.obs_encoder,
        ).to(device)
        
        # EMA policy
        self.policy_ema = copy.deepcopy(self.policy)
        for p in self.policy_ema.parameters():
            p.requires_grad = False
        
        # Q-Networks
        self.critic = MultiModalDoubleQNetwork(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            obs_encoder=self.obs_encoder,
        ).to(device)
        
        # Target Q-networks
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        # Collect trainable parameters
        policy_params = list(self.policy.parameters())
        q_params = list(self.critic.parameters())
        
        if obs_mode in ["image", "state_image"] and not freeze_vision_encoder:
            vision_params = [p for p in self.obs_encoder.vision_encoder.parameters() if p.requires_grad]
            policy_params.extend(vision_params)
        
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=learning_rate_policy)
        self.q_optimizer = torch.optim.Adam(q_params, lr=learning_rate_q)
        
        self.total_steps = 0
        self.ema_decay = 0.999
    
    def _update_ema(self):
        """Update EMA policy."""
        with torch.no_grad():
            for ema_p, p in zip(self.policy_ema.parameters(), self.policy.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
    
    def _linear_interpolate(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_expand = t.unsqueeze(-1) if t.dim() == 1 else t
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
    
    @torch.no_grad()
    def _sample_actions_batch(
        self,
        obs_features: torch.Tensor,
        num_steps: Optional[int] = None,
        use_ema: bool = True
    ) -> torch.Tensor:
        batch_size = obs_features.shape[0]
        num_steps = num_steps or self.num_flow_steps
        net = self.policy_ema if use_ema else self.policy
        
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            v = net(x, t, obs_features=obs_features)
            x = x + v * dt
        
        return torch.clamp(x, -1.0, 1.0)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        # Parse batch
        states = batch.get("states", batch.get("obs", batch.get("state")))
        next_states = batch.get("next_states", batch.get("next_obs"))
        images = batch.get("images", batch.get("image"))
        next_images = batch.get("next_images", batch.get("next_image"))
        actions = batch["actions"] if "actions" in batch else batch["action"]
        rewards = batch["rewards"] if "rewards" in batch else batch["reward"]
        dones = batch["dones"] if "dones" in batch else batch["done"]
        
        if states is not None:
            states = states.to(self.device)
            next_states = next_states.to(self.device)
        if images is not None:
            images = images.to(self.device)
            next_images = next_images.to(self.device)
            if images.dim() == 4 and images.shape[-1] in [1, 3, 4]:
                images = images.permute(0, 3, 1, 2)
                next_images = next_images.permute(0, 3, 1, 2)
            if images.max() > 1.0:
                images = images / 255.0
                next_images = next_images / 255.0
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        batch_size = actions.shape[0]
        
        # Scale rewards for Q-value stability
        scaled_rewards = rewards * self.reward_scale
        
        # Update Q-Networks
        with torch.no_grad():
            next_obs_features = self.obs_encoder(state=next_states, image=next_images)
            next_actions = self._sample_actions_batch(next_obs_features)
            
            target_q1, target_q2 = self.critic_target(next_actions, obs_features=next_obs_features)
            target_q = torch.min(target_q1, target_q2)
            target_q = scaled_rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * target_q
            
            # Clip Q targets to prevent explosion
            if self.q_target_clip is not None:
                target_q = torch.clamp(target_q, -self.q_target_clip, self.q_target_clip)
        
        obs_features = self.obs_encoder(state=states, image=images)
        current_q1, current_q2 = self.critic(actions, obs_features=obs_features)
        
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.q_optimizer.step()
        
        # Update Policy
        policy_loss_dict = self._update_policy(obs_features.detach(), actions)
        
        # Update Targets
        soft_update(self.critic_target, self.critic, self.tau)
        self._update_ema()
        
        self.total_steps += 1
        
        return {"q_loss": q_loss.item(), **policy_loss_dict}
    
    def _update_policy(
        self,
        obs_features: torch.Tensor,
        expert_actions: torch.Tensor
    ) -> Dict[str, float]:
        batch_size = obs_features.shape[0]
        
        # Flow Matching Loss
        x_0 = torch.randn_like(expert_actions)
        t = torch.rand(batch_size, device=self.device)
        x_t = self._linear_interpolate(x_0, expert_actions, t)
        v_target = expert_actions - x_0
        v_pred = self.policy(x_t, t, obs_features=obs_features)
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # Consistency Loss
        t_cons = 0.05 + torch.rand(batch_size, device=self.device) * 0.9
        delta_t = torch.rand(batch_size, device=self.device) * (0.99 - t_cons)
        delta_t = torch.clamp(delta_t, min=0.02)
        t_next = torch.clamp(t_cons + delta_t, max=0.99)
        
        x_t_cons = self._linear_interpolate(x_0, expert_actions, t_cons)
        x_t_next = self._linear_interpolate(x_0, expert_actions, t_next)
        
        with torch.no_grad():
            v_teacher = self.policy_ema(x_t_next, t_next, obs_features=obs_features)
            t_next_expand = t_next.unsqueeze(-1)
            pred_x1 = x_t_next + (1 - t_next_expand) * v_teacher
        
        v_cons_target = pred_x1 - x_0
        v_cons_pred = self.policy(x_t_cons, t_cons, obs_features=obs_features)
        consistency_loss = F.mse_loss(v_cons_pred, v_cons_target)
        
        # Q-Value Maximization
        x_for_q = torch.randn(batch_size, self.action_dim, device=self.device)
        dt = 1.0 / self.num_flow_steps
        
        with torch.no_grad():
            for i in range(self.num_flow_steps - 1):
                t_step = torch.full((batch_size,), i * dt, device=self.device)
                v = self.policy(x_for_q, t_step, obs_features=obs_features)
                x_for_q = x_for_q + v * dt
        
        t_last = torch.full((batch_size,), (self.num_flow_steps - 1) * dt, device=self.device)
        v_last = self.policy(x_for_q, t_last, obs_features=obs_features)
        actions_for_q = x_for_q + v_last * dt
        actions_for_q = torch.clamp(actions_for_q, -1.0, 1.0)
        
        q_value = self.critic.q1_forward(actions_for_q, obs_features=obs_features)
        q_loss = -q_value.mean()
        
        # Total Loss
        total_loss = self.bc_weight * flow_loss + self.consistency_weight * consistency_loss + self.alpha * q_loss
        
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
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False
    ) -> np.ndarray:
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        
        obs_features = self.obs_encoder(state=state, image=image)
        net = self.policy_ema if deterministic else self.policy
        
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        dt = 1.0 / self.num_flow_steps
        
        for i in range(self.num_flow_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            v = net(x, t, obs_features=obs_features)
            x = x + v * dt
        
        action = torch.clamp(x, -1.0, 1.0)
        return action.cpu().numpy().squeeze()
    
    def save(self, path: str):
        checkpoint = {
            "policy": self.policy.state_dict(),
            "policy_ema": self.policy_ema.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "obs_encoder": self.obs_encoder.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "obs_mode": self.obs_mode,
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy_ema.load_state_dict(checkpoint["policy_ema"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.total_steps = checkpoint["total_steps"]
