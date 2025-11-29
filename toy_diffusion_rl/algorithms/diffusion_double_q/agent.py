"""
Diffusion Double Q Learning Agent - Unified Version

Combines Diffusion Policy with Double Q-Learning for offline RL.
Supports multiple observation modes through configurable encoder.

References:
- Diffusion-QL: https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL
- IDQL: https://github.com/philippe-eecs/IDQL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
import copy

try:
    from ...common.networks import MLP, TimestepEmbedding, MultiModalNoisePredictor, MultiModalDoubleQNetwork
    from ...common.utils import DiffusionHelper, soft_update
    from ...common.obs_encoder import ObservationEncoder, create_obs_encoder
except ImportError:
    try:
        from toy_diffusion_rl.common.networks import MLP, TimestepEmbedding, MultiModalNoisePredictor, MultiModalDoubleQNetwork
        from toy_diffusion_rl.common.utils import DiffusionHelper, soft_update
        from toy_diffusion_rl.common.obs_encoder import ObservationEncoder, create_obs_encoder
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from common.networks import MLP, TimestepEmbedding, MultiModalNoisePredictor, MultiModalDoubleQNetwork
        from common.utils import DiffusionHelper, soft_update
        from common.obs_encoder import ObservationEncoder, create_obs_encoder


class DiffusionDoubleQAgent:
    """Unified Diffusion Double Q Agent with multi-modal support.
    
    Combines diffusion-based actor with twin Q-networks.
    Supports state-only, image-only, or state+image observations.
    
    Args:
        action_dim: Dimension of action space
        obs_mode: Observation mode ("state", "image", or "state_image")
        state_dim: Dimension of state space
        image_shape: Image shape as (H, W, C)
        hidden_dims: Hidden layer dimensions
        num_diffusion_steps: Number of diffusion steps
        vision_encoder_type: Type of vision encoder
        vision_output_dim: Output dimension of vision encoder
        freeze_vision_encoder: Whether to freeze vision encoder
        learning_rate_actor: Learning rate for actor
        learning_rate_critic: Learning rate for critics
        gamma: Discount factor
        tau: Soft update coefficient
        alpha: Weight for Q-value term
        bc_weight: Weight for BC loss
        device: Device for computation
    """
    
    def __init__(
        self,
        action_dim: int,
        obs_mode: str = "state",
        state_dim: Optional[int] = None,
        image_shape: Optional[Tuple[int, int, int]] = None,
        hidden_dims: List[int] = [256, 256],
        num_diffusion_steps: int = 10,
        vision_encoder_type: str = "cnn",
        vision_output_dim: int = 128,
        freeze_vision_encoder: bool = False,
        learning_rate_actor: float = 3e-4,
        learning_rate_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 1.0,
        bc_weight: float = 1.0,
        noise_schedule: str = "linear",
        device: str = "cpu"
    ):
        self.action_dim = action_dim
        self.obs_mode = obs_mode
        self.state_dim = state_dim
        self.image_shape = image_shape
        self.num_diffusion_steps = num_diffusion_steps
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.bc_weight = bc_weight
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
        
        # Actor: Diffusion noise predictor
        self.actor = MultiModalNoisePredictor(
            action_dim=action_dim,
            hidden_dims=hidden_dims + [256],
            obs_encoder=self.obs_encoder,
        ).to(device)
        
        # Critics: Double Q-networks
        self.critic = MultiModalDoubleQNetwork(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            obs_encoder=self.obs_encoder,
        ).to(device)
        
        # Target critics
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        # Diffusion helper
        self.diffusion = DiffusionHelper(
            num_diffusion_steps=num_diffusion_steps,
            schedule=noise_schedule,
            device=device
        )
        
        # Collect trainable parameters
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        
        if obs_mode in ["image", "state_image"] and not freeze_vision_encoder:
            vision_params = [p for p in self.obs_encoder.vision_encoder.parameters() if p.requires_grad]
            actor_params.extend(vision_params)
        
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=learning_rate_actor)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=learning_rate_critic)
        
        self.total_steps = 0
    
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
        
        # Update Critics
        with torch.no_grad():
            next_obs_features = self.obs_encoder(state=next_states, image=next_images)
            next_actions = self._sample_actions_batch(next_obs_features)
            
            target_q1, target_q2 = self.critic_target(
                next_actions, obs_features=next_obs_features
            )
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        obs_features = self.obs_encoder(state=states, image=images)
        current_q1, current_q2 = self.critic(actions, obs_features=obs_features)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss_dict = self._update_actor(obs_features.detach(), actions)
        
        # Update Target Networks
        soft_update(self.critic_target, self.critic, self.tau)
        
        self.total_steps += 1
        
        return {"critic_loss": critic_loss.item(), **actor_loss_dict}
    
    def _update_actor(
        self,
        obs_features: torch.Tensor,
        expert_actions: torch.Tensor
    ) -> Dict[str, float]:
        """Update actor with combined diffusion and Q loss."""
        batch_size = obs_features.shape[0]
        
        # Diffusion (BC) Loss
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(expert_actions)
        noisy_actions = self.diffusion.q_sample(expert_actions, t, noise)
        
        t_normalized = t.float() / self.num_diffusion_steps
        predicted_noise = self.actor(noisy_actions, t_normalized, obs_features=obs_features)
        
        bc_loss = F.mse_loss(predicted_noise, noise)
        
        # Q-Value Maximization Loss
        with torch.no_grad():
            generated_actions = self._sample_actions_batch(obs_features)
        
        t_last = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        noise_for_q = torch.randn_like(expert_actions)
        noisy_for_q = self.diffusion.q_sample(generated_actions.detach(), t_last, noise_for_q)
        
        t_normalized_q = t_last.float() / self.num_diffusion_steps
        pred_noise_q = self.actor(noisy_for_q, t_normalized_q, obs_features=obs_features)
        
        alpha_bar = self.diffusion.alphas_bar[t_last].view(-1, 1)
        actions_for_q = (noisy_for_q - torch.sqrt(1 - alpha_bar) * pred_noise_q) / torch.sqrt(alpha_bar)
        actions_for_q = torch.clamp(actions_for_q, -1.0, 1.0)
        
        q_value = self.critic.q1_forward(actions_for_q, obs_features=obs_features)
        q_loss = -q_value.mean()
        
        actor_loss = self.bc_weight * bc_loss + self.alpha * q_loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return {
            "actor_loss": actor_loss.item(),
            "bc_loss": bc_loss.item(),
            "q_loss": q_loss.item(),
            "q_mean": q_value.mean().item()
        }
    
    def _sample_actions_batch(self, obs_features: torch.Tensor) -> torch.Tensor:
        """Sample actions for a batch of observations."""
        batch_size = obs_features.shape[0]
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        steps = min(self.num_diffusion_steps, 5)
        step_indices = list(range(0, self.num_diffusion_steps, self.num_diffusion_steps // steps))[::-1]
        
        for t in step_indices:
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            predicted_noise = self.actor(x, t_normalized, obs_features=obs_features)
            x = self.diffusion.p_sample(x, t_batch, predicted_noise, clip_denoised=True)
        
        return torch.clamp(x, -1.0, 1.0)
    
    @torch.no_grad()
    def sample_action(
        self, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = True
    ) -> np.ndarray:
        """Sample an action for an observation."""
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        
        self.actor.eval()
        
        obs_features = self.obs_encoder(state=state, image=image)
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        steps = self.num_diffusion_steps if deterministic else 5
        step_indices = list(range(0, self.num_diffusion_steps, max(1, self.num_diffusion_steps // steps)))[::-1]
        
        for t in step_indices:
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            predicted_noise = self.actor(x, t_normalized, obs_features=obs_features)
            x = self.diffusion.p_sample(x, t_batch, predicted_noise, clip_denoised=True)
        
        x = torch.clamp(x, -1.0, 1.0)
        return x.cpu().numpy().squeeze()
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "obs_encoder": self.obs_encoder.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "obs_mode": self.obs_mode,
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_steps = checkpoint["total_steps"]
