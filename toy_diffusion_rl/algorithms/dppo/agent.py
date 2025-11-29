"""
DPPO - Diffusion Policy Policy Optimization

Online RL fine-tuning of diffusion policies using PPO-style updates.
Treats the diffusion denoising process as a sequence of actions in
an augmented MDP.

Key feature: Partial chain fine-tuning - only the last K denoising steps
are trainable, while earlier steps remain frozen (from pretrained policy).
This improves stability and sample efficiency.

Supports multiple observation modes: state-only, image-only, or state+image.

References:
- DPPO: https://diffusion-ppo.github.io/
- Official code: https://github.com/irom-princeton/dppo
- PPO: Proximal Policy Optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
import copy

try:
    from ...common.networks import (
        DiffusionNoisePredictor, ValueNetwork, TimestepEmbedding, MLP,
        MultiModalNoisePredictor, MultiModalValueNetwork
    )
    from ...common.utils import DiffusionHelper
    from ...common.replay_buffer import RolloutBuffer
    from ...common.obs_encoder import ObservationEncoder, create_obs_encoder
except ImportError:
    try:
        from toy_diffusion_rl.common.networks import (
            DiffusionNoisePredictor, ValueNetwork, TimestepEmbedding, MLP,
            MultiModalNoisePredictor, MultiModalValueNetwork
        )
        from toy_diffusion_rl.common.utils import DiffusionHelper
        from toy_diffusion_rl.common.replay_buffer import RolloutBuffer
        from toy_diffusion_rl.common.obs_encoder import ObservationEncoder, create_obs_encoder
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from common.networks import (
            DiffusionNoisePredictor, ValueNetwork, TimestepEmbedding, MLP,
            MultiModalNoisePredictor, MultiModalValueNetwork
        )
        from common.utils import DiffusionHelper
        from common.replay_buffer import RolloutBuffer
        from common.obs_encoder import ObservationEncoder, create_obs_encoder

# Alias for backwards compatibility
MultiModalDiffusionActor = MultiModalNoisePredictor


class DPPOAgent:
    """Unified DPPO Agent supporting multiple observation modalities.
    
    This agent handles state-only, image-only, or state+image observations
    through a configurable observation encoder. The core diffusion policy
    and PPO training logic remain the same regardless of observation mode.
    
    Key features:
    1. Partial chain fine-tuning: only last K denoising steps are trainable
    2. Multi-modal observations: state, image, or state+image
    3. Shared vision encoder between actor and critic
    
    Args:
        action_dim: Dimension of action space
        obs_mode: Observation mode ("state", "image", or "state_image")
        state_dim: Dimension of state space (required for state/state_image modes)
        image_shape: Image shape as (H, W, C) (required for image/state_image modes)
        hidden_dims: Hidden layer dimensions
        num_diffusion_steps: Number of diffusion steps
        ft_denoising_steps: Number of denoising steps to fine-tune (from the end)
        vision_encoder_type: Type of vision encoder ("cnn" or "dinov2")
        vision_output_dim: Output dimension of vision encoder
        freeze_vision_encoder: Whether to freeze vision encoder weights
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_ratio: PPO clip ratio
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm
        ppo_epochs: Number of PPO epochs per update
        batch_size: Batch size for PPO updates
        noise_schedule: Diffusion noise schedule ("linear" or "cosine")
        device: Device for computation
    """
    
    def __init__(
        self,
        action_dim: int,
        obs_mode: str = "state",
        state_dim: Optional[int] = None,
        image_shape: Optional[Tuple[int, int, int]] = None,
        hidden_dims: List[int] = [256, 256],
        num_diffusion_steps: int = 5,
        ft_denoising_steps: int = 3,
        vision_encoder_type: str = "cnn",
        vision_output_dim: int = 128,
        freeze_vision_encoder: bool = False,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        noise_schedule: str = "linear",
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.obs_mode = obs_mode
        self.state_dim = state_dim
        self.image_shape = image_shape
        self.num_diffusion_steps = num_diffusion_steps
        self.ft_denoising_steps = min(ft_denoising_steps, num_diffusion_steps)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
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
        
        obs_feature_dim = self.obs_encoder.output_dim
        
        # Base policy (frozen, for early denoising steps)
        self.actor = MultiModalDiffusionActor(
            action_dim=action_dim,
            hidden_dims=hidden_dims + [256],
            obs_encoder=self.obs_encoder,
        ).to(device)
        for p in self.actor.net.parameters():
            p.requires_grad = False
        for p in self.actor.time_embed.parameters():
            p.requires_grad = False
        
        # Fine-tuned policy (trainable, for last K denoising steps)
        self.actor_ft = MultiModalDiffusionActor(
            action_dim=action_dim,
            hidden_dims=hidden_dims + [256],
            obs_encoder=self.obs_encoder,  # Share encoder
        ).to(device)
        
        # Value function
        self.value_net = MultiModalValueNetwork(
            hidden_dims=hidden_dims,
            obs_encoder=self.obs_encoder,  # Share encoder
        ).to(device)
        
        # Diffusion helper
        self.diffusion = DiffusionHelper(
            num_diffusion_steps=num_diffusion_steps,
            schedule=noise_schedule,
            device=device,
        )
        
        # Collect trainable parameters
        trainable_params = [
            {"params": self.actor_ft.net.parameters(), "lr": learning_rate},
            {"params": self.actor_ft.time_embed.parameters(), "lr": learning_rate},
            {"params": self.value_net.net.parameters(), "lr": learning_rate},
        ]
        
        # Add vision encoder params if trainable
        if obs_mode in ["image", "state_image"] and not freeze_vision_encoder:
            vision_params = [p for p in self.obs_encoder.vision_encoder.parameters() if p.requires_grad]
            if vision_params:
                trainable_params.append({"params": vision_params, "lr": learning_rate * 0.1})
        
        self.optimizer = torch.optim.Adam(trainable_params)
        
        # Old policy for PPO ratio computation
        self.old_actor_ft = MultiModalDiffusionActor(
            action_dim=action_dim,
            hidden_dims=hidden_dims + [256],
            obs_encoder=self.obs_encoder,
        ).to(device)
        self._sync_old_policy()
        for p in self.old_actor_ft.parameters():
            p.requires_grad = False
        
        # Noise std for stochastic denoising
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=device))
        
        self.total_steps = 0
    
    def _sync_old_policy(self):
        """Sync old policy with current policy."""
        self.old_actor_ft.net.load_state_dict(self.actor_ft.net.state_dict())
        self.old_actor_ft.time_embed.load_state_dict(self.actor_ft.time_embed.state_dict())
    
    def _get_actor_for_step(self, t: int) -> MultiModalDiffusionActor:
        """Get appropriate actor for timestep.
        
        Fine-tuned actor is used for the last ft_denoising_steps steps.
        """
        if t < self.ft_denoising_steps:
            return self.actor_ft
        else:
            return self.actor
    
    def _parse_observation(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray], torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Parse observation into state and image tensors.
        
        Args:
            obs: Observation (array, dict, or tensor)
        
        Returns:
            Tuple of (state_tensor, image_tensor)
        """
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
    
    def _compute_action_log_prob(
        self,
        state: Optional[torch.Tensor],
        image: Optional[torch.Tensor],
        action: torch.Tensor,
        use_old_policy: bool = False,
        return_entropy: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute log probability of action under diffusion policy.
        
        Only computes log prob for fine-tuned denoising steps.
        """
        batch_size = action.shape[0]
        device = action.device
        
        total_log_prob = torch.zeros(batch_size, device=device)
        total_entropy = torch.zeros(batch_size, device=device)
        
        std = torch.exp(self.log_std).clamp(min=0.01, max=1.0)
        
        # Pre-compute observation features (shared across timesteps)
        with torch.no_grad() if use_old_policy else torch.enable_grad():
            obs_features = self.obs_encoder(state=state, image=image)
        
        for t in range(self.ft_denoising_steps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            # Get noisy action at time t
            a_t = self.diffusion.q_sample(action, t_batch, noise=torch.randn_like(action))
            
            # Predict noise
            if use_old_policy:
                pred_noise = self.old_actor_ft(a_t, t_normalized, obs_features=obs_features)
            else:
                pred_noise = self.actor_ft(a_t, t_normalized, obs_features=obs_features)
            
            # Compute predicted clean action
            alpha_bar = self.diffusion.alphas_bar[t]
            pred_a0 = (a_t - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            pred_a0 = torch.clamp(pred_a0, -1.0, 1.0)
            
            # Compute log probability
            if t == 0:
                dist = torch.distributions.Normal(pred_a0, std)
                log_prob = dist.log_prob(action).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
            else:
                scale = std * np.sqrt(t / self.num_diffusion_steps)
                dist = torch.distributions.Normal(pred_a0, scale)
                log_prob = dist.log_prob(action).sum(dim=-1) * 0.1
                entropy = dist.entropy().sum(dim=-1)
            
            total_log_prob = total_log_prob + log_prob
            total_entropy = total_entropy + entropy
        
        if return_entropy:
            return total_log_prob, total_entropy / self.ft_denoising_steps
        return total_log_prob, None
    
    @torch.no_grad()
    def sample_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """Sample action and return log prob and value.
        
        Uses partial chain: frozen actor for early steps, trainable for final steps.
        
        Args:
            obs: Observation
            deterministic: If True, don't add exploration noise
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        
        # Pre-compute observation features
        obs_features = self.obs_encoder(state=state, image=image)
        
        # Diffusion sampling with partial chain
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for t in reversed(range(self.num_diffusion_steps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            actor = self._get_actor_for_step(t)
            pred_noise = actor(x, t_normalized, obs_features=obs_features)
            x = self.diffusion.p_sample(x, t_batch, pred_noise, clip_denoised=True)
            
            # Add exploration noise for fine-tuned steps
            if not deterministic and 0 < t < self.ft_denoising_steps:
                noise_scale = torch.exp(self.log_std) * 0.1
                x = x + noise_scale * torch.randn_like(x)
        
        action = torch.clamp(x, -1.0, 1.0)
        
        # Compute log prob and value
        with torch.enable_grad():
            log_prob, _ = self._compute_action_log_prob(
                state, image, action, use_old_policy=False
            )
        
        value = self.value_net(obs_features=obs_features)
        
        return (
            action.cpu().numpy().squeeze(),
            log_prob.detach().cpu().numpy().squeeze(),
            value.cpu().numpy().squeeze(),
        )
    
    @torch.no_grad()
    def sample_action_pretrain(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = True,
    ) -> np.ndarray:
        """Sample action using only pretrained actor (no partial chain).
        
        Use this for evaluation after pretrain but before fine-tuning.
        """
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        
        obs_features = self.obs_encoder(state=state, image=image)
        
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for t in reversed(range(self.num_diffusion_steps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            # Only use frozen pretrained actor
            pred_noise = self.actor(x, t_normalized, obs_features=obs_features)
            x = self.diffusion.p_sample(x, t_batch, pred_noise, clip_denoised=True)
        
        action = torch.clamp(x, -1.0, 1.0)
        return action.cpu().numpy().squeeze()
    
    def collect_rollout(
        self,
        env,
        rollout_steps: int = 2048,
    ) -> Dict[str, np.ndarray]:
        """Collect rollout for PPO update.
        
        Args:
            env: Gymnasium environment
            rollout_steps: Number of steps to collect
        
        Returns:
            Buffer dictionary with collected experience
        """
        states_list = []
        images_list = []
        actions_list = []
        rewards_list = []
        values_list = []
        log_probs_list = []
        dones_list = []
        
        obs, _ = env.reset()
        
        for _ in range(rollout_steps):
            action, log_prob, value = self.sample_action(obs)
            
            # Store observation
            if self.obs_mode == "state":
                states_list.append(obs if isinstance(obs, np.ndarray) else obs)
            elif self.obs_mode == "image":
                images_list.append(obs if isinstance(obs, np.ndarray) else obs)
            else:
                states_list.append(obs["state"])
                images_list.append(obs["image"])
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            actions_list.append(action)
            rewards_list.append(reward)
            values_list.append(value)
            log_probs_list.append(log_prob)
            dones_list.append(done)
            
            obs = next_obs if not done else env.reset()[0]
        
        # Compute last value for GAE
        with torch.no_grad():
            _, _, last_value = self.sample_action(obs)
        
        buffer = {
            "actions": np.array(actions_list),
            "rewards": np.array(rewards_list),
            "values": np.array(values_list),
            "log_probs": np.array(log_probs_list),
            "dones": np.array(dones_list),
            "last_value": last_value,
        }
        
        if states_list:
            buffer["states"] = np.array(states_list)
        if images_list:
            buffer["images"] = np.array(images_list)
        
        return self._compute_gae(buffer)
    
    def _compute_gae(self, buffer: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute GAE advantages and returns."""
        rewards = buffer["rewards"]
        values = buffer["values"]
        dones = buffer["dones"]
        last_value = buffer["last_value"]
        
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        buffer["advantages"] = advantages
        buffer["returns"] = advantages + values
        
        return buffer
    
    def update(self, buffer: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update policy using PPO.
        
        Args:
            buffer: Rollout buffer from collect_rollout
        
        Returns:
            Dictionary of training metrics
        """
        self._sync_old_policy()
        
        # Prepare tensors
        states = None
        images = None
        
        if "states" in buffer:
            states = torch.FloatTensor(buffer["states"]).to(self.device)
        if "images" in buffer:
            images = torch.FloatTensor(buffer["images"]).to(self.device)
            if images.dim() == 4 and images.shape[-1] in [1, 3, 4]:
                images = images.permute(0, 3, 1, 2)
            if images.max() > 1.0:
                images = images / 255.0
        
        actions = torch.FloatTensor(buffer["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(buffer["log_probs"]).to(self.device)
        advantages = torch.FloatTensor(buffer["advantages"]).to(self.device)
        returns = torch.FloatTensor(buffer["returns"]).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        n_samples = len(actions)
        
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx] if states is not None else None
                batch_images = images[batch_idx] if images is not None else None
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Compute new log probs and entropy
                new_log_probs, entropy = self._compute_action_log_prob(
                    batch_states, batch_images, batch_actions,
                    use_old_policy=False, return_entropy=True,
                )
                
                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                obs_features = self.obs_encoder(state=batch_states, image=batch_images)
                values = self.value_net(obs_features=obs_features).squeeze()
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                all_params = list(self.actor_ft.net.parameters()) + list(self.value_net.net.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        self.total_steps += n_samples
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "ft_steps": self.ft_denoising_steps,
        }
    
    def pretrain_bc(
        self,
        dataloader,
        num_steps: int = 1000,
    ) -> Dict[str, float]:
        """Pretrain with behavior cloning.
        
        Args:
            dataloader: PyTorch DataLoader yielding batches
            num_steps: Number of BC steps
        
        Returns:
            Dictionary with BC metrics
        """
        # Enable gradients for frozen actor during pretraining
        for p in self.actor.net.parameters():
            p.requires_grad = True
        for p in self.actor.time_embed.parameters():
            p.requires_grad = True
        
        params = (
            list(self.actor.net.parameters()) +
            list(self.actor.time_embed.parameters()) +
            list(self.actor_ft.net.parameters()) +
            list(self.actor_ft.time_embed.parameters())
        )
        
        # Add vision encoder params if trainable
        if self.obs_mode in ["image", "state_image"] and self.obs_encoder.vision_encoder is not None:
            vision_params = [p for p in self.obs_encoder.vision_encoder.parameters() if p.requires_grad]
            params.extend(vision_params)
        
        bc_optimizer = torch.optim.Adam(params, lr=1e-4)
        
        total_loss = 0
        step = 0
        data_iter = iter(dataloader)
        
        while step < num_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Parse batch
            states = batch.get("obs", batch.get("state"))
            if states is not None:
                states = states.to(self.device)
            
            images = batch.get("image")
            if images is not None:
                images = images.to(self.device)
            
            actions = batch["action"].to(self.device)
            batch_size = actions.shape[0]
            
            # Compute observation features
            obs_features = self.obs_encoder(state=states, image=images)
            
            # Sample timesteps and add noise
            t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
            noise = torch.randn_like(actions)
            noisy_actions = self.diffusion.q_sample(actions, t, noise)
            t_normalized = t.float() / self.num_diffusion_steps
            
            # Predict noise with both actors
            pred_noise_frozen = self.actor(noisy_actions, t_normalized, obs_features=obs_features)
            pred_noise_ft = self.actor_ft(noisy_actions, t_normalized, obs_features=obs_features)
            
            loss = F.mse_loss(pred_noise_frozen, noise) + F.mse_loss(pred_noise_ft, noise)
            
            bc_optimizer.zero_grad()
            loss.backward()
            bc_optimizer.step()
            
            total_loss += loss.item()
            step += 1
        
        # Freeze base actor after pretraining
        for p in self.actor.net.parameters():
            p.requires_grad = False
        for p in self.actor.time_embed.parameters():
            p.requires_grad = False
        
        self._sync_old_policy()
        
        return {"bc_loss": total_loss / num_steps}
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "actor": self.actor.state_dict(),
            "actor_ft": self.actor_ft.state_dict(),
            "value_net": self.value_net.state_dict(),
            "obs_encoder": self.obs_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "log_std": self.log_std,
            "total_steps": self.total_steps,
            "obs_mode": self.obs_mode,
            "config": {
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "image_shape": self.image_shape,
                "num_diffusion_steps": self.num_diffusion_steps,
                "ft_denoising_steps": self.ft_denoising_steps,
            },
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_ft.load_state_dict(checkpoint["actor_ft"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.log_std.data = checkpoint["log_std"]
        self.total_steps = checkpoint["total_steps"]
        self._sync_old_policy()
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights into both frozen and trainable actors.
        
        Args:
            checkpoint_path: Path to pretrained diffusion policy checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Try different key names
        if "policy" in checkpoint:
            weights = checkpoint["policy"]
        elif "actor" in checkpoint:
            weights = checkpoint["actor"]
        else:
            weights = checkpoint
        
        # Load into frozen actor
        self.actor.load_state_dict(weights, strict=False)
        
        # Initialize trainable actor from pretrained
        self.actor_ft.load_state_dict(self.actor.state_dict())
        self._sync_old_policy()
