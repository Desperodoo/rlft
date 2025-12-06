"""
ReinFlow - Unified Online RL Fine-tuning for Flow Matching Policies

Fine-tunes a pretrained flow-matching policy using online RL.
The key insight is to use a NoisyFlowMLP wrapper that learns 
time-dependent exploration noise via a separate noise network.

This unified version supports multiple observation modes:
- "state": State vector only
- "image": Image observation only  
- "state_image": Both state and image (multimodal)

Key features:
- NoisyFlowMLP: Wraps base flow policy with learnable noise injection
- explore_noise_net: Predicts noise_std from time_emb and cond_emb
- ObservationEncoder: Unified handling of different observation modes

References:
- ReinFlow: https://reinflow.github.io/
- Code: https://github.com/ReinFlow/ReinFlow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
import copy

try:
    from ...common.networks import (
        FlowVelocityPredictor, ValueNetwork, MLP,
        MultiModalVelocityPredictor, MultiModalNoisyFlowMLP, MultiModalValueNetwork,
        MultiModalShortCutVelocityPredictor, MultiModalNoisyShortCutFlowMLP
    )
    from ...common.replay_buffer import RolloutBuffer
    from ...common.obs_encoder import ObservationEncoder, create_obs_encoder
except ImportError:
    try:
        from toy_diffusion_rl.common.networks import (
            FlowVelocityPredictor, ValueNetwork, MLP,
            MultiModalVelocityPredictor, MultiModalNoisyFlowMLP, MultiModalValueNetwork,
            MultiModalShortCutVelocityPredictor, MultiModalNoisyShortCutFlowMLP
        )
        from toy_diffusion_rl.common.replay_buffer import RolloutBuffer
        from toy_diffusion_rl.common.obs_encoder import ObservationEncoder, create_obs_encoder
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from common.networks import (
            FlowVelocityPredictor, ValueNetwork, MLP,
            MultiModalVelocityPredictor, MultiModalNoisyFlowMLP, MultiModalValueNetwork,
            MultiModalShortCutVelocityPredictor, MultiModalNoisyShortCutFlowMLP
        )
        from common.replay_buffer import RolloutBuffer
        from common.obs_encoder import ObservationEncoder, create_obs_encoder

# Alias for backwards compatibility
MultiModalFlowVelocityPredictor = MultiModalVelocityPredictor


class ReinFlowAgent:
    """Unified ReinFlow Agent for online RL fine-tuning.
    
    Supports multiple observation modalities through a configurable
    observation encoder. The core flow matching and PPO training logic
    remain the same regardless of observation mode.
    
    Key components:
    1. MultiModalNoisyFlowMLP with learnable exploration noise
    2. ObservationEncoder for unified multimodal handling
    3. Value function for advantage estimation
    4. PPO-style policy gradient updates
    
    Args:
        action_dim: Dimension of action space
        obs_mode: Observation mode ("state", "image", or "state_image")
        state_dim: Dimension of state space (required for state/state_image)
        image_shape: Image shape as (H, W, C) (required for image/state_image)
        hidden_dims: Hidden layer dimensions
        num_flow_steps: Number of flow integration steps
        noise_scheduler_type: Type of noise schedule ('const', 'learn', 'learn_decay')
        vision_encoder_type: Type of vision encoder ("cnn" or "dinov2")
        vision_output_dim: Output dimension of vision encoder
        freeze_vision_encoder: Whether to freeze vision encoder weights
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_ratio: PPO clip ratio
        entropy_coef: Entropy coefficient
        value_coef: Value loss coefficient
        max_grad_norm: Maximum gradient norm
        ppo_epochs: Number of PPO epochs per update
        batch_size: Batch size for updates
        device: Device for computation
    """
    
    def __init__(
        self,
        action_dim: int,
        obs_mode: str = "state",
        state_dim: Optional[int] = None,
        image_shape: Optional[Tuple[int, int, int]] = None,
        hidden_dims: List[int] = [256, 256],
        num_flow_steps: int = 10,
        noise_scheduler_type: str = "learn",
        vision_encoder_type: str = "cnn",
        vision_output_dim: int = 128,
        freeze_vision_encoder: bool = False,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
        use_ema: bool = True,
        ema_decay: float = 0.999,
        use_shortcut: bool = False,
        self_consistency_k: float = 0.25,
        max_denoising_steps: int = 8,
        delta: float = 1e-5
    ):
        self.action_dim = action_dim
        self.obs_mode = obs_mode
        self.state_dim = state_dim
        self.image_shape = image_shape
        self.num_flow_steps = num_flow_steps
        self.noise_scheduler_type = noise_scheduler_type
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device
        
        # ShortCut flow settings
        self.use_shortcut = use_shortcut
        self.self_consistency_k = self_consistency_k
        self.max_denoising_steps = max_denoising_steps
        self.delta = delta
        
        # Create observation encoder
        self.obs_encoder = create_obs_encoder(
            obs_mode=obs_mode,
            state_dim=state_dim,
            image_shape=image_shape,
            vision_encoder_type=vision_encoder_type,
            vision_output_dim=vision_output_dim,
            freeze_vision=freeze_vision_encoder,
        ).to(device)
        
        # Policy with noisy flow (choose ShortCut or standard based on use_shortcut)
        if use_shortcut:
            self.policy = MultiModalNoisyShortCutFlowMLP(
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                noise_scheduler_type=noise_scheduler_type,
                obs_encoder=self.obs_encoder,
            ).to(device)
        else:
            self.policy = MultiModalNoisyFlowMLP(
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                noise_scheduler_type=noise_scheduler_type,
                obs_encoder=self.obs_encoder,
            ).to(device)
        
        # Create EMA network for base_net
        if use_ema:
            self.base_net_ema = copy.deepcopy(self.policy.base_net)
            for param in self.base_net_ema.parameters():
                param.requires_grad = False
        else:
            self.base_net_ema = None
        
        # Value network (shares encoder with policy)
        self.value_net = MultiModalValueNetwork(
            hidden_dims=hidden_dims,
            obs_encoder=self.obs_encoder,
        ).to(device)
        
        # Old policy for PPO (match the policy type)
        if use_shortcut:
            self.old_policy = MultiModalNoisyShortCutFlowMLP(
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                noise_scheduler_type=noise_scheduler_type,
                obs_encoder=self.obs_encoder,
            ).to(device)
        else:
            self.old_policy = MultiModalNoisyFlowMLP(
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                noise_scheduler_type=noise_scheduler_type,
                obs_encoder=self.obs_encoder,
            ).to(device)
        self._sync_old_policy()
        for p in self.old_policy.parameters():
            p.requires_grad = False
        
        # Collect trainable parameters
        trainable_params = [
            {"params": self.policy.base_net.parameters(), "lr": learning_rate},
            {"params": self.policy.time_embed.parameters(), "lr": learning_rate},
            {"params": self.policy.obs_embed.parameters(), "lr": learning_rate},
            {"params": self.policy.explore_noise_net.parameters(), "lr": learning_rate},
            {"params": self.value_net.net.parameters(), "lr": learning_rate},
        ]
        
        # Add vision encoder params if trainable
        if obs_mode in ["image", "state_image"] and not freeze_vision_encoder:
            vision_params = [p for p in self.obs_encoder.vision_encoder.parameters() if p.requires_grad]
            if vision_params:
                trainable_params.append({"params": vision_params, "lr": learning_rate * 0.1})
        
        self.optimizer = torch.optim.Adam(trainable_params)
        
        self.total_steps = 0
    
    def _sync_old_policy(self):
        """Sync old policy with current policy."""
        self.old_policy.base_net.load_state_dict(self.policy.base_net.state_dict())
        self.old_policy.time_embed.load_state_dict(self.policy.time_embed.state_dict())
        self.old_policy.obs_embed.load_state_dict(self.policy.obs_embed.state_dict())
        self.old_policy.explore_noise_net.load_state_dict(self.policy.explore_noise_net.state_dict())
    
    def _update_ema(self):
        """Update EMA network with exponential moving average of base_net weights."""
        if self.base_net_ema is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.base_net_ema.parameters(), self.policy.base_net.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)
    
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
    
    def _integrate_flow(
        self,
        state: Optional[torch.Tensor],
        image: Optional[torch.Tensor],
        num_steps: Optional[int] = None,
        sample_noise: bool = True,
        return_chain: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Integrate the stochastic flow to sample action.
        
        Uses Euler-Maruyama for SDE integration with learned noise.
        Following the ReinFlow paper: track chain for accurate log prob computation.
        
        Args:
            state: State tensor
            image: Image tensor
            num_steps: Number of integration steps
            sample_noise: Whether to sample stochastic noise
            return_chain: Whether to return the full trajectory chain
            
        Returns:
            Tuple of (final_action, log_prob, chain) where chain is optional
        """
        num_steps = num_steps or self.num_flow_steps
        batch_size = state.shape[0] if state is not None else image.shape[0]
        dt = 1.0 / num_steps
        
        # Pre-compute observation features
        obs_features = self.obs_encoder(state=state, image=image)
        
        # Start from Gaussian noise
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Store chain if needed (for accurate log prob computation during update)
        if return_chain:
            chain = torch.zeros(batch_size, num_steps + 1, self.action_dim, device=self.device)
            chain[:, 0] = x
        
        # Track log probability using transition probabilities
        log_prob = torch.zeros(batch_size, device=self.device)
        
        # Initial distribution log prob: N(0, 1)
        log_prob_init = -0.5 * (x ** 2).sum(dim=-1) - 0.5 * self.action_dim * np.log(2 * np.pi)
        log_prob = log_prob + log_prob_init
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            
            # Get velocity and learned noise scale
            if self.use_shortcut:
                # ShortCut model needs step size d
                d = torch.full((batch_size,), dt, device=self.device)
                velocity, noise, noise_std = self.policy(
                    x, t, d, obs_features=obs_features, sample_noise=sample_noise
                )
            else:
                velocity, noise, noise_std = self.policy(
                    x, t, obs_features=obs_features, sample_noise=sample_noise
                )
            
            # Deterministic update
            x_mean = x + velocity * dt
            
            # Stochastic update with noise
            if sample_noise:
                # Clamp noise_std for stability
                noise_std_clamped = torch.clamp(noise_std, min=0.01, max=0.5)
                x_new = x_mean + noise_std_clamped * noise
                
                # Log prob of transition: N(x_new | x_mean, noise_std^2)
                # This is: -0.5 * ((x_new - x_mean) / noise_std)^2 - log(noise_std) - 0.5*log(2*pi)
                log_prob_trans = -0.5 * ((noise) ** 2).sum(dim=-1)
                log_prob_trans = log_prob_trans - noise_std_clamped.log().sum(dim=-1)
                log_prob_trans = log_prob_trans - 0.5 * self.action_dim * np.log(2 * np.pi)
                log_prob = log_prob + log_prob_trans
            else:
                x_new = x_mean
            
            x = x_new
            
            if return_chain:
                chain[:, i + 1] = x
        
        action = torch.clamp(x, -1.0, 1.0)
        
        # Normalize log_prob by number of steps for stability
        log_prob = log_prob / (num_steps + 1)
        
        if return_chain:
            return action, log_prob, chain
        return action, log_prob, None
    
    def _compute_log_prob(
        self,
        state: Optional[torch.Tensor],
        image: Optional[torch.Tensor],
        action: torch.Tensor,
        use_old_policy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probability of action under policy.
        
        Uses inverse flow approach: given action (final x), run backward pass to 
        estimate the initial x0, then compute the log prob of the trajectory.
        
        This is an approximation - the exact approach would require saving the chain.
        """
        batch_size = action.shape[0]
        
        # Pre-compute observation features
        obs_features = self.obs_encoder(state=state, image=image)
        
        policy = self.old_policy if use_old_policy else self.policy
        
        # Forward flow from fixed noise to get "expected" action trajectory
        # Use action's mean as reference - run flow deterministically
        dt = 1.0 / self.num_flow_steps
        
        # Key insight: For PPO, we care about the CHANGE in log_prob, not absolute value
        # Use a local Gaussian approximation around the deterministic flow path
        
        # Run deterministic flow to get expected trajectory
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Compute accumulated noise scale for local approximation
        total_noise_var = torch.zeros(batch_size, self.action_dim, device=self.device)
        
        for i in range(self.num_flow_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            
            if self.use_shortcut:
                d = torch.full((batch_size,), dt, device=self.device)
                velocity, _, noise_std = policy(x, t, d, obs_features=obs_features, sample_noise=False)
            else:
                velocity, _, noise_std = policy(x, t, obs_features=obs_features, sample_noise=False)
            
            # Accumulate variance from stochastic steps
            # Each step contributes noise_std^2 to total variance
            noise_std_clamped = torch.clamp(noise_std, min=0.01, max=0.3)
            total_noise_var = total_noise_var + noise_std_clamped ** 2
            
            # Deterministic step
            x = x + velocity * dt
        
        mean_action = torch.clamp(x, -1.0, 1.0)
        
        # Total std is sqrt of accumulated variance
        total_std = torch.sqrt(total_noise_var + 1e-6)
        total_std = torch.clamp(total_std, min=0.05, max=0.8)
        
        # Compute log prob using local Gaussian approximation
        # This captures how "likely" the action is under the current policy
        diff = action - mean_action
        log_prob = -0.5 * (diff / total_std) ** 2 - torch.log(total_std)
        log_prob = log_prob - 0.5 * np.log(2 * np.pi)
        log_prob = log_prob.sum(dim=-1)
        
        # Clamp for stability
        log_prob = torch.clamp(log_prob, min=-20.0, max=5.0)
        
        # Entropy based on accumulated noise
        entropy = 0.5 * (1 + np.log(2 * np.pi)) + torch.log(total_std)
        entropy = entropy.sum(dim=-1)
        
        return log_prob, entropy
    
    @torch.no_grad()
    def sample_action_pretrain(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Sample action using only base velocity network (deterministic ODE).
        
        Use for evaluation after pretrain but before fine-tuning.
        Uses EMA network if available for more stable inference.
        """
        state, image = self._parse_observation(obs)
        batch_size = state.shape[0] if state is not None else image.shape[0]
        
        obs_features = self.obs_encoder(state=state, image=image)
        
        # Use EMA network if available, otherwise use base_net
        velocity_net = self.base_net_ema if self.base_net_ema is not None else self.policy.base_net
        
        dt = 1.0 / self.num_flow_steps
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for i in range(self.num_flow_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            if self.use_shortcut:
                d = torch.full((batch_size,), dt, device=self.device)
                velocity = velocity_net(x, t, d, obs_features=obs_features)
            else:
                velocity = velocity_net(x, t, obs_features=obs_features)
            x = x + velocity * dt
        
        action = torch.clamp(x, -1.0, 1.0)
        return action.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def sample_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Sample action and return log prob and value.
        
        Args:
            obs: Observation
            deterministic: If True, don't sample noise
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state, image = self._parse_observation(obs)
        
        # Sample from stochastic flow (no need to return chain for inference)
        action, log_prob, _ = self._integrate_flow(
            state, image, sample_noise=not deterministic, return_chain=False
        )
        
        # Get value
        obs_features = self.obs_encoder(state=state, image=image)
        value = self.value_net(obs_features=obs_features)
        
        return (
            action.cpu().numpy().squeeze(),
            log_prob.cpu().numpy().squeeze(),
            value.cpu().numpy().squeeze()
        )
    
    def collect_rollout(
        self,
        env,
        rollout_steps: int = 2048
    ) -> Dict[str, np.ndarray]:
        """Collect experience for online update.
        
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
        
        # Compute last value
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
            buffer: Rollout buffer
            
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
                
                # Compute new log probs
                new_log_probs, entropy = self._compute_log_prob(
                    batch_states, batch_images, batch_actions, use_old_policy=False
                )
                
                # PPO ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                obs_features = self.obs_encoder(state=batch_states, image=batch_images)
                values = self.value_net(obs_features=obs_features).squeeze()
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        self.total_steps += n_samples
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates
        }
    
    def pretrain_bc_step(
        self,
        actions: torch.Tensor,
        obs_features: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, float]:
        """Single step of BC pretraining with ShortCut flow support.
        
        This method is designed for external training loops that need
        intermediate evaluation. Use this when you need fine-grained control
        over the training process.
        
        Args:
            actions: Target actions (B, action_dim)
            obs_features: Pre-computed observation features from obs_encoder
            optimizer: External optimizer for updating parameters
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Dictionary with step metrics:
            - 'loss': Total loss value
            - 'fm_loss': Flow matching loss (if use_shortcut)
            - 'sc_loss': Self-consistency loss (if use_shortcut)
        """
        if self.use_shortcut:
            # ShortCut flow with self-consistency loss
            loss, fm_loss, sc_loss = self._compute_shortcut_loss(
                actions, obs_features
            )
            result = {
                'loss': loss.item(),
                'fm_loss': fm_loss,
                'sc_loss': sc_loss
            }
        else:
            # Standard flow matching loss (no sigma_min)
            batch_size = actions.shape[0]
            x_0 = torch.randn_like(actions)
            t = torch.rand(batch_size, device=self.device)
            
            t_expand = t.unsqueeze(-1)
            x_t = (1 - t_expand) * x_0 + t_expand * actions
            target_v = actions - x_0
            
            pred_v = self.policy.base_net(x_t, t, obs_features=obs_features)
            loss = F.mse_loss(pred_v, target_v)
            result = {'loss': loss.item()}
        
        optimizer.zero_grad()
        loss.backward()
        
        # Get trainable params for gradient clipping
        bc_params = list(self.policy.base_net.parameters())
        if self.obs_mode in ["image", "state_image"] and self.obs_encoder.vision_encoder is not None:
            vision_params = [p for p in self.obs_encoder.vision_encoder.parameters() if p.requires_grad]
            bc_params.extend(vision_params)
        
        torch.nn.utils.clip_grad_norm_(bc_params, max_grad_norm)
        optimizer.step()
        
        # Update EMA
        self._update_ema()
        
        return result
    
    def get_pretrain_parameters(self) -> list:
        """Get parameters for BC pretraining optimizer.
        
        Returns:
            List of parameters to optimize during pretraining
        """
        bc_params = list(self.policy.base_net.parameters())
        
        # Add vision encoder params if trainable
        if self.obs_mode in ["image", "state_image"] and self.obs_encoder.vision_encoder is not None:
            vision_params = [p for p in self.obs_encoder.vision_encoder.parameters() if p.requires_grad]
            bc_params.extend(vision_params)
        
        return bc_params
    
    def pretrain_bc(
        self,
        dataloader,
        num_steps: int = 1000,
        learning_rate: float = 3e-4,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, float]:
        """Pretrain with behavior cloning using flow matching objective.
        
        If use_shortcut is True, uses ShortCut flow with self-consistency loss:
        - 25% of batch: self-consistency loss (two-step velocity average target)
        - 75% of batch: standard flow matching loss
        
        Args:
            dataloader: PyTorch DataLoader yielding batches
            num_steps: Number of BC steps
            learning_rate: Learning rate for BC optimizer
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Dictionary with BC metrics
        """
        bc_optimizer = torch.optim.Adam(self.get_pretrain_parameters(), lr=learning_rate)
        
        total_loss = 0
        total_fm_loss = 0
        total_sc_loss = 0
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
            
            # Compute observation features
            obs_features = self.obs_encoder(state=states, image=images)
            
            # Use pretrain_bc_step for single step training
            step_metrics = self.pretrain_bc_step(
                actions=actions,
                obs_features=obs_features,
                optimizer=bc_optimizer,
                max_grad_norm=max_grad_norm
            )
            
            total_loss += step_metrics['loss']
            if 'fm_loss' in step_metrics:
                total_fm_loss += step_metrics['fm_loss']
            if 'sc_loss' in step_metrics:
                total_sc_loss += step_metrics['sc_loss']
            
            step += 1
        
        self._sync_old_policy()
        
        result = {"bc_loss": total_loss / num_steps}
        if self.use_shortcut:
            result["fm_loss"] = total_fm_loss / num_steps
            result["sc_loss"] = total_sc_loss / num_steps
        return result
    
    def _compute_shortcut_loss(
        self,
        actions: torch.Tensor,
        obs_features: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute ShortCut flow loss with self-consistency.
        
        Following ReinFlow's ShortCutFlow:
        - 25% of batch uses self-consistency loss (two-step velocity average as target)
        - 75% of batch uses standard flow matching loss
        
        Args:
            actions: Target actions (B, action_dim), i.e., x_1
            obs_features: Pre-computed observation features
            
        Returns:
            Tuple of (total_loss, fm_loss_value, sc_loss_value)
        """
        B = actions.shape[0]
        k_num = int(B * self.self_consistency_k)  # Number for self-consistency
        
        # ============================================================
        # Self-consistency part (first k_num samples)
        # ============================================================
        x_sc_1 = actions[:k_num]
        obs_sc = obs_features[:k_num]
        
        # Sample step size: d = 1 / 2^d_base, where d_base in [0, max_denoising_steps-1)
        d_base_sc = torch.randint(0, self.max_denoising_steps - 1, (k_num,), device=self.device)
        d_sc = 1.0 / (2 ** d_base_sc.float())  # Step size for prediction
        d_bootstrap_sc = d_sc / 2  # Half step size for bootstrap
        dt_sections_sc = 2 ** d_base_sc  # Number of time sections
        
        # Sample time t from discrete grid compatible with step size
        t_idx_sc = torch.cat([
            torch.randint(0, max(1, int(dt_sections_sc[i])), (1,), device=self.device) 
            for i in range(k_num)
        ], dim=0)
        t_sc = t_idx_sc.float() / dt_sections_sc.float()
        
        # Sample initial noise
        x0_sc = torch.randn_like(x_sc_1)
        
        # Interpolate to get x_t
        t_expand_sc = t_sc.unsqueeze(-1)
        x_t_sc = (1 - (1 - self.delta) * t_expand_sc) * x0_sc + t_expand_sc * x_sc_1
        
        # Two-step velocity averaging (self-consistency target)
        with torch.no_grad():
            # First step: velocity at (x_t, t) with half step size
            v_t_sc = self.policy.base_net(x_t_sc, t_sc, d_bootstrap_sc, obs_features=obs_sc)
            
            # Advance by d_bootstrap
            d_bootstrap_expand = d_bootstrap_sc.unsqueeze(-1)
            x_t2_sc = x_t_sc + v_t_sc * d_bootstrap_expand
            t2_sc = t_sc + d_bootstrap_sc
            
            # Second step: velocity at (x_t2, t2) with half step size
            v_t2_sc = self.policy.base_net(x_t2_sc, t2_sc, d_bootstrap_sc, obs_features=obs_sc)
            
            # Target: average of two velocities (self-consistency)
            v_target_sc = (v_t_sc + v_t2_sc) / 2
        
        # Predict velocity with full step size d_sc
        v_pred_sc = self.policy.base_net(x_t_sc, t_sc, d_sc, obs_features=obs_sc)
        sc_loss = F.mse_loss(v_pred_sc, v_target_sc)
        
        # ============================================================
        # Flow matching part (remaining B - k_num samples)
        # ============================================================
        x_fm_1 = actions[k_num:]
        obs_fm = obs_features[k_num:]
        fm_batch_size = B - k_num
        
        # Standard flow matching: d=0 means no self-consistency
        t_fm = torch.rand(fm_batch_size, device=self.device)
        x0_fm = torch.randn_like(x_fm_1)
        
        t_expand_fm = t_fm.unsqueeze(-1)
        x_t_fm = (1 - (1 - self.delta) * t_expand_fm) * x0_fm + t_expand_fm * x_fm_1
        v_target_fm = x_fm_1 - (1 - self.delta) * x0_fm
        
        # For flow matching, d=0 (or very small)
        d_fm = torch.zeros(fm_batch_size, device=self.device)
        
        v_pred_fm = self.policy.base_net(x_t_fm, t_fm, d_fm, obs_features=obs_fm)
        fm_loss = F.mse_loss(v_pred_fm, v_target_fm)
        
        # Combined loss
        total_loss = sc_loss + fm_loss
        
        return total_loss, fm_loss.item(), sc_loss.item()
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "obs_encoder": self.obs_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "obs_mode": self.obs_mode,
            "noise_scheduler_type": self.noise_scheduler_type,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "use_shortcut": self.use_shortcut,
            "config": {
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "image_shape": self.image_shape,
                "num_flow_steps": self.num_flow_steps,
                "self_consistency_k": self.self_consistency_k,
                "max_denoising_steps": self.max_denoising_steps,
                "delta": self.delta,
            },
        }
        if self.base_net_ema is not None:
            checkpoint["base_net_ema"] = self.base_net_ema.state_dict()
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
        # Load EMA if available
        if "base_net_ema" in checkpoint and self.base_net_ema is not None:
            self.base_net_ema.load_state_dict(checkpoint["base_net_ema"])
        self._sync_old_policy()
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained flow matching weights.
        
        Args:
            checkpoint_path: Path to pretrained checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "policy" in checkpoint:
            state_dict = checkpoint["policy"]
        elif "velocity_net" in checkpoint:
            state_dict = checkpoint["velocity_net"]
        else:
            state_dict = checkpoint
        
        # Load into base network
        self.policy.base_net.load_state_dict(state_dict, strict=False)
        self._sync_old_policy()
