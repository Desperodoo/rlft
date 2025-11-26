"""
ReinFlow - Online RL Fine-tuning for Flow Matching Policies

Fine-tunes a pretrained flow-matching policy using online RL.
The key insight is to use a NoisyFlowMLP wrapper that learns 
time-dependent exploration noise via a separate noise network.

Key features:
- NoisyFlowMLP: Wraps base flow policy with learnable noise injection
- explore_noise_net: Predicts noise_std from time_emb and cond_emb
- Supports different noise scheduler types ('const', 'learn', 'learn_decay')

References:
- ReinFlow: https://reinflow.github.io/
- Code: https://github.com/ReinFlow/ReinFlow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import copy

from ...common.networks import FlowVelocityPredictor, ValueNetwork
from ...common.replay_buffer import RolloutBuffer


class NoisyFlowMLP(nn.Module):
    """Noisy Flow MLP wrapper for RL fine-tuning.
    
    Wraps a base flow velocity network with learnable noise injection.
    The noise is predicted by a separate network based on time and state embeddings.
    
    Architecture:
    - base_net: Pretrained flow velocity predictor (can be frozen)
    - explore_noise_net: Learns to predict time-dependent noise std
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        noise_scheduler_type: Type of noise schedule ('const', 'learn', 'learn_decay')
        init_noise_std: Initial noise standard deviation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256, 256],
        noise_scheduler_type: str = "learn",
        init_noise_std: float = 0.1
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.noise_scheduler_type = noise_scheduler_type
        self.init_noise_std = init_noise_std
        
        # Base velocity network (deterministic flow)
        self.base_net = FlowVelocityPredictor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        )
        
        # Time embedding network (shared for noise prediction)
        self.time_embed_dim = 64
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # State embedding for noise prediction
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # Exploration noise prediction network
        # Input: time_embed + state_embed
        # Output: noise_std (per action dimension)
        self.explore_noise_net = nn.Sequential(
            nn.Linear(self.time_embed_dim * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
            nn.Softplus()  # Ensure positive noise std
        )
        
        # Initialize noise network to output init_noise_std
        self._init_noise_net()
        
        # Learnable decay factor for 'learn_decay' mode
        if noise_scheduler_type == "learn_decay":
            self.log_decay = nn.Parameter(torch.zeros(1))
            
        # Constant noise std for 'const' mode
        if noise_scheduler_type == "const":
            self.register_buffer("const_noise_std", torch.ones(action_dim) * init_noise_std)
            
    def _init_noise_net(self):
        """Initialize noise network to output init_noise_std."""
        # Initialize last layer bias to achieve init_noise_std
        # Softplus(x) = log(1 + exp(x)), so x = log(exp(std) - 1)
        init_bias = np.log(np.exp(self.init_noise_std) - 1)
        nn.init.constant_(self.explore_noise_net[-2].bias, init_bias)
        nn.init.zeros_(self.explore_noise_net[-2].weight)
        
    def get_noise_std(
        self,
        t: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Get exploration noise std based on time and state.
        
        Args:
            t: Current timestep in [0, 1]
            state: Conditioning state
            
        Returns:
            Noise std of shape (batch_size, action_dim)
        """
        if self.noise_scheduler_type == "const":
            batch_size = state.shape[0]
            return self.const_noise_std.unsqueeze(0).expand(batch_size, -1)
            
        # Compute embeddings
        time_emb = self.time_embed(t.unsqueeze(-1))  # (B, 64)
        state_emb = self.state_embed(state)  # (B, 64)
        
        # Concatenate and predict noise
        combined = torch.cat([time_emb, state_emb], dim=-1)  # (B, 128)
        noise_std = self.explore_noise_net(combined)  # (B, action_dim)
        
        if self.noise_scheduler_type == "learn_decay":
            # Apply time-dependent decay: noise decreases as t -> 1
            decay_factor = torch.exp(self.log_decay)
            time_scale = torch.exp(-decay_factor * t).unsqueeze(-1)  # (B, 1)
            noise_std = noise_std * time_scale
            
        # Clamp noise std to reasonable range
        noise_std = noise_std.clamp(min=0.001, max=0.5)
        
        return noise_std
    
    def forward(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        sample_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: compute velocity and optionally sample exploration noise.
        
        Args:
            state: Conditioning state
            x: Current position in flow
            t: Current timestep
            sample_noise: Whether to sample noise
            
        Returns:
            Tuple of (velocity, noise, noise_std)
        """
        # Get base velocity (deterministic)
        velocity = self.base_net(state, x, t)
        
        # Get time and state dependent noise std
        noise_std = self.get_noise_std(t, state)
        
        # Sample noise if requested
        if sample_noise:
            noise = torch.randn_like(velocity)
        else:
            noise = torch.zeros_like(velocity)
            
        return velocity, noise, noise_std


class StochasticFlowPolicy(nn.Module):
    """Stochastic Flow Policy using NoisyFlowMLP for RL fine-tuning.
    
    This is a wrapper that uses NoisyFlowMLP as the backbone.
    Kept for backward compatibility.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        noise_scheduler_type: Type of noise schedule
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256, 256],
        noise_scheduler_type: str = "learn"
    ):
        super().__init__()
        
        self.noisy_flow = NoisyFlowMLP(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            noise_scheduler_type=noise_scheduler_type
        )
        
        self.action_dim = action_dim
        
    @property
    def velocity_net(self):
        """Access to base velocity network for pretraining."""
        return self.noisy_flow.base_net
        
    def get_noise_scale(self, t: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        """Get noise scale for timestep t.
        
        Args:
            t: Timestep in [0, 1]
            state: Conditioning state (required for learned noise)
            
        Returns:
            Noise scale
        """
        if state is None:
            # Fallback for backward compatibility
            return torch.ones(1, device=t.device) * 0.1
        return self.noisy_flow.get_noise_std(t, state)
    
    def forward(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: compute velocity and optionally sample noise.
        
        Args:
            state: Conditioning state
            x: Current position in flow
            t: Current timestep
            sample: Whether to sample noise
            
        Returns:
            Tuple of (velocity, noise, noise_scale)
        """
        return self.noisy_flow(state, x, t, sample_noise=sample)


class ReinFlowAgent:
    """ReinFlow Agent for online RL fine-tuning of flow policies.
    
    Key components:
    1. NoisyFlowMLP with learnable exploration noise network
    2. Frozen base policy + trainable fine-tuned policy (optional)
    3. Value function for advantage estimation
    4. PPO-style policy gradient updates
    
    Training phases:
    1. Offline pretraining: Standard flow matching on expert data
    2. Online fine-tuning: Policy gradient with learned exploration noise
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        num_flow_steps: Number of flow integration steps
        noise_scheduler_type: Type of noise schedule ('const', 'learn', 'learn_decay')
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_ratio: PPO clip ratio
        entropy_coef: Entropy coefficient
        device: Device for computation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        num_flow_steps: int = 10,
        noise_scheduler_type: str = "learn",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_flow_steps = num_flow_steps
        self.noise_scheduler_type = noise_scheduler_type
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device
        
        # Stochastic flow policy with NoisyFlowMLP
        self.policy = StochasticFlowPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims + [256],
            noise_scheduler_type=noise_scheduler_type
        ).to(device)
        
        # Value function
        self.value_net = ValueNetwork(
            state_dim=state_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        # Old policy for PPO
        self.old_policy = copy.deepcopy(self.policy)
        for p in self.old_policy.parameters():
            p.requires_grad = False
            
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=learning_rate
        )
        
        self.total_steps = 0
        
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained flow matching weights into base network.
        
        Only loads weights for the base velocity network, not the noise network.
        
        Args:
            checkpoint_path: Path to pretrained checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Try different key names
        if "policy" in checkpoint:
            state_dict = checkpoint["policy"]
        elif "velocity_net" in checkpoint:
            state_dict = checkpoint["velocity_net"]
        else:
            state_dict = checkpoint
            
        # Load into base network only
        self.policy.noisy_flow.base_net.load_state_dict(state_dict, strict=False)
        
        # Also update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        
    def offline_pretrain(
        self,
        dataset: Dict[str, np.ndarray],
        num_steps: int = 5000,
        batch_size: int = 256
    ) -> Dict[str, float]:
        """Pretrain flow policy on offline data.
        
        Uses standard flow matching objective on expert demonstrations.
        Only trains the base velocity network, not the noise network.
        
        Args:
            dataset: Dictionary with 'states' and 'actions'
            num_steps: Number of training steps
            batch_size: Batch size
            
        Returns:
            Dictionary of training metrics
        """
        states = torch.FloatTensor(dataset["states"]).to(self.device)
        actions = torch.FloatTensor(dataset["actions"]).to(self.device)
        n_samples = states.shape[0]
        
        # Only pretrain the base velocity network
        pretrain_optimizer = torch.optim.Adam(
            self.policy.velocity_net.parameters(), lr=1e-4
        )
        
        # Use same sigma_min as FlowMatchingPolicy for consistency
        sigma_min = 0.001
        
        total_loss = 0
        
        for step in range(num_steps):
            idx = torch.randint(0, n_samples, (batch_size,))
            batch_states = states[idx]
            batch_actions = actions[idx]  # x_1 (target)
            
            # Flow matching loss (consistent with FlowMatchingPolicy)
            x_0 = torch.randn_like(batch_actions)  # Noise
            t = torch.rand(batch_size, device=self.device)
            
            # Interpolation with sigma_min (matching FlowMatchingPolicy)
            t_expand = t.unsqueeze(-1)
            sigma_t = 1 - (1 - sigma_min) * t_expand
            x_t = sigma_t * x_0 + t_expand * batch_actions
            
            # Target velocity (matching FlowMatchingPolicy)
            target_v = batch_actions - (1 - sigma_min) * x_0
            
            # Predict velocity (using base network directly)
            pred_v = self.policy.velocity_net(batch_states, x_t, t)
            
            loss = F.mse_loss(pred_v, target_v)
            
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()
            
            total_loss += loss.item()
            
        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        return {"pretrain_loss": total_loss / num_steps}
    
    def _integrate_flow(
        self,
        state: torch.Tensor,
        num_steps: Optional[int] = None,
        sample_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate the stochastic flow to sample action.
        
        Uses Euler-Maruyama for SDE integration with learned noise.
        
        Args:
            state: Conditioning state
            num_steps: Number of integration steps
            sample_noise: Whether to sample stochastic noise
            
        Returns:
            Tuple of (final_action, log_prob)
        """
        num_steps = num_steps or self.num_flow_steps
        batch_size = state.shape[0]
        dt = 1.0 / num_steps
        
        # Start from Gaussian noise
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Track log probability
        log_prob = torch.zeros(batch_size, device=self.device)
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            
            # Get velocity and learned noise scale
            velocity, noise, noise_std = self.policy(
                state, x, t, sample=sample_noise
            )
            
            # Euler-Maruyama step: dx = v * dt + sigma * sqrt(dt) * dW
            x_new = x + velocity * dt
            if sample_noise:
                x_new = x_new + noise_std * noise * np.sqrt(dt)
                
                # Log probability of this noise sample (Gaussian)
                log_prob = log_prob - 0.5 * (noise ** 2).sum(dim=-1)
                log_prob = log_prob - 0.5 * self.action_dim * np.log(2 * np.pi)
                
            x = x_new
            
        # Clip final action
        action = torch.clamp(x, -1.0, 1.0)
        
        return action, log_prob
    
    def _compute_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        policy: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probability of action under policy.
        
        This is approximate since we don't have the exact integration path.
        We use a surrogate based on the learned noise injection.
        
        Args:
            state: State
            action: Action
            policy: Policy to evaluate
            
        Returns:
            Tuple of (log_prob, entropy)
        """
        batch_size = state.shape[0]
        
        # Approximate: treat final action as coming from Gaussian centered at
        # deterministic flow output with learned noise std
        
        # Run deterministic flow
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        dt = 1.0 / self.num_flow_steps
        
        for i in range(self.num_flow_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            velocity, _, noise_std = policy(state, x, t, sample=False)
            x = x + velocity * dt
            
        mean_action = torch.clamp(x, -1.0, 1.0)
        
        # Get final noise scale from the learned noise network
        t_final = torch.ones(batch_size, device=self.device) * 0.9
        _, _, noise_std = policy(state, mean_action, t_final, sample=False)
        
        # Use the learned per-dimension noise std
        std = noise_std + 0.01  # Add small epsilon for stability
        
        # Log prob under Gaussian
        dist = torch.distributions.Normal(mean_action, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy
    
    @torch.no_grad()
    def sample_action_pretrain(
        self,
        state: np.ndarray
    ) -> np.ndarray:
        """Sample action using only pretrained velocity network (deterministic ODE).
        
        This should be used for evaluation after pretrain but before fine-tuning.
        Uses pure ODE integration without stochastic noise injection.
        
        Args:
            state: Current state
            
        Returns:
            Sampled action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        batch_size = state.shape[0]
        dt = 1.0 / self.num_flow_steps
        
        # Start from Gaussian noise
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Pure ODE integration using only base velocity network
        for i in range(self.num_flow_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            
            # Only use base velocity prediction (no noise)
            velocity = self.policy.velocity_net(state, x, t)
            x = x + velocity * dt
            
        action = torch.clamp(x, -1.0, 1.0)
        
        return action.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def sample_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Sample action and return log prob and value.
        
        Args:
            state: Current state
            deterministic: If True, don't sample noise
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Sample from stochastic flow
        action, log_prob = self._integrate_flow(
            state, sample_noise=not deterministic
        )
        
        # Get value
        value = self.value_net(state)
        
        return (
            action.cpu().numpy().squeeze(),
            log_prob.cpu().numpy().squeeze(),
            value.cpu().numpy().squeeze()
        )
    
    def collect_rollout(
        self,
        env,
        rollout_steps: int = 2048
    ) -> RolloutBuffer:
        """Collect experience for online update.
        
        Args:
            env: Gymnasium environment
            rollout_steps: Number of steps to collect
            
        Returns:
            RolloutBuffer with experience
        """
        buffer = RolloutBuffer(
            capacity=rollout_steps,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        state, _ = env.reset()
        
        for _ in range(rollout_steps):
            action, log_prob, value = self.sample_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            buffer.add(
                state=state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done
            )
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state
                
        # Compute returns and advantages
        with torch.no_grad():
            last_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            last_value = self.value_net(last_state).item()
            
        buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        return buffer
    
    def online_update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Update policy using collected experience.
        
        Uses PPO-style clipped objective.
        
        Args:
            buffer: RolloutBuffer with experience
            
        Returns:
            Dictionary of training metrics
        """
        # Save old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for epoch in range(self.ppo_epochs):
            for batch in buffer.get_batches(self.batch_size, shuffle=True):
                states = batch["states"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                
                # Compute new log probs
                new_log_probs, entropy = self._compute_log_prob(
                    states, actions, self.policy
                )
                
                # PPO ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Clipped objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = self.value_net(states).squeeze()
                value_loss = F.mse_loss(values, returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Update policy
                self.policy_optimizer.zero_grad()
                (policy_loss + self.entropy_coef * entropy_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy_optimizer.step()
                
                # Update value
                self.value_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), self.max_grad_norm
                )
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
                
        self.total_steps += len(buffer)
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "noise_scheduler_type": self.noise_scheduler_type
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self.total_steps = checkpoint["total_steps"]
        if "noise_scheduler_type" in checkpoint:
            self.noise_scheduler_type = checkpoint["noise_scheduler_type"]
        self.old_policy.load_state_dict(self.policy.state_dict())
