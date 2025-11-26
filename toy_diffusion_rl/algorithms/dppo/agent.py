"""
DPPO - Diffusion Policy Policy Optimization

Online RL fine-tuning of diffusion policies using PPO-style updates.
Treats the diffusion denoising process as a sequence of actions in
an augmented MDP.

Key feature: Partial chain fine-tuning - only the last K denoising steps
are trainable, while earlier steps remain frozen (from pretrained policy).
This improves stability and sample efficiency.

References:
- DPPO: https://diffusion-ppo.github.io/
- Official code: https://github.com/irom-princeton/dppo
- PPO: Proximal Policy Optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple
import copy

from ...common.networks import DiffusionNoisePredictor, ValueNetwork
from ...common.utils import DiffusionHelper
from ...common.replay_buffer import RolloutBuffer


class DPPOAgent:
    """Diffusion Policy Policy Optimization Agent.
    
    Key ideas:
    1. Model diffusion denoising as a sequential decision process
    2. Each denoising step is an "action" in an augmented MDP
    3. Use PPO to update the diffusion policy for higher returns
    4. Partial chain fine-tuning: only last K steps are trainable
    
    The log-probability of the final action is computed as the sum of
    log-probabilities of each denoising step (only for trainable steps).
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        num_diffusion_steps: Number of diffusion steps
        ft_denoising_steps: Number of denoising steps to fine-tune (from the end)
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_ratio: PPO clip ratio
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm
        device: Device for computation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        num_diffusion_steps: int = 5,
        ft_denoising_steps: int = 3,  # Number of steps to fine-tune (last K steps)
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
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.ft_denoising_steps = min(ft_denoising_steps, num_diffusion_steps)  # Can't exceed total
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device
        
        # Base policy (frozen, for early denoising steps)
        self.actor = DiffusionNoisePredictor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims + [256]
        ).to(device)
        for p in self.actor.parameters():
            p.requires_grad = False
        
        # Fine-tuned policy (trainable, for last K denoising steps)
        self.actor_ft = DiffusionNoisePredictor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims + [256]
        ).to(device)
        
        # Legacy alias for backward compatibility
        self.policy = self.actor_ft
        
        # Value function
        self.value_net = ValueNetwork(
            state_dim=state_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        # Diffusion helper
        self.diffusion = DiffusionHelper(
            num_diffusion_steps=num_diffusion_steps,
            schedule=noise_schedule,
            device=device
        )
        
        # Optimizer - only train actor_ft and value_net
        self.optimizer = torch.optim.Adam([
            {"params": self.actor_ft.parameters(), "lr": learning_rate},
            {"params": self.value_net.parameters(), "lr": learning_rate}
        ])
        
        # For storing old policy (for PPO ratio computation)
        self.old_actor_ft = copy.deepcopy(self.actor_ft)
        for p in self.old_actor_ft.parameters():
            p.requires_grad = False
        
        # Legacy alias
        self.old_policy = self.old_actor_ft
            
        # Noise std for stochastic denoising (trainable)
        self.log_std = nn.Parameter(
            torch.zeros(action_dim, device=device)
        )
        
        self.total_steps = 0
        
        # Compute the timestep threshold for switching from frozen to trainable
        # Steps 0 to (num_diffusion_steps - ft_denoising_steps - 1) use frozen actor
        # Steps (num_diffusion_steps - ft_denoising_steps) to (num_diffusion_steps - 1) use actor_ft
        self.ft_start_step = self.num_diffusion_steps - self.ft_denoising_steps
        
    def _get_actor_for_step(self, t: int) -> nn.Module:
        """Get the appropriate actor for the given timestep.
        
        Args:
            t: Diffusion timestep (0 = clean, num_steps-1 = noise)
            
        Returns:
            Frozen actor for early steps, trainable actor_ft for final steps
        """
        # Reverse diffusion goes from t=num_steps-1 (noise) down to t=0 (clean)
        # We want to fine-tune the LAST K steps, i.e., t < ft_denoising_steps
        if t < self.ft_denoising_steps:
            return self.actor_ft
        else:
            return self.actor
            
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights into both frozen and trainable actors.
        
        Args:
            checkpoint_path: Path to pretrained diffusion policy checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load into frozen actor
        if "policy" in checkpoint:
            self.actor.load_state_dict(checkpoint["policy"])
        elif "actor" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor"])
            
        # Initialize trainable actor_ft from pretrained weights
        self.actor_ft.load_state_dict(self.actor.state_dict())
        self.old_actor_ft.load_state_dict(self.actor_ft.state_dict())
        
    def _compute_action_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        use_old_policy: bool = False,
        return_entropy: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute log probability of an action under diffusion policy.
        
        We model the reverse diffusion as a sequence of Gaussian steps:
        p(a_{t-1} | a_t, s) = N(mu_theta(a_t, t, s), sigma^2)
        
        The total log prob is the sum over fine-tuned denoising steps only.
        
        Args:
            state: Current state
            action: Final action
            use_old_policy: Whether to use old policy for PPO
            return_entropy: Whether to return entropy
            
        Returns:
            Log probability and optionally entropy
        """
        batch_size = state.shape[0]
        
        # We need to trace back the diffusion path
        # Forward diffusion: add noise to action
        total_log_prob = torch.zeros(batch_size, device=self.device)
        total_entropy = torch.zeros(batch_size, device=self.device)
        
        std = torch.exp(self.log_std).clamp(min=0.01, max=1.0)
        
        # Only compute log prob for fine-tuned steps (last K steps)
        for t in range(self.ft_denoising_steps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            # Get noisy version at time t
            if t == self.ft_denoising_steps - 1:
                # Get the intermediate state from frozen policy at the switching point
                a_t = self.diffusion.q_sample(
                    action, 
                    t_batch,
                    noise=torch.randn_like(action)
                )
            else:
                # Use forward diffusion to get a_t from action
                a_t = self.diffusion.q_sample(
                    action, 
                    t_batch,
                    noise=torch.randn_like(action)
                )
            
            # Select appropriate policy
            if use_old_policy:
                pred_noise = self.old_actor_ft(state, a_t, t_normalized)
            else:
                pred_noise = self.actor_ft(state, a_t, t_normalized)
            
            # Expected next step (mean of reverse transition)
            alpha_bar = self.diffusion.alphas_bar[t]
            pred_a0 = (a_t - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            pred_a0 = torch.clamp(pred_a0, -1.0, 1.0)
            
            # Compute log probability of the actual transition
            # This is approximate: we compute prob of final action given predicted
            if t == 0:
                # At t=0, action should match prediction
                dist = torch.distributions.Normal(pred_a0, std)
                log_prob = dist.log_prob(action).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
            else:
                # Intermediate steps contribute less
                dist = torch.distributions.Normal(pred_a0, std * np.sqrt(t / self.num_diffusion_steps))
                log_prob = dist.log_prob(action).sum(dim=-1) * 0.1  # Down-weight intermediate
                entropy = dist.entropy().sum(dim=-1)
                
            total_log_prob = total_log_prob + log_prob
            total_entropy = total_entropy + entropy
            
        if return_entropy:
            return total_log_prob, total_entropy / self.ft_denoising_steps
        return total_log_prob, None
    
    @torch.no_grad()
    @torch.no_grad()
    def sample_action_pretrain(
        self, 
        state: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """Sample action using only the pretrained actor (no partial chain).
        
        This should be used for evaluation after pretrain but before fine-tuning.
        Uses only the frozen base actor for all diffusion steps.
        
        Args:
            state: Current state
            deterministic: If True, don't add noise
            
        Returns:
            Sampled action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        batch_size = state.shape[0]
        
        # Full diffusion sampling using only pretrained actor
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for t in reversed(range(self.num_diffusion_steps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            # Only use the frozen pretrained actor
            pred_noise = self.actor(state, x, t_normalized)
            x = self.diffusion.p_sample(x, t_batch, pred_noise, clip_denoised=True)
                
        action = torch.clamp(x, -1.0, 1.0)
        
        return action.cpu().numpy().squeeze()
    
    def sample_action(
        self, 
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Sample action and return log prob and value.
        
        Uses partial chain: frozen actor for early steps, trainable actor_ft for final steps.
        
        Args:
            state: Current state
            deterministic: If True, use mean prediction
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        batch_size = state.shape[0]
        
        # Diffusion sampling with partial chain
        x = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for t in reversed(range(self.num_diffusion_steps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_normalized = t_batch.float() / self.num_diffusion_steps
            
            # Select appropriate actor based on timestep
            # Use frozen actor for early steps, trainable actor_ft for final steps
            actor = self._get_actor_for_step(t)
            pred_noise = actor(state, x, t_normalized)
            x = self.diffusion.p_sample(x, t_batch, pred_noise, clip_denoised=True)
            
            # Add exploration noise only for fine-tuned steps (unless deterministic)
            if not deterministic and t > 0 and t < self.ft_denoising_steps:
                std = torch.exp(self.log_std) * 0.1
                x = x + std * torch.randn_like(x)
                
        action = torch.clamp(x, -1.0, 1.0)
        
        # Compute log prob (only for fine-tuned steps)
        log_prob, _ = self._compute_action_log_prob(state, action, use_old_policy=False)
        
        # Compute value
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
        """Collect rollout for PPO update.
        
        Args:
            env: Gymnasium environment
            rollout_steps: Number of steps to collect
            
        Returns:
            RolloutBuffer with collected experience
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
    
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Update policy using PPO.
        
        Only updates the trainable actor_ft (fine-tuned policy for last K steps).
        
        Args:
            buffer: RolloutBuffer with collected experience
            
        Returns:
            Dictionary of training metrics
        """
        # Copy current actor_ft to old policy
        self.old_actor_ft.load_state_dict(self.actor_ft.state_dict())
        
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
                
                # Compute new log probs and entropy using current actor_ft
                new_log_probs, entropy = self._compute_action_log_prob(
                    states, actions, use_old_policy=False, return_entropy=True
                )
                
                # PPO ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Clipped objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = self.value_net(states).squeeze()
                value_loss = F.mse_loss(values, returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Optimize (only actor_ft and value_net are trainable)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_ft.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
                
        self.total_steps += len(buffer)
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "ft_steps": self.ft_denoising_steps
        }
    
    def pretrain_bc(
        self,
        dataset: Dict[str, np.ndarray],
        num_steps: int = 1000,
        batch_size: int = 256
    ) -> Dict[str, float]:
        """Pretrain with behavior cloning.
        
        Trains both frozen actor and trainable actor_ft with the same weights.
        
        Args:
            dataset: Dictionary with 'states' and 'actions'
            num_steps: Number of BC steps
            batch_size: Batch size
            
        Returns:
            Dictionary of BC metrics
        """
        states = torch.FloatTensor(dataset["states"]).to(self.device)
        actions = torch.FloatTensor(dataset["actions"]).to(self.device)
        n_samples = states.shape[0]
        
        # Temporarily enable gradients for frozen actor during pretraining
        for p in self.actor.parameters():
            p.requires_grad = True
            
        bc_optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.actor_ft.parameters()), 
            lr=1e-4
        )
        
        total_loss = 0
        for step in range(num_steps):
            idx = torch.randint(0, n_samples, (batch_size,))
            batch_states = states[idx]
            batch_actions = actions[idx]
            
            # Sample timesteps
            t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
            noise = torch.randn_like(batch_actions)
            noisy_actions = self.diffusion.q_sample(batch_actions, t, noise)
            t_normalized = t.float() / self.num_diffusion_steps
            
            # Train both actors with same loss
            pred_noise_frozen = self.actor(batch_states, noisy_actions, t_normalized)
            pred_noise_ft = self.actor_ft(batch_states, noisy_actions, t_normalized)
            
            loss = F.mse_loss(pred_noise_frozen, noise) + F.mse_loss(pred_noise_ft, noise)
            
            bc_optimizer.zero_grad()
            loss.backward()
            bc_optimizer.step()
            
            total_loss += loss.item()
            
        # Freeze the base actor after pretraining
        for p in self.actor.parameters():
            p.requires_grad = False
            
        # Update old policy
        self.old_actor_ft.load_state_dict(self.actor_ft.state_dict())
            
        return {"bc_loss": total_loss / num_steps}
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_ft": self.actor_ft.state_dict(),
            "value_net": self.value_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "log_std": self.log_std,
            "total_steps": self.total_steps,
            "ft_denoising_steps": self.ft_denoising_steps
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_ft.load_state_dict(checkpoint["actor_ft"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.log_std.data = checkpoint["log_std"]
        self.total_steps = checkpoint["total_steps"]
        if "ft_denoising_steps" in checkpoint:
            self.ft_denoising_steps = checkpoint["ft_denoising_steps"]
            self.ft_start_step = self.num_diffusion_steps - self.ft_denoising_steps
        self.old_actor_ft.load_state_dict(self.actor_ft.state_dict())
