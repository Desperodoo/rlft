"""
VecEnv Rollout Collection for ManiSkill3 Environments.

Provides a VecEnv-compatible rollout collection function for GPU-parallelized
environments like ManiSkill3. This is designed to work with DPPO and ReinFlow
agents for efficient parallel sampling.

Key differences from single-env collect_rollout:
1. Handles batched observations (num_envs, ...)
2. Works with torch tensors on GPU
3. Efficiently handles parallel resets on episode termination
4. Supports state_image observation mode by default
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union, Any, List
import gymnasium as gym


def collect_rollout_vecenv(
    agent,
    vec_env,
    rollout_steps: int = 2048,
    device: str = "cuda",
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """Collect rollout from a vectorized environment (e.g., ManiSkill3).
    
    This function is designed for GPU-parallelized environments where:
    - Observations and actions are batched torch tensors
    - Multiple environments run in parallel
    - Episode resets happen asynchronously
    
    Args:
        agent: DPPO or ReinFlow agent with sample_action method
        vec_env: Vectorized environment (e.g., ManiSkillPickCubeEnv with num_envs > 1)
        rollout_steps: Number of steps PER ENVIRONMENT to collect
        device: Device for tensor operations
    
    Returns:
        Buffer dictionary with collected experience:
        - states: (rollout_steps * num_envs, state_dim) if applicable
        - images: (rollout_steps * num_envs, C, H, W) if applicable
        - actions: (rollout_steps * num_envs, action_dim)
        - rewards: (rollout_steps * num_envs,)
        - values: (rollout_steps * num_envs,)
        - log_probs: (rollout_steps * num_envs,)
        - dones: (rollout_steps * num_envs,)
        - advantages: (rollout_steps * num_envs,)
        - returns: (rollout_steps * num_envs,)
    
    Example:
        >>> from envs.maniskill_env import make_maniskill_env
        >>> vec_env = make_maniskill_env(obs_mode="state_image", num_envs=16)
        >>> buffer = collect_rollout_vecenv(agent, vec_env, rollout_steps=256)
        >>> metrics = agent.update(buffer)
    """
    num_envs = getattr(vec_env, 'num_envs', 1)
    obs_mode = getattr(agent, 'obs_mode', 'state_image')
    
    # Storage lists (will be flattened at the end)
    all_states = []
    all_images = []
    all_actions = []
    all_rewards = []
    all_values = []
    all_log_probs = []
    all_dones = []
    
    # Track per-env step count for proper GAE computation
    env_step_rewards = [[] for _ in range(num_envs)]
    env_step_values = [[] for _ in range(num_envs)]
    env_step_dones = [[] for _ in range(num_envs)]
    
    # Reset environment
    obs, info = vec_env.reset()
    
    for step in range(rollout_steps):
        # Convert observation to proper format for agent
        # Note: ManiSkill3 uses "rgb" key, but we support "image" for backward compatibility
        if obs_mode == "state_image":
            if isinstance(obs, dict):
                state_batch = obs["state"]
                image_batch = obs.get("rgb", obs.get("image"))
            else:
                raise ValueError("Expected dict observation for state_image mode")
        elif obs_mode == "state":
            state_batch = obs if not isinstance(obs, dict) else obs.get("state", obs)
            image_batch = None
        else:  # image
            state_batch = None
            image_batch = obs if not isinstance(obs, dict) else obs.get("rgb", obs.get("image"))
        
        # Sample actions from agent (batched)
        with torch.no_grad():
            # Prepare observation for agent
            if obs_mode == "state_image":
                agent_obs = {"state": state_batch, "image": image_batch}
            elif obs_mode == "state":
                agent_obs = state_batch
            else:
                agent_obs = image_batch
            
            # Agent's sample_action handles single obs, we need batched version
            actions, log_probs, values = _sample_action_batched(
                agent, agent_obs, obs_mode, device
            )
        
        # Store observations
        if state_batch is not None:
            if isinstance(state_batch, torch.Tensor):
                all_states.append(state_batch.cpu().numpy())
            else:
                all_states.append(state_batch)
        
        if image_batch is not None:
            if isinstance(image_batch, torch.Tensor):
                all_images.append(image_batch.cpu().numpy())
            else:
                all_images.append(image_batch)
        
        # Step environment
        if isinstance(actions, torch.Tensor):
            step_actions = actions
        else:
            step_actions = torch.from_numpy(actions).float()
            if torch.cuda.is_available() and device == "cuda":
                step_actions = step_actions.cuda()
        
        next_obs, rewards, terminated, truncated, info = vec_env.step(step_actions)
        dones = terminated | truncated if isinstance(terminated, torch.Tensor) else (terminated | truncated)
        
        # Convert to numpy
        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy()
        else:
            rewards_np = np.array(rewards)
        
        if isinstance(dones, torch.Tensor):
            dones_np = dones.cpu().numpy()
        else:
            dones_np = np.array(dones)
        
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions
        
        if isinstance(values, torch.Tensor):
            values_np = values.cpu().numpy()
        else:
            values_np = np.array(values)
        
        if isinstance(log_probs, torch.Tensor):
            log_probs_np = log_probs.cpu().numpy()
        else:
            log_probs_np = np.array(log_probs)
        
        # Store transition data
        all_actions.append(actions_np)
        all_rewards.append(rewards_np)
        all_values.append(values_np)
        all_log_probs.append(log_probs_np)
        all_dones.append(dones_np)
        
        # Track per-env for GAE
        for i in range(num_envs):
            env_step_rewards[i].append(rewards_np[i] if rewards_np.ndim > 0 else rewards_np)
            env_step_values[i].append(values_np[i] if values_np.ndim > 0 else values_np)
            env_step_dones[i].append(dones_np[i] if dones_np.ndim > 0 else dones_np)
        
        # Update observation (ManiSkill3 handles auto-reset internally)
        obs = next_obs
    
    # Compute last values for GAE
    with torch.no_grad():
        if obs_mode == "state_image":
            image_batch = obs.get("rgb", obs.get("image"))
            agent_obs = {"state": obs["state"], "image": image_batch}
        elif obs_mode == "state":
            agent_obs = obs if not isinstance(obs, dict) else obs.get("state", obs)
        else:
            agent_obs = obs if not isinstance(obs, dict) else obs.get("rgb", obs.get("image"))
        
        _, _, last_values = _sample_action_batched(agent, agent_obs, obs_mode, device)
        if isinstance(last_values, torch.Tensor):
            last_values = last_values.cpu().numpy()
    
    # Stack and flatten collected data
    # Shape: (rollout_steps, num_envs, ...) -> (rollout_steps * num_envs, ...)
    buffer = {}
    
    if all_states:
        states_arr = np.stack(all_states, axis=0)  # (steps, num_envs, state_dim)
        buffer["states"] = states_arr.reshape(-1, states_arr.shape[-1])
    
    if all_images:
        images_arr = np.stack(all_images, axis=0)  # (steps, num_envs, H, W, C)
        # Flatten first two dimensions
        buffer["images"] = images_arr.reshape(-1, *images_arr.shape[2:])
    
    actions_arr = np.stack(all_actions, axis=0)
    buffer["actions"] = actions_arr.reshape(-1, actions_arr.shape[-1]) if actions_arr.ndim > 2 else actions_arr.flatten()
    
    rewards_arr = np.stack(all_rewards, axis=0)
    buffer["rewards"] = rewards_arr.flatten()
    
    values_arr = np.stack(all_values, axis=0)
    buffer["values"] = values_arr.flatten()
    
    log_probs_arr = np.stack(all_log_probs, axis=0)
    buffer["log_probs"] = log_probs_arr.flatten()
    
    dones_arr = np.stack(all_dones, axis=0)
    buffer["dones"] = dones_arr.flatten()
    
    # Compute per-env last values then flatten
    buffer["last_value"] = last_values.flatten() if last_values.ndim > 0 else last_values
    
    # Compute GAE per environment then flatten
    buffer = _compute_gae_vecenv(buffer, num_envs, rollout_steps, agent.gamma, agent.gae_lambda)
    
    return buffer


def _sample_action_batched(
    agent,
    obs: Union[np.ndarray, Dict, torch.Tensor],
    obs_mode: str,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample actions for a batch of observations.
    
    Handles the batched input/output for vectorized environments.
    Note: Supports both "rgb" (ManiSkill3) and "image" keys for compatibility.
    """
    # Parse observations based on mode
    if obs_mode == "state_image":
        state = obs["state"] if isinstance(obs, dict) else None
        image = obs.get("rgb", obs.get("image")) if isinstance(obs, dict) else None
    elif obs_mode == "state":
        state = obs
        image = None
    else:
        state = None
        image = obs.get("rgb", obs.get("image")) if isinstance(obs, dict) else obs
    
    # Convert to tensors if needed
    if state is not None:
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
        else:
            state = state.to(device)
    
    if image is not None:
        if isinstance(image, np.ndarray):
            # Handle HWC -> CHW conversion if needed
            if image.ndim == 4 and image.shape[-1] in [1, 3, 4]:
                image = np.transpose(image, (0, 3, 1, 2))
            image = torch.FloatTensor(image).to(device)
        else:
            if image.ndim == 4 and image.shape[-1] in [1, 3, 4]:
                image = image.permute(0, 3, 1, 2)
            image = image.to(device)
        
        # Normalize if needed
        if image.max() > 1.0:
            image = image / 255.0
    
    batch_size = state.shape[0] if state is not None else image.shape[0]
    
    # Get observation features
    obs_features = agent.obs_encoder(state=state, image=image)
    
    # Check if agent is DPPO or ReinFlow and sample accordingly
    if hasattr(agent, 'actor_ft'):
        # DPPO agent
        actions, log_probs, values = _sample_dppo_batched(
            agent, obs_features, batch_size, device
        )
    elif hasattr(agent, 'policy'):
        # ReinFlow agent
        actions, log_probs, values = _sample_reinflow_batched(
            agent, state, image, obs_features, batch_size, device
        )
    else:
        raise ValueError("Unknown agent type")
    
    return actions, log_probs, values


def _sample_dppo_batched(
    agent,
    obs_features: torch.Tensor,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample from DPPO agent for a batch."""
    # Diffusion sampling
    x = torch.randn(batch_size, agent.action_dim, device=device)
    
    for t in reversed(range(agent.num_diffusion_steps)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        t_normalized = t_batch.float() / agent.num_diffusion_steps
        
        actor = agent._get_actor_for_step(t)
        pred_noise = actor(x, t_normalized, obs_features=obs_features)
        x = agent.diffusion.p_sample(x, t_batch, pred_noise, clip_denoised=True)
        
        # Add exploration noise for fine-tuned steps
        if 0 < t < agent.ft_denoising_steps:
            noise_scale = torch.exp(agent.log_std) * 0.1
            x = x + noise_scale * torch.randn_like(x)
    
    actions = torch.clamp(x, -1.0, 1.0)
    
    # Compute log prob (simplified for batched inference)
    std = torch.exp(agent.log_std).clamp(min=0.01, max=1.0)
    dist = torch.distributions.Normal(actions, std)
    log_probs = dist.log_prob(actions).sum(dim=-1)
    
    # Get values
    values = agent.value_net(obs_features=obs_features).squeeze(-1)
    
    return actions, log_probs, values


def _sample_reinflow_batched(
    agent,
    state: Optional[torch.Tensor],
    image: Optional[torch.Tensor],
    obs_features: torch.Tensor,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample from ReinFlow agent for a batch."""
    dt = 1.0 / agent.num_flow_steps
    x = torch.randn(batch_size, agent.action_dim, device=device)
    
    log_prob = torch.zeros(batch_size, device=device)
    
    for i in range(agent.num_flow_steps):
        t = torch.full((batch_size,), i * dt, device=device)
        
        # Get velocity and noise from noisy flow
        velocity, noise_std = agent.policy(x, t, obs_features=obs_features, return_noise_std=True)
        
        # Sample noise
        eps = torch.randn_like(x)
        x_next = x + velocity * dt + noise_std * np.sqrt(dt) * eps
        
        # Accumulate log prob
        log_prob = log_prob - 0.5 * (eps ** 2).sum(dim=-1)
        
        x = x_next
    
    actions = torch.clamp(x, -1.0, 1.0)
    
    # Get values
    values = agent.value_net(obs_features=obs_features).squeeze(-1)
    
    return actions, log_prob, values


def _compute_gae_vecenv(
    buffer: Dict[str, np.ndarray],
    num_envs: int,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
) -> Dict[str, np.ndarray]:
    """Compute GAE for vectorized environment rollouts.
    
    The data is stored as (rollout_steps * num_envs,) but we need to
    compute GAE per environment, then flatten back.
    """
    total_samples = rollout_steps * num_envs
    
    # Reshape to (rollout_steps, num_envs)
    rewards = buffer["rewards"].reshape(rollout_steps, num_envs)
    values = buffer["values"].reshape(rollout_steps, num_envs)
    dones = buffer["dones"].reshape(rollout_steps, num_envs)
    last_values = buffer["last_value"]
    if last_values.ndim == 0:
        last_values = np.full(num_envs, last_values)
    
    advantages = np.zeros((rollout_steps, num_envs))
    
    for env_idx in range(num_envs):
        last_gae = 0.0
        for t in reversed(range(rollout_steps)):
            if t == rollout_steps - 1:
                next_value = last_values[env_idx]
            else:
                next_value = values[t + 1, env_idx]
            
            next_non_terminal = 1.0 - dones[t, env_idx]
            delta = rewards[t, env_idx] + gamma * next_value * next_non_terminal - values[t, env_idx]
            advantages[t, env_idx] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    
    # Flatten back
    buffer["advantages"] = advantages.flatten()
    buffer["returns"] = buffer["advantages"] + buffer["values"]
    
    return buffer


if __name__ == "__main__":
    print("VecEnv Rollout Collection Module")
    print("This module provides collect_rollout_vecenv for ManiSkill3 environments.")
    print("\nUsage:")
    print("  from common.vecenv_rollout import collect_rollout_vecenv")
    print("  buffer = collect_rollout_vecenv(agent, vec_env, rollout_steps=256)")
