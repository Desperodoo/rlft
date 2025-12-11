"""
Online RL Fine-tuning Script for ReinFlow

Fine-tunes a pretrained flow matching policy (from AW-ShortCut Flow)
using PPO-style policy gradient with environment interaction.

Three-stage pipeline:
- Stage 1: ShortCut Flow BC pretrain (pure BC) → train_offline_rl.py --algorithm shortcut_flow
- Stage 2: AW-ShortCut Flow offline RL (Q-weighted BC) → train_offline_rl.py --algorithm aw_shortcut_flow
- Stage 3: ReinFlow online RL (this script, PPO fine-tuning)

Key features:
- SMDP formulation: executes full act_horizon before policy update
- Critic warmup: trains value network before policy updates
- Noise scheduling: controllable exploration decay
- Compatible with AW-ShortCut Flow checkpoints

Usage:
    python train_online_finetune.py \
        --env_id PickCube-v1 \
        --pretrained_path runs/aw_shortcut_flow-PickCube-v1-seed0/best_model.pt \
        --num_inference_steps 8 \
        --critic_warmup_steps 5000

Based on ManiSkill PPO implementation and ReinFlow paper.
"""

ALGO_NAME = "ReinFlow"

import os
import random
import time
import warnings
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict, Any
import json

# Suppress SAPIEN PhysX CPU warnings when using GPU training with CPU eval
os.environ["SAPIEN_LOG_LEVEL"] = "error"
warnings.filterwarnings("ignore", message=".*PhysX CPU system.*")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import AgentWrapper, build_state_obs_extractor, convert_obs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.algorithms.reinflow import ReinFlowAgent, VectorizedRolloutBuffer, compute_smdp_rewards
from diffusion_policy.evaluate import evaluate


@dataclass
class Args:
    # Experiment settings
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances"""

    # Environment settings
    env_id: str = "LiftPegUpright-v1"
    """the id of the environment"""
    num_envs: int = 50
    """number of parallel environments for training"""
    num_eval_envs: int = 50
    """number of parallel eval environments"""
    max_episode_steps: Optional[int] = None
    """max episode steps"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode"""
    obs_mode: str = "rgb"
    """observation mode: rgb, depth, or rgb+depth"""
    sim_backend: str = "physx_cuda"
    """simulation backend (physx_cuda for parallel training)"""

    # Pretrained checkpoint
    pretrained_path: str = "runs/awsc-LiftPegUpright-v1-seed0/checkpoints/best_eval_success_once.pt"
    """path to pretrained AW-ShortCut Flow checkpoint"""
    freeze_visual_encoder: bool = True
    """whether to freeze visual encoder during fine-tuning"""

    # Training settings
    total_updates: int = 20_000
    """total training timesteps (env steps, not policy steps)"""
    rollout_steps: int = 128
    """number of SMDP chunks to collect before each update"""
    ppo_epochs: int = 2
    """number of PPO epochs per update"""
    minibatch_size: int = 6400
    """minibatch size for PPO updates"""
    lr: float = 3e-5
    """learning rate for policy"""
    lr_critic: float = 1e-4
    """learning rate for value network"""
    max_grad_norm: float = 0.5
    """maximum gradient norm for clipping"""

    # PPO hyperparameters
    gamma: float = 0.99
    """discount factor"""
    gae_lambda: float = 0.95
    """GAE lambda"""
    clip_ratio: float = 0.2
    """PPO clip ratio"""
    entropy_coef: float = 0.01
    """entropy coefficient"""
    value_coef: float = 0.5
    """value loss coefficient"""

    # Policy settings
    obs_horizon: int = 2
    """observation horizon"""
    act_horizon: int = 8
    """action execution horizon (full chunk for SMDP)"""
    pred_horizon: int = 16
    """action prediction horizon"""
    num_inference_steps: int = 8
    """number of flow integration steps (fixed mode)"""
    ema_decay: float = 0.999
    """EMA decay rate"""

    # Network architecture (defaults match train_offline_rl.py for checkpoint compatibility)
    diffusion_step_embed_dim: int = 64
    """timestep embedding dimension (must match pretrained checkpoint)"""
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    """U-Net channel dimensions (must match pretrained checkpoint)"""
    n_groups: int = 8
    """GroupNorm groups (must match pretrained checkpoint)"""
    visual_feature_dim: int = 256
    """visual encoder output dimension (must match pretrained checkpoint)"""

    # Noise scheduling
    min_noise_std: float = 0.01
    """minimum exploration noise std"""
    max_noise_std: float = 0.3
    """maximum exploration noise std"""
    noise_decay_type: Literal["constant", "linear", "exponential"] = "linear"
    """noise decay schedule type"""
    noise_decay_steps: int = 500000
    """steps over which to decay exploration noise"""

    # Critic warmup
    critic_warmup_steps: int = 5000
    """number of steps to train critic only (policy frozen)"""

    # Reward processing
    normalize_rewards: bool = True
    """whether to normalize rewards using running mean/std"""
    normalize_advantages: bool = True
    """whether to normalize advantages (recommended for stable training)"""
    clip_value_loss: bool = True
    """whether to clip value loss to reduce vloss explosion"""
    value_clip_range: float = 10.0
    """clip value predictions to [-range, range] for stable training"""

    # Logging settings (in number of updates, not timesteps)
    log_freq: int = 1
    """logging frequency (in number of updates)"""
    eval_freq: int = 10
    """evaluation frequency (in number of updates)"""
    save_freq: int = 10000000
    """checkpoint save frequency (in number of updates)"""
    num_eval_episodes: int = 50
    """number of evaluation episodes"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningMeanStd:
    """Running mean/std calculator using Welford's algorithm."""
    
    def __init__(self, epsilon: float = 1e-8, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # Avoid division by zero
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray):
        """Update running statistics with a batch of values."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.epsilon)


class RewardNormalizer:
    """Running mean/std for reward normalization."""
    
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.rms = RunningMeanStd(epsilon=epsilon)
    
    def update(self, rewards: np.ndarray):
        """Update running statistics."""
        self.rms.update(rewards.flatten())
    
    def normalize(self, reward: float) -> float:
        """Normalize a single reward by dividing by std."""
        return reward / (self.rms.std + self.epsilon)


def make_train_envs(env_id: str, num_envs: int, sim_backend: str, 
                    control_mode: str, obs_mode: str, 
                    max_episode_steps: Optional[int] = None,
                    record_video: bool = False,
                    video_path: str = None):
    """Create parallel training environments."""
    env_kwargs = dict(
        obs_mode="rgbd" if "rgb" in obs_mode or "depth" in obs_mode else "state",
        control_mode=control_mode,
        sim_backend=sim_backend,
        num_envs=num_envs,
    )
    
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps
    
    if record_video and video_path:
        env_kwargs["enable_shadow"] = True
        env_kwargs["render_mode"] = "rgb_array"
    
    env = gym.make(env_id, **env_kwargs)
    
    # Wrap for RGBD observations
    if "rgb" in obs_mode or "depth" in obs_mode:
        include_rgb = "rgb" in obs_mode
        include_depth = "depth" in obs_mode
        env = FlattenRGBDObservationWrapper(
            env, 
            rgb=include_rgb, 
            depth=include_depth, 
            state=True
        )
    
    return env


def main():
    args = tyro.cli(Args)
    
    # Generate experiment name
    if args.exp_name is None:
        args.exp_name = f"reinflow-{args.env_id}-seed{args.seed}"
    
    run_name = f"{args.exp_name}__{int(time.time())}"
    
    # Set up logging
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize logging
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )
    
    writer = SummaryWriter(log_dir)
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environments
    print("Creating training environments...")
    train_envs = make_train_envs(
        env_id=args.env_id,
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        obs_mode=args.obs_mode,
        max_episode_steps=args.max_episode_steps,
    )
    
    print("Creating evaluation environments...")
    eval_env_kwargs = dict(
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        obs_mode=args.obs_mode,  # Must match training obs_mode for AgentWrapper
        render_mode="rgb_array",  # For video recording compatibility
    )
    eval_other_kwargs = dict(
        obs_horizon=args.obs_horizon,
    )
    eval_envs = make_eval_envs(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        sim_backend=args.sim_backend,
        env_kwargs=eval_env_kwargs,
        other_kwargs=eval_other_kwargs,
        video_dir=f"{log_dir}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],  # Flatten RGBD observations before FrameStack
    )
    
    # Get environment info
    obs_space = train_envs.single_observation_space
    act_space = train_envs.single_action_space
    act_dim = act_space.shape[0]
    
    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")
    print(f"Action dimension: {act_dim}")
    
    # Build state observation extractor and get state dimension
    # Note: state_obs_extractor is NOT used for online training since FlattenRGBDObservationWrapper
    # already flattens the state. We keep it only for compatibility.
    state_obs_extractor = build_state_obs_extractor(args.env_id)
    # State dimension is simply the flattened state size from observation space
    state_dim = train_envs.single_observation_space["state"].shape[0]
    print(f"State dimension: {state_dim}")
    
    # Check observation mode
    include_rgb = "rgb" in args.obs_mode
    include_depth = "depth" in args.obs_mode
    
    # Build visual encoder
    if include_rgb or include_depth:
        if include_rgb and include_depth:
            in_channels = 4  # RGB + Depth
        elif include_rgb:
            in_channels = 3
        else:
            in_channels = 3  # Depth repeated to 3 channels
        
        visual_encoder = PlainConv(
            in_channels=in_channels,
            out_dim=args.visual_feature_dim,
            pool_feature_map=True,  # Must match pretrained checkpoint
        ).to(device)
        
        obs_dim = (args.visual_feature_dim + state_dim) * args.obs_horizon
    else:
        visual_encoder = nn.Identity()
        obs_dim = state_dim * args.obs_horizon
    
    print(f"Observation dimension (flattened): {obs_dim}")
    
    # Build velocity network (ShortCut Flow architecture)
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=act_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=args.unet_dims,
        n_groups=args.n_groups,
    ).to(device)
    
    # Build ReinFlow agent
    agent = ReinFlowAgent(
        velocity_net=velocity_net,
        obs_dim=obs_dim,
        act_dim=act_dim,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        act_horizon=args.act_horizon,
        num_inference_steps=args.num_inference_steps,
        ema_decay=args.ema_decay,
        use_ema=True,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        min_noise_std=args.min_noise_std,
        max_noise_std=args.max_noise_std,
        noise_decay_type=args.noise_decay_type,
        noise_decay_steps=args.noise_decay_steps,
        critic_warmup_steps=args.critic_warmup_steps,
    ).to(device)
    
    # Load pretrained checkpoint
    if args.pretrained_path:
        print(f"Loading pretrained model from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        
        # Load velocity network weights
        if "velocity_net" in checkpoint:
            agent.load_from_aw_shortcut_flow(args.pretrained_path, device=str(device))
        else:
            agent.load_pretrained(args.pretrained_path, device=str(device))
        
        # Load visual encoder if available
        if include_rgb or include_depth:
            if "visual_encoder" in checkpoint:
                visual_encoder.load_state_dict(checkpoint["visual_encoder"])
                print("Loaded visual encoder from checkpoint")
        
        print("Pretrained model loaded successfully")
    
    # Freeze visual encoder if specified
    if args.freeze_visual_encoder and (include_rgb or include_depth):
        for param in visual_encoder.parameters():
            param.requires_grad = False
        print("Visual encoder frozen")
    
    # Set up optimizers
    policy_params = list(agent.noisy_velocity_net.parameters())
    critic_params = list(agent.value_net.parameters())
    
    if not args.freeze_visual_encoder and (include_rgb or include_depth):
        policy_params += list(visual_encoder.parameters())
    
    policy_optimizer = optim.AdamW(policy_params, lr=args.lr)
    critic_optimizer = optim.AdamW(critic_params, lr=args.lr_critic)
    
    # Create vectorized rollout buffer (following official PPO implementation)
    buffer = VectorizedRolloutBuffer(
        num_steps=args.rollout_steps,
        num_envs=args.num_envs,
        obs_dim=obs_dim,
        pred_horizon=args.pred_horizon,
        act_dim=act_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        normalize_advantages=args.normalize_advantages,
        device=str(device),
    )
    
    # Create agent wrapper for evaluation (aligns with train_offline_rl.py)
    agent_wrapper = AgentWrapper(
        agent, visual_encoder, include_rgb, include_depth, 
        args.obs_horizon, args.act_horizon
    ).to(device)
    
    # Helper function to encode observations (following train_offline_rl.py pattern)
    def encode_observations(obs_seq):
        """Encode observation sequence to get conditioning features.
        
        Args:
            obs_seq: dict with 'state' and optionally 'rgb'/'depth'
                    shapes: state [B, T, state_dim], rgb [B, T, H, W, C]
        
        Returns:
            obs_cond: [B, obs_horizon * (visual_dim + state_dim)]
        """
        B = obs_seq["state"].shape[0]
        T = obs_seq["state"].shape[1]
        
        features_list = []
        
        # Visual features
        if include_rgb or include_depth:
            if include_rgb:
                rgb = obs_seq["rgb"]  # [B, T, H, W, C]
                # Convert to [B*T, C, H, W] and normalize
                rgb_flat = rgb.reshape(B * T, *rgb.shape[2:])
                rgb_flat = rgb_flat.permute(0, 3, 1, 2).float() / 255.0
                
                if include_depth and "depth" in obs_seq:
                    depth = obs_seq["depth"]  # [B, T, H, W, 1]
                    depth_flat = depth.reshape(B * T, *depth.shape[2:])
                    depth_flat = depth_flat.permute(0, 3, 1, 2).float()
                    visual_input = torch.cat([rgb_flat, depth_flat], dim=1)
                else:
                    visual_input = rgb_flat
            elif include_depth:
                depth = obs_seq["depth"]
                depth_flat = depth.reshape(B * T, *depth.shape[2:])
                depth_flat = depth_flat.permute(0, 3, 1, 2).float()
                # Repeat depth to 3 channels for encoder
                visual_input = depth_flat.repeat(1, 3, 1, 1)
            
            visual_feat = visual_encoder(visual_input)  # [B*T, visual_dim]
            visual_feat = visual_feat.view(B, T, -1)  # [B, T, visual_dim]
            features_list.append(visual_feat)
        
        # State features - already flattened by FlattenRGBDObservationWrapper
        state = obs_seq["state"]  # [B, T, state_dim]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(device)
        features_list.append(state)
        
        # Concatenate features: [B, T, visual_dim + state_dim]
        obs_features = torch.cat(features_list, dim=-1)
        
        # Flatten to [B, T * (visual_dim + state_dim)]
        obs_cond = obs_features.reshape(B, -1)
        
        return obs_cond
    
    # Training statistics
    reward_normalizer = RewardNormalizer(gamma=args.gamma)
    global_step = 0
    num_updates = 0
    start_time = time.time()
    
    timings = defaultdict(float)
    best_eval_metrics = defaultdict(float)
    
    # Episode tracking
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_successes = deque(maxlen=100)
    
    # Observation buffer
    obs, _ = train_envs.reset()
    obs_deque = deque(maxlen=args.obs_horizon)
    for _ in range(args.obs_horizon):
        obs_deque.append(obs)
    
    # Per-env episode tracking
    env_episode_rewards = np.zeros(args.num_envs)
    env_episode_lengths = np.zeros(args.num_envs, dtype=int)
    
    # Calculate expected number of updates
    timesteps_per_update = args.num_envs * args.act_horizon * args.rollout_steps
    total_timesteps = args.total_updates * timesteps_per_update
    
    print("\nStarting training...")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Timesteps per update: {timesteps_per_update} ({args.num_envs} envs × {args.act_horizon} act × {args.rollout_steps} rollout)")
    print(f"Expected updates: {args.total_updates}")
    print(f"Log every {args.log_freq} updates, Eval every {args.eval_freq} updates, Save every {args.save_freq} updates")
    print(f"Critic warmup steps: {args.critic_warmup_steps}")
    print(f"Noise schedule: {args.noise_decay_type}, {args.max_noise_std} -> {args.min_noise_std} over {args.noise_decay_steps} steps")
    
    # Track done state for GAE (following ppo_rgb.py)
    next_done = torch.zeros(args.num_envs, device=device)
    
    pbar = tqdm(total=args.total_updates, desc="Training")
    
    while num_updates < args.total_updates:
        # Collect rollouts
        buffer.reset()
        
        for rollout_step in range(args.rollout_steps):
            # Process observations: stack obs_deque along time dimension
            obs_seq = {
                k: torch.stack([o[k] for o in obs_deque], dim=1)
                for k in obs.keys()
            }
            
            # Encode observations using unified function
            with torch.no_grad():
                obs_cond = encode_observations(obs_seq)
                
                # Get action with exploration
                actions, chains = agent.get_action(
                    obs_cond, 
                    deterministic=False, 
                    use_ema=False,
                    return_chains=True,
                )
                
                # Compute value and log_prob
                values = agent.compute_value(obs_cond)
                log_probs, _ = agent.compute_action_log_prob(
                    obs_cond, actions, chains=chains, use_old_policy=False
                )
            
            # Execute action chunk (SMDP)
            chunk_rewards = []
            chunk_dones = []
            
            for step_idx in range(args.act_horizon):
                action = actions[:, step_idx, :]
                next_obs, rewards, terminations, truncations, infos = train_envs.step(action)
                
                dones = terminations | truncations
                
                chunk_rewards.append(rewards)
                chunk_dones.append(dones)
                
                # Update per-env tracking
                for i in range(args.num_envs):
                    env_episode_rewards[i] += rewards[i].item()
                    env_episode_lengths[i] += 1
                    
                    if dones[i]:
                        episode_rewards.append(env_episode_rewards[i])
                        episode_lengths.append(env_episode_lengths[i])
                        
                        if "success" in infos:
                            success = infos["success"][i].item() if hasattr(infos["success"][i], "item") else infos["success"][i]
                        else:
                            success = rewards[i].item() > 0
                        episode_successes.append(float(success))
                        
                        # Reset tracking
                        env_episode_rewards[i] = 0.0
                        env_episode_lengths[i] = 0
                
                obs_deque.append(next_obs)
                obs = next_obs
                global_step += args.num_envs
                
                # Check if any episode ended - break to handle episode boundary
                if dones.any():
                    break
            
            # Compute SMDP rewards (vectorized)
            chunk_rewards_tensor = torch.stack(chunk_rewards)  # (chunk_len, num_envs)
            chunk_dones_tensor = torch.stack(chunk_dones)      # (chunk_len, num_envs)
            chunk_len = len(chunk_rewards)
            
            # Cumulative discounted reward for each env
            cum_rewards = torch.zeros(args.num_envs, device=device)
            chunk_done_flags = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            discount = 1.0
            
            for i in range(chunk_len):
                # Only add reward if episode hasn't ended yet for this env
                still_running = ~chunk_done_flags
                cum_rewards = cum_rewards + discount * chunk_rewards_tensor[i] * still_running.float()
                
                # Update done flags
                chunk_done_flags = chunk_done_flags | chunk_dones_tensor[i]
                discount *= args.gamma
            
            # Optional: normalize rewards
            if args.normalize_rewards:
                reward_normalizer.update(cum_rewards.cpu().numpy())
                cum_rewards = torch.tensor(
                    [reward_normalizer.normalize(r.item()) for r in cum_rewards],
                    device=device
                )
            
            # Add to buffer (vectorized)
            buffer.add(
                obs=obs_cond,
                actions=actions,
                rewards=cum_rewards,
                values=values,
                log_probs=log_probs,
                dones=chunk_done_flags.float(),
            )
            
            # Handle final values for episodes that ended (following ppo_rgb.py)
            if chunk_done_flags.any():
                done_mask = chunk_done_flags
                # Get value estimate for the new state after reset
                with torch.no_grad():
                    # The obs after episode end is already the new episode's first obs
                    final_obs_seq = {
                        k: torch.stack([o[k] for o in obs_deque], dim=1)
                        for k in obs.keys()
                    }
                    final_obs_cond = encode_observations(final_obs_seq)
                    final_vals = agent.compute_value(final_obs_cond[done_mask])
                buffer.set_final_values(rollout_step, done_mask, final_vals)
            
            # Update next_done for GAE
            next_done = chunk_done_flags.float()
        
        # Compute returns and advantages
        with torch.no_grad():
            # Get last value for bootstrapping
            obs_seq = {
                k: torch.stack([o[k] for o in obs_deque], dim=1)
                for k in obs.keys()
            }
            
            obs_cond = encode_observations(obs_seq)
            next_value = agent.compute_value(obs_cond).squeeze(-1)
        
        # Vectorized GAE computation (following ppo_rgb.py)
        # Note: advantage normalization is handled inside buffer if enabled
        buffer.compute_returns_and_advantages(
            next_value=next_value,
            next_done=next_done,
        )
        
        # PPO update
        last_tick = time.time()
        agent.sync_old_policy()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0
        
        for epoch in range(args.ppo_epochs):
            batches = buffer.get_batches(args.minibatch_size, shuffle=True)
            
            for batch in batches:
                # Forward pass with value clipping to prevent vloss explosion
                loss_dict = agent.compute_ppo_loss(
                    obs_cond=batch["obs"],
                    actions=batch["actions"],
                    old_log_probs=batch["log_probs"],
                    advantages=batch["advantages"],
                    returns=batch["returns"],
                    old_values=batch.get("values"),  # For value clipping
                    clip_value=args.clip_value_loss,
                    value_clip_range=args.value_clip_range,
                )
                
                # Update noise schedule
                agent.update_noise_schedule()
                
                # Backward pass
                if num_updates < args.critic_warmup_steps:
                    # During warmup, only update critic
                    critic_optimizer.zero_grad()
                    loss_dict["loss"].backward()
                    nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                    critic_optimizer.step()
                else:
                    # Full PPO update
                    policy_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    loss_dict["loss"].backward()
                    nn.utils.clip_grad_norm_(policy_params, args.max_grad_norm)
                    nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                    policy_optimizer.step()
                    critic_optimizer.step()
                
                total_policy_loss += loss_dict["policy_loss"].item()
                total_value_loss += loss_dict["value_loss"].item()
                total_entropy += loss_dict["entropy"].item()
                num_batches += 1
        
        # Update EMA
        agent.update_ema()
        num_updates += 1
        
        # Timing for PPO update
        timings["forward"] += time.time() - last_tick
        
        # Compute average losses for this update (always compute for pbar)
        avg_policy_loss = total_policy_loss / max(1, num_batches)
        avg_value_loss = total_value_loss / max(1, num_batches)
        avg_entropy = total_entropy / max(1, num_batches)
        min_noise, max_noise = agent.get_current_noise_std()
        
        # Prepare losses dict for pbar display (always available)
        losses_dict = {
            "ploss": avg_policy_loss,
            "vloss": avg_value_loss,
            "ent": avg_entropy,
            "noise": max_noise,
            "episode_reward": np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0,
            "success_rate": np.mean(episode_successes) if len(episode_successes) > 0 else 0.0,
        }
        if num_updates < args.critic_warmup_steps:
            losses_dict["warmup"] = 1.0
        
        # Logging (periodic)
        if num_updates % args.log_freq == 0 and num_updates > 0:
            last_tick = time.time()
            
            # Log to tensorboard
            writer.add_scalar("charts/learning_rate", policy_optimizer.param_groups[0]["lr"], num_updates)
            writer.add_scalar("losses/policy_loss", avg_policy_loss, num_updates)
            writer.add_scalar("losses/value_loss", avg_value_loss, num_updates)
            writer.add_scalar("losses/entropy", avg_entropy, num_updates)
            writer.add_scalar("losses/noise_std_max", max_noise, num_updates)
            writer.add_scalar("losses/in_warmup", float(bool(num_updates < args.critic_warmup_steps)), num_updates)
            
            # Log timings
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, num_updates)
            
            if len(episode_rewards) > 0:
                writer.add_scalar("charts/episode_reward", np.mean(episode_rewards), num_updates)
                writer.add_scalar("charts/episode_length", np.mean(episode_lengths), num_updates)
                writer.add_scalar("charts/success_rate", np.mean(episode_successes), num_updates)
            
            timings["logging"] += time.time() - last_tick
        
        # Evaluation (using official evaluate function, aligned with train_offline_rl.py)
        if num_updates % args.eval_freq == 0:
            last_tick = time.time()
            # ReinFlow-specific: use deterministic=True, use_ema=True for evaluation
            eval_metrics = evaluate(
                args.num_eval_episodes, 
                agent_wrapper, 
                eval_envs, 
                device, 
                args.sim_backend,
                agent_kwargs={"deterministic": True, "use_ema": True},
            )
            timings["eval"] += time.time() - last_tick
            
            print(f"\nEvaluated {len(eval_metrics['success_at_end'])} episodes")
            
            # Average metrics and log (aligned with offline format)
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], num_updates)
                print(f"  {k}: {eval_metrics[k]:.4f}")
            
            # Track best metrics for checkpointing
            for k in ["success_once", "success_at_end"]:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    checkpoint_path = f"{log_dir}/checkpoint_best_eval_{k}.pt"
                    checkpoint = {
                        "agent": agent.state_dict(),
                        "visual_encoder": visual_encoder.state_dict() if include_rgb or include_depth else None,
                        "config": vars(args),
                    }
                    torch.save(checkpoint, checkpoint_path)
                    print(f"New best {k}: {eval_metrics[k]:.4f}. Saving checkpoint.")
            
            if args.track:
                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=num_updates)
        
        # Save checkpoint (periodic)
        if num_updates % args.save_freq == 0 and num_updates > 0:
            checkpoint_path = f"{log_dir}/checkpoint_{num_updates}.pt"
            checkpoint = {
                "agent": agent.state_dict(),
                "visual_encoder": visual_encoder.state_dict() if include_rgb or include_depth else None,
                "policy_optimizer": policy_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "global_step": global_step,
                "num_updates": num_updates,
                "config": vars(args),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Update progress bar with losses (aligned with offline format)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in losses_dict.items()})
        pbar.update(args.num_envs * args.act_horizon * args.rollout_steps)
    
    pbar.close()
    
    # Final save
    final_path = f"{log_dir}/final_model.pt"
    checkpoint = {
        "agent": agent.state_dict(),
        "visual_encoder": visual_encoder.state_dict() if include_rgb or include_depth else None,
        "config": vars(args),
    }
    torch.save(checkpoint, final_path)
    print(f"Final model saved to {final_path}")
    
    # Cleanup
    train_envs.close()
    eval_envs.close()
    writer.close()
    
    if args.track:
        wandb.finish()
    
    print("Training complete!")


if __name__ == "__main__":
    main()
