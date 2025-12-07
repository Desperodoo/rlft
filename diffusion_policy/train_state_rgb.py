"""
Diffusion Policy with State + RGB Multi-Modal Observations.

This script trains a Diffusion Policy that uses both proprioceptive state
information (agent qpos/qvel, task-specific features) and RGB images as input.

The key difference from train.py (state-only) and train_rgbd.py (visual-only):
- Combines state vector with visual features for richer observation
- Uses FlattenRGBDObservationWrapper to process observations
- Supports state_dict+rgb demonstration format from ManiSkill

Usage:
    python train_state_rgb.py --env-id PickCube-v1 \
        --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state_dict+rgb.pd_ee_delta_pos.physx_cpu.h5 \
        --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" \
        --num-demos 100 --max_episode_steps 100 \
        --total_iters 30000 \
        --exp-name diffusion_policy-PickCube-v1-state_rgb
"""

ALGO_NAME = "BC_Diffusion_state_rgb_UNet"

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor,
                                    worker_init_fn)


@dataclass
class Args:
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
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PickCube-v1"
    """the id of the environment"""
    demo_path: str = (
        "~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state_dict+rgb.pd_ee_delta_pos.physx_cpu.h5"
    )
    """the path of demo dataset (state_dict+rgb format)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = (
        16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    )
    diffusion_step_embed_dim: int = 64  # not very important
    unet_dims: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )  # default setting is about ~4.5M params
    n_groups: int = (
        8  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila
    )
    # Vision encoder arguments
    visual_feature_dim: int = 256
    """output dimension of visual encoder"""

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_ee_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


def reorder_keys(d, ref_dict):
    """Reorder dictionary keys to match reference dictionary order."""
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


def load_state_rgb_demo_dataset(data_path, num_traj=None, state_keys=None):
    """Load state_dict+rgb demonstration dataset.
    
    This function loads demonstrations in the state_dict+rgb format which contains:
    - obs/agent/qpos, qvel: proprioceptive state
    - obs/extra/*: task-specific features
    - obs/sensor_data/*/rgb: RGB images
    - actions: action sequences
    
    Args:
        data_path: Path to HDF5 file
        num_traj: Number of trajectories to load (None for all)
        state_keys: Dict specifying which state keys to include.
                   Format: {'agent': ['qpos', 'qvel'], 'extra': ['tcp_pose', ...]}
                   If None, includes all available keys.
    
    Returns:
        dict with keys:
            - 'observations': list of dicts, each with 'state' and 'rgb' keys
            - 'actions': list of np.ndarray
    """
    import h5py
    
    print(f"Loading state_dict+rgb HDF5 file: {data_path}")
    data_path = os.path.expanduser(data_path)
    
    trajectories = {'observations': [], 'actions': []}
    
    with h5py.File(data_path, 'r') as f:
        keys = sorted(f.keys(), key=lambda x: int(x.split('_')[-1]))
        if num_traj is not None:
            keys = keys[:num_traj]
        
        for traj_key in keys:
            traj = f[traj_key]
            obs = traj['obs']
            
            # Extract state components
            state_parts = []
            
            # Agent state (qpos, qvel)
            if 'agent' in obs:
                agent_keys = state_keys.get('agent', list(obs['agent'].keys())) if state_keys else list(obs['agent'].keys())
                for k in agent_keys:
                    if k in obs['agent']:
                        state_parts.append(obs['agent'][k][:])
            
            # Extra state (task-specific features)
            if 'extra' in obs:
                extra_keys = state_keys.get('extra', list(obs['extra'].keys())) if state_keys else list(obs['extra'].keys())
                for k in extra_keys:
                    if k in obs['extra']:
                        data = obs['extra'][k][:]
                        # Handle boolean arrays
                        if data.dtype == bool:
                            data = data.astype(np.float32)
                        # Ensure 2D shape
                        if len(data.shape) == 1:
                            data = data.reshape(-1, 1)
                        state_parts.append(data)
            
            # Concatenate state
            state = np.concatenate(state_parts, axis=-1).astype(np.float32)
            
            # Extract RGB images
            rgb = None
            if 'sensor_data' in obs:
                rgb_parts = []
                for cam_name in obs['sensor_data'].keys():
                    if 'rgb' in obs['sensor_data'][cam_name]:
                        rgb_parts.append(obs['sensor_data'][cam_name]['rgb'][:])
                if rgb_parts:
                    # Concatenate along channel dimension if multiple cameras
                    # Shape: (L, H, W, C*num_cameras)
                    rgb = np.concatenate(rgb_parts, axis=-1)
            
            # Store observations
            obs_dict = {
                'state': state,
                'rgb': rgb
            }
            trajectories['observations'].append(obs_dict)
            
            # Extract actions
            actions = traj['actions'][:].astype(np.float32)
            trajectories['actions'].append(actions)
    
    print(f"Loaded {len(trajectories['actions'])} trajectories")
    print(f"State dim: {trajectories['observations'][0]['state'].shape[-1]}")
    if trajectories['observations'][0]['rgb'] is not None:
        print(f"RGB shape: {trajectories['observations'][0]['rgb'].shape[1:]}")
    
    return trajectories


class SmallDemoDataset_StateRGB(Dataset):
    """Dataset for State+RGB demonstrations.
    
    Handles multi-modal observations with both state vectors and RGB images.
    """
    
    def __init__(self, data_path, device, num_traj, obs_horizon, pred_horizon, control_mode, state_keys=None):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        
        # Load dataset with optional state key filtering
        trajectories = load_state_rgb_demo_dataset(data_path, num_traj=num_traj, state_keys=state_keys)
        
        # Move data to device
        for i in range(len(trajectories['observations'])):
            obs = trajectories['observations'][i]
            obs['state'] = torch.from_numpy(obs['state']).to(device)
            if obs['rgb'] is not None:
                # Keep RGB as uint8 to save memory, convert to float during training
                obs['rgb'] = torch.from_numpy(obs['rgb']).to(device)
            trajectories['actions'][i] = torch.from_numpy(
                trajectories['actions'][i]
            ).to(device)
        
        # Determine action padding for delta controllers
        if 'delta_pos' in control_mode or control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros(
                (trajectories['actions'][0].shape[1] - 1,), device=device
            )
        else:
            self.pad_action_arm = torch.zeros(
                (trajectories['actions'][0].shape[1] - 1,), device=device
            )
        
        # Pre-compute slices
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories
        self.has_rgb = trajectories['observations'][0]['rgb'] is not None
    
    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories['actions'][traj_idx].shape
        
        obs_traj = self.trajectories['observations'][traj_idx]
        
        # Get state sequence
        state_seq = obs_traj['state'][max(0, start):start + self.obs_horizon]
        if start < 0:
            pad_state = state_seq[0:1].repeat(-start, 1)
            state_seq = torch.cat((pad_state, state_seq), dim=0)
        
        # Get RGB sequence
        rgb_seq = None
        if self.has_rgb:
            rgb_seq = obs_traj['rgb'][max(0, start):start + self.obs_horizon]
            if start < 0:
                pad_rgb = rgb_seq[0:1].repeat(-start, 1, 1, 1)
                rgb_seq = torch.cat((pad_rgb, rgb_seq), dim=0)
        
        # Get action sequence
        act_seq = self.trajectories['actions'][traj_idx][max(0, start):end]
        if start < 0:
            act_seq = torch.cat([act_seq[0:1].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            gripper_action = act_seq[-1, -1:]
            pad_action = torch.cat((self.pad_action_arm, gripper_action), dim=0)
            act_seq = torch.cat([act_seq, pad_action.unsqueeze(0).repeat(end - L, 1)], dim=0)
        
        assert state_seq.shape[0] == self.obs_horizon
        assert act_seq.shape[0] == self.pred_horizon
        
        obs_dict = {'state': state_seq}
        if rgb_seq is not None:
            obs_dict['rgb'] = rgb_seq
        
        return {
            'observations': obs_dict,
            'actions': act_seq,
        }

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    """Diffusion Policy Agent with State+RGB multi-modal observations."""
    
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        
        # Get observation and action dimensions
        obs_space = env.single_observation_space
        assert len(env.single_action_space.shape) == 1
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        
        self.act_dim = env.single_action_space.shape[0]
        
        # State dimension
        self.state_dim = obs_space["state"].shape[1]  # (obs_horizon, state_dim)
        
        # Check for RGB
        self.include_rgb = "rgb" in obs_space.keys()
        
        # Visual encoder
        self.visual_feature_dim = args.visual_feature_dim if self.include_rgb else 0
        if self.include_rgb:
            # Get number of channels (could be multiple cameras concatenated)
            rgb_shape = obs_space["rgb"].shape  # (obs_horizon, H, W, C)
            total_channels = rgb_shape[-1]
            
            self.visual_encoder = PlainConv(
                in_channels=total_channels,
                out_dim=self.visual_feature_dim,
                pool_feature_map=True
            )
        else:
            self.visual_encoder = None
        
        # Total observation dimension for conditioning
        obs_cond_dim = self.obs_horizon * (self.state_dim + self.visual_feature_dim)
        
        # Noise prediction network (U-Net)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=obs_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )

    def encode_obs(self, obs_seq, eval_mode=False):
        """Encode observations into conditioning vector.
        
        Args:
            obs_seq: dict with 'state' (B, obs_horizon, state_dim) 
                     and optionally 'rgb' (B, obs_horizon, H, W, C)
            eval_mode: whether in evaluation mode (affects augmentation)
        
        Returns:
            Conditioning vector of shape (B, obs_horizon * (state_dim + visual_dim))
        """
        batch_size = obs_seq['state'].shape[0]
        features = []
        
        # Process state
        state_feat = obs_seq['state']  # (B, obs_horizon, state_dim)
        features.append(state_feat)
        
        # Process RGB if available
        if self.include_rgb and 'rgb' in obs_seq:
            rgb = obs_seq['rgb']  # (B, obs_horizon, H, W, C)
            
            # Convert to float and normalize
            rgb = rgb.float() / 255.0
            
            # Reshape for visual encoder: (B*obs_horizon, C, H, W)
            rgb = rgb.permute(0, 1, 4, 2, 3)  # (B, obs_horizon, C, H, W)
            rgb = rgb.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
            
            # Encode
            visual_feat = self.visual_encoder(rgb)  # (B*obs_horizon, visual_dim)
            visual_feat = visual_feat.reshape(
                batch_size, self.obs_horizon, self.visual_feature_dim
            )  # (B, obs_horizon, visual_dim)
            
            features.append(visual_feat)
        
        # Concatenate features along the last dimension
        combined = torch.cat(features, dim=-1)  # (B, obs_horizon, state_dim + visual_dim)
        
        # Flatten for conditioning
        return combined.flatten(start_dim=1)  # (B, obs_horizon * total_dim)

    def compute_loss(self, obs_seq, action_seq):
        """Compute diffusion training loss."""
        B = obs_seq['state'].shape[0]
        
        # observation as FiLM conditioning
        obs_cond = self.encode_obs(
            obs_seq, eval_mode=False
        )  # (B, obs_horizon * obs_dim)
        
        # Sample noise
        noise = torch.randn(
            (B, self.pred_horizon, self.act_dim), 
            device=obs_seq['state'].device
        )
        
        # Sample diffusion timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=obs_seq['state'].device
        ).long()
        
        # Add noise to actions
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        
        # Predict noise
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )
        
        return F.mse_loss(noise_pred, noise)
    
    def get_action(self, obs_seq):
        """Generate actions via diffusion denoising.
        
        Args:
            obs_seq: dict with 'state' and optionally 'rgb'
        
        Returns:
            Actions of shape (B, act_horizon, act_dim)
        """
        B = obs_seq['state'].shape[0]
        device = obs_seq['state'].device
        
        with torch.no_grad():
            # Handle permutation for evaluation (env returns H, W, C format)
            if self.include_rgb and 'rgb' in obs_seq:
                # Check if we need to permute (evaluation uses H, W, C format)
                if obs_seq['rgb'].shape[-1] != obs_seq['rgb'].shape[-2]:
                    # Already in (B, obs_horizon, H, W, C) format for eval
                    pass
            
            # Encode observations
            obs_cond = self.encode_obs(obs_seq, eval_mode=True)
            
            # Initialize from noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=device
            )
            
            # Denoise
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample
        
        # Extract action horizon
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]


def save_ckpt(run_name, tag, agent, ema_agent, ema):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save(
        {
            "agent": agent.state_dict(),
            "ema_agent": ema_agent.state_dict(),
        },
        f"runs/{run_name}/checkpoints/{tag}.pt",
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[:-len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    
    # Validate demo path and control mode
    args.demo_path = os.path.expanduser(args.demo_path)
    if args.demo_path.endswith(".h5"):
        import json
        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert control_mode == args.control_mode, \
                f"Control mode mismatch. Dataset: {control_mode}, Args: {args.control_mode}"
    
    # Validate horizon settings
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Create evaluation environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="rgbd",  # Use rgbd to get both state and images
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )
    assert args.max_episode_steps is not None, \
        "max_episode_steps must be specified for imitation learning"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[partial(FlattenRGBDObservationWrapper, rgb=True, depth=False, state=True)],
    )
    
    # Get state keys from a temporary environment to ensure consistency
    # between dataset loading and environment observations
    tmp_env = gym.make(args.env_id, obs_mode="rgbd", **{k: v for k, v in env_kwargs.items() if k not in ['obs_mode', 'render_mode', 'human_render_camera_configs']})
    tmp_obs, _ = tmp_env.reset()
    state_keys = {
        'agent': list(tmp_obs['agent'].keys()),
        'extra': list(tmp_obs['extra'].keys())
    }
    tmp_env.close()
    print(f"Using state keys from environment: {state_keys}")
    
    # Logging
    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(
            **env_kwargs, 
            num_envs=args.num_eval_envs, 
            env_id=args.env_id, 
            env_horizon=args.max_episode_steps
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy-StateRGB",
            tags=["diffusion_policy", "state_rgb", "multimodal"],
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )
    
    # Create dataset
    dataset = SmallDemoDataset_StateRGB(
        data_path=args.demo_path,
        device=device,
        num_traj=args.num_demos,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        control_mode=args.control_mode,
        state_keys=state_keys,
    )
    
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )

    agent = Agent(envs, args).to(device)
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()) / 1e6:.2f}M")
    
    # Optimizer
    optimizer = optim.AdamW(
        params=agent.parameters(),
        lr=args.lr,
        betas=(0.95, 0.999),
        weight_decay=1e-6
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    # define evaluation and logging functions
    def evaluate_and_save_best(iteration):
        if iteration % args.eval_freq == 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate(
                args.num_eval_episodes, ema_agent, envs, device, args.sim_backend
            )
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}", agent, ema_agent, ema)
                    print(f"New best {k}: {eval_metrics[k]:.4f}. Saving checkpoint.")
    
    def log_metrics(iteration):
        if iteration % args.log_freq == 0:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()
    for iteration, data_batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick

        # forward and compute loss
        last_tick = time.time()
        total_loss = agent.compute_loss(
            obs_seq=data_batch["observations"],
            action_seq=data_batch["actions"],
        )
        timings["forward"] += time.time() - last_tick

        # backward
        last_tick = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()  # step lr scheduler every batch, this is different from standard pytorch behavior
        timings["backward"] += time.time() - last_tick

        # ema step
        last_tick = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - last_tick

        # Evaluation
        evaluate_and_save_best(iteration)
        log_metrics(iteration)

        # Checkpoint
        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration), agent, ema_agent, ema)
        
        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        last_tick = time.time()

    evaluate_and_save_best(args.total_iters)
    log_metrics(args.total_iters)

    envs.close()
    writer.close()
