#!/usr/bin/env python3
"""
Validate All Diffusion/Flow RL Algorithms on ManiSkill3 PickCube Task (Offline)

This script tests all unified algorithm implementations on the ManiSkill3 PickCube-v1
robotics task with state_image (multimodal) observation mode.

Features:
- Video recording of evaluation episodes (first 3 episodes per eval)
- Detailed loss metrics tracking (main_loss, bc_loss, q_loss, critic_loss, etc.)
- Multi-panel training progress visualization

Algorithms (Behavior Cloning / Offline RL):
1. Diffusion Policy (BC baseline)
2. Flow Matching Policy (BC baseline)
3. Consistency Flow Policy (single-step generation)
4. Reflected Flow Policy (bounded action spaces)
5. Diffusion Double Q (Offline RL)
6. CPQL (Offline RL with Consistency Models)
7. DPPO (pretrain only, no online finetuning)
8. ReinFlow (pretrain only, no online finetuning)

Observation modes:
- "state": State vector only (dim=42)
- "state_image": Both state and image (multimodal)

Evaluation:
- Success Rate: Percentage of episodes where cube is successfully picked
- Average Reward: Mean episode reward

Usage:
    # Train and evaluate all algorithms on ManiSkill3 dataset
    python scripts/validate_maniskill_offline.py \
        --dataset_path data/maniskill_pickcube_1000.h5 \
        --obs_mode state_image \
        --num_steps 50000 \
        --eval_interval 5000 \
        --record_video \
        --video_dir ./results/videos
"""

# Filter third-party deprecation warnings before imports
import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*fork_rng without explicitly specifying.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*env\\..*to get variables from other wrappers.*", category=UserWarning)

import os
import sys
import argparse
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
workspace_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, workspace_dir)

# Import ManiSkill environment
from toy_diffusion_rl.envs.maniskill_env import make_maniskill_env, check_maniskill_available

# Import agents
from toy_diffusion_rl.algorithms.diffusion_policy.agent import DiffusionPolicyAgent
from toy_diffusion_rl.algorithms.flow_matching.fm_policy import FlowMatchingPolicy
from toy_diffusion_rl.algorithms.flow_matching.consistency_flow import ConsistencyFlowPolicy
from toy_diffusion_rl.algorithms.flow_matching.reflected_flow import ReflectedFlowPolicy
from toy_diffusion_rl.algorithms.diffusion_double_q.agent import DiffusionDoubleQAgent
from toy_diffusion_rl.algorithms.cpql.agent import CPQLAgent
from toy_diffusion_rl.algorithms.dppo.agent import DPPOAgent
from toy_diffusion_rl.algorithms.reinflow.agent import ReinFlowAgent


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    """Load HDF5 dataset for ManiSkill3 PickCube.
    
    Args:
        path: Path to HDF5 file
        
    Returns:
        Dictionary with:
            - obs: (N, 42) state observations
            - images: (N, H, W, 3) RGB images (if available)
            - actions: (N, 7) actions
            - rewards: (N,) rewards
            - next_obs: (N, 42) next states
            - next_images: (N, H, W, 3) next images (if available)
            - dones: (N,) done flags
    """
    print(f"Loading dataset from {path}...")
    
    with h5py.File(path, 'r') as f:
        # Print metadata
        print(f"  Metadata:")
        for key in f.attrs:
            print(f"    {key}: {f.attrs[key]}")
        
        dataset = {}
        
        # Load fields
        if "obs" in f:
            dataset["obs"] = f["obs"][:]
        if "next_obs" in f:
            dataset["next_obs"] = f["next_obs"][:]
        if "images" in f:
            dataset["images"] = f["images"][:]
        if "next_images" in f:
            dataset["next_images"] = f["next_images"][:]
        
        dataset["actions"] = f["actions"][:]
        dataset["rewards"] = f["rewards"][:]
        dataset["dones"] = f["dones"][:]
        
        print(f"  Dataset size: {len(dataset['actions'])} transitions")
        print(f"  Action shape: {dataset['actions'].shape}")
        if "obs" in dataset:
            print(f"  State shape: {dataset['obs'].shape}")
        if "images" in dataset:
            print(f"  Image shape: {dataset['images'].shape}")
    
    return dataset


def prepare_batch(
    dataset: Dict[str, np.ndarray],
    indices: np.ndarray,
    obs_mode: str,
    device: str
) -> Dict[str, torch.Tensor]:
    """Prepare a batch for training.
    
    Args:
        dataset: Full dataset dict
        indices: Batch indices
        obs_mode: "state" or "state_image"
        device: Target device
        
    Returns:
        Batch dictionary with tensors
    """
    batch = {}
    
    # Actions (always present)
    batch["actions"] = torch.FloatTensor(dataset["actions"][indices]).to(device)
    batch["rewards"] = torch.FloatTensor(dataset["rewards"][indices]).to(device)
    batch["dones"] = torch.FloatTensor(dataset["dones"][indices]).to(device)
    
    if obs_mode == "state":
        batch["states"] = torch.FloatTensor(dataset["obs"][indices]).to(device)
        if "next_obs" in dataset:
            batch["next_states"] = torch.FloatTensor(dataset["next_obs"][indices]).to(device)
    
    else:  # state_image
        batch["states"] = torch.FloatTensor(dataset["obs"][indices]).to(device)
        
        images = dataset["images"][indices].astype(np.float32) / 255.0
        batch["images"] = torch.FloatTensor(images).permute(0, 3, 1, 2).contiguous().to(device)
        
        if "next_obs" in dataset:
            batch["next_states"] = torch.FloatTensor(dataset["next_obs"][indices]).to(device)
        if "next_images" in dataset:
            next_images = dataset["next_images"][indices].astype(np.float32) / 255.0
            batch["next_images"] = torch.FloatTensor(next_images).permute(0, 3, 1, 2).contiguous().to(device)
    
    return batch


def save_video(frames: List[np.ndarray], video_path: str, fps: int = 30) -> None:
    """Save frames as video using OpenCV.
    
    Args:
        frames: List of RGB frames, each (H, W, 3) numpy array
        video_path: Output video path (.mp4)
        fps: Frames per second
    """
    if not frames:
        return
    
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    video_writer.release()


def evaluate_agent(
    agent,
    env,
    obs_mode: str,
    num_episodes: int = 10,
    max_steps: int = 50,
    use_pretrain_sample: bool = False,
    verbose: bool = False,
    record_video: bool = False,
    video_dir: Optional[str] = None,
    algo_name: str = "Agent",
    eval_step: int = 0,
    num_video_episodes: int = 3,
) -> Dict[str, Any]:
    """Evaluate agent on ManiSkill3 environment with optional video recording.
    
    Args:
        agent: Agent to evaluate
        env: ManiSkill3 environment (single env with use_numpy=True)
        obs_mode: Observation mode
        num_episodes: Number of evaluation episodes
        max_steps: Max steps per episode
        use_pretrain_sample: Use sample_action_pretrain for DPPO/ReinFlow
        verbose: Print per-episode info
        record_video: Whether to record videos
        video_dir: Directory to save videos
        algo_name: Algorithm name for video naming
        eval_step: Current training step (for video naming)
        num_video_episodes: Number of episodes to record (hardcoded first N)
        
    Returns:
        Dict with:
            - success_rate: Fraction of successful episodes
            - avg_reward: Average episode reward
            - avg_length: Average episode length
            - episode_rewards: List of per-episode rewards
            - episode_successes: List of per-episode success flags
    """
    successes = 0
    total_reward = 0
    total_length = 0
    episode_rewards = []
    episode_successes = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0
        success = False
        frames = []
        
        # Record video only for first N episodes
        should_record = record_video and ep < num_video_episodes and video_dir is not None
        
        for step in range(max_steps):
            # Capture frame for video (before action)
            if should_record:
                try:
                    frame = env.render()
                    if frame is not None:
                        if isinstance(frame, torch.Tensor):
                            frame = frame.cpu().numpy()
                        if frame.dtype != np.uint8:
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)
                        frames.append(frame)
                except Exception:
                    pass  # Skip if render fails
            
            # Prepare observation for agent
            if obs_mode == "state":
                agent_obs = obs
            else:  # state_image
                agent_obs = {"state": obs["state"], "image": obs["image"]}
            
            # Get action
            with torch.no_grad():
                if use_pretrain_sample and hasattr(agent, 'sample_action_pretrain'):
                    result = agent.sample_action_pretrain(agent_obs)
                else:
                    result = agent.sample_action(agent_obs)
                
                action = result[0] if isinstance(result, tuple) else result
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if action.ndim > 1:
                    action = action[0]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            # Check for success
            if info.get("success", False):
                success = True
            
            if terminated or truncated:
                # Capture final frame
                if should_record:
                    try:
                        frame = env.render()
                        if frame is not None:
                            if isinstance(frame, torch.Tensor):
                                frame = frame.cpu().numpy()
                            if frame.dtype != np.uint8:
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                else:
                                    frame = frame.astype(np.uint8)
                            frames.append(frame)
                    except Exception:
                        pass
                break
        
        total_length += step + 1
        total_reward += ep_reward
        episode_rewards.append(ep_reward)
        episode_successes.append(success)
        
        if success:
            successes += 1
        
        # Save video for this episode
        if should_record and frames:
            status = "success" if success else "fail"
            video_subdir = os.path.join(video_dir, algo_name, f"step_{eval_step:06d}")
            video_path = os.path.join(video_subdir, f"ep_{ep+1:02d}_{status}.mp4")
            save_video(frames, video_path, fps=30)
            if verbose:
                print(f"    Saved video: {video_path}")
        
        if verbose:
            print(f"  Episode {ep+1}: reward={ep_reward:.2f}, length={step+1}, success={success}")
    
    return {
        "success_rate": successes / num_episodes,
        "avg_reward": total_reward / num_episodes,
        "avg_length": total_length / num_episodes,
        "episode_rewards": episode_rewards,
        "episode_successes": episode_successes,
    }


def train_and_evaluate_all(
    dataset: Dict[str, np.ndarray],
    obs_mode: str,
    state_dim: int,
    action_dim: int,
    image_shape: Tuple[int, int, int],
    num_steps: int,
    eval_interval: int,
    eval_episodes: int,
    device: str,
    seed: int = 42,
    checkpoint_dir: Optional[str] = None,
    max_episode_steps: int = 50,
    record_video: bool = False,
    video_dir: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, List]], Dict[str, Dict[str, Any]]]:
    """Train all algorithms and track metrics.
    
    Args:
        dataset: Training dataset
        obs_mode: Observation mode
        state_dim: State dimension
        action_dim: Action dimension
        image_shape: Image shape (H, W, C)
        num_steps: Total training steps
        eval_interval: Steps between evaluations
        eval_episodes: Episodes per evaluation
        device: Device to use
        seed: Random seed
        checkpoint_dir: Directory to save best checkpoints
        max_episode_steps: Max steps per episode
        record_video: Whether to record evaluation videos
        video_dir: Directory to save videos (required if record_video=True)
        
    Returns:
        Tuple of:
        - results: Dict[algo_name -> Dict[metric_name -> List[values]]]
        - best_checkpoints: Dict[algo_name -> Dict with 'state_dict', 'step', 'success_rate', 'use_pretrain']
    """
    results = {}
    best_checkpoints = {}
    batch_size = 128
    n_samples = len(dataset["actions"])
    
    # Setup video directory
    if record_video and video_dir:
        os.makedirs(video_dir, exist_ok=True)
        print(f"Video recording enabled, saving to: {video_dir}")
    
    # Create checkpoint directory
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Temp directory for saving best models during training
    import tempfile
    temp_ckpt_dir = tempfile.mkdtemp(prefix="best_ckpt_ms3_")
    
    # Create evaluation environment (single env with numpy)
    print("Creating evaluation environment...")
    eval_env = make_maniskill_env(
        task="PickCube-v1",
        obs_mode=obs_mode,
        num_envs=1,
        seed=seed + 1000,
        max_episode_steps=max_episode_steps,
        image_size=image_shape[0],
        use_numpy=True,
    )
    print("Evaluation environment created!")
    
    # Common agent kwargs
    common_kwargs = {
        "action_dim": action_dim,
        "obs_mode": obs_mode,
        "device": device,
        "hidden_dims": [256, 256],
    }
    if obs_mode in ["state", "state_image"]:
        common_kwargs["state_dim"] = state_dim
    if obs_mode == "state_image":
        common_kwargs["image_shape"] = image_shape
    
    eval_steps = list(range(0, num_steps + 1, eval_interval))
    if eval_steps[-1] != num_steps:
        eval_steps.append(num_steps)
    
    # ==================== 1. Diffusion Policy ====================
    algo_name = "DiffusionPolicy"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {
        "steps": [], "success_rate": [], "avg_reward": [],
        "main_loss": []  # Track main training loss
    }
    loss_buffer = []  # Buffer to collect losses between evals
    best_success = -1.0
    
    try:
        agent = DiffusionPolicyAgent(
            num_diffusion_steps=100,
            **common_kwargs
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(
                    agent, eval_env, obs_mode, eval_episodes, max_episode_steps,
                    record_video=record_video, video_dir=video_dir,
                    algo_name=algo_name, eval_step=step
                )
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                # Record average loss since last eval
                avg_loss = np.mean(loss_buffer) if loss_buffer else 0.0
                results[algo_name]["main_loss"].append(avg_loss)
                loss_buffer.clear()
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}, Loss={avg_loss:.4f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": False,
                        "agent_class": "DiffusionPolicyAgent",
                        "agent_kwargs": {"num_diffusion_steps": 100, **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                train_info = agent.train_step(batch)
                if train_info and "loss" in train_info:
                    loss_buffer.append(train_info["loss"])
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 2. Flow Matching ====================
    algo_name = "FlowMatching"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {
        "steps": [], "success_rate": [], "avg_reward": [],
        "main_loss": []
    }
    loss_buffer = []
    best_success = -1.0
    
    try:
        agent = FlowMatchingPolicy(
            num_inference_steps=20,
            **common_kwargs
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(
                    agent, eval_env, obs_mode, eval_episodes, max_episode_steps,
                    record_video=record_video, video_dir=video_dir,
                    algo_name=algo_name, eval_step=step
                )
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                avg_loss = np.mean(loss_buffer) if loss_buffer else 0.0
                results[algo_name]["main_loss"].append(avg_loss)
                loss_buffer.clear()
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}, Loss={avg_loss:.4f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": False,
                        "agent_class": "FlowMatchingPolicy",
                        "agent_kwargs": {"num_inference_steps": 20, **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                train_info = agent.train_step(batch)
                if train_info and "loss" in train_info:
                    loss_buffer.append(train_info["loss"])
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 3. Consistency Flow ====================
    algo_name = "ConsistencyFlow"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {
        "steps": [], "success_rate": [], "avg_reward": [],
        "main_loss": []
    }
    loss_buffer = []
    best_success = -1.0
    
    try:
        agent = ConsistencyFlowPolicy(
            num_inference_steps=5,
            flow_batch_ratio=0.7,
            consistency_batch_ratio=0.3,
            **common_kwargs
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(
                    agent, eval_env, obs_mode, eval_episodes, max_episode_steps,
                    record_video=record_video, video_dir=video_dir,
                    algo_name=algo_name, eval_step=step
                )
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                avg_loss = np.mean(loss_buffer) if loss_buffer else 0.0
                results[algo_name]["main_loss"].append(avg_loss)
                loss_buffer.clear()
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}, Loss={avg_loss:.4f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": False,
                        "agent_class": "ConsistencyFlowPolicy",
                        "agent_kwargs": {"num_inference_steps": 5, "flow_batch_ratio": 0.7, "consistency_batch_ratio": 0.3, **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                train_info = agent.train_step(batch)
                if train_info and "loss" in train_info:
                    loss_buffer.append(train_info["loss"])
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 4. Reflected Flow ====================
    algo_name = "ReflectedFlow"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {
        "steps": [], "success_rate": [], "avg_reward": [],
        "main_loss": []
    }
    loss_buffer = []
    best_success = -1.0
    
    try:
        agent = ReflectedFlowPolicy(
            num_inference_steps=20,
            reflection_mode="hard",
            **common_kwargs
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(
                    agent, eval_env, obs_mode, eval_episodes, max_episode_steps,
                    record_video=record_video, video_dir=video_dir,
                    algo_name=algo_name, eval_step=step
                )
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                avg_loss = np.mean(loss_buffer) if loss_buffer else 0.0
                results[algo_name]["main_loss"].append(avg_loss)
                loss_buffer.clear()
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}, Loss={avg_loss:.4f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": False,
                        "agent_class": "ReflectedFlowPolicy",
                        "agent_kwargs": {"num_inference_steps": 20, "reflection_mode": "hard", **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                train_info = agent.train_step(batch)
                if train_info and "loss" in train_info:
                    loss_buffer.append(train_info["loss"])
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 5. Diffusion QL ====================
    algo_name = "DiffusionQL"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {
        "steps": [], "success_rate": [], "avg_reward": [],
        "main_loss": [], "critic_loss": [], "actor_loss": [],
        "bc_loss": [], "q_loss": [], "q_mean": []
    }
    loss_buffers = defaultdict(list)
    best_success = -1.0
    
    try:
        agent = DiffusionDoubleQAgent(
            num_diffusion_steps=100,
            **common_kwargs
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(
                    agent, eval_env, obs_mode, eval_episodes, max_episode_steps,
                    record_video=record_video, video_dir=video_dir,
                    algo_name=algo_name, eval_step=step
                )
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                
                # Aggregate loss metrics
                for key in ["critic_loss", "actor_loss", "bc_loss", "q_loss", "q_mean"]:
                    avg_val = np.mean(loss_buffers[key]) if loss_buffers[key] else 0.0
                    results[algo_name][key].append(avg_val)
                # main_loss is the sum of critic_loss + actor_loss
                main_loss = np.mean(loss_buffers["critic_loss"]) + np.mean(loss_buffers["actor_loss"]) if loss_buffers["critic_loss"] else 0.0
                results[algo_name]["main_loss"].append(main_loss)
                
                # Clear buffers
                for key in loss_buffers:
                    loss_buffers[key].clear()
                
                q_mean_val = results[algo_name]["q_mean"][-1]
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}, Loss={main_loss:.4f}, Q={q_mean_val:.2f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": False,
                        "agent_class": "DiffusionDoubleQAgent",
                        "agent_kwargs": {"num_diffusion_steps": 100, **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                train_info = agent.train_step(batch)
                if train_info:
                    for key in ["critic_loss", "actor_loss", "bc_loss", "q_loss", "q_mean"]:
                        if key in train_info:
                            loss_buffers[key].append(train_info[key])
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 6. CPQL ====================
    algo_name = "CPQL"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {
        "steps": [], "success_rate": [], "avg_reward": [],
        "main_loss": [], "q_loss": [], "flow_loss": [], "consistency_loss": []
    }
    loss_buffers = defaultdict(list)
    best_success = -1.0
    
    try:
        agent = CPQLAgent(
            sigma_max=80.0,
            sigma_min=0.002,
            rho=7.0,
            **common_kwargs
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(
                    agent, eval_env, obs_mode, eval_episodes, max_episode_steps,
                    record_video=record_video, video_dir=video_dir,
                    algo_name=algo_name, eval_step=step
                )
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                
                # Aggregate loss metrics
                for key in ["q_loss", "flow_loss", "consistency_loss"]:
                    avg_val = np.mean(loss_buffers[key]) if loss_buffers[key] else 0.0
                    results[algo_name][key].append(avg_val)
                # main_loss is the sum
                main_loss = sum(np.mean(loss_buffers[k]) if loss_buffers[k] else 0.0 
                               for k in ["q_loss", "flow_loss", "consistency_loss"])
                results[algo_name]["main_loss"].append(main_loss)
                
                # Clear buffers
                for key in loss_buffers:
                    loss_buffers[key].clear()
                
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}, Loss={main_loss:.4f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": False,
                        "agent_class": "CPQLAgent",
                        "agent_kwargs": {"sigma_max": 80.0, "sigma_min": 0.002, "rho": 7.0, **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                train_info = agent.train_step(batch)
                if train_info:
                    for key in ["q_loss", "flow_loss", "consistency_loss"]:
                        if key in train_info:
                            loss_buffers[key].append(train_info[key])
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 7. DPPO (pretrain only) ====================
    algo_name = "DPPO"
    print(f"\n{'='*60}")
    print(f"Training {algo_name} (BC pretrain only)...")
    print(f"{'='*60}")
    
    results[algo_name] = {
        "steps": [], "success_rate": [], "avg_reward": [],
        "main_loss": []
    }
    loss_buffer = []
    best_success = -1.0
    
    try:
        agent = DPPOAgent(
            num_diffusion_steps=100,
            ft_denoising_steps=5,
            **common_kwargs
        )
        
        # Enable gradients for pretraining
        for p in agent.actor.parameters():
            p.requires_grad = True
        dppo_optimizer = torch.optim.Adam(
            list(agent.actor.parameters()) + list(agent.actor_ft.parameters()),
            lr=1e-4
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(
                    agent, eval_env, obs_mode, eval_episodes, max_episode_steps, use_pretrain_sample=True,
                    record_video=record_video, video_dir=video_dir,
                    algo_name=algo_name, eval_step=step
                )
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                avg_loss = np.mean(loss_buffer) if loss_buffer else 0.0
                results[algo_name]["main_loss"].append(avg_loss)
                loss_buffer.clear()
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}, Loss={avg_loss:.4f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": True,
                        "agent_class": "DPPOAgent",
                        "agent_kwargs": {"num_diffusion_steps": 100, "ft_denoising_steps": 5, **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                batch_actions = batch["actions"]
                
                # Get observation features
                state_input = batch.get("states")
                image_input = batch.get("images")
                obs_features = agent.obs_encoder(state=state_input, image=image_input)
                
                # Diffusion BC loss
                t = torch.randint(0, agent.num_diffusion_steps, (len(batch_actions),), device=device)
                noise = torch.randn_like(batch_actions)
                noisy_actions = agent.diffusion.q_sample(batch_actions, t, noise)
                t_normalized = t.float() / agent.num_diffusion_steps
                
                pred_noise = agent.actor(noisy_actions, t_normalized, obs_features=obs_features)
                pred_noise_ft = agent.actor_ft(noisy_actions, t_normalized, obs_features=obs_features)
                loss = F.mse_loss(pred_noise, noise) + F.mse_loss(pred_noise_ft, noise)
                
                dppo_optimizer.zero_grad()
                loss.backward()
                dppo_optimizer.step()
                
                loss_buffer.append(loss.item())
        
        # Freeze actor after pretraining
        for p in agent.actor.parameters():
            p.requires_grad = False
            
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 8. ReinFlow (pretrain only) ====================
    algo_name = "ReinFlow"
    print(f"\n{'='*60}")
    print(f"Training {algo_name} (BC pretrain only)...")
    print(f"{'='*60}")
    
    results[algo_name] = {
        "steps": [], "success_rate": [], "avg_reward": [],
        "main_loss": []
    }
    loss_buffer = []
    best_success = -1.0
    
    try:
        agent = ReinFlowAgent(
            num_flow_steps=10,
            noise_scheduler_type="learn",
            **common_kwargs
        )
        
        # Create optimizer for pretrain
        pretrain_optimizer = torch.optim.Adam(
            agent.policy.base_net.parameters(), lr=1e-4
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(
                    agent, eval_env, obs_mode, eval_episodes, max_episode_steps, use_pretrain_sample=True,
                    record_video=record_video, video_dir=video_dir,
                    algo_name=algo_name, eval_step=step
                )
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                avg_loss = np.mean(loss_buffer) if loss_buffer else 0.0
                results[algo_name]["main_loss"].append(avg_loss)
                loss_buffer.clear()
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}, Loss={avg_loss:.4f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": True,
                        "agent_class": "ReinFlowAgent",
                        "agent_kwargs": {"num_flow_steps": 10, "noise_scheduler_type": "learn", **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                batch_actions = batch["actions"]
                
                # Get observation features
                state_input = batch.get("states")
                image_input = batch.get("images")
                obs_features = agent.obs_encoder(state=state_input, image=image_input)
                
                # Flow matching loss
                x_0 = torch.randn_like(batch_actions)
                t = torch.rand(len(batch_actions), device=device)
                t_expand = t.unsqueeze(-1)
                x_t = (1 - t_expand) * x_0 + t_expand * batch_actions
                target_v = batch_actions - x_0
                pred_v = agent.policy.base_net(x_t, t, obs_features=obs_features)
                loss = F.mse_loss(pred_v, target_v)
                
                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()
                
                loss_buffer.append(loss.item())
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    eval_env.close()
    return results, best_checkpoints


def print_summary(results: Dict[str, Dict[str, List]]):
    """Print summary table of final results."""
    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY (ManiSkill3 PickCube)")
    print("=" * 70)
    print(f"{'Algorithm':<20} {'Success Rate':>15} {'Avg Reward':>15}")
    print("-" * 70)
    
    for algo_name in results:
        if results[algo_name]["success_rate"]:
            final_success = results[algo_name]["success_rate"][-1]
            final_reward = results[algo_name]["avg_reward"][-1]
            print(f"{algo_name:<20} {final_success:>14.1%} {final_reward:>15.2f}")
        else:
            print(f"{algo_name:<20} {'N/A':>15} {'N/A':>15}")
    
    print("=" * 70)


def save_results(results: Dict, output_path: str):
    """Save results to file."""
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for algo, metrics in results.items():
        results_json[algo] = {
            k: [float(v) for v in vals] if isinstance(vals, list) else vals
            for k, vals in metrics.items()
        }
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to {output_path}")


def plot_training_progress(
    results: Dict[str, Dict[str, List]],
    save_path: str,
    title: str = "ManiSkill3 PickCube Training Progress"
):
    """Plot training progress curves for all algorithms with multiple metrics.
    
    Creates a 2x3 grid showing:
    - Success Rate, Average Reward, Main Training Loss (top row)
    - BC/Actor Loss, Critic/Q Loss, Q Value Mean (bottom row)
    """
    # Filter algorithms that have results
    algorithms = [name for name in results.keys() if results[name].get("steps")]
    
    if not algorithms:
        print("No results to plot")
        return
    
    # Color palette
    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
    ]
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # ============ Row 1: Success Rate, Reward, Main Loss ============
    
    # Plot 1: Success Rate
    ax1 = axes[0, 0]
    for i, algo_name in enumerate(algorithms):
        steps = results[algo_name]["steps"]
        success_rates = results[algo_name].get("success_rate", [])
        if steps and success_rates:
            color = colors[i % len(colors)]
            ax1.plot(steps, [s * 100 for s in success_rates], 
                    label=algo_name, color=color, marker='o', markersize=3, linewidth=1.5)
    ax1.set_xlabel("Training Steps", fontsize=11)
    ax1.set_ylabel("Success Rate (%)", fontsize=11)
    ax1.set_title("Success Rate", fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)
    
    # Plot 2: Average Reward
    ax2 = axes[0, 1]
    for i, algo_name in enumerate(algorithms):
        steps = results[algo_name]["steps"]
        avg_rewards = results[algo_name].get("avg_reward", [])
        if steps and avg_rewards:
            color = colors[i % len(colors)]
            ax2.plot(steps, avg_rewards, 
                    label=algo_name, color=color, marker='o', markersize=3, linewidth=1.5)
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("Average Reward", fontsize=11)
    ax2.set_title("Average Reward", fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Main Training Loss
    ax3 = axes[0, 2]
    for i, algo_name in enumerate(algorithms):
        steps = results[algo_name]["steps"]
        main_loss = results[algo_name].get("main_loss", [])
        if steps and main_loss:
            color = colors[i % len(colors)]
            ax3.plot(steps, main_loss, 
                    label=algo_name, color=color, marker='o', markersize=3, linewidth=1.5)
    ax3.set_xlabel("Training Steps", fontsize=11)
    ax3.set_ylabel("Loss", fontsize=11)
    ax3.set_title("Main Training Loss", fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # ============ Row 2: BC/Actor Loss, Critic/Q Loss, Q Mean ============
    
    # Plot 4: BC Loss / Actor Loss (for RL algorithms)
    ax4 = axes[1, 0]
    has_bc_data = False
    for i, algo_name in enumerate(algorithms):
        steps = results[algo_name]["steps"]
        # Try bc_loss first, then actor_loss
        bc_loss = results[algo_name].get("bc_loss", results[algo_name].get("actor_loss", []))
        if steps and bc_loss and any(v != 0 for v in bc_loss):
            color = colors[i % len(colors)]
            ax4.plot(steps, bc_loss, 
                    label=algo_name, color=color, marker='o', markersize=3, linewidth=1.5)
            has_bc_data = True
    ax4.set_xlabel("Training Steps", fontsize=11)
    ax4.set_ylabel("Loss", fontsize=11)
    ax4.set_title("BC / Actor Loss", fontsize=12, fontweight='bold')
    if has_bc_data:
        ax4.legend(loc='best', fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No BC/Actor loss data", ha='center', va='center', transform=ax4.transAxes)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Critic / Q Loss
    ax5 = axes[1, 1]
    has_q_data = False
    for i, algo_name in enumerate(algorithms):
        steps = results[algo_name]["steps"]
        # Try critic_loss first, then q_loss
        q_loss = results[algo_name].get("critic_loss", results[algo_name].get("q_loss", []))
        if steps and q_loss and any(v != 0 for v in q_loss):
            color = colors[i % len(colors)]
            ax5.plot(steps, q_loss, 
                    label=algo_name, color=color, marker='o', markersize=3, linewidth=1.5)
            has_q_data = True
    ax5.set_xlabel("Training Steps", fontsize=11)
    ax5.set_ylabel("Loss", fontsize=11)
    ax5.set_title("Critic / Q Loss", fontsize=12, fontweight='bold')
    if has_q_data:
        ax5.legend(loc='best', fontsize=8)
    else:
        ax5.text(0.5, 0.5, "No Critic/Q loss data\n(BC algorithms only)", ha='center', va='center', transform=ax5.transAxes)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Q Value Mean
    ax6 = axes[1, 2]
    has_qval_data = False
    for i, algo_name in enumerate(algorithms):
        steps = results[algo_name]["steps"]
        q_mean = results[algo_name].get("q_mean", [])
        if steps and q_mean and any(v != 0 for v in q_mean):
            color = colors[i % len(colors)]
            ax6.plot(steps, q_mean, 
                    label=algo_name, color=color, marker='o', markersize=3, linewidth=1.5)
            has_qval_data = True
    ax6.set_xlabel("Training Steps", fontsize=11)
    ax6.set_ylabel("Q Value", fontsize=11)
    ax6.set_title("Q Value Mean", fontsize=12, fontweight='bold')
    if has_qval_data:
        ax6.legend(loc='best', fontsize=8)
    else:
        ax6.text(0.5, 0.5, "No Q value data\n(BC algorithms only)", ha='center', va='center', transform=ax6.transAxes)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate all algorithms on ManiSkill3 PickCube task (offline)"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to HDF5 dataset (e.g., data/maniskill_pickcube_1000.h5)"
    )
    parser.add_argument(
        "--obs_mode", type=str, default="state_image",
        choices=["state", "state_image"],
        help="Observation mode"
    )
    parser.add_argument("--num_steps", type=int, default=50000, help="Training steps")
    parser.add_argument("--eval_interval", type=int, default=5000, help="Steps between evaluations")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Episodes per evaluation")
    parser.add_argument("--max_episode_steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    # New arguments for video recording
    parser.add_argument("--record_video", action="store_true", help="Record evaluation videos (first 3 episodes per eval)")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory to save videos (default: output_dir/videos)")
    
    args = parser.parse_args()
    
    # Check ManiSkill3 availability
    if not check_maniskill_available():
        print("Error: ManiSkill3 is not available.")
        print("Please run this script in the rlft_ms3 conda environment.")
        sys.exit(1)
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    set_seed(args.seed)
    
    # Check dataset
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset not found at {args.dataset_path}")
        print("Generate dataset first using:")
        print("  python scripts/generate_maniskill_dataset.py --num_episodes 1000 --num_envs 100 --output_path data/maniskill_pickcube_1000.h5")
        sys.exit(1)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Get dimensions
    state_dim = dataset["obs"].shape[1] if "obs" in dataset else 42
    action_dim = dataset["actions"].shape[1]
    
    # Determine image shape from dataset
    if "images" in dataset:
        image_shape = dataset["images"].shape[1:]  # (H, W, C)
    else:
        image_shape = (128, 128, 3)
    
    # Setup video directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.record_video:
        video_dir = args.video_dir if args.video_dir else os.path.join(args.output_dir, f"videos_{timestamp}")
    else:
        video_dir = None
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Observation mode: {args.obs_mode}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Image shape: {image_shape}")
    print(f"  Training steps: {args.num_steps}")
    print(f"  Eval interval: {args.eval_interval}")
    print(f"  Eval episodes: {args.eval_episodes}")
    print(f"  Record video: {args.record_video}")
    if video_dir:
        print(f"  Video directory: {video_dir}")
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.output_dir, f"maniskill_checkpoints_{timestamp}")
    
    # Train and evaluate
    results, best_checkpoints = train_and_evaluate_all(
        dataset=dataset,
        obs_mode=args.obs_mode,
        state_dim=state_dim,
        action_dim=action_dim,
        image_shape=image_shape,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        device=device,
        seed=args.seed,
        checkpoint_dir=checkpoint_dir,
        max_episode_steps=args.max_episode_steps,
        record_video=args.record_video,
        video_dir=video_dir,
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"maniskill_pickcube_{args.obs_mode}_{timestamp}.json"
    )
    save_results(results, output_path)
    
    # Save detailed metrics as JSON (includes all loss metrics)
    detailed_path = os.path.join(
        args.output_dir,
        f"detailed_metrics_{timestamp}.json"
    )
    save_results(results, detailed_path)
    print(f"Detailed metrics saved to {detailed_path}")
    
    # Plot training progress
    plot_path = os.path.join(
        args.output_dir,
        f"maniskill_pickcube_{args.obs_mode}_{timestamp}.png"
    )
    plot_training_progress(
        results=results,
        save_path=plot_path,
        title=f"ManiSkill3 PickCube Training Progress ({args.obs_mode} mode)"
    )
    
    # Save best checkpoints info
    if best_checkpoints:
        print("\n" + "=" * 70)
        print("BEST CHECKPOINTS SUMMARY")
        print("=" * 70)
        import shutil
        for algo_name, ckpt in best_checkpoints.items():
            final_path = os.path.join(checkpoint_dir, f"{algo_name}_best.pt")
            if os.path.exists(ckpt["temp_path"]):
                shutil.copy2(ckpt["temp_path"], final_path)
                ckpt["final_path"] = final_path
            print(f"  {algo_name}: step={ckpt['step']}, success={ckpt['success_rate']:.1%} -> {final_path}")
        print("=" * 70)
    
    print("\nValidation complete!")
    print(f"Results: {output_path}")
    print(f"Plot: {plot_path}")
    print(f"Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
