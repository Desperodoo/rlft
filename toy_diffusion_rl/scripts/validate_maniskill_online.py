#!/usr/bin/env python3
"""
Online Fine-tuning Validation for DPPO and ReinFlow on ManiSkill3 Tasks.

This script performs online RL fine-tuning for DPPO and ReinFlow agents that have been
pre-trained with behavior cloning on offline data. It uses the same VecEnv data 
collection pipeline as generate_maniskill_dataset.py to ensure consistency.

Supported Tasks:
- PickCube-v1: Pick up a cube and place it at a goal position
- PegInsertionSide-v1: Insert a peg into a box from the side

Key Features:
- Loads pre-trained checkpoints from offline training (validate_maniskill_offline.py)
- Uses ManiSkillVectorEnv for parallel data collection (same as dataset generation)
- Properly handles observation normalization (state) and action denormalization
- Supports auto_reset vectorized environment with correct episode boundary handling
- Records training metrics and periodic evaluations

Algorithms:
- DPPO: Partial chain fine-tuning (last K denoising steps trainable)
- ReinFlow: Noisy flow matching with learnable exploration noise

Usage:
    # Fine-tune from a checkpoint on PickCube
    python scripts/validate_maniskill_online.py \
        --task PickCube-v1 \
        --checkpoint_dir ./results/maniskill_checkpoints_YYYYMMDD_HHMMSS \
        --obs_mode state_image \
        --num_iters 100 \
        --rollout_steps 2048 \
        --num_envs 20

    # Fine-tune on PegInsertionSide
    python scripts/validate_maniskill_online.py \
        --task PegInsertionSide-v1 \
        --checkpoint_dir ./results/maniskill_peginsertionside_checkpoints_YYYYMMDD \
        --obs_mode state_image \
        --num_iters 200

    # Fine-tune specific algorithm only
    python scripts/validate_maniskill_online.py \
        --task PickCube-v1 \
        --checkpoint_dir ./results/maniskill_checkpoints_YYYYMMDD_HHMMSS \
        --algorithm DPPO \
        --num_iters 100

Note:
    This script requires the rlft_ms3 conda environment with ManiSkill3 installed.
    Pre-trained checkpoints must include normalizer_state for proper action/state handling.
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
import json
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
workspace_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, workspace_dir)

# Import ManiSkill environment
from toy_diffusion_rl.envs.maniskill_env import make_maniskill_env, check_maniskill_available, TASK_DEFAULTS

# Import agents
from toy_diffusion_rl.algorithms.dppo.agent import DPPOAgent
from toy_diffusion_rl.algorithms.reinflow.agent import ReinFlowAgent
from toy_diffusion_rl.common.normalizer import DataNormalizer

# Import evaluate_agent from offline validation script (reuse with video recording support)
from toy_diffusion_rl.scripts.validate_maniskill_offline import evaluate_agent


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def to_numpy(x):
    """Convert tensor or array to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy().copy()
    elif isinstance(x, np.ndarray):
        return x.copy()
    else:
        return np.array(x)


def load_checkpoint_info(checkpoint_dir: str, algo_name: str) -> Dict[str, Any]:
    """Load checkpoint info for a specific algorithm.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        algo_name: Algorithm name (DPPO or ReinFlow)
        
    Returns:
        Dictionary with checkpoint info including normalizer_state
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"{algo_name}_best.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Check for normalizer state in checkpoint
    info = {
        "path": checkpoint_path,
        "checkpoint": checkpoint,
    }
    
    # Look for normalizer state (could be in checkpoint or in a separate metadata file)
    metadata_path = checkpoint_path.replace(".pt", "_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if "normalizer_state" in metadata:
                info["normalizer_state"] = metadata["normalizer_state"]
    
    return info


def create_agent(
    algo_name: str,
    action_dim: int,
    state_dim: int,
    image_shape: Tuple[int, int, int],
    obs_mode: str,
    device: str,
    checkpoint_path: Optional[str] = None,
) -> Union[DPPOAgent, ReinFlowAgent]:
    """Create and optionally load agent.
    
    Args:
        algo_name: Algorithm name (DPPO or ReinFlow)
        action_dim: Action dimension
        state_dim: State dimension
        image_shape: Image shape (H, W, C)
        obs_mode: Observation mode
        device: Device to use
        checkpoint_path: Optional path to load checkpoint
        
    Returns:
        Agent instance
    """
    common_kwargs = {
        "action_dim": action_dim,
        "obs_mode": obs_mode,
        "device": device,
        "hidden_dims": [256, 256],
        "batch_size": 512,
    }
    
    if obs_mode in ["state", "state_image"]:
        common_kwargs["state_dim"] = state_dim
    if obs_mode == "state_image":
        common_kwargs["image_shape"] = image_shape
    
    if algo_name == "DPPO":
        agent = DPPOAgent(
            num_diffusion_steps=100,
            ft_denoising_steps=5,
            **common_kwargs
        )
    elif algo_name == "ReinFlow":
        agent = ReinFlowAgent(
            num_flow_steps=10,
            noise_scheduler_type="learn",
            **common_kwargs
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint: {checkpoint_path}")
        agent.load(checkpoint_path)
    
    return agent


def collect_rollout_parallel(
    agent: Union[DPPOAgent, ReinFlowAgent],
    env,
    rollout_steps: int,
    obs_mode: str,
    num_envs: int,
    normalizer: Optional[DataNormalizer] = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Collect rollout data using parallel VecEnv.
    
    This function collects data in a format suitable for PPO/GAE computation.
    Data is stored in (num_steps, num_envs, ...) format to preserve trajectory
    structure for correct advantage estimation.
    
    Key features:
    - Uses VecEnv with auto_reset=True
    - Maintains per-environment trajectory structure for GAE
    - Handles state normalization and action denormalization via normalizer
    - Handles episode boundaries correctly with auto_reset
    
    Args:
        agent: DPPO or ReinFlow agent
        env: ManiSkill3 VecEnv (already created with ManiSkillVectorEnv wrapper)
        rollout_steps: Number of steps per environment to collect
        obs_mode: Observation mode ("state" or "state_image")
        num_envs: Number of parallel environments
        normalizer: Optional normalizer for state/action normalization
        verbose: Print progress info
        
    Returns:
        Buffer dictionary with collected experience, ready for agent.update()
    """
    # Calculate steps per env
    steps_per_env = rollout_steps // num_envs
    if steps_per_env < 1:
        steps_per_env = 1
    
    # Storage arrays - shape: (steps_per_env, num_envs, ...)
    # This preserves trajectory structure for correct GAE computation
    states_buffer = []
    images_buffer = []
    actions_buffer = []
    rewards_buffer = []
    values_buffer = []
    log_probs_buffer = []
    dones_buffer = []
    
    # Per-environment tracking for statistics
    env_rewards = np.zeros(num_envs)
    env_lengths = np.zeros(num_envs, dtype=np.int32)
    completed_episodes = 0
    total_reward = 0.0
    
    # Initial reset
    obs, info = env.reset()
    
    # Detect device from observation
    if obs_mode == "state":
        device = obs.device if hasattr(obs, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = obs["state"].device if hasattr(obs.get("state"), 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pbar = tqdm(total=steps_per_env, desc="Collecting rollout") if verbose else None
    
    for step in range(steps_per_env):
        # ======== Prepare observation for agent ========
        if obs_mode == "state":
            state_tensor = obs
            if normalizer is not None:
                state_normalized = normalizer.normalize_state(state_tensor)
            else:
                state_normalized = state_tensor
            agent_obs = state_normalized
            state_np = to_numpy(state_normalized)
        else:
            state_tensor = obs["state"]
            image_tensor = obs.get("rgb", obs.get("image"))
            
            if normalizer is not None:
                state_normalized = normalizer.normalize_state(state_tensor)
            else:
                state_normalized = state_tensor
            
            if image_tensor is not None:
                if image_tensor.ndim == 4:  # (N, H, W, C)
                    image_nchw = image_tensor.permute(0, 3, 1, 2).contiguous()
                else:
                    image_nchw = image_tensor
                
                if image_nchw.dtype == torch.uint8:
                    image_normalized = image_nchw.float() / 255.0
                elif image_nchw.max() > 1.0:
                    image_normalized = image_nchw / 255.0
                else:
                    image_normalized = image_nchw
            else:
                image_normalized = None
            
            agent_obs = {"state": state_normalized, "image": image_normalized}
            state_np = to_numpy(state_normalized)
            image_np = to_numpy(image_normalized) if image_normalized is not None else None
        
        # ======== Sample action from agent ========
        with torch.no_grad():
            action, log_prob, value = agent.sample_action(agent_obs)
        
        # Convert action to tensor
        if isinstance(action, np.ndarray):
            action_tensor = torch.from_numpy(action).float().to(device)
        else:
            action_tensor = action.to(device)
        
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.shape[0] != num_envs:
            action_tensor = action_tensor.expand(num_envs, -1)
        
        action_np = to_numpy(action_tensor)
        
        # Denormalize action for env
        if normalizer is not None:
            action_for_env = normalizer.unnormalize_action(action_tensor)
        else:
            action_for_env = action_tensor
        
        # ======== Store data for this step (all envs) ========
        states_buffer.append(state_np.copy())
        if obs_mode != "state":
            images_buffer.append(image_np.copy())
        actions_buffer.append(action_np.copy())
        
        # Handle batched log_prob and value
        if isinstance(log_prob, np.ndarray):
            if log_prob.ndim == 0:
                log_prob_np = np.full(num_envs, float(log_prob))
            else:
                log_prob_np = log_prob.copy()
        else:
            log_prob_np = np.full(num_envs, float(log_prob))
        
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                value_np = np.full(num_envs, float(value))
            else:
                value_np = value.flatten().copy()
        else:
            value_np = np.full(num_envs, float(value))
        
        log_probs_buffer.append(log_prob_np)
        values_buffer.append(value_np)
        
        # ======== Step environment ========
        next_obs, rewards, terminated, truncated, info = env.step(action_for_env)
        
        # Convert to numpy
        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy()
        else:
            rewards_np = np.array(rewards)
        
        if isinstance(terminated, torch.Tensor):
            terminated_np = terminated.cpu().numpy()
        else:
            terminated_np = np.array(terminated)
        
        if isinstance(truncated, torch.Tensor):
            truncated_np = truncated.cpu().numpy()
        else:
            truncated_np = np.array(truncated)
        
        dones = terminated_np | truncated_np
        
        rewards_buffer.append(rewards_np.copy())
        dones_buffer.append(dones.copy())
        
        # Track statistics
        env_rewards += rewards_np
        env_lengths += 1
        
        for i in range(num_envs):
            if dones[i]:
                total_reward += env_rewards[i]
                completed_episodes += 1
                env_rewards[i] = 0
                env_lengths[i] = 0
        
        obs = next_obs
        
        if pbar:
            pbar.update(1)
    
    if pbar:
        pbar.close()
    
    # ======== Compute last value for GAE (per environment) ========
    with torch.no_grad():
        if obs_mode == "state":
            if normalizer is not None:
                final_obs = normalizer.normalize_state(obs)
            else:
                final_obs = obs
        else:
            state_tensor = obs["state"]
            image_tensor = obs.get("rgb", obs.get("image"))
            
            if normalizer is not None:
                state_normalized = normalizer.normalize_state(state_tensor)
            else:
                state_normalized = state_tensor
            
            if image_tensor is not None:
                if image_tensor.ndim == 4:
                    image_nchw = image_tensor.permute(0, 3, 1, 2).contiguous()
                else:
                    image_nchw = image_tensor
                if image_nchw.dtype == torch.uint8:
                    image_normalized = image_nchw.float() / 255.0
                elif image_nchw.max() > 1.0:
                    image_normalized = image_nchw / 255.0
                else:
                    image_normalized = image_nchw
            else:
                image_normalized = None
            
            final_obs = {"state": state_normalized, "image": image_normalized}
        
        _, _, last_values = agent.sample_action(final_obs)
        if isinstance(last_values, np.ndarray):
            if last_values.ndim == 0:
                last_values = np.full(num_envs, float(last_values))
            else:
                last_values = last_values.flatten()
        else:
            last_values = np.full(num_envs, float(last_values))
    
    # ======== Convert to arrays: (steps_per_env, num_envs, ...) ========
    states_arr = np.array(states_buffer)  # (T, N, state_dim)
    actions_arr = np.array(actions_buffer)  # (T, N, action_dim)
    rewards_arr = np.array(rewards_buffer)  # (T, N)
    values_arr = np.array(values_buffer)  # (T, N)
    log_probs_arr = np.array(log_probs_buffer)  # (T, N)
    dones_arr = np.array(dones_buffer)  # (T, N)
    
    if images_buffer:
        images_arr = np.array(images_buffer)  # (T, N, C, H, W)
    
    # ======== Compute GAE per environment ========
    advantages = np.zeros_like(rewards_arr)  # (T, N)
    
    for env_idx in range(num_envs):
        last_gae = 0.0
        for t in reversed(range(steps_per_env)):
            if t == steps_per_env - 1:
                next_value = last_values[env_idx]
            else:
                next_value = values_arr[t + 1, env_idx]
            
            next_non_terminal = 1.0 - float(dones_arr[t, env_idx])
            delta = rewards_arr[t, env_idx] + agent.gamma * next_value * next_non_terminal - values_arr[t, env_idx]
            advantages[t, env_idx] = last_gae = delta + agent.gamma * agent.gae_lambda * next_non_terminal * last_gae
    
    returns_arr = advantages + values_arr
    
    # ======== Flatten to (T*N, ...) for PPO update ========
    buffer = {
        "states": states_arr.reshape(-1, states_arr.shape[-1]),  # (T*N, state_dim)
        "actions": actions_arr.reshape(-1, actions_arr.shape[-1]),  # (T*N, action_dim)
        "rewards": rewards_arr.flatten(),  # (T*N,)
        "values": values_arr.flatten(),  # (T*N,)
        "log_probs": log_probs_arr.flatten(),  # (T*N,)
        "dones": dones_arr.flatten(),  # (T*N,)
        "advantages": advantages.flatten(),  # (T*N,)
        "returns": returns_arr.flatten(),  # (T*N,)
        "last_value": float(last_values.mean()),  # For compatibility
    }
    
    if images_buffer:
        # images: (T, N, C, H, W) -> (T*N, C, H, W)
        buffer["images"] = images_arr.reshape(-1, *images_arr.shape[2:])
    
    # Statistics
    avg_reward = total_reward / max(completed_episodes, 1)
    if verbose:
        print(f"  Collected {steps_per_env * num_envs} steps, {completed_episodes} episodes, avg_reward={avg_reward:.2f}")
    
    return buffer


def online_finetune(
    algo_name: str,
    agent: Union[DPPOAgent, ReinFlowAgent],
    train_env,
    eval_env,
    normalizer: DataNormalizer,
    num_iters: int,
    rollout_steps: int,
    num_envs: int,
    obs_mode: str,
    eval_interval: int,
    eval_episodes: int,
    save_interval: int,
    save_dir: str,
    max_episode_steps: int = 50,
    record_video: bool = False,
    video_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, List]:
    """Perform online fine-tuning loop.
    
    Args:
        algo_name: Algorithm name for logging
        agent: DPPO or ReinFlow agent
        train_env: Training VecEnv
        eval_env: Evaluation VecEnv
        normalizer: Data normalizer
        num_iters: Number of PPO update iterations
        rollout_steps: Steps per rollout
        num_envs: Number of parallel training envs
        obs_mode: Observation mode
        eval_interval: Iterations between evaluations
        eval_episodes: Episodes per evaluation
        save_interval: Iterations between checkpoint saves
        save_dir: Directory to save checkpoints
        max_episode_steps: Max steps per episode for evaluation
        record_video: Whether to record evaluation videos
        video_dir: Directory to save videos
        verbose: Print progress
        
    Returns:
        Dictionary with training metrics over iterations
    """
    results = {
        "iters": [],
        "success_rate": [],
        "avg_reward": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
    }
    
    best_success_rate = -1.0
    
    for iter_idx in range(num_iters):
        # ======== Collect rollout ========
        buffer = collect_rollout_parallel(
            agent=agent,
            env=train_env,
            rollout_steps=rollout_steps,
            obs_mode=obs_mode,
            num_envs=num_envs,
            normalizer=normalizer,
            verbose=False,
        )
        
        # ======== Update policy ========
        update_info = agent.update(buffer)
        
        # ======== Logging ========
        if (iter_idx + 1) % eval_interval == 0 or iter_idx == 0:
            # Evaluate using the shared evaluate_agent from validate_maniskill_offline
            eval_metrics = evaluate_agent(
                agent=agent,
                env=eval_env,
                obs_mode=obs_mode,
                num_episodes=eval_episodes,
                max_steps=max_episode_steps,
                use_pretrain_sample=False,  # Use fine-tuned policy for evaluation
                verbose=verbose,
                record_video=record_video,
                video_dir=video_dir,
                algo_name=algo_name,
                eval_step=iter_idx + 1,
                normalizer=normalizer,
            )
            
            results["iters"].append(iter_idx + 1)
            results["success_rate"].append(eval_metrics["success_rate"])
            results["avg_reward"].append(eval_metrics["avg_reward"])
            results["policy_loss"].append(update_info.get("policy_loss", 0.0))
            results["value_loss"].append(update_info.get("value_loss", 0.0))
            results["entropy"].append(update_info.get("entropy", 0.0))
            
            if verbose:
                print(f"  Iter {iter_idx+1}/{num_iters}: "
                      f"Success={eval_metrics['success_rate']:.1%}, "
                      f"Reward={eval_metrics['avg_reward']:.2f}, "
                      f"PolicyLoss={update_info.get('policy_loss', 0.0):.4f}, "
                      f"ValueLoss={update_info.get('value_loss', 0.0):.4f}, "
                      f"Entropy={update_info.get('entropy', 0.0):.4f}")
            
            # Save best checkpoint
            if eval_metrics["success_rate"] > best_success_rate:
                best_success_rate = eval_metrics["success_rate"]
                best_path = os.path.join(save_dir, f"{algo_name}_online_best.pt")
                agent.save(best_path)
                
                # Also save metadata
                metadata = {
                    "iter": iter_idx + 1,
                    "success_rate": best_success_rate,
                    "avg_reward": eval_metrics["avg_reward"],
                    "normalizer_state": normalizer.state_dict(),
                }
                metadata_path = best_path.replace(".pt", "_metadata.json")
                # Convert tensor stats to lists for JSON serialization
                serializable_metadata = {}
                for k, v in metadata.items():
                    if k == "normalizer_state":
                        serializable_metadata[k] = {
                            nk: {sk: sv.tolist() if hasattr(sv, 'tolist') else sv 
                                 for sk, sv in nv.items()} if isinstance(nv, dict) else nv
                            for nk, nv in v.items()
                        }
                    else:
                        serializable_metadata[k] = v
                with open(metadata_path, 'w') as f:
                    json.dump(serializable_metadata, f, indent=2)
                
                if verbose:
                    print(f"    -> New best! Saved to {best_path}")
        
        # ======== Periodic checkpoint ========
        if (iter_idx + 1) % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"{algo_name}_iter_{iter_idx+1:04d}.pt")
            agent.save(ckpt_path)
            if verbose:
                print(f"    Checkpoint saved: {ckpt_path}")
    
    return results


def print_summary(results: Dict[str, Dict[str, List]]):
    """Print summary of online fine-tuning results."""
    print("\n" + "=" * 70)
    print("ONLINE FINE-TUNING SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<15} {'Final Success':>15} {'Final Reward':>15} {'Best Success':>15}")
    print("-" * 70)
    
    for algo_name, metrics in results.items():
        if metrics["success_rate"]:
            final_success = metrics["success_rate"][-1]
            final_reward = metrics["avg_reward"][-1]
            best_success = max(metrics["success_rate"])
            print(f"{algo_name:<15} {final_success:>14.1%} {final_reward:>15.2f} {best_success:>14.1%}")
        else:
            print(f"{algo_name:<15} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
    
    print("=" * 70)


def plot_training_progress(
    results: Dict[str, Dict[str, List]],
    save_path: str,
    title: str = "ManiSkill3 Online Fine-tuning Progress"
):
    """Plot training progress curves."""
    import matplotlib.pyplot as plt
    
    algorithms = list(results.keys())
    if not algorithms:
        print("No results to plot")
        return
    
    colors = {'DPPO': '#1f77b4', 'ReinFlow': '#ff7f0e'}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Success Rate
    ax = axes[0, 0]
    for algo in algorithms:
        if results[algo]["iters"]:
            ax.plot(results[algo]["iters"], 
                   [s * 100 for s in results[algo]["success_rate"]],
                   label=algo, color=colors.get(algo, 'blue'), marker='o', markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Average Reward
    ax = axes[0, 1]
    for algo in algorithms:
        if results[algo]["iters"]:
            ax.plot(results[algo]["iters"], results[algo]["avg_reward"],
                   label=algo, color=colors.get(algo, 'blue'), marker='o', markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Reward")
    ax.set_title("Average Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Policy Loss
    ax = axes[0, 2]
    for algo in algorithms:
        if results[algo]["iters"] and results[algo]["policy_loss"]:
            ax.plot(results[algo]["iters"], results[algo]["policy_loss"],
                   label=algo, color=colors.get(algo, 'blue'), marker='o', markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Policy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Value Loss
    ax = axes[1, 0]
    for algo in algorithms:
        if results[algo]["iters"] and results[algo]["value_loss"]:
            ax.plot(results[algo]["iters"], results[algo]["value_loss"],
                   label=algo, color=colors.get(algo, 'blue'), marker='o', markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value Loss")
    ax.set_title("Value Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Entropy
    ax = axes[1, 1]
    for algo in algorithms:
        if results[algo]["iters"] and results[algo]["entropy"]:
            ax.plot(results[algo]["iters"], results[algo]["entropy"],
                   label=algo, color=colors.get(algo, 'blue'), marker='o', markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy")
    ax.set_title("Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Empty subplot for now
    axes[1, 2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Online fine-tuning for DPPO and ReinFlow on ManiSkill3 tasks"
    )
    
    # Task selection
    parser.add_argument(
        "--task", type=str, default="PickCube-v1",
        choices=["PickCube-v1", "PegInsertionSide-v1"],
        help="ManiSkill3 task name"
    )
    
    # Checkpoint loading
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Directory containing pretrained checkpoints (from validate_maniskill_offline.py)"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None,
        help="Path to dataset (for creating normalizer if not loading from checkpoint)"
    )
    
    # Algorithm selection
    parser.add_argument(
        "--algorithm", type=str, default="all",
        choices=["DPPO", "ReinFlow", "all"],
        help="Which algorithm to fine-tune"
    )
    
    # Environment settings
    parser.add_argument(
        "--obs_mode", type=str, default="state_image",
        choices=["state", "state_image"],
        help="Observation mode"
    )
    parser.add_argument(
        "--num_envs", type=int, default=20,
        help="Number of parallel training environments"
    )
    parser.add_argument(
        "--num_eval_envs", type=int, default=20,
        help="Number of parallel evaluation environments"
    )
    parser.add_argument(
        "--max_episode_steps", type=int, default=None,
        help="Max steps per episode (default: task-dependent)"
    )
    parser.add_argument(
        "--image_size", type=int, default=128,
        help="Image size for state_image mode"
    )
    
    # Training settings
    parser.add_argument("--num_iters", type=int, default=100, help="Number of PPO update iterations")
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--eval_interval", type=int, default=10, help="Iterations between evaluations")
    parser.add_argument("--eval_episodes", type=int, default=20, help="Episodes per evaluation")
    parser.add_argument("--save_interval", type=int, default=50, help="Iterations between checkpoint saves")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--record_video", action="store_true", help="Record evaluation videos")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory to save videos (default: output_dir/videos)")
    
    args = parser.parse_args()
    
    # Check ManiSkill3 availability
    if not check_maniskill_available():
        print("Error: ManiSkill3 is not available.")
        print("Please run this script in the rlft_ms3 conda environment.")
        sys.exit(1)
    
    # Get task defaults
    task_config = TASK_DEFAULTS.get(args.task, {})
    
    # Set max_episode_steps from task defaults if not specified
    if args.max_episode_steps is None:
        args.max_episode_steps = task_config.get("max_episode_steps", 50)
        print(f"Using task default max_episode_steps={args.max_episode_steps} for {args.task}")
    
    # Get control mode from task defaults
    control_mode = task_config.get("control_mode", "pd_ee_delta_pose")
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    set_seed(args.seed)
    
    # Determine algorithms to fine-tune
    if args.algorithm == "all":
        algorithms = ["DPPO", "ReinFlow"]
    else:
        algorithms = [args.algorithm]
    
    # Setup dimensions from task defaults
    state_dim = task_config.get("state_dim", 42)
    action_dim = task_config.get("action_dim", 7)
    image_shape = (args.image_size, args.image_size, 3)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_short = args.task.replace("-v1", "").lower()
    save_dir = os.path.join(args.output_dir, f"online_finetuning_{task_short}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup video directory
    if args.record_video:
        video_dir = args.video_dir if args.video_dir else os.path.join(save_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
    else:
        video_dir = None
    
    print("\n" + "=" * 60)
    print("ManiSkill3 Online Fine-tuning")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Algorithms: {algorithms}")
    print(f"Observation mode: {args.obs_mode}")
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Control mode: {control_mode}")
    print(f"Max episode steps: {args.max_episode_steps}")
    print(f"Number of training envs: {args.num_envs}")
    print(f"Number of eval envs: {args.num_eval_envs}")
    print(f"Rollout steps: {args.rollout_steps}")
    print(f"Number of iterations: {args.num_iters}")
    print(f"Save directory: {save_dir}")
    if args.record_video:
        print(f"Video directory: {video_dir}")
    print("=" * 60)
    
    # Create environments
    print("\nCreating training environment...")
    train_env = make_maniskill_env(
        task=args.task,
        obs_mode=args.obs_mode,
        num_envs=args.num_envs,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        image_size=args.image_size,
    )
    print(f"  Training env created with {args.num_envs} parallel envs")
    
    print("Creating evaluation environment...")
    eval_env = make_maniskill_env(
        task=args.task,
        obs_mode=args.obs_mode,
        num_envs=args.num_eval_envs,
        seed=args.seed + 1000,
        max_episode_steps=args.max_episode_steps,
        image_size=args.image_size,
    )
    print(f"  Eval env created with {args.num_eval_envs} parallel envs")
    
    # Load or create normalizer
    normalizer = None
    if args.checkpoint_dir:
        # Try to load normalizer from checkpoint metadata
        for algo in algorithms:
            try:
                ckpt_info = load_checkpoint_info(args.checkpoint_dir, algo)
                if "normalizer_state" in ckpt_info:
                    normalizer = DataNormalizer(action_mode='limits', obs_mode='limits')
                    normalizer.load_state_dict(ckpt_info["normalizer_state"])
                    normalizer = normalizer.to(device)
                    print(f"Loaded normalizer from {algo} checkpoint")
                    break
            except FileNotFoundError:
                continue
    
    if normalizer is None and args.dataset_path:
        # Create normalizer from dataset
        import h5py
        print(f"Creating normalizer from dataset: {args.dataset_path}")
        with h5py.File(args.dataset_path, 'r') as f:
            actions = f["actions"][:]
            states = f["obs"][:] if "obs" in f else None
        normalizer = DataNormalizer(action_mode='limits', obs_mode='limits')
        normalizer.fit(actions=actions, states=states)
        normalizer = normalizer.to(device)
    
    if normalizer is None:
        print("Warning: No normalizer available. Creating identity normalizer.")
        # Create dummy normalizer that does no transformation
        normalizer = DataNormalizer(action_mode='limits', obs_mode='limits')
        # Fit with dummy data (identity mapping)
        dummy_actions = np.array([[-1.0] * action_dim, [1.0] * action_dim])
        dummy_states = np.array([[-1.0] * state_dim, [1.0] * state_dim])
        normalizer.fit(actions=dummy_actions, states=dummy_states)
        normalizer = normalizer.to(device)
    
    # Run fine-tuning for each algorithm
    all_results = {}
    
    for algo_name in algorithms:
        print(f"\n{'='*60}")
        print(f"Fine-tuning {algo_name}...")
        print(f"{'='*60}")
        
        # Create agent
        checkpoint_path = None
        if args.checkpoint_dir:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{algo_name}_best.pt")
            if not os.path.exists(checkpoint_path):
                print(f"  Warning: Checkpoint not found at {checkpoint_path}")
                print(f"  Training from scratch...")
                checkpoint_path = None
        
        try:
            agent = create_agent(
                algo_name=algo_name,
                action_dim=action_dim,
                state_dim=state_dim,
                image_shape=image_shape,
                obs_mode=args.obs_mode,
                device=device,
                checkpoint_path=checkpoint_path,
            )
            
            # Evaluate before fine-tuning
            print("  Evaluating before fine-tuning...")
            pre_metrics = evaluate_agent(
                agent=agent,
                env=eval_env,
                obs_mode=args.obs_mode,
                num_episodes=args.eval_episodes,
                max_steps=args.max_episode_steps,
                use_pretrain_sample=True,  # Use pretrain sampling before fine-tuning
                verbose=False,
                record_video=args.record_video,
                video_dir=video_dir,
                algo_name=f"{algo_name}_pre",
                eval_step=0,
                normalizer=normalizer,
            )
            print(f"  Pre-finetune: Success={pre_metrics['success_rate']:.1%}, Reward={pre_metrics['avg_reward']:.2f}")
            
            # Run fine-tuning
            print("  Starting online fine-tuning...")
            algo_save_dir = os.path.join(save_dir, algo_name)
            os.makedirs(algo_save_dir, exist_ok=True)
            
            results = online_finetune(
                algo_name=algo_name,
                agent=agent,
                train_env=train_env,
                eval_env=eval_env,
                normalizer=normalizer,
                num_iters=args.num_iters,
                rollout_steps=args.rollout_steps,
                num_envs=args.num_envs,
                obs_mode=args.obs_mode,
                eval_interval=args.eval_interval,
                eval_episodes=args.eval_episodes,
                save_interval=args.save_interval,
                save_dir=algo_save_dir,
                max_episode_steps=args.max_episode_steps,
                record_video=args.record_video,
                video_dir=video_dir,
                verbose=True,
            )
            
            all_results[algo_name] = results
            
        except Exception as e:
            print(f"  Error during {algo_name} fine-tuning: {e}")
            import traceback
            traceback.print_exc()
            all_results[algo_name] = {"iters": [], "success_rate": [], "avg_reward": [], 
                                       "policy_loss": [], "value_loss": [], "entropy": []}
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump({k: {kk: [float(v) for v in vv] for kk, vv in v.items()} 
                   for k, v in all_results.items()}, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Plot progress
    plot_path = os.path.join(save_dir, "training_progress.png")
    plot_training_progress(all_results, plot_path)
    
    print(f"\nOnline fine-tuning complete!")
    print(f"  Results: {results_path}")
    print(f"  Plot: {plot_path}")
    print(f"  Checkpoints: {save_dir}")


if __name__ == "__main__":
    main()
