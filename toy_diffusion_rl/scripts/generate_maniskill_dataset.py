#!/usr/bin/env python3
"""
Generate Offline Dataset from ManiSkill3 PickCube Environment.

This script generates an offline dataset for training diffusion/flow policies
on the ManiSkill3 PickCube-v1 task. Supports parallel data collection with
GPU-accelerated environments.

Features:
- Uses ManiSkill3's PickCube-v1 with pd_ee_delta_pose control
- Supports state, image, and state_image observation modes
- Expert policy from maniskill_env.py for demonstration collection
- GPU-parallel collection for efficiency
- Saves to HDF5 format compatible with existing dataset loaders

Usage:
    # Single env collection
    python generate_maniskill_dataset.py --num_episodes 200 --output_path data/maniskill_pickcube.h5
    
    # Parallel collection (faster)
    python generate_maniskill_dataset.py --num_episodes 200 --num_envs 16 --output_path data/maniskill_pickcube.h5

Note:
    This script requires the rlft_ms3 conda environment with ManiSkill3 installed.
"""

# Filter third-party deprecation warnings before imports
import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*fork_rng without explicitly specifying.*", category=UserWarning)

import argparse
import os
import sys
import h5py
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from envs.maniskill_env import (
        make_maniskill_env,
        ManiSkillExpertPolicy,
        check_maniskill_available,
    )
except ImportError:
    print("Error: ManiSkill3 environment module not found.")
    print("Make sure you're in the rlft_ms3 conda environment and ManiSkill3 is installed.")
    sys.exit(1)


def to_numpy(x):
    """Convert tensor or array to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy().copy()
    elif isinstance(x, np.ndarray):
        return x.copy()
    else:
        return np.array(x)


def collect_episodes_single(
    env,
    expert: ManiSkillExpertPolicy,
    num_episodes: int,
    max_episode_steps: int = 50,
    obs_mode: str = "state_image",
    verbose: bool = True,
) -> Dict[str, List]:
    """Collect episodes with single environment.
    
    Returns:
        Dictionary with lists of transitions
    """
    data = {
        "obs": [],
        "next_obs": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "episode_ids": [],
        "timesteps": [],
    }
    
    if obs_mode in ["image", "state_image"]:
        data["images"] = []
        data["next_images"] = []
    
    successes = 0
    
    episodes = tqdm(range(num_episodes), desc="Collecting episodes") if verbose else range(num_episodes)
    
    for ep in episodes:
        obs, info = env.reset()
        expert.reset()
        
        for step in range(max_episode_steps):
            # Get raw obs for expert (contains structured obs with extra.tcp_pose, etc.)
            raw_obs = info.get("raw_obs", obs)
            
            action = expert.get_action(raw_obs, info)
            
            # Store current observation
            if obs_mode == "state":
                data["obs"].append(obs.copy())
            elif obs_mode == "image":
                data["images"].append(obs.copy())
            else:  # state_image
                data["obs"].append(obs["state"].copy())
                data["images"].append(obs["rgb"].copy())
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store next observation
            if obs_mode == "state":
                data["next_obs"].append(next_obs.copy())
            elif obs_mode == "image":
                data["next_images"].append(next_obs.copy())
            else:  # state_image
                data["next_obs"].append(next_obs["state"].copy())
                data["next_images"].append(next_obs["rgb"].copy())
            
            data["actions"].append(action)
            data["rewards"].append(reward)
            data["dones"].append(done)
            data["episode_ids"].append(ep)
            data["timesteps"].append(step)
            
            obs = next_obs
            
            if done:
                if info.get("success", False) or reward > 0:
                    successes += 1
                break
    
    if verbose:
        print(f"\nSuccess rate: {successes}/{num_episodes} = {100*successes/num_episodes:.1f}%")
    
    return data


def collect_episodes_parallel(
    env,
    expert: ManiSkillExpertPolicy,
    num_episodes: int,
    max_episode_steps: int = 50,
    obs_mode: str = "state_image",
    num_envs: int = 16,
    verbose: bool = True,
) -> Dict[str, List]:
    """Collect episodes with parallel environments using ManiSkillVectorEnv.
    
    Uses the official ManiSkillVectorEnv wrapper with auto_reset=True to properly
    handle episode resets. When an episode ends, next_obs is the new episode's
    initial observation, and info["final_observation"] contains the old episode's
    final observation.
    
    Note: The environment should already be wrapped with ManiSkillVectorEnv by
    make_maniskill_env when num_envs > 1.
    """
    # The env should already be properly wrapped by make_maniskill_env
    vec_env = env
    
    data = {
        "obs": [],
        "next_obs": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "episode_ids": [],
        "timesteps": [],
    }
    
    if obs_mode in ["image", "state_image"]:
        data["images"] = []
        data["next_images"] = []
    
    # Per-environment tracking - each env gets a unique episode ID
    env_episode_ids = np.arange(num_envs, dtype=np.int32)  # [0, 1, ..., num_envs-1]
    env_timesteps = np.zeros(num_envs, dtype=np.int32)
    env_experts = [ManiSkillExpertPolicy() for _ in range(num_envs)]
    
    # Next available episode ID
    next_episode_id = num_envs
    completed_episodes = 0
    successes = 0
    
    obs, info = vec_env.reset()
    
    # Detect device from observation
    if obs_mode == "state":
        device = obs.device if hasattr(obs, 'device') else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = obs["state"].device if hasattr(obs.get("state"), 'device') else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Reset all experts
    for exp in env_experts:
        exp.reset()
    
    pbar = tqdm(total=num_episodes, desc="Collecting episodes") if verbose else None
    
    while completed_episodes < num_episodes:
        # For expert policy, we need privileged state info (obj_pose, etc.)
        # Use raw_obs from info which contains structured observation with extra.tcp_pose, etc.
        raw_obs = info.get("raw_obs", obs)
        
        # Get actions from expert policy for each environment
        actions = []
        for i in range(num_envs):
            # Extract single env observation from batch
            single_raw_obs = {}
            if isinstance(raw_obs, dict):
                for key, val in raw_obs.items():
                    if isinstance(val, dict):
                        single_raw_obs[key] = {}
                        for k2, v2 in val.items():
                            if hasattr(v2, '__getitem__') and hasattr(v2, 'shape') and len(v2.shape) > 0:
                                single_raw_obs[key][k2] = v2[i]
                            else:
                                single_raw_obs[key][k2] = v2
                    elif hasattr(val, '__getitem__') and hasattr(val, 'shape') and len(val.shape) > 0:
                        single_raw_obs[key] = val[i]
                    else:
                        single_raw_obs[key] = val
            else:
                single_raw_obs = raw_obs[i] if hasattr(raw_obs, '__getitem__') else raw_obs
            
            action = env_experts[i].get_action(single_raw_obs, info)
            actions.append(action)
        actions = np.stack(actions, axis=0)
        
        # Prepare observations for storage
        if obs_mode == "state":
            state_np = to_numpy(obs)
        elif obs_mode == "image":
            image_np = to_numpy(obs)
        else:  # state_image
            state_np = to_numpy(obs["state"])
            image_np = to_numpy(obs["rgb"])
        
        # Step all environments
        actions_tensor = torch.from_numpy(actions).float().to(device)
        
        next_obs, rewards, terminated, truncated, info = vec_env.step(actions_tensor)
        
        # Convert to numpy
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.cpu().numpy()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.cpu().numpy()
        dones = terminated | truncated
        
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        
        # Prepare next observations for storage
        # Note: With auto_reset=True, when an episode ends, next_obs is already the new episode's
        # initial observation. For simplicity, we use next_obs directly for all cases.
        # This is acceptable for offline RL datasets.
        if obs_mode == "state":
            next_state_np = to_numpy(next_obs)
        elif obs_mode == "image":
            next_image_np = to_numpy(next_obs)
        else:  # state_image
            next_state_np = to_numpy(next_obs["state"])
            next_image_np = to_numpy(next_obs["rgb"])
        
        # Store transitions for each environment
        for i in range(num_envs):
            # Only store if this environment's episode is still needed
            if env_episode_ids[i] < num_episodes:
                # Store current observation
                if obs_mode == "state":
                    data["obs"].append(state_np[i].copy())
                elif obs_mode == "image":
                    data["images"].append(image_np[i].copy())
                else:  # state_image
                    data["obs"].append(state_np[i].copy())
                    data["images"].append(image_np[i].copy())
                
                # Store next observation (always use next_obs for simplicity)
                is_done = dones[i]
                if obs_mode == "state":
                    data["next_obs"].append(next_state_np[i].copy())
                elif obs_mode == "image":
                    data["next_images"].append(next_image_np[i].copy())
                else:
                    data["next_obs"].append(next_state_np[i].copy())
                    data["next_images"].append(next_image_np[i].copy())
                
                data["actions"].append(actions[i].copy())
                data["rewards"].append(float(rewards[i]))
                data["dones"].append(is_done)
                data["episode_ids"].append(int(env_episode_ids[i]))
                data["timesteps"].append(int(env_timesteps[i]))
                
                env_timesteps[i] += 1
                
                # Check if episode ended
                if is_done:
                    # Check success from final_info (for auto_reset VecEnv)
                    # When auto_reset=True, info["success"] is the new episode's success,
                    # while info["final_info"]["success"] is the completed episode's success
                    final_info = info.get("final_info", None)
                    if final_info is not None and "success" in final_info:
                        success_tensor = final_info["success"]
                        if hasattr(success_tensor, '__getitem__'):
                            if success_tensor[i]:
                                successes += 1
                        elif success_tensor:
                            successes += 1
                    else:
                        # Fallback for single env or no auto_reset
                        success_tensor = info.get("success", None)
                        if success_tensor is not None:
                            if hasattr(success_tensor, '__getitem__'):
                                if success_tensor[i]:
                                    successes += 1
                            elif success_tensor:
                                successes += 1
                    
                    completed_episodes += 1
                    if pbar:
                        pbar.update(1)
                    
                    # Assign new episode ID to this environment
                    if next_episode_id < num_episodes:
                        env_episode_ids[i] = next_episode_id
                        next_episode_id += 1
                    else:
                        env_episode_ids[i] = num_episodes  # Mark as done
                    
                    env_timesteps[i] = 0
                    # Reset expert for this environment
                    # next_obs[i] is already the new episode's initial observation
                    env_experts[i].reset()
        
        obs = next_obs
        
        # Early exit if all needed episodes are collected
        if completed_episodes >= num_episodes:
            break
    
    if pbar:
        pbar.close()
    
    if verbose:
        print(f"\nSuccess rate: {successes}/{num_episodes} = {100*successes/num_episodes:.1f}%")
    
    return data


def save_dataset(
    data: Dict[str, List],
    output_path: str,
    obs_mode: str = "state_image",
    compression: str = "gzip",
):
    """Save collected data to HDF5 file.
    
    Args:
        data: Dictionary with lists of transitions
        output_path: Path to save HDF5 file
        obs_mode: Observation mode
        compression: Compression algorithm
    """
    import json
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Compute episode statistics
    episode_ids = np.array(data["episode_ids"])
    timesteps = np.array(data["timesteps"])
    unique_episodes = len(np.unique(episode_ids))
    
    # Sort data by episode_id first, then by timestep within each episode
    # This ensures data is stored in correct episode order for offline RL
    print("Sorting data by episode_id and timestep...")
    sort_indices = np.lexsort((timesteps, episode_ids))  # Sort by episode_id first, then timestep
    
    # Apply sorting to all data fields
    def sort_array(arr):
        return [arr[i] for i in sort_indices]
    
    data = {key: sort_array(val) for key, val in data.items()}
    
    with h5py.File(output_path, "w") as f:
        # Store observations
        if "obs" in data and data["obs"]:
            obs_arr = np.array(data["obs"], dtype=np.float32)
            f.create_dataset("obs", data=obs_arr, compression=compression)
        
        if "next_obs" in data and data["next_obs"]:
            next_obs_arr = np.array(data["next_obs"], dtype=np.float32)
            f.create_dataset("next_obs", data=next_obs_arr, compression=compression)
        
        # Store images (as uint8 to save space)
        if "images" in data and data["images"]:
            images_arr = np.array(data["images"], dtype=np.uint8)
            f.create_dataset("images", data=images_arr, compression=compression)
        
        if "next_images" in data and data["next_images"]:
            next_images_arr = np.array(data["next_images"], dtype=np.uint8)
            f.create_dataset("next_images", data=next_images_arr, compression=compression)
        
        # Store other data
        f.create_dataset("actions", data=np.array(data["actions"], dtype=np.float32), compression=compression)
        f.create_dataset("rewards", data=np.array(data["rewards"], dtype=np.float32), compression=compression)
        f.create_dataset("dones", data=np.array(data["dones"], dtype=bool), compression=compression)
        f.create_dataset("episode_ids", data=np.array(data["episode_ids"], dtype=np.int32), compression=compression)
        f.create_dataset("timesteps", data=np.array(data["timesteps"], dtype=np.int32), compression=compression)
        
        # Store metadata as JSON string for compatibility with load_dataset
        metadata = {
            "obs_mode": obs_mode,
            "num_episodes": unique_episodes,
            "num_transitions": len(data["actions"]),
            "task": "PickCube-v1",
            "control_mode": "pd_ee_delta_pose",
        }
        f.attrs["metadata"] = json.dumps(metadata)
        
        # Also store as individual attrs for backwards compatibility
        f.attrs["obs_mode"] = obs_mode
        f.attrs["num_episodes"] = unique_episodes
        f.attrs["num_transitions"] = len(data["actions"])
        f.attrs["task"] = "PickCube-v1"
        f.attrs["control_mode"] = "pd_ee_delta_pose"
    
    print(f"\nDataset saved to: {output_path}")
    print(f"  - Observation mode: {obs_mode}")
    print(f"  - Number of episodes: {unique_episodes}")
    print(f"  - Number of transitions: {len(data['actions'])}")
    
    if "obs" in data and data["obs"]:
        print(f"  - State dimension: {np.array(data['obs'][0]).shape}")
    if "images" in data and data["images"]:
        print(f"  - Image shape: {np.array(data['images'][0]).shape}")
    print(f"  - Action dimension: {np.array(data['actions'][0]).shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate offline dataset from ManiSkill3 PickCube environment"
    )
    parser.add_argument(
        "--obs_mode",
        type=str,
        default="state_image",
        choices=["state", "image", "state_image"],
        help="Observation mode"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=200,
        help="Number of episodes to collect"
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=50,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments (>1 uses GPU simulation)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Image size for image observations"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/maniskill_pickcube_state_image.h5",
        help="Output path for dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Check ManiSkill3 availability
    if not check_maniskill_available():
        print("Error: ManiSkill3 is not installed.")
        print("Please activate the rlft_ms3 environment and install ManiSkill3:")
        print("  conda activate rlft_ms3")
        print("  pip install mani_skill")
        sys.exit(1)
    
    print("=" * 60)
    print("ManiSkill3 Dataset Generation")
    print("=" * 60)
    print(f"Task: PickCube-v1")
    print(f"Observation mode: {args.obs_mode}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Number of parallel envs: {args.num_envs}")
    print(f"Output path: {args.output_path}")
    print("=" * 60)
    
    # Create environment
    print("\nCreating environment...")
    env = make_maniskill_env(
        task="PickCube-v1",
        obs_mode=args.obs_mode,
        num_envs=args.num_envs,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        image_size=args.image_size,
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
    )
    
    print(f"Environment created successfully!")
    print(f"  - State dim: {getattr(env, 'state_dim', 'N/A')}")
    action_dim = env.action_space.shape[-1]
    print(f"  - Action dim: {action_dim}")
    print(f"  - Image shape: {getattr(env, 'image_shape', 'N/A')}")
    
    # Create expert policy
    expert = ManiSkillExpertPolicy()
    
    # Collect episodes
    print("\nCollecting demonstrations...")
    if args.num_envs == 1:
        data = collect_episodes_single(
            env=env,
            expert=expert,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
            obs_mode=args.obs_mode,
            verbose=True,
        )
    else:
        data = collect_episodes_parallel(
            env=env,
            expert=expert,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
            obs_mode=args.obs_mode,
            num_envs=args.num_envs,
            verbose=True,
        )
    
    # Save dataset
    save_dataset(data, args.output_path, args.obs_mode)
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
