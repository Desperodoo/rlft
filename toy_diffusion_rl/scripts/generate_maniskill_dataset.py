#!/usr/bin/env python3
"""
Generate Offline Dataset from ManiSkill3 PickCube Environment.

This script generates an offline dataset for training diffusion/flow policies
on the ManiSkill3 PickCube-v1 task. Supports parallel data collection with
GPU-accelerated environments.

Features:
- Uses ManiSkill3's PickCube-v1 with pd_ee_delta_pose control
- Supports state, image, and state_image observation modes
- Scripted expert policy for demonstration collection
- GPU-parallel collection for efficiency
- Saves to HDF5 format compatible with existing dataset loaders

Usage:
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
from typing import Dict, List, Tuple, Optional

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


class ManiSkillScriptedExpert:
    """Improved scripted expert policy for ManiSkill3 PickCube task.
    
    Uses privileged state information to compute actions:
    1. Move to pre-grasp position above cube
    2. Lower and close gripper
    3. Lift and move to goal
    4. Lower and release
    
    This policy works with the raw ManiSkill3 observation dict.
    """
    
    def __init__(
        self,
        approach_height: float = 0.08,
        grasp_height: float = 0.02,
        lift_height: float = 0.15,
        position_gain: float = 10.0,
        max_action: float = 1.0,
    ):
        self.approach_height = approach_height
        self.grasp_height = grasp_height
        self.lift_height = lift_height
        self.position_gain = position_gain
        self.max_action = max_action
        
        self.phase = 0
        self.grasp_counter = 0
    
    def reset(self):
        """Reset policy state for new episode."""
        self.phase = 0
        self.grasp_counter = 0
    
    def get_action(
        self,
        raw_obs: Dict,
        env_idx: int = 0,
    ) -> np.ndarray:
        """Compute expert action from raw ManiSkill3 observation.
        
        Args:
            raw_obs: Raw observation dict from ManiSkill3
            env_idx: Environment index for batched observations
        
        Returns:
            Action array (7,) for pd_ee_delta_pose: [dx, dy, dz, dax, day, daz, gripper]
        """
        # Extract positions from observation
        extra = raw_obs.get("extra", {})
        
        # Get TCP (end-effector) position
        tcp_pose = extra.get("tcp_pose", None)
        if tcp_pose is None:
            # Fallback: use agent qpos
            agent = raw_obs.get("agent", {})
            qpos = agent.get("qpos", np.zeros(9))
            if isinstance(qpos, torch.Tensor):
                qpos = qpos.cpu().numpy()
            if qpos.ndim > 1:
                qpos = qpos[env_idx]
            gripper_pos = np.array([0.0, 0.0, 0.1])  # Default
        else:
            if isinstance(tcp_pose, torch.Tensor):
                tcp_pose = tcp_pose.cpu().numpy()
            if tcp_pose.ndim > 1:
                tcp_pose = tcp_pose[env_idx]
            gripper_pos = tcp_pose[:3]
        
        # Get cube position
        obj_pose = extra.get("obj_pose", None)
        if obj_pose is None:
            cube_pos = np.array([0.0, 0.0, 0.02])
        else:
            if isinstance(obj_pose, torch.Tensor):
                obj_pose = obj_pose.cpu().numpy()
            if obj_pose.ndim > 1:
                obj_pose = obj_pose[env_idx]
            cube_pos = obj_pose[:3]
        
        # Get goal position
        goal_pos = extra.get("goal_pos", None)
        if goal_pos is None:
            goal_pos = cube_pos + np.array([0.1, 0.1, 0.1])
        else:
            if isinstance(goal_pos, torch.Tensor):
                goal_pos = goal_pos.cpu().numpy()
            if goal_pos.ndim > 1:
                goal_pos = goal_pos[env_idx]
            if len(goal_pos.shape) > 0 and goal_pos.shape[-1] >= 3:
                goal_pos = goal_pos[:3]
        
        # Initialize action
        action = np.zeros(7, dtype=np.float32)
        
        # Compute position errors
        cube_above = cube_pos.copy()
        cube_above[2] += self.approach_height
        
        cube_grasp = cube_pos.copy()
        cube_grasp[2] += self.grasp_height
        
        goal_above = goal_pos.copy()
        goal_above[2] = max(goal_above[2], cube_pos[2] + self.lift_height)
        
        # Phase 0: Move to pre-grasp position
        if self.phase == 0:
            delta = cube_above - gripper_pos
            action[:3] = self.position_gain * delta
            action[6] = 1.0  # Open gripper
            
            if np.linalg.norm(delta) < 0.02:
                self.phase = 1
        
        # Phase 1: Lower to grasp
        elif self.phase == 1:
            delta = cube_grasp - gripper_pos
            action[:3] = self.position_gain * delta
            action[6] = 1.0  # Keep open
            
            if np.linalg.norm(delta) < 0.015:
                self.phase = 2
                self.grasp_counter = 0
        
        # Phase 2: Close gripper
        elif self.phase == 2:
            action[:3] = 0.0
            action[6] = -1.0  # Close gripper
            self.grasp_counter += 1
            
            if self.grasp_counter > 10:
                self.phase = 3
        
        # Phase 3: Lift and move to goal
        elif self.phase == 3:
            delta = goal_above - gripper_pos
            action[:3] = self.position_gain * delta
            action[6] = -1.0  # Keep closed
            
            if np.linalg.norm(delta[:2]) < 0.02 and delta[2] < 0.02:
                self.phase = 4
        
        # Phase 4: Lower to goal
        elif self.phase == 4:
            delta = goal_pos - gripper_pos
            action[:3] = self.position_gain * delta
            action[6] = -1.0  # Keep closed
            
            if np.linalg.norm(delta) < 0.03:
                self.phase = 5
        
        # Phase 5: Release
        elif self.phase == 5:
            action[:3] = 0.0
            action[6] = 1.0  # Open gripper
        
        # Clip actions
        action[:3] = np.clip(action[:3], -self.max_action, self.max_action)
        action[6] = np.clip(action[6], -1.0, 1.0)
        
        return action


def collect_episodes_single(
    env,
    expert: ManiSkillScriptedExpert,
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
            # Get raw obs for expert
            raw_obs = info.get("raw_obs", obs)
            if isinstance(raw_obs, dict) and "extra" not in raw_obs:
                # If wrapped, try to get from info
                raw_obs = obs
            
            action = expert.get_action(raw_obs)
            
            # Store current observation
            if obs_mode == "state":
                data["obs"].append(obs.copy())
            elif obs_mode == "image":
                data["images"].append(obs.copy())
            else:  # state_image
                data["obs"].append(obs["state"].copy())
                data["images"].append(obs["image"].copy())
            
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
                data["next_images"].append(next_obs["image"].copy())
            
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
    expert: ManiSkillScriptedExpert,
    num_episodes: int,
    max_episode_steps: int = 50,
    obs_mode: str = "state_image",
    num_envs: int = 16,
    verbose: bool = True,
) -> Dict[str, List]:
    """Collect episodes with parallel environments.
    
    Note: This is more complex due to asynchronous episode endings.
    For simplicity, we collect a fixed number of steps and track episodes.
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
    
    # Per-environment tracking
    env_episode_ids = np.zeros(num_envs, dtype=np.int32)
    env_timesteps = np.zeros(num_envs, dtype=np.int32)
    env_experts = [ManiSkillScriptedExpert() for _ in range(num_envs)]
    
    current_episode = 0
    successes = 0
    
    obs, info = env.reset()
    
    # Reset all experts
    for exp in env_experts:
        exp.reset()
    
    steps_per_episode = max_episode_steps
    total_steps_needed = num_episodes * steps_per_episode
    steps_collected = 0
    
    pbar = tqdm(total=num_episodes, desc="Collecting episodes") if verbose else None
    
    while current_episode < num_episodes:
        # Get actions for all environments
        raw_obs = info.get("raw_obs", obs)
        
        actions = []
        for i in range(num_envs):
            action = env_experts[i].get_action(raw_obs, env_idx=i)
            actions.append(action)
        actions = np.stack(actions, axis=0)
        
        # Store current observations (convert Tensor to numpy if needed)
        if obs_mode == "state":
            state = obs if not isinstance(obs, dict) else obs
            state_np = to_numpy(state)
            for i in range(num_envs):
                if env_episode_ids[i] < num_episodes:
                    data["obs"].append(state_np[i].copy() if state_np.ndim > 1 else state_np.copy())
        elif obs_mode == "image":
            image = obs if not isinstance(obs, dict) else obs
            image_np = to_numpy(image)
            for i in range(num_envs):
                if env_episode_ids[i] < num_episodes:
                    data["images"].append(image_np[i].copy() if image_np.ndim > 3 else image_np.copy())
        else:  # state_image
            state_np = to_numpy(obs["state"])
            image_np = to_numpy(obs["image"])
            for i in range(num_envs):
                if env_episode_ids[i] < num_episodes:
                    data["obs"].append(state_np[i].copy())
                    data["images"].append(image_np[i].copy())
        
        # Step all environments
        actions_tensor = torch.from_numpy(actions).float()
        if torch.cuda.is_available():
            actions_tensor = actions_tensor.cuda()
        
        next_obs, rewards, terminated, truncated, info = env.step(actions_tensor)
        
        # Convert to numpy first to avoid cross-GPU tensor operations
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.cpu().numpy()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.cpu().numpy()
        dones = terminated | truncated
        
        # Convert rewards to numpy
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        
        # Store transitions (convert Tensor to numpy if needed)
        if obs_mode == "state":
            next_state = next_obs if not isinstance(next_obs, dict) else next_obs
            next_state_np = to_numpy(next_state)
        elif obs_mode == "image":
            next_image = next_obs if not isinstance(next_obs, dict) else next_obs
            next_image_np = to_numpy(next_image)
        else:  # state_image
            next_state_np = to_numpy(next_obs["state"])
            next_image_np = to_numpy(next_obs["image"])
        
        for i in range(num_envs):
            if env_episode_ids[i] < num_episodes:
                if obs_mode == "state":
                    data["next_obs"].append(next_state_np[i].copy() if next_state_np.ndim > 1 else next_state_np.copy())
                elif obs_mode == "image":
                    data["next_images"].append(next_image_np[i].copy() if next_image_np.ndim > 3 else next_image_np.copy())
                else:
                    data["next_obs"].append(next_state_np[i].copy())
                    data["next_images"].append(next_image_np[i].copy())
                
                data["actions"].append(actions[i].copy())
                data["rewards"].append(float(rewards[i]))
                data["dones"].append(bool(dones[i]))
                data["episode_ids"].append(int(env_episode_ids[i]))
                data["timesteps"].append(int(env_timesteps[i]))
                
                env_timesteps[i] += 1
                
                # Check if episode ended
                if dones[i] or env_timesteps[i] >= max_episode_steps:
                    # Check success
                    if isinstance(info, dict):
                        success = info.get("success", np.zeros(num_envs))
                        if isinstance(success, torch.Tensor):
                            success = success.cpu().numpy()
                        if success[i] or rewards[i] > 0:
                            successes += 1
                    
                    # Reset this environment's tracking
                    env_episode_ids[i] = current_episode + 1
                    env_timesteps[i] = 0
                    env_experts[i].reset()
                    
                    current_episode += 1
                    if pbar:
                        pbar.update(1)
                    
                    if current_episode >= num_episodes:
                        break
        
        obs = next_obs
    
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
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
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
        
        # Store metadata
        f.attrs["obs_mode"] = obs_mode
        f.attrs["num_episodes"] = len(set(data["episode_ids"]))
        f.attrs["num_transitions"] = len(data["actions"])
        f.attrs["task"] = "PickCube-v1"
        f.attrs["control_mode"] = "pd_ee_delta_pose"
    
    print(f"\nDataset saved to: {output_path}")
    print(f"  - Observation mode: {obs_mode}")
    print(f"  - Number of episodes: {len(set(data['episode_ids']))}")
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
        use_numpy=(args.num_envs == 1),
    )
    
    print(f"Environment created successfully!")
    print(f"  - State dim: {getattr(env, 'state_dim', 'N/A')}")
    print(f"  - Action dim: {env.action_dim}")
    print(f"  - Image shape: {getattr(env, 'image_shape', 'N/A')}")
    
    # Create expert policy
    expert = ManiSkillScriptedExpert()
    
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
