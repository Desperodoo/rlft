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


class ManiSkillScriptedExpert:
    """Scripted expert policy for ManiSkill3 PickCube task.
    
    PickCube is a "pick and hold" task - the goal is to pick up the cube
    and hold it at a goal position in the AIR (not place it down).
    
    Uses privileged state information to compute actions:
    1. Move to pre-grasp position above cube
    2. Lower to cube
    3. Close gripper
    4. Move to goal position (in the air) and hold
    
    This policy works with both:
    - Raw ManiSkill3 observation dict (state_dict mode)
    - Flat state tensor (state mode) with known structure
    
    Flat state structure for PickCube-v1 (42 dims):
    - [0:9]   qpos (joint positions)
    - [9:18]  qvel (joint velocities)  
    - [18:19] is_grasped
    - [19:26] tcp_pose (3 pos + 4 quat)
    - [26:29] goal_pos
    - [29:36] obj_pose (3 pos + 4 quat)
    - [36:39] tcp_to_obj_pos
    - [39:42] obj_to_goal_pos
    
    Action space (pd_ee_delta_pose): [dx, dy, dz, dax, day, daz, gripper]
    - position delta: world frame
    - rotation delta: axis-angle in end-effector frame
    """
    
    # Indices for flat state tensor
    TCP_POS_IDX = 19  # tcp_pose[:3]
    TCP_QUAT_IDX = 22  # tcp_pose[3:7] - quaternion (w, x, y, z) or (x, y, z, w)
    GOAL_POS_IDX = 26  # goal_pos
    OBJ_POS_IDX = 29   # obj_pose[:3]
    OBJ_QUAT_IDX = 32  # obj_pose[3:7]
    
    # Target orientation: gripper pointing down
    # ManiSkill3 uses (x, y, z, w) quaternion format
    # Initial gripper pose is approximately (x=0, y=1, z=0, w=0) - 180 deg around Y axis
    # This corresponds to gripper pointing down with fingers along X axis
    # We want to maintain this orientation
    TARGET_QUAT_XYZW = np.array([0.0, 1.0, 0.0, 0.0])  # (x, y, z, w) - 180 deg around Y
    
    def __init__(
        self,
        approach_height: float = 0.08,
        grasp_height: float = 0.02,
        position_gain: float = 10.0,
        orientation_gain: float = 0.5,  # Lower gain for smoother orientation correction
        max_action: float = 1.0,
        use_orientation_control: bool = True,
    ):
        self.approach_height = approach_height
        self.grasp_height = grasp_height
        self.position_gain = position_gain
        self.orientation_gain = orientation_gain
        self.max_action = max_action
        self.use_orientation_control = use_orientation_control
        
        self.phase = 0
        self.grasp_counter = 0
        # Store initial cube position for reference
        self.initial_cube_z = None
    
    def reset(self):
        """Reset policy state for new episode."""
        self.phase = 0
        self.grasp_counter = 0
        self.initial_cube_z = None
    
    @staticmethod
    def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
        """Convert quaternion from (x, y, z, w) to (w, x, y, z) format."""
        return np.array([q[3], q[0], q[1], q[2]])
    
    @staticmethod
    def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
        """Convert quaternion from (w, x, y, z) to (x, y, z, w) format."""
        return np.array([q[1], q[2], q[3], q[0]])
    
    @staticmethod
    def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to axis-angle representation.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            axis_angle: [ax, ay, az] where magnitude is the angle
        """
        # Normalize quaternion
        q = q / (np.linalg.norm(q) + 1e-8)
        w, x, y, z = q
        
        # Compute angle
        angle = 2.0 * np.arccos(np.clip(abs(w), -1.0, 1.0))
        
        # Compute axis
        sin_half = np.sin(angle / 2.0)
        if sin_half < 1e-6:
            return np.zeros(3)
        
        axis = np.array([x, y, z]) / sin_half
        
        # Handle quaternion sign ambiguity
        if w < 0:
            angle = -angle
        
        return axis * angle
    
    @staticmethod
    def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions (w, x, y, z format)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    
    @staticmethod
    def quat_conjugate(q: np.ndarray) -> np.ndarray:
        """Compute quaternion conjugate (inverse for unit quaternion)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def compute_orientation_error(
        self,
        current_quat_xyzw: np.ndarray,
        target_quat_xyzw: np.ndarray = None,
    ) -> np.ndarray:
        """Compute orientation error as axis-angle.
        
        Args:
            current_quat_xyzw: Current orientation [x, y, z, w] (ManiSkill3 format)
            target_quat_xyzw: Target orientation [x, y, z, w], defaults to pointing down
            
        Returns:
            axis_angle: Rotation error [dax, day, daz]
        """
        if target_quat_xyzw is None:
            target_quat_xyzw = self.TARGET_QUAT_XYZW
        
        # Convert to (w, x, y, z) format for computation
        current_quat = self.quat_xyzw_to_wxyz(current_quat_xyzw)
        target_quat = self.quat_xyzw_to_wxyz(target_quat_xyzw)
        
        # Normalize quaternions
        current_quat = current_quat / (np.linalg.norm(current_quat) + 1e-8)
        target_quat = target_quat / (np.linalg.norm(target_quat) + 1e-8)
        
        # Compute relative rotation: q_error = q_target * q_current^-1
        q_error = self.quat_multiply(target_quat, self.quat_conjugate(current_quat))
        
        # Convert to axis-angle
        return self.quat_to_axis_angle(q_error)
    
    def _extract_positions_from_flat(
        self,
        obs: np.ndarray,
        env_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract gripper, cube, goal positions and TCP quaternion from flat state tensor.
        
        Args:
            obs: Flat state tensor of shape (num_envs, 42) or (42,)
            env_idx: Environment index
            
        Returns:
            gripper_pos, cube_pos, goal_pos, tcp_quat (each shape (3,) or (4,))
        """
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if obs.ndim > 1:
            obs = obs[env_idx]
        
        gripper_pos = obs[self.TCP_POS_IDX:self.TCP_POS_IDX + 3].copy()
        goal_pos = obs[self.GOAL_POS_IDX:self.GOAL_POS_IDX + 3].copy()
        cube_pos = obs[self.OBJ_POS_IDX:self.OBJ_POS_IDX + 3].copy()
        
        # Extract TCP quaternion (assuming w, x, y, z format)
        tcp_quat = obs[self.TCP_QUAT_IDX:self.TCP_QUAT_IDX + 4].copy()
        
        return gripper_pos, cube_pos, goal_pos, tcp_quat
    
    def _extract_positions_from_dict(
        self,
        raw_obs: Dict,
        env_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract gripper, cube, goal positions and TCP quaternion from observation dict.
        
        Args:
            raw_obs: Raw observation dict from ManiSkill3 (state_dict mode)
            env_idx: Environment index
            
        Returns:
            gripper_pos, cube_pos, goal_pos, tcp_quat (each shape (3,) or (4,))
        """
        extra = raw_obs.get("extra", {})
        
        # Get TCP (end-effector) pose (position + quaternion)
        tcp_pose = extra.get("tcp_pose", None)
        if tcp_pose is None:
            gripper_pos = np.array([0.0, 0.0, 0.1])
            tcp_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        else:
            if isinstance(tcp_pose, torch.Tensor):
                tcp_pose = tcp_pose.cpu().numpy()
            if tcp_pose.ndim > 1:
                tcp_pose = tcp_pose[env_idx]
            gripper_pos = tcp_pose[:3].copy()
            # tcp_pose is [x, y, z, qw, qx, qy, qz] or [x, y, z, qx, qy, qz, qw]
            # ManiSkill3 uses (w, x, y, z) convention internally
            tcp_quat = tcp_pose[3:7].copy()
        
        # Get cube position
        obj_pose = extra.get("obj_pose", None)
        if obj_pose is None:
            cube_pos = np.array([0.0, 0.0, 0.02])
        else:
            if isinstance(obj_pose, torch.Tensor):
                obj_pose = obj_pose.cpu().numpy()
            if obj_pose.ndim > 1:
                obj_pose = obj_pose[env_idx]
            cube_pos = obj_pose[:3].copy()
        
        # Get goal position
        goal_pos = extra.get("goal_pos", None)
        if goal_pos is None:
            goal_pos = cube_pos + np.array([0.1, 0.1, 0.2])
        else:
            if isinstance(goal_pos, torch.Tensor):
                goal_pos = goal_pos.cpu().numpy()
            if goal_pos.ndim > 1:
                goal_pos = goal_pos[env_idx]
            if len(goal_pos.shape) > 0 and goal_pos.shape[-1] >= 3:
                goal_pos = goal_pos[:3].copy()
        
        return gripper_pos, cube_pos, goal_pos, tcp_quat
    
    def get_action(
        self,
        obs: Union[Dict, np.ndarray, torch.Tensor],
        env_idx: int = 0,
    ) -> np.ndarray:
        """Compute expert action from observation.
        
        Args:
            obs: Either a dict (state_dict mode) or tensor/array (state mode)
            env_idx: Environment index for batched observations
        
        Returns:
            Action array (7,) for pd_ee_delta_pose: [dx, dy, dz, dax, day, daz, gripper]
        """
        # Extract positions and orientation based on observation type
        if isinstance(obs, dict):
            gripper_pos, cube_pos, goal_pos, tcp_quat = self._extract_positions_from_dict(obs, env_idx)
        else:
            gripper_pos, cube_pos, goal_pos, tcp_quat = self._extract_positions_from_flat(obs, env_idx)
        
        # Store initial cube z position
        if self.initial_cube_z is None:
            self.initial_cube_z = cube_pos[2]
        
        # Initialize action
        action = np.zeros(7, dtype=np.float32)
        
        # Compute orientation correction to maintain gripper pointing down
        if self.use_orientation_control:
            # Compute orientation error as axis-angle
            orientation_error = self.compute_orientation_error(tcp_quat)
            # Apply orientation correction (scaled by gain)
            action[3:6] = self.orientation_gain * orientation_error
        
        # Compute target positions
        cube_above = cube_pos.copy()
        cube_above[2] = self.initial_cube_z + self.approach_height
        
        cube_grasp = cube_pos.copy()
        cube_grasp[2] = self.initial_cube_z + self.grasp_height
        
        # Phase 0: Move to pre-grasp position (above cube)
        if self.phase == 0:
            delta = cube_above - gripper_pos
            action[:3] = self.position_gain * delta
            action[6] = 1.0  # Open gripper
            
            if np.linalg.norm(delta) < 0.02:
                self.phase = 1
        
        # Phase 1: Lower to grasp position
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
        
        # Phase 3: Move to goal position (in the air) and HOLD
        elif self.phase == 3:
            delta = goal_pos - gripper_pos
            action[:3] = self.position_gain * delta
            action[6] = -1.0  # Keep closed - don't let go!
            
            # Stay in this phase - keep holding at goal position
        
        # Clip actions
        action[:3] = np.clip(action[:3], -self.max_action, self.max_action)
        action[3:6] = np.clip(action[3:6], -self.max_action, self.max_action)  # Clip rotation
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
    """Collect episodes with parallel environments using ManiSkillVectorEnv.
    
    Uses the official ManiSkillVectorEnv wrapper with auto_reset=True to properly
    handle episode resets. When an episode ends, next_obs is the new episode's
    initial observation, and info["final_observation"] contains the old episode's
    final observation.
    """
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    
    # Wrap environment with ManiSkillVectorEnv for proper auto-reset behavior
    vec_env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)
    
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
    env_experts = [ManiSkillScriptedExpert() for _ in range(num_envs)]
    
    # Next available episode ID
    next_episode_id = num_envs
    completed_episodes = 0
    successes = 0
    
    obs, info = vec_env.reset()
    
    # Reset all experts
    for exp in env_experts:
        exp.reset()
    
    pbar = tqdm(total=num_episodes, desc="Collecting episodes") if verbose else None
    
    while completed_episodes < num_episodes:
        # For expert policy, we need privileged state info (obj_pose, etc.)
        # Access the base ManiSkill environment observation which contains full state info
        # including obj_pose in the 'extra' dict
        try:
            privileged_obs = vec_env.base_env.get_obs()
        except AttributeError:
            # Fallback: if base_env doesn't exist, try to unwrap
            if hasattr(vec_env, 'env'):
                try:
                    privileged_obs = vec_env.env.unwrapped.get_obs()
                except:
                    privileged_obs = obs  # Last resort fallback
            else:
                privileged_obs = obs
        
        # Get actions from expert policy using privileged observation
        actions = []
        for i in range(num_envs):
            action = env_experts[i].get_action(privileged_obs, env_idx=i)
            actions.append(action)
        actions = np.stack(actions, axis=0)
        
        # Prepare observations for storage
        if obs_mode == "state":
            state_np = to_numpy(obs)
        elif obs_mode == "image":
            image_np = to_numpy(obs)
        else:  # state_image
            state_np = to_numpy(obs["state"])
            image_np = to_numpy(obs["image"])
        
        # Step all environments
        actions_tensor = torch.from_numpy(actions).float()
        # Make sure actions are on the same device as the environment
        if hasattr(vec_env, 'device'):
            actions_tensor = actions_tensor.to(vec_env.device)
        elif torch.cuda.is_available():
            actions_tensor = actions_tensor.cuda()
        
        next_obs, rewards, terminated, truncated, info = vec_env.step(actions_tensor)
        
        # Convert to numpy
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.cpu().numpy()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.cpu().numpy()
        dones = terminated | truncated
        
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        
        # Get final observations for environments that just ended
        # With auto_reset=True, next_obs is the NEW episode's initial obs
        # info["final_observation"] contains the OLD episode's final obs
        final_obs_available = "final_observation" in info
        if final_obs_available:
            final_obs = info["final_observation"]
            if obs_mode == "state":
                final_state_np = to_numpy(final_obs)
            elif obs_mode == "image":
                final_image_np = to_numpy(final_obs)
            else:  # state_image
                final_state_np = to_numpy(final_obs["state"]) if isinstance(final_obs, dict) else to_numpy(final_obs)
                final_image_np = to_numpy(final_obs["image"]) if isinstance(final_obs, dict) else None
        
        # Prepare next observations for storage (for non-done envs)
        if obs_mode == "state":
            next_state_np = to_numpy(next_obs)
        elif obs_mode == "image":
            next_image_np = to_numpy(next_obs)
        else:  # state_image
            next_state_np = to_numpy(next_obs["state"])
            next_image_np = to_numpy(next_obs["image"])
        
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
                
                # Store next observation
                # For done environments, use final_observation from info
                is_done = dones[i]
                if is_done and final_obs_available:
                    # Use final observation from the completed episode
                    if obs_mode == "state":
                        data["next_obs"].append(final_state_np[i].copy())
                    elif obs_mode == "image":
                        data["next_images"].append(final_image_np[i].copy() if final_image_np is not None else next_image_np[i].copy())
                    else:
                        data["next_obs"].append(final_state_np[i].copy())
                        data["next_images"].append(final_image_np[i].copy() if final_image_np is not None else next_image_np[i].copy())
                else:
                    # Use next_obs for continuing episodes
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
                    # Check success from final_info
                    if "final_info" in info:
                        final_info = info["final_info"]
                        if isinstance(final_info, dict):
                            success = final_info.get("success", np.zeros(num_envs))
                            if isinstance(success, torch.Tensor):
                                success = success.cpu().numpy()
                            if hasattr(success, '__len__') and len(success) > i:
                                if success[i]:
                                    successes += 1
                            elif success:
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
    unique_episodes = len(np.unique(episode_ids))
    
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
