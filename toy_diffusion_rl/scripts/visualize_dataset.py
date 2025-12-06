#!/usr/bin/env python3
"""
Visualize Dataset Episodes as Videos.

This script reads episodes from an HDF5 dataset and saves them as video files.
Supports both converted datasets (flat format) and original ManiSkill3 trajectory files.

Usage:
    # From converted dataset
    python scripts/visualize_dataset.py \
        --dataset_path data/peg_insertion_side_filtered.h5 \
        --output_dir results/dataset_videos \
        --num_episodes 10

    # From original ManiSkill3 trajectory
    python scripts/visualize_dataset.py \
        --dataset_path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state_dict+rgb.pd_joint_pos.physx_cuda.h5 \
        --output_dir results/trajectory_videos \
        --num_episodes 10 \
        --original_format
"""

import argparse
import os
import sys
import h5py
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List


def save_video(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """Save frames as video file."""
    if not frames:
        return
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        writer.write(frame_bgr)
    
    writer.release()


def visualize_converted_dataset(
    dataset_path: str,
    output_dir: str,
    num_episodes: Optional[int] = None,
    fps: int = 30,
    add_info_overlay: bool = True,
):
    """Visualize episodes from converted HDF5 dataset (flat format)."""
    print(f"Loading converted dataset: {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as f:
        images = f['images'][:]
        episode_ids = f['episode_ids'][:]
        timesteps = f['timesteps'][:]
        rewards = f['rewards'][:]
        actions = f['actions'][:]
        dones = f['dones'][:]
        
        unique_episodes = np.unique(episode_ids)
        if num_episodes is not None:
            unique_episodes = unique_episodes[:num_episodes]
        
        print(f"Found {len(np.unique(episode_ids))} episodes, visualizing {len(unique_episodes)}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for ep_idx, ep_id in enumerate(tqdm(unique_episodes, desc="Creating videos")):
            ep_mask = episode_ids == ep_id
            ep_images = images[ep_mask]
            ep_rewards = rewards[ep_mask]
            ep_actions = actions[ep_mask]
            ep_timesteps = timesteps[ep_mask]
            
            # Sort by timestep
            sort_idx = np.argsort(ep_timesteps)
            ep_images = ep_images[sort_idx]
            ep_rewards = ep_rewards[sort_idx]
            ep_actions = ep_actions[sort_idx]
            
            frames = []
            cumulative_reward = 0
            
            for t, (img, reward, action) in enumerate(zip(ep_images, ep_rewards, ep_actions)):
                cumulative_reward += reward
                
                # Ensure uint8
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                if add_info_overlay:
                    # Scale up for better text visibility
                    scale = 4
                    h, w = img.shape[:2]
                    img_large = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
                    
                    # Add text overlay
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 1
                    color = (255, 255, 255)
                    
                    # Episode info
                    cv2.putText(img_large, f"Episode {ep_id}", (10, 25), font, font_scale, color, thickness)
                    cv2.putText(img_large, f"Step {t}/{len(ep_images)}", (10, 50), font, font_scale, color, thickness)
                    cv2.putText(img_large, f"Reward: {reward:.2f}", (10, 75), font, font_scale, color, thickness)
                    cv2.putText(img_large, f"Cumulative: {cumulative_reward:.1f}", (10, 100), font, font_scale, color, thickness)
                    
                    # Action info (first 4 dims)
                    action_str = f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}, ...]"
                    cv2.putText(img_large, action_str, (10, h * scale - 10), font, font_scale * 0.8, color, thickness)
                    
                    frames.append(img_large)
                else:
                    frames.append(img)
            
            # Save video
            total_reward = ep_rewards.sum()
            video_path = os.path.join(output_dir, f"episode_{ep_id:03d}_len{len(ep_images)}_reward{total_reward:.0f}.mp4")
            save_video(frames, video_path, fps=fps)
        
        print(f"\nVideos saved to: {output_dir}")


def visualize_original_trajectory(
    trajectory_path: str,
    output_dir: str,
    num_episodes: Optional[int] = None,
    fps: int = 30,
    add_info_overlay: bool = True,
    max_episode_length: Optional[int] = None,
):
    """Visualize episodes from original ManiSkill3 trajectory file."""
    print(f"Loading original trajectory: {trajectory_path}")
    
    with h5py.File(trajectory_path, 'r') as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith('traj_')],
                          key=lambda x: int(x.split('_')[1]))
        
        if num_episodes is not None:
            traj_keys = traj_keys[:num_episodes]
        
        print(f"Found {len([k for k in f.keys() if k.startswith('traj_')])} trajectories, visualizing {len(traj_keys)}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for traj_key in tqdm(traj_keys, desc="Creating videos"):
            traj = f[traj_key]
            traj_id = int(traj_key.split('_')[1])
            
            # Get data
            actions = traj['actions'][:]
            rewards = traj['rewards'][:] if 'rewards' in traj else np.zeros(len(actions))
            success = traj['success'][:] if 'success' in traj else np.zeros(len(actions), dtype=bool)
            
            # Find effective length (up to first success)
            if success[0]:
                # Skip corrupted trajectories
                continue
            
            success_indices = np.where(success)[0]
            if len(success_indices) > 0:
                T_effective = success_indices[0] + 1
            else:
                T_effective = len(actions)
            
            if max_episode_length is not None and T_effective > max_episode_length:
                T_effective = max_episode_length
            
            # Get images
            obs_group = traj['obs']
            frames = []
            cumulative_reward = 0
            
            # Find camera with RGB
            sensor_data = obs_group['sensor_data']
            cam_name = None
            for name in sensor_data.keys():
                if 'rgb' in sensor_data[name]:
                    cam_name = name
                    break
            
            if cam_name is None:
                print(f"Warning: No RGB data in {traj_key}, skipping")
                continue
            
            for t in range(T_effective):
                # Get image
                img = sensor_data[cam_name]['rgb'][t]
                reward = rewards[t]
                action = actions[t]
                cumulative_reward += reward
                
                # Ensure uint8
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                if add_info_overlay:
                    # Scale up for better text visibility
                    scale = 4
                    h, w = img.shape[:2]
                    img_large = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
                    
                    # Add text overlay
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 1
                    color = (255, 255, 255)
                    
                    # Episode info
                    cv2.putText(img_large, f"Trajectory {traj_id}", (10, 25), font, font_scale, color, thickness)
                    cv2.putText(img_large, f"Step {t}/{T_effective}", (10, 50), font, font_scale, color, thickness)
                    cv2.putText(img_large, f"Reward: {reward:.2f}", (10, 75), font, font_scale, color, thickness)
                    cv2.putText(img_large, f"Cumulative: {cumulative_reward:.1f}", (10, 100), font, font_scale, color, thickness)
                    
                    # Success flag
                    if t == T_effective - 1 and len(success_indices) > 0:
                        cv2.putText(img_large, "SUCCESS!", (10, 125), font, font_scale, (0, 255, 0), thickness + 1)
                    
                    # Action info
                    is_zero_action = np.allclose(action, 0)
                    action_str = f"Action: [{'ZERO' if is_zero_action else f'{action[0]:.2f}, {action[1]:.2f}, ...'}]"
                    cv2.putText(img_large, action_str, (10, h * scale - 10), font, font_scale * 0.8, color, thickness)
                    
                    frames.append(img_large)
                else:
                    frames.append(img)
            
            # Save video
            total_reward = rewards[:T_effective].sum()
            status = "success" if len(success_indices) > 0 else "incomplete"
            video_path = os.path.join(output_dir, f"traj_{traj_id:03d}_{status}_len{T_effective}_reward{total_reward:.0f}.mp4")
            save_video(frames, video_path, fps=fps)
        
        print(f"\nVideos saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset episodes as videos")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to HDF5 dataset (converted or original trajectory)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/dataset_videos",
        help="Output directory for videos"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to visualize (default: 10)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video FPS (default: 30)"
    )
    parser.add_argument(
        "--original_format",
        action="store_true",
        help="Input is original ManiSkill3 trajectory format (not converted)"
    )
    parser.add_argument(
        "--no_overlay",
        action="store_true",
        help="Disable info overlay on video"
    )
    parser.add_argument(
        "--max_episode_length",
        type=int,
        default=None,
        help="Maximum episode length to visualize (for original format)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset not found: {args.dataset_path}")
        sys.exit(1)
    
    if args.original_format:
        visualize_original_trajectory(
            trajectory_path=args.dataset_path,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes,
            fps=args.fps,
            add_info_overlay=not args.no_overlay,
            max_episode_length=args.max_episode_length,
        )
    else:
        visualize_converted_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes,
            fps=args.fps,
            add_info_overlay=not args.no_overlay,
        )


if __name__ == "__main__":
    main()
