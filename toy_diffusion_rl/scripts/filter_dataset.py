#!/usr/bin/env python3
"""
Filter ManiSkill3 Dataset by Episode Quality.

This script filters the ManiSkill3 PickCube dataset to keep only high-quality
successful trajectories based on cumulative episode reward.

Usage:
    python scripts/filter_dataset.py \
        --input_path data/maniskill_pickcube_state_image_2k.h5 \
        --output_path data/maniskill_pickcube_state_image_filtered.h5 \
        --min_reward 100 \
        --remove_rotation

Options:
    --min_reward: Minimum cumulative episode reward to keep (default: 100)
    --max_length: Maximum episode length to keep (default: None)
    --remove_rotation: Remove zero rotation dimensions (7D -> 4D actions)
    --normalize_rewards: Normalize rewards to [-1, 1] range
"""

import argparse
import os
import sys
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


def analyze_dataset(path: str) -> dict:
    """Analyze dataset and return statistics."""
    with h5py.File(path, 'r') as f:
        rewards = f['rewards'][:]
        episode_ids = f['episode_ids'][:]
        actions = f['actions'][:]
        timesteps = f['timesteps'][:]
    
    # Per-episode statistics
    ep_stats = []
    for ep in np.unique(episode_ids):
        mask = episode_ids == ep
        ep_reward = rewards[mask].sum()
        ep_len = mask.sum()
        ep_stats.append({
            'episode_id': int(ep),
            'total_reward': float(ep_reward),
            'length': int(ep_len),
        })
    
    ep_stats = sorted(ep_stats, key=lambda x: x['total_reward'], reverse=True)
    
    # Action statistics
    action_stats = {
        'mean': actions.mean(axis=0).tolist(),
        'std': actions.std(axis=0).tolist(),
        'min': actions.min(axis=0).tolist(),
        'max': actions.max(axis=0).tolist(),
    }
    
    # Check if rotation dims are all zeros
    if actions.shape[1] == 7:
        rotation_is_zero = np.allclose(actions[:, 3:6], 0, atol=1e-6)
    else:
        rotation_is_zero = False
    
    return {
        'num_episodes': len(ep_stats),
        'num_transitions': len(rewards),
        'episode_stats': ep_stats,
        'action_stats': action_stats,
        'rotation_is_zero': rotation_is_zero,
        'reward_range': [float(rewards.min()), float(rewards.max())],
    }


def filter_dataset(
    input_path: str,
    output_path: str,
    min_reward: float = 100.0,
    max_reward: float = None,
    min_length: int = None,
    max_length: int = None,
    remove_rotation: bool = False,
    normalize_rewards: bool = False,
    verbose: bool = True,
):
    """Filter dataset by episode quality.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        min_reward: Minimum cumulative episode reward
        max_reward: Maximum cumulative episode reward (optional)
        min_length: Minimum episode length (optional)
        max_length: Maximum episode length (optional)
        remove_rotation: Remove rotation dimensions from actions (7D -> 4D)
        normalize_rewards: Normalize rewards to [-1, 1]
        verbose: Print progress
    """
    if verbose:
        print(f"Loading dataset: {input_path}")
    
    with h5py.File(input_path, 'r') as f:
        # Load all data
        obs = f['obs'][:]
        next_obs = f['next_obs'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        dones = f['dones'][:]
        episode_ids = f['episode_ids'][:]
        timesteps = f['timesteps'][:]
        
        # Load images if present
        has_images = 'images' in f
        if has_images:
            images = f['images'][:]
            next_images = f['next_images'][:]
        
        # Load metadata
        metadata = {}
        if 'metadata' in f.attrs:
            metadata = json.loads(f.attrs['metadata'])
        obs_mode = f.attrs.get('obs_mode', 'state_image')
    
    if verbose:
        print(f"Original dataset: {len(np.unique(episode_ids))} episodes, {len(rewards)} transitions")
    
    # Calculate per-episode statistics
    ep_rewards = {}
    ep_lengths = {}
    for ep in np.unique(episode_ids):
        mask = episode_ids == ep
        ep_rewards[ep] = rewards[mask].sum()
        ep_lengths[ep] = mask.sum()
    
    # Filter episodes
    keep_episodes = set()
    for ep, total_reward in ep_rewards.items():
        ep_len = ep_lengths[ep]
        
        # Apply filters
        if min_reward is not None and total_reward < min_reward:
            continue
        if max_reward is not None and total_reward > max_reward:
            continue
        if min_length is not None and ep_len < min_length:
            continue
        if max_length is not None and ep_len > max_length:
            continue
        
        keep_episodes.add(ep)
    
    if verbose:
        print(f"Keeping {len(keep_episodes)} episodes after filtering")
    
    # Create mask for transitions to keep
    keep_mask = np.array([ep in keep_episodes for ep in episode_ids])
    
    # Filter data
    filtered_obs = obs[keep_mask]
    filtered_next_obs = next_obs[keep_mask]
    filtered_actions = actions[keep_mask]
    filtered_rewards = rewards[keep_mask]
    filtered_dones = dones[keep_mask]
    filtered_episode_ids = episode_ids[keep_mask]
    filtered_timesteps = timesteps[keep_mask]
    
    if has_images:
        filtered_images = images[keep_mask]
        filtered_next_images = next_images[keep_mask]
    
    # Remove rotation dimensions if requested
    if remove_rotation and filtered_actions.shape[1] == 7:
        if verbose:
            print("Removing rotation dimensions (7D -> 4D actions)")
        # Keep only [dx, dy, dz, gripper] - indices 0,1,2,6
        filtered_actions = np.concatenate([
            filtered_actions[:, :3],  # position
            filtered_actions[:, 6:7],  # gripper
        ], axis=1)
    
    # Normalize rewards if requested
    if normalize_rewards:
        if verbose:
            print("Normalizing rewards to [-1, 0] range (Gym-compatible)")
        r_min, r_max = filtered_rewards.min(), filtered_rewards.max()
        if r_max > r_min:
            # Scale to [-1, 0] range to match Gym sparse reward structure
            # This maps: r_min -> -1, r_max -> 0
            filtered_rewards = (filtered_rewards - r_max) / (r_max - r_min)
            if verbose:
                print(f"  Original range: [{r_min:.4f}, {r_max:.4f}]")
                print(f"  Normalized range: [{filtered_rewards.min():.4f}, {filtered_rewards.max():.4f}]")
    
    # Re-number episode IDs to be consecutive
    old_to_new_ep = {old: new for new, old in enumerate(sorted(keep_episodes))}
    filtered_episode_ids = np.array([old_to_new_ep[ep] for ep in filtered_episode_ids])
    
    # Save filtered dataset
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    if verbose:
        print(f"Saving filtered dataset to: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('obs', data=filtered_obs, compression='gzip')
        f.create_dataset('next_obs', data=filtered_next_obs, compression='gzip')
        f.create_dataset('actions', data=filtered_actions, compression='gzip')
        f.create_dataset('rewards', data=filtered_rewards, compression='gzip')
        f.create_dataset('dones', data=filtered_dones, compression='gzip')
        f.create_dataset('episode_ids', data=filtered_episode_ids, compression='gzip')
        f.create_dataset('timesteps', data=filtered_timesteps, compression='gzip')
        
        if has_images:
            f.create_dataset('images', data=filtered_images, compression='gzip')
            f.create_dataset('next_images', data=filtered_next_images, compression='gzip')
        
        # Update metadata
        new_metadata = {
            'obs_mode': obs_mode,
            'num_episodes': len(keep_episodes),
            'num_transitions': len(filtered_actions),
            'task': metadata.get('task', 'PickCube-v1'),
            'control_mode': metadata.get('control_mode', 'pd_ee_delta_pose'),
            'filter_settings': {
                'min_reward': min_reward,
                'max_reward': max_reward,
                'min_length': min_length,
                'max_length': max_length,
                'remove_rotation': remove_rotation,
                'normalize_rewards': normalize_rewards,
            }
        }
        f.attrs['metadata'] = json.dumps(new_metadata)
        f.attrs['obs_mode'] = obs_mode
        f.attrs['num_episodes'] = len(keep_episodes)
        f.attrs['num_transitions'] = len(filtered_actions)
    
    if verbose:
        print(f"\nFiltered dataset statistics:")
        print(f"  - Episodes: {len(keep_episodes)}")
        print(f"  - Transitions: {len(filtered_actions)}")
        print(f"  - Obs shape: {filtered_obs.shape}")
        print(f"  - Action shape: {filtered_actions.shape}")
        print(f"  - Reward range: [{filtered_rewards.min():.4f}, {filtered_rewards.max():.4f}]")
        
        # Quality distribution of kept episodes
        kept_rewards = [ep_rewards[ep] for ep in keep_episodes]
        print(f"  - Kept episode reward range: [{min(kept_rewards):.1f}, {max(kept_rewards):.1f}]")
        print(f"  - Kept episode reward mean: {np.mean(kept_rewards):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Filter ManiSkill3 dataset by quality")
    parser.add_argument('--input_path', type=str, required=True, help='Input HDF5 file')
    parser.add_argument('--output_path', type=str, required=True, help='Output HDF5 file')
    parser.add_argument('--min_reward', type=float, default=100.0, help='Minimum episode reward')
    parser.add_argument('--max_reward', type=float, default=None, help='Maximum episode reward')
    parser.add_argument('--min_length', type=int, default=None, help='Minimum episode length')
    parser.add_argument('--max_length', type=int, default=None, help='Maximum episode length')
    parser.add_argument('--remove_rotation', action='store_true', help='Remove rotation dims (7D->4D)')
    parser.add_argument('--normalize_rewards', action='store_true', help='Normalize rewards to [-1,1]')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze, do not filter')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        print("="*60)
        print("Dataset Analysis")
        print("="*60)
        stats = analyze_dataset(args.input_path)
        
        print(f"\nDataset: {args.input_path}")
        print(f"Episodes: {stats['num_episodes']}")
        print(f"Transitions: {stats['num_transitions']}")
        print(f"Reward range: {stats['reward_range']}")
        print(f"Rotation dims are zero: {stats['rotation_is_zero']}")
        
        print(f"\nAction statistics:")
        print(f"  Mean: {stats['action_stats']['mean']}")
        print(f"  Std:  {stats['action_stats']['std']}")
        
        # Quality distribution
        ep_stats = stats['episode_stats']
        high = sum(1 for e in ep_stats if e['total_reward'] > 100)
        medium = sum(1 for e in ep_stats if 50 < e['total_reward'] <= 100)
        low = sum(1 for e in ep_stats if e['total_reward'] <= 50)
        
        print(f"\nQuality distribution:")
        print(f"  High (>100):   {high}/{len(ep_stats)} = {100*high/len(ep_stats):.1f}%")
        print(f"  Medium (50-100): {medium}/{len(ep_stats)} = {100*medium/len(ep_stats):.1f}%")
        print(f"  Low (<=50):    {low}/{len(ep_stats)} = {100*low/len(ep_stats):.1f}%")
        
        print(f"\nTop 10 episodes by reward:")
        for e in ep_stats[:10]:
            print(f"  Episode {e['episode_id']}: reward={e['total_reward']:.1f}, length={e['length']}")
    else:
        filter_dataset(
            input_path=args.input_path,
            output_path=args.output_path,
            min_reward=args.min_reward,
            max_reward=args.max_reward,
            min_length=args.min_length,
            max_length=args.max_length,
            remove_rotation=args.remove_rotation,
            normalize_rewards=args.normalize_rewards,
            verbose=True,
        )
        print("\nDone!")


if __name__ == "__main__":
    main()
