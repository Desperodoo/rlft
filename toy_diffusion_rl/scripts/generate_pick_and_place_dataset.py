#!/usr/bin/env python3
"""
Generate Offline Dataset for Pick-and-Place Task.

This script generates expert demonstration data for the Fetch Pick-and-Place
environment using a scripted expert policy. The data is saved in HDF5 format
for efficient loading during offline pretraining.

Usage:
    python scripts/generate_pick_and_place_dataset.py \
        --backend fetch \
        --obs_mode state_image \
        --num_episodes 200 \
        --max_episode_steps 50 \
        --output_path data/fetch_pick_and_place_state_image.h5
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h5py
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from envs.pick_and_place import make_pick_and_place_env, FetchExpertPolicy


def collect_episode(
    env,
    expert: FetchExpertPolicy,
    obs_mode: str,
) -> Dict[str, np.ndarray]:
    """Collect a single episode using expert policy.
    
    Args:
        env: Pick-and-place environment
        expert: Expert policy instance
        obs_mode: Observation mode ("state", "image", or "state_image")
    
    Returns:
        Dictionary containing episode data
    """
    obs, info = env.reset()
    expert.reset()
    
    # Storage for episode data
    states = []
    images = []
    actions = []
    rewards = []
    dones = []
    next_states = []
    next_images = []
    
    done = False
    step = 0
    
    while not done:
        # Get current state (always record state for analysis)
        if obs_mode == "state":
            current_state = obs
            current_image = None
        elif obs_mode == "image":
            # For image-only mode, we still need dict_obs for expert
            current_state = None
            current_image = obs
        else:  # state_image
            current_state = obs["state"]
            current_image = obs["image"]
        
        # Get dict observation for expert policy
        dict_obs = env.get_dict_obs()
        
        # Get expert action
        action = expert.get_action(dict_obs)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Get next state
        if obs_mode == "state":
            next_state = next_obs
            next_image = None
        elif obs_mode == "image":
            next_state = None
            next_image = next_obs
        else:  # state_image
            next_state = next_obs["state"]
            next_image = next_obs["image"]
        
        # Store transition
        if current_state is not None:
            states.append(current_state)
            next_states.append(next_state)
        if current_image is not None:
            images.append(current_image)
            next_images.append(next_image)
        
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        
        obs = next_obs
        step += 1
    
    # Convert to numpy arrays
    episode_data = {
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=bool),
    }
    
    if len(states) > 0:
        episode_data["obs"] = np.array(states, dtype=np.float32)
        episode_data["next_obs"] = np.array(next_states, dtype=np.float32)
    
    if len(images) > 0:
        episode_data["images"] = np.array(images, dtype=np.uint8)
        episode_data["next_images"] = np.array(next_images, dtype=np.uint8)
    
    return episode_data


def generate_dataset(
    backend: str,
    obs_mode: str,
    num_episodes: int,
    max_episode_steps: int,
    output_path: str,
    seed: int = 0,
    image_size: int = 128,
    reward_type: str = "sparse",
) -> Dict[str, any]:
    """Generate offline dataset.
    
    Args:
        backend: Environment backend ("fetch")
        obs_mode: Observation mode ("state", "image", or "state_image")
        num_episodes: Number of episodes to collect
        max_episode_steps: Maximum steps per episode
        output_path: Path to save HDF5 file
        seed: Random seed
        image_size: Image resolution for rendering
        reward_type: Reward type ("sparse" or "dense")
    
    Returns:
        Statistics dictionary
    """
    # Create environment
    # For data collection, always use state_image internally to capture both
    internal_obs_mode = "state_image" if obs_mode != "state" else "state"
    
    env = make_pick_and_place_env(
        backend=backend,
        obs_mode=internal_obs_mode,
        seed=seed,
        max_episode_steps=max_episode_steps,
        image_size=image_size,
        reward_type=reward_type,
    )
    
    expert = FetchExpertPolicy()
    
    # Collect episodes
    all_obs = []
    all_next_obs = []
    all_images = []
    all_next_images = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_episode_ids = []
    all_timesteps = []
    
    total_successes = 0
    total_rewards = 0
    
    print(f"Collecting {num_episodes} episodes...")
    for ep_idx in tqdm(range(num_episodes)):
        # Set different seed for each episode
        env.reset(seed=seed + ep_idx)
        
        episode_data = collect_episode(env, expert, internal_obs_mode)
        
        ep_len = len(episode_data["actions"])
        ep_reward = episode_data["rewards"].sum()
        
        # Track success (sparse reward: success if reward > -episode_length)
        if reward_type == "sparse" and ep_reward > -ep_len:
            total_successes += 1
        total_rewards += ep_reward
        
        # Append to global storage
        if "obs" in episode_data:
            all_obs.append(episode_data["obs"])
            all_next_obs.append(episode_data["next_obs"])
        if "images" in episode_data:
            all_images.append(episode_data["images"])
            all_next_images.append(episode_data["next_images"])
        
        all_actions.append(episode_data["actions"])
        all_rewards.append(episode_data["rewards"])
        all_dones.append(episode_data["dones"])
        all_episode_ids.append(np.full(ep_len, ep_idx, dtype=np.int32))
        all_timesteps.append(np.arange(ep_len, dtype=np.int32))
    
    env.close()
    
    # Concatenate all data
    dataset = {
        "actions": np.concatenate(all_actions, axis=0),
        "rewards": np.concatenate(all_rewards, axis=0),
        "dones": np.concatenate(all_dones, axis=0),
        "episode_ids": np.concatenate(all_episode_ids, axis=0),
        "timesteps": np.concatenate(all_timesteps, axis=0),
    }
    
    if len(all_obs) > 0:
        dataset["obs"] = np.concatenate(all_obs, axis=0)
        dataset["next_obs"] = np.concatenate(all_next_obs, axis=0)
    
    if len(all_images) > 0:
        dataset["images"] = np.concatenate(all_images, axis=0)
        dataset["next_images"] = np.concatenate(all_next_images, axis=0)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Save to HDF5
    print(f"Saving dataset to {output_path}...")
    with h5py.File(output_path, "w") as f:
        # Store metadata
        f.attrs["backend"] = backend
        f.attrs["obs_mode"] = obs_mode
        f.attrs["num_episodes"] = num_episodes
        f.attrs["max_episode_steps"] = max_episode_steps
        f.attrs["image_size"] = image_size
        f.attrs["reward_type"] = reward_type
        f.attrs["seed"] = seed
        f.attrs["total_transitions"] = len(dataset["actions"])
        f.attrs["success_rate"] = total_successes / num_episodes
        f.attrs["average_reward"] = total_rewards / num_episodes
        
        # Store data
        for key, data in dataset.items():
            # Use compression for images
            if "image" in key:
                f.create_dataset(
                    key,
                    data=data,
                    compression="gzip",
                    compression_opts=4,
                    chunks=True,
                )
            else:
                f.create_dataset(key, data=data)
    
    # Print statistics
    stats = {
        "total_transitions": len(dataset["actions"]),
        "num_episodes": num_episodes,
        "success_rate": total_successes / num_episodes,
        "average_reward": total_rewards / num_episodes,
        "obs_shape": dataset["obs"].shape if "obs" in dataset else None,
        "image_shape": dataset["images"].shape if "images" in dataset else None,
        "action_shape": dataset["actions"].shape,
    }
    
    print("\n" + "=" * 50)
    print("Dataset Statistics:")
    print("=" * 50)
    print(f"  Total transitions: {stats['total_transitions']:,}")
    print(f"  Number of episodes: {stats['num_episodes']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Average reward: {stats['average_reward']:.2f}")
    if stats["obs_shape"]:
        print(f"  Observation shape: {stats['obs_shape']}")
    if stats["image_shape"]:
        print(f"  Image shape: {stats['image_shape']}")
    print(f"  Action shape: {stats['action_shape']}")
    print("=" * 50)
    
    # Print file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size:.2f} MB")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate offline dataset for Pick-and-Place task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        default="fetch",
        choices=["fetch"],
        help="Environment backend",
    )
    parser.add_argument(
        "--obs_mode",
        type=str,
        default="state",
        choices=["state", "image", "state_image"],
        help="Observation mode",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=200,
        help="Number of episodes to collect",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=50,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output HDF5 file path (default: data/fetch_pick_and_place_{obs_mode}.h5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Image resolution for rendering",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default="sparse",
        choices=["sparse", "dense"],
        help="Reward type",
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        args.output_path = f"data/fetch_pick_and_place_{args.obs_mode}.h5"
    
    print("=" * 50)
    print("Pick-and-Place Dataset Generation")
    print("=" * 50)
    print(f"  Backend: {args.backend}")
    print(f"  Observation mode: {args.obs_mode}")
    print(f"  Number of episodes: {args.num_episodes}")
    print(f"  Max episode steps: {args.max_episode_steps}")
    print(f"  Image size: {args.image_size}")
    print(f"  Reward type: {args.reward_type}")
    print(f"  Output path: {args.output_path}")
    print(f"  Seed: {args.seed}")
    print("=" * 50)
    
    generate_dataset(
        backend=args.backend,
        obs_mode=args.obs_mode,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        output_path=args.output_path,
        seed=args.seed,
        image_size=args.image_size,
        reward_type=args.reward_type,
    )
    
    print(f"\n✓ Dataset saved to {args.output_path}")


if __name__ == "__main__":
    main()
