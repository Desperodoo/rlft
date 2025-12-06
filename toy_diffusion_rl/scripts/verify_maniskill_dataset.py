#!/usr/bin/env python3
"""
Verify ManiSkill3 Dataset Integrity.

This script validates that a converted ManiSkill3 dataset has the correct format
and data integrity for use with ManiSkillOfflineDataset.

Checks performed:
1. All required fields exist (obs, next_obs, images, next_images, actions, rewards, dones, episode_ids, timesteps)
2. Data types are correct (images uint8, others float32/int32/bool)
3. Episode IDs and timesteps are properly sorted
4. Image values are in valid range (0-255)
5. State/action/reward statistics are reasonable
6. Dataset can be loaded with ManiSkillOfflineDataset

Usage:
    python scripts/verify_maniskill_dataset.py --dataset_path data/peg_insertion_side_1k.h5
"""

import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)

import argparse
import os
import sys
import h5py
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_dataset(dataset_path: str, verbose: bool = True) -> Dict:
    """Verify dataset integrity and return statistics.
    
    Args:
        dataset_path: Path to HDF5 dataset file
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with verification results
    """
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "statistics": {},
    }
    
    def log(msg, level="info"):
        if verbose:
            prefix = {"info": "✓", "warn": "⚠", "error": "✗"}
            print(f"  {prefix.get(level, ' ')} {msg}")
    
    def add_error(msg):
        results["errors"].append(msg)
        results["passed"] = False
        log(msg, "error")
    
    def add_warning(msg):
        results["warnings"].append(msg)
        log(msg, "warn")
    
    if verbose:
        print("=" * 60)
        print("ManiSkill Dataset Verification")
        print("=" * 60)
        print(f"Dataset: {dataset_path}")
        print()
    
    # Check file exists
    if not os.path.exists(dataset_path):
        add_error(f"Dataset file not found: {dataset_path}")
        return results
    
    # Open and verify
    try:
        with h5py.File(dataset_path, "r") as f:
            # 1. Check required fields
            if verbose:
                print("1. Checking required fields...")
            
            required_fields = [
                "obs", "next_obs", "images", "next_images",
                "actions", "rewards", "dones", "episode_ids", "timesteps"
            ]
            
            for field in required_fields:
                if field not in f:
                    add_error(f"Missing required field: {field}")
                else:
                    log(f"Found field: {field} - shape {f[field].shape}, dtype {f[field].dtype}")
            
            if not results["passed"]:
                return results
            
            # Load data
            obs = f["obs"][:]
            next_obs = f["next_obs"][:]
            images = f["images"][:]
            next_images = f["next_images"][:]
            actions = f["actions"][:]
            rewards = f["rewards"][:]
            dones = f["dones"][:]
            episode_ids = f["episode_ids"][:]
            timesteps = f["timesteps"][:]
            
            # 2. Check data types
            if verbose:
                print("\n2. Checking data types...")
            
            if obs.dtype != np.float32:
                add_warning(f"obs dtype is {obs.dtype}, expected float32")
            else:
                log(f"obs dtype: {obs.dtype} ✓")
            
            if next_obs.dtype != np.float32:
                add_warning(f"next_obs dtype is {next_obs.dtype}, expected float32")
            else:
                log(f"next_obs dtype: {next_obs.dtype} ✓")
            
            if images.dtype != np.uint8:
                add_error(f"images dtype is {images.dtype}, expected uint8")
            else:
                log(f"images dtype: {images.dtype} ✓")
            
            if next_images.dtype != np.uint8:
                add_error(f"next_images dtype is {next_images.dtype}, expected uint8")
            else:
                log(f"next_images dtype: {next_images.dtype} ✓")
            
            if actions.dtype != np.float32:
                add_warning(f"actions dtype is {actions.dtype}, expected float32")
            else:
                log(f"actions dtype: {actions.dtype} ✓")
            
            if rewards.dtype != np.float32:
                add_warning(f"rewards dtype is {rewards.dtype}, expected float32")
            else:
                log(f"rewards dtype: {rewards.dtype} ✓")
            
            if dones.dtype != bool and dones.dtype != np.bool_:
                add_warning(f"dones dtype is {dones.dtype}, expected bool")
            else:
                log(f"dones dtype: {dones.dtype} ✓")
            
            if episode_ids.dtype != np.int32:
                add_warning(f"episode_ids dtype is {episode_ids.dtype}, expected int32")
            else:
                log(f"episode_ids dtype: {episode_ids.dtype} ✓")
            
            if timesteps.dtype != np.int32:
                add_warning(f"timesteps dtype is {timesteps.dtype}, expected int32")
            else:
                log(f"timesteps dtype: {timesteps.dtype} ✓")
            
            # 3. Check episode_ids and timesteps ordering
            if verbose:
                print("\n3. Checking episode/timestep ordering...")
            
            # Check that data is sorted by episode_id first, then timestep
            is_sorted = True
            for i in range(1, len(episode_ids)):
                if episode_ids[i] < episode_ids[i-1]:
                    is_sorted = False
                    add_error(f"episode_ids not sorted at index {i}: {episode_ids[i-1]} -> {episode_ids[i]}")
                    break
                elif episode_ids[i] == episode_ids[i-1]:
                    if timesteps[i] != timesteps[i-1] + 1:
                        is_sorted = False
                        add_error(f"timesteps not sequential at index {i}: episode={episode_ids[i]}, timestep {timesteps[i-1]} -> {timesteps[i]}")
                        break
            
            if is_sorted:
                log("Episode IDs and timesteps are properly sorted ✓")
            
            # Check timesteps reset at episode boundaries
            unique_episodes = np.unique(episode_ids)
            for ep in unique_episodes[:5]:  # Check first 5 episodes
                ep_mask = episode_ids == ep
                ep_timesteps = timesteps[ep_mask]
                if ep_timesteps[0] != 0:
                    add_warning(f"Episode {ep} starts at timestep {ep_timesteps[0]} instead of 0")
                if not np.array_equal(ep_timesteps, np.arange(len(ep_timesteps))):
                    add_error(f"Episode {ep} timesteps not sequential: {ep_timesteps[:10]}...")
            
            log(f"Checked timestep sequence for {min(5, len(unique_episodes))} episodes ✓")
            
            # 4. Check image values
            if verbose:
                print("\n4. Checking image values...")
            
            if images.min() < 0 or images.max() > 255:
                add_error(f"Image values out of range [0, 255]: min={images.min()}, max={images.max()}")
            else:
                log(f"Image value range: [{images.min()}, {images.max()}] ✓")
            
            if next_images.min() < 0 or next_images.max() > 255:
                add_error(f"Next image values out of range [0, 255]: min={next_images.min()}, max={next_images.max()}")
            else:
                log(f"Next image value range: [{next_images.min()}, {next_images.max()}] ✓")
            
            # Check image shape
            if len(images.shape) != 4 or images.shape[3] != 3:
                add_error(f"Image shape unexpected: {images.shape}, expected (N, H, W, 3)")
            else:
                log(f"Image shape: {images.shape} (N, H, W, C) ✓")
            
            # 5. Check data statistics
            if verbose:
                print("\n5. Checking data statistics...")
            
            stats = {
                "num_transitions": len(actions),
                "num_episodes": len(unique_episodes),
                "state_dim": obs.shape[1],
                "action_dim": actions.shape[1],
                "image_shape": images.shape[1:],
                "obs_mean": obs.mean(axis=0),
                "obs_std": obs.std(axis=0),
                "action_mean": actions.mean(axis=0),
                "action_std": actions.std(axis=0),
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
                "reward_min": rewards.min(),
                "reward_max": rewards.max(),
                "done_rate": dones.mean(),
            }
            results["statistics"] = stats
            
            log(f"Number of transitions: {stats['num_transitions']}")
            log(f"Number of episodes: {stats['num_episodes']}")
            log(f"State dimension: {stats['state_dim']}")
            log(f"Action dimension: {stats['action_dim']}")
            log(f"Image shape: {stats['image_shape']}")
            log(f"Reward range: [{stats['reward_min']:.4f}, {stats['reward_max']:.4f}]")
            log(f"Reward mean±std: {stats['reward_mean']:.4f} ± {stats['reward_std']:.4f}")
            log(f"Done rate: {stats['done_rate']*100:.2f}%")
            
            # Episode length statistics
            ep_lengths = []
            for ep in unique_episodes:
                ep_lengths.append(np.sum(episode_ids == ep))
            ep_lengths = np.array(ep_lengths)
            stats["ep_length_mean"] = ep_lengths.mean()
            stats["ep_length_std"] = ep_lengths.std()
            stats["ep_length_min"] = ep_lengths.min()
            stats["ep_length_max"] = ep_lengths.max()
            
            log(f"Episode length: {stats['ep_length_mean']:.1f} ± {stats['ep_length_std']:.1f} (range: {stats['ep_length_min']}-{stats['ep_length_max']})")
            
            # Check metadata
            if verbose:
                print("\n6. Checking metadata...")
            
            if "metadata" in f.attrs:
                try:
                    metadata = json.loads(f.attrs["metadata"])
                    log(f"Metadata found: {list(metadata.keys())}")
                    for k, v in metadata.items():
                        log(f"  {k}: {v}")
                except json.JSONDecodeError:
                    add_warning("Metadata exists but is not valid JSON")
            else:
                add_warning("No metadata attribute found")
            
            # Check individual attributes
            for attr in ["obs_mode", "task", "control_mode", "num_episodes", "num_transitions"]:
                if attr in f.attrs:
                    log(f"Attribute {attr}: {f.attrs[attr]}")
                else:
                    add_warning(f"Missing attribute: {attr}")
            
            # 7. Test loading with ManiSkillOfflineDataset
            if verbose:
                print("\n7. Testing dataset loader compatibility...")
            
            try:
                from common.dataset_loader import ManiSkillOfflineDataset
                
                dataset = ManiSkillOfflineDataset(
                    h5_path=dataset_path,
                    obs_mode="state_image",
                    load_to_memory=False,  # Don't load to memory for quick test
                )
                
                # Test getting a sample
                sample = dataset[0]
                required_keys = ["obs", "next_obs", "image", "next_image", "action", "reward", "done"]
                for key in required_keys:
                    if key not in sample:
                        add_error(f"Dataset sample missing key: {key}")
                    else:
                        log(f"Sample key '{key}': shape={sample[key].shape if hasattr(sample[key], 'shape') else 'scalar'}")
                
                log(f"ManiSkillOfflineDataset loading successful ✓")
                
            except ImportError as e:
                add_warning(f"Could not import ManiSkillOfflineDataset: {e}")
            except Exception as e:
                add_error(f"Failed to load with ManiSkillOfflineDataset: {e}")
    
    except Exception as e:
        add_error(f"Failed to open dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Verification Summary")
        print("=" * 60)
        
        if results["passed"]:
            print("✓ All checks passed!")
        else:
            print(f"✗ {len(results['errors'])} error(s) found:")
            for err in results["errors"]:
                print(f"  - {err}")
        
        if results["warnings"]:
            print(f"⚠ {len(results['warnings'])} warning(s):")
            for warn in results["warnings"]:
                print(f"  - {warn}")
    
    return results


def compare_with_original(
    converted_path: str,
    original_json_path: str,
    verbose: bool = True
) -> Dict:
    """Compare converted dataset with original demo metadata to calculate replay success rate.
    
    Args:
        converted_path: Path to converted HDF5 dataset
        original_json_path: Path to original trajectory.json metadata
        verbose: Whether to print detailed output
    
    Returns:
        Comparison statistics including replay success rate
    """
    stats = {}
    
    if verbose:
        print("\n" + "=" * 60)
        print("Replay Success Rate Analysis")
        print("=" * 60)
    
    # Load original metadata
    try:
        with open(original_json_path, "r") as f:
            original_meta = json.load(f)
        
        original_episodes = len(original_meta["episodes"])
        stats["original_episodes"] = original_episodes
        
        if verbose:
            print(f"Original demo episodes: {original_episodes}")
    except Exception as e:
        if verbose:
            print(f"Could not load original metadata: {e}")
        return stats
    
    # Load converted dataset
    try:
        with h5py.File(converted_path, "r") as f:
            episode_ids = f["episode_ids"][:]
            converted_episodes = len(np.unique(episode_ids))
            stats["converted_episodes"] = converted_episodes
            
            if verbose:
                print(f"Converted dataset episodes: {converted_episodes}")
            
            # Calculate success rate
            success_rate = converted_episodes / original_episodes if original_episodes > 0 else 0
            stats["replay_success_rate"] = success_rate
            
            if verbose:
                print(f"Replay success rate: {success_rate*100:.1f}% ({converted_episodes}/{original_episodes})")
    except Exception as e:
        if verbose:
            print(f"Could not load converted dataset: {e}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Verify ManiSkill3 dataset integrity"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to HDF5 dataset file to verify"
    )
    parser.add_argument(
        "--original_json",
        type=str,
        default=None,
        help="Path to original trajectory.json for replay success rate calculation"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    results = verify_dataset(args.dataset_path, verbose=not args.quiet)
    
    if args.original_json:
        compare_stats = compare_with_original(
            args.dataset_path,
            args.original_json,
            verbose=not args.quiet
        )
        results["replay_stats"] = compare_stats
    
    # Exit with appropriate code
    if results["passed"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
