#!/usr/bin/env python3
"""
Verify observation processing consistency between offline and online pipelines.

This script compares:
1. build_state_obs_extractor (offline pipeline)
2. FlattenRGBDObservationWrapper / flatten_state_dict (online pipeline)

To ensure state extraction logic is identical.
"""

import numpy as np
import torch
import gymnasium as gym
from functools import partial

# ManiSkill imports
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils import common

# Our imports
import sys
sys.path.insert(0, "/home/amax/rlft/diffusion_policy")
from diffusion_policy.utils import build_state_obs_extractor, convert_obs


def get_raw_obs_from_env(env_id: str, obs_mode: str = "rgb", control_mode: str = "pd_ee_delta_pose"):
    """Get raw observation from environment without wrapper."""
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        num_envs=1,
        sim_backend="cpu",
    )
    obs, _ = env.reset()
    env.close()
    return obs


def get_wrapped_obs_from_env(env_id: str, obs_mode: str = "rgb", control_mode: str = "pd_ee_delta_pose"):
    """Get observation from environment WITH FlattenRGBDObservationWrapper."""
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        num_envs=1,
        sim_backend="cpu",
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    obs, _ = env.reset()
    env.close()
    return obs


def extract_state_offline(obs, env_id: str):
    """
    Simulate offline pipeline state extraction.
    Uses build_state_obs_extractor from utils.py
    """
    state_obs_extractor = build_state_obs_extractor(env_id)
    
    # This is what convert_obs does internally
    states_to_stack = state_obs_extractor(obs)
    
    # Process each state component
    processed_states = []
    for i, s in enumerate(states_to_stack):
        arr = np.array(s)
        # Handle dimension expansion (same as self_contained_mode)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        # Handle bool conversion
        if arr.dtype == np.bool_:
            arr = arr.astype(np.float32)
        # Handle float64 -> float32
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        processed_states.append(arr)
        print(f"  State component {i}: shape={arr.shape}, dtype={arr.dtype}")
    
    # Stack horizontally
    state = np.hstack(processed_states)
    return state


def extract_state_online(obs):
    """
    Simulate online pipeline state extraction.
    Uses FlattenRGBDObservationWrapper output directly.
    """
    # The wrapper already provides flattened state
    state = obs["state"]
    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()
    return state


def compare_state_extraction(env_id: str, obs_mode: str = "rgb"):
    """Compare state extraction between offline and online pipelines."""
    print(f"\n{'='*60}")
    print(f"Testing: {env_id} (obs_mode={obs_mode})")
    print('='*60)
    
    # Get raw observation (for offline processing)
    print("\n1. Getting raw observation (no wrapper)...")
    raw_obs = get_raw_obs_from_env(env_id, obs_mode=obs_mode)
    
    # Show raw observation structure
    print("\n   Raw observation structure:")
    def show_structure(d, indent=3):
        for k, v in d.items():
            if isinstance(v, dict):
                print(" " * indent + f"{k}:")
                show_structure(v, indent + 3)
            elif isinstance(v, (np.ndarray, torch.Tensor)):
                shape = v.shape if hasattr(v, 'shape') else 'N/A'
                dtype = v.dtype if hasattr(v, 'dtype') else type(v)
                print(" " * indent + f"{k}: shape={shape}, dtype={dtype}")
            else:
                print(" " * indent + f"{k}: {type(v)}")
    show_structure(raw_obs)
    
    # Extract state using offline method
    print("\n2. Extracting state using OFFLINE method (build_state_obs_extractor)...")
    offline_state = extract_state_offline(raw_obs, env_id)
    print(f"   Offline state shape: {offline_state.shape}")
    print(f"   Offline state dtype: {offline_state.dtype}")
    print(f"   Offline state first 10 values: {offline_state[0, :10]}")
    
    # Get wrapped observation (for online processing)
    print("\n3. Getting wrapped observation (FlattenRGBDObservationWrapper)...")
    wrapped_obs = get_wrapped_obs_from_env(env_id, obs_mode=obs_mode)
    
    print("\n   Wrapped observation structure:")
    for k, v in wrapped_obs.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            shape = v.shape if hasattr(v, 'shape') else 'N/A'
            dtype = v.dtype if hasattr(v, 'dtype') else type(v)
            print(f"      {k}: shape={shape}, dtype={dtype}")
    
    # Extract state using online method
    print("\n4. Extracting state using ONLINE method (wrapper output)...")
    online_state = extract_state_online(wrapped_obs)
    print(f"   Online state shape: {online_state.shape}")
    print(f"   Online state dtype: {online_state.dtype}")
    print(f"   Online state first 10 values: {online_state[0, :10]}")
    
    # Compare
    print("\n5. Comparing states...")
    
    # Check shapes
    offline_flat = offline_state.flatten()
    online_flat = online_state.flatten()
    
    print(f"   Offline flattened shape: {offline_flat.shape}")
    print(f"   Online flattened shape: {online_flat.shape}")
    
    if offline_flat.shape != online_flat.shape:
        print(f"\n   ⚠️  SHAPE MISMATCH!")
        print(f"   Offline: {offline_flat.shape}")
        print(f"   Online: {online_flat.shape}")
        
        # Try to find which components differ
        print("\n   Analyzing component-by-component...")
        return False
    
    # Check values
    diff = np.abs(offline_flat - online_flat)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"   Max absolute difference: {max_diff}")
    print(f"   Mean absolute difference: {mean_diff}")
    
    if max_diff < 1e-5:
        print("\n   ✅ States are IDENTICAL (within tolerance)")
        return True
    else:
        print("\n   ⚠️  States DIFFER!")
        
        # Find where they differ
        diff_indices = np.where(diff > 1e-5)[0]
        print(f"   Differing indices: {diff_indices[:20]}...")
        
        for idx in diff_indices[:5]:
            print(f"   Index {idx}: offline={offline_flat[idx]:.6f}, online={online_flat[idx]:.6f}")
        
        return False


def verify_rgb_format(env_id: str, obs_mode: str = "rgb"):
    """Verify RGB format differences between pipelines."""
    print(f"\n{'='*60}")
    print(f"RGB Format Verification: {env_id}")
    print('='*60)
    
    # Get raw observation
    raw_obs = get_raw_obs_from_env(env_id, obs_mode=obs_mode)
    
    # Get wrapped observation
    wrapped_obs = get_wrapped_obs_from_env(env_id, obs_mode=obs_mode)
    
    # Check RGB in raw obs
    if "sensor_data" in raw_obs:
        print("\n1. Raw observation RGB (from sensor_data):")
        for cam_name, cam_data in raw_obs["sensor_data"].items():
            if "rgb" in cam_data:
                rgb = cam_data["rgb"]
                print(f"   {cam_name}/rgb: shape={rgb.shape}, dtype={rgb.dtype}")
    
    # Check RGB in wrapped obs
    if "rgb" in wrapped_obs:
        rgb = wrapped_obs["rgb"]
        print(f"\n2. Wrapped observation RGB:")
        print(f"   rgb: shape={rgb.shape}, dtype={rgb.dtype}")
        print(f"   Format: {'NHWC' if rgb.shape[-1] in [3, 6, 9, 12] else 'NCHW'}")


def main():
    """Run verification for multiple environments."""
    test_envs = [
        "PegInsertionSide-v1",
        "LiftPegUpright-v1",
        # "PickCube-v1",
    ]
    
    results = {}
    
    for env_id in test_envs:
        try:
            result = compare_state_extraction(env_id, obs_mode="rgb")
            results[env_id] = result
            
            # Also verify RGB format
            verify_rgb_format(env_id, obs_mode="rgb")
            
        except Exception as e:
            print(f"\n❌ Error testing {env_id}: {e}")
            import traceback
            traceback.print_exc()
            results[env_id] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for env_id, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {env_id}: {status}")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
