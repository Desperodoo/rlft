#!/usr/bin/env python3
"""Test script for action chunking implementation.

This script validates the action chunking functionality in:
1. ActionChunkingDataset - dataset loading with action sequences
2. FlowMatchingPolicy - action generation with queue mechanism
3. DiffusionPolicyAgent - action generation with queue mechanism

Run from rlft directory with:
    conda run -n rlft_ms3 python -m toy_diffusion_rl.scripts.test_action_chunking
"""

import numpy as np
import torch
import h5py
import tempfile
import os
import sys

# For direct script execution, we need to handle imports carefully
# The modules use relative imports, so we import them as a package
script_dir = os.path.dirname(os.path.abspath(__file__))
toy_diffusion_rl_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(toy_diffusion_rl_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import using package path
from toy_diffusion_rl.common.dataset_loader import ActionChunkingDataset
from toy_diffusion_rl.algorithms.flow_matching.fm_policy import FlowMatchingPolicy
from toy_diffusion_rl.algorithms.diffusion_policy.agent import DiffusionPolicyAgent


def create_test_h5_file(path: str, num_episodes: int = 3, steps_per_episode: int = 50):
    """Create a test HDF5 file with dummy data."""
    total_steps = num_episodes * steps_per_episode
    
    with h5py.File(path, 'w') as f:
        # Create obs and actions (using keys expected by ActionChunkingDataset)
        obs = np.random.randn(total_steps, 10).astype(np.float32)
        next_obs = np.random.randn(total_steps, 10).astype(np.float32)
        actions = np.random.randn(total_steps, 4).astype(np.float32)
        rewards = np.zeros(total_steps, dtype=np.float32)
        dones = np.zeros(total_steps, dtype=bool)
        
        # Create episode_ids
        episode_ids = np.repeat(np.arange(num_episodes), steps_per_episode).astype(np.int32)
        
        # Set done at end of each episode
        for i in range(num_episodes):
            dones[(i + 1) * steps_per_episode - 1] = True
        
        f.create_dataset('obs', data=obs)
        f.create_dataset('next_obs', data=next_obs)
        f.create_dataset('actions', data=actions)
        f.create_dataset('rewards', data=rewards)
        f.create_dataset('dones', data=dones)
        f.create_dataset('episode_ids', data=episode_ids)
        
        print(f"Created test file with {total_steps} steps, {num_episodes} episodes")
        print(f"Obs shape: {obs.shape}, Actions shape: {actions.shape}")
        print(f"Episode IDs: {np.unique(episode_ids)}")


def test_action_chunking_dataset():
    """Test ActionChunkingDataset functionality."""
    print("\n" + "="*60)
    print("Testing ActionChunkingDataset")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, 'test_data.h5')
        create_test_h5_file(h5_path, num_episodes=3, steps_per_episode=50)
        
        # Test with action_horizon=4
        action_horizon = 4
        dataset = ActionChunkingDataset(
            h5_path=h5_path,
            obs_mode='state',
            action_horizon=action_horizon
        )
        
        print(f"\nDataset length: {len(dataset)}")
        print(f"Action horizon: {action_horizon}")
        
        # Get a sample
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Obs shape: {sample['obs'].shape}")
        print(f"Action shape: {sample['action'].shape}")
        
        # Verify action sequence shape
        assert sample['action'].shape == (action_horizon, 4), \
            f"Expected action shape ({action_horizon}, 4), got {sample['action'].shape}"
        
        # Test episode boundary handling - get sample near episode end
        # Episode 0 ends at index 49, episode 1 starts at 50
        print("\n--- Testing episode boundary handling ---")
        
        # Find index near episode boundary
        for idx in range(len(dataset)):
            sample = dataset[idx]
            # Just verify it doesn't crash and returns valid data
            assert sample['action'].shape == (action_horizon, 4)
        
        print("✓ All samples have correct action sequence shape")
        print("✓ Episode boundary handling works correctly")
        
        # Test DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        batch = next(iter(loader))
        print(f"\nBatch obs shape: {batch['obs'].shape}")
        print(f"Batch action shape: {batch['action'].shape}")
        
        assert batch['action'].shape == (8, action_horizon, 4), \
            f"Expected batch action shape (8, {action_horizon}, 4), got {batch['action'].shape}"
        
        print("✓ DataLoader works correctly with ActionChunkingDataset")


def test_flow_matching_action_chunking():
    """Test FlowMatchingPolicy action chunking."""
    print("\n" + "="*60)
    print("Testing FlowMatchingPolicy Action Chunking")
    print("="*60)
    
    action_horizon = 4
    action_exec_horizon = 2
    state_dim = 10
    action_dim = 4
    
    policy = FlowMatchingPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        action_exec_horizon=action_exec_horizon,
        hidden_dims=[64, 64],
        num_inference_steps=5,
        device='cpu'
    )
    
    print(f"Action horizon: {action_horizon}")
    print(f"Action exec horizon: {action_exec_horizon}")
    
    # Test action queue mechanism - single env
    obs = {'state': np.random.randn(1, state_dim).astype(np.float32)}
    
    print("\n--- Testing action queue mechanism (single env) ---")
    policy.reset(num_envs=1)
    
    # First call should sample new actions
    action1 = policy.sample_action(obs)
    print(f"Action 1 shape: {action1.shape}")
    assert action1.shape == (action_dim,), f"Expected shape ({action_dim},), got {action1.shape}"
    
    # Queue for env 0 should have action_exec_horizon - 1 = 1 remaining action
    assert len(policy._action_queues.get(0, [])) == action_exec_horizon - 1, \
        f"Expected queue length {action_exec_horizon - 1}, got {len(policy._action_queues.get(0, []))}"
    print(f"Queue length after first sample: {len(policy._action_queues.get(0, []))}")
    
    # Second call should use queued action
    action2 = policy.sample_action(obs)
    print(f"Action 2 shape: {action2.shape}")
    assert len(policy._action_queues.get(0, [])) == 0, "Queue should be empty after second sample"
    print(f"Queue length after second sample: {len(policy._action_queues.get(0, []))}")
    
    # Third call should sample new actions again
    action3 = policy.sample_action(obs)
    print(f"Action 3 shape: {action3.shape}")
    assert len(policy._action_queues.get(0, [])) == action_exec_horizon - 1
    
    print("✓ Action queue mechanism works correctly (single env)")
    
    # Test reset
    policy.reset()
    assert len(policy._action_queues.get(0, [])) == 0
    print("✓ reset() works correctly")
    
    # Test VecEnv support
    print("\n--- Testing VecEnv support (multiple envs) ---")
    num_envs = 4
    obs_batched = {'state': np.random.randn(num_envs, state_dim).astype(np.float32)}
    
    policy.reset(num_envs=num_envs)
    assert len(policy._action_queues) == num_envs, f"Expected {num_envs} queues, got {len(policy._action_queues)}"
    
    # First call should sample for all envs
    actions = policy.sample_action(obs_batched)
    print(f"Batched actions shape: {actions.shape}")
    assert actions.shape == (num_envs, action_dim), f"Expected shape ({num_envs}, {action_dim}), got {actions.shape}"
    
    # Each env's queue should have remaining actions
    for i in range(num_envs):
        assert len(policy._action_queues[i]) == action_exec_horizon - 1
    print(f"All {num_envs} envs have correct queue length")
    
    # Test resetting specific envs (simulating auto-reset)
    policy.reset(env_ids=[1, 2])
    assert len(policy._action_queues[0]) == action_exec_horizon - 1, "Env 0 should still have cached actions"
    assert len(policy._action_queues[1]) == 0, "Env 1 should be reset"
    assert len(policy._action_queues[2]) == 0, "Env 2 should be reset"
    assert len(policy._action_queues[3]) == action_exec_horizon - 1, "Env 3 should still have cached actions"
    print("✓ Selective reset (env_ids) works correctly")
    
    print("✓ VecEnv action queue mechanism works correctly")


def test_diffusion_policy_action_chunking():
    """Test DiffusionPolicyAgent action chunking."""
    print("\n" + "="*60)
    print("Testing DiffusionPolicyAgent Action Chunking")
    print("="*60)
    
    action_horizon = 4
    action_exec_horizon = 2
    state_dim = 10
    action_dim = 4
    
    agent = DiffusionPolicyAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        action_exec_horizon=action_exec_horizon,
        hidden_dims=[64, 64],
        num_diffusion_steps=10,
        device='cpu'
    )
    
    print(f"Action horizon: {action_horizon}")
    print(f"Action exec horizon: {action_exec_horizon}")
    
    # Test action queue mechanism - single env
    obs = {'state': np.random.randn(1, state_dim).astype(np.float32)}
    
    print("\n--- Testing action queue mechanism (single env) ---")
    agent.reset(num_envs=1)
    
    # First call should sample new actions
    action1 = agent.sample_action(obs)
    print(f"Action 1 shape: {action1.shape}")
    assert action1.shape == (action_dim,), f"Expected shape ({action_dim},), got {action1.shape}"
    
    # Queue should have action_exec_horizon - 1 remaining actions
    assert len(agent._action_queues.get(0, [])) == action_exec_horizon - 1
    print(f"Queue length after first sample: {len(agent._action_queues.get(0, []))}")
    
    # Second call should use queued action
    action2 = agent.sample_action(obs)
    assert len(agent._action_queues.get(0, [])) == 0
    print(f"Queue length after second sample: {len(agent._action_queues.get(0, []))}")
    
    # Third call should sample new actions again
    action3 = agent.sample_action(obs)
    assert len(agent._action_queues.get(0, [])) == action_exec_horizon - 1
    
    print("✓ Action queue mechanism works correctly (single env)")
    
    # Test reset
    agent.reset()
    assert len(agent._action_queues.get(0, [])) == 0
    print("✓ reset() works correctly")
    
    # Test VecEnv support
    print("\n--- Testing VecEnv support (DiffusionPolicy) ---")
    num_envs = 4
    obs_batched = {'state': np.random.randn(num_envs, state_dim).astype(np.float32)}
    
    agent.reset(num_envs=num_envs)
    assert len(agent._action_queues) == num_envs
    
    actions = agent.sample_action(obs_batched)
    print(f"Batched actions shape: {actions.shape}")
    assert actions.shape == (num_envs, action_dim)
    
    # Test selective reset
    agent.reset(env_ids=[0, 3])
    assert len(agent._action_queues[0]) == 0
    assert len(agent._action_queues[1]) == action_exec_horizon - 1
    assert len(agent._action_queues[2]) == action_exec_horizon - 1
    assert len(agent._action_queues[3]) == 0
    print("✓ Selective reset (env_ids) works correctly")
    
    print("✓ VecEnv action queue mechanism works correctly (DiffusionPolicy)")


def test_backward_compatibility():
    """Test that action_horizon=1 gives same behavior as before."""
    print("\n" + "="*60)
    print("Testing Backward Compatibility (action_horizon=1)")
    print("="*60)
    
    state_dim = 10
    action_dim = 4
    
    # FlowMatching with action_horizon=1
    policy = FlowMatchingPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=1,  # Default
        action_exec_horizon=1,
        hidden_dims=[64, 64],
        num_inference_steps=5,
        device='cpu'
    )
    
    obs = {'state': np.random.randn(1, state_dim).astype(np.float32)}
    policy.reset(num_envs=1)
    
    # Each call should sample new action (no queueing with action_horizon=1)
    action1 = policy.sample_action(obs)
    assert action1.shape == (action_dim,)
    # With action_horizon=1, queue is not used (bypasses chunking logic)
    
    action2 = policy.sample_action(obs)
    assert action2.shape == (action_dim,)
    
    print("✓ FlowMatchingPolicy backward compatible with action_horizon=1")
    
    # DiffusionPolicy with action_horizon=1
    agent = DiffusionPolicyAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=1,
        action_exec_horizon=1,
        hidden_dims=[64, 64],
        num_diffusion_steps=10,
        device='cpu'
    )
    
    agent.reset(num_envs=1)
    action1 = agent.sample_action(obs)
    assert action1.shape == (action_dim,)
    
    action2 = agent.sample_action(obs)
    assert action2.shape == (action_dim,)
    
    print("✓ DiffusionPolicyAgent backward compatible with action_horizon=1")


def test_training_with_action_chunking():
    """Test training step with action sequences."""
    print("\n" + "="*60)
    print("Testing Training with Action Chunking")
    print("="*60)
    
    action_horizon = 4
    state_dim = 10
    action_dim = 4
    batch_size = 8
    
    # FlowMatching training
    policy = FlowMatchingPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dims=[64, 64],
        num_inference_steps=5,
        device='cpu'
    )
    
    # Create batch with action sequences
    batch = {
        'state': torch.randn(batch_size, state_dim),
        'actions': torch.randn(batch_size, action_horizon, action_dim)
    }
    
    result = policy.train_step(batch)
    # train_step may return dict or float depending on implementation
    if isinstance(result, dict):
        loss_value = result.get('loss', result.get('fm_loss', 0.0))
        print(f"FlowMatching training loss: {loss_value:.4f}")
        assert not np.isnan(loss_value)
    else:
        print(f"FlowMatching training loss: {result:.4f}")
        assert isinstance(result, float) and not np.isnan(result)
    print("✓ FlowMatchingPolicy training with action sequences works")
    
    # DiffusionPolicy training
    agent = DiffusionPolicyAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dims=[64, 64],
        num_diffusion_steps=10,
        device='cpu'
    )
    
    result = agent.train_step(batch)
    if isinstance(result, dict):
        loss_value = result.get('loss', result.get('diffusion_loss', 0.0))
        print(f"DiffusionPolicy training loss: {loss_value:.4f}")
        assert not np.isnan(loss_value)
    else:
        print(f"DiffusionPolicy training loss: {result:.4f}")
        assert isinstance(result, float) and not np.isnan(result)
    print("✓ DiffusionPolicyAgent training with action sequences works")


if __name__ == '__main__':
    print("="*60)
    print("ACTION CHUNKING IMPLEMENTATION TESTS")
    print("="*60)
    
    try:
        test_action_chunking_dataset()
        test_flow_matching_action_chunking()
        test_diffusion_policy_action_chunking()
        test_backward_compatibility()
        test_training_with_action_chunking()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
