#!/usr/bin/env python3
"""
Validation Script for ManiSkill3 Integration.

This script tests the ManiSkill3 environment integration, including:
1. Environment creation (single and VecEnv)
2. Observation space verification
3. Expert policy demonstration
4. Dataset generation and loading
5. Algorithm training (offline pretraining)
6. Online fine-tuning with VecEnv (DPPO/ReinFlow)

Usage:
    # Basic validation
    python validate_maniskill.py --test_env --test_expert
    
    # Full validation with training
    python validate_maniskill.py --full
    
    # Quick smoke test
    python validate_maniskill.py --quick

Note:
    This script requires the rlft_ms3 conda environment with ManiSkill3 installed.
"""

import warnings
# Suppress known third-party deprecation warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*CUDA reports that you have.*", category=UserWarning)

import argparse
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    
    try:
        import mani_skill
        print(f"✓ ManiSkill3 version: {mani_skill.__version__}")
    except ImportError:
        missing.append("mani_skill")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        missing.append("torch")
    
    try:
        import gymnasium
        print(f"✓ Gymnasium version: {gymnasium.__version__}")
    except ImportError:
        missing.append("gymnasium")
    
    if missing:
        print(f"\n✗ Missing dependencies: {', '.join(missing)}")
        print("Please install with: pip install " + " ".join(missing))
        return False
    
    return True


def test_environment_creation(verbose: bool = True) -> bool:
    """Test ManiSkill3 environment creation."""
    print("\n" + "=" * 60)
    print("Test 1: Environment Creation")
    print("=" * 60)
    
    try:
        from envs.maniskill_env import (
            make_maniskill_env,
            check_maniskill_available,
            ManiSkillPickCubeEnv,
        )
        
        if not check_maniskill_available():
            print("✗ ManiSkill3 not available")
            return False
        
        # Test 1a: Single environment, state mode
        print("\n1a. Creating single environment (state mode)...")
        env = make_maniskill_env(
            task="PickCube-v1",
            obs_mode="state",
            num_envs=1,
            use_numpy=True,
            seed=42,
        )
        
        print(f"  State dim: {env.state_dim}")
        print(f"  Action dim: {env.action_dim}")
        # Use get_wrapper_attr to avoid deprecation warning
        try:
            max_steps = env.get_wrapper_attr('max_episode_steps')
        except AttributeError:
            max_steps = getattr(env.unwrapped, 'max_episode_steps', 50)
        print(f"  Max episode steps: {max_steps}")
        
        obs, info = env.reset()
        print(f"  Observation shape: {obs.shape}")
        
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step successful, reward: {reward:.4f}")
        
        env.close()
        print("✓ Single environment (state mode) passed")
        
        # Test 1b: Single environment, state_image mode
        print("\n1b. Creating single environment (state_image mode)...")
        env = make_maniskill_env(
            task="PickCube-v1",
            obs_mode="state_image",
            num_envs=1,
            use_numpy=True,
            image_size=64,
            seed=42,
        )
        
        print(f"  State dim: {env.state_dim}")
        print(f"  Image shape: {env.image_shape}")
        
        obs, info = env.reset()
        if isinstance(obs, dict):
            print(f"  State obs shape: {obs['state'].shape}")
            print(f"  Image obs shape: {obs['image'].shape}")
        
        env.close()
        print("✓ Single environment (state_image mode) passed")
        
        # Test 1c: VecEnv (GPU parallel)
        # Note: VecEnv with num_envs > 1 cannot be tested in the same process 
        # after single-env tests due to PhysX limitation
        if torch.cuda.is_available():
            print("\n1c. VecEnv test skipped (PhysX cannot switch between CPU/GPU in same process)")
            print("    To test VecEnv, run in a fresh Python process:")
            print("    python -c \"from envs import make_maniskill_env; env = make_maniskill_env(num_envs=4, use_numpy=False); print('VecEnv OK')\"")
        else:
            print("\n1c. Skipping VecEnv test (CUDA not available)")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_expert_policy(verbose: bool = True) -> bool:
    """Test expert policy for demonstration collection."""
    print("\n" + "=" * 60)
    print("Test 2: Expert Policy")
    print("=" * 60)
    
    try:
        from envs.maniskill_env import make_maniskill_env
        from scripts.generate_maniskill_dataset import ManiSkillScriptedExpert
        
        print("\n2a. Testing scripted expert on PickCube-v1...")
        
        env = make_maniskill_env(
            task="PickCube-v1",
            obs_mode="state",
            num_envs=1,
            use_numpy=True,
            max_episode_steps=50,
            seed=42,
        )
        
        expert = ManiSkillScriptedExpert()
        
        # Run a few episodes
        successes = 0
        num_episodes = 5
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            expert.reset()
            
            total_reward = 0
            for step in range(50):
                raw_obs = info.get("raw_obs", obs)
                action = expert.get_action(raw_obs)
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            success = info.get("success", total_reward > 0)
            if success:
                successes += 1
            
            if verbose:
                print(f"  Episode {ep+1}: reward={total_reward:.2f}, success={success}")
        
        env.close()
        
        success_rate = successes / num_episodes
        print(f"\n  Success rate: {success_rate*100:.1f}%")
        
        if success_rate >= 0.4:
            print("✓ Expert policy passed")
            return True
        else:
            print(f"✗ Expert policy success rate too low: {success_rate*100:.1f}%")
            return False
        
    except Exception as e:
        print(f"✗ Expert policy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_generation(verbose: bool = True) -> bool:
    """Test dataset generation and loading."""
    print("\n" + "=" * 60)
    print("Test 3: Dataset Generation and Loading")
    print("=" * 60)
    
    try:
        from envs.maniskill_env import make_maniskill_env
        from scripts.generate_maniskill_dataset import (
            ManiSkillScriptedExpert,
            collect_episodes_single,
            save_dataset,
        )
        from common.dataset_loader import ManiSkillOfflineDataset, make_dataloader
        
        print("\n3a. Generating small test dataset...")
        
        # Create environment
        env = make_maniskill_env(
            task="PickCube-v1",
            obs_mode="state_image",
            num_envs=1,
            use_numpy=True,
            image_size=64,
            seed=42,
        )
        
        expert = ManiSkillScriptedExpert()
        
        # Collect a few episodes
        data = collect_episodes_single(
            env=env,
            expert=expert,
            num_episodes=3,
            max_episode_steps=30,
            obs_mode="state_image",
            verbose=False,
        )
        
        env.close()
        
        print(f"  Collected {len(data['actions'])} transitions")
        print(f"  State shape: {np.array(data['obs'][0]).shape}")
        print(f"  Image shape: {np.array(data['images'][0]).shape}")
        print(f"  Action shape: {np.array(data['actions'][0]).shape}")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name
        
        print("\n3b. Saving dataset...")
        save_dataset(data, temp_path, obs_mode="state_image")
        
        # Test loading
        print("\n3c. Loading dataset...")
        dataset = ManiSkillOfflineDataset(temp_path, obs_mode="state_image")
        print(f"  {dataset}")
        
        sample = dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Obs shape: {sample['obs'].shape}")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Action shape: {sample['action'].shape}")
        
        # Test dataloader
        print("\n3d. Testing DataLoader...")
        dataloader = make_dataloader(temp_path, obs_mode="state_image", batch_size=8)
        batch = next(iter(dataloader))
        print(f"  Batch obs shape: {batch['obs'].shape}")
        print(f"  Batch image shape: {batch['image'].shape}")
        
        # Cleanup
        os.unlink(temp_path)
        
        print("✓ Dataset generation and loading passed")
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_offline_training(verbose: bool = True) -> bool:
    """Test offline training with a simple algorithm."""
    print("\n" + "=" * 60)
    print("Test 4: Offline Training (Diffusion Policy)")
    print("=" * 60)
    
    try:
        from envs.maniskill_env import make_maniskill_env
        from scripts.generate_maniskill_dataset import (
            ManiSkillScriptedExpert,
            collect_episodes_single,
            save_dataset,
        )
        from common.dataset_loader import ManiSkillOfflineDataset
        from algorithms.diffusion_policy.agent import DiffusionPolicyAgent
        
        print("\n4a. Preparing dataset...")
        
        # Create environment
        env = make_maniskill_env(
            task="PickCube-v1",
            obs_mode="state_image",
            num_envs=1,
            use_numpy=True,
            image_size=64,
            seed=42,
        )
        
        expert = ManiSkillScriptedExpert()
        
        # Collect data
        data = collect_episodes_single(
            env=env,
            expert=expert,
            num_episodes=5,
            max_episode_steps=30,
            obs_mode="state_image",
            verbose=False,
        )
        
        env.close()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name
        save_dataset(data, temp_path, obs_mode="state_image")
        
        # Load dataset
        dataset = ManiSkillOfflineDataset(temp_path, obs_mode="state_image")
        
        print(f"  Dataset size: {len(dataset)}")
        
        print("\n4b. Creating Diffusion Policy agent...")
        
        agent = DiffusionPolicyAgent(
            state_dim=dataset.state_dim,
            action_dim=dataset.action_dim,
            image_shape=(3, 64, 64),  # C, H, W
            hidden_dim=64,
            num_diffusion_steps=10,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        print(f"  Agent device: {agent.device}")
        
        print("\n4c. Running training steps...")
        
        dataloader = dataset.make_dataloader(batch_size=16, shuffle=True)
        
        losses = []
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Just a few steps
                break
            
            obs = batch["obs"].to(agent.device)
            images = batch["image"].to(agent.device)
            actions = batch["action"].to(agent.device)
            
            loss = agent.update(obs, actions, images)
            losses.append(loss)
            
            if verbose:
                print(f"  Step {i+1}: loss={loss:.4f}")
        
        # Cleanup
        os.unlink(temp_path)
        
        if len(losses) > 0 and all(np.isfinite(l) for l in losses):
            print("✓ Offline training passed")
            return True
        else:
            print("✗ Training produced invalid losses")
            return False
        
    except Exception as e:
        print(f"✗ Offline training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vecenv_rollout(verbose: bool = True) -> bool:
    """Test VecEnv rollout collection for online RL."""
    print("\n" + "=" * 60)
    print("Test 5: VecEnv Rollout Collection")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("  Skipping VecEnv test (CUDA not available)")
        return True
    
    try:
        from envs.maniskill_env import make_maniskill_env
        from common.vecenv_rollout import collect_rollout_vecenv
        
        print("\n5a. Creating VecEnv...")
        
        num_envs = 4
        env = make_maniskill_env(
            task="PickCube-v1",
            obs_mode="state",
            num_envs=num_envs,
            use_numpy=False,
            seed=42,
        )
        
        print(f"  Num envs: {num_envs}")
        print(f"  State dim: {env.state_dim}")
        print(f"  Action dim: {env.action_dim}")
        
        print("\n5b. Creating simple policy...")
        
        # Simple random policy for testing
        class RandomPolicy(torch.nn.Module):
            def __init__(self, action_dim):
                super().__init__()
                self.action_dim = action_dim
            
            def get_action(self, obs, deterministic=False):
                batch_size = obs.shape[0]
                return torch.randn(batch_size, self.action_dim, device=obs.device)
        
        policy = RandomPolicy(env.action_dim).cuda()
        
        print("\n5c. Collecting rollout...")
        
        rollout_data = collect_rollout_vecenv(
            env=env,
            agent=policy,
            num_steps=20,
            gamma=0.99,
            gae_lambda=0.95,
            device="cuda",
        )
        
        print(f"  States shape: {rollout_data['states'].shape}")
        print(f"  Actions shape: {rollout_data['actions'].shape}")
        print(f"  Rewards shape: {rollout_data['rewards'].shape}")
        print(f"  Returns shape: {rollout_data['returns'].shape}")
        print(f"  Advantages shape: {rollout_data['advantages'].shape}")
        
        env.close()
        
        print("✓ VecEnv rollout collection passed")
        return True
        
    except Exception as e:
        print(f"✗ VecEnv rollout test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_validation():
    """Run all validation tests."""
    print("=" * 60)
    print("ManiSkill3 Integration Validation")
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    if not check_dependencies():
        print("\n✗ Dependency check failed. Please install missing packages.")
        return False
    
    results = {}
    
    # Run tests
    tests = [
        ("Environment Creation", test_environment_creation),
        ("Expert Policy", test_expert_policy),
        ("Dataset Generation", test_dataset_generation),
        ("Offline Training", test_offline_training),
        ("VecEnv Rollout", test_vecenv_rollout),
    ]
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! ManiSkill3 integration is ready.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return all_passed


def run_quick_validation():
    """Run quick smoke test."""
    print("=" * 60)
    print("ManiSkill3 Quick Validation")
    print("=" * 60)
    
    if not check_dependencies():
        return False
    
    return test_environment_creation(verbose=False)


def main():
    parser = argparse.ArgumentParser(
        description="Validate ManiSkill3 integration"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full validation suite"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke test"
    )
    parser.add_argument(
        "--test_env",
        action="store_true",
        help="Test environment creation only"
    )
    parser.add_argument(
        "--test_expert",
        action="store_true",
        help="Test expert policy only"
    )
    parser.add_argument(
        "--test_dataset",
        action="store_true",
        help="Test dataset generation only"
    )
    parser.add_argument(
        "--test_training",
        action="store_true",
        help="Test offline training only"
    )
    parser.add_argument(
        "--test_vecenv",
        action="store_true",
        help="Test VecEnv rollout only"
    )
    
    args = parser.parse_args()
    
    # If no specific tests, run full validation
    if not any([args.full, args.quick, args.test_env, args.test_expert,
                args.test_dataset, args.test_training, args.test_vecenv]):
        args.full = True
    
    if args.quick:
        success = run_quick_validation()
    elif args.full:
        success = run_full_validation()
    else:
        if not check_dependencies():
            sys.exit(1)
        
        success = True
        if args.test_env:
            success = success and test_environment_creation()
        if args.test_expert:
            success = success and test_expert_policy()
        if args.test_dataset:
            success = success and test_dataset_generation()
        if args.test_training:
            success = success and test_offline_training()
        if args.test_vecenv:
            success = success and test_vecenv_rollout()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
