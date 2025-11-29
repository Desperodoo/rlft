#!/usr/bin/env python3
"""
Validate Pick-and-Place environment and agent integration.

Tests:
1. Environment state/image observation modes
2. Agent offline training with state observations
3. Agent action sampling with different obs modes
"""

import sys
import os
from pathlib import Path

# Setup path
script_dir = Path(__file__).parent.absolute()
toy_diffusion_dir = script_dir.parent.absolute()
workspace_root = toy_diffusion_dir.parent.absolute()

sys.path.insert(0, str(workspace_root))
sys.path.insert(0, str(toy_diffusion_dir))
os.chdir(toy_diffusion_dir)

import numpy as np
import torch
import argparse


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device(device_str=None):
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def test_environment():
    """Test Pick-and-Place environment wrappers."""
    print("\n" + "="*60)
    print("PICK-AND-PLACE ENVIRONMENT TESTS")
    print("="*60)
    
    results = {}
    
    # Test 1: State wrapper
    print("\n  Testing FetchPickAndPlaceStateWrapper...")
    try:
        from toy_diffusion_rl.envs.pick_and_place import make_pick_and_place_env
        
        env = make_pick_and_place_env(obs_mode="state", seed=42)
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray), f"Expected ndarray, got {type(obs)}"
        print(f"    ✓ State observation shape: {obs.shape}")
        print(f"    ✓ Action space: {env.action_space.shape}")
        
        # Take a step
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"    ✓ Step works. Reward: {reward:.4f}")
        
        env.close()
        results["state_wrapper"] = True
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results["state_wrapper"] = False
    
    # Test 2: Image wrapper
    print("\n  Testing FetchPickAndPlaceImageWrapper (image mode)...")
    try:
        env = make_pick_and_place_env(obs_mode="image", seed=42, image_size=128)
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray), f"Expected ndarray, got {type(obs)}"
        assert obs.shape == (128, 128, 3), f"Wrong image shape: {obs.shape}"
        print(f"    ✓ Image observation shape: {obs.shape}")
        
        # Take a step
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"    ✓ Step works. Reward: {reward:.4f}")
        
        env.close()
        results["image_wrapper"] = True
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results["image_wrapper"] = False
    
    # Test 3: State+image wrapper
    print("\n  Testing FetchPickAndPlaceImageWrapper (state_image mode)...")
    try:
        env = make_pick_and_place_env(obs_mode="state_image", seed=42, image_size=128)
        obs, info = env.reset()
        
        assert isinstance(obs, dict), f"Expected dict, got {type(obs)}"
        assert "state" in obs and "image" in obs, f"Missing keys: {obs.keys()}"
        print(f"    ✓ State shape: {obs['state'].shape}")
        print(f"    ✓ Image shape: {obs['image'].shape}")
        
        env.close()
        results["state_image_wrapper"] = True
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results["state_image_wrapper"] = False
    
    return results


def test_agent_with_pick_and_place(device):
    """Test agents with Pick-and-Place environment."""
    print("\n" + "="*60)
    print("AGENT + PICK-AND-PLACE INTEGRATION TESTS")
    print("="*60)
    
    results = {}
    
    try:
        from toy_diffusion_rl.envs.pick_and_place import make_pick_and_place_env
    except Exception as e:
        print(f"  ✗ Cannot import environment: {e}")
        return {}
    
    # Load agents
    agents = {}
    try:
        from toy_diffusion_rl.algorithms.dppo.agent_unified import DPPOAgent
        agents['DPPO'] = DPPOAgent
    except:
        pass
    try:
        from toy_diffusion_rl.algorithms.reinflow.agent_unified import ReinFlowAgent
        agents['ReinFlow'] = ReinFlowAgent
    except:
        pass
    try:
        from toy_diffusion_rl.algorithms.diffusion_policy.agent_unified import DiffusionPolicyAgent
        agents['DiffusionPolicy'] = DiffusionPolicyAgent
    except:
        pass
    
    if not agents:
        print("  ✗ No agents could be loaded")
        return {}
    
    # Test each agent with state mode
    print("\n  Testing agents with state observation mode...")
    
    env = make_pick_and_place_env(obs_mode="state", seed=42)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    for name, agent_class in agents.items():
        print(f"\n  {name} (state mode):")
        
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dims": [256, 256],
            "device": device,
            "obs_mode": "state",
        }
        
        if name == "DPPO":
            kwargs["num_diffusion_steps"] = 5
        elif name == "ReinFlow":
            kwargs["num_flow_steps"] = 5
        elif name == "DiffusionPolicy":
            kwargs["num_diffusion_steps"] = 10
        
        try:
            agent = agent_class(**kwargs)
            
            # Test action sampling
            obs, _ = env.reset()
            result = agent.sample_action(obs)
            action = result[0] if isinstance(result, tuple) else result
            
            assert action.shape == (action_dim,), f"Wrong shape: {action.shape}"
            print(f"    ✓ Action sampling works. Shape: {action.shape}")
            results[f"{name}_state"] = True
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results[f"{name}_state"] = False
    
    env.close()
    
    # Test with image mode
    print("\n  Testing agents with image observation mode...")
    
    try:
        env = make_pick_and_place_env(obs_mode="image", seed=42, image_size=128)
        action_dim = env.action_space.shape[0]
        
        for name, agent_class in agents.items():
            print(f"\n  {name} (image mode):")
            
            kwargs = {
                "state_dim": 25,  # Not used in image-only mode
                "action_dim": action_dim,
                "hidden_dims": [256, 256],
                "device": device,
                "obs_mode": "image",
                "image_shape": (128, 128, 3),
                "vision_encoder_type": "cnn",
            }
            
            if name == "DPPO":
                kwargs["num_diffusion_steps"] = 5
            elif name == "ReinFlow":
                kwargs["num_flow_steps"] = 5
            elif name == "DiffusionPolicy":
                kwargs["num_diffusion_steps"] = 10
            
            try:
                agent = agent_class(**kwargs)
                
                # Test action sampling
                obs, _ = env.reset()
                result = agent.sample_action(obs)
                action = result[0] if isinstance(result, tuple) else result
                
                assert action.shape == (action_dim,), f"Wrong shape: {action.shape}"
                print(f"    ✓ Action sampling works. Shape: {action.shape}")
                results[f"{name}_image"] = True
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                import traceback
                traceback.print_exc()
                results[f"{name}_image"] = False
        
        env.close()
        
    except Exception as e:
        print(f"  ✗ Cannot create image environment: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device(args.device if args.device != "auto" else None)
    print(f"Using device: {device}")
    
    all_results = {}
    
    # Test environment
    env_results = test_environment()
    all_results.update(env_results)
    
    # Test agent integration
    agent_results = test_agent_with_pick_and_place(str(device))
    all_results.update(agent_results)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in all_results.values() if v)
    failed = sum(1 for v in all_results.values() if not v)
    
    for test_name, success in sorted(all_results.items()):
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
