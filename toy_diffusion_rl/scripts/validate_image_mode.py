#!/usr/bin/env python3
"""
Validate unified agents with image observation mode.

This script tests that all unified agents work with image observations.
"""

import sys
import os
from pathlib import Path

# Setup path - go up to workspace root (rlft)
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
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device(device_str=None):
    """Get device for computation."""
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_unified_agents():
    """Load all unified agent classes."""
    agents = {}
    
    try:
        from toy_diffusion_rl.algorithms.diffusion_policy.agent_unified import DiffusionPolicyAgent
        agents['DiffusionPolicy'] = DiffusionPolicyAgent
    except Exception as e:
        print(f"  ✗ DiffusionPolicy: {e}")
    
    try:
        from toy_diffusion_rl.algorithms.flow_matching.fm_policy_unified import FlowMatchingPolicy
        agents['FlowMatching'] = FlowMatchingPolicy
    except Exception as e:
        print(f"  ✗ FlowMatching: {e}")
    
    try:
        from toy_diffusion_rl.algorithms.diffusion_double_q.agent_unified import DiffusionDoubleQAgent
        agents['DiffusionDoubleQ'] = DiffusionDoubleQAgent
    except Exception as e:
        print(f"  ✗ DiffusionDoubleQ: {e}")
    
    try:
        from toy_diffusion_rl.algorithms.cpql.agent_unified import CPQLAgent
        agents['CPQL'] = CPQLAgent
    except Exception as e:
        print(f"  ✗ CPQL: {e}")
    
    try:
        from toy_diffusion_rl.algorithms.dppo.agent_unified import DPPOAgent
        agents['DPPO'] = DPPOAgent
    except Exception as e:
        print(f"  ✗ DPPO: {e}")
    
    try:
        from toy_diffusion_rl.algorithms.reinflow.agent_unified import ReinFlowAgent
        agents['ReinFlow'] = ReinFlowAgent
    except Exception as e:
        print(f"  ✗ ReinFlow: {e}")
    
    return agents


def test_image_mode_instantiation(agents, device):
    """Test that agents can be instantiated with image mode."""
    results = {}
    
    print("\n" + "="*60)
    print("IMAGE MODE INSTANTIATION TESTS")
    print("="*60)
    
    image_shape = (128, 128, 3)  # HWC format
    action_dim = 2
    
    for name, agent_class in agents.items():
        print(f"\n  Testing {name}...")
        
        kwargs = {
            "state_dim": 4,  # Even for image mode, some agents need state_dim
            "action_dim": action_dim,
            "hidden_dims": [64, 64],
            "device": device,
            "obs_mode": "image",
            "image_shape": image_shape,
            "vision_encoder_type": "cnn",
        }
        
        # Algorithm-specific params
        if name == "DPPO":
            kwargs["num_diffusion_steps"] = 3
        elif name == "ReinFlow":
            kwargs["num_flow_steps"] = 3
        elif name == "DiffusionPolicy":
            kwargs["num_diffusion_steps"] = 5
        elif name == "FlowMatching":
            kwargs["num_inference_steps"] = 3
        elif name == "DiffusionDoubleQ":
            kwargs["num_diffusion_steps"] = 3
        
        try:
            agent = agent_class(**kwargs)
            print(f"    ✓ Agent created with image mode")
            results[name] = True
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    return results


def test_image_mode_action_sampling(agents, device):
    """Test action sampling with image observations."""
    results = {}
    
    print("\n" + "="*60)
    print("IMAGE MODE ACTION SAMPLING TESTS")
    print("="*60)
    
    image_shape = (128, 128, 3)
    action_dim = 2
    batch_size = 4
    
    for name, agent_class in agents.items():
        print(f"\n  Testing {name}...")
        
        kwargs = {
            "state_dim": 4,
            "action_dim": action_dim,
            "hidden_dims": [64, 64],
            "device": device,
            "obs_mode": "image",
            "image_shape": image_shape,
            "vision_encoder_type": "cnn",
        }
        
        if name == "DPPO":
            kwargs["num_diffusion_steps"] = 3
        elif name == "ReinFlow":
            kwargs["num_flow_steps"] = 3
        elif name == "DiffusionPolicy":
            kwargs["num_diffusion_steps"] = 5
        elif name == "FlowMatching":
            kwargs["num_inference_steps"] = 3
        elif name == "DiffusionDoubleQ":
            kwargs["num_diffusion_steps"] = 3
        
        try:
            agent = agent_class(**kwargs)
            
            # Create dummy image observation (HWC format)
            image_obs = np.random.rand(128, 128, 3).astype(np.float32)
            
            # Sample action
            result = agent.sample_action(image_obs)
            action = result[0] if isinstance(result, tuple) else result
            
            assert action.shape == (action_dim,), f"Wrong action shape: {action.shape}"
            print(f"    ✓ Action sampling works. Action shape: {action.shape}")
            results[name] = True
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    return results


def test_state_image_mode_action_sampling(agents, device):
    """Test action sampling with combined state+image observations."""
    results = {}
    
    print("\n" + "="*60)
    print("STATE+IMAGE MODE ACTION SAMPLING TESTS")
    print("="*60)
    
    image_shape = (128, 128, 3)
    state_dim = 4
    action_dim = 2
    
    for name, agent_class in agents.items():
        print(f"\n  Testing {name}...")
        
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dims": [64, 64],
            "device": device,
            "obs_mode": "state_image",
            "image_shape": image_shape,
            "vision_encoder_type": "cnn",
        }
        
        if name == "DPPO":
            kwargs["num_diffusion_steps"] = 3
        elif name == "ReinFlow":
            kwargs["num_flow_steps"] = 3
        elif name == "DiffusionPolicy":
            kwargs["num_diffusion_steps"] = 5
        elif name == "FlowMatching":
            kwargs["num_inference_steps"] = 3
        elif name == "DiffusionDoubleQ":
            kwargs["num_diffusion_steps"] = 3
        
        try:
            agent = agent_class(**kwargs)
            
            # Create dummy combined observation
            # Format: dict with 'state' and 'image' keys
            obs = {
                'state': np.random.rand(state_dim).astype(np.float32),
                'image': np.random.rand(128, 128, 3).astype(np.float32),
            }
            
            # Sample action
            result = agent.sample_action(obs)
            action = result[0] if isinstance(result, tuple) else result
            
            assert action.shape == (action_dim,), f"Wrong action shape: {action.shape}"
            print(f"    ✓ Action sampling works. Action shape: {action.shape}")
            results[name] = True
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device(args.device if args.device != "auto" else None)
    print(f"Using device: {device}")
    
    # Load agents
    print("\nLoading unified agents...")
    agents = load_unified_agents()
    print(f"Loaded {len(agents)} agents: {list(agents.keys())}")
    
    if not agents:
        print("✗ No agents could be loaded!")
        return False
    
    # Run tests
    all_results = {}
    
    # Test 1: Image mode instantiation
    results = test_image_mode_instantiation(agents, str(device))
    all_results.update({f"{k}_image_init": v for k, v in results.items()})
    
    # Test 2: Image mode action sampling
    results = test_image_mode_action_sampling(agents, str(device))
    all_results.update({f"{k}_image_sample": v for k, v in results.items()})
    
    # Test 3: State+image mode action sampling
    results = test_state_image_mode_action_sampling(agents, str(device))
    all_results.update({f"{k}_state_image_sample": v for k, v in results.items()})
    
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
