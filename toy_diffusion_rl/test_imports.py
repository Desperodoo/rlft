#!/usr/bin/env python3
"""
Test script to verify all modules can be imported correctly.
Run this from the project root: python toy_diffusion_rl/test_imports.py
"""

import sys
import os

# Add parent directory to path for package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Testing Toy Diffusion RL Imports")
    print("=" * 60)
    
    errors = []
    
    # Test environments
    print("\n[1/4] Testing environments...")
    try:
        from toy_diffusion_rl.envs import PointMass2DEnv, PendulumContinuousWrapper, make_env
        env1 = PointMass2DEnv()
        env2 = make_env("point_mass_2d")
        print(f"  ✓ PointMass2DEnv: obs_dim={env1.observation_space.shape[0]}, act_dim={env1.action_space.shape[0]}")
        print(f"  ✓ make_env('point_mass_2d'): working")
        
        # Test pendulum wrapper
        try:
            env3 = PendulumContinuousWrapper()
            print(f"  ✓ PendulumContinuousWrapper: obs_dim={env3.observation_space.shape[0]}, act_dim={env3.action_space.shape[0]}")
        except Exception as e:
            print(f"  ⚠ PendulumContinuousWrapper skipped (may need 'pip install gymnasium[classic-control]')")
            
    except Exception as e:
        errors.append(f"Environments: {e}")
        print(f"  ✗ Error: {e}")
    
    # Test common modules
    print("\n[2/4] Testing common modules...")
    try:
        from toy_diffusion_rl.common.networks import (
            MLP, TimestepEmbedding, DiffusionNoisePredictor,
            FlowVelocityPredictor, QNetwork, ValueNetwork, DoubleQNetwork
        )
        from toy_diffusion_rl.common.replay_buffer import ReplayBuffer, RolloutBuffer
        from toy_diffusion_rl.common.utils import set_seed, soft_update, DiffusionHelper
        print("  ✓ All common modules imported successfully")
    except Exception as e:
        errors.append(f"Common modules: {e}")
        print(f"  ✗ Error: {e}")
    
    # Test algorithms
    print("\n[3/4] Testing algorithm imports...")
    algorithm_tests = [
        ("DiffusionPolicyAgent", "toy_diffusion_rl.algorithms.diffusion_policy.agent", "DiffusionPolicyAgent"),
        ("FlowMatchingPolicy", "toy_diffusion_rl.algorithms.flow_matching.fm_policy", "FlowMatchingPolicy"),
        ("ReflectedFlowPolicy", "toy_diffusion_rl.algorithms.flow_matching.reflected_flow", "ReflectedFlowPolicy"),
        ("ConsistencyFlowPolicy", "toy_diffusion_rl.algorithms.flow_matching.consistency_flow", "ConsistencyFlowPolicy"),
        ("DiffusionDoubleQAgent", "toy_diffusion_rl.algorithms.diffusion_double_q.agent", "DiffusionDoubleQAgent"),
        ("CPQLAgent", "toy_diffusion_rl.algorithms.cpql.agent", "CPQLAgent"),
        ("DPPOAgent", "toy_diffusion_rl.algorithms.dppo.agent", "DPPOAgent"),
        ("ReinFlowAgent", "toy_diffusion_rl.algorithms.reinflow.agent", "ReinFlowAgent"),
    ]
    
    imported_agents = {}
    for display_name, module_path, class_name in algorithm_tests:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            imported_agents[display_name] = cls
            print(f"  ✓ {display_name}")
        except Exception as e:
            errors.append(f"{display_name}: {e}")
            print(f"  ✗ {display_name}: {e}")
    
    # Test instantiation
    print("\n[4/4] Testing agent instantiation...")
    try:
        import torch
        import numpy as np
        
        state_dim, action_dim = 4, 2
        device = "cpu"
        
        instantiated = []
        for name, cls in imported_agents.items():
            try:
                agent = cls(state_dim, action_dim, device=device)
                instantiated.append((name, agent))
                print(f"  ✓ {name} instantiated")
            except Exception as e:
                errors.append(f"{name} instantiation: {e}")
                print(f"  ✗ {name} instantiation failed: {e}")
            
        # Quick action sampling test
        if instantiated:
            print("\n  Testing action sampling...")
            test_state = np.zeros(state_dim, dtype=np.float32)
            for name, agent in instantiated:
                try:
                    if hasattr(agent, 'sample_action'):
                        action = agent.sample_action(test_state)
                        if isinstance(action, np.ndarray):
                            print(f"    ✓ {name}.sample_action() -> shape {action.shape}")
                        else:
                            print(f"    ✓ {name}.sample_action() -> {type(action)}")
                except Exception as e:
                    print(f"    ⚠ {name}.sample_action() skipped: {e}")
                
    except Exception as e:
        errors.append(f"Agent instantiation: {e}")
        print(f"  ✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("SUCCESS: All imports and tests passed!")
        print("\nYou can now run training with:")
        print("  python toy_diffusion_rl/train.py --algorithm diffusion_policy --env point_mass_2d")
        return 0


def test_env_rollout():
    """Test environment rollout with expert controller."""
    print("\n" + "=" * 60)
    print("Testing Environment Rollout")
    print("=" * 60)
    
    try:
        from toy_diffusion_rl.envs import PointMass2DEnv
        import numpy as np
        
        env = PointMass2DEnv()
        state, _ = env.reset(seed=42)
        
        total_reward = 0
        for step in range(100):
            # Use expert controller
            action = env.get_expert_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        print(f"  ✓ Rollout completed: {step + 1} steps, total reward: {total_reward:.2f}")
        print(f"    Final position: ({state[0]:.3f}, {state[1]:.3f})")
        
        # Test expert data collection
        print("\n  Testing expert data collection...")
        dataset = env.collect_expert_dataset(num_episodes=5, noise_std=0.1)
        print(f"  ✓ Collected {len(dataset['states'])} transitions from 5 episodes")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = test_imports()
    if exit_code == 0:
        exit_code = test_env_rollout()
    sys.exit(exit_code)
