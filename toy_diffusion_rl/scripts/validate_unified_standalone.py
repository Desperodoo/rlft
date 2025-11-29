#!/usr/bin/env python3
"""
Standalone validation script for unified multimodal agents.

This script tests that all unified agents can train offline 
on particle environment with state-only mode.
"""

import sys
import os
from pathlib import Path

# Setup path - go up to workspace root (rlft)
script_dir = Path(__file__).parent.absolute()  # scripts/
toy_diffusion_dir = script_dir.parent.absolute()  # toy_diffusion_rl/
workspace_root = toy_diffusion_dir.parent.absolute()  # rlft/

# Add workspace root to path for package imports
sys.path.insert(0, str(workspace_root))
# Also add toy_diffusion_rl for direct imports
sys.path.insert(0, str(toy_diffusion_dir))
os.chdir(toy_diffusion_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


# ====== Minimal imports from common ======
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get device for computation."""
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ====== Import environment directly ======
from toy_diffusion_rl.envs.multimodal_particle import MultimodalParticleEnv


# ====== Simple Replay Buffer ======
class SimpleBuffer:
    """Simple replay buffer for testing."""
    
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 10000, device: str = "cpu"):
        self.device = device
        self.capacity = capacity
        self.states = torch.zeros(capacity, state_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.next_states = torch.zeros(capacity, state_dim, device=device)
        self.dones = torch.zeros(capacity, device=device)
        self.size = 0
        self.ptr = 0
    
    def load_dataset(self, data: Dict[str, np.ndarray]):
        """Load dataset into buffer."""
        n = len(data['states'])
        self.states[:n] = torch.from_numpy(data['states']).float().to(self.device)
        self.actions[:n] = torch.from_numpy(data['actions']).float().to(self.device)
        self.rewards[:n] = torch.from_numpy(data['rewards']).float().to(self.device)
        self.next_states[:n] = torch.from_numpy(data['next_states']).float().to(self.device)
        self.dones[:n] = torch.from_numpy(data['dones']).float().to(self.device)
        self.size = n
        self.ptr = n
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch from buffer."""
        indices = np.random.randint(0, self.size, batch_size)
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
        }


# ====== Load unified agents ======
def load_unified_agents():
    """Dynamically load unified agent classes."""
    agents = {}
    
    # DiffusionPolicy
    try:
        from toy_diffusion_rl.algorithms.diffusion_policy.agent_unified import DiffusionPolicyAgent
        agents['DiffusionPolicy'] = DiffusionPolicyAgent
        print("  ✓ DiffusionPolicy loaded")
    except Exception as e:
        print(f"  ✗ DiffusionPolicy failed: {e}")
    
    # FlowMatching
    try:
        from toy_diffusion_rl.algorithms.flow_matching.fm_policy_unified import FlowMatchingPolicy
        agents['FlowMatching'] = FlowMatchingPolicy
        print("  ✓ FlowMatching loaded")
    except Exception as e:
        print(f"  ✗ FlowMatching failed: {e}")
    
    # DiffusionDoubleQ
    try:
        from toy_diffusion_rl.algorithms.diffusion_double_q.agent_unified import DiffusionDoubleQAgent
        agents['DiffusionDoubleQ'] = DiffusionDoubleQAgent
        print("  ✓ DiffusionDoubleQ loaded")
    except Exception as e:
        print(f"  ✗ DiffusionDoubleQ failed: {e}")
    
    # CPQL
    try:
        from toy_diffusion_rl.algorithms.cpql.agent_unified import CPQLAgent
        agents['CPQL'] = CPQLAgent
        print("  ✓ CPQL loaded")
    except Exception as e:
        print(f"  ✗ CPQL failed: {e}")
    
    # DPPO
    try:
        from toy_diffusion_rl.algorithms.dppo.agent_unified import DPPOAgent
        agents['DPPO'] = DPPOAgent
        print("  ✓ DPPO loaded")
    except Exception as e:
        print(f"  ✗ DPPO failed: {e}")
    
    # ReinFlow
    try:
        from toy_diffusion_rl.algorithms.reinflow.agent_unified import ReinFlowAgent
        agents['ReinFlow'] = ReinFlowAgent
        print("  ✓ ReinFlow loaded")
    except Exception as e:
        print(f"  ✗ ReinFlow failed: {e}")
    
    return agents


def test_offline_training(
    agent_class,
    agent_name: str,
    env: MultimodalParticleEnv,
    num_steps: int = 50,
    batch_size: int = 32,
    device: str = "cuda",
):
    """Test offline training on particle environment."""
    print(f"\n  Testing {agent_name} - Offline Training...")
    
    # Skip offline test for online-only algorithms
    if agent_name in ["DPPO", "ReinFlow"]:
        print(f"    ⊘ Skipped (online-only algorithm)")
        return None  # Return None to indicate skip, not fail
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create agent with appropriate parameters
    agent_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dims": [64, 64],  # Small network for fast testing
        "device": device,
        "obs_mode": "state",
    }
    
    # Add algorithm-specific parameters
    if agent_name == "DPPO":
        agent_kwargs["num_diffusion_steps"] = 3
    elif agent_name == "ReinFlow":
        agent_kwargs["num_flow_steps"] = 3
    elif agent_name == "DiffusionPolicy":
        agent_kwargs["num_diffusion_steps"] = 5
    elif agent_name == "FlowMatching":
        agent_kwargs["num_inference_steps"] = 3
    elif agent_name == "DiffusionDoubleQ":
        agent_kwargs["num_diffusion_steps"] = 3
    elif agent_name == "CPQL":
        pass  # Use defaults
    
    try:
        agent = agent_class(**agent_kwargs)
    except Exception as e:
        print(f"    ✗ Agent creation failed: {e}")
        return False
    
    # Collect expert dataset
    expert_data = env.collect_expert_dataset(num_episodes=20)
    
    # Create replay buffer
    buffer = SimpleBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    buffer.load_dataset(expert_data)
    
    # Training loop
    losses = []
    try:
        for step in range(num_steps):
            batch = buffer.sample(batch_size)
            metrics = agent.train_step(batch)
            loss = metrics.get('loss', metrics.get('actor_loss', 0))
            losses.append(loss)
        
        avg_loss = np.mean(losses[-10:])
        print(f"    ✓ Training completed. Avg loss: {avg_loss:.4f}")
        return True
        
    except Exception as e:
        print(f"    ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_sampling(
    agent_class,
    agent_name: str,
    env: MultimodalParticleEnv,
    device: str = "cuda",
):
    """Test action sampling."""
    print(f"  Testing {agent_name} - Action Sampling...")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dims": [64, 64],
        "device": device,
        "obs_mode": "state",
    }
    
    if agent_name == "DPPO":
        agent_kwargs["num_diffusion_steps"] = 3
    elif agent_name == "ReinFlow":
        agent_kwargs["num_flow_steps"] = 3
    elif agent_name == "DiffusionPolicy":
        agent_kwargs["num_diffusion_steps"] = 5
    elif agent_name == "FlowMatching":
        agent_kwargs["num_inference_steps"] = 3
    elif agent_name == "DiffusionDoubleQ":
        agent_kwargs["num_diffusion_steps"] = 3
    
    try:
        agent = agent_class(**agent_kwargs)
        state, _ = env.reset()
        
        result = agent.sample_action(state)
        action = result[0] if isinstance(result, tuple) else result
        
        assert action.shape == (action_dim,), f"Wrong action shape: {action.shape}"
        print(f"    ✓ Action sampling works. Action shape: {action.shape}")
        return True
        
    except Exception as e:
        print(f"    ✗ Action sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_online_training(
    agent_class,
    agent_name: str,
    env: MultimodalParticleEnv,
    num_iterations: int = 2,
    rollout_steps: int = 32,
    device: str = "cuda",
):
    """Test online training (DPPO, ReinFlow only)."""
    print(f"  Testing {agent_name} - Online Training...")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dims": [64, 64],
        "device": device,
        "obs_mode": "state",
    }
    
    if agent_name == "DPPO":
        agent_kwargs["num_diffusion_steps"] = 3
    elif agent_name == "ReinFlow":
        agent_kwargs["num_flow_steps"] = 3
    
    try:
        agent = agent_class(**agent_kwargs)
        
        for iteration in range(num_iterations):
            rollout = agent.collect_rollout(env, rollout_steps)
            metrics = agent.update(rollout)  # Use 'update' instead of 'online_update'
        
        print(f"    ✓ Online training completed")
        return True
        
    except Exception as e:
        print(f"    ✗ Online training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=50)
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device if args.device != "auto" else None)
    print(f"Using device: {device}")
    
    # Create environment
    env = MultimodalParticleEnv(
        distribution_type="ring",
        num_modes=8,
        scale=2.0,
        noise_std=0.1,
        context_dim=0,
        reward_type="density"
    )
    print(f"Environment: MultimodalParticle (ring)")
    print(f"  State dim: {env.observation_space.shape[0]}")
    print(f"  Action dim: {env.action_space.shape[0]}")
    
    # Load agents
    print("\nLoading unified agents...")
    agents = load_unified_agents()
    
    if not agents:
        print("\n✗ No agents could be loaded!")
        return False
    
    # Run tests
    results = {}
    
    print("\n" + "="*60)
    print("OFFLINE TRAINING TESTS")
    print("="*60)
    
    offline_agents = ["DiffusionPolicy", "FlowMatching", "DiffusionDoubleQ", "CPQL"]
    online_agents = ["DPPO", "ReinFlow"]
    
    for name in offline_agents + online_agents:
        if name in agents:
            success = test_offline_training(
                agents[name], name, env,
                num_steps=args.num_steps,
                device=str(device)
            )
            results[f"{name}_offline"] = success
    
    print("\n" + "="*60)
    print("ACTION SAMPLING TESTS")
    print("="*60)
    
    for name in offline_agents + online_agents:
        if name in agents:
            success = test_action_sampling(
                agents[name], name, env,
                device=str(device)
            )
            results[f"{name}_sample"] = success
    
    print("\n" + "="*60)
    print("ONLINE TRAINING TESTS (DPPO, ReinFlow)")
    print("="*60)
    
    for name in online_agents:
        if name in agents:
            success = test_online_training(
                agents[name], name, env,
                device=str(device)
            )
            results[f"{name}_online"] = success
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for test_name, success in sorted(results.items()):
        if success is True:
            status = "✓ PASS"
        elif success is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
