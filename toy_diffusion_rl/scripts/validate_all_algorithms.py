"""
Validate All Diffusion/Flow RL Algorithms on Multimodal Particle Environment

This script tests all implemented algorithms:
1. Diffusion Policy (BC baseline)
2. Flow Matching Policy (BC baseline)
3. Consistency Flow Policy (single-step generation)
4. Reflected Flow Policy (bounded action spaces)
5. Diffusion Double Q (Offline RL)
6. CPQL (Offline RL with Consistency Models)
7. DPPO (Online RL fine-tuning)
8. ReinFlow (Online RL fine-tuning)

Each algorithm is trained on a multimodal distribution (clusters, ring, etc.)
and we visualize the learned distribution vs the target.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from toy_diffusion_rl.envs import MultimodalParticleEnv, make_env
from toy_diffusion_rl.algorithms.diffusion_policy.agent import DiffusionPolicyAgent
from toy_diffusion_rl.algorithms.flow_matching.fm_policy import FlowMatchingPolicy
from toy_diffusion_rl.algorithms.flow_matching.consistency_flow import ConsistencyFlowPolicy
from toy_diffusion_rl.algorithms.flow_matching.reflected_flow import ReflectedFlowPolicy
from toy_diffusion_rl.algorithms.diffusion_double_q.agent import DiffusionDoubleQAgent
from toy_diffusion_rl.algorithms.cpql.agent import CPQLAgent
from toy_diffusion_rl.algorithms.dppo.agent import DPPOAgent
from toy_diffusion_rl.algorithms.reinflow.agent import ReinFlowAgent


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_offline_dataset(
    env: MultimodalParticleEnv,
    num_samples: int = 5000
) -> Dict[str, np.ndarray]:
    """Create offline dataset from environment."""
    dataset = env.collect_offline_dataset(num_samples=num_samples)
    return dataset


def train_diffusion_policy(
    dataset: Dict[str, np.ndarray],
    state_dim: int,
    action_dim: int,
    num_steps: int = 2000,
    device: str = "cpu"
) -> DiffusionPolicyAgent:
    """Train Diffusion Policy (BC)."""
    agent = DiffusionPolicyAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        num_diffusion_steps=20,
        device=device
    )
    
    # Convert dataset to tensors
    states = torch.FloatTensor(dataset["states"]).to(device)
    actions = torch.FloatTensor(dataset["actions"]).to(device)
    n_samples = len(states)
    batch_size = 256
    
    total_loss = 0
    for step in range(num_steps):
        idx = np.random.randint(0, n_samples, batch_size)
        batch = {
            "states": states[idx],
            "actions": actions[idx],
        }
        metrics = agent.train_step(batch)
        total_loss = metrics['loss']
        
    print(f"  Diffusion Policy - Final loss: {total_loss:.4f}")
    return agent


def train_flow_matching(
    dataset: Dict[str, np.ndarray],
    state_dim: int,
    action_dim: int,
    num_steps: int = 2000,
    device: str = "cpu"
) -> FlowMatchingPolicy:
    """Train Flow Matching Policy (BC)."""
    agent = FlowMatchingPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        num_inference_steps=20,
        device=device
    )
    
    # Convert dataset to tensors
    states = torch.FloatTensor(dataset["states"]).to(device)
    actions = torch.FloatTensor(dataset["actions"]).to(device)
    n_samples = len(states)
    batch_size = 256
    
    total_loss = 0
    for step in range(num_steps):
        idx = np.random.randint(0, n_samples, batch_size)
        batch = {
            "states": states[idx],
            "actions": actions[idx],
        }
        metrics = agent.train_step(batch)
        total_loss = metrics['loss']
        
    print(f"  Flow Matching - Final loss: {total_loss:.4f}")
    return agent


def train_diffusion_double_q(
    dataset: Dict[str, np.ndarray],
    state_dim: int,
    action_dim: int,
    num_steps: int = 2000,
    device: str = "cpu"
) -> DiffusionDoubleQAgent:
    """Train Diffusion Double Q (Offline RL)."""
    agent = DiffusionDoubleQAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        num_diffusion_steps=20,
        device=device
    )
    
    # Train
    batch_size = 256
    n_samples = len(dataset["states"])
    
    for step in range(num_steps):
        idx = np.random.randint(0, n_samples, batch_size)
        batch = {
            "states": torch.FloatTensor(dataset["states"][idx]).to(device),
            "actions": torch.FloatTensor(dataset["actions"][idx]).to(device),
            "rewards": torch.FloatTensor(dataset["rewards"][idx]).to(device),
            "next_states": torch.FloatTensor(dataset["next_states"][idx]).to(device),
            "dones": torch.FloatTensor(dataset["dones"][idx]).to(device),
        }
        metrics = agent.train_step(batch)
        
    print(f"  Diffusion QL - Actor loss: {metrics['actor_loss']:.4f}, Q loss: {metrics['q_loss']:.4f}")
    return agent


def train_cpql(
    dataset: Dict[str, np.ndarray],
    state_dim: int,
    action_dim: int,
    num_steps: int = 2000,
    device: str = "cpu"
) -> CPQLAgent:
    """Train CPQL (Consistency Policy Q-Learning)."""
    agent = CPQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        device=device
    )
    
    # Train
    batch_size = 256
    n_samples = len(dataset["states"])
    
    for step in range(num_steps):
        idx = np.random.randint(0, n_samples, batch_size)
        batch = {
            "states": torch.FloatTensor(dataset["states"][idx]).to(device),
            "actions": torch.FloatTensor(dataset["actions"][idx]).to(device),
            "rewards": torch.FloatTensor(dataset["rewards"][idx]).to(device),
            "next_states": torch.FloatTensor(dataset["next_states"][idx]).to(device),
            "dones": torch.FloatTensor(dataset["dones"][idx]).to(device),
        }
        metrics = agent.train_step(batch)
        
    print(f"  CPQL - Consistency loss: {metrics['consistency_loss']:.4f}, BC loss: {metrics['bc_loss']:.4f}")
    return agent


def train_dppo(
    dataset: Dict[str, np.ndarray],
    state_dim: int,
    action_dim: int,
    pretrain_steps: int = 1000,
    device: str = "cpu"
) -> DPPOAgent:
    """Train DPPO (pretrain only, no online env for this toy task)."""
    agent = DPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        num_diffusion_steps=10,
        ft_denoising_steps=5,
        device=device
    )
    
    # Pretrain with BC
    metrics = agent.pretrain_bc(dataset, num_steps=pretrain_steps, batch_size=256)
    print(f"  DPPO - BC loss: {metrics['bc_loss']:.4f}")
    return agent


def train_reinflow(
    dataset: Dict[str, np.ndarray],
    state_dim: int,
    action_dim: int,
    pretrain_steps: int = 1000,
    device: str = "cpu"
) -> ReinFlowAgent:
    """Train ReinFlow (pretrain only)."""
    agent = ReinFlowAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        num_flow_steps=10,
        noise_scheduler_type="learn",
        device=device
    )
    
    # Pretrain with flow matching
    metrics = agent.offline_pretrain(dataset, num_steps=pretrain_steps, batch_size=256)
    print(f"  ReinFlow - Pretrain loss: {metrics['pretrain_loss']:.4f}")
    return agent


def generate_samples(
    agent,
    state: np.ndarray,
    num_samples: int = 500,
    device: str = "cpu"
) -> np.ndarray:
    """Generate samples from a trained agent."""
    samples = []
    
    for _ in range(num_samples):
        with torch.no_grad():
            # Each agent has different sample_action signatures
            # Most accept just state, some have optional params
            action = agent.sample_action(state)
            if isinstance(action, tuple):
                action = action[0]  # Some agents return (action, log_prob, value)
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            if action.ndim > 1:
                action = action[0]  # Remove batch dimension
            samples.append(action)
    
    samples = np.array(samples)
    return samples


def visualize_training_progress(
    env: MultimodalParticleEnv,
    progress_results: Dict[str, Dict[int, np.ndarray]],
    emd_history: Dict[str, Dict[int, float]],
    save_path: str,
    title: str = ""
):
    """Visualize training progress for all algorithms like DDBM paper figure.
    
    Creates a grid where:
    - Each row is an algorithm
    - Each column is a training checkpoint (step)
    - Last column is the target distribution
    - EMD/MMD is shown below each subplot
    
    Args:
        env: Environment with target distribution
        progress_results: Dict[algorithm_name -> Dict[step -> samples]]
        emd_history: Dict[algorithm_name -> Dict[step -> emd_value]]
        save_path: Path to save figure
        title: Figure title
    """
    algorithms = list(progress_results.keys())
    n_algorithms = len(algorithms)
    
    # Get all checkpoints (steps) from any algorithm
    sample_algo = algorithms[0]
    checkpoints = sorted(progress_results[sample_algo].keys())
    n_checkpoints = len(checkpoints)
    
    # Create figure: rows = algorithms, cols = checkpoints + 1 (for target)
    fig, axes = plt.subplots(
        n_algorithms, n_checkpoints + 1,
        figsize=(2.2 * (n_checkpoints + 1), 2.5 * n_algorithms)
    )
    
    if n_algorithms == 1:
        axes = axes.reshape(1, -1)
    
    # Get target samples once
    target_samples = env.sample_from_distribution(1000)
    
    # Color scheme
    sample_color = '#1f77b4'  # Blue for generated samples
    target_color = '#ff7f0e'  # Orange for target
    
    for row_idx, algo_name in enumerate(algorithms):
        algo_progress = progress_results[algo_name]
        algo_emd = emd_history[algo_name]
        
        # Plot each checkpoint
        for col_idx, step in enumerate(checkpoints):
            ax = axes[row_idx, col_idx]
            
            if step in algo_progress:
                samples = algo_progress[step]
                samples_scaled = samples * env.scale * 1.5
                
                ax.scatter(samples_scaled[:, 0], samples_scaled[:, 1],
                          alpha=0.5, s=3, c=sample_color, rasterized=True)
                
                emd_val = algo_emd.get(step, 0)
                
                # Title only on first row
                if row_idx == 0:
                    ax.set_title(f'Step={step}', fontsize=10)
                    
                # EMD annotation
                ax.text(0.5, -0.15, f'EMD={emd_val:.4f}', 
                       transform=ax.transAxes, ha='center', fontsize=8)
            
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Algorithm name on the left
            if col_idx == 0:
                ax.set_ylabel(algo_name, fontsize=10, rotation=0, ha='right', va='center')
                ax.yaxis.set_label_coords(-0.3, 0.5)
        
        # Plot target in last column
        ax = axes[row_idx, -1]
        ax.scatter(target_samples[:, 0], target_samples[:, 1],
                  alpha=0.5, s=3, c=target_color, rasterized=True)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        if row_idx == 0:
            ax.set_title('Target', fontsize=10)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training progress visualization to {save_path}")


def train_with_progress_tracking(
    env: MultimodalParticleEnv,
    dataset: Dict[str, np.ndarray],
    state_dim: int,
    action_dim: int,
    num_steps: int,
    device: str,
    eval_checkpoints: List[int] = None
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[int, float]]]:
    """Train all algorithms and track progress at checkpoints.
    
    Args:
        env: Environment for evaluation
        dataset: Training dataset
        state_dim: State dimension
        action_dim: Action dimension
        num_steps: Total training steps
        device: Device to use
        eval_checkpoints: List of steps at which to evaluate
        
    Returns:
        Tuple of (progress_results, emd_history)
        - progress_results: Dict[algo_name -> Dict[step -> samples]]
        - emd_history: Dict[algo_name -> Dict[step -> emd_value]]
    """
    if eval_checkpoints is None:
        # Default: evaluate at 0%, 20%, 40%, 60%, 80%, 100% of training
        eval_checkpoints = [0, num_steps//5, 2*num_steps//5, 3*num_steps//5, 4*num_steps//5, num_steps]
    
    progress_results = {}
    emd_history = {}
    
    # Dummy state for generation
    state = np.zeros(state_dim, dtype=np.float32)
    
    # Prepare data tensors
    states_t = torch.FloatTensor(dataset["states"]).to(device)
    actions_t = torch.FloatTensor(dataset["actions"]).to(device)
    rewards_t = torch.FloatTensor(dataset["rewards"]).to(device)
    next_states_t = torch.FloatTensor(dataset["next_states"]).to(device)
    dones_t = torch.FloatTensor(dataset["dones"]).to(device)
    n_samples = len(states_t)
    batch_size = 256
    
    def evaluate_agent(agent, algo_name, use_pretrain_sample=False):
        """Evaluate agent and return samples and EMD.
        
        Args:
            agent: The agent to evaluate
            algo_name: Name of the algorithm (for logging)
            use_pretrain_sample: If True, use sample_action_pretrain() method
                               (for DPPO/ReinFlow before fine-tuning)
        """
        samples = []
        for _ in range(500):
            with torch.no_grad():
                # Use pretrain sampling for DPPO/ReinFlow (pure ODE/diffusion without RL noise)
                if use_pretrain_sample and hasattr(agent, 'sample_action_pretrain'):
                    action = agent.sample_action_pretrain(state)
                else:
                    action = agent.sample_action(state)
                    
                if isinstance(action, tuple):
                    action = action[0]
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if action.ndim > 1:
                    action = action[0]
                samples.append(action)
        samples = np.array(samples)
        samples_scaled = samples * env.scale * 1.5
        metrics = env.evaluate_samples(samples_scaled)
        return samples, metrics['wasserstein_approx']
    
    # ==================== 1. Diffusion Policy ====================
    print("\n1. Training Diffusion Policy with progress tracking...")
    algo_name = "Diffusion Policy"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = DiffusionPolicyAgent(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=[256, 256], num_diffusion_steps=20, device=device
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = evaluate_agent(agent, algo_name)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = {"states": states_t[idx], "actions": actions_t[idx]}
                agent.train_step(batch)
    except Exception as e:
        print(f"  Error: {e}")
    
    # ==================== 2. Flow Matching ====================
    print("\n2. Training Flow Matching with progress tracking...")
    algo_name = "Flow Matching"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = FlowMatchingPolicy(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=[256, 256], num_inference_steps=20, device=device
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = evaluate_agent(agent, algo_name)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = {"states": states_t[idx], "actions": actions_t[idx]}
                agent.train_step(batch)
    except Exception as e:
        print(f"  Error: {e}")
    
    # ==================== 3. Consistency Flow ====================
    print("\n3. Training Consistency Flow with progress tracking...")
    algo_name = "Consistency Flow"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = ConsistencyFlowPolicy(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=[256, 256], num_inference_steps=5,  # Use 5 steps for better quality
            flow_batch_ratio=0.7, consistency_batch_ratio=0.3,
            consistency_weight=1.0, device=device
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = evaluate_agent(agent, algo_name)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = {"states": states_t[idx], "actions": actions_t[idx]}
                agent.train_step(batch)
    except Exception as e:
        print(f"  Error: {e}")
    
    # ==================== 4. Reflected Flow ====================
    print("\n4. Training Reflected Flow with progress tracking...")
    algo_name = "Reflected Flow"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = ReflectedFlowPolicy(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=[256, 256], num_inference_steps=20,
            reflection_mode="hard", device=device
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = evaluate_agent(agent, algo_name)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = {"states": states_t[idx], "actions": actions_t[idx]}
                agent.train_step(batch)
    except Exception as e:
        print(f"  Error: {e}")
    
    # ==================== 5. Diffusion QL ====================
    print("\n5. Training Diffusion QL with progress tracking...")
    algo_name = "Diffusion QL"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = DiffusionDoubleQAgent(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=[256, 256], num_diffusion_steps=20, device=device
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = evaluate_agent(agent, algo_name)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = {
                    "states": states_t[idx], "actions": actions_t[idx],
                    "rewards": rewards_t[idx], "next_states": next_states_t[idx],
                    "dones": dones_t[idx]
                }
                agent.train_step(batch)
    except Exception as e:
        print(f"  Error: {e}")
    
    # ==================== 6. CPQL ====================
    print("\n6. Training CPQL with progress tracking...")
    algo_name = "CPQL"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = CPQLAgent(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=[256, 256], sigma_max=80.0, sigma_min=0.002,
            rho=7.0, device=device
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = evaluate_agent(agent, algo_name)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = {
                    "states": states_t[idx], "actions": actions_t[idx],
                    "rewards": rewards_t[idx], "next_states": next_states_t[idx],
                    "dones": dones_t[idx]
                }
                agent.train_step(batch)
    except Exception as e:
        print(f"  Error: {e}")
    
    # ==================== 7. DPPO ====================
    print("\n7. Training DPPO with progress tracking...")
    algo_name = "DPPO"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = DPPOAgent(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=[256, 256], num_diffusion_steps=10,
            ft_denoising_steps=5, device=device
        )
        
        # Enable gradients for pretraining
        for p in agent.actor.parameters():
            p.requires_grad = True
        dppo_optimizer = torch.optim.Adam(
            list(agent.actor.parameters()) + list(agent.actor_ft.parameters()), 
            lr=1e-4
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                # Use pretrain sampling (only base actor, full diffusion)
                samples, emd = evaluate_agent(agent, algo_name, use_pretrain_sample=True)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch_states = states_t[idx]
                batch_actions = actions_t[idx]
                
                # Diffusion BC loss
                t = torch.randint(0, agent.num_diffusion_steps, (batch_size,), device=device)
                noise = torch.randn_like(batch_actions)
                noisy_actions = agent.diffusion.q_sample(batch_actions, t, noise)
                t_normalized = t.float() / agent.num_diffusion_steps
                
                pred_noise = agent.actor(batch_states, noisy_actions, t_normalized)
                pred_noise_ft = agent.actor_ft(batch_states, noisy_actions, t_normalized)
                loss = F.mse_loss(pred_noise, noise) + F.mse_loss(pred_noise_ft, noise)
                
                dppo_optimizer.zero_grad()
                loss.backward()
                dppo_optimizer.step()
                
        # Freeze actor after pretraining
        for p in agent.actor.parameters():
            p.requires_grad = False
    except Exception as e:
        print(f"  Error: {e}")
    
    # ==================== 8. ReinFlow ====================
    print("\n8. Training ReinFlow with progress tracking...")
    algo_name = "ReinFlow"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = ReinFlowAgent(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=[256, 256], num_flow_steps=10,
            noise_scheduler_type="learn", device=device
        )
        
        # Create optimizer for pretrain
        pretrain_optimizer = torch.optim.Adam(
            agent.policy.velocity_net.parameters(), lr=1e-4
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                # Use pretrain sampling (pure ODE integration without noise network)
                samples, emd = evaluate_agent(agent, algo_name, use_pretrain_sample=True)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch_states = states_t[idx]
                batch_actions = actions_t[idx]
                
                # Flow matching loss
                x_0 = torch.randn_like(batch_actions)
                t = torch.rand(batch_size, device=device)
                t_expand = t.unsqueeze(-1)
                x_t = (1 - t_expand) * x_0 + t_expand * batch_actions
                target_v = batch_actions - x_0
                pred_v = agent.policy.velocity_net(batch_states, x_t, t)
                loss = F.mse_loss(pred_v, target_v)
                
                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()
    except Exception as e:
        print(f"  Error: {e}")
    
    return progress_results, emd_history


def run_progress_validation(
    distribution_type: str = "clusters",
    num_modes: int = 8,
    num_samples: int = 5000,
    num_train_steps: int = 2000,
    device: str = "cpu",
    output_dir: str = "./results",
    num_checkpoints: int = 6
):
    """Run validation with training progress visualization.
    
    Creates a figure showing how each algorithm learns over training steps,
    similar to the DDBM paper figure.
    """
    print(f"\n{'='*60}")
    print(f"Training Progress Validation on: {distribution_type} distribution")
    print(f"{'='*60}")
    
    # Create environment
    env = make_env(
        distribution_type,
        num_modes=num_modes,
        scale=2.0,
        noise_std=0.15
    )
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Create offline dataset
    print("\nCollecting offline dataset...")
    dataset = create_offline_dataset(env, num_samples=num_samples)
    print(f"Dataset size: {len(dataset['states'])} samples")
    
    # Define checkpoints
    checkpoints = [int(i * num_train_steps / (num_checkpoints - 1)) for i in range(num_checkpoints)]
    checkpoints = sorted(set(checkpoints))  # Remove duplicates and sort
    print(f"Evaluation checkpoints: {checkpoints}")
    
    # Train with progress tracking
    progress_results, emd_history = train_with_progress_tracking(
        env=env,
        dataset=dataset,
        state_dim=state_dim,
        action_dim=action_dim,
        num_steps=num_train_steps,
        device=device,
        eval_checkpoints=checkpoints
    )
    
    # Create visualization
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Progress visualization (like DDBM figure)
    progress_save_path = os.path.join(
        output_dir, 
        f"progress_{distribution_type}_{timestamp}.png"
    )
    visualize_training_progress(
        env=env,
        progress_results=progress_results,
        emd_history=emd_history,
        save_path=progress_save_path,
        title=f"Training Progress on {distribution_type.title()} Distribution"
    )

    
    # Print summary
    print(f"\n{'='*60}")
    print("Final Evaluation Metrics")
    print(f"{'='*60}")
    print(f"{'Algorithm':<20} {'Final EMD':>12}")
    print("-" * 34)
    
    for name in progress_results:
        final_step = max(emd_history[name].keys())
        final_emd = emd_history[name][final_step]
        print(f"{name:<20} {final_emd:>12.4f}")
    
    return progress_results, emd_history


def main():
    parser = argparse.ArgumentParser(description="Validate all diffusion/flow RL algorithms")
    parser.add_argument("--distribution", type=str, default="clusters",
                       choices=["clusters", "ring", "double_ring", "spiral", "moons", "grid"],
                       help="Type of target distribution")
    parser.add_argument("--num_modes", type=int, default=8, help="Number of modes")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--num_steps", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Run on all distribution types")
    parser.add_argument("--num_checkpoints", type=int, default=6,
                       help="Number of checkpoints for progress visualization")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print(f"Using device: {args.device}")
    
    if args.all:
        distributions = ["clusters", "ring", "double_ring", "spiral", "moons", "grid"]
        for dist in distributions:
            run_progress_validation(
                distribution_type=dist,
                num_modes=args.num_modes,
                num_samples=args.num_samples,
                num_train_steps=args.num_steps,
                device=args.device,
                output_dir=args.output_dir,
                num_checkpoints=args.num_checkpoints
            )
    else:
        run_progress_validation(
            distribution_type=args.distribution,
            num_modes=args.num_modes,
            num_samples=args.num_samples,
            num_train_steps=args.num_steps,
            device=args.device,
            output_dir=args.output_dir,
            num_checkpoints=args.num_checkpoints
        )

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
