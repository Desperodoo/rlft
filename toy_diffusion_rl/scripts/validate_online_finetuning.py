"""
Validate Online Fine-tuning for DPPO and ReinFlow

This script validates the online RL components of:
1. DPPO - Diffusion Policy Policy Optimization
2. ReinFlow - Flow Matching + Online RL Fine-tuning

The validation flow:
1. Pretrain on offline data (BC/Flow Matching)
2. Fine-tune online using environment rewards
3. Compare pretrain vs fine-tuned performance

Environment: MultimodalParticleEnv with reward-based learning
- State: dummy context (unconditional generation)
- Action: 2D point to generate
- Reward: Based on density/distance to target distribution
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
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from toy_diffusion_rl.envs import MultimodalParticleEnv, make_env
from toy_diffusion_rl.algorithms.dppo.agent import DPPOAgent
from toy_diffusion_rl.algorithms.reinflow.agent import ReinFlowAgent


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_agent(
    agent,
    env: MultimodalParticleEnv,
    num_samples: int = 500,
    use_pretrain_sample: bool = False
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Evaluate agent by generating samples and computing metrics.
    
    Args:
        agent: DPPO or ReinFlow agent
        env: Environment for evaluation metrics
        num_samples: Number of samples to generate
        use_pretrain_sample: If True, use pretrain sampling (no RL exploration noise)
        
    Returns:
        samples: Generated samples (num_samples, 2)
        metrics: Dictionary with evaluation metrics
    """
    state = np.zeros(env.observation_space.shape[0], dtype=np.float32)
    samples = []
    
    for _ in range(num_samples):
        with torch.no_grad():
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
    
    # Compute metrics
    eval_metrics = env.evaluate_samples(samples_scaled)
    
    return samples, eval_metrics


def collect_online_rewards(
    agent,
    env: MultimodalParticleEnv,
    num_episodes: int = 100
) -> float:
    """Collect average reward from environment interaction.
    
    Args:
        agent: DPPO or ReinFlow agent
        env: Environment
        num_episodes: Number of episodes to run
        
    Returns:
        Average reward
    """
    total_reward = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.sample_action(state)
            if isinstance(action, tuple):
                action = action[0]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            if action.ndim > 1:
                action = action[0]
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            state = next_state
            
        total_reward += episode_reward
        
    return total_reward / num_episodes


def visualize_online_progress(
    progress_data: Dict[str, Dict],
    save_path: str,
    title: str = "Online Fine-tuning Progress"
):
    """Visualize online fine-tuning progress.
    
    Args:
        progress_data: Dict with algorithm name -> {
            'pretrain_samples': samples before fine-tuning,
            'finetune_samples': list of samples at each checkpoint,
            'pretrain_emd': EMD before fine-tuning,
            'finetune_emd': list of EMD at each checkpoint,
            'rewards': list of average rewards at each checkpoint,
            'iterations': list of iteration numbers
        }
        save_path: Path to save figure
        title: Figure title
    """
    n_algos = len(progress_data)
    fig, axes = plt.subplots(n_algos, 4, figsize=(16, 4 * n_algos))
    
    if n_algos == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, (algo_name, data) in enumerate(progress_data.items()):
        # Column 1: Pretrain distribution
        ax = axes[row_idx, 0]
        samples = data.get('pretrain_samples', np.zeros((10, 2)))
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5, c='blue')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title(f'Pretrain (EMD={data.get("pretrain_emd", 0):.4f})')
        ax.set_ylabel(algo_name, fontsize=12)
        
        # Column 2: Final fine-tuned distribution
        ax = axes[row_idx, 1]
        finetune_samples = data.get('finetune_samples', [np.zeros((10, 2))])
        final_samples = finetune_samples[-1] if len(finetune_samples) > 0 else np.zeros((10, 2))
        ax.scatter(final_samples[:, 0], final_samples[:, 1], alpha=0.5, s=5, c='red')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        finetune_emd = data.get('finetune_emd', [0])
        final_emd = finetune_emd[-1] if len(finetune_emd) > 0 else 0
        ax.set_title(f'Fine-tuned (EMD={final_emd:.4f})')
        
        # Column 3: EMD curve
        ax = axes[row_idx, 2]
        iterations = data.get('iterations', [0])
        emd_values = [data.get('pretrain_emd', 0)] + list(data.get('finetune_emd', []))
        iter_values = [0] + list(iterations)
        ax.plot(iter_values, emd_values, 'b-o', linewidth=2, markersize=4)
        ax.axhline(y=data.get('pretrain_emd', 0), color='gray', linestyle='--', alpha=0.5, label='Pretrain')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('EMD ↓')
        ax.set_title('EMD over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Column 4: Reward curve
        ax = axes[row_idx, 3]
        rewards = data.get('rewards', [0])
        iter_values = iterations if len(iterations) > 0 else [0]
        ax.plot(iter_values, rewards, 'g-o', linewidth=2, markersize=4)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Avg Reward ↑')
        ax.set_title('Reward over Training')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved progress visualization to {save_path}")


def train_dppo_online(
    env: MultimodalParticleEnv,
    dataset: Dict[str, np.ndarray],
    pretrain_steps: int = 5000,
    online_iterations: int = 50,
    rollout_steps: int = 512,
    device: str = "cpu",
    eval_interval: int = 5
) -> Tuple[DPPOAgent, Dict]:
    """Train DPPO with pretrain + online fine-tuning.
    
    Args:
        env: Environment for online interaction
        dataset: Offline dataset for pretraining
        pretrain_steps: Number of BC pretraining steps
        online_iterations: Number of PPO update iterations
        rollout_steps: Steps per rollout
        device: Compute device
        eval_interval: Evaluate every N iterations
        
    Returns:
        agent: Trained agent
        progress: Training progress data
    """
    print("\n" + "="*60)
    print("Training DPPO (Pretrain + Online Fine-tuning)")
    print("="*60)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create agent
    agent = DPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        num_diffusion_steps=10,
        ft_denoising_steps=5,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ppo_epochs=5,
        batch_size=64,
        device=device
    )
    
    # Phase 1: Pretrain with BC
    print("\nPhase 1: Behavior Cloning Pretrain...")
    metrics = agent.pretrain_bc(dataset, num_steps=pretrain_steps, batch_size=256)
    print(f"  BC Loss: {metrics['bc_loss']:.4f}")
    
    # Evaluate pretrain
    pretrain_samples, pretrain_metrics = evaluate_agent(
        agent, env, num_samples=500, use_pretrain_sample=True
    )
    print(f"  Pretrain EMD: {pretrain_metrics['wasserstein_approx']:.4f}")
    print(f"  Pretrain Mode Coverage: {pretrain_metrics['mode_coverage']:.2%}")
    
    # Progress tracking
    progress = {
        'pretrain_samples': pretrain_samples,
        'pretrain_emd': pretrain_metrics['wasserstein_approx'],
        'finetune_samples': [],
        'finetune_emd': [],
        'rewards': [],
        'iterations': [],
        'policy_losses': [],
        'value_losses': []
    }
    
    # Phase 2: Online Fine-tuning with PPO
    print("\nPhase 2: Online Fine-tuning with PPO...")
    
    for iteration in tqdm(range(1, online_iterations + 1), desc="Online Training"):
        # Collect rollout
        buffer = agent.collect_rollout(env, rollout_steps=rollout_steps)
        
        # Update policy
        update_metrics = agent.update(buffer)
        
        progress['policy_losses'].append(update_metrics['policy_loss'])
        progress['value_losses'].append(update_metrics['value_loss'])
        
        # Evaluate periodically
        if iteration % eval_interval == 0:
            # Generate samples for visualization
            samples, eval_metrics = evaluate_agent(agent, env, num_samples=500)
            
            # Compute average reward
            avg_reward = collect_online_rewards(agent, env, num_episodes=50)
            
            progress['finetune_samples'].append(samples)
            progress['finetune_emd'].append(eval_metrics['wasserstein_approx'])
            progress['rewards'].append(avg_reward)
            progress['iterations'].append(iteration)
            
            print(f"\n  Iter {iteration}: EMD={eval_metrics['wasserstein_approx']:.4f}, "
                  f"Reward={avg_reward:.2f}, "
                  f"Policy Loss={update_metrics['policy_loss']:.4f}")
    
    return agent, progress


def train_reinflow_online(
    env: MultimodalParticleEnv,
    dataset: Dict[str, np.ndarray],
    pretrain_steps: int = 5000,
    online_iterations: int = 50,
    rollout_steps: int = 512,
    device: str = "cpu",
    eval_interval: int = 5
) -> Tuple[ReinFlowAgent, Dict]:
    """Train ReinFlow with pretrain + online fine-tuning.
    
    Args:
        env: Environment for online interaction
        dataset: Offline dataset for pretraining
        pretrain_steps: Number of flow matching pretraining steps
        online_iterations: Number of PPO update iterations
        rollout_steps: Steps per rollout
        device: Compute device
        eval_interval: Evaluate every N iterations
        
    Returns:
        agent: Trained agent
        progress: Training progress data
    """
    print("\n" + "="*60)
    print("Training ReinFlow (Pretrain + Online Fine-tuning)")
    print("="*60)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create agent
    agent = ReinFlowAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        num_flow_steps=10,
        noise_scheduler_type="learn",
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ppo_epochs=5,
        batch_size=64,
        device=device
    )
    
    # Phase 1: Pretrain with Flow Matching
    print("\nPhase 1: Flow Matching Pretrain...")
    metrics = agent.offline_pretrain(dataset, num_steps=pretrain_steps, batch_size=256)
    print(f"  Pretrain Loss: {metrics['pretrain_loss']:.4f}")
    
    # Evaluate pretrain
    pretrain_samples, pretrain_metrics = evaluate_agent(
        agent, env, num_samples=500, use_pretrain_sample=True
    )
    print(f"  Pretrain EMD: {pretrain_metrics['wasserstein_approx']:.4f}")
    print(f"  Pretrain Mode Coverage: {pretrain_metrics['mode_coverage']:.2%}")
    
    # Progress tracking
    progress = {
        'pretrain_samples': pretrain_samples,
        'pretrain_emd': pretrain_metrics['wasserstein_approx'],
        'finetune_samples': [],
        'finetune_emd': [],
        'rewards': [],
        'iterations': [],
        'policy_losses': [],
        'value_losses': []
    }
    
    # Phase 2: Online Fine-tuning with PPO
    print("\nPhase 2: Online Fine-tuning with PPO...")
    
    for iteration in tqdm(range(1, online_iterations + 1), desc="Online Training"):
        # Collect rollout
        buffer = agent.collect_rollout(env, rollout_steps=rollout_steps)
        
        # Update policy
        update_metrics = agent.online_update(buffer)
        
        progress['policy_losses'].append(update_metrics['policy_loss'])
        progress['value_losses'].append(update_metrics['value_loss'])
        
        # Evaluate periodically
        if iteration % eval_interval == 0:
            # Generate samples for visualization (use deterministic mode after fine-tuning)
            samples, eval_metrics = evaluate_agent(agent, env, num_samples=500)
            
            # Compute average reward
            avg_reward = collect_online_rewards(agent, env, num_episodes=50)
            
            progress['finetune_samples'].append(samples)
            progress['finetune_emd'].append(eval_metrics['wasserstein_approx'])
            progress['rewards'].append(avg_reward)
            progress['iterations'].append(iteration)
            
            print(f"\n  Iter {iteration}: EMD={eval_metrics['wasserstein_approx']:.4f}, "
                  f"Reward={avg_reward:.2f}, "
                  f"Policy Loss={update_metrics['policy_loss']:.4f}")
    
    return agent, progress


def run_online_validation(
    distribution_type: str = "ring",
    num_modes: int = 8,
    num_samples: int = 5000,
    pretrain_steps: int = 5000,
    online_iterations: int = 50,
    rollout_steps: int = 512,
    device: str = "cpu",
    output_dir: str = "./results",
    seed: int = 42
):
    """Run online fine-tuning validation for DPPO and ReinFlow.
    
    Args:
        distribution_type: Type of target distribution
        num_modes: Number of modes
        num_samples: Number of offline samples
        pretrain_steps: Pretraining steps
        online_iterations: Online PPO iterations
        rollout_steps: Steps per rollout
        device: Compute device
        output_dir: Output directory for results
        seed: Random seed
    """
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Online Fine-tuning Validation")
    print(f"Distribution: {distribution_type}, Modes: {num_modes}")
    print(f"{'='*60}")
    
    # Create environment with reward for online learning
    # Use 'density' reward type for meaningful RL signal
    env = MultimodalParticleEnv(
        distribution_type=distribution_type,
        num_modes=num_modes,
        scale=2.0,
        noise_std=0.15,
        reward_type="density",  # Reward based on log-density
        max_episode_steps=1  # Single-step episode for generation
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Reward type: density (log probability under target distribution)")
    
    # Collect offline dataset
    print("\nCollecting offline dataset...")
    dataset = env.collect_offline_dataset(num_samples=num_samples)
    print(f"Dataset size: {len(dataset['states'])} samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Store results
    all_progress = {}
    
    # ==================== Train DPPO ====================
    dppo_agent, dppo_progress = train_dppo_online(
        env=env,
        dataset=dataset,
        pretrain_steps=pretrain_steps,
        online_iterations=online_iterations,
        rollout_steps=rollout_steps,
        device=device,
        eval_interval=5
    )
    all_progress['DPPO'] = dppo_progress
    
    # ==================== Train ReinFlow ====================
    reinflow_agent, reinflow_progress = train_reinflow_online(
        env=env,
        dataset=dataset,
        pretrain_steps=pretrain_steps,
        online_iterations=online_iterations,
        rollout_steps=rollout_steps,
        device=device,
        eval_interval=5
    )
    all_progress['ReinFlow'] = reinflow_progress
    
    # ==================== Visualization ====================
    save_path = os.path.join(output_dir, f"online_finetuning_{distribution_type}_{timestamp}.png")
    visualize_online_progress(
        all_progress,
        save_path=save_path,
        title=f"Online Fine-tuning on {distribution_type.title()} Distribution"
    )
    
    # ==================== Final Summary ====================
    print("\n" + "="*60)
    print("Final Summary")
    print("="*60)
    
    print(f"\n{'Algorithm':<15} {'Pretrain EMD':>15} {'Final EMD':>15} {'EMD Change':>15}")
    print("-" * 62)
    
    for algo_name, progress in all_progress.items():
        pretrain_emd = progress['pretrain_emd']
        final_emd = progress['finetune_emd'][-1] if progress['finetune_emd'] else pretrain_emd
        change = final_emd - pretrain_emd
        change_str = f"{change:+.4f}" if change != 0 else "0.0000"
        print(f"{algo_name:<15} {pretrain_emd:>15.4f} {final_emd:>15.4f} {change_str:>15}")
    
    # Save detailed results
    results_path = os.path.join(output_dir, f"online_results_{distribution_type}_{timestamp}.txt")
    with open(results_path, 'w') as f:
        f.write(f"Online Fine-tuning Results\n")
        f.write(f"Distribution: {distribution_type}\n")
        f.write(f"Num Modes: {num_modes}\n")
        f.write(f"Pretrain Steps: {pretrain_steps}\n")
        f.write(f"Online Iterations: {online_iterations}\n")
        f.write(f"Rollout Steps: {rollout_steps}\n")
        f.write(f"\n{'='*60}\n\n")
        
        for algo_name, progress in all_progress.items():
            f.write(f"{algo_name}:\n")
            f.write(f"  Pretrain EMD: {progress['pretrain_emd']:.4f}\n")
            if progress['finetune_emd']:
                f.write(f"  Final EMD: {progress['finetune_emd'][-1]:.4f}\n")
                f.write(f"  EMD History: {progress['finetune_emd']}\n")
                f.write(f"  Reward History: {progress['rewards']}\n")
            f.write("\n")
    
    print(f"\nResults saved to {results_path}")
    
    return all_progress


def main():
    parser = argparse.ArgumentParser(description="Validate online fine-tuning for DPPO and ReinFlow")
    parser.add_argument("--distribution", type=str, default="ring",
                       choices=["clusters", "ring", "double_ring", "spiral", "moons", "grid"],
                       help="Type of target distribution")
    parser.add_argument("--num_modes", type=int, default=8, help="Number of modes")
    parser.add_argument("--num_samples", type=int, default=5000, help="Offline dataset size")
    parser.add_argument("--pretrain_steps", type=int, default=5000, help="BC pretrain steps")
    parser.add_argument("--online_iterations", type=int, default=50, help="Online PPO iterations")
    parser.add_argument("--rollout_steps", type=int, default=512, help="Steps per rollout")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    run_online_validation(
        distribution_type=args.distribution,
        num_modes=args.num_modes,
        num_samples=args.num_samples,
        pretrain_steps=args.pretrain_steps,
        online_iterations=args.online_iterations,
        rollout_steps=args.rollout_steps,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print("\nOnline fine-tuning validation complete!")


if __name__ == "__main__":
    main()
