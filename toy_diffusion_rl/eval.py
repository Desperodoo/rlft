#!/usr/bin/env python3
"""
Evaluation script for Toy Diffusion RL algorithms.

Usage:
    python eval.py --checkpoint results/experiment/best_model.pt --algorithm diffusion_policy --env point_mass_2d
    python eval.py --checkpoint results/experiment/best_model.pt --config results/experiment/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from envs import make_env
from common import set_seed, get_device
from algorithms import (
    DiffusionPolicyAgent,
    FlowMatchingPolicy,
    ReflectedFlowPolicy,
    ConsistencyFlowPolicy,
    DiffusionDoubleQAgent,
    CPQLAgent,
    DPPOAgent,
    ReinFlowAgent
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_agent(algorithm: str, env, config: dict, device: str):
    """Create agent based on algorithm name."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    hidden_dims = config.get("network", {}).get("hidden_dims", [256, 256])
    
    if algorithm == "diffusion_policy":
        diffusion_config = config.get("diffusion", {})
        return DiffusionPolicyAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            num_diffusion_steps=diffusion_config.get("num_diffusion_steps", 100),
            device=device
        )
    
    elif algorithm == "flow_matching":
        flow_config = config.get("flow", {})
        variant = flow_config.get("variant", "vanilla")
        
        if variant == "reflected":
            return ReflectedFlowPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                device=device
            )
        elif variant == "consistency":
            return ConsistencyFlowPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                device=device
            )
        else:
            return FlowMatchingPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                device=device
            )
    
    elif algorithm == "diffusion_double_q":
        return DiffusionDoubleQAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            device=device
        )
    
    elif algorithm == "cpql":
        return CPQLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            device=device
        )
    
    elif algorithm == "dppo":
        return DPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            device=device
        )
    
    elif algorithm == "reinflow":
        return ReinFlowAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            device=device
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate_episode(agent, env, render: bool = False):
    """Run one evaluation episode and collect trajectory data."""
    states = []
    actions = []
    rewards = []
    
    state, _ = env.reset()
    done = False
    
    while not done:
        states.append(state.copy())
        
        result = agent.sample_action(state)
        action = result[0] if isinstance(result, tuple) else result
        actions.append(action.copy() if isinstance(action, np.ndarray) else np.array([action]))
        
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        
        if render:
            env.render()
    
    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "total_return": sum(rewards)
    }


def evaluate(agent, env, num_episodes: int = 100, verbose: bool = True):
    """Evaluate agent over multiple episodes."""
    results = []
    
    for i in range(num_episodes):
        episode_result = evaluate_episode(agent, env)
        results.append(episode_result)
        
        if verbose and (i + 1) % 10 == 0:
            avg_return = np.mean([r["total_return"] for r in results])
            print(f"Episode {i+1}/{num_episodes}, Avg Return: {avg_return:.2f}")
    
    returns = [r["total_return"] for r in results]
    
    stats = {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "min_return": np.min(returns),
        "max_return": np.max(returns),
        "median_return": np.median(returns)
    }
    
    return results, stats


def plot_results(results, save_path: str = None):
    """Plot evaluation results."""
    returns = [r["total_return"] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Return distribution
    axes[0].hist(returns, bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(returns), color='r', linestyle='--', label=f'Mean: {np.mean(returns):.2f}')
    axes[0].set_xlabel('Episode Return')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Return Distribution')
    axes[0].legend()
    
    # Returns over episodes
    axes[1].plot(returns, alpha=0.7)
    axes[1].axhline(np.mean(returns), color='r', linestyle='--', label='Mean')
    axes[1].fill_between(range(len(returns)), 
                         np.mean(returns) - np.std(returns),
                         np.mean(returns) + np.std(returns),
                         alpha=0.2, color='r')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Return')
    axes[1].set_title('Returns per Episode')
    axes[1].legend()
    
    # Episode lengths
    lengths = [len(r["rewards"]) for r in results]
    axes[2].hist(lengths, bins=20, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Episode Length')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Episode Length Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_trajectory(result, env_name: str, save_path: str = None):
    """Plot a sample trajectory (for 2D environments)."""
    if env_name != "point_mass_2d":
        print("Trajectory plotting only supported for point_mass_2d")
        return
    
    states = result["states"]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot trajectory
    ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=1.5, alpha=0.7, label='Trajectory')
    ax.scatter(states[0, 0], states[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(states[-1, 0], states[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
    ax.scatter(0, 0, c='gold', s=150, marker='*', label='Goal', zorder=5)
    
    # Plot bounds
    ax.axhline(y=-5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Trajectory (Return: {result["total_return"]:.2f})')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Toy Diffusion RL algorithms")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--algorithm", type=str, default=None,
                        help="Algorithm name (inferred from config if not provided)")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment name (inferred from config if not provided)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save plots to checkpoint directory")
    parser.add_argument("--plot_trajectory", action="store_true",
                        help="Plot sample trajectory")
    
    args = parser.parse_args()
    
    # Try to find config in checkpoint directory
    checkpoint_dir = os.path.dirname(args.checkpoint)
    config_path = args.config or os.path.join(checkpoint_dir, "config.yaml")
    
    if os.path.exists(config_path):
        config = load_config(config_path)
        algorithm = args.algorithm or config.get("algorithm")
        env_name = args.env or config.get("env", {}).get("name")
    else:
        config = {}
        algorithm = args.algorithm
        env_name = args.env
    
    if not algorithm or not env_name:
        raise ValueError("Algorithm and environment must be specified (via args or config)")
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create environment and agent
    env = make_env(env_name)
    agent = create_agent(algorithm, env, config, str(device))
    
    # Load checkpoint
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Evaluate
    print(f"\nEvaluating {algorithm} on {env_name}...")
    print(f"Running {args.num_episodes} episodes...")
    
    results, stats = evaluate(agent, env, args.num_episodes)
    
    # Print statistics
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"  Mean Return:   {stats['mean_return']:.2f} ± {stats['std_return']:.2f}")
    print(f"  Median Return: {stats['median_return']:.2f}")
    print(f"  Min Return:    {stats['min_return']:.2f}")
    print(f"  Max Return:    {stats['max_return']:.2f}")
    print("="*50)
    
    # Save/show plots
    if args.save_plots:
        plot_save_path = os.path.join(checkpoint_dir, "eval_results.png")
        plot_results(results, save_path=plot_save_path)
        
        if args.plot_trajectory:
            # Plot best trajectory
            best_idx = np.argmax([r["total_return"] for r in results])
            traj_save_path = os.path.join(checkpoint_dir, "best_trajectory.png")
            plot_trajectory(results[best_idx], env_name, save_path=traj_save_path)
    else:
        plot_results(results)
        
        if args.plot_trajectory:
            best_idx = np.argmax([r["total_return"] for r in results])
            plot_trajectory(results[best_idx], env_name)
    
    # Save statistics
    stats_path = os.path.join(checkpoint_dir, "eval_stats.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)
    print(f"\nStatistics saved to {stats_path}")


if __name__ == "__main__":
    main()
