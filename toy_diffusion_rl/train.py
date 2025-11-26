#!/usr/bin/env python3
"""
Training script for Toy Diffusion RL algorithms.

Usage:
    python train.py --algorithm diffusion_policy --env point_mass_2d --seed 42
    python train.py --config configs/diffusion_policy_pendulum.yaml
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from envs import make_env
from common import ReplayBuffer, set_seed, get_device
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
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if "_base_" in config:
        base_path = Path(config_path).parent / config["_base_"]
        base_config = load_config(str(base_path))
        # Merge configs (current overrides base)
        base_config = deep_merge(base_config, config)
        del base_config["_base_"]
        return base_config
    
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def create_agent(algorithm: str, env, config: dict, device: str):
    """Create agent based on algorithm name."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    hidden_dims = config.get("network", {}).get("hidden_dims", [256, 256])
    learning_rate = config.get("optimizer", {}).get("learning_rate", 3e-4)
    
    if algorithm == "diffusion_policy":
        diffusion_config = config.get("diffusion", {})
        return DiffusionPolicyAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            num_diffusion_steps=diffusion_config.get("num_diffusion_steps", 100),
            noise_schedule=diffusion_config.get("noise_schedule", "linear"),
            learning_rate=learning_rate,
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
                learning_rate=learning_rate,
                num_inference_steps=flow_config.get("num_inference_steps", 10),
                device=device
            )
        elif variant == "consistency":
            return ConsistencyFlowPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                learning_rate=learning_rate,
                device=device
            )
        else:  # vanilla
            return FlowMatchingPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                learning_rate=learning_rate,
                num_inference_steps=flow_config.get("num_inference_steps", 10),
                device=device
            )
    
    elif algorithm == "diffusion_double_q":
        diffusion_config = config.get("diffusion", {})
        rl_config = config.get("rl", {})
        return DiffusionDoubleQAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            num_diffusion_steps=diffusion_config.get("num_diffusion_steps", 10),
            gamma=rl_config.get("gamma", 0.99),
            tau=rl_config.get("tau", 0.005),
            alpha=rl_config.get("alpha", 1.0),
            device=device
        )
    
    elif algorithm == "cpql":
        rl_config = config.get("rl", {})
        consistency_config = config.get("consistency", {})
        return CPQLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            gamma=rl_config.get("gamma", 0.99),
            tau=rl_config.get("tau", 0.005),
            alpha=rl_config.get("alpha", 0.1),
            device=device
        )
    
    elif algorithm == "dppo":
        diffusion_config = config.get("diffusion", {})
        ppo_config = config.get("ppo", {})
        return DPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            num_diffusion_steps=diffusion_config.get("num_diffusion_steps", 5),
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            clip_ratio=ppo_config.get("clip_ratio", 0.2),
            device=device
        )
    
    elif algorithm == "reinflow":
        flow_config = config.get("flow", {})
        ppo_config = config.get("ppo", {})
        return ReinFlowAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            num_flow_steps=flow_config.get("num_flow_steps", 10),
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            clip_ratio=ppo_config.get("clip_ratio", 0.2),
            device=device
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate(agent, env, num_episodes: int = 10) -> float:
    """Evaluate agent on environment."""
    returns = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            if hasattr(agent, "sample_action"):
                result = agent.sample_action(state)
                action = result[0] if isinstance(result, tuple) else result
            else:
                action = agent.sample_action(state)
                
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
            
        returns.append(episode_return)
        
    return np.mean(returns)


def train_offline(agent, env, buffer, config: dict, save_dir: str):
    """Training loop for offline algorithms (Diffusion Policy, Flow Matching, etc.)."""
    training_config = config.get("training", {})
    total_steps = training_config.get("total_steps", 20000)
    batch_size = training_config.get("batch_size", 256)
    eval_interval = training_config.get("eval_interval", 1000)
    eval_episodes = training_config.get("eval_episodes", 10)
    save_interval = training_config.get("save_interval", 10000)
    
    # Initialize EMA if available
    if hasattr(agent, "_init_ema"):
        agent._init_ema()
    
    history = {"loss": [], "eval_return": [], "eval_step": []}
    best_return = float("-inf")
    
    pbar = tqdm(range(total_steps), desc="Training")
    
    for step in pbar:
        batch = buffer.sample(batch_size)
        metrics = agent.train_step(batch)
        
        history["loss"].append(metrics.get("loss", 0))
        
        # Evaluation
        if (step + 1) % eval_interval == 0:
            eval_return = evaluate(agent, env, eval_episodes)
            history["eval_return"].append(eval_return)
            history["eval_step"].append(step + 1)
            
            if eval_return > best_return:
                best_return = eval_return
                agent.save(os.path.join(save_dir, "best_model.pt"))
                
            pbar.set_postfix({
                "loss": f"{metrics.get('loss', 0):.4f}",
                "eval": f"{eval_return:.2f}",
                "best": f"{best_return:.2f}"
            })
            
        # Save checkpoint
        if (step + 1) % save_interval == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_{step+1}.pt"))
            
    # Final save
    agent.save(os.path.join(save_dir, "final_model.pt"))
    
    return history


def train_online(agent, env, config: dict, save_dir: str):
    """Training loop for online algorithms (DPPO, ReinFlow)."""
    training_config = config.get("training", {})
    total_iterations = training_config.get("total_iterations", 100)
    rollout_steps = training_config.get("rollout_steps", 2048)
    eval_interval = training_config.get("eval_interval", 10)
    
    # Optional pretraining
    pretrain_config = config.get("pretrain", {})
    if pretrain_config.get("enabled", False):
        print("Pretraining with behavior cloning...")
        expert_data = env.collect_expert_dataset(
            num_episodes=pretrain_config.get("expert_episodes", 50),
            noise_std=pretrain_config.get("expert_noise", 0.1)
        )
        
        if hasattr(agent, "pretrain_bc"):
            bc_metrics = agent.pretrain_bc(
                expert_data, 
                num_steps=pretrain_config.get("num_steps", 2000)
            )
            print(f"Pretrain BC loss: {bc_metrics.get('bc_loss', 0):.4f}")
        elif hasattr(agent, "offline_pretrain"):
            pretrain_metrics = agent.offline_pretrain(
                expert_data,
                num_steps=pretrain_config.get("num_steps", 5000)
            )
            print(f"Pretrain loss: {pretrain_metrics.get('pretrain_loss', 0):.4f}")
    
    history = {"policy_loss": [], "value_loss": [], "eval_return": [], "eval_step": []}
    best_return = float("-inf")
    
    pbar = tqdm(range(total_iterations), desc="Training Online")
    
    for iteration in pbar:
        # Collect rollout
        buffer = agent.collect_rollout(env, rollout_steps)
        
        # Update
        if hasattr(agent, "update"):
            metrics = agent.update(buffer)
        elif hasattr(agent, "online_update"):
            metrics = agent.online_update(buffer)
        else:
            raise ValueError("Agent must have update or online_update method")
            
        history["policy_loss"].append(metrics.get("policy_loss", 0))
        history["value_loss"].append(metrics.get("value_loss", 0))
        
        # Evaluation
        if (iteration + 1) % eval_interval == 0:
            eval_return = evaluate(agent, env, 10)
            history["eval_return"].append(eval_return)
            history["eval_step"].append(iteration + 1)
            
            if eval_return > best_return:
                best_return = eval_return
                agent.save(os.path.join(save_dir, "best_model.pt"))
                
            pbar.set_postfix({
                "policy_loss": f"{metrics.get('policy_loss', 0):.4f}",
                "eval": f"{eval_return:.2f}",
                "best": f"{best_return:.2f}"
            })
    
    # Final save
    agent.save(os.path.join(save_dir, "final_model.pt"))
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Toy Diffusion RL algorithms")
    parser.add_argument("--algorithm", type=str, default="diffusion_policy",
                        choices=["diffusion_policy", "flow_matching", "diffusion_double_q",
                                 "cpql", "dppo", "reinflow"],
                        help="Algorithm to train")
    parser.add_argument("--env", type=str, default="point_mass_2d",
                        choices=["point_mass_2d", "pendulum"],
                        help="Environment to train on")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (overrides algorithm/env)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["cpu", "cuda", "auto"],
                        help="Device to use")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        algorithm = config.get("algorithm", args.algorithm)
        env_name = config.get("env", {}).get("name", args.env)
    else:
        config = {}
        algorithm = args.algorithm
        env_name = args.env
    
    # Setup
    seed = config.get("training", {}).get("seed", args.seed)
    set_seed(seed)
    
    device = get_device(args.device if args.device != "auto" else None)
    print(f"Using device: {device}")
    
    # Create save directory
    exp_name = args.exp_name or f"{algorithm}_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Create environment
    env = make_env(env_name)
    print(f"Environment: {env_name}")
    print(f"  State dim: {env.observation_space.shape[0]}")
    print(f"  Action dim: {env.action_space.shape[0]}")
    
    # Create agent
    agent = create_agent(algorithm, env, config, str(device))
    print(f"Algorithm: {algorithm}")
    
    # Train based on algorithm type
    online_algorithms = ["dppo", "reinflow"]
    
    if algorithm in online_algorithms:
        # Online RL training
        history = train_online(agent, env, config, save_dir)
    else:
        # Offline / BC training
        # Collect expert data
        expert_config = config.get("expert_data", {})
        print("Collecting expert demonstrations...")
        expert_data = env.collect_expert_dataset(
            num_episodes=expert_config.get("num_episodes", 100),
            noise_std=expert_config.get("noise_std", 0.1)
        )
        print(f"  Collected {len(expert_data['states'])} transitions")
        
        # Create buffer
        buffer = ReplayBuffer(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=str(device)
        )
        buffer.load_dataset(expert_data)
        
        history = train_offline(agent, env, buffer, config, save_dir)
    
    # Save history
    np.savez(os.path.join(save_dir, "history.npz"), **history)
    
    print(f"\nTraining complete! Results saved to {save_dir}")
    
    # Final evaluation
    final_return = evaluate(agent, env, 20)
    print(f"Final evaluation return: {final_return:.2f}")


if __name__ == "__main__":
    main()
