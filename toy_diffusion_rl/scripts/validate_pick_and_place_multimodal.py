#!/usr/bin/env python3
"""
Validate All Diffusion/Flow RL Algorithms on Pick-and-Place Task (Multimodal)

This script tests all unified algorithm implementations on the Fetch Pick-and-Place
robotics task with state_image (multimodal) observation mode.

Algorithms:
1. Diffusion Policy (BC baseline)
2. Flow Matching Policy (BC baseline)
3. Consistency Flow Policy (single-step generation)
4. Reflected Flow Policy (bounded action spaces)
5. Diffusion Double Q (Offline RL)
6. CPQL (Offline RL with Consistency Models)
7. DPPO (Online RL fine-tuning)
8. ReinFlow (Online RL fine-tuning)

Observation modes:
- "state": State vector only (dim=25)
- "image": Image observation only (128x128x3)
- "state_image": Both state and image (multimodal)

Evaluation:
- Success Rate: Percentage of episodes where object reaches goal
- Average Reward: Mean episode reward
"""

import os
import sys
import argparse
import copy
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
workspace_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, workspace_dir)

# Import environment
from toy_diffusion_rl.envs.pick_and_place import make_pick_and_place_env

# Import agents
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


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    """Load HDF5 dataset for Pick-and-Place.
    
    Args:
        path: Path to HDF5 file
        
    Returns:
        Dictionary with:
            - obs: (N, 25) state observations
            - images: (N, 128, 128, 3) RGB images (if available)
            - actions: (N, 4) actions
            - rewards: (N,) rewards
            - next_obs: (N, 25) next states
            - next_images: (N, 128, 128, 3) next images (if available)
            - dones: (N,) done flags
    """
    print(f"Loading dataset from {path}...")
    
    with h5py.File(path, 'r') as f:
        # Print metadata
        print(f"  Metadata:")
        for key in f.attrs:
            print(f"    {key}: {f.attrs[key]}")
        
        dataset = {}
        
        # Load required fields
        if "obs" in f:
            dataset["obs"] = f["obs"][:]
        if "next_obs" in f:
            dataset["next_obs"] = f["next_obs"][:]
        if "images" in f:
            dataset["images"] = f["images"][:]
        if "next_images" in f:
            dataset["next_images"] = f["next_images"][:]
        
        dataset["actions"] = f["actions"][:]
        dataset["rewards"] = f["rewards"][:]
        dataset["dones"] = f["dones"][:]
        
        print(f"  Dataset size: {len(dataset['actions'])} transitions")
        print(f"  Action shape: {dataset['actions'].shape}")
        if "obs" in dataset:
            print(f"  State shape: {dataset['obs'].shape}")
        if "images" in dataset:
            print(f"  Image shape: {dataset['images'].shape}")
    
    return dataset


def prepare_batch(
    dataset: Dict[str, np.ndarray],
    indices: np.ndarray,
    obs_mode: str,
    device: str
) -> Dict[str, torch.Tensor]:
    """Prepare a batch for training.
    
    Args:
        dataset: Full dataset dict
        indices: Batch indices
        obs_mode: "state", "image", or "state_image"
        device: Target device
        
    Returns:
        Batch dictionary with tensors
    """
    batch = {}
    
    # Actions (always present)
    batch["actions"] = torch.FloatTensor(dataset["actions"][indices]).to(device)
    batch["rewards"] = torch.FloatTensor(dataset["rewards"][indices]).to(device)
    batch["dones"] = torch.FloatTensor(dataset["dones"][indices]).to(device)
    
    if obs_mode == "state":
        batch["states"] = torch.FloatTensor(dataset["obs"][indices]).to(device)
        if "next_obs" in dataset:
            batch["next_states"] = torch.FloatTensor(dataset["next_obs"][indices]).to(device)
    
    elif obs_mode == "image":
        # Images: (N, H, W, C) -> (N, C, H, W), normalize to [0, 1]
        images = dataset["images"][indices].astype(np.float32) / 255.0
        batch["images"] = torch.FloatTensor(images).permute(0, 3, 1, 2).contiguous().to(device)
        
        if "next_images" in dataset:
            next_images = dataset["next_images"][indices].astype(np.float32) / 255.0
            batch["next_images"] = torch.FloatTensor(next_images).permute(0, 3, 1, 2).contiguous().to(device)
    
    else:  # state_image
        batch["states"] = torch.FloatTensor(dataset["obs"][indices]).to(device)
        
        images = dataset["images"][indices].astype(np.float32) / 255.0
        batch["images"] = torch.FloatTensor(images).permute(0, 3, 1, 2).contiguous().to(device)
        
        if "next_obs" in dataset:
            batch["next_states"] = torch.FloatTensor(dataset["next_obs"][indices]).to(device)
        if "next_images" in dataset:
            next_images = dataset["next_images"][indices].astype(np.float32) / 255.0
            batch["next_images"] = torch.FloatTensor(next_images).permute(0, 3, 1, 2).contiguous().to(device)
    
    return batch


def create_obs_for_agent(
    state: Optional[np.ndarray],
    image: Optional[np.ndarray],
    obs_mode: str
):
    """Create observation in the format expected by agent.sample_action().
    
    Args:
        state: (state_dim,) state array or None
        image: (H, W, C) image array or None
        obs_mode: "state", "image", or "state_image"
        
    Returns:
        Observation in agent-expected format
    """
    if obs_mode == "state":
        return state
    elif obs_mode == "image":
        return {"image": image}
    else:  # state_image
        return {"state": state, "image": image}


def evaluate_agent(
    agent,
    env,
    obs_mode: str,
    num_episodes: int = 10,
    max_steps: int = 50,
    use_pretrain_sample: bool = False,
    verbose: bool = False
) -> Dict[str, float]:
    """Evaluate agent on environment.
    
    Args:
        agent: Agent to evaluate
        env: Environment
        obs_mode: Observation mode
        num_episodes: Number of evaluation episodes
        max_steps: Max steps per episode
        use_pretrain_sample: Use sample_action_pretrain for DPPO/ReinFlow
        verbose: Print per-episode info
        
    Returns:
        Dict with:
            - success_rate: Fraction of successful episodes
            - avg_reward: Average episode reward
            - avg_length: Average episode length
    """
    successes = 0
    total_reward = 0
    total_length = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0
        
        for step in range(max_steps):
            # Prepare observation for agent
            if obs_mode == "state":
                agent_obs = obs
            elif obs_mode == "image":
                agent_obs = {"image": obs}
            else:  # state_image
                agent_obs = {"state": obs["state"], "image": obs["image"]}
            
            # Get action
            with torch.no_grad():
                if use_pretrain_sample and hasattr(agent, 'sample_action_pretrain'):
                    result = agent.sample_action_pretrain(agent_obs)
                else:
                    result = agent.sample_action(agent_obs)
                
                action = result[0] if isinstance(result, tuple) else result
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if action.ndim > 1:
                    action = action[0]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            if terminated or truncated:
                break
        
        total_length += step + 1
        total_reward += ep_reward
        
        # Success: sparse reward environment, success if object reached goal
        if ep_reward > -max_steps:  # Got some positive reward
            successes += 1
        
        if verbose:
            print(f"  Episode {ep+1}: reward={ep_reward:.2f}, length={step+1}")
    
    return {
        "success_rate": successes / num_episodes,
        "avg_reward": total_reward / num_episodes,
        "avg_length": total_length / num_episodes,
    }


def train_and_evaluate_all(
    dataset: Dict[str, np.ndarray],
    obs_mode: str,
    state_dim: int,
    action_dim: int,
    image_shape: Tuple[int, int, int],
    num_steps: int,
    eval_interval: int,
    eval_episodes: int,
    device: str,
    seed: int = 42,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, List]], Dict[str, Dict[str, Any]]]:
    """Train all algorithms and track metrics.
    
    Args:
        dataset: Training dataset
        obs_mode: Observation mode
        state_dim: State dimension
        action_dim: Action dimension
        image_shape: Image shape (H, W, C)
        num_steps: Total training steps
        eval_interval: Steps between evaluations
        eval_episodes: Episodes per evaluation
        device: Device to use
        seed: Random seed
        checkpoint_dir: Directory to save best checkpoints
        
    Returns:
        Tuple of:
        - results: Dict[algo_name -> Dict[metric_name -> List[values]]]
        - best_checkpoints: Dict[algo_name -> Dict with 'state_dict', 'step', 'success_rate', 'use_pretrain']
    """
    results = {}
    best_checkpoints = {}  # Store best checkpoint info for each algorithm
    batch_size = 128
    n_samples = len(dataset["actions"])
    
    # Create checkpoint directory
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Temp directory for saving best models during training
    import tempfile
    temp_ckpt_dir = tempfile.mkdtemp(prefix="best_ckpt_")
    
    # Create evaluation environment
    eval_env = make_pick_and_place_env(
        obs_mode=obs_mode,
        seed=seed + 1000,
        image_size=image_shape[0],
        reward_type="sparse"
    )
    
    # Common agent kwargs
    common_kwargs = {
        "action_dim": action_dim,
        "obs_mode": obs_mode,
        "device": device,
        "hidden_dims": [256, 256],
    }
    if obs_mode in ["state", "state_image"]:
        common_kwargs["state_dim"] = state_dim
    if obs_mode in ["image", "state_image"]:
        common_kwargs["image_shape"] = image_shape
    
    eval_steps = list(range(0, num_steps + 1, eval_interval))
    if eval_steps[-1] != num_steps:
        eval_steps.append(num_steps)
    
    # ==================== 1. Diffusion Policy ====================
    algo_name = "DiffusionPolicy"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {"steps": [], "success_rate": [], "avg_reward": []}
    best_success = -1.0
    best_step = 0
    
    try:
        agent = DiffusionPolicyAgent(
            num_diffusion_steps=20,
            **common_kwargs
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(agent, eval_env, obs_mode, eval_episodes)
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}")
                
                # Save best checkpoint
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    best_step = step
                    # Save model to temp file
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": False,
                        "agent_class": "DiffusionPolicyAgent",
                        "agent_kwargs": {"num_diffusion_steps": 20, **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                agent.train_step(batch)
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 2. Flow Matching ====================
    algo_name = "FlowMatching"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {"steps": [], "success_rate": [], "avg_reward": []}
    best_success = -1.0
    best_step = 0
    
    try:
        agent = FlowMatchingPolicy(
            num_inference_steps=20,
            **common_kwargs
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(agent, eval_env, obs_mode, eval_episodes)
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    best_step = step
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": False,
                        "agent_class": "FlowMatchingPolicy",
                        "agent_kwargs": {"num_inference_steps": 20, **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                agent.train_step(batch)
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 3. Consistency Flow ====================
    algo_name = "ConsistencyFlow"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {"steps": [], "success_rate": [], "avg_reward": []}
    best_success = -1.0
    best_step = 0
    
    try:
        agent = ConsistencyFlowPolicy(
            num_inference_steps=5,
            flow_batch_ratio=0.7,
            consistency_batch_ratio=0.3,
            **common_kwargs
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(agent, eval_env, obs_mode, eval_episodes)
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    best_step = step
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": False,
                        "agent_class": "ConsistencyFlowPolicy",
                        "agent_kwargs": {"num_inference_steps": 5, "flow_batch_ratio": 0.7, "consistency_batch_ratio": 0.3, **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                agent.train_step(batch)
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 4. Reflected Flow ====================
    algo_name = "ReflectedFlow"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {"steps": [], "success_rate": [], "avg_reward": []}
    best_success = -1.0
    best_step = 0
    
    try:
        agent = ReflectedFlowPolicy(
            num_inference_steps=20,
            reflection_mode="hard",
            **common_kwargs
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(agent, eval_env, obs_mode, eval_episodes)
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    best_step = step
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": False,
                        "agent_class": "ReflectedFlowPolicy",
                        "agent_kwargs": {"num_inference_steps": 20, "reflection_mode": "hard", **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                agent.train_step(batch)
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 5. Diffusion QL ====================
    algo_name = "DiffusionQL"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {"steps": [], "success_rate": [], "avg_reward": []}
    best_success = -1.0
    best_step = 0
    
    try:
        agent = DiffusionDoubleQAgent(
            num_diffusion_steps=20,
            **common_kwargs
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(agent, eval_env, obs_mode, eval_episodes)
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    best_step = step
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": False,
                        "agent_class": "DiffusionDoubleQAgent",
                        "agent_kwargs": {"num_diffusion_steps": 20, **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                agent.train_step(batch)
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 6. CPQL ====================
    algo_name = "CPQL"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {"steps": [], "success_rate": [], "avg_reward": []}
    best_success = -1.0
    best_step = 0
    
    try:
        agent = CPQLAgent(
            sigma_max=80.0,
            sigma_min=0.002,
            rho=7.0,
            **common_kwargs
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(agent, eval_env, obs_mode, eval_episodes)
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    best_step = step
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": False,
                        "agent_class": "CPQLAgent",
                        "agent_kwargs": {"sigma_max": 80.0, "sigma_min": 0.002, "rho": 7.0, **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                agent.train_step(batch)
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 7. DPPO ====================
    algo_name = "DPPO"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {"steps": [], "success_rate": [], "avg_reward": []}
    best_success = -1.0
    best_step = 0
    
    try:
        agent = DPPOAgent(
            num_diffusion_steps=10,
            ft_denoising_steps=5,
            **common_kwargs
        )
        
        # Enable gradients for pretraining
        for p in agent.actor.parameters():
            p.requires_grad = True
        dppo_optimizer = torch.optim.Adam(
            list(agent.actor.parameters()) + list(agent.actor_ft.parameters()),
            lr=1e-4
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(agent, eval_env, obs_mode, eval_episodes, use_pretrain_sample=True)
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    best_step = step
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": True,
                        "agent_class": "DPPOAgent",
                        "agent_kwargs": {"num_diffusion_steps": 10, "ft_denoising_steps": 5, **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                batch_actions = batch["actions"]
                
                # Get observation features
                state_input = batch.get("states")
                image_input = batch.get("images")
                obs_features = agent.obs_encoder(state=state_input, image=image_input)
                
                # Diffusion BC loss
                t = torch.randint(0, agent.num_diffusion_steps, (len(batch_actions),), device=device)
                noise = torch.randn_like(batch_actions)
                noisy_actions = agent.diffusion.q_sample(batch_actions, t, noise)
                t_normalized = t.float() / agent.num_diffusion_steps
                
                pred_noise = agent.actor(noisy_actions, t_normalized, obs_features=obs_features)
                pred_noise_ft = agent.actor_ft(noisy_actions, t_normalized, obs_features=obs_features)
                loss = F.mse_loss(pred_noise, noise) + F.mse_loss(pred_noise_ft, noise)
                
                dppo_optimizer.zero_grad()
                loss.backward()
                dppo_optimizer.step()
        
        # Freeze actor after pretraining
        for p in agent.actor.parameters():
            p.requires_grad = False
            
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 8. ReinFlow ====================
    algo_name = "ReinFlow"
    print(f"\n{'='*60}")
    print(f"Training {algo_name}...")
    print(f"{'='*60}")
    
    results[algo_name] = {"steps": [], "success_rate": [], "avg_reward": []}
    best_success = -1.0
    best_step = 0
    
    try:
        agent = ReinFlowAgent(
            num_flow_steps=10,
            noise_scheduler_type="learn",
            **common_kwargs
        )
        
        # Create optimizer for pretrain
        pretrain_optimizer = torch.optim.Adam(
            agent.policy.base_net.parameters(), lr=1e-4
        )
        
        for step in tqdm(range(num_steps + 1), desc=algo_name):
            if step in eval_steps:
                metrics = evaluate_agent(agent, eval_env, obs_mode, eval_episodes, use_pretrain_sample=True)
                results[algo_name]["steps"].append(step)
                results[algo_name]["success_rate"].append(metrics["success_rate"])
                results[algo_name]["avg_reward"].append(metrics["avg_reward"])
                print(f"  Step {step}: Success={metrics['success_rate']:.2%}, Reward={metrics['avg_reward']:.2f}")
                
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    best_step = step
                    temp_path = os.path.join(temp_ckpt_dir, f"{algo_name}_best.pt")
                    agent.save(temp_path)
                    best_checkpoints[algo_name] = {
                        "temp_path": temp_path,
                        "step": step,
                        "success_rate": best_success,
                        "use_pretrain": True,
                        "agent_class": "ReinFlowAgent",
                        "agent_kwargs": {"num_flow_steps": 10, "noise_scheduler_type": "learn", **common_kwargs},
                    }
                    print(f"    -> New best! Saving checkpoint at step {step}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = prepare_batch(dataset, idx, obs_mode, device)
                batch_actions = batch["actions"]
                
                # Get observation features
                state_input = batch.get("states")
                image_input = batch.get("images")
                obs_features = agent.obs_encoder(state=state_input, image=image_input)
                
                # Flow matching loss
                x_0 = torch.randn_like(batch_actions)
                t = torch.rand(len(batch_actions), device=device)
                t_expand = t.unsqueeze(-1)
                x_t = (1 - t_expand) * x_0 + t_expand * batch_actions
                target_v = batch_actions - x_0
                pred_v = agent.policy.base_net(x_t, t, obs_features=obs_features)
                loss = F.mse_loss(pred_v, target_v)
                
                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    eval_env.close()
    return results, best_checkpoints


def print_summary(results: Dict[str, Dict[str, List]]):
    """Print summary table of final results."""
    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<20} {'Success Rate':>15} {'Avg Reward':>15}")
    print("-" * 70)
    
    for algo_name in results:
        if results[algo_name]["success_rate"]:
            final_success = results[algo_name]["success_rate"][-1]
            final_reward = results[algo_name]["avg_reward"][-1]
            print(f"{algo_name:<20} {final_success:>14.1%} {final_reward:>15.2f}")
        else:
            print(f"{algo_name:<20} {'N/A':>15} {'N/A':>15}")
    
    print("=" * 70)


def save_results(results: Dict, output_path: str):
    """Save results to file."""
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for algo, metrics in results.items():
        results_json[algo] = {
            k: [float(v) for v in vals] if isinstance(vals, list) else vals
            for k, vals in metrics.items()
        }
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to {output_path}")


def evaluate_with_video(
    agent,
    obs_mode: str,
    image_shape: Tuple[int, int, int],
    num_episodes: int,
    video_dir: str,
    algo_name: str,
    use_pretrain_sample: bool = False,
    max_steps: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate agent and record videos.
    
    Args:
        agent: Agent to evaluate
        obs_mode: Observation mode
        image_shape: Image shape (H, W, C)
        num_episodes: Number of episodes to record
        video_dir: Directory to save videos
        algo_name: Algorithm name for video naming
        use_pretrain_sample: Use sample_action_pretrain for DPPO/ReinFlow
        max_steps: Max steps per episode
        seed: Random seed
        
    Returns:
        Dict with success_rate and avg_reward
    """
    import cv2
    
    os.makedirs(video_dir, exist_ok=True)
    
    # Create environment with rgb_array render mode
    env = make_pick_and_place_env(
        obs_mode=obs_mode,
        seed=seed,
        image_size=image_shape[0],
        reward_type="sparse"
    )
    
    successes = 0
    total_reward = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0
        frames = []
        
        for step in range(max_steps):
            # Render frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Prepare observation for agent
            if obs_mode == "state":
                agent_obs = obs
            elif obs_mode == "image":
                agent_obs = {"image": obs}
            else:  # state_image
                agent_obs = {"state": obs["state"], "image": obs["image"]}
            
            # Get action
            with torch.no_grad():
                if use_pretrain_sample and hasattr(agent, 'sample_action_pretrain'):
                    result = agent.sample_action_pretrain(agent_obs)
                else:
                    result = agent.sample_action(agent_obs)
                
                action = result[0] if isinstance(result, tuple) else result
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if action.ndim > 1:
                    action = action[0]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            if terminated or truncated:
                # Add final frame
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                break
        
        total_reward += ep_reward
        success = ep_reward > -max_steps
        if success:
            successes += 1
        
        # Save video
        if frames:
            video_path = os.path.join(video_dir, f"{algo_name}_ep{ep+1}_{'success' if success else 'fail'}.mp4")
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
            video_writer.release()
            print(f"    Saved video: {video_path}")
    
    env.close()
    
    return {
        "success_rate": successes / num_episodes,
        "avg_reward": total_reward / num_episodes,
    }


def create_agent_from_checkpoint(
    checkpoint: Dict[str, Any]
) -> Any:
    """Create agent from checkpoint info and load weights.
    
    Args:
        checkpoint: Dict with 'temp_path', 'agent_class', 'agent_kwargs'
        
    Returns:
        Agent with loaded weights
    """
    agent_class_name = checkpoint["agent_class"]
    agent_kwargs = checkpoint["agent_kwargs"]
    temp_path = checkpoint["temp_path"]
    
    # Map class name to class
    class_map = {
        "DiffusionPolicyAgent": DiffusionPolicyAgent,
        "FlowMatchingPolicy": FlowMatchingPolicy,
        "ConsistencyFlowPolicy": ConsistencyFlowPolicy,
        "ReflectedFlowPolicy": ReflectedFlowPolicy,
        "DiffusionDoubleQAgent": DiffusionDoubleQAgent,
        "CPQLAgent": CPQLAgent,
        "DPPOAgent": DPPOAgent,
        "ReinFlowAgent": ReinFlowAgent,
    }
    
    agent_class = class_map[agent_class_name]
    agent = agent_class(**agent_kwargs)
    agent.load(temp_path)
    
    return agent


def plot_training_progress(
    results: Dict[str, Dict[str, List]],
    save_path: str,
    title: str = "Pick-and-Place Training Progress"
):
    """Plot training progress curves for all algorithms.
    
    Creates two subplots:
    - Left: Success Rate vs Training Steps
    - Right: Average Reward vs Training Steps
    
    Args:
        results: Dict[algo_name -> Dict[metric_name -> List[values]]]
        save_path: Path to save the figure
        title: Title for the figure
    """
    # Filter algorithms that have results
    algorithms = [name for name in results.keys() if results[name].get("steps")]
    
    if not algorithms:
        print("No results to plot")
        return
    
    # Color palette for different algorithms
    colors = [
        '#1f77b4',  # Blue - DiffusionPolicy
        '#ff7f0e',  # Orange - FlowMatching
        '#2ca02c',  # Green - ConsistencyFlow
        '#d62728',  # Red - ReflectedFlow
        '#9467bd',  # Purple - DiffusionQL
        '#8c564b',  # Brown - CPQL
        '#e377c2',  # Pink - DPPO
        '#7f7f7f',  # Gray - ReinFlow
    ]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Success Rate
    ax1 = axes[0]
    for i, algo_name in enumerate(algorithms):
        steps = results[algo_name]["steps"]
        success_rates = results[algo_name]["success_rate"]
        if steps and success_rates:
            color = colors[i % len(colors)]
            ax1.plot(steps, [s * 100 for s in success_rates], 
                    label=algo_name, color=color, marker='o', markersize=4, linewidth=2)
    
    ax1.set_xlabel("Training Steps", fontsize=12)
    ax1.set_ylabel("Success Rate (%)", fontsize=12)
    ax1.set_title("Success Rate vs Training Steps", fontsize=13)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)
    
    # Plot Average Reward
    ax2 = axes[1]
    for i, algo_name in enumerate(algorithms):
        steps = results[algo_name]["steps"]
        avg_rewards = results[algo_name]["avg_reward"]
        if steps and avg_rewards:
            color = colors[i % len(colors)]
            ax2.plot(steps, avg_rewards, 
                    label=algo_name, color=color, marker='o', markersize=4, linewidth=2)
    
    ax2.set_xlabel("Training Steps", fontsize=12)
    ax2.set_ylabel("Average Reward", fontsize=12)
    ax2.set_title("Average Reward vs Training Steps", fontsize=13)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Overall title
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate all algorithms on Pick-and-Place task"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None,
        help="Path to HDF5 dataset. If not provided, generates a new one."
    )
    parser.add_argument(
        "--obs_mode", type=str, default="state_image",
        choices=["state", "image", "state_image"],
        help="Observation mode"
    )
    parser.add_argument("--num_steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--eval_interval", type=int, default=2000, help="Steps between evaluations")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Episodes per evaluation")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument(
        "--generate_dataset", action="store_true",
        help="Generate dataset if not found"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=200,
        help="Number of episodes for dataset generation"
    )
    parser.add_argument(
        "--video_eval_episodes", type=int, default=10,
        help="Number of episodes for video evaluation with best checkpoint"
    )
    parser.add_argument(
        "--skip_video", action="store_true",
        help="Skip video recording for best checkpoints"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    set_seed(args.seed)
    
    # Default dataset path
    if args.dataset_path is None:
        args.dataset_path = os.path.join(
            parent_dir, "data", f"fetch_pick_and_place_{args.obs_mode}.h5"
        )
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        if args.generate_dataset:
            print(f"Dataset not found. Generating {args.num_episodes} episodes...")
            
            # Import and run dataset generation
            from generate_pick_and_place_dataset import generate_dataset
            
            os.makedirs(os.path.dirname(args.dataset_path) or ".", exist_ok=True)
            generate_dataset(
                backend="fetch",
                obs_mode=args.obs_mode,
                num_episodes=args.num_episodes,
                max_episode_steps=50,
                output_path=args.dataset_path,
                seed=args.seed,
            )
        else:
            print(f"Error: Dataset not found at {args.dataset_path}")
            print("Use --generate_dataset to generate a new one, or provide --dataset_path")
            return
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Get dimensions
    state_dim = dataset["obs"].shape[1] if "obs" in dataset else 25
    action_dim = dataset["actions"].shape[1]
    image_shape = (128, 128, 3)  # Fixed for Fetch environment
    
    print(f"\nConfiguration:")
    print(f"  Observation mode: {args.obs_mode}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Image shape: {image_shape}")
    print(f"  Training steps: {args.num_steps}")
    print(f"  Eval interval: {args.eval_interval}")
    print(f"  Eval episodes: {args.eval_episodes}")
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.output_dir, f"checkpoints_{timestamp}")
    
    # Train and evaluate
    results, best_checkpoints = train_and_evaluate_all(
        dataset=dataset,
        obs_mode=args.obs_mode,
        state_dim=state_dim,
        action_dim=action_dim,
        image_shape=image_shape,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        device=device,
        seed=args.seed,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir,
        f"pick_and_place_{args.obs_mode}_{timestamp}.json"
    )
    save_results(results, output_path)
    
    # Plot training progress
    plot_path = os.path.join(
        args.output_dir,
        f"pick_and_place_{args.obs_mode}_{timestamp}.png"
    )
    plot_training_progress(
        results=results,
        save_path=plot_path,
        title=f"Pick-and-Place Training Progress ({args.obs_mode} mode)"
    )
    
    # Save best checkpoints to disk
    if best_checkpoints:
        print("\n" + "=" * 70)
        print("BEST CHECKPOINTS SUMMARY")
        print("=" * 70)
        import shutil
        for algo_name, ckpt in best_checkpoints.items():
            # Copy from temp location to final location
            final_path = os.path.join(checkpoint_dir, f"{algo_name}_best.pt")
            if os.path.exists(ckpt["temp_path"]):
                shutil.copy2(ckpt["temp_path"], final_path)
                ckpt["final_path"] = final_path
            print(f"  {algo_name}: step={ckpt['step']}, success={ckpt['success_rate']:.1%} -> {final_path}")
        print("=" * 70)
    
    # Video evaluation with best checkpoints
    if not args.skip_video and best_checkpoints:
        print("\n" + "=" * 70)
        print("VIDEO EVALUATION WITH BEST CHECKPOINTS")
        print("=" * 70)
        
        video_dir = os.path.join(args.output_dir, f"videos_{timestamp}")
        video_results = {}
        
        for algo_name, ckpt in best_checkpoints.items():
            print(f"\nEvaluating {algo_name} (best step={ckpt['step']}, train success={ckpt['success_rate']:.1%})...")
            
            try:
                # Create agent and load best checkpoint
                agent = create_agent_from_checkpoint(ckpt)
                
                # Evaluate with video
                metrics = evaluate_with_video(
                    agent=agent,
                    obs_mode=args.obs_mode,
                    image_shape=image_shape,
                    num_episodes=args.video_eval_episodes,
                    video_dir=os.path.join(video_dir, algo_name),
                    algo_name=algo_name,
                    use_pretrain_sample=ckpt["use_pretrain"],
                    max_steps=50,
                    seed=args.seed + 2000,
                )
                
                video_results[algo_name] = metrics
                print(f"  Final: Success={metrics['success_rate']:.1%}, Reward={metrics['avg_reward']:.2f}")
                
            except Exception as e:
                print(f"  Error during video evaluation: {e}")
                import traceback
                traceback.print_exc()
        
        # Print video evaluation summary
        if video_results:
            print("\n" + "=" * 70)
            print("VIDEO EVALUATION SUMMARY")
            print("=" * 70)
            print(f"{'Algorithm':<20} {'Best Step':>12} {'Train Success':>15} {'Eval Success':>15} {'Eval Reward':>12}")
            print("-" * 70)
            for algo_name in video_results:
                ckpt = best_checkpoints[algo_name]
                metrics = video_results[algo_name]
                print(f"{algo_name:<20} {ckpt['step']:>12} {ckpt['success_rate']:>14.1%} {metrics['success_rate']:>14.1%} {metrics['avg_reward']:>12.2f}")
            print("=" * 70)
    
    print("\nValidation complete!")


if __name__ == "__main__":
    main()
