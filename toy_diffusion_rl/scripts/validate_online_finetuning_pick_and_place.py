"""
Validate DPPO and ReinFlow online fine-tuning for Pick-and-Place task.

This script validates that online reinforcement learning algorithms
(DPPO and ReinFlow) can properly:
1. Pretrain using BC/Flow Matching on the Pick-and-Place dataset
2. Collect online rollouts in the Pick-and-Place environment
3. Update policies using PPO
4. Improve success rate through online fine-tuning

Usage:
    python validate_online_finetuning_pick_and_place.py --algorithm dppo --obs_mode state
    python validate_online_finetuning_pick_and_place.py --algorithm reinflow --obs_mode state_image
    python validate_online_finetuning_pick_and_place.py --all --obs_mode state
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
workspace_dir = os.path.dirname(project_dir)
sys.path.insert(0, project_dir)
sys.path.insert(0, workspace_dir)

from toy_diffusion_rl.envs import make_pick_and_place_env
from toy_diffusion_rl.common.dataset_loader import PickAndPlaceOfflineDataset
from toy_diffusion_rl.algorithms.dppo.agent import DPPOAgent
from toy_diffusion_rl.algorithms.reinflow.agent import ReinFlowAgent


def get_default_dataset_path(obs_mode: str) -> str:
    """Get default dataset path based on observation mode."""
    base_path = Path(__file__).parent.parent / "data"
    
    if obs_mode == "state":
        # Check if state-only dataset exists, otherwise use state_image
        state_path = base_path / "fetch_pick_and_place_state_1000ep.h5"
        if state_path.exists():
            return str(state_path)
        # Fall back to state_image dataset (will only use state part)
        return str(base_path / "fetch_pick_and_place_state_image_1000ep.h5")
    elif obs_mode == "state_image":
        return str(base_path / "fetch_pick_and_place_state_image_1000ep.h5")
    else:
        return str(base_path / "fetch_pick_and_place_state_image_1000ep.h5")


def evaluate_policy(
    agent,
    env,
    num_episodes: int = 20,
    max_steps: int = 50,
) -> dict:
    """Evaluate policy on Pick-and-Place task.
    
    Returns:
        Dictionary with success_rate, mean_return, std_return
    """
    returns = []
    successes = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        success = False
        
        for _ in range(max_steps):
            # Get action from agent
            with torch.no_grad():
                result = agent.sample_action(obs, deterministic=True)
                # sample_action returns (action, log_prob, value)
                action = result[0] if isinstance(result, tuple) else result
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            
            # Check success
            if info.get("is_success", False):
                success = True
            
            if terminated or truncated:
                break
        
        returns.append(episode_return)
        successes.append(float(success))
    
    return {
        "success_rate": np.mean(successes),
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
    }


def save_checkpoint(
    agent,
    checkpoint_dir: Path,
    checkpoint_name: str,
    metadata: dict = None,
):
    """Save agent checkpoint.
    
    Args:
        agent: Agent to save
        checkpoint_dir: Directory to save checkpoint
        checkpoint_name: Name of the checkpoint file (without extension)
        metadata: Optional metadata to save alongside checkpoint
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pt"
    
    # Save agent
    agent.save(str(checkpoint_path))
    
    # Save metadata if provided
    if metadata:
        metadata_path = checkpoint_dir / f"{checkpoint_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    print(f"    💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def train_dppo_online(
    obs_mode: str = "state",
    dataset_path: str = None,
    pretrain_steps: int = 2000,
    online_iterations: int = 50,
    rollout_steps: int = 512,
    eval_frequency: int = 10,
    eval_episodes: int = 10,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_checkpoints: bool = True,
    checkpoint_dir: Path = None,
) -> dict:
    """Train DPPO agent with online fine-tuning on Pick-and-Place.
    
    Args:
        obs_mode: "state" or "state_image"
        dataset_path: Path to offline dataset (None for default)
        pretrain_steps: Number of BC pretraining steps
        online_iterations: Number of online PPO iterations
        rollout_steps: Steps per rollout collection
        eval_frequency: Evaluate every N iterations
        eval_episodes: Number of episodes for evaluation
        seed: Random seed
        device: Torch device
    
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"DPPO Online Fine-tuning (obs_mode={obs_mode})")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    if dataset_path is None:
        dataset_path = get_default_dataset_path(obs_mode)
    
    if not os.path.exists(dataset_path):
        print(f"  ✗ Dataset not found: {dataset_path}")
        return {"error": f"Dataset not found: {dataset_path}"}
    
    print(f"  Loading dataset: {dataset_path}")
    dataset = PickAndPlaceOfflineDataset(dataset_path, obs_mode=obs_mode)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    # Create environment
    print("  Creating Pick-and-Place environment...")
    env = make_pick_and_place_env(obs_mode=obs_mode, seed=seed)
    eval_env = make_pick_and_place_env(obs_mode=obs_mode, seed=seed + 100)
    
    # Get dimensions
    state_dim = dataset.state_dim  # Will be None if not available
    image_shape = dataset.image_shape  # Will be None if not available
    action_dim = dataset.action_dim
    
    print(f"  State dim: {state_dim}, Image shape: {image_shape}, Action dim: {action_dim}")
    
    # Create DPPO agent
    print("  Creating DPPO agent...")
    agent = DPPOAgent(
        action_dim=action_dim,
        obs_mode=obs_mode,
        state_dim=state_dim,
        image_shape=image_shape,
        hidden_dims=[256, 256],
        num_diffusion_steps=5,
        ft_denoising_steps=3,
        device=device,
    )
    
    # Initial evaluation
    print("\n  Initial evaluation (before pretraining)...")
    initial_eval = evaluate_policy(agent, eval_env, num_episodes=eval_episodes)
    print(f"    Success rate: {initial_eval['success_rate']:.2%}")
    print(f"    Mean return: {initial_eval['mean_return']:.3f}")
    
    # Phase 1: BC Pretraining
    print(f"\n  Phase 1: BC Pretraining ({pretrain_steps} steps)...")
    start_time = time.time()
    
    # Use pretrain_bc which directly iterates over dataloader
    loss_info = agent.pretrain_bc(
        dataloader=dataloader,
        num_steps=pretrain_steps,
    )
    
    pretrain_time = time.time() - start_time
    print(f"  BC Pretraining done in {pretrain_time:.1f}s, Final Loss: {loss_info.get('bc_loss', 0):.4f}")
    
    # Post-pretrain evaluation
    print("\n  Evaluation after pretraining...")
    pretrain_eval = evaluate_policy(agent, eval_env, num_episodes=eval_episodes)
    print(f"    Success rate: {pretrain_eval['success_rate']:.2%}")
    print(f"    Mean return: {pretrain_eval['mean_return']:.3f}")
    
    # Save pretrain checkpoint
    if save_checkpoints and checkpoint_dir is not None:
        save_checkpoint(
            agent, checkpoint_dir, "pretrain",
            metadata={
                "step": pretrain_steps,
                "phase": "pretrain",
                "success_rate": pretrain_eval["success_rate"],
                "mean_return": pretrain_eval["mean_return"],
            }
        )
    
    # Phase 2: Online Fine-tuning
    print(f"\n  Phase 2: Online Fine-tuning ({online_iterations} iterations)...")
    
    results = {
        "algorithm": "dppo",
        "obs_mode": obs_mode,
        "pretrain_steps": pretrain_steps,
        "online_iterations": online_iterations,
        "initial_success_rate": initial_eval["success_rate"],
        "pretrain_success_rate": pretrain_eval["success_rate"],
        "training_log": [],
    }
    
    best_success_rate = pretrain_eval["success_rate"]
    best_checkpoint_iter = None
    
    for iteration in range(online_iterations):
        # Collect rollout
        buffer = agent.collect_rollout(env, rollout_steps=rollout_steps)
        
        # Update policy
        update_info = agent.update(buffer)
        
        # Log progress
        if (iteration + 1) % eval_frequency == 0:
            eval_result = evaluate_policy(agent, eval_env, num_episodes=eval_episodes)
            
            results["training_log"].append({
                "iteration": iteration + 1,
                "success_rate": eval_result["success_rate"],
                "mean_return": eval_result["mean_return"],
                "policy_loss": update_info.get("policy_loss", 0),
                "value_loss": update_info.get("value_loss", 0),
            })
            
            print(f"    Iter {iteration+1}/{online_iterations}: "
                  f"Success={eval_result['success_rate']:.2%}, "
                  f"Return={eval_result['mean_return']:.3f}, "
                  f"PolicyLoss={update_info.get('policy_loss', 0):.4f}")
            
            # Save checkpoint at each evaluation
            if save_checkpoints and checkpoint_dir is not None:
                save_checkpoint(
                    agent, checkpoint_dir, f"iter_{iteration+1:04d}",
                    metadata={
                        "iteration": iteration + 1,
                        "phase": "online",
                        "success_rate": eval_result["success_rate"],
                        "mean_return": eval_result["mean_return"],
                        "policy_loss": update_info.get("policy_loss", 0),
                        "value_loss": update_info.get("value_loss", 0),
                    }
                )
            
            if eval_result["success_rate"] > best_success_rate:
                best_success_rate = eval_result["success_rate"]
                best_checkpoint_iter = iteration + 1
    
    # Final evaluation
    print("\n  Final evaluation...")
    final_eval = evaluate_policy(agent, eval_env, num_episodes=eval_episodes * 2)
    results["final_success_rate"] = final_eval["success_rate"]
    results["final_mean_return"] = final_eval["mean_return"]
    results["best_success_rate"] = best_success_rate
    results["best_checkpoint_iter"] = best_checkpoint_iter
    
    print(f"\n  DPPO Results:")
    print(f"    Initial success rate: {initial_eval['success_rate']:.2%}")
    print(f"    After pretraining: {pretrain_eval['success_rate']:.2%}")
    print(f"    Final success rate: {final_eval['success_rate']:.2%}")
    print(f"    Best success rate: {best_success_rate:.2%}")
    if best_checkpoint_iter:
        print(f"    Best checkpoint: iter_{best_checkpoint_iter:04d}")
    
    # Check if online fine-tuning worked
    improvement = final_eval["success_rate"] - pretrain_eval["success_rate"]
    if improvement > 0:
        print(f"  ✓ Online fine-tuning improved performance by {improvement:.2%}")
        results["status"] = "success"
    else:
        print(f"  ! No improvement from online fine-tuning (may need more iterations)")
        results["status"] = "no_improvement"
    
    env.close()
    eval_env.close()
    
    return results


def train_reinflow_online(
    obs_mode: str = "state",
    dataset_path: str = None,
    pretrain_steps: int = 2000,
    online_iterations: int = 50,
    rollout_steps: int = 512,
    eval_frequency: int = 10,
    eval_episodes: int = 10,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_checkpoints: bool = True,
    checkpoint_dir: Path = None,
) -> dict:
    """Train ReinFlow agent with online fine-tuning on Pick-and-Place.
    
    Args:
        obs_mode: "state" or "state_image"
        dataset_path: Path to offline dataset (None for default)
        pretrain_steps: Number of flow matching pretraining steps
        online_iterations: Number of online PPO iterations
        rollout_steps: Steps per rollout collection
        eval_frequency: Evaluate every N iterations
        eval_episodes: Number of episodes for evaluation
        seed: Random seed
        device: Torch device
        save_checkpoints: Whether to save checkpoints
        checkpoint_dir: Directory for saving checkpoints
    
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"ReinFlow Online Fine-tuning (obs_mode={obs_mode})")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    if dataset_path is None:
        dataset_path = get_default_dataset_path(obs_mode)
    
    if not os.path.exists(dataset_path):
        print(f"  ✗ Dataset not found: {dataset_path}")
        return {"error": f"Dataset not found: {dataset_path}"}
    
    print(f"  Loading dataset: {dataset_path}")
    dataset = PickAndPlaceOfflineDataset(dataset_path, obs_mode=obs_mode)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    # Create environment
    print("  Creating Pick-and-Place environment...")
    env = make_pick_and_place_env(obs_mode=obs_mode, seed=seed)
    eval_env = make_pick_and_place_env(obs_mode=obs_mode, seed=seed + 100)
    
    # Get dimensions
    state_dim = dataset.state_dim  # Will be None if not available
    image_shape = dataset.image_shape  # Will be None if not available
    action_dim = dataset.action_dim
    
    print(f"  State dim: {state_dim}, Image shape: {image_shape}, Action dim: {action_dim}")
    
    # Create ReinFlow agent
    print("  Creating ReinFlow agent...")
    agent = ReinFlowAgent(
        action_dim=action_dim,
        obs_mode=obs_mode,
        state_dim=state_dim,
        image_shape=image_shape,
        hidden_dims=[256, 256],
        num_flow_steps=5,
        device=device,
    )
    
    # Initial evaluation
    print("\n  Initial evaluation (before pretraining)...")
    initial_eval = evaluate_policy(agent, eval_env, num_episodes=eval_episodes)
    print(f"    Success rate: {initial_eval['success_rate']:.2%}")
    print(f"    Mean return: {initial_eval['mean_return']:.3f}")
    
    # Phase 1: Flow Matching Pretraining (using BC objective)
    print(f"\n  Phase 1: Flow Matching Pretraining ({pretrain_steps} steps)...")
    start_time = time.time()
    
    # Use pretrain_bc which directly iterates over dataloader
    loss_info = agent.pretrain_bc(
        dataloader=dataloader,
        num_steps=pretrain_steps,
    )
    
    pretrain_time = time.time() - start_time
    print(f"  Flow Pretraining done in {pretrain_time:.1f}s, Final Loss: {loss_info.get('bc_loss', 0):.4f}")
    
    # Post-pretrain evaluation
    print("\n  Evaluation after pretraining...")
    pretrain_eval = evaluate_policy(agent, eval_env, num_episodes=eval_episodes)
    print(f"    Success rate: {pretrain_eval['success_rate']:.2%}")
    print(f"    Mean return: {pretrain_eval['mean_return']:.3f}")
    
    # Save pretrain checkpoint
    if save_checkpoints and checkpoint_dir is not None:
        save_checkpoint(
            agent, checkpoint_dir, "pretrain",
            metadata={
                "step": pretrain_steps,
                "phase": "pretrain",
                "success_rate": pretrain_eval["success_rate"],
                "mean_return": pretrain_eval["mean_return"],
            }
        )
    
    # Phase 2: Online Fine-tuning
    print(f"\n  Phase 2: Online Fine-tuning ({online_iterations} iterations)...")
    
    results = {
        "algorithm": "reinflow",
        "obs_mode": obs_mode,
        "pretrain_steps": pretrain_steps,
        "online_iterations": online_iterations,
        "initial_success_rate": initial_eval["success_rate"],
        "pretrain_success_rate": pretrain_eval["success_rate"],
        "training_log": [],
    }
    
    best_success_rate = pretrain_eval["success_rate"]
    best_checkpoint_iter = None
    
    for iteration in range(online_iterations):
        # Collect rollout
        buffer = agent.collect_rollout(env, rollout_steps=rollout_steps)
        
        # Update policy
        update_info = agent.update(buffer)
        
        # Log progress
        if (iteration + 1) % eval_frequency == 0:
            eval_result = evaluate_policy(agent, eval_env, num_episodes=eval_episodes)
            
            results["training_log"].append({
                "iteration": iteration + 1,
                "success_rate": eval_result["success_rate"],
                "mean_return": eval_result["mean_return"],
                "policy_loss": update_info.get("policy_loss", 0),
                "value_loss": update_info.get("value_loss", 0),
            })
            
            print(f"    Iter {iteration+1}/{online_iterations}: "
                  f"Success={eval_result['success_rate']:.2%}, "
                  f"Return={eval_result['mean_return']:.3f}, "
                  f"PolicyLoss={update_info.get('policy_loss', 0):.4f}")
            
            # Save checkpoint at each evaluation
            if save_checkpoints and checkpoint_dir is not None:
                save_checkpoint(
                    agent, checkpoint_dir, f"iter_{iteration+1:04d}",
                    metadata={
                        "iteration": iteration + 1,
                        "phase": "online",
                        "success_rate": eval_result["success_rate"],
                        "mean_return": eval_result["mean_return"],
                        "policy_loss": update_info.get("policy_loss", 0),
                        "value_loss": update_info.get("value_loss", 0),
                    }
                )
            
            if eval_result["success_rate"] > best_success_rate:
                best_success_rate = eval_result["success_rate"]
                best_checkpoint_iter = iteration + 1
    
    # Final evaluation
    print("\n  Final evaluation...")
    final_eval = evaluate_policy(agent, eval_env, num_episodes=eval_episodes * 2)
    results["final_success_rate"] = final_eval["success_rate"]
    results["final_mean_return"] = final_eval["mean_return"]
    results["best_success_rate"] = best_success_rate
    results["best_checkpoint_iter"] = best_checkpoint_iter
    
    print(f"\n  ReinFlow Results:")
    print(f"    Initial success rate: {initial_eval['success_rate']:.2%}")
    print(f"    After pretraining: {pretrain_eval['success_rate']:.2%}")
    print(f"    Final success rate: {final_eval['success_rate']:.2%}")
    print(f"    Best success rate: {best_success_rate:.2%}")
    if best_checkpoint_iter:
        print(f"    Best checkpoint: iter_{best_checkpoint_iter:04d}")
    
    # Check if online fine-tuning worked
    improvement = final_eval["success_rate"] - pretrain_eval["success_rate"]
    if improvement > 0:
        print(f"  ✓ Online fine-tuning improved performance by {improvement:.2%}")
        results["status"] = "success"
    else:
        print(f"  ! No improvement from online fine-tuning (may need more iterations)")
        results["status"] = "no_improvement"
    
    env.close()
    eval_env.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate DPPO/ReinFlow online fine-tuning for Pick-and-Place"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["dppo", "reinflow"],
        help="Algorithm to validate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate both algorithms",
    )
    parser.add_argument(
        "--obs_mode",
        type=str,
        default="state",
        choices=["state", "state_image"],
        help="Observation mode",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to offline dataset",
    )
    parser.add_argument(
        "--pretrain_steps",
        type=int,
        default=2000,
        help="Number of pretraining steps",
    )
    parser.add_argument(
        "--online_iterations",
        type=int,
        default=50,
        help="Number of online training iterations",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=512,
        help="Steps per rollout collection",
    )
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=10,
        help="Evaluate every N iterations",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="Save model checkpoints during training",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints (default: checkpoints/<algo>_<obs_mode>_<timestamp>)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Pick-and-Place Online Fine-tuning Validation")
    print("=" * 70)
    print(f"  Observation Mode: {args.obs_mode}")
    print(f"  Pretrain Steps: {args.pretrain_steps}")
    print(f"  Online Iterations: {args.online_iterations}")
    print(f"  Rollout Steps: {args.rollout_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Save Checkpoints: {args.save_checkpoints}")
    
    algorithms = []
    if args.all:
        algorithms = ["dppo", "reinflow"]
    elif args.algorithm:
        algorithms = [args.algorithm]
    else:
        print("Error: Must specify --algorithm or --all")
        return
    
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for algorithm in algorithms:
        # Set up checkpoint directory for this algorithm
        if args.save_checkpoints:
            if args.checkpoint_dir:
                ckpt_dir = Path(args.checkpoint_dir) / f"{algorithm}_{args.obs_mode}"
            else:
                ckpt_dir = Path(__file__).parent.parent.parent / "checkpoints" / f"{algorithm}_{args.obs_mode}_{timestamp}"
        else:
            ckpt_dir = None
        
        if algorithm == "dppo":
            result = train_dppo_online(
                obs_mode=args.obs_mode,
                dataset_path=args.dataset_path,
                pretrain_steps=args.pretrain_steps,
                online_iterations=args.online_iterations,
                rollout_steps=args.rollout_steps,
                eval_frequency=args.eval_frequency,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
                save_checkpoints=args.save_checkpoints,
                checkpoint_dir=ckpt_dir,
            )
        else:  # reinflow
            result = train_reinflow_online(
                obs_mode=args.obs_mode,
                dataset_path=args.dataset_path,
                pretrain_steps=args.pretrain_steps,
                online_iterations=args.online_iterations,
                rollout_steps=args.rollout_steps,
                eval_frequency=args.eval_frequency,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
                save_checkpoints=args.save_checkpoints,
                checkpoint_dir=ckpt_dir,
            )
        
        all_results[algorithm] = result
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for alg, result in all_results.items():
        if "error" in result:
            print(f"  {alg.upper()}: ERROR - {result['error']}")
        else:
            print(f"  {alg.upper()}:")
            print(f"    Pretrain -> Final: {result['pretrain_success_rate']:.2%} -> {result['final_success_rate']:.2%}")
            print(f"    Best Success Rate: {result['best_success_rate']:.2%}")
            print(f"    Status: {result['status']}")
    
    # Save results
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(__file__).parent.parent.parent / "results"
        results_path.mkdir(exist_ok=True)
        
        filename = f"online_finetuning_pick_and_place_{args.obs_mode}_{timestamp}.json"
        with open(results_path / filename, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to: {results_path / filename}")


if __name__ == "__main__":
    main()
