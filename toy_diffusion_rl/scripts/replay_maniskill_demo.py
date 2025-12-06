#!/usr/bin/env python3
"""
Custom ManiSkill3 Demo Replay Script with Success/Failure Separation

This script replays ManiSkill3 demonstration trajectories and optionally separates
successful and failed trajectories into different output files.

Key Features:
- Replay trajectories with GPU simulation backend for accuracy
- Use --use-env-states for deterministic replay
- Split successful and failed trajectories into separate files
- Generate detailed replay status report
- Support for multiple observation modes (state, rgbd, etc.)

Based on ManiSkill3's official replay_trajectory.py but with added functionality
for analyzing and filtering trajectory replay results.

Usage:
    # Basic replay with success/failure separation
    python scripts/replay_maniskill_demo.py \
        --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
        --output-dir ./data/peg_insertion_replayed \
        --obs-mode state_dict+rgb \
        --use-env-states \
        --save-split \
        --num-envs 64 \
        --backend gpu

    # Full replay with all options
    python scripts/replay_maniskill_demo.py \
        --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
        --output-dir ./data/peg_insertion_replayed \
        --obs-mode state_dict+rgb \
        --use-env-states \
        --save-split \
        --record-rewards \
        --reward-mode dense \
        --num-envs 64 \
        --backend gpu \
        --count 100

Note:
    This script requires the rlft_ms3 conda environment with ManiSkill3 installed.
"""

import os
import sys
import argparse
import json
import copy
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import h5py
from tqdm import tqdm

# Suppress warnings before importing torch/mani_skill
import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)

import torch
import gymnasium as gym

# Import ManiSkill modules
try:
    import mani_skill.envs
    from mani_skill.trajectory import utils as trajectory_utils
    from mani_skill.utils import common, io_utils, wrappers
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.envs.utils.system.backend import CPU_SIM_BACKENDS
    MANISKILL_AVAILABLE = True
except ImportError:
    MANISKILL_AVAILABLE = False
    print("Warning: ManiSkill3 not available. Please install mani_skill package.")


@dataclass
class ReplayStatus:
    """Status of a single trajectory replay."""
    episode_id: int
    original_success: bool
    replay_success: bool
    original_steps: int
    replay_steps: int
    reason: str  # "success", "failed_task", "timeout", "physics_divergence", etc.
    final_reward: float = 0.0


@dataclass
class ReplayReport:
    """Summary report of replay results."""
    total_episodes: int
    successful_replays: int
    failed_replays: int
    success_rate: float
    failure_reasons: Dict[str, int]
    statuses: List[ReplayStatus]
    
    def to_dict(self) -> dict:
        return {
            "total_episodes": self.total_episodes,
            "successful_replays": self.successful_replays,
            "failed_replays": self.failed_replays,
            "success_rate": self.success_rate,
            "failure_reasons": self.failure_reasons,
            "statuses": [asdict(s) for s in self.statuses],
        }


def sanity_check_and_format_seed(episode: dict) -> None:
    """Sanity check trajectory seed and reformat reset kwargs seed if needed."""
    if "seed" in episode["reset_kwargs"]:
        if isinstance(episode["reset_kwargs"]["seed"], list):
            assert len(episode["reset_kwargs"]["seed"]) == 1, \
                f"Found multiple seeds for trajectory (id={episode['episode_id']})"
            episode["reset_kwargs"]["seed"] = episode["reset_kwargs"]["seed"][0]
        assert episode["reset_kwargs"]["seed"] == episode["episode_seed"], \
            f"Seed mismatch for trajectory (id={episode['episode_id']})"
    else:
        episode["reset_kwargs"]["seed"] = episode["episode_seed"]


def replay_parallelized_gpu(
    env,
    episodes: List[dict],
    trajectories: h5py.File,
    use_env_states: bool = True,
    use_first_env_state: bool = False,
    num_envs: int = 64,
    verbose: bool = False,
) -> Tuple[List[ReplayStatus], Dict[int, bool]]:
    """Replay trajectories using GPU-parallelized simulation.
    
    Args:
        env: ManiSkill environment (wrapped with RecordEpisode if saving)
        episodes: List of episode metadata dicts
        trajectories: HDF5 file containing trajectory data
        use_env_states: Whether to replay using environment states
        use_first_env_state: Whether to use only the first env state
        num_envs: Number of parallel environments
        verbose: Print progress info
        
    Returns:
        Tuple of (list of ReplayStatus, dict mapping episode_id to success)
    """
    statuses = []
    success_map = {}
    
    # Pad episodes to fill batch
    episode_pad = (num_envs - len(episodes) % num_envs) % num_envs
    padded_episodes = list(episodes) + [episodes[-1]] * episode_pad
    batches = [padded_episodes[i:i+num_envs] for i in range(0, len(padded_episodes), num_envs)]
    
    pbar = tqdm(total=len(episodes), desc="Replaying trajectories") if verbose else None
    
    for batch_idx, episode_batch in enumerate(batches):
        trajectory_ids = [ep["episode_id"] for ep in episode_batch]
        episode_lens = np.array([ep["elapsed_steps"] for ep in episode_batch])
        max_steps = max(episode_lens)
        
        # Prepare seeds and reset
        seeds = torch.tensor(
            [ep["episode_seed"] for ep in episode_batch],
            device=env.unwrapped.device,
        )
        env.reset(seed=seeds)
        
        # Prepare batched env states and actions
        env_states_list = []
        actions_batch = []
        
        for i, traj_id in enumerate(trajectory_ids):
            traj = trajectories[f"traj_{traj_id}"]
            episode = episode_batch[i]
            sanity_check_and_format_seed(episode)
            
            env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
            actions = np.array(traj["actions"])
            
            # Pad to max length
            while len(env_states) < max_steps + 1:
                env_states.append(env_states[-1])
            if len(actions) < max_steps:
                actions = np.concatenate([
                    actions,
                    np.zeros((max_steps - len(actions), actions.shape[1]))
                ], axis=0)
            
            env_states_list.append(env_states)
            actions_batch.append(actions)
        
        # Convert to batched format
        env_states_batch = []
        for t in range(max_steps + 1):
            env_states_batch.append(
                trajectory_utils.list_of_dicts_to_dict(
                    [env_states_list[i][t] for i in range(len(env_states_list))]
                )
            )
        actions_batch = np.stack(actions_batch, axis=1)  # (T, N, action_dim)
        
        # Set initial state if needed
        if use_first_env_state or use_env_states:
            env.unwrapped.set_state_dict(env_states_batch[0])
        
        # Track per-environment info
        env_final_success = np.zeros(num_envs, dtype=bool)
        env_final_rewards = np.zeros(num_envs)
        env_steps = np.zeros(num_envs, dtype=np.int32)
        
        # Replay trajectory
        for t in range(max_steps):
            action = actions_batch[t]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if use_env_states and t < max_steps - 1:
                env.unwrapped.set_state_dict(env_states_batch[t + 1])
            
            # Update tracking
            reward_np = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else np.array(reward)
            env_final_rewards += reward_np
            
            # Mark environments that reached their original length
            active_mask = (t < episode_lens - 1)
            env_steps = np.where(active_mask, env_steps + 1, env_steps)
            
            # Check success for environments at their final step
            if "success" in info:
                success_np = info["success"].cpu().numpy() if isinstance(info["success"], torch.Tensor) else np.array(info["success"])
                at_final_step = (t == episode_lens - 1)
                env_final_success = np.where(at_final_step, success_np, env_final_success)
        
        # Record status for each episode in batch (excluding padding)
        actual_batch_size = num_envs if batch_idx < len(batches) - 1 else (len(episodes) % num_envs or num_envs)
        for i in range(actual_batch_size):
            if batch_idx * num_envs + i >= len(episodes):
                continue
                
            episode = episode_batch[i]
            episode_id = episode["episode_id"]
            original_success = episode.get("success", True)  # Assume success if not specified
            replay_success = bool(env_final_success[i])
            
            if replay_success:
                reason = "success"
            elif not original_success:
                reason = "original_failed"
            else:
                reason = "physics_divergence"
            
            status = ReplayStatus(
                episode_id=episode_id,
                original_success=original_success,
                replay_success=replay_success,
                original_steps=episode["elapsed_steps"],
                replay_steps=int(episode_lens[i]),
                reason=reason,
                final_reward=float(env_final_rewards[i]),
            )
            statuses.append(status)
            success_map[episode_id] = replay_success
            
            if pbar:
                pbar.update(1)
    
    if pbar:
        pbar.close()
    
    return statuses, success_map


def copy_trajectory_to_file(
    src_h5: h5py.File,
    dst_h5: h5py.File,
    episode_id: int,
    new_episode_id: int,
) -> None:
    """Copy a trajectory from source to destination HDF5 file."""
    src_key = f"traj_{episode_id}"
    dst_key = f"traj_{new_episode_id}"
    
    if src_key in src_h5:
        src_h5.copy(src_key, dst_h5, name=dst_key)


def split_trajectories(
    traj_path: str,
    success_map: Dict[int, bool],
    output_dir: str,
    json_data: dict,
    replayed_episodes: Optional[List[dict]] = None,
) -> Tuple[str, str]:
    """Split trajectories into success and failure files.
    
    Args:
        traj_path: Path to original trajectory file
        success_map: Dict mapping episode_id to success status
        output_dir: Output directory
        json_data: Original trajectory metadata
        replayed_episodes: List of episodes that were actually replayed (if None, use all)
        
    Returns:
        Tuple of (success_file_path, failure_file_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    success_path = os.path.join(output_dir, "trajectory_success.h5")
    failure_path = os.path.join(output_dir, "trajectory_failed.h5")
    
    success_episodes = []
    failure_episodes = []
    
    # Use only replayed episodes if specified
    episodes_to_process = replayed_episodes if replayed_episodes is not None else json_data["episodes"]
    # Filter to only episodes that are in the success_map (i.e., were actually replayed)
    episodes_to_process = [ep for ep in episodes_to_process if ep["episode_id"] in success_map]
    
    with h5py.File(traj_path, 'r') as src_h5:
        with h5py.File(success_path, 'w') as success_h5, \
             h5py.File(failure_path, 'w') as failure_h5:
            
            success_idx = 0
            failure_idx = 0
            
            for episode in episodes_to_process:
                episode_id = episode["episode_id"]
                is_success = success_map.get(episode_id, False)
                
                if is_success:
                    copy_trajectory_to_file(src_h5, success_h5, episode_id, success_idx)
                    episode_copy = copy.deepcopy(episode)
                    episode_copy["episode_id"] = success_idx
                    success_episodes.append(episode_copy)
                    success_idx += 1
                else:
                    copy_trajectory_to_file(src_h5, failure_h5, episode_id, failure_idx)
                    episode_copy = copy.deepcopy(episode)
                    episode_copy["episode_id"] = failure_idx
                    failure_episodes.append(episode_copy)
                    failure_idx += 1
    
    # Save metadata JSON files
    success_json = copy.deepcopy(json_data)
    success_json["episodes"] = success_episodes
    with open(success_path.replace(".h5", ".json"), 'w') as f:
        json.dump(success_json, f, indent=2)
    
    failure_json = copy.deepcopy(json_data)
    failure_json["episodes"] = failure_episodes
    with open(failure_path.replace(".h5", ".json"), 'w') as f:
        json.dump(failure_json, f, indent=2)
    
    print(f"Success trajectories: {len(success_episodes)} -> {success_path}")
    print(f"Failed trajectories: {len(failure_episodes)} -> {failure_path}")
    
    return success_path, failure_path


def main():
    parser = argparse.ArgumentParser(
        description="Replay ManiSkill3 trajectories with success/failure separation"
    )
    
    # Input/Output
    parser.add_argument(
        "--traj-path", type=str, required=True,
        help="Path to the trajectory .h5 file to replay"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for split trajectories and report (default: same as input)"
    )
    
    # Replay options
    parser.add_argument(
        "--obs-mode", "-o", type=str, default=None,
        help="Target observation mode (e.g., state_dict+rgb, rgbd)"
    )
    parser.add_argument(
        "--backend", "-b", type=str, default=None,
        choices=["cpu", "gpu", "physx_cpu", "physx_cuda"],
        help="Simulation backend (default: same as original)"
    )
    parser.add_argument(
        "--use-env-states", action="store_true",
        help="Replay using environment states for deterministic replay"
    )
    parser.add_argument(
        "--use-first-env-state", action="store_true",
        help="Use only the first env state to set initial state"
    )
    parser.add_argument(
        "--num-envs", "-n", type=int, default=64,
        help="Number of parallel environments for GPU replay"
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Number of trajectories to replay (default: all)"
    )
    
    # Reward recording
    parser.add_argument(
        "--record-rewards", action="store_true",
        help="Record rewards during replay"
    )
    parser.add_argument(
        "--reward-mode", type=str, default=None,
        help="Reward mode (sparse, dense, normalized_dense)"
    )
    
    # Output options
    parser.add_argument(
        "--save-split", action="store_true",
        help="Save success and failure trajectories to separate files"
    )
    parser.add_argument(
        "--save-report", action="store_true", default=True,
        help="Save detailed replay status report"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose progress information"
    )
    
    args = parser.parse_args()
    
    # Check ManiSkill availability
    if not MANISKILL_AVAILABLE:
        print("Error: ManiSkill3 is not available.")
        print("Please install mani_skill package and run in the rlft_ms3 environment.")
        sys.exit(1)
    
    # Validate input
    traj_path = os.path.expanduser(args.traj_path)
    if not os.path.exists(traj_path):
        print(f"Error: Trajectory file not found: {traj_path}")
        sys.exit(1)
    
    # Load trajectory metadata
    json_path = traj_path.replace(".h5", ".json")
    if not os.path.exists(json_path):
        print(f"Error: Trajectory metadata not found: {json_path}")
        sys.exit(1)
    
    json_data = io_utils.load_json(json_path)
    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"].copy()
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(traj_path)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine backend
    if args.backend is None:
        args.backend = ori_env_kwargs.get("sim_backend", "physx_cpu")
    elif args.backend == "gpu":
        args.backend = "physx_cuda"
    elif args.backend == "cpu":
        args.backend = "physx_cpu"
    
    # Setup environment kwargs
    env_kwargs = ori_env_kwargs.copy()
    env_kwargs["sim_backend"] = args.backend
    env_kwargs["num_envs"] = args.num_envs
    
    if args.obs_mode is not None:
        env_kwargs["obs_mode"] = args.obs_mode
    if args.reward_mode is not None:
        env_kwargs["reward_mode"] = args.reward_mode
    
    print("\n" + "=" * 60)
    print("ManiSkill3 Trajectory Replay")
    print("=" * 60)
    print(f"Task: {env_id}")
    print(f"Trajectory: {traj_path}")
    print(f"Backend: {args.backend}")
    print(f"Num envs: {args.num_envs}")
    print(f"Use env states: {args.use_env_states}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)
    
    # Get episodes to replay
    episodes = json_data["episodes"]
    if args.count is not None:
        episodes = episodes[:args.count]
    print(f"\nTotal episodes to replay: {len(episodes)}")
    
    # Create environment
    print("\nCreating environment...")
    env = gym.make(env_id, **env_kwargs)
    print(f"  Environment created: {env_id}")
    
    # Open trajectory file
    with h5py.File(traj_path, 'r') as trajectories:
        # Replay trajectories
        print("\nReplaying trajectories...")
        if args.backend not in CPU_SIM_BACKENDS:
            # GPU parallelized replay
            statuses, success_map = replay_parallelized_gpu(
                env=env,
                episodes=episodes,
                trajectories=trajectories,
                use_env_states=args.use_env_states,
                use_first_env_state=args.use_first_env_state,
                num_envs=args.num_envs,
                verbose=args.verbose or True,
            )
        else:
            print("Warning: CPU replay not fully implemented, using basic loop...")
            # Simple CPU replay (not parallelized)
            statuses = []
            success_map = {}
            pbar = tqdm(episodes, desc="Replaying")
            for episode in pbar:
                episode_id = episode["episode_id"]
                sanity_check_and_format_seed(episode)
                
                env.reset(**episode["reset_kwargs"])
                traj = trajectories[f"traj_{episode_id}"]
                actions = np.array(traj["actions"])
                
                if args.use_env_states or args.use_first_env_state:
                    env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
                    env.set_state_dict(env_states[0])
                
                total_reward = 0.0
                final_success = False
                for t, action in enumerate(actions):
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    if args.use_env_states and t < len(actions) - 1:
                        env.set_state_dict(env_states[t + 1])
                    if terminated or truncated:
                        final_success = info.get("success", False)
                        break
                
                status = ReplayStatus(
                    episode_id=episode_id,
                    original_success=episode.get("success", True),
                    replay_success=final_success,
                    original_steps=episode["elapsed_steps"],
                    replay_steps=t + 1,
                    reason="success" if final_success else "failed",
                    final_reward=total_reward,
                )
                statuses.append(status)
                success_map[episode_id] = final_success
    
    env.close()
    
    # Generate report
    successful = sum(1 for s in statuses if s.replay_success)
    failed = len(statuses) - successful
    
    failure_reasons = {}
    for s in statuses:
        if not s.replay_success:
            reason = s.reason
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    report = ReplayReport(
        total_episodes=len(statuses),
        successful_replays=successful,
        failed_replays=failed,
        success_rate=successful / len(statuses) if statuses else 0.0,
        failure_reasons=failure_reasons,
        statuses=statuses,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("REPLAY SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {report.total_episodes}")
    print(f"Successful replays: {report.successful_replays}")
    print(f"Failed replays: {report.failed_replays}")
    print(f"Success rate: {report.success_rate:.1%}")
    if failure_reasons:
        print("\nFailure reasons:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print("=" * 60)
    
    # Save report
    if args.save_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(args.output_dir, f"replay_report_{timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved: {report_path}")
    
    # Split trajectories if requested
    if args.save_split:
        print("\nSplitting trajectories...")
        success_path, failure_path = split_trajectories(
            traj_path=traj_path,
            success_map=success_map,
            output_dir=args.output_dir,
            json_data=json_data,
        )
    
    print("\nReplay complete!")


if __name__ == "__main__":
    # spawn is needed for multiprocessing with CUDA
    mp.set_start_method("spawn", force=True)
    main()
