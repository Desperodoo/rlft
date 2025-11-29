"""
Validate All Diffusion/Flow RL Algorithms on Multimodal Particle Environment

This script tests all unified (multimodal-capable) algorithm implementations:
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

Supports observation modes:
- "state": State vector only
- "image": Image observation only
- "state_image": Both state and image (multimodal)
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

# Import agents (unified versions are now the default)
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


def create_synthetic_images(states: np.ndarray, image_shape: Tuple[int, int, int]) -> np.ndarray:
    """Create synthetic images from states for testing image mode.
    
    Creates simple 2D Gaussian blobs. If states has only 1D, uses it for x and 0 for y.
    
    Args:
        states: (N, D) array of states (D can be 1 or 2)
        image_shape: (H, W, C) image shape
        
    Returns:
        (N, H, W, C) array of images
    """
    H, W, C = image_shape
    N = states.shape[0]
    images = np.zeros((N, H, W, C), dtype=np.float32)
    
    # Create coordinate grids
    y_coords = np.linspace(-4, 4, H)
    x_coords = np.linspace(-4, 4, W)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    state_dim = states.shape[1] if states.ndim > 1 else 1
    
    for i in range(N):
        # Center of Gaussian at state position
        if state_dim >= 2:
            cx, cy = states[i, 0], states[i, 1]
        else:
            # For 1D state, use x and center y at 0
            cx = states[i, 0] if states.ndim > 1 else states[i]
            cy = 0.0
        
        # Gaussian blob
        sigma = 0.5
        gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        
        # Normalize to [0, 1]
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min() + 1e-6)
        
        # Apply to all channels
        for c in range(C):
            images[i, :, :, c] = gaussian
    
    return images


def prepare_batch_for_obs_mode(
    batch: Dict[str, torch.Tensor],
    obs_mode: str,
    image_shape: Tuple[int, int, int],
    device: str
) -> Dict[str, torch.Tensor]:
    """Prepare batch according to observation mode.
    
    Args:
        batch: Original batch with states, actions, etc.
        obs_mode: "state", "image", or "state_image"
        image_shape: (H, W, C) for image generation
        device: Target device
        
    Returns:
        Prepared batch with appropriate observations
    """
    result = {}
    
    # Get states as numpy for image generation
    states_np = batch["states"].cpu().numpy() if isinstance(batch["states"], torch.Tensor) else batch["states"]
    
    if obs_mode == "state":
        result["states"] = batch["states"].to(device)
    elif obs_mode == "image":
        # Generate synthetic images
        images = create_synthetic_images(states_np, image_shape)
        # Use .contiguous() after permute to avoid view errors
        result["images"] = torch.FloatTensor(images).permute(0, 3, 1, 2).contiguous().to(device)  # (N, C, H, W)
    else:  # state_image
        result["states"] = batch["states"].to(device)
        images = create_synthetic_images(states_np, image_shape)
        result["images"] = torch.FloatTensor(images).permute(0, 3, 1, 2).contiguous().to(device)
    
    # Copy other fields
    for key in ["actions", "rewards", "next_states", "dones"]:
        if key in batch:
            result[key] = batch[key].to(device)
    
    return result


def create_obs_for_sampling(
    state: np.ndarray,
    obs_mode: str,
    image_shape: Tuple[int, int, int]
) -> Dict[str, np.ndarray]:
    """Create observation dict for sampling.
    
    Args:
        state: (state_dim,) or (1, state_dim) state array
        obs_mode: "state", "image", or "state_image"
        image_shape: (H, W, C) for image generation
        
    Returns:
        Observation dict with appropriate fields
    """
    if state.ndim == 1:
        state = state.reshape(1, -1)
    
    if obs_mode == "state":
        return state[0]
    elif obs_mode == "image":
        images = create_synthetic_images(state, image_shape)
        return {"image": images[0]}
    else:  # state_image
        images = create_synthetic_images(state, image_shape)
        return {"state": state[0], "image": images[0]}


def evaluate_agent(
    agent,
    state: np.ndarray,
    obs_mode: str,
    image_shape: Tuple[int, int, int],
    env: MultimodalParticleEnv,
    num_samples: int = 500,
    use_pretrain_sample: bool = False
) -> Tuple[np.ndarray, float]:
    """Evaluate agent and return samples and EMD.
    
    Args:
        agent: Agent to evaluate
        state: Base state for sampling
        obs_mode: Observation mode
        image_shape: Image shape for generating synthetic images
        env: Environment for EMD computation
        num_samples: Number of samples to generate
        use_pretrain_sample: Use sample_action_pretrain for DPPO/ReinFlow
        
    Returns:
        (samples, emd) tuple
    """
    samples = []
    for _ in range(num_samples):
        with torch.no_grad():
            obs = create_obs_for_sampling(state, obs_mode, image_shape)
            
            # Use pretrain sampling for DPPO/ReinFlow
            if use_pretrain_sample and hasattr(agent, 'sample_action_pretrain'):
                action = agent.sample_action_pretrain(obs)
            else:
                action = agent.sample_action(obs)
            
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


def train_with_progress_tracking(
    env: MultimodalParticleEnv,
    dataset: Dict[str, np.ndarray],
    state_dim: int,
    action_dim: int,
    obs_mode: str,
    image_shape: Tuple[int, int, int],
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
        obs_mode: Observation mode ("state", "image", "state_image")
        image_shape: Image shape as (H, W, C)
        num_steps: Total training steps
        device: Device to use
        eval_checkpoints: List of steps at which to evaluate
        
    Returns:
        Tuple of (progress_results, emd_history)
        - progress_results: Dict[algo_name -> Dict[step -> samples]]
        - emd_history: Dict[algo_name -> Dict[step -> emd_value]]
    """
    if eval_checkpoints is None:
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
    
    def get_batch(idx):
        """Get batch prepared for obs_mode."""
        batch = {
            "states": states_t[idx],
            "actions": actions_t[idx],
            "rewards": rewards_t[idx],
            "next_states": next_states_t[idx],
            "dones": dones_t[idx],
        }
        return prepare_batch_for_obs_mode(batch, obs_mode, image_shape, device)
    
    def eval_agent_wrapper(agent, algo_name, use_pretrain=False):
        return evaluate_agent(
            agent, state, obs_mode, image_shape, env,
            num_samples=500, use_pretrain_sample=use_pretrain
        )
    
    # Common agent kwargs
    common_kwargs = {
        "action_dim": action_dim,
        "obs_mode": obs_mode,
        "device": device,
    }
    if obs_mode in ["state", "state_image"]:
        common_kwargs["state_dim"] = state_dim
    if obs_mode in ["image", "state_image"]:
        common_kwargs["image_shape"] = image_shape

    # ==================== 1. Diffusion Policy ====================
    print("\n1. Training Diffusion Policy with progress tracking...")
    algo_name = "Diffusion Policy"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = DiffusionPolicyAgent(
            hidden_dims=[256, 256],
            num_diffusion_steps=20,
            **common_kwargs
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = eval_agent_wrapper(agent, algo_name)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = get_batch(idx)
                agent.train_step(batch)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 2. Flow Matching ====================
    print("\n2. Training Flow Matching with progress tracking...")
    algo_name = "Flow Matching"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = FlowMatchingPolicy(
            hidden_dims=[256, 256],
            num_inference_steps=20,
            **common_kwargs
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = eval_agent_wrapper(agent, algo_name)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = get_batch(idx)
                agent.train_step(batch)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 3. Consistency Flow ====================
    print("\n3. Training Consistency Flow with progress tracking...")
    algo_name = "Consistency Flow"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = ConsistencyFlowPolicy(
            hidden_dims=[256, 256],
            num_inference_steps=5,
            flow_batch_ratio=0.7,
            consistency_batch_ratio=0.3,
            **common_kwargs
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = eval_agent_wrapper(agent, algo_name)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = get_batch(idx)
                agent.train_step(batch)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 4. Reflected Flow ====================
    print("\n4. Training Reflected Flow with progress tracking...")
    algo_name = "Reflected Flow"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = ReflectedFlowPolicy(
            hidden_dims=[256, 256],
            num_inference_steps=20,
            reflection_mode="hard",
            **common_kwargs
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = eval_agent_wrapper(agent, algo_name)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = get_batch(idx)
                agent.train_step(batch)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 5. Diffusion QL ====================
    print("\n5. Training Diffusion QL with progress tracking...")
    algo_name = "Diffusion QL"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = DiffusionDoubleQAgent(
            hidden_dims=[256, 256],
            num_diffusion_steps=20,
            **common_kwargs
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = eval_agent_wrapper(agent, algo_name)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = get_batch(idx)
                agent.train_step(batch)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 6. CPQL ====================
    print("\n6. Training CPQL with progress tracking...")
    algo_name = "CPQL"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = CPQLAgent(
            hidden_dims=[256, 256],
            sigma_max=80.0,
            sigma_min=0.002,
            rho=7.0,
            **common_kwargs
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = eval_agent_wrapper(agent, algo_name)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = get_batch(idx)
                agent.train_step(batch)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 7. DPPO ====================
    print("\n7. Training DPPO with progress tracking...")
    algo_name = "DPPO"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = DPPOAgent(
            hidden_dims=[256, 256],
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
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = eval_agent_wrapper(agent, algo_name, use_pretrain=True)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = get_batch(idx)
                
                batch_actions = batch["actions"]
                
                # Get observation features
                state_input = batch.get("states")
                image_input = batch.get("images")
                obs_features = agent.obs_encoder(state=state_input, image=image_input)
                
                # Diffusion BC loss
                t = torch.randint(0, agent.num_diffusion_steps, (batch_size,), device=device)
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
    print("\n8. Training ReinFlow with progress tracking...")
    algo_name = "ReinFlow"
    progress_results[algo_name] = {}
    emd_history[algo_name] = {}
    
    try:
        agent = ReinFlowAgent(
            hidden_dims=[256, 256],
            num_flow_steps=10,
            noise_scheduler_type="learn",
            **common_kwargs
        )
        
        # Create optimizer for pretrain (use base_net, not velocity_net)
        pretrain_optimizer = torch.optim.Adam(
            agent.policy.base_net.parameters(), lr=1e-4
        )
        
        for step in range(num_steps + 1):
            if step in eval_checkpoints:
                samples, emd = eval_agent_wrapper(agent, algo_name, use_pretrain=True)
                progress_results[algo_name][step] = samples
                emd_history[algo_name][step] = emd
                print(f"  Step {step}: EMD = {emd:.4f}")
            
            if step < num_steps:
                idx = np.random.randint(0, n_samples, batch_size)
                batch = get_batch(idx)
                
                batch_actions = batch["actions"]
                
                # Get observation features
                state_input = batch.get("states")
                image_input = batch.get("images")
                obs_features = agent.obs_encoder(state=state_input, image=image_input)
                
                # Flow matching loss (use base_net, not velocity_net)
                x_0 = torch.randn_like(batch_actions)
                t = torch.rand(batch_size, device=device)
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
    
    return progress_results, emd_history


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
    """
    # Filter algorithms that have results
    algorithms = [name for name in progress_results.keys() if progress_results[name]]
    n_algorithms = len(algorithms)
    
    if n_algorithms == 0:
        print("No results to visualize")
        return
    
    # Get all checkpoints (steps) from first algorithm with results
    sample_algo = algorithms[0]
    checkpoints = sorted(progress_results[sample_algo].keys())
    n_checkpoints = len(checkpoints)
    
    if n_checkpoints == 0:
        print("No checkpoints to visualize")
        return
    
    # Create figure
    fig, axes = plt.subplots(
        n_algorithms, n_checkpoints + 1,
        figsize=(2.2 * (n_checkpoints + 1), 2.5 * n_algorithms)
    )
    
    if n_algorithms == 1:
        axes = axes.reshape(1, -1)
    
    # Get target samples
    target_samples = env.sample_from_distribution(1000)
    
    # Color scheme
    sample_color = '#1f77b4'
    target_color = '#ff7f0e'
    
    for row_idx, algo_name in enumerate(algorithms):
        algo_progress = progress_results[algo_name]
        algo_emd = emd_history[algo_name]
        
        for col_idx, step in enumerate(checkpoints):
            ax = axes[row_idx, col_idx]
            
            if step in algo_progress:
                samples = algo_progress[step]
                samples_scaled = samples * env.scale * 1.5
                
                ax.scatter(samples_scaled[:, 0], samples_scaled[:, 1],
                          alpha=0.5, s=3, c=sample_color, rasterized=True)
                
                emd_val = algo_emd.get(step, 0)
                
                if row_idx == 0:
                    ax.set_title(f'Step={step}', fontsize=10)
                    
                ax.text(0.5, -0.15, f'EMD={emd_val:.4f}', 
                       transform=ax.transAxes, ha='center', fontsize=8)
            
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
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


def run_progress_validation(
    distribution_type: str = "clusters",
    obs_mode: str = "state",
    num_modes: int = 8,
    num_samples: int = 5000,
    num_train_steps: int = 2000,
    device: str = "cpu",
    output_dir: str = "./results",
    num_checkpoints: int = 6
):
    """Run validation with training progress visualization.
    
    Args:
        distribution_type: Type of target distribution
        obs_mode: "state", "image", or "state_image"
        num_modes: Number of modes in distribution
        num_samples: Number of training samples
        num_train_steps: Number of training steps
        device: Device to use
        output_dir: Output directory
        num_checkpoints: Number of evaluation checkpoints
    """
    print(f"\n{'='*60}")
    print(f"Training Progress Validation on: {distribution_type} distribution")
    print(f"Observation mode: {obs_mode}")
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
    image_shape = (64, 64, 3)  # Fixed image shape for synthetic images
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Image shape: {image_shape}")
    
    # Create offline dataset
    print("\nCollecting offline dataset...")
    dataset = create_offline_dataset(env, num_samples=num_samples)
    print(f"Dataset size: {len(dataset['states'])} samples")
    
    # Define checkpoints
    checkpoints = [int(i * num_train_steps / (num_checkpoints - 1)) for i in range(num_checkpoints)]
    checkpoints = sorted(set(checkpoints))
    print(f"Evaluation checkpoints: {checkpoints}")
    
    # Train with progress tracking
    progress_results, emd_history = train_with_progress_tracking(
        env=env,
        dataset=dataset,
        state_dim=state_dim,
        action_dim=action_dim,
        obs_mode=obs_mode,
        image_shape=image_shape,
        num_steps=num_train_steps,
        device=device,
        eval_checkpoints=checkpoints
    )
    
    # Create visualization
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    progress_save_path = os.path.join(
        output_dir, 
        f"progress_multimodal_{distribution_type}_{obs_mode}_{timestamp}.png"
    )
    visualize_training_progress(
        env=env,
        progress_results=progress_results,
        emd_history=emd_history,
        save_path=progress_save_path,
        title=f"Training Progress on {distribution_type.title()} ({obs_mode} mode)"
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("Final Evaluation Metrics")
    print(f"{'='*60}")
    print(f"{'Algorithm':<20} {'Final EMD':>12}")
    print("-" * 34)
    
    for name in progress_results:
        if emd_history[name]:
            final_step = max(emd_history[name].keys())
            final_emd = emd_history[name][final_step]
            print(f"{name:<20} {final_emd:>12.4f}")
        else:
            print(f"{name:<20} {'N/A':>12}")
    
    return progress_results, emd_history


def main():
    parser = argparse.ArgumentParser(description="Validate all unified diffusion/flow RL algorithms")
    parser.add_argument("--distribution", type=str, default="clusters",
                       choices=["clusters", "ring", "double_ring", "spiral", "moons", "grid"],
                       help="Type of target distribution")
    parser.add_argument("--obs_mode", type=str, default="state",
                       choices=["state", "image", "state_image"],
                       help="Observation mode")
    parser.add_argument("--num_modes", type=int, default=8, help="Number of modes")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--num_steps", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Run on all distribution types")
    parser.add_argument("--all_obs_modes", action="store_true", help="Run on all observation modes")
    parser.add_argument("--num_checkpoints", type=int, default=6,
                       help="Number of checkpoints for progress visualization")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print(f"Using device: {args.device}")
    
    distributions = ["clusters", "ring", "double_ring", "spiral", "moons", "grid"] if args.all else [args.distribution]
    obs_modes = ["state", "image", "state_image"] if args.all_obs_modes else [args.obs_mode]
    
    for dist in distributions:
        for obs_mode in obs_modes:
            run_progress_validation(
                distribution_type=dist,
                obs_mode=obs_mode,
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
