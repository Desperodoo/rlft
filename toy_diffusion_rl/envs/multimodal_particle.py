"""
Multimodal Particle Distribution Environment

A toy environment for validating diffusion/flow-matching policies.
The task is to generate samples from a target multi-modal distribution
(clusters, rings, spirals, etc.).

This environment supports:
1. Offline training: Generate expert data from target distribution
2. Online fine-tuning: Use rewards based on sample quality/coverage

The simple 2D setting allows easy visualization of learned distributions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List, Literal
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io
from PIL import Image


class MultimodalParticleEnv(gym.Env):
    """Multimodal Particle Distribution Environment.
    
    The agent learns to generate samples from a target multi-modal distribution.
    
    State: Context/condition for conditional generation (can be fixed or varied)
    Action: Generated 2D point
    Reward: Based on how well the point matches the target distribution
    
    Supported distributions:
    - 'clusters': Multiple Gaussian clusters
    - 'ring': Points on a ring/circle
    - 'double_ring': Two concentric rings
    - 'spiral': Spiral pattern
    - 'moons': Two crescent moons (sklearn-style)
    - 'grid': Regular grid of clusters
    - 'mixture': GMM with specified means and covariances
    
    Args:
        distribution_type: Type of target distribution
        num_modes: Number of modes/clusters (for applicable types)
        scale: Overall scale of the distribution
        noise_std: Noise standard deviation for samples
        context_dim: Dimension of context vector (0 for unconditional)
        reward_type: 'density', 'distance', or 'coverage'
        max_episode_steps: Max steps per episode
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    DISTRIBUTION_TYPES = [
        'clusters', 'ring', 'double_ring', 'spiral', 
        'moons', 'grid', 'mixture', 'checkerboard'
    ]
    
    def __init__(
        self,
        distribution_type: str = "clusters",
        num_modes: int = 8,
        scale: float = 2.0,
        noise_std: float = 0.1,
        context_dim: int = 0,
        context_type: str = "fixed",  # 'fixed', 'random', 'mode_id'
        reward_type: str = "density",  # 'density', 'distance', 'coverage'
        max_episode_steps: int = 1,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        assert distribution_type in self.DISTRIBUTION_TYPES, \
            f"Unknown distribution: {distribution_type}. Choose from {self.DISTRIBUTION_TYPES}"
        
        self.distribution_type = distribution_type
        self.num_modes = num_modes
        self.scale = scale
        self.noise_std = noise_std
        self.context_dim = context_dim
        self.context_type = context_type
        self.reward_type = reward_type
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Set random seed
        self.np_random = np.random.default_rng(seed)
        
        # Action: 2D point to generate
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation: context vector
        if context_dim > 0:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(context_dim,),
                dtype=np.float32
            )
        else:
            # Dummy observation for unconditional generation
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        
        # Setup distribution parameters
        self._setup_distribution()
        
        self.current_context = None
        self.current_target_mode = None
        self.steps = 0
        self.generated_samples = []
        
    def _setup_distribution(self):
        """Setup parameters for the target distribution."""
        if self.distribution_type == "clusters":
            # Clusters arranged in a circle
            angles = np.linspace(0, 2 * np.pi, self.num_modes, endpoint=False)
            self.mode_centers = np.stack([
                self.scale * np.cos(angles),
                self.scale * np.sin(angles)
            ], axis=1)
            self.mode_stds = np.ones(self.num_modes) * self.noise_std
            
        elif self.distribution_type == "ring":
            # Single ring
            self.ring_radius = self.scale
            self.ring_width = self.noise_std
            
        elif self.distribution_type == "double_ring":
            # Two concentric rings
            self.inner_radius = self.scale * 0.5
            self.outer_radius = self.scale
            self.ring_width = self.noise_std
            
        elif self.distribution_type == "spiral":
            # Spiral parameters
            self.spiral_turns = 2.0
            self.spiral_scale = self.scale
            
        elif self.distribution_type == "moons":
            # Two crescent moons
            self.moon_radius = self.scale
            self.moon_noise = self.noise_std
            
        elif self.distribution_type == "grid":
            # Grid of clusters
            grid_size = int(np.sqrt(self.num_modes))
            x = np.linspace(-self.scale, self.scale, grid_size)
            y = np.linspace(-self.scale, self.scale, grid_size)
            xx, yy = np.meshgrid(x, y)
            self.mode_centers = np.stack([xx.flatten(), yy.flatten()], axis=1)
            self.num_modes = len(self.mode_centers)
            self.mode_stds = np.ones(self.num_modes) * self.noise_std
            
        elif self.distribution_type == "mixture":
            # Custom GMM (default: 4 clusters with different shapes)
            self.mode_centers = np.array([
                [self.scale, self.scale],
                [-self.scale, self.scale],
                [-self.scale, -self.scale],
                [self.scale, -self.scale]
            ])
            self.mode_covs = [
                np.array([[0.2, 0.1], [0.1, 0.2]]) * self.noise_std**2,
                np.array([[0.3, 0.0], [0.0, 0.1]]) * self.noise_std**2,
                np.array([[0.1, 0.0], [0.0, 0.3]]) * self.noise_std**2,
                np.array([[0.2, -0.1], [-0.1, 0.2]]) * self.noise_std**2,
            ]
            self.num_modes = 4
            
        elif self.distribution_type == "checkerboard":
            # Checkerboard pattern
            self.board_size = 4
            self.cell_size = self.scale / self.board_size
            
    def sample_from_distribution(
        self, 
        n_samples: int = 1,
        mode_idx: Optional[int] = None
    ) -> np.ndarray:
        """Sample points from the target distribution.
        
        Args:
            n_samples: Number of samples to generate
            mode_idx: If specified, sample only from this mode
            
        Returns:
            Array of shape (n_samples, 2)
        """
        if self.distribution_type == "clusters":
            if mode_idx is not None:
                centers = self.mode_centers[mode_idx:mode_idx+1]
                idx = np.zeros(n_samples, dtype=int)
            else:
                idx = self.np_random.integers(0, self.num_modes, n_samples)
                centers = self.mode_centers[idx]
            noise = self.np_random.normal(0, self.noise_std, (n_samples, 2))
            return (centers + noise).astype(np.float32)
            
        elif self.distribution_type == "ring":
            angles = self.np_random.uniform(0, 2 * np.pi, n_samples)
            radii = self.ring_radius + self.np_random.normal(0, self.ring_width, n_samples)
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            return np.stack([x, y], axis=1).astype(np.float32)
            
        elif self.distribution_type == "double_ring":
            # Sample from inner or outer ring
            is_outer = self.np_random.random(n_samples) > 0.5
            radii = np.where(is_outer, self.outer_radius, self.inner_radius)
            radii = radii + self.np_random.normal(0, self.ring_width, n_samples)
            angles = self.np_random.uniform(0, 2 * np.pi, n_samples)
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            return np.stack([x, y], axis=1).astype(np.float32)
            
        elif self.distribution_type == "spiral":
            t = self.np_random.uniform(0, 1, n_samples)
            angle = t * self.spiral_turns * 2 * np.pi
            radius = t * self.spiral_scale
            noise = self.np_random.normal(0, self.noise_std, (n_samples, 2))
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            return (np.stack([x, y], axis=1) + noise).astype(np.float32)
            
        elif self.distribution_type == "moons":
            # Upper moon
            n_upper = n_samples // 2
            angles_upper = self.np_random.uniform(0, np.pi, n_upper)
            x_upper = self.moon_radius * np.cos(angles_upper)
            y_upper = self.moon_radius * np.sin(angles_upper)
            
            # Lower moon (shifted and flipped)
            n_lower = n_samples - n_upper
            angles_lower = self.np_random.uniform(0, np.pi, n_lower)
            x_lower = self.moon_radius * np.cos(angles_lower) + self.moon_radius
            y_lower = -self.moon_radius * np.sin(angles_lower) + 0.5
            
            x = np.concatenate([x_upper, x_lower])
            y = np.concatenate([y_upper, y_lower])
            noise = self.np_random.normal(0, self.moon_noise, (n_samples, 2))
            return (np.stack([x, y], axis=1) + noise).astype(np.float32)
            
        elif self.distribution_type == "grid":
            if mode_idx is not None:
                centers = self.mode_centers[mode_idx:mode_idx+1]
                idx = np.zeros(n_samples, dtype=int)
            else:
                idx = self.np_random.integers(0, self.num_modes, n_samples)
                centers = self.mode_centers[idx]
            noise = self.np_random.normal(0, self.noise_std, (n_samples, 2))
            return (centers + noise).astype(np.float32)
            
        elif self.distribution_type == "mixture":
            samples = []
            for _ in range(n_samples):
                if mode_idx is not None:
                    idx = mode_idx
                else:
                    idx = self.np_random.integers(0, self.num_modes)
                sample = self.np_random.multivariate_normal(
                    self.mode_centers[idx], self.mode_covs[idx]
                )
                samples.append(sample)
            return np.array(samples, dtype=np.float32)
            
        elif self.distribution_type == "checkerboard":
            samples = []
            for _ in range(n_samples):
                while True:
                    x = self.np_random.uniform(-self.scale, self.scale)
                    y = self.np_random.uniform(-self.scale, self.scale)
                    # Determine cell
                    cx = int((x + self.scale) / (2 * self.scale) * self.board_size)
                    cy = int((y + self.scale) / (2 * self.scale) * self.board_size)
                    # Checkerboard pattern
                    if (cx + cy) % 2 == 0:
                        samples.append([x, y])
                        break
            return np.array(samples, dtype=np.float32)
            
    def compute_log_density(self, points: np.ndarray) -> np.ndarray:
        """Compute log density of points under the target distribution.
        
        Args:
            points: Array of shape (n, 2)
            
        Returns:
            Array of shape (n,) with log densities
        """
        points = np.atleast_2d(points)
        n = len(points)
        
        if self.distribution_type in ["clusters", "grid"]:
            # GMM density
            log_probs = []
            for i in range(self.num_modes):
                diff = points - self.mode_centers[i]
                log_prob = -0.5 * np.sum(diff**2, axis=1) / (self.noise_std**2)
                log_prob -= np.log(2 * np.pi * self.noise_std**2)
                log_probs.append(log_prob)
            log_probs = np.array(log_probs)  # (num_modes, n)
            # Log-sum-exp for mixture
            max_log = np.max(log_probs, axis=0)
            log_density = max_log + np.log(np.sum(np.exp(log_probs - max_log), axis=0))
            log_density -= np.log(self.num_modes)  # Uniform mixture weights
            return log_density
            
        elif self.distribution_type == "ring":
            radii = np.linalg.norm(points, axis=1)
            diff = radii - self.ring_radius
            log_density = -0.5 * diff**2 / (self.ring_width**2)
            return log_density
            
        elif self.distribution_type == "double_ring":
            radii = np.linalg.norm(points, axis=1)
            diff_inner = radii - self.inner_radius
            diff_outer = radii - self.outer_radius
            log_prob_inner = -0.5 * diff_inner**2 / (self.ring_width**2)
            log_prob_outer = -0.5 * diff_outer**2 / (self.ring_width**2)
            # Log-sum-exp
            max_log = np.maximum(log_prob_inner, log_prob_outer)
            log_density = max_log + np.log(np.exp(log_prob_inner - max_log) + np.exp(log_prob_outer - max_log))
            return log_density
            
        else:
            # For complex distributions, use kernel density estimation
            # or return a simpler proxy
            samples = self.sample_from_distribution(1000)
            distances = np.linalg.norm(points[:, None] - samples[None, :], axis=2)
            min_distances = np.min(distances, axis=1)
            return -min_distances  # Negative distance as proxy for density
            
    def compute_distance_to_modes(self, points: np.ndarray) -> np.ndarray:
        """Compute minimum distance from points to mode centers.
        
        Args:
            points: Array of shape (n, 2)
            
        Returns:
            Array of shape (n,) with distances
        """
        points = np.atleast_2d(points)
        
        if hasattr(self, 'mode_centers'):
            distances = np.linalg.norm(
                points[:, None] - self.mode_centers[None, :], axis=2
            )
            return np.min(distances, axis=1)
        else:
            # For non-cluster distributions, sample reference points
            ref_samples = self.sample_from_distribution(100)
            distances = np.linalg.norm(
                points[:, None] - ref_samples[None, :], axis=2
            )
            return np.min(distances, axis=1)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.
        
        Args:
            seed: Random seed
            options: Can contain 'mode_idx' for conditional generation
            
        Returns:
            observation: Context vector
            info: Additional information
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            
        self.steps = 0
        self.generated_samples = []
        
        # Setup context
        if options and "mode_idx" in options:
            self.current_target_mode = options["mode_idx"]
        else:
            self.current_target_mode = self.np_random.integers(0, self.num_modes) \
                if hasattr(self, 'mode_centers') else None
        
        if self.context_dim > 0:
            if self.context_type == "mode_id" and self.current_target_mode is not None:
                # One-hot encoding of mode
                self.current_context = np.zeros(self.context_dim, dtype=np.float32)
                if self.current_target_mode < self.context_dim:
                    self.current_context[self.current_target_mode] = 1.0
            elif self.context_type == "random":
                self.current_context = self.np_random.normal(0, 1, self.context_dim).astype(np.float32)
            else:
                self.current_context = np.zeros(self.context_dim, dtype=np.float32)
        else:
            self.current_context = np.zeros(1, dtype=np.float32)
            
        info = {
            "target_mode": self.current_target_mode,
            "distribution_type": self.distribution_type
        }
        
        return self.current_context.copy(), info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step (generate a sample).
        
        Args:
            action: Generated 2D point (in [-1, 1], will be scaled)
            
        Returns:
            observation: Same context (unchanged)
            reward: Quality of the generated sample
            terminated: Always False (one-step episode)
            truncated: True after max_episode_steps
            info: Additional information
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Scale action to distribution range
        scaled_action = action * self.scale * 1.5
        
        self.generated_samples.append(scaled_action)
        self.steps += 1
        
        # Compute reward
        if self.reward_type == "density":
            log_density = self.compute_log_density(scaled_action.reshape(1, -1))[0]
            reward = float(log_density)
        elif self.reward_type == "distance":
            distance = self.compute_distance_to_modes(scaled_action.reshape(1, -1))[0]
            reward = -float(distance)
        else:  # coverage
            reward = 0.0
            
        terminated = False
        truncated = self.steps >= self.max_episode_steps
        
        info = {
            "generated_point": scaled_action,
            "reward_type": self.reward_type
        }
        
        return self.current_context.copy(), reward, terminated, truncated, info
    
    def render(self):
        """Render the current state."""
        if self.render_mode is None:
            return None
            
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot target distribution samples
        target_samples = self.sample_from_distribution(500)
        ax.scatter(
            target_samples[:, 0], target_samples[:, 1],
            alpha=0.3, s=10, c='blue', label='Target'
        )
        
        # Plot generated samples
        if len(self.generated_samples) > 0:
            gen_samples = np.array(self.generated_samples)
            ax.scatter(
                gen_samples[:, 0], gen_samples[:, 1],
                alpha=0.7, s=30, c='red', marker='x', label='Generated'
            )
        
        ax.set_xlim(-self.scale * 2, self.scale * 2)
        ax.set_ylim(-self.scale * 2, self.scale * 2)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f'Distribution: {self.distribution_type}')
        ax.grid(True, alpha=0.3)
        
        if self.render_mode == "human":
            plt.show()
            return None
        else:  # rgb_array
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            return np.array(img)
    
    # ==================== Offline Data Generation ====================
    
    def collect_offline_dataset(
        self,
        num_samples: int = 10000,
        include_conditions: bool = True
    ) -> Dict[str, np.ndarray]:
        """Collect offline dataset for training.
        
        For diffusion/flow models, we just need samples from the target distribution.
        For conditional models, we also provide conditioning information.
        
        Args:
            num_samples: Number of samples to collect
            include_conditions: Whether to include conditioning info
            
        Returns:
            Dataset dictionary
        """
        # Sample from target distribution
        actions = self.sample_from_distribution(num_samples)
        
        # Normalize to [-1, 1] for action space
        actions_normalized = actions / (self.scale * 1.5)
        actions_normalized = np.clip(actions_normalized, -1.0, 1.0)
        
        # Generate states/conditions
        if self.context_dim > 0 and include_conditions:
            if self.context_type == "mode_id" and hasattr(self, 'mode_centers'):
                # Assign each sample to nearest mode
                distances = np.linalg.norm(
                    actions[:, None] - self.mode_centers[None, :], axis=2
                )
                mode_ids = np.argmin(distances, axis=1)
                states = np.zeros((num_samples, self.context_dim), dtype=np.float32)
                for i, mode_id in enumerate(mode_ids):
                    if mode_id < self.context_dim:
                        states[i, mode_id] = 1.0
            else:
                states = np.zeros((num_samples, self.context_dim), dtype=np.float32)
        else:
            states = np.zeros((num_samples, max(1, self.context_dim)), dtype=np.float32)
        
        # For diffusion training, we mainly need states and actions
        # We can add dummy rewards/next_states for compatibility
        return {
            "states": states.astype(np.float32),
            "actions": actions_normalized.astype(np.float32),
            "actions_unnormalized": actions.astype(np.float32),
            "rewards": np.zeros(num_samples, dtype=np.float32),
            "next_states": states.astype(np.float32),
            "dones": np.ones(num_samples, dtype=np.float32)
        }
    
    def collect_expert_dataset(
        self,
        num_episodes: int = 100,
        noise_std: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """Collect expert dataset (alias for collect_offline_dataset).
        
        Args:
            num_episodes: Treated as num_samples
            noise_std: Additional noise (ignored for this env)
            
        Returns:
            Dataset dictionary
        """
        return self.collect_offline_dataset(num_samples=num_episodes * 10)
    
    # ==================== Visualization Utilities ====================
    
    def visualize_distribution(
        self,
        generated_samples: Optional[np.ndarray] = None,
        title: str = "",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[np.ndarray]:
        """Visualize target and generated distributions.
        
        Args:
            generated_samples: Optional array of generated samples (n, 2)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            RGB array if not showing
        """
        fig, axes = plt.subplots(1, 2 if generated_samples is not None else 1, 
                                  figsize=(12, 5) if generated_samples is not None else (6, 5))
        
        if generated_samples is None:
            axes = [axes]
        
        # Plot target distribution
        target_samples = self.sample_from_distribution(1000)
        axes[0].scatter(
            target_samples[:, 0], target_samples[:, 1],
            alpha=0.5, s=5, c='blue'
        )
        axes[0].set_xlim(-self.scale * 2, self.scale * 2)
        axes[0].set_ylim(-self.scale * 2, self.scale * 2)
        axes[0].set_aspect('equal')
        axes[0].set_title(f'Target: {self.distribution_type}')
        axes[0].grid(True, alpha=0.3)
        
        # Plot generated distribution
        if generated_samples is not None:
            axes[1].scatter(
                generated_samples[:, 0], generated_samples[:, 1],
                alpha=0.5, s=5, c='red'
            )
            axes[1].set_xlim(-self.scale * 2, self.scale * 2)
            axes[1].set_ylim(-self.scale * 2, self.scale * 2)
            axes[1].set_aspect('equal')
            axes[1].set_title('Generated')
            axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        if show:
            plt.show()
            return None
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            return np.array(img)
    
    def evaluate_samples(
        self,
        generated_samples: np.ndarray,
        num_reference: int = 1000
    ) -> Dict[str, float]:
        """Evaluate quality of generated samples.
        
        Computes various metrics:
        - Mean log density
        - Mode coverage (how many modes are covered)
        - MMD (Maximum Mean Discrepancy) to target
        
        Args:
            generated_samples: Array of shape (n, 2)
            num_reference: Number of reference samples from target
            
        Returns:
            Dictionary of evaluation metrics
        """
        generated_samples = np.atleast_2d(generated_samples)
        n_gen = len(generated_samples)
        
        # Reference samples
        ref_samples = self.sample_from_distribution(num_reference)
        
        # Mean log density
        log_densities = self.compute_log_density(generated_samples)
        mean_log_density = np.mean(log_densities)
        
        # Mode coverage (for cluster-based distributions)
        if hasattr(self, 'mode_centers'):
            # Check which modes are covered
            distances = np.linalg.norm(
                generated_samples[:, None] - self.mode_centers[None, :], axis=2
            )
            nearest_modes = np.argmin(distances, axis=1)
            covered_modes = len(np.unique(nearest_modes))
            mode_coverage = covered_modes / self.num_modes
        else:
            mode_coverage = 1.0
            
        # MMD with RBF kernel
        def rbf_kernel(x, y, sigma=1.0):
            dist = np.sum((x[:, None] - y[None, :]) ** 2, axis=2)
            return np.exp(-dist / (2 * sigma ** 2))
        
        sigma = self.scale / 2
        k_xx = rbf_kernel(generated_samples, generated_samples, sigma)
        k_yy = rbf_kernel(ref_samples, ref_samples, sigma)
        k_xy = rbf_kernel(generated_samples, ref_samples, sigma)
        
        mmd = np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)
        
        # Wasserstein distance approximation (using sorted 1D projections)
        def wasserstein_1d(x, y):
            return np.mean(np.abs(np.sort(x) - np.sort(y[:len(x)])))
        
        w_x = wasserstein_1d(generated_samples[:, 0], ref_samples[:, 0])
        w_y = wasserstein_1d(generated_samples[:, 1], ref_samples[:, 1])
        wasserstein_approx = (w_x + w_y) / 2
        
        return {
            "mean_log_density": float(mean_log_density),
            "mode_coverage": float(mode_coverage),
            "mmd": float(mmd),
            "wasserstein_approx": float(wasserstein_approx)
        }


# ==================== Convenience Functions ====================

def make_multimodal_env(
    distribution: str = "clusters",
    num_modes: int = 8,
    **kwargs
) -> MultimodalParticleEnv:
    """Create a multimodal particle environment.
    
    Args:
        distribution: Type of distribution
        num_modes: Number of modes
        **kwargs: Additional arguments
        
    Returns:
        Environment instance
    """
    return MultimodalParticleEnv(
        distribution_type=distribution,
        num_modes=num_modes,
        **kwargs
    )
