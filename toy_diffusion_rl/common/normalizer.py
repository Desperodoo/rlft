"""
Data Normalizers for Diffusion/Flow Policies.

This module provides normalizers for actions and observations, similar to
the LinearNormalizer in Diffusion Policy. Key features:
- Fit normalizers to dataset statistics
- Normalize inputs to [-1, 1] range (limits mode) or zero mean, unit std (gaussian mode)
- Unnormalize outputs for environment interaction
- Save/load normalizer state

Reference: 
    https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/common/normalizer.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union


class SingleFieldNormalizer(nn.Module):
    """Normalizer for a single field (action or observation).
    
    Supports two modes:
    - 'limits': Scale to [-1, 1] based on min/max
    - 'gaussian': Zero mean, unit std normalization
    
    Args:
        mode: Normalization mode ('limits' or 'gaussian')
        output_min: Minimum output value for limits mode
        output_max: Maximum output value for limits mode
        range_eps: Small epsilon to avoid division by zero
    """
    
    def __init__(
        self,
        mode: str = 'limits',
        output_min: float = -1.0,
        output_max: float = 1.0,
        range_eps: float = 1e-4,
    ):
        super().__init__()
        self.mode = mode
        self.output_min = output_min
        self.output_max = output_max
        self.range_eps = range_eps
        self.fitted = False
        
        # Parameters will be registered after fitting
        self.register_buffer('scale', None)
        self.register_buffer('offset', None)
        self.register_buffer('input_min', None)
        self.register_buffer('input_max', None)
        self.register_buffer('input_mean', None)
        self.register_buffer('input_std', None)
    
    @torch.no_grad()
    def fit(self, data: Union[torch.Tensor, np.ndarray]):
        """Fit normalizer to data.
        
        Args:
            data: Input data of shape (N, D) or (N,)
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        data = data.float()
        
        # Flatten if needed
        if len(data.shape) == 1:
            data = data.unsqueeze(-1)
        
        # Compute statistics
        self.input_min = data.min(dim=0)[0]
        self.input_max = data.max(dim=0)[0]
        self.input_mean = data.mean(dim=0)
        self.input_std = data.std(dim=0)
        
        # Compute scale and offset
        if self.mode == 'limits':
            input_range = self.input_max - self.input_min
            # Handle constant dimensions
            ignore_dim = input_range < self.range_eps
            input_range[ignore_dim] = self.output_max - self.output_min
            
            self.scale = (self.output_max - self.output_min) / input_range
            self.offset = self.output_min - self.scale * self.input_min
            # Center constant dimensions
            self.offset[ignore_dim] = (self.output_max + self.output_min) / 2 - self.input_min[ignore_dim]
        
        elif self.mode == 'gaussian':
            # Handle constant dimensions
            ignore_dim = self.input_std < self.range_eps
            std = self.input_std.clone()
            std[ignore_dim] = 1.0
            
            self.scale = 1.0 / std
            self.offset = -self.input_mean * self.scale
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        self.fitted = True
    
    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Normalize input data.
        
        Args:
            x: Input data of shape (..., D)
            
        Returns:
            Normalized data of same shape
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        x = x.to(dtype=self.scale.dtype, device=self.scale.device)
        return x * self.scale + self.offset
    
    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Unnormalize data back to original scale.
        
        Args:
            x: Normalized data of shape (..., D)
            
        Returns:
            Unnormalized data of same shape
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        x = x.to(dtype=self.scale.dtype, device=self.scale.device)
        return (x - self.offset) / self.scale
    
    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Alias for normalize."""
        return self.normalize(x)
    
    def get_stats(self) -> Dict[str, torch.Tensor]:
        """Get input statistics."""
        return {
            'min': self.input_min,
            'max': self.input_max,
            'mean': self.input_mean,
            'std': self.input_std,
        }


class DataNormalizer(nn.Module):
    """Multi-field normalizer for actions and observations.
    
    Handles both action and observation normalization with separate
    normalizers for each field. Can normalize dict observations with
    multiple keys (e.g., {'state': ..., 'image': ...}).
    
    Args:
        action_mode: Normalization mode for actions ('limits' or 'gaussian')
        obs_mode: Normalization mode for observations ('limits' or 'gaussian')
        normalize_images: Whether to normalize images (usually False, just scale to [0,1])
    
    Example:
        >>> normalizer = DataNormalizer()
        >>> normalizer.fit(actions=actions, states=states)
        >>> 
        >>> # Training: normalize inputs
        >>> norm_actions = normalizer.normalize_action(actions)
        >>> norm_states = normalizer.normalize_state(states)
        >>> 
        >>> # Inference: unnormalize outputs
        >>> actions = normalizer.unnormalize_action(model_output)
    """
    
    def __init__(
        self,
        action_mode: str = 'limits',
        obs_mode: str = 'limits',
        normalize_images: bool = False,
    ):
        super().__init__()
        self.action_mode = action_mode
        self.obs_mode = obs_mode
        self.normalize_images = normalize_images
        
        self.action_normalizer = SingleFieldNormalizer(mode=action_mode)
        self.state_normalizer = SingleFieldNormalizer(mode=obs_mode)
        
        self.fitted = False
    
    @torch.no_grad()
    def fit(
        self,
        actions: Optional[Union[torch.Tensor, np.ndarray]] = None,
        states: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        """Fit normalizers to data.
        
        Args:
            actions: Action data of shape (N, action_dim)
            states: State data of shape (N, state_dim)
        """
        if actions is not None:
            self.action_normalizer.fit(actions)
        
        if states is not None:
            self.state_normalizer.fit(states)
        
        self.fitted = True
    
    def fit_from_dataset(self, dataset: Dict[str, np.ndarray]):
        """Fit normalizers from a dataset dictionary.
        
        Args:
            dataset: Dictionary with 'actions' and optionally 'obs' keys
        """
        actions = dataset.get('actions')
        states = dataset.get('obs', dataset.get('states'))
        
        self.fit(actions=actions, states=states)
    
    def normalize_action(self, actions: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Normalize actions."""
        return self.action_normalizer.normalize(actions)
    
    def unnormalize_action(self, actions: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Unnormalize actions (for environment interaction)."""
        return self.action_normalizer.unnormalize(actions)
    
    def normalize_state(self, states: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Normalize states."""
        return self.state_normalizer.normalize(states)
    
    def unnormalize_state(self, states: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Unnormalize states."""
        return self.state_normalizer.unnormalize(states)
    
    def normalize_image(self, images: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Normalize images to [0, 1] range."""
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        images = images.float()
        if images.max() > 1.0:
            images = images / 255.0
        
        return images
    
    def normalize_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize a batch dictionary.
        
        Args:
            batch: Dictionary with 'actions', 'states', and optionally 'images'
            
        Returns:
            Normalized batch dictionary
        """
        result = {}
        
        if 'actions' in batch:
            result['actions'] = self.normalize_action(batch['actions'])
        
        if 'states' in batch:
            result['states'] = self.normalize_state(batch['states'])
        
        if 'next_states' in batch:
            result['next_states'] = self.normalize_state(batch['next_states'])
        
        if 'images' in batch:
            result['images'] = self.normalize_image(batch['images'])
        
        if 'next_images' in batch:
            result['next_images'] = self.normalize_image(batch['next_images'])
        
        # Copy other fields as-is
        for key in batch:
            if key not in result:
                result[key] = batch[key]
        
        return result
    
    def state_dict(self) -> Dict:
        """Get state dictionary for saving."""
        return {
            'action_normalizer': {
                'scale': self.action_normalizer.scale,
                'offset': self.action_normalizer.offset,
                'input_min': self.action_normalizer.input_min,
                'input_max': self.action_normalizer.input_max,
                'input_mean': self.action_normalizer.input_mean,
                'input_std': self.action_normalizer.input_std,
                'fitted': self.action_normalizer.fitted,
            },
            'state_normalizer': {
                'scale': self.state_normalizer.scale,
                'offset': self.state_normalizer.offset,
                'input_min': self.state_normalizer.input_min,
                'input_max': self.state_normalizer.input_max,
                'input_mean': self.state_normalizer.input_mean,
                'input_std': self.state_normalizer.input_std,
                'fitted': self.state_normalizer.fitted,
            },
            'action_mode': self.action_mode,
            'obs_mode': self.obs_mode,
            'fitted': self.fitted,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dictionary."""
        # Load action normalizer
        an = state_dict['action_normalizer']
        self.action_normalizer.scale = an['scale']
        self.action_normalizer.offset = an['offset']
        self.action_normalizer.input_min = an['input_min']
        self.action_normalizer.input_max = an['input_max']
        self.action_normalizer.input_mean = an['input_mean']
        self.action_normalizer.input_std = an['input_std']
        self.action_normalizer.fitted = an['fitted']
        
        # Load state normalizer
        sn = state_dict['state_normalizer']
        self.state_normalizer.scale = sn['scale']
        self.state_normalizer.offset = sn['offset']
        self.state_normalizer.input_min = sn['input_min']
        self.state_normalizer.input_max = sn['input_max']
        self.state_normalizer.input_mean = sn['input_mean']
        self.state_normalizer.input_std = sn['input_std']
        self.state_normalizer.fitted = sn['fitted']
        
        self.action_mode = state_dict.get('action_mode', 'limits')
        self.obs_mode = state_dict.get('obs_mode', 'limits')
        self.fitted = state_dict.get('fitted', True)
    
    def to(self, device):
        """Move normalizer to device."""
        super().to(device)
        return self
    
    def get_action_stats(self) -> Dict[str, torch.Tensor]:
        """Get action normalization statistics."""
        return self.action_normalizer.get_stats()
    
    def get_state_stats(self) -> Dict[str, torch.Tensor]:
        """Get state normalization statistics."""
        return self.state_normalizer.get_stats()


def create_normalizer_from_dataset(
    dataset: Dict[str, np.ndarray],
    action_mode: str = 'limits',
    obs_mode: str = 'limits',
) -> DataNormalizer:
    """Factory function to create and fit a normalizer from dataset.
    
    Args:
        dataset: Dataset dictionary with 'actions' and 'obs' keys
        action_mode: Normalization mode for actions
        obs_mode: Normalization mode for observations
        
    Returns:
        Fitted DataNormalizer
    """
    normalizer = DataNormalizer(action_mode=action_mode, obs_mode=obs_mode)
    normalizer.fit_from_dataset(dataset)
    return normalizer


def create_normalizer_from_env(
    env,
    obs_mode: str = 'limits',
) -> DataNormalizer:
    """Create normalizer using environment action space bounds.
    
    This is the recommended approach for ManiSkill environments, as it uses
    the true action space bounds instead of data statistics.
    
    The action normalizer maps:
        action_space.low -> -1
        action_space.high -> +1
    
    Reference:
        ManiSkill official diffusion policy uses environment-level normalization:
        https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/diffusion_policy/train.py
    
    Args:
        env: Gymnasium environment with action_space attribute
        obs_mode: Normalization mode for observations ('limits' or 'gaussian')
        
    Returns:
        DataNormalizer with action normalizer fitted to env bounds
    """
    normalizer = DataNormalizer(action_mode='limits', obs_mode=obs_mode)
    
    # Get action space bounds
    action_low = env.action_space.low
    action_high = env.action_space.high
    
    if isinstance(action_low, np.ndarray):
        action_low = action_low.astype(np.float32)
        action_high = action_high.astype(np.float32)
    
    # Verify action space is bounded
    if np.any(np.isinf(action_low)) or np.any(np.isinf(action_high)):
        raise ValueError(
            "Environment action space is unbounded. "
            "Cannot use environment-level normalization."
        )
    
    # Fit action normalizer to environment bounds
    # Create dummy 2D array for fit (min and max values)
    dummy_actions = np.stack([action_low, action_high], axis=0)
    normalizer.action_normalizer.fit(dummy_actions)
    
    return normalizer


def normalize_action_with_env_bounds(
    action: Union[torch.Tensor, np.ndarray],
    action_low: Union[torch.Tensor, np.ndarray],
    action_high: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Normalize action from [low, high] to [-1, 1].
    
    This is a simple utility function for direct normalization without
    creating a full normalizer object.
    
    Args:
        action: Action to normalize
        action_low: Lower bounds of action space
        action_high: Upper bounds of action space
        
    Returns:
        Normalized action in [-1, 1]
    """
    if isinstance(action, np.ndarray):
        action = torch.from_numpy(action)
    if isinstance(action_low, np.ndarray):
        action_low = torch.from_numpy(action_low)
    if isinstance(action_high, np.ndarray):
        action_high = torch.from_numpy(action_high)
    
    # Normalize: (action - low) / (high - low) * 2 - 1
    # Maps low -> -1, high -> +1
    scale = 2.0 / (action_high - action_low + 1e-8)
    offset = -1.0 - scale * action_low
    
    return action * scale + offset


def unnormalize_action_with_env_bounds(
    action: Union[torch.Tensor, np.ndarray],
    action_low: Union[torch.Tensor, np.ndarray],
    action_high: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Unnormalize action from [-1, 1] to [low, high].
    
    Args:
        action: Normalized action in [-1, 1]
        action_low: Lower bounds of action space
        action_high: Upper bounds of action space
        
    Returns:
        Action in [low, high]
    """
    if isinstance(action, np.ndarray):
        action = torch.from_numpy(action)
    if isinstance(action_low, np.ndarray):
        action_low = torch.from_numpy(action_low)
    if isinstance(action_high, np.ndarray):
        action_high = torch.from_numpy(action_high)
    
    # Unnormalize: (action + 1) / 2 * (high - low) + low
    # Maps -1 -> low, +1 -> high
    return (action + 1.0) / 2.0 * (action_high - action_low) + action_low


if __name__ == "__main__":
    print("Testing DataNormalizer...")
    
    # Create dummy data
    np.random.seed(42)
    actions = np.random.uniform(-0.5, 0.5, (1000, 7)).astype(np.float32)
    states = np.random.uniform(-10, 10, (1000, 42)).astype(np.float32)
    
    # Test fitting
    normalizer = DataNormalizer(action_mode='limits', obs_mode='limits')
    normalizer.fit(actions=actions, states=states)
    
    print("Action stats:", normalizer.get_action_stats())
    print("State stats:", normalizer.get_state_stats())
    
    # Test normalization
    norm_actions = normalizer.normalize_action(actions)
    print(f"\nOriginal actions: min={actions.min():.3f}, max={actions.max():.3f}")
    print(f"Normalized actions: min={norm_actions.min():.3f}, max={norm_actions.max():.3f}")
    
    # Test unnormalization
    unnorm_actions = normalizer.unnormalize_action(norm_actions)
    print(f"Unnormalized actions: min={unnorm_actions.min():.3f}, max={unnorm_actions.max():.3f}")
    
    # Verify round-trip
    diff = torch.abs(unnorm_actions - torch.from_numpy(actions)).max()
    print(f"Round-trip error: {diff:.6f}")
    
    # Test batch normalization
    batch = {
        'actions': torch.from_numpy(actions[:32]),
        'states': torch.from_numpy(states[:32]),
    }
    norm_batch = normalizer.normalize_batch(batch)
    print(f"\nBatch normalized - actions: [{norm_batch['actions'].min():.2f}, {norm_batch['actions'].max():.2f}]")
    print(f"Batch normalized - states: [{norm_batch['states'].min():.2f}, {norm_batch['states'].max():.2f}]")
    
    # Test save/load
    state = normalizer.state_dict()
    new_normalizer = DataNormalizer()
    new_normalizer.load_state_dict(state)
    
    norm_actions2 = new_normalizer.normalize_action(actions)
    diff = torch.abs(norm_actions - norm_actions2).max()
    print(f"\nSave/load error: {diff:.6f}")
    
    print("\n✓ All normalizer tests passed!")
