"""
Neural Network Modules for Diffusion/Flow RL

Contains:
- MLP: Basic multi-layer perceptron
- TimestepEmbedding: Sinusoidal timestep encoding
- DiffusionNoisePredictor: Noise prediction network for diffusion
- FlowVelocityPredictor: Velocity field network for flow matching
- QNetwork: Q-value network for RL
- ValueNetwork: State value network
- DoubleQNetwork: Twin Q-networks for TD3/SAC style algorithms

Vision-enabled variants:
- VisionDiffusionNoisePredictor: Supports optional image inputs
- VisionFlowVelocityPredictor: Supports optional image inputs
- VisionValueNetwork: Supports optional image inputs
- VisionQNetwork: Supports optional image inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Tuple, Union


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding.
    
    Encodes a scalar timestep into a high-dimensional vector using
    sinusoidal position encoding, similar to Transformer position encodings.
    
    Args:
        embedding_dim: Dimension of the output embedding
        max_period: Maximum period for the sinusoidal encoding
    """
    
    def __init__(self, embedding_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        
        # Learnable projection
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute timestep embeddings.
        
        Args:
            timesteps: Tensor of shape (batch_size,) with values in [0, 1]
            
        Returns:
            Embeddings of shape (batch_size, embedding_dim)
        """
        # Ensure timesteps is float and has correct shape
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        timesteps = timesteps.float()
        
        half_dim = self.embedding_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        
        args = timesteps[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Handle odd embedding dimensions
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
            
        return self.mlp(embedding)


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function
        output_activation: Activation for output layer (None for no activation)
        dropout: Dropout rate (0 for no dropout)
        layer_norm: Whether to use layer normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU(),
        output_activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
        layer_norm: bool = False
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation)
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionNoisePredictor(nn.Module):
    """Noise prediction network for Diffusion Policy.
    
    Predicts the noise added to an action given the noisy action,
    current state, and diffusion timestep.
    
    Architecture:
    - Timestep embedding
    - Concatenate [state, noisy_action, time_embedding]
    - MLP to predict noise
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        time_embed_dim: Dimension of timestep embedding
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        time_embed_dim: int = 64
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_embed_dim = time_embed_dim
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_embed_dim)
        
        # Main network
        input_dim = state_dim + action_dim + time_embed_dim
        self.net = MLP(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=nn.SiLU(),
            output_activation=None
        )
        
        # Initialize output layer to zero for stability
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        state: torch.Tensor, 
        noisy_action: torch.Tensor, 
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise in noisy action.
        
        Args:
            state: Current state, shape (batch_size, state_dim)
            noisy_action: Noisy action, shape (batch_size, action_dim)
            timestep: Diffusion timestep in [0, 1], shape (batch_size,)
            
        Returns:
            Predicted noise, shape (batch_size, action_dim)
        """
        # Get timestep embedding
        t_embed = self.time_embed(timestep)
        
        # Concatenate inputs
        x = torch.cat([state, noisy_action, t_embed], dim=-1)
        
        # Predict noise
        return self.net(x)


class FlowVelocityPredictor(nn.Module):
    """Velocity field predictor for Flow Matching Policy.
    
    Predicts the velocity field v(x_t, t | s) for the ODE:
    dx/dt = v(x, t | s)
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        time_embed_dim: Dimension of timestep embedding
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        time_embed_dim: int = 64
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_embed_dim)
        
        # Main network
        input_dim = state_dim + action_dim + time_embed_dim
        self.net = MLP(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=nn.SiLU(),
            output_activation=None
        )
        
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """Predict velocity field.
        
        Args:
            state: Current state, shape (batch_size, state_dim)
            action: Current action in flow, shape (batch_size, action_dim)
            timestep: Flow timestep in [0, 1], shape (batch_size,)
            
        Returns:
            Predicted velocity, shape (batch_size, action_dim)
        """
        t_embed = self.time_embed(timestep)
        x = torch.cat([state, action, t_embed], dim=-1)
        return self.net(x)


class QNetwork(nn.Module):
    """Q-value network for state-action pairs.
    
    Estimates Q(s, a) - the expected return starting from state s,
    taking action a, and following the policy thereafter.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        
        self.net = MLP(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=nn.ReLU()
        )
        
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Compute Q-value.
        
        Args:
            state: State, shape (batch_size, state_dim)
            action: Action, shape (batch_size, action_dim)
            
        Returns:
            Q-value, shape (batch_size, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class ValueNetwork(nn.Module):
    """State value network V(s).
    
    Estimates the expected return starting from state s
    and following the policy.
    
    Args:
        state_dim: Dimension of state space
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        
        self.net = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=nn.ReLU()
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute state value.
        
        Args:
            state: State, shape (batch_size, state_dim)
            
        Returns:
            Value, shape (batch_size, 1)
        """
        return self.net(state)


class DoubleQNetwork(nn.Module):
    """Twin Q-networks for Double Q-learning / TD3 style algorithms.
    
    Uses two independent Q-networks and takes the minimum for
    target computation to reduce overestimation bias.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)
        
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both Q-values.
        
        Args:
            state: State, shape (batch_size, state_dim)
            action: Action, shape (batch_size, action_dim)
            
        Returns:
            Tuple of (Q1, Q2), each shape (batch_size, 1)
        """
        return self.q1(state, action), self.q2(state, action)
    
    def q1_forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Compute only Q1 value (for efficiency during policy update)."""
        return self.q1(state, action)
    
    def min_q(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Compute minimum of both Q-values.
        
        Args:
            state: State, shape (batch_size, state_dim)
            action: Action, shape (batch_size, action_dim)
            
        Returns:
            min(Q1, Q2), shape (batch_size, 1)
        """
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


# ============================================================================
# Vision-Enabled Network Variants
# ============================================================================

class VisionDiffusionNoisePredictor(nn.Module):
    """Noise prediction network with optional vision encoder support.
    
    Extends DiffusionNoisePredictor to handle multimodal observations
    (state + image) for robotic manipulation tasks.
    
    Args:
        state_dim: Dimension of state space (0 for image-only)
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        time_embed_dim: Dimension of timestep embedding
        vision_encoder: Optional vision encoder module
        vision_output_dim: Output dimension of vision encoder
        obs_mode: Observation mode ("state", "image", or "state_image")
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        time_embed_dim: int = 64,
        vision_encoder: Optional[nn.Module] = None,
        vision_output_dim: int = 128,
        obs_mode: str = "state",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_embed_dim = time_embed_dim
        self.obs_mode = obs_mode
        self.vision_encoder = vision_encoder
        self.vision_output_dim = vision_output_dim if vision_encoder else 0
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_embed_dim)
        
        # Compute total input dimension
        if obs_mode == "state":
            total_obs_dim = state_dim
        elif obs_mode == "image":
            total_obs_dim = self.vision_output_dim
        else:  # state_image
            total_obs_dim = state_dim + self.vision_output_dim
        
        self.total_obs_dim = total_obs_dim
        
        # Main network
        input_dim = total_obs_dim + action_dim + time_embed_dim
        self.net = MLP(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=nn.SiLU(),
            output_activation=None
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _encode_obs(
        self,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode observations based on obs_mode.
        
        Args:
            state: State tensor (B, state_dim) or None
            image: Image tensor (B, C, H, W) or None
            
        Returns:
            Encoded observation tensor (B, total_obs_dim)
        """
        if self.obs_mode == "state":
            return state
        elif self.obs_mode == "image":
            return self.vision_encoder(image)
        else:  # state_image
            image_features = self.vision_encoder(image)
            return torch.cat([state, image_features], dim=-1)
    
    def forward(
        self, 
        noisy_action: torch.Tensor, 
        timestep: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise in noisy action.
        
        Args:
            noisy_action: Noisy action, shape (B, action_dim)
            timestep: Diffusion timestep in [0, 1], shape (B,)
            state: State tensor (B, state_dim) or None
            image: Image tensor (B, C, H, W) or None
            
        Returns:
            Predicted noise, shape (B, action_dim)
        """
        # Encode observations
        obs_features = self._encode_obs(state, image)
        
        # Get timestep embedding
        t_embed = self.time_embed(timestep)
        
        # Concatenate inputs
        x = torch.cat([obs_features, noisy_action, t_embed], dim=-1)
        
        # Predict noise
        return self.net(x)


class VisionFlowVelocityPredictor(nn.Module):
    """Velocity field predictor with optional vision encoder support.
    
    Extends FlowVelocityPredictor to handle multimodal observations
    (state + image) for robotic manipulation tasks.
    
    Args:
        state_dim: Dimension of state space (0 for image-only)
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        time_embed_dim: Dimension of timestep embedding
        vision_encoder: Optional vision encoder module
        vision_output_dim: Output dimension of vision encoder
        obs_mode: Observation mode ("state", "image", or "state_image")
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        time_embed_dim: int = 64,
        vision_encoder: Optional[nn.Module] = None,
        vision_output_dim: int = 128,
        obs_mode: str = "state",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_mode = obs_mode
        self.vision_encoder = vision_encoder
        self.vision_output_dim = vision_output_dim if vision_encoder else 0
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_embed_dim)
        
        # Compute total input dimension
        if obs_mode == "state":
            total_obs_dim = state_dim
        elif obs_mode == "image":
            total_obs_dim = self.vision_output_dim
        else:  # state_image
            total_obs_dim = state_dim + self.vision_output_dim
        
        self.total_obs_dim = total_obs_dim
        
        # Main network
        input_dim = total_obs_dim + action_dim + time_embed_dim
        self.net = MLP(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=nn.SiLU(),
            output_activation=None
        )
    
    def _encode_obs(
        self,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode observations based on obs_mode."""
        if self.obs_mode == "state":
            return state
        elif self.obs_mode == "image":
            return self.vision_encoder(image)
        else:  # state_image
            image_features = self.vision_encoder(image)
            return torch.cat([state, image_features], dim=-1)
        
    def forward(
        self, 
        action: torch.Tensor, 
        timestep: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity field.
        
        Args:
            action: Current action in flow, shape (B, action_dim)
            timestep: Flow timestep in [0, 1], shape (B,)
            state: State tensor (B, state_dim) or None
            image: Image tensor (B, C, H, W) or None
            
        Returns:
            Predicted velocity, shape (B, action_dim)
        """
        obs_features = self._encode_obs(state, image)
        t_embed = self.time_embed(timestep)
        x = torch.cat([obs_features, action, t_embed], dim=-1)
        return self.net(x)


class VisionValueNetwork(nn.Module):
    """State value network with optional vision encoder support.
    
    Extends ValueNetwork to handle multimodal observations.
    
    Args:
        state_dim: Dimension of state space (0 for image-only)
        hidden_dims: List of hidden layer dimensions
        vision_encoder: Optional vision encoder module
        vision_output_dim: Output dimension of vision encoder
        obs_mode: Observation mode ("state", "image", or "state_image")
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256],
        vision_encoder: Optional[nn.Module] = None,
        vision_output_dim: int = 128,
        obs_mode: str = "state",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.obs_mode = obs_mode
        self.vision_encoder = vision_encoder
        self.vision_output_dim = vision_output_dim if vision_encoder else 0
        
        # Compute total input dimension
        if obs_mode == "state":
            total_obs_dim = state_dim
        elif obs_mode == "image":
            total_obs_dim = self.vision_output_dim
        else:  # state_image
            total_obs_dim = state_dim + self.vision_output_dim
        
        self.total_obs_dim = total_obs_dim
        
        self.net = MLP(
            input_dim=total_obs_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=nn.ReLU()
        )
    
    def _encode_obs(
        self,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode observations based on obs_mode."""
        if self.obs_mode == "state":
            return state
        elif self.obs_mode == "image":
            return self.vision_encoder(image)
        else:  # state_image
            image_features = self.vision_encoder(image)
            return torch.cat([state, image_features], dim=-1)
        
    def forward(
        self, 
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute state value.
        
        Args:
            state: State tensor (B, state_dim) or None
            image: Image tensor (B, C, H, W) or None
            
        Returns:
            Value, shape (B, 1)
        """
        obs_features = self._encode_obs(state, image)
        return self.net(obs_features)


class VisionQNetwork(nn.Module):
    """Q-value network with optional vision encoder support.
    
    Args:
        state_dim: Dimension of state space (0 for image-only)
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        vision_encoder: Optional vision encoder module
        vision_output_dim: Output dimension of vision encoder
        obs_mode: Observation mode ("state", "image", or "state_image")
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        vision_encoder: Optional[nn.Module] = None,
        vision_output_dim: int = 128,
        obs_mode: str = "state",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_mode = obs_mode
        self.vision_encoder = vision_encoder
        self.vision_output_dim = vision_output_dim if vision_encoder else 0
        
        # Compute total input dimension
        if obs_mode == "state":
            total_obs_dim = state_dim
        elif obs_mode == "image":
            total_obs_dim = self.vision_output_dim
        else:  # state_image
            total_obs_dim = state_dim + self.vision_output_dim
        
        self.total_obs_dim = total_obs_dim
        
        self.net = MLP(
            input_dim=total_obs_dim + action_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=nn.ReLU()
        )
    
    def _encode_obs(
        self,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode observations based on obs_mode."""
        if self.obs_mode == "state":
            return state
        elif self.obs_mode == "image":
            return self.vision_encoder(image)
        else:  # state_image
            image_features = self.vision_encoder(image)
            return torch.cat([state, image_features], dim=-1)
        
    def forward(
        self, 
        action: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Q-value.
        
        Args:
            action: Action, shape (B, action_dim)
            state: State tensor (B, state_dim) or None
            image: Image tensor (B, C, H, W) or None
            
        Returns:
            Q-value, shape (B, 1)
        """
        obs_features = self._encode_obs(state, image)
        x = torch.cat([obs_features, action], dim=-1)
        return self.net(x)


class VisionDoubleQNetwork(nn.Module):
    """Twin Q-networks with optional vision encoder support.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        vision_encoder: Optional vision encoder module (shared between Q1 and Q2)
        vision_output_dim: Output dimension of vision encoder
        obs_mode: Observation mode
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        vision_encoder: Optional[nn.Module] = None,
        vision_output_dim: int = 128,
        obs_mode: str = "state",
    ):
        super().__init__()
        
        self.obs_mode = obs_mode
        self.vision_encoder = vision_encoder
        
        # Q1 and Q2 share the vision encoder but have separate MLPs
        self.q1 = VisionQNetwork(
            state_dim, action_dim, hidden_dims,
            vision_encoder=None,  # Will use shared encoder
            vision_output_dim=vision_output_dim,
            obs_mode=obs_mode,
        )
        self.q2 = VisionQNetwork(
            state_dim, action_dim, hidden_dims,
            vision_encoder=None,
            vision_output_dim=vision_output_dim,
            obs_mode=obs_mode,
        )
        
        # Override to use shared encoder
        self.vision_output_dim = vision_output_dim if vision_encoder else 0
    
    def _encode_obs(
        self,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode observations using shared vision encoder."""
        if self.obs_mode == "state":
            return state
        elif self.obs_mode == "image":
            return self.vision_encoder(image)
        else:  # state_image
            image_features = self.vision_encoder(image)
            return torch.cat([state, image_features], dim=-1)
        
    def forward(
        self, 
        action: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both Q-values with shared feature encoding."""
        obs_features = self._encode_obs(state, image)
        x = torch.cat([obs_features, action], dim=-1)
        return self.q1.net(x), self.q2.net(x)
    
    def min_q(
        self, 
        action: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute minimum of both Q-values."""
        q1, q2 = self.forward(action, state, image)
        return torch.min(q1, q2)


# ============================================================================
# MultiModal Network Variants (using ObservationEncoder)
# ============================================================================
# These networks use ObservationEncoder for flexible observation handling.
# They support pre-computed observation features (obs_features) for efficiency.

class MultiModalNoisePredictor(nn.Module):
    """Unified noise predictor for diffusion policies with multi-modal observation support.
    
    Used by: DPPO, DiffusionPolicy, DiffusionDoubleQ
    
    Args:
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions for noise predictor
        time_embed_dim: Dimension of timestep embedding
        obs_encoder: ObservationEncoder for processing observations
        obs_dim: Observation dimension (used if obs_encoder is None)
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        time_embed_dim: int = 64,
        obs_encoder: Optional['ObservationEncoder'] = None,
        obs_dim: int = 0,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.obs_encoder = obs_encoder
        
        if obs_encoder is not None:
            obs_feature_dim = obs_encoder.output_dim
        else:
            obs_feature_dim = obs_dim
        
        self.obs_feature_dim = obs_feature_dim
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_embed_dim)
        
        # Noise prediction network
        input_dim = obs_feature_dim + action_dim + time_embed_dim
        self.net = MLP(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=nn.SiLU(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stability."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        obs_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise in noisy action.
        
        Args:
            noisy_action: Noisy action (B, action_dim)
            timestep: Normalized timestep in [0, 1] (B,)
            state: Optional state vector (B, state_dim)
            image: Optional image tensor (B, C, H, W)
            obs_features: Pre-computed observation features (B, obs_feature_dim)
                         If provided, state and image are ignored.
        
        Returns:
            Predicted noise (B, action_dim)
        """
        if obs_features is not None:
            features = obs_features
        elif self.obs_encoder is not None:
            features = self.obs_encoder(state=state, image=image)
        else:
            features = state
        
        t_embed = self.time_embed(timestep)
        x = torch.cat([features, noisy_action, t_embed], dim=-1)
        return self.net(x)


class MultiModalVelocityPredictor(nn.Module):
    """Unified velocity predictor for flow matching with multi-modal observation support.
    
    Used by: ReinFlow, FlowMatching, CPQL
    
    Args:
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        time_embed_dim: Dimension of timestep embedding
        obs_encoder: ObservationEncoder for processing observations
        obs_dim: Observation dimension (used if obs_encoder is None)
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        time_embed_dim: int = 64,
        obs_encoder: Optional['ObservationEncoder'] = None,
        obs_dim: int = 0,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.obs_encoder = obs_encoder
        
        if obs_encoder is not None:
            obs_feature_dim = obs_encoder.output_dim
        else:
            obs_feature_dim = obs_dim
        
        self.obs_feature_dim = obs_feature_dim
        
        # Simple timestep embedding (linear)
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Velocity prediction network
        input_dim = obs_feature_dim + action_dim + time_embed_dim
        self.net = MLP(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=nn.SiLU(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        obs_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity at position x and time t.
        
        Args:
            x: Current position in flow (B, action_dim)
            t: Current timestep in [0, 1] (B,)
            state: Optional state vector (B, state_dim)
            image: Optional image tensor (B, C, H, W)
            obs_features: Pre-computed observation features
        
        Returns:
            Predicted velocity (B, action_dim)
        """
        if obs_features is not None:
            features = obs_features
        elif self.obs_encoder is not None:
            features = self.obs_encoder(state=state, image=image)
        else:
            features = state
        
        t_embed = self.time_embed(t.unsqueeze(-1))
        combined = torch.cat([features, x, t_embed], dim=-1)
        return self.net(combined)


class MultiModalValueNetwork(nn.Module):
    """Unified value network with multi-modal observation support.
    
    Used by: DPPO, ReinFlow
    
    Args:
        hidden_dims: Hidden layer dimensions
        obs_encoder: ObservationEncoder for processing observations
        obs_dim: Observation dimension (used if obs_encoder is None)
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [256, 256],
        obs_encoder: Optional['ObservationEncoder'] = None,
        obs_dim: int = 0,
    ):
        super().__init__()
        
        self.obs_encoder = obs_encoder
        
        if obs_encoder is not None:
            input_dim = obs_encoder.output_dim
        else:
            input_dim = obs_dim
        
        self.net = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=nn.ReLU(),
        )
    
    def forward(
        self,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        obs_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute state value.
        
        Args:
            state: Optional state vector (B, state_dim)
            image: Optional image tensor (B, C, H, W)
            obs_features: Pre-computed observation features
        
        Returns:
            Value estimate (B, 1)
        """
        if obs_features is not None:
            features = obs_features
        elif self.obs_encoder is not None:
            features = self.obs_encoder(state=state, image=image)
        else:
            features = state
        
        return self.net(features)


class MultiModalDoubleQNetwork(nn.Module):
    """Unified double Q-network with multi-modal observation support.
    
    Used by: DiffusionDoubleQ, CPQL
    
    Args:
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        obs_encoder: ObservationEncoder for processing observations
        obs_dim: Observation dimension (used if obs_encoder is None)
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        obs_encoder: Optional['ObservationEncoder'] = None,
        obs_dim: int = 0,
    ):
        super().__init__()
        
        self.obs_encoder = obs_encoder
        
        if obs_encoder is not None:
            obs_feature_dim = obs_encoder.output_dim
        else:
            obs_feature_dim = obs_dim
        
        input_dim = obs_feature_dim + action_dim
        
        self.q1 = MLP(input_dim=input_dim, output_dim=1, hidden_dims=hidden_dims, activation=nn.ReLU())
        self.q2 = MLP(input_dim=input_dim, output_dim=1, hidden_dims=hidden_dims, activation=nn.ReLU())
        
        # Initialize Q-networks with small weights to prevent large initial Q-values
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable Q-value estimation."""
        for module in [self.q1, self.q2]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        action: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        obs_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both Q-values.
        
        Args:
            action: Action (B, action_dim)
            state: Optional state vector (B, state_dim)
            image: Optional image tensor (B, C, H, W)
            obs_features: Pre-computed observation features
        
        Returns:
            Tuple of (Q1, Q2), each shape (B, 1)
        """
        if obs_features is not None:
            features = obs_features
        elif self.obs_encoder is not None:
            features = self.obs_encoder(state=state, image=image)
        else:
            features = state
        
        sa = torch.cat([features, action], dim=-1)
        return self.q1(sa), self.q2(sa)
    
    def q1_forward(
        self,
        action: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        obs_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute only Q1 value (for efficiency during policy update)."""
        if obs_features is not None:
            features = obs_features
        elif self.obs_encoder is not None:
            features = self.obs_encoder(state=state, image=image)
        else:
            features = state
        
        sa = torch.cat([features, action], dim=-1)
        return self.q1(sa)
    
    def min_q(
        self,
        action: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        obs_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute minimum of both Q-values."""
        q1, q2 = self.forward(action, state, image, obs_features)
        return torch.min(q1, q2)


class MultiModalNoisyFlowMLP(nn.Module):
    """Noisy Flow MLP with multi-modal observation support.
    
    Wraps a base flow velocity network with learnable noise injection.
    The noise is predicted by a separate network based on time and 
    observation embeddings.
    
    Used by: ReinFlow
    
    Args:
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        noise_scheduler_type: Type of noise schedule ('const', 'learn', 'learn_decay')
        init_noise_std: Initial noise standard deviation
        obs_encoder: ObservationEncoder for processing observations
        obs_dim: Observation dimension (used if obs_encoder is None)
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        noise_scheduler_type: str = "learn",
        init_noise_std: float = 0.1,
        obs_encoder: Optional['ObservationEncoder'] = None,
        obs_dim: int = 0,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.noise_scheduler_type = noise_scheduler_type
        self.init_noise_std = init_noise_std
        self.obs_encoder = obs_encoder
        
        if obs_encoder is not None:
            obs_feature_dim = obs_encoder.output_dim
        else:
            obs_feature_dim = obs_dim
        
        self.obs_feature_dim = obs_feature_dim
        
        # Base velocity network
        self.base_net = MultiModalVelocityPredictor(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            obs_encoder=obs_encoder,
            obs_dim=obs_dim,
        )
        
        # Time embedding for noise prediction
        self.time_embed_dim = 64
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # Observation embedding for noise prediction
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_feature_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # Exploration noise prediction network
        self.explore_noise_net = nn.Sequential(
            nn.Linear(self.time_embed_dim * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
            nn.Softplus()
        )
        
        self._init_noise_net()
        
        if noise_scheduler_type == "learn_decay":
            self.log_decay = nn.Parameter(torch.zeros(1))
            
        if noise_scheduler_type == "const":
            self.register_buffer("const_noise_std", torch.ones(action_dim) * init_noise_std)
    
    def _init_noise_net(self):
        """Initialize noise network to output init_noise_std."""
        init_bias = np.log(np.exp(self.init_noise_std) - 1)
        nn.init.constant_(self.explore_noise_net[-2].bias, init_bias)
        nn.init.zeros_(self.explore_noise_net[-2].weight)
    
    def get_noise_std(
        self,
        t: torch.Tensor,
        obs_features: torch.Tensor
    ) -> torch.Tensor:
        """Get exploration noise std based on time and observations.
        
        Args:
            t: Current timestep in [0, 1]
            obs_features: Encoded observation features
            
        Returns:
            Noise std of shape (batch_size, action_dim)
        """
        if self.noise_scheduler_type == "const":
            batch_size = obs_features.shape[0]
            return self.const_noise_std.unsqueeze(0).expand(batch_size, -1)
        
        time_emb = self.time_embed(t.unsqueeze(-1))
        obs_emb = self.obs_embed(obs_features)
        
        combined = torch.cat([time_emb, obs_emb], dim=-1)
        noise_std = self.explore_noise_net(combined)
        
        if self.noise_scheduler_type == "learn_decay":
            decay_factor = torch.exp(self.log_decay)
            time_scale = torch.exp(-decay_factor * t).unsqueeze(-1)
            noise_std = noise_std * time_scale
        
        return noise_std.clamp(min=0.001, max=0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        obs_features: Optional[torch.Tensor] = None,
        sample_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: compute velocity and optionally sample exploration noise.
        
        Args:
            x: Current position in flow (B, action_dim)
            t: Current timestep in [0, 1] (B,)
            state: Optional state tensor
            image: Optional image tensor
            obs_features: Pre-computed observation features
            sample_noise: Whether to sample noise
            
        Returns:
            Tuple of (velocity, noise, noise_std)
        """
        # Get observation features
        if obs_features is not None:
            features = obs_features
        elif self.obs_encoder is not None:
            features = self.obs_encoder(state=state, image=image)
        else:
            features = state
        
        # Get base velocity
        velocity = self.base_net(x, t, obs_features=features)
        
        # Get noise std
        noise_std = self.get_noise_std(t, features)
        
        if sample_noise:
            noise = torch.randn_like(velocity)
        else:
            noise = torch.zeros_like(velocity)
        
        return velocity, noise, noise_std
    
    def forward_with_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        obs_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict velocity and return noise std.
        
        Returns:
            Tuple of (velocity, noise_std)
        """
        if obs_features is None:
            if self.obs_encoder is not None:
                obs_features = self.obs_encoder(state=state, image=image)
            else:
                obs_features = state
        
        velocity = self.base_net(x, t, obs_features=obs_features)
        noise_std = self.get_noise_std(t, obs_features)
        
        return velocity, noise_std


# ============================================================================
# ShortCut Flow Networks (for ReinFlow with self-consistency)
# ============================================================================
# These networks support step size `d` as additional input for ShortCut models.
# Reference: ReinFlow (https://github.com/ReinFlow/ReinFlow)

class MultiModalShortCutVelocityPredictor(nn.Module):
    """ShortCut velocity predictor with step size conditioning.
    
    Unlike standard flow matching that only takes time t, ShortCut models
    also condition on step size d, enabling self-consistency training.
    
    The embedding combines:
    - t_embed: sinusoidal embedding of time t
    - d_embed: sinusoidal embedding of step size d
    - These are concatenated and projected to td_embed_dim
    
    Reference: ShortCutFlow from ReinFlow paper.
    
    Args:
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        td_embed_dim: Dimension of combined time-step embedding
        obs_encoder: ObservationEncoder for processing observations
        obs_dim: Observation dimension (used if obs_encoder is None)
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        td_embed_dim: int = 64,
        obs_encoder: Optional['ObservationEncoder'] = None,
        obs_dim: int = 0,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.td_embed_dim = td_embed_dim
        self.obs_encoder = obs_encoder
        
        if obs_encoder is not None:
            obs_feature_dim = obs_encoder.output_dim
        else:
            obs_feature_dim = obs_dim
        
        self.obs_feature_dim = obs_feature_dim
        
        # Sinusoidal embedding for both t and d (shared architecture)
        # Following ReinFlow: use SinusoidalPosEmb style
        self.map_t = nn.Sequential(
            nn.Linear(1, td_embed_dim),
            nn.SiLU(),
            nn.Linear(td_embed_dim, td_embed_dim)
        )
        self.map_d = nn.Sequential(
            nn.Linear(1, td_embed_dim),
            nn.SiLU(),
            nn.Linear(td_embed_dim, td_embed_dim)
        )
        
        # Project concatenated [t_emb, d_emb] to td_embed_dim
        self.td_embed = nn.Sequential(
            nn.Linear(2 * td_embed_dim, td_embed_dim),
            nn.Mish(),  # ReinFlow uses Mish
            nn.Linear(td_embed_dim, td_embed_dim)
        )
        
        # Velocity prediction network
        input_dim = obs_feature_dim + action_dim + td_embed_dim
        self.net = MLP(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=nn.SiLU(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        d: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        obs_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity at position x, time t, with step size d.
        
        Args:
            x: Current position in flow (B, action_dim)
            t: Current timestep in [0, 1] (B,)
            d: Step size in [0, 1] (B,), e.g., 1/num_steps
            state: Optional state vector (B, state_dim)
            image: Optional image tensor (B, C, H, W)
            obs_features: Pre-computed observation features
        
        Returns:
            Predicted velocity (B, action_dim)
        """
        if obs_features is not None:
            features = obs_features
        elif self.obs_encoder is not None:
            features = self.obs_encoder(state=state, image=image)
        else:
            features = state
        
        # Embed time t and step size d separately
        t_emb = self.map_t(t.unsqueeze(-1))
        d_emb = self.map_d(d.unsqueeze(-1))
        
        # Combine embeddings
        td_emb = self.td_embed(torch.cat([t_emb, d_emb], dim=-1))
        
        # Concatenate and predict
        combined = torch.cat([features, x, td_emb], dim=-1)
        return self.net(combined)


class MultiModalNoisyShortCutFlowMLP(nn.Module):
    """Noisy ShortCut Flow MLP with multi-modal observation support.
    
    Extends MultiModalShortCutVelocityPredictor with learnable exploration noise,
    similar to MultiModalNoisyFlowMLP but with step size conditioning.
    
    Used by: ReinFlow (online fine-tuning with ShortCut model)
    
    Args:
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        noise_scheduler_type: Type of noise schedule ('const', 'learn', 'learn_decay')
        init_noise_std: Initial noise standard deviation
        obs_encoder: ObservationEncoder for processing observations
        obs_dim: Observation dimension (used if obs_encoder is None)
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        noise_scheduler_type: str = "learn",
        init_noise_std: float = 0.1,
        obs_encoder: Optional['ObservationEncoder'] = None,
        obs_dim: int = 0,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.noise_scheduler_type = noise_scheduler_type
        self.init_noise_std = init_noise_std
        self.obs_encoder = obs_encoder
        
        if obs_encoder is not None:
            obs_feature_dim = obs_encoder.output_dim
        else:
            obs_feature_dim = obs_dim
        
        self.obs_feature_dim = obs_feature_dim
        
        # Base ShortCut velocity network
        self.base_net = MultiModalShortCutVelocityPredictor(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            obs_encoder=obs_encoder,
            obs_dim=obs_dim,
        )
        
        # Time embedding for noise prediction
        self.time_embed_dim = 64
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # Observation embedding for noise prediction
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_feature_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # Exploration noise prediction network
        self.explore_noise_net = nn.Sequential(
            nn.Linear(self.time_embed_dim * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
            nn.Softplus()
        )
        
        self._init_noise_net()
        
        if noise_scheduler_type == "learn_decay":
            self.log_decay = nn.Parameter(torch.zeros(1))
            
        if noise_scheduler_type == "const":
            self.register_buffer("const_noise_std", torch.ones(action_dim) * init_noise_std)
    
    def _init_noise_net(self):
        """Initialize noise network to output init_noise_std."""
        init_bias = np.log(np.exp(self.init_noise_std) - 1)
        nn.init.constant_(self.explore_noise_net[-2].bias, init_bias)
        nn.init.zeros_(self.explore_noise_net[-2].weight)
    
    def get_noise_std(
        self,
        t: torch.Tensor,
        obs_features: torch.Tensor
    ) -> torch.Tensor:
        """Get exploration noise std based on time and observations."""
        if self.noise_scheduler_type == "const":
            batch_size = obs_features.shape[0]
            return self.const_noise_std.unsqueeze(0).expand(batch_size, -1)
        
        time_emb = self.time_embed(t.unsqueeze(-1))
        obs_emb = self.obs_embed(obs_features)
        
        combined = torch.cat([time_emb, obs_emb], dim=-1)
        noise_std = self.explore_noise_net(combined)
        
        if self.noise_scheduler_type == "learn_decay":
            decay_factor = torch.exp(self.log_decay)
            time_scale = torch.exp(-decay_factor * t).unsqueeze(-1)
            noise_std = noise_std * time_scale
        
        return noise_std.clamp(min=0.001, max=0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        d: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        obs_features: Optional[torch.Tensor] = None,
        sample_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: compute velocity and optionally sample exploration noise.
        
        Args:
            x: Current position in flow (B, action_dim)
            t: Current timestep in [0, 1] (B,)
            d: Step size in [0, 1] (B,)
            state: Optional state tensor
            image: Optional image tensor
            obs_features: Pre-computed observation features
            sample_noise: Whether to sample noise
            
        Returns:
            Tuple of (velocity, noise, noise_std)
        """
        # Get observation features
        if obs_features is not None:
            features = obs_features
        elif self.obs_encoder is not None:
            features = self.obs_encoder(state=state, image=image)
        else:
            features = state
        
        # Get base velocity with step size conditioning
        velocity = self.base_net(x, t, d, obs_features=features)
        
        # Get noise std
        noise_std = self.get_noise_std(t, features)
        
        if sample_noise:
            noise = torch.randn_like(velocity)
        else:
            noise = torch.zeros_like(velocity)
        
        return velocity, noise, noise_std
