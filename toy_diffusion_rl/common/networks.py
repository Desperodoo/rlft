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
    
    Supports Action Chunking: when action_horizon > 1, predicts noise for
    a sequence of actions instead of a single action.
    
    Args:
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions for noise predictor
        time_embed_dim: Dimension of timestep embedding
        obs_encoder: ObservationEncoder for processing observations
        obs_dim: Observation dimension (used if obs_encoder is None)
        action_horizon: Number of future actions to predict (1 = no chunking)
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        time_embed_dim: int = 64,
        obs_encoder: Optional['ObservationEncoder'] = None,
        obs_dim: int = 0,
        action_horizon: int = 1,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_encoder = obs_encoder
        
        # Total output dimension: action_dim * action_horizon (flattened)
        self.output_dim = action_dim * action_horizon
        
        if obs_encoder is not None:
            obs_feature_dim = obs_encoder.output_dim
        else:
            obs_feature_dim = obs_dim
        
        self.obs_feature_dim = obs_feature_dim
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_embed_dim)
        
        # Noise prediction network
        # Input: obs_features + flattened noisy action sequence + time embedding
        input_dim = obs_feature_dim + self.output_dim + time_embed_dim
        self.net = MLP(
            input_dim=input_dim,
            output_dim=self.output_dim,
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
            noisy_action: Noisy action tensor
                - If action_horizon == 1: (B, action_dim)
                - If action_horizon > 1: (B, action_horizon, action_dim) or (B, action_horizon * action_dim)
            timestep: Normalized timestep in [0, 1] (B,)
            state: Optional state vector (B, state_dim)
            image: Optional image tensor (B, C, H, W)
            obs_features: Pre-computed observation features (B, obs_feature_dim)
                         If provided, state and image are ignored.
        
        Returns:
            Predicted noise with same shape as input noisy_action
        """
        if obs_features is not None:
            features = obs_features
        elif self.obs_encoder is not None:
            features = self.obs_encoder(state=state, image=image)
        else:
            features = state
        
        # Remember input shape for output reshaping
        input_shape = noisy_action.shape
        batch_size = noisy_action.shape[0]
        
        # Flatten action sequence if needed
        if noisy_action.dim() == 3:  # (B, T, action_dim)
            noisy_action_flat = noisy_action.view(batch_size, -1)
        else:  # Already flat (B, action_dim) or (B, T * action_dim)
            noisy_action_flat = noisy_action
        
        t_embed = self.time_embed(timestep)
        x = torch.cat([features, noisy_action_flat, t_embed], dim=-1)
        predicted_noise_flat = self.net(x)
        
        # Reshape output to match input shape
        if len(input_shape) == 3:  # Input was (B, T, action_dim)
            return predicted_noise_flat.view(batch_size, self.action_horizon, self.action_dim)
        else:
            return predicted_noise_flat


class MultiModalVelocityPredictor(nn.Module):
    """Unified velocity predictor for flow matching with multi-modal observation support.
    
    Used by: ReinFlow, FlowMatching, CPQL
    
    Supports Action Chunking: when action_horizon > 1, predicts velocity for
    a sequence of actions instead of a single action.
    
    Args:
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        time_embed_dim: Dimension of timestep embedding
        obs_encoder: ObservationEncoder for processing observations
        obs_dim: Observation dimension (used if obs_encoder is None)
        action_horizon: Number of future actions to predict (1 = no chunking)
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        time_embed_dim: int = 64,
        obs_encoder: Optional['ObservationEncoder'] = None,
        obs_dim: int = 0,
        action_horizon: int = 1,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_encoder = obs_encoder
        
        # Total output dimension: action_dim * action_horizon (flattened)
        self.output_dim = action_dim * action_horizon
        
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
        # Input: obs_features + flattened action sequence + time embedding
        input_dim = obs_feature_dim + self.output_dim + time_embed_dim
        self.net = MLP(
            input_dim=input_dim,
            output_dim=self.output_dim,
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
            x: Current position in flow
                - If action_horizon == 1: (B, action_dim)
                - If action_horizon > 1: (B, action_horizon, action_dim) or (B, action_horizon * action_dim)
            t: Current timestep in [0, 1] (B,)
            state: Optional state vector (B, state_dim)
            image: Optional image tensor (B, C, H, W)
            obs_features: Pre-computed observation features
        
        Returns:
            Predicted velocity with same shape as input x
        """
        if obs_features is not None:
            features = obs_features
        elif self.obs_encoder is not None:
            features = self.obs_encoder(state=state, image=image)
        else:
            features = state
        
        # Remember input shape for output reshaping
        input_shape = x.shape
        batch_size = x.shape[0]
        
        # Flatten action sequence if needed
        if x.dim() == 3:  # (B, T, action_dim)
            x_flat = x.view(batch_size, -1)
        else:  # Already flat
            x_flat = x
        
        t_embed = self.time_embed(t.unsqueeze(-1))
        combined = torch.cat([features, x_flat, t_embed], dim=-1)
        predicted_velocity_flat = self.net(combined)
        
        # Reshape output to match input shape
        if len(input_shape) == 3:  # Input was (B, T, action_dim)
            return predicted_velocity_flat.view(batch_size, self.action_horizon, self.action_dim)
        else:
            return predicted_velocity_flat


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


# ============================================================================
# 1D U-Net Architecture for Diffusion Policy
# ============================================================================
# Reference: ManiSkill official diffusion policy implementation
# https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/diffusion_policy/diffusion_policy/conditional_unet1d.py

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep.
    
    Standard positional encoding used in transformers and diffusion models.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embedding.
        
        Args:
            x: Timestep tensor of shape (batch_size,)
            
        Returns:
            Embedding of shape (batch_size, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    """1D downsampling using strided convolution."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """1D upsampling using transposed convolution."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish activation block."""
    
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 8,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning.
    
    Takes two inputs: x (feature) and cond (conditioning).
    x is passed through 2 Conv1dBlock stacked together with residual connection.
    cond is applied to x with FiLM (Feature-wise Linear Modulation).
    
    Reference: https://arxiv.org/abs/1709.07871
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        
        # FiLM modulation: predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))  # (B, cond_channels) -> (B, cond_channels, 1)
        )
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass with conditioning.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, horizon)
            cond: Conditioning tensor of shape (batch_size, cond_dim)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, horizon)
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        
        # FiLM modulation: scale and bias
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """1D U-Net for noise prediction in diffusion policy.
    
    This is a simplified version of the official ManiSkill implementation.
    Uses FiLM conditioning to incorporate timestep and observation embeddings.
    
    Architecture:
    - Encoder: Series of ConditionalResidualBlock1D with downsampling
    - Middle: ConditionalResidualBlock1D without resolution change
    - Decoder: Series of ConditionalResidualBlock1D with upsampling and skip connections
    
    Args:
        input_dim: Dimension of input actions
        global_cond_dim: Dimension of global conditioning (obs_horizon * obs_dim)
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration
        down_dims: Channel size for each UNet level
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
    
    Reference:
        https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/diffusion_policy/diffusion_policy/conditional_unet1d.py
    """
    
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: List[int] = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        
        # Diffusion step encoder
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        
        # Total conditioning dimension
        cond_dim = dsed + global_cond_dim
        
        # Build encoder (down) modules
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
        
        # Middle modules
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups),
        ])
        
        # Build decoder (up) modules
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        # Final conv
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        
        self.down_modules = down_modules
        self.up_modules = up_modules
        
        # Print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"ConditionalUnet1D: {n_params / 1e6:.2f}M parameters")
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            sample: Noisy action sequence of shape (B, T, input_dim)
            timestep: Diffusion timestep(s), shape (B,) or scalar
            global_cond: Global conditioning of shape (B, global_cond_dim)
            
        Returns:
            Predicted noise of shape (B, T, input_dim)
        """
        # Convert to (B, C, T) for 1D convolutions
        sample = sample.moveaxis(-1, -2)  # (B, T, C) -> (B, C, T)
        
        # Handle timestep
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        
        # Broadcast to batch dimension
        timestep = timestep.expand(sample.shape[0])
        
        # Encode diffusion step
        global_feature = self.diffusion_step_encoder(timestep)
        
        # Concatenate with global conditioning
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)
        
        # Encoder
        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        
        # Middle
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        
        # Decoder with skip connections
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        
        # Final conv
        x = self.final_conv(x)
        
        # Convert back to (B, T, C)
        x = x.moveaxis(-1, -2)  # (B, C, T) -> (B, T, C)
        return x


class MultiModalUnet1DNoisePredictor(nn.Module):
    """1D U-Net noise predictor with multi-modal observation support.
    
    Wraps ConditionalUnet1D to work with the ObservationEncoder interface.
    
    This is the preferred network for diffusion policy with action chunking,
    as the U-Net architecture preserves temporal structure better than MLP.
    
    Args:
        action_dim: Dimension of action space
        obs_encoder: ObservationEncoder for processing observations
        obs_dim: Observation dimension (used if obs_encoder is None)
        action_horizon: Number of actions in sequence (pred_horizon)
        diffusion_step_embed_dim: Size of diffusion step embedding
        down_dims: Channel sizes for each U-Net level
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
    """
    
    def __init__(
        self,
        action_dim: int,
        obs_encoder: Optional['ObservationEncoder'] = None,
        obs_dim: int = 0,
        action_horizon: int = 16,
        diffusion_step_embed_dim: int = 64,
        down_dims: List[int] = [64, 128, 256],
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_encoder = obs_encoder
        
        if obs_encoder is not None:
            obs_feature_dim = obs_encoder.output_dim
        else:
            obs_feature_dim = obs_dim
        
        self.obs_feature_dim = obs_feature_dim
        
        # 1D U-Net for noise prediction
        # global_cond_dim = obs_feature_dim (single observation embedding)
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
    
    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        obs_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise in noisy action sequence.
        
        Args:
            noisy_action: Noisy action tensor of shape (B, T, action_dim) or (B, action_dim)
            timestep: Diffusion timestep (can be normalized [0,1] or integer)
            state: Optional state tensor (B, state_dim)
            image: Optional image tensor (B, C, H, W)
            obs_features: Pre-computed observation features (B, obs_feature_dim)
            
        Returns:
            Predicted noise with same shape as noisy_action
        """
        # Get observation features
        if obs_features is not None:
            features = obs_features
        elif self.obs_encoder is not None:
            features = self.obs_encoder(state=state, image=image)
        else:
            features = state
        
        # Remember input shape
        input_shape = noisy_action.shape
        batch_size = noisy_action.shape[0]
        
        # Ensure 3D input for U-Net: (B, T, action_dim)
        if noisy_action.dim() == 2:
            # (B, action_dim) -> (B, 1, action_dim)
            noisy_action = noisy_action.unsqueeze(1)
        
        # Convert normalized timestep to integer if needed
        if timestep.dtype == torch.float32 or timestep.dtype == torch.float64:
            # Assume timestep is normalized [0, 1], convert to integer
            # This is approximate - exact conversion depends on num_diffusion_steps
            timestep = (timestep * 100).long()
        
        # Forward through U-Net
        predicted_noise = self.unet(noisy_action, timestep, global_cond=features)
        
        # Restore original shape
        if len(input_shape) == 2:
            predicted_noise = predicted_noise.squeeze(1)
        
        return predicted_noise