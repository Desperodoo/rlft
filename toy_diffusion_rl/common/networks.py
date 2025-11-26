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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Tuple


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
