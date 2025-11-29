"""
Unified Observation Encoder for Multi-Modal Inputs.

This module provides a unified interface for encoding observations that may
consist of state vectors, images, or both. All agents can use this encoder
to support flexible observation modes without duplicating code.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union, Any

try:
    from .vision_encoders import CNNEncoder, DINOv2Encoder, make_vision_encoder
except ImportError:
    try:
        from toy_diffusion_rl.common.vision_encoders import CNNEncoder, DINOv2Encoder, make_vision_encoder
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from vision_encoders import CNNEncoder, DINOv2Encoder, make_vision_encoder


class ObservationEncoder(nn.Module):
    """Unified observation encoder supporting state, image, or state+image inputs.
    
    This encoder provides a consistent interface for all algorithms to process
    different observation modalities. It handles:
    - State-only: Pass through or optional MLP projection
    - Image-only: Vision encoder to embedding
    - State+Image: Concatenation of state and image embeddings
    
    Args:
        obs_mode: Observation mode ("state", "image", or "state_image")
        state_dim: Dimension of state vector (required for state/state_image modes)
        image_shape: Shape of image as (H, W, C) (required for image/state_image modes)
        output_dim: Output embedding dimension
        vision_encoder_type: Type of vision encoder ("cnn" or "dinov2")
        vision_encoder_kwargs: Additional kwargs for vision encoder
        state_projection: Whether to project state through MLP (default: False)
        state_hidden_dims: Hidden dims for state projection MLP
        freeze_vision: Whether to freeze vision encoder (for pretrained models)
    
    Example:
        >>> # State-only mode
        >>> encoder = ObservationEncoder(
        ...     obs_mode="state", state_dim=31, output_dim=31
        ... )
        >>> state = torch.randn(32, 31)
        >>> features = encoder(state=state)
        
        >>> # Image-only mode
        >>> encoder = ObservationEncoder(
        ...     obs_mode="image", image_shape=(128, 128, 3), output_dim=128,
        ...     vision_encoder_type="cnn"
        ... )
        >>> image = torch.randn(32, 3, 128, 128)
        >>> features = encoder(image=image)
        
        >>> # State+Image mode
        >>> encoder = ObservationEncoder(
        ...     obs_mode="state_image", state_dim=31, image_shape=(128, 128, 3),
        ...     output_dim=159, vision_encoder_type="cnn"
        ... )
        >>> features = encoder(state=state, image=image)
    """
    
    def __init__(
        self,
        obs_mode: str = "state",
        state_dim: Optional[int] = None,
        image_shape: Optional[Tuple[int, int, int]] = None,
        output_dim: Optional[int] = None,
        vision_encoder_type: str = "cnn",
        vision_encoder_kwargs: Optional[Dict[str, Any]] = None,
        state_projection: bool = False,
        state_hidden_dims: Optional[list] = None,
        freeze_vision: bool = False,
    ):
        super().__init__()
        
        self.obs_mode = obs_mode
        self.state_dim = state_dim
        self.image_shape = image_shape
        self.vision_encoder_type = vision_encoder_type
        self.freeze_vision = freeze_vision
        
        # Validate inputs
        if obs_mode in ["state", "state_image"] and state_dim is None:
            raise ValueError(f"state_dim required for obs_mode='{obs_mode}'")
        if obs_mode in ["image", "state_image"] and image_shape is None:
            raise ValueError(f"image_shape required for obs_mode='{obs_mode}'")
        
        # Vision encoder output dim
        vision_kwargs = vision_encoder_kwargs or {}
        if obs_mode in ["image", "state_image"]:
            # Default vision output dim
            vision_output_dim = vision_kwargs.get("output_dim", 128)
        else:
            vision_output_dim = 0
        
        # Compute output dimension
        if obs_mode == "state":
            if state_projection:
                self._output_dim = output_dim or state_dim
            else:
                self._output_dim = state_dim
        elif obs_mode == "image":
            self._output_dim = vision_output_dim
        else:  # state_image
            self._output_dim = state_dim + vision_output_dim
        
        # Override with explicit output_dim if provided
        if output_dim is not None:
            self._output_dim = output_dim
        
        # Build vision encoder
        self.vision_encoder = None
        if obs_mode in ["image", "state_image"]:
            # Remove output_dim from kwargs to avoid duplicate
            vision_kwargs_copy = vision_kwargs.copy()
            vis_output_dim = vision_kwargs_copy.pop("output_dim", 128)
            
            self.vision_encoder = make_vision_encoder(
                encoder_type=vision_encoder_type,
                image_shape=image_shape,
                output_dim=vis_output_dim,
                freeze=freeze_vision,
                **vision_kwargs_copy
            )
            self.vision_output_dim = vis_output_dim
        else:
            self.vision_output_dim = 0
        
        # Build state projection (optional)
        self.state_projector = None
        if state_projection and obs_mode in ["state", "state_image"]:
            hidden_dims = state_hidden_dims or [256]
            layers = []
            in_dim = state_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.ReLU(),
                ])
                in_dim = h_dim
            # Final projection to match state_dim (for additive combination)
            if obs_mode == "state":
                layers.append(nn.Linear(in_dim, output_dim or state_dim))
            else:
                layers.append(nn.Linear(in_dim, state_dim))
            self.state_projector = nn.Sequential(*layers)
    
    @property
    def output_dim(self) -> int:
        """Output dimension of the encoder."""
        return self._output_dim
    
    def forward(
        self,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode observations.
        
        Args:
            state: State vector of shape (B, state_dim)
            image: Image tensor of shape (B, C, H, W), values in [0, 1] or [0, 255]
        
        Returns:
            Encoded features of shape (B, output_dim)
        """
        features = []
        
        # Process state
        if self.obs_mode in ["state", "state_image"]:
            if state is None:
                raise ValueError(f"state required for obs_mode='{self.obs_mode}'")
            if self.state_projector is not None:
                state_feat = self.state_projector(state)
            else:
                state_feat = state
            features.append(state_feat)
        
        # Process image
        if self.obs_mode in ["image", "state_image"]:
            if image is None:
                raise ValueError(f"image required for obs_mode='{self.obs_mode}'")
            image_feat = self.vision_encoder(image)
            features.append(image_feat)
        
        # Concatenate features
        if len(features) == 1:
            return features[0]
        else:
            return torch.cat(features, dim=-1)
    
    def get_state_dim(self) -> int:
        """Get effective state dimension (for algorithms that need it separately)."""
        if self.state_projector is not None:
            return self.state_dim  # Original state dim before projection
        return self.state_dim or 0
    
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state only (for value functions that only use state)."""
        if self.state_projector is not None:
            return self.state_projector(state)
        return state
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image only."""
        if self.vision_encoder is None:
            raise ValueError("No vision encoder available")
        return self.vision_encoder(image)


def create_obs_encoder(
    obs_mode: str,
    state_dim: Optional[int] = None,
    image_shape: Optional[Tuple[int, int, int]] = None,
    vision_encoder_type: str = "cnn",
    vision_output_dim: int = 128,
    freeze_vision: bool = False,
    **kwargs
) -> ObservationEncoder:
    """Factory function to create observation encoder.
    
    Args:
        obs_mode: Observation mode ("state", "image", or "state_image")
        state_dim: State dimension
        image_shape: Image shape (H, W, C)
        vision_encoder_type: Vision encoder type ("cnn" or "dinov2")
        vision_output_dim: Vision encoder output dimension
        freeze_vision: Whether to freeze vision encoder
        **kwargs: Additional arguments
    
    Returns:
        ObservationEncoder instance
    """
    vision_kwargs = {"output_dim": vision_output_dim}
    
    return ObservationEncoder(
        obs_mode=obs_mode,
        state_dim=state_dim,
        image_shape=image_shape,
        vision_encoder_type=vision_encoder_type,
        vision_encoder_kwargs=vision_kwargs,
        freeze_vision=freeze_vision,
        **kwargs
    )


class MultiModalMLP(nn.Module):
    """MLP that accepts multi-modal observations via ObservationEncoder.
    
    This is a convenience wrapper that combines an ObservationEncoder with
    an MLP head. Useful for value networks and simple policy networks.
    
    Args:
        obs_encoder: ObservationEncoder instance
        output_dim: Output dimension of MLP
        hidden_dims: Hidden layer dimensions
        activation: Activation function
    """
    
    def __init__(
        self,
        obs_encoder: ObservationEncoder,
        output_dim: int,
        hidden_dims: list = [256, 256],
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        
        self.obs_encoder = obs_encoder
        
        # Build MLP
        layers = []
        in_dim = obs_encoder.output_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                activation,
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        state: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        features = self.obs_encoder(state=state, image=image)
        return self.mlp(features)


if __name__ == "__main__":
    print("Testing ObservationEncoder...")
    
    # Test state-only mode
    print("\n1. Testing state-only mode:")
    encoder = ObservationEncoder(obs_mode="state", state_dim=31)
    state = torch.randn(32, 31)
    features = encoder(state=state)
    print(f"   Input: state {state.shape}")
    print(f"   Output: {features.shape}, output_dim={encoder.output_dim}")
    
    # Test image-only mode
    print("\n2. Testing image-only mode:")
    encoder = ObservationEncoder(
        obs_mode="image",
        image_shape=(128, 128, 3),
        vision_encoder_type="cnn",
        vision_encoder_kwargs={"output_dim": 128},
    )
    image = torch.randn(32, 3, 128, 128)
    features = encoder(image=image)
    print(f"   Input: image {image.shape}")
    print(f"   Output: {features.shape}, output_dim={encoder.output_dim}")
    
    # Test state+image mode
    print("\n3. Testing state+image mode:")
    encoder = ObservationEncoder(
        obs_mode="state_image",
        state_dim=31,
        image_shape=(128, 128, 3),
        vision_encoder_type="cnn",
        vision_encoder_kwargs={"output_dim": 128},
    )
    features = encoder(state=state, image=image)
    print(f"   Input: state {state.shape}, image {image.shape}")
    print(f"   Output: {features.shape}, output_dim={encoder.output_dim}")
    
    # Test factory function
    print("\n4. Testing create_obs_encoder factory:")
    encoder = create_obs_encoder(
        obs_mode="state_image",
        state_dim=31,
        image_shape=(128, 128, 3),
        vision_encoder_type="cnn",
        vision_output_dim=64,
    )
    features = encoder(state=state, image=image)
    print(f"   Output: {features.shape}, output_dim={encoder.output_dim}")
    
    # Test MultiModalMLP
    print("\n5. Testing MultiModalMLP:")
    value_net = MultiModalMLP(
        obs_encoder=encoder,
        output_dim=1,
        hidden_dims=[256, 256],
    )
    value = value_net(state=state, image=image)
    print(f"   Value output: {value.shape}")
    
    print("\n✓ All ObservationEncoder tests passed!")
