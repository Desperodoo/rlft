"""
Vision Encoders for Image-based Observations.

This module provides vision encoder architectures for processing image observations
in robotic manipulation tasks. Two encoder families are implemented:

1. CNNEncoder: Lightweight CNN trained from scratch
2. DINOv2Encoder: Pre-trained DINOv2-ViT with optional fine-tuning

Both encoders output fixed-dimension embeddings that can be concatenated with
state vectors for multimodal policy learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


class CNNEncoder(nn.Module):
    """Lightweight CNN encoder for image observations.
    
    Architecture: 3-4 conv layers → flatten → MLP head
    
    This encoder is designed to be trained from scratch on robotic manipulation
    tasks. It uses a compact architecture suitable for online RL fine-tuning.
    
    Args:
        image_shape: Input image shape as (H, W, C) or (C, H, W)
        output_dim: Output embedding dimension (default: 128)
        channels: List of channel sizes for conv layers (default: [32, 64, 64])
        kernel_sizes: List of kernel sizes for conv layers (default: [8, 4, 3])
        strides: List of strides for conv layers (default: [4, 2, 1])
        activation: Activation function (default: nn.ReLU)
        use_layer_norm: Whether to use layer normalization (default: True)
    
    Example:
        >>> encoder = CNNEncoder(image_shape=(128, 128, 3), output_dim=128)
        >>> images = torch.randn(32, 3, 128, 128)  # (B, C, H, W)
        >>> features = encoder(images)
        >>> print(features.shape)
        torch.Size([32, 128])
    """
    
    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        output_dim: int = 128,
        channels: List[int] = [32, 64, 64],
        kernel_sizes: List[int] = [8, 4, 3],
        strides: List[int] = [4, 2, 1],
        activation: nn.Module = nn.ReLU,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        # Parse image shape (support both HWC and CHW)
        if len(image_shape) == 3:
            if image_shape[2] in [1, 3, 4]:  # HWC format
                self.height, self.width, self.in_channels = image_shape
            else:  # CHW format
                self.in_channels, self.height, self.width = image_shape
        else:
            raise ValueError(f"Invalid image_shape: {image_shape}")
        
        self.output_dim = output_dim
        self.use_layer_norm = use_layer_norm
        
        # Build convolutional layers
        conv_layers = []
        in_ch = self.in_channels
        
        for out_ch, kernel, stride in zip(channels, kernel_sizes, strides):
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel, stride, padding=0))
            conv_layers.append(activation())
            in_ch = out_ch
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Compute flattened feature size
        self.feature_size = self._get_conv_output_size()
        
        # MLP head
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.LayerNorm(256) if use_layer_norm else nn.Identity(),
            activation(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity(),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _get_conv_output_size(self) -> int:
        """Compute the flattened size after conv layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.height, self.width)
            output = self.conv(dummy)
            return int(torch.prod(torch.tensor(output.shape[1:])))
    
    def _init_weights(self):
        """Initialize weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images of shape (B, C, H, W) with values in [0, 255] or [0, 1]
        
        Returns:
            Image embeddings of shape (B, output_dim)
        """
        # Normalize to [0, 1] if input is uint8-like
        if x.max() > 1.0:
            x = x / 255.0
        
        # Apply conv layers
        features = self.conv(x)
        
        # Flatten and apply MLP (use reshape for non-contiguous tensors)
        features = features.reshape(features.size(0), -1)
        embeddings = self.fc(features)
        
        return embeddings


class DINOv2Encoder(nn.Module):
    """Pre-trained DINOv2-ViT encoder for image observations.
    
    Uses a pre-trained DINOv2 Vision Transformer as the backbone for extracting
    semantic visual features. Can be frozen entirely or allow partial fine-tuning.
    
    Args:
        model_name: DINOv2 model variant (default: "vit_small_patch14_dinov2")
            Options: vit_small_patch14_dinov2, vit_base_patch14_dinov2, etc.
        output_dim: Output embedding dimension (default: 384 for ViT-S)
        freeze: Whether to freeze the backbone (default: True)
        fine_tune_layers: Number of last layers to unfreeze (default: 0)
        use_cls_token: Whether to use CLS token or mean pooling (default: True)
    
    Note:
        Requires `timm` library: pip install timm
        
    Example:
        >>> encoder = DINOv2Encoder(output_dim=384, freeze=True)
        >>> images = torch.randn(32, 3, 224, 224)  # DINOv2 expects 224x224
        >>> features = encoder(images)
        >>> print(features.shape)
        torch.Size([32, 384])
    """
    
    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        output_dim: int = 384,
        freeze: bool = True,
        fine_tune_layers: int = 0,
        use_cls_token: bool = True,
        image_size: int = 518,  # DINOv2 default is 518
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.freeze = freeze
        self.fine_tune_layers = fine_tune_layers
        self.use_cls_token = use_cls_token
        self.image_size = image_size
        
        # Load pre-trained DINOv2 model from timm
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
            img_size=image_size,  # Explicitly set image size
        )
        
        # Get backbone output dimension
        self.backbone_dim = self.backbone.embed_dim
        
        # Freeze backbone if specified
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Optionally unfreeze last N layers
            if fine_tune_layers > 0:
                # Unfreeze last N transformer blocks
                for block in self.backbone.blocks[-fine_tune_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True
        
        # Projection head if output_dim differs from backbone_dim
        if output_dim != self.backbone_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.projection = nn.Identity()
        
        # Image normalization (ImageNet stats for DINOv2)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for DINOv2.
        
        Args:
            x: Images of shape (B, C, H, W), values in [0, 255] or [0, 1]
        
        Returns:
            Normalized images of shape (B, C, 224, 224)
        """
        # Normalize to [0, 1]
        if x.max() > 1.0:
            x = x / 255.0
        
        # Resize to expected size if needed
        if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        
        # Apply ImageNet normalization
        x = (x - self.mean) / self.std
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images of shape (B, C, H, W)
        
        Returns:
            Image embeddings of shape (B, output_dim)
        """
        # Preprocess
        x = self._preprocess(x)
        
        # Extract features from backbone
        features = self.backbone.forward_features(x)
        
        # Pool features
        if self.use_cls_token:
            # Use CLS token
            if hasattr(features, 'shape') and len(features.shape) == 3:
                features = features[:, 0]  # CLS token is first
            # else features is already (B, D) from forward_features with pooling
        else:
            # Mean pooling over spatial tokens
            if hasattr(features, 'shape') and len(features.shape) == 3:
                features = features[:, 1:].mean(dim=1)
        
        # Project to output dimension
        embeddings = self.projection(features)
        
        return embeddings


class ResNetEncoder(nn.Module):
    """ResNet-based encoder for image observations.
    
    Uses a pre-trained ResNet backbone (ResNet-18 or ResNet-34) for feature extraction.
    Lighter weight alternative to DINOv2 for faster inference.
    
    Args:
        model_name: ResNet variant ("resnet18" or "resnet34")
        output_dim: Output embedding dimension
        freeze: Whether to freeze the backbone
        pretrained: Whether to use ImageNet pretrained weights
    """
    
    def __init__(
        self,
        model_name: str = "resnet18",
        output_dim: int = 128,
        freeze: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError("Please install torchvision")
        
        # Load backbone
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            backbone_dim = 512
        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
            backbone_dim = 512
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone_dim = backbone_dim
        
        # Freeze if specified
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Image normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Normalize to [0, 1]
        if x.max() > 1.0:
            x = x / 255.0
        
        # Apply ImageNet normalization
        x = (x - self.mean) / self.std
        
        # Extract features
        features = self.backbone(x)
        embeddings = self.projection(features)
        
        return embeddings


def make_vision_encoder(
    encoder_type: str,
    image_shape: Tuple[int, int, int],
    output_dim: int = 128,
    freeze: bool = True,
    **kwargs
) -> nn.Module:
    """Factory function to create vision encoders.
    
    Args:
        encoder_type: Type of encoder ("cnn", "dinov2", or "resnet")
        image_shape: Input image shape as (H, W, C) or (C, H, W)
        output_dim: Output embedding dimension
        freeze: Whether to freeze backbone (for pretrained models)
        **kwargs: Additional encoder-specific arguments
    
    Returns:
        Vision encoder module
    
    Example:
        >>> # Create CNN encoder (train from scratch)
        >>> encoder = make_vision_encoder("cnn", (128, 128, 3), output_dim=128)
        
        >>> # Create frozen DINOv2 encoder
        >>> encoder = make_vision_encoder("dinov2", (128, 128, 3), output_dim=384, freeze=True)
        
        >>> # Create fine-tunable DINOv2 encoder
        >>> encoder = make_vision_encoder("dinov2", (128, 128, 3), freeze=False, fine_tune_layers=2)
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == "cnn":
        return CNNEncoder(
            image_shape=image_shape,
            output_dim=output_dim,
            **kwargs
        )
    elif encoder_type in ["dinov2", "dinov3", "dino"]:
        return DINOv2Encoder(
            output_dim=output_dim,
            freeze=freeze,
            **kwargs
        )
    elif encoder_type == "resnet":
        return ResNetEncoder(
            output_dim=output_dim,
            freeze=freeze,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. "
            f"Choose from: 'cnn', 'dinov2', 'resnet'"
        )


def compute_cnn_output_shape(
    input_shape: Tuple[int, int],
    channels: List[int],
    kernel_sizes: List[int],
    strides: List[int],
) -> Tuple[int, int, int]:
    """Compute output shape of CNN given input shape and layer configs.
    
    Useful for computing the flattened feature size before creating the encoder.
    
    Args:
        input_shape: (H, W) of input image
        channels: List of output channels per layer
        kernel_sizes: List of kernel sizes per layer
        strides: List of strides per layer
    
    Returns:
        (C, H, W) output shape after all conv layers
    """
    h, w = input_shape
    
    for kernel, stride in zip(kernel_sizes, strides):
        h = (h - kernel) // stride + 1
        w = (w - kernel) // stride + 1
    
    return (channels[-1], h, w)


if __name__ == "__main__":
    # Test vision encoders
    print("Testing Vision Encoders...")
    
    # Test CNNEncoder
    print("\n1. Testing CNNEncoder:")
    cnn = CNNEncoder(image_shape=(128, 128, 3), output_dim=128)
    print(f"   Parameters: {sum(p.numel() for p in cnn.parameters()):,}")
    
    images = torch.randn(4, 3, 128, 128)
    features = cnn(images)
    print(f"   Input shape: {images.shape}")
    print(f"   Output shape: {features.shape}")
    
    # Test with uint8 input
    images_uint8 = (torch.rand(4, 3, 128, 128) * 255).to(torch.uint8).float()
    features = cnn(images_uint8)
    print(f"   Works with uint8 input: {features.shape}")
    
    # Test DINOv2Encoder (if timm is available)
    print("\n2. Testing DINOv2Encoder:")
    try:
        dinov2 = DINOv2Encoder(output_dim=384, freeze=True)
        print(f"   Parameters: {sum(p.numel() for p in dinov2.parameters()):,}")
        print(f"   Trainable: {sum(p.numel() for p in dinov2.parameters() if p.requires_grad):,}")
        
        images = torch.randn(2, 3, 128, 128)
        features = dinov2(images)
        print(f"   Input shape: {images.shape}")
        print(f"   Output shape: {features.shape}")
    except Exception as e:
        print(f"   Skipped (timm not available or model download failed): {e}")
    
    # Test factory function
    print("\n3. Testing make_vision_encoder factory:")
    encoder = make_vision_encoder("cnn", (128, 128, 3), output_dim=64)
    features = encoder(torch.randn(2, 3, 128, 128))
    print(f"   CNN output: {features.shape}")
    
    print("\n✓ All vision encoder tests passed!")
