"""DPPO - Diffusion Policy Policy Optimization implementation.

Unified version supporting multiple observation modes:
- "state": State vector only
- "image": Image observation only
- "state_image": Both state and image (multimodal)
"""

from .agent import DPPOAgent

__all__ = ["DPPOAgent"]
