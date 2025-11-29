"""Consistency Policy Q-Learning (CPQL) implementation.

Unified version supporting multiple observation modes:
- "state": State vector only
- "image": Image observation only
- "state_image": Both state and image (multimodal)
"""

from .agent import CPQLAgent

__all__ = ["CPQLAgent"]
