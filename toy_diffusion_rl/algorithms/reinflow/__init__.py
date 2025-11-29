"""ReinFlow - Online RL fine-tuning for flow-matching policies.

Unified version supporting multiple observation modes:
- "state": State vector only
- "image": Image observation only
- "state_image": Both state and image (multimodal)
"""

from .agent import ReinFlowAgent

__all__ = ["ReinFlowAgent"]
