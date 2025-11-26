"""Common environment interfaces and registrations."""

from .point_mass_2d import PointMass2DEnv
from .pendulum_continuous_wrapper import PendulumContinuousWrapper
from .multimodal_particle import MultimodalParticleEnv, make_multimodal_env

__all__ = [
    "PointMass2DEnv", 
    "PendulumContinuousWrapper", 
    "MultimodalParticleEnv",
    "make_multimodal_env",
    "make_env"
]


def make_env(env_name: str, **kwargs):
    """Create an environment by name.
    
    Args:
        env_name: Name of the environment
            - 'point_mass_2d': 2D point mass navigation
            - 'pendulum': Continuous pendulum control
            - 'multimodal': Multimodal particle distribution
            - 'clusters', 'ring', 'double_ring', etc.: Specific distribution types
        **kwargs: Additional environment parameters
        
    Returns:
        Environment instance
    """
    if env_name == "point_mass_2d":
        return PointMass2DEnv(**kwargs)
    elif env_name == "pendulum":
        return PendulumContinuousWrapper(**kwargs)
    elif env_name == "multimodal":
        return MultimodalParticleEnv(**kwargs)
    elif env_name in MultimodalParticleEnv.DISTRIBUTION_TYPES:
        return MultimodalParticleEnv(distribution_type=env_name, **kwargs)
    else:
        raise ValueError(
            f"Unknown environment: {env_name}. "
            f"Choose from: point_mass_2d, pendulum, multimodal, "
            f"or distribution types: {MultimodalParticleEnv.DISTRIBUTION_TYPES}"
        )
