"""Common environment interfaces and registrations."""

from .point_mass_2d import PointMass2DEnv
from .pendulum_continuous_wrapper import PendulumContinuousWrapper
from .multimodal_particle import MultimodalParticleEnv, make_multimodal_env

# Gymnasium Robotics environments (optional, requires gymnasium-robotics)
try:
    from .pick_and_place import (
        make_pick_and_place_env,
        FetchPickAndPlaceEnv,
        FetchExpertPolicy,
    )
    GYMNASIUM_ROBOTICS_AVAILABLE = True
except ImportError:
    GYMNASIUM_ROBOTICS_AVAILABLE = False
    # Provide dummy functions for environments without gymnasium-robotics
    def make_pick_and_place_env(*args, **kwargs):
        raise ImportError(
            "gymnasium-robotics is not installed. Please use the rlft environment:\n"
            "  conda activate rlft\n"
            "Or install: pip install gymnasium-robotics"
        )
    FetchPickAndPlaceEnv = None
    FetchExpertPolicy = None

# ManiSkill3 environments (optional, requires rlft_ms3 environment)
try:
    from .maniskill_env import (
        make_maniskill_env,
        ManiSkillPickCubeEnv,
        ManiSkillExpertPolicy,
        check_maniskill_available,
    )
    MANISKILL_AVAILABLE = True
except ImportError:
    MANISKILL_AVAILABLE = False
    # Provide dummy functions for environments without ManiSkill3
    def make_maniskill_env(*args, **kwargs):
        raise ImportError(
            "ManiSkill3 is not installed. Please use the rlft_ms3 environment:\n"
            "  conda activate rlft_ms3\n"
            "Or install ManiSkill3: pip install mani_skill"
        )
    def check_maniskill_available():
        return False
    ManiSkillPickCubeEnv = None
    ManiSkillExpertPolicy = None

__all__ = [
    "PointMass2DEnv", 
    "PendulumContinuousWrapper", 
    "MultimodalParticleEnv",
    "make_multimodal_env",
    "make_env",
    # Gymnasium Robotics (optional)
    "make_pick_and_place_env",
    "FetchPickAndPlaceEnv",
    "FetchExpertPolicy",
    "GYMNASIUM_ROBOTICS_AVAILABLE",
    # ManiSkill3 (optional)
    "make_maniskill_env",
    "ManiSkillPickCubeEnv",
    "ManiSkillExpertPolicy",
    "check_maniskill_available",
    "MANISKILL_AVAILABLE",
]


def make_env(env_name: str, **kwargs):
    """Create an environment by name.
    
    Args:
        env_name: Name of the environment
            - 'point_mass_2d': 2D point mass navigation
            - 'pendulum': Continuous pendulum control
            - 'multimodal': Multimodal particle distribution
            - 'clusters', 'ring', 'double_ring', etc.: Specific distribution types
            - 'pick_and_place': Fetch Pick-and-Place manipulation task
            - 'maniskill' or 'pickcube': ManiSkill3 PickCube task (requires rlft_ms3 env)
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
    elif env_name == "pick_and_place":
        return make_pick_and_place_env(**kwargs)
    elif env_name in ["maniskill", "pickcube", "PickCube-v1"]:
        return make_maniskill_env(**kwargs)
    else:
        raise ValueError(
            f"Unknown environment: {env_name}. "
            f"Choose from: point_mass_2d, pendulum, multimodal, pick_and_place, maniskill, "
            f"or distribution types: {MultimodalParticleEnv.DISTRIBUTION_TYPES}"
        )
