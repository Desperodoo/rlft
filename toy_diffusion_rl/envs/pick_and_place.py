"""
Pick-and-Place Environment Wrappers for Fetch Robot.

This module provides environment wrappers for the FetchPickAndPlace task
from gymnasium-robotics, supporting both state and image observations.

Reference:
    https://robotics.farama.org/envs/fetch/pick_and_place/
"""

import gymnasium as gym
import gymnasium_robotics  # Register robotics environments
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
import cv2


class FetchPickAndPlaceStateWrapper(gym.ObservationWrapper):
    """Wrapper that flattens Fetch dict observation into a 1D state vector.
    
    Concatenates:
        - observation: robot state (gripper pos, vel, etc.)
        - achieved_goal: current object position
        - desired_goal: target position
    
    Total state dimension: 25 (10 + 3 + 3 + 3 + 3 + 3 = 25 for FetchPickAndPlace-v3)
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # Get dimensions from wrapped env
        obs_space = env.observation_space
        obs_dim = obs_space["observation"].shape[0]
        achieved_dim = obs_space["achieved_goal"].shape[0]
        desired_dim = obs_space["desired_goal"].shape[0]
        
        total_dim = obs_dim + achieved_dim + desired_dim
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )
    
    def observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten dict observation to 1D array."""
        return np.concatenate([
            obs["observation"],
            obs["achieved_goal"],
            obs["desired_goal"]
        ]).astype(np.float32)


class FetchPickAndPlaceImageWrapper(gym.Wrapper):
    """Wrapper that provides image observations for Fetch environment.
    
    Supports three observation modes:
        - "image": Returns only the rendered image
        - "state_image": Returns dict with both state and image
        - "state": Returns only state (delegates to FetchPickAndPlaceStateWrapper)
    
    Image is resized to (image_size, image_size, 3) and returned as uint8.
    """
    
    def __init__(
        self,
        env: gym.Env,
        obs_mode: str = "state_image",
        image_size: int = 128,
    ):
        super().__init__(env)
        
        assert obs_mode in ["image", "state_image"], \
            f"obs_mode must be 'image' or 'state_image', got {obs_mode}"
        
        self.obs_mode = obs_mode
        self.image_size = image_size
        
        # Get state dimension from wrapped env
        obs_space = env.observation_space
        obs_dim = obs_space["observation"].shape[0]
        achieved_dim = obs_space["achieved_goal"].shape[0]
        desired_dim = obs_space["desired_goal"].shape[0]
        self.state_dim = obs_dim + achieved_dim + desired_dim
        
        # Define observation space
        image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(image_size, image_size, 3),
            dtype=np.uint8
        )
        
        if obs_mode == "image":
            self.observation_space = image_space
        else:  # state_image
            self.observation_space = gym.spaces.Dict({
                "state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.state_dim,),
                    dtype=np.float32
                ),
                "image": image_space
            })
    
    def _get_state(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract flattened state from dict observation."""
        return np.concatenate([
            obs["observation"],
            obs["achieved_goal"],
            obs["desired_goal"]
        ]).astype(np.float32)
    
    def _get_image(self) -> np.ndarray:
        """Render and resize image from environment."""
        # Render RGB image
        image = self.env.render()
        
        # Resize to target size
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA
            )
        
        return image.astype(np.uint8)
    
    def _make_obs(self, obs: Dict[str, np.ndarray]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Construct observation based on obs_mode."""
        image = self._get_image()
        
        if self.obs_mode == "image":
            return image
        else:  # state_image
            return {
                "state": self._get_state(obs),
                "image": image
            }
    
    def reset(self, **kwargs) -> Tuple[Union[np.ndarray, Dict], Dict]:
        """Reset environment and return observation."""
        obs, info = self.env.reset(**kwargs)
        return self._make_obs(obs), info
    
    def step(self, action: np.ndarray) -> Tuple[Union[np.ndarray, Dict], float, bool, bool, Dict]:
        """Step environment and return observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store original dict obs for expert policy access
        info["dict_obs"] = obs
        
        return self._make_obs(obs), reward, terminated, truncated, info


class FetchPickAndPlaceEnv(gym.Wrapper):
    """Unified wrapper for FetchPickAndPlace with configurable observation mode.
    
    This is the main entry point for creating Pick-and-Place environments.
    It automatically applies the appropriate wrappers based on obs_mode.
    
    Args:
        obs_mode: Observation mode - "state", "image", or "state_image"
        image_size: Size of rendered images (default: 128)
        reward_type: "sparse" or "dense" reward (default: "sparse")
    """
    
    def __init__(
        self,
        obs_mode: str = "state",
        image_size: int = 128,
        reward_type: str = "sparse",
        **kwargs
    ):
        # Create base Fetch environment with rgb_array render mode
        render_mode = "rgb_array" if obs_mode in ["image", "state_image"] else None
        
        base_env = gym.make(
            "FetchPickAndPlace-v4",
            render_mode=render_mode,
            reward_type=reward_type,
            **kwargs
        )
        
        # Apply appropriate wrapper
        if obs_mode == "state":
            wrapped_env = FetchPickAndPlaceStateWrapper(base_env)
        elif obs_mode in ["image", "state_image"]:
            wrapped_env = FetchPickAndPlaceImageWrapper(
                base_env,
                obs_mode=obs_mode,
                image_size=image_size
            )
        else:
            raise ValueError(f"Unknown obs_mode: {obs_mode}")
        
        super().__init__(wrapped_env)
        
        self.obs_mode = obs_mode
        self.image_size = image_size
        self.reward_type = reward_type
        
        # Store dimensions for easy access
        if obs_mode == "state":
            self.state_dim = self.observation_space.shape[0]
            self.image_shape = None
        elif obs_mode == "image":
            self.state_dim = None
            self.image_shape = (image_size, image_size, 3)
        else:  # state_image
            self.state_dim = self.observation_space["state"].shape[0]
            self.image_shape = (image_size, image_size, 3)
    
    def get_dict_obs(self) -> Dict[str, np.ndarray]:
        """Get the original dict observation from base Fetch env.
        
        Useful for scripted expert policies that need gripper/object positions.
        """
        # Access the underlying Fetch env
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        # Get observation from MuJoCo
        return base_env._get_obs()


def make_pick_and_place_env(
    backend: str = "fetch",
    obs_mode: str = "state",
    seed: int = 0,
    max_episode_steps: int = 50,
    image_size: int = 128,
    reward_type: str = "sparse",
) -> gym.Env:
    """Factory function to create Pick-and-Place environments.
    
    Args:
        backend: Environment backend ("fetch" is currently the only supported option)
        obs_mode: Observation mode - "state", "image", or "state_image"
        seed: Random seed for environment
        max_episode_steps: Maximum steps per episode (default: 50)
        image_size: Size of rendered images for image modes (default: 128)
        reward_type: "sparse" or "dense" reward (default: "sparse")
    
    Returns:
        Gymnasium environment with specified configuration
    
    Example:
        >>> env = make_pick_and_place_env(obs_mode="state_image", seed=42)
        >>> obs, info = env.reset()
        >>> print(obs["state"].shape, obs["image"].shape)
        (25,) (128, 128, 3)
    """
    if backend != "fetch":
        raise ValueError(f"Unknown backend: {backend}. Only 'fetch' is supported.")
    
    # Create wrapped environment
    env = FetchPickAndPlaceEnv(
        obs_mode=obs_mode,
        image_size=image_size,
        reward_type=reward_type,
        max_episode_steps=max_episode_steps,
    )
    
    # Set seed
    env.reset(seed=seed)
    
    return env


class FetchExpertPolicy:
    """Scripted expert policy for Fetch Pick-and-Place task.
    
    Uses a phased heuristic approach:
        Phase 1: Move gripper above the object
        Phase 2: Lower gripper to object and close gripper
        Phase 3: Lift object and move toward goal
        Phase 4: Lower to goal and release
    
    This policy is used for generating expert demonstration datasets.
    """
    
    def __init__(
        self,
        approach_height: float = 0.05,
        grip_threshold: float = 0.02,
        goal_threshold: float = 0.03,
        max_action: float = 1.0,
        gain: float = 10.0,
    ):
        """
        Args:
            approach_height: Height above object for approach (meters)
            grip_threshold: Distance threshold for gripping
            goal_threshold: Distance threshold for goal reaching
            max_action: Maximum action magnitude
            gain: Proportional control gain
        """
        self.approach_height = approach_height
        self.grip_threshold = grip_threshold
        self.goal_threshold = goal_threshold
        self.max_action = max_action
        self.gain = gain
        
        self.phase = 0
        self.has_object = False
    
    def reset(self):
        """Reset policy state for new episode."""
        self.phase = 0
        self.has_object = False
    
    def get_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute expert action given observation dict.
        
        Args:
            obs_dict: Dict with keys "observation", "achieved_goal", "desired_goal"
                observation[0:3]: gripper position
                observation[3:6]: object position
                observation[6:9]: object relative position
                achieved_goal[0:3]: current object position
                desired_goal[0:3]: target position
        
        Returns:
            Action array of shape (4,): [dx, dy, dz, gripper]
        """
        # Parse observation
        obs = obs_dict["observation"]
        gripper_pos = obs[0:3]
        object_pos = obs[3:6]
        desired_goal = obs_dict["desired_goal"]
        
        # Compute distances
        gripper_to_object = object_pos - gripper_pos
        object_to_goal = desired_goal - object_pos
        
        dist_gripper_object = np.linalg.norm(gripper_to_object)
        dist_object_goal = np.linalg.norm(object_to_goal)
        
        # Initialize action
        action = np.zeros(4)
        
        # Phase 0: Move above object
        if self.phase == 0:
            target = object_pos.copy()
            target[2] += self.approach_height
            
            delta = target - gripper_pos
            action[0:3] = self.gain * delta
            action[3] = 1.0  # Open gripper
            
            if np.linalg.norm(delta) < self.grip_threshold:
                self.phase = 1
        
        # Phase 1: Lower to object and close gripper
        elif self.phase == 1:
            delta = object_pos - gripper_pos
            action[0:3] = self.gain * delta
            
            if dist_gripper_object < self.grip_threshold:
                action[3] = -1.0  # Close gripper
                self.has_object = True
                self.phase = 2
            else:
                action[3] = 1.0  # Open gripper
        
        # Phase 2: Lift and move toward goal
        elif self.phase == 2:
            target = desired_goal.copy()
            target[2] += self.approach_height
            
            delta = target - gripper_pos
            action[0:3] = self.gain * delta
            action[3] = -1.0  # Keep gripper closed
            
            if np.linalg.norm(delta[:2]) < self.goal_threshold:
                self.phase = 3
        
        # Phase 3: Lower to goal and release
        elif self.phase == 3:
            delta = desired_goal - gripper_pos
            action[0:3] = self.gain * delta
            
            if dist_object_goal < self.goal_threshold:
                action[3] = 1.0  # Open gripper
            else:
                action[3] = -1.0  # Keep closed until at goal
        
        # Clip actions to valid range
        action[0:3] = np.clip(action[0:3], -self.max_action, self.max_action)
        action[3] = np.clip(action[3], -1.0, 1.0)
        
        return action.astype(np.float32)


if __name__ == "__main__":
    # Test environment creation
    print("Testing Pick-and-Place Environment...")
    
    # Test state mode
    print("\n1. Testing state mode:")
    env = make_pick_and_place_env(obs_mode="state", seed=42)
    obs, info = env.reset()
    print(f"   State shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")
    
    # Test with random actions
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Sample reward: {reward}")
    env.close()
    
    # Test state_image mode
    print("\n2. Testing state_image mode:")
    env = make_pick_and_place_env(obs_mode="state_image", seed=42)
    obs, info = env.reset()
    print(f"   State shape: {obs['state'].shape}")
    print(f"   Image shape: {obs['image'].shape}")
    env.close()
    
    # Test image mode
    print("\n3. Testing image mode:")
    env = make_pick_and_place_env(obs_mode="image", seed=42)
    obs, info = env.reset()
    print(f"   Image shape: {obs.shape}")
    env.close()
    
    # Test expert policy
    print("\n4. Testing expert policy:")
    env = make_pick_and_place_env(obs_mode="state", seed=42, reward_type="sparse")
    expert = FetchExpertPolicy()
    
    total_reward = 0
    successes = 0
    n_episodes = 10
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        expert.reset()
        ep_reward = 0
        
        for step in range(50):
            # Get dict obs for expert
            dict_obs = env.get_dict_obs()
            action = expert.get_action(dict_obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            if terminated or truncated:
                break
        
        total_reward += ep_reward
        if ep_reward > -50:  # Success if got positive reward
            successes += 1
    
    print(f"   Expert success rate: {successes}/{n_episodes}")
    print(f"   Average episode reward: {total_reward / n_episodes:.2f}")
    env.close()
    
    print("\n✓ All tests passed!")
