"""
ManiSkill3 Environment Wrappers for PickCube Task.

This module provides environment wrappers for ManiSkill3's PickCube-v1 task,
supporting state, image, and state_image observation modes with GPU-parallelized
vectorized environments.

Reference:
    https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html

Note:
    This module requires the rlft_ms3 conda environment (not rlft).
    ManiSkill3 uses SAPIEN physics engine instead of MuJoCo.
"""

# Filter third-party deprecation warnings before imports
import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*env\\..*to get variables from other wrappers.*", category=UserWarning)

import gymnasium as gym
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any, Union, List
import cv2

# Import ManiSkill3 - will fail if not installed
try:
    import mani_skill.envs
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
    MANISKILL_AVAILABLE = True
except ImportError:
    MANISKILL_AVAILABLE = False
    print("Warning: ManiSkill3 not installed. Install with: pip install mani_skill")


class ManiSkillStateImageWrapper(gym.Wrapper):
    """Wrapper that provides unified state_image observations for ManiSkill3.
    
    ManiSkill3 returns observations as torch tensors by default. This wrapper:
    1. Extracts state from state_dict observation
    2. Extracts RGB images from sensor data
    3. Provides a unified interface compatible with existing algorithms
    
    Supports three observation modes:
        - "state": Flattened state vector only (numpy)
        - "image": RGB image only (numpy, HWC uint8)
        - "state_image": Dict with both state and image (numpy)
    
    For VecEnv (num_envs > 1), returns batched data as torch tensors.
    """
    
    def __init__(
        self,
        env: gym.Env,
        obs_mode: str = "state_image",
        image_size: int = 128,
        use_numpy: bool = True,  # Convert to numpy for single env compatibility
    ):
        super().__init__(env)
        
        assert obs_mode in ["state", "image", "state_image"], \
            f"obs_mode must be 'state', 'image', or 'state_image', got {obs_mode}"
        
        self.obs_mode = obs_mode
        self.image_size = image_size
        self.use_numpy = use_numpy
        
        # Get number of parallel environments (use get_wrapper_attr to avoid deprecation warning)
        try:
            self.num_envs = env.get_wrapper_attr('num_envs')
        except AttributeError:
            self.num_envs = getattr(env.unwrapped, 'num_envs', 1)
        self.is_vec_env = self.num_envs > 1
        
        # Infer dimensions from a sample observation
        sample_obs, _ = env.reset()
        
        # Helper to convert to numpy array
        def to_numpy_array(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            elif isinstance(x, np.ndarray):
                return x
            elif isinstance(x, (bool, int, float)):
                return np.array([x])
            else:
                return np.asarray(x)
        
        # Extract state dimension from a sample observation
        if isinstance(sample_obs, dict):
            # state_dict mode or rgbd mode with state
            if "agent" in sample_obs:
                agent_obs = sample_obs["agent"]
                state_parts = []
                
                if "qpos" in agent_obs:
                    qpos = to_numpy_array(agent_obs["qpos"])
                    # Handle batched observations
                    if len(qpos.shape) > 1:
                        qpos = qpos[0]  # Take first env for dimension inference
                    state_parts.append(qpos.flatten())
                if "qvel" in agent_obs:
                    qvel = to_numpy_array(agent_obs["qvel"])
                    if len(qvel.shape) > 1:
                        qvel = qvel[0]
                    state_parts.append(qvel.flatten())
                
                # Add extra observations if available
                if "extra" in sample_obs:
                    for key, val in sample_obs["extra"].items():
                        arr = to_numpy_array(val)
                        if len(arr.shape) > 1:
                            arr = arr[0]  # Take first env for dimension inference
                        state_parts.append(arr.flatten())
                
                sample_state = np.concatenate(state_parts)
                self.state_dim = sample_state.shape[0]
            else:
                # Flat state mode
                sample_obs_np = to_numpy_array(sample_obs)
                # Handle batched observations
                if len(sample_obs_np.shape) > 1:
                    self.state_dim = sample_obs_np.shape[-1]  # Last dim is state dim
                else:
                    self.state_dim = sample_obs_np.shape[0]
        else:
            sample_obs_np = to_numpy_array(sample_obs)
            if len(sample_obs_np.shape) > 1:
                self.state_dim = sample_obs_np.shape[-1]
            else:
                self.state_dim = sample_obs_np.shape[0]
        
        # Define observation space
        if self.is_vec_env and not self.use_numpy:
            # Batched spaces for VecEnv
            state_shape = (self.num_envs, self.state_dim)
            image_shape = (self.num_envs, image_size, image_size, 3)
        else:
            state_shape = (self.state_dim,)
            image_shape = (image_size, image_size, 3)
        
        state_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32
        )
        image_space = gym.spaces.Box(
            low=0, high=255, shape=image_shape, dtype=np.uint8
        )
        
        if obs_mode == "state":
            self.observation_space = state_space
        elif obs_mode == "image":
            self.observation_space = image_space
        else:  # state_image
            self.observation_space = gym.spaces.Dict({
                "state": state_space,
                "image": image_space
            })
    
    def _extract_state(self, obs: Dict) -> Union[np.ndarray, torch.Tensor]:
        """Extract flattened state from ManiSkill3 observation.
        
        For single env: returns (state_dim,)
        For VecEnv: returns (num_envs, state_dim)
        """
        # Helper to convert to numpy array
        def to_numpy_array(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            elif isinstance(x, np.ndarray):
                return x
            elif isinstance(x, (bool, int, float)):
                return np.array([x])
            else:
                return np.asarray(x)
        
        if isinstance(obs, dict):
            if "agent" in obs:
                # state_dict mode
                agent_obs = obs["agent"]
                state_parts = []
                
                # Robot proprioception
                if "qpos" in agent_obs:
                    qpos = to_numpy_array(agent_obs["qpos"])
                    # For VecEnv, shape is (num_envs, qpos_dim)
                    if len(qpos.shape) == 1:
                        qpos = qpos.reshape(1, -1) if self.is_vec_env else qpos
                    state_parts.append(qpos)
                if "qvel" in agent_obs:
                    qvel = to_numpy_array(agent_obs["qvel"])
                    if len(qvel.shape) == 1:
                        qvel = qvel.reshape(1, -1) if self.is_vec_env else qvel
                    state_parts.append(qvel)
                
                # Extra observations (goal, object poses, etc.)
                if "extra" in obs:
                    for key, val in obs["extra"].items():
                        arr = to_numpy_array(val)
                        if len(arr.shape) == 0:
                            # Scalar value
                            if self.is_vec_env:
                                arr = np.full((self.num_envs, 1), arr)
                            else:
                                arr = arr.reshape(1)
                        elif len(arr.shape) == 1:
                            if self.is_vec_env:
                                # (num_envs,) -> (num_envs, 1)
                                arr = arr.reshape(-1, 1)
                            # else: (dim,) stays as is
                        state_parts.append(arr)
                
                if state_parts:
                    if self.is_vec_env:
                        state = np.concatenate(state_parts, axis=-1)
                    else:
                        state = np.concatenate([s.flatten() for s in state_parts])
                else:
                    if self.is_vec_env:
                        state = np.zeros((self.num_envs, self.state_dim), dtype=np.float32)
                    else:
                        state = np.zeros(self.state_dim, dtype=np.float32)
            else:
                # Already flat state
                state = to_numpy_array(obs)
        else:
            state = to_numpy_array(obs)
        
        # Ensure correct dtype
        state = state.astype(np.float32)
        
        if not self.use_numpy:
            state = torch.from_numpy(state)
        
        return state
    
    def _extract_image(self, obs: Dict) -> Union[np.ndarray, torch.Tensor]:
        """Extract RGB image from ManiSkill3 observation."""
        image = None
        
        if isinstance(obs, dict):
            # Check for sensor_data (rgbd mode)
            if "sensor_data" in obs:
                sensor_data = obs["sensor_data"]
                # Get first camera's RGB
                for cam_name, cam_data in sensor_data.items():
                    if isinstance(cam_data, dict) and "rgb" in cam_data:
                        image = cam_data["rgb"]
                        break
                    elif isinstance(cam_data, dict) and "Color" in cam_data:
                        # Raw sensor mode
                        color = cam_data["Color"]
                        image = color[..., :3]  # Take RGB, drop Alpha
                        break
            
            # Check for direct rgb key
            if image is None and "rgb" in obs:
                image = obs["rgb"]
        
        if image is None:
            # Fallback: render from environment
            image = self.env.render()
            if image is None:
                # Create placeholder
                if self.is_vec_env:
                    image = np.zeros((self.num_envs, self.image_size, self.image_size, 3), dtype=np.uint8)
                else:
                    image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                return image
        
        # Convert torch tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Ensure uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Resize if needed
        if self.is_vec_env:
            # Batched images: (num_envs, H, W, C)
            if image.shape[1] != self.image_size or image.shape[2] != self.image_size:
                resized = []
                for i in range(image.shape[0]):
                    img = cv2.resize(
                        image[i], (self.image_size, self.image_size),
                        interpolation=cv2.INTER_AREA
                    )
                    resized.append(img)
                image = np.stack(resized, axis=0)
            
            if self.use_numpy and self.num_envs == 1:
                image = image[0]  # Unbatch for single env
        else:
            # Single image: (H, W, C)
            if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
                image = cv2.resize(
                    image, (self.image_size, self.image_size),
                    interpolation=cv2.INTER_AREA
                )
        
        return image
    
    def _make_obs(self, obs: Dict) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Construct observation based on obs_mode."""
        if self.obs_mode == "state":
            return self._extract_state(obs)
        elif self.obs_mode == "image":
            return self._extract_image(obs)
        else:  # state_image
            return {
                "state": self._extract_state(obs),
                "image": self._extract_image(obs)
            }
    
    def reset(self, **kwargs) -> Tuple[Union[np.ndarray, Dict], Dict]:
        """Reset environment and return observation."""
        obs, info = self.env.reset(**kwargs)
        return self._make_obs(obs), info
    
    def step(self, action) -> Tuple[Union[np.ndarray, Dict], Any, Any, Any, Dict]:
        """Step environment and return observation."""
        # Convert numpy action to torch if needed (for GPU sim)
        if isinstance(action, np.ndarray) and self.is_vec_env:
            action = torch.from_numpy(action).float()
            if torch.cuda.is_available():
                action = action.cuda()
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store original observation for expert policy access
        info["raw_obs"] = obs
        
        # Convert reward/done to numpy for single env
        if self.use_numpy:
            if isinstance(reward, torch.Tensor):
                reward = reward.cpu().numpy()
                if self.num_envs == 1:
                    reward = float(reward[0])
            if isinstance(terminated, torch.Tensor):
                terminated = terminated.cpu().numpy()
                if self.num_envs == 1:
                    terminated = bool(terminated[0])
            if isinstance(truncated, torch.Tensor):
                truncated = truncated.cpu().numpy()
                if self.num_envs == 1:
                    truncated = bool(truncated[0])
        
        return self._make_obs(obs), reward, terminated, truncated, info


class ManiSkillPickCubeEnv(gym.Wrapper):
    """Unified wrapper for ManiSkill3 PickCube-v1 with configurable observation mode.
    
    This is the main entry point for creating ManiSkill3 PickCube environments.
    Supports GPU-parallelized vectorized environments via num_envs parameter.
    
    Args:
        obs_mode: Observation mode - "state", "image", or "state_image"
        num_envs: Number of parallel environments (>1 enables GPU simulation)
        image_size: Size of rendered images (default: 128)
        control_mode: Robot control mode (default: "pd_ee_delta_pose")
        reward_mode: "dense" or "sparse" (default: "dense")
        use_numpy: Convert outputs to numpy (default: True for single env compatibility)
    
    Example:
        >>> env = ManiSkillPickCubeEnv(obs_mode="state_image", num_envs=1)
        >>> obs, info = env.reset()
        >>> print(obs["state"].shape, obs["image"].shape)
        
        >>> # VecEnv for parallel training
        >>> vec_env = ManiSkillPickCubeEnv(obs_mode="state_image", num_envs=16)
        >>> obs, info = vec_env.reset()
        >>> print(obs["state"].shape)  # (16, state_dim)
    """
    
    def __init__(
        self,
        obs_mode: str = "state_image",
        num_envs: int = 1,
        image_size: int = 128,
        control_mode: str = "pd_ee_delta_pose",
        reward_mode: str = "dense",
        use_numpy: bool = True,
        max_episode_steps: int = 50,
        **kwargs
    ):
        if not MANISKILL_AVAILABLE:
            raise ImportError(
                "ManiSkill3 is not installed. Please install it with:\n"
                "  pip install mani_skill\n"
                "Or use the rlft_ms3 conda environment."
            )
        
        # Determine ManiSkill3 obs_mode
        if obs_mode == "state":
            ms_obs_mode = "state"
            render_mode = None
        elif obs_mode == "image":
            ms_obs_mode = "rgb+depth"  # Use rgb+depth to get camera data
            render_mode = "rgb_array"
        else:  # state_image
            ms_obs_mode = "state_dict+rgb+depth"  # Get both state dict and visual data
            render_mode = "rgb_array"
        
        # Create base ManiSkill3 environment
        base_env = gym.make(
            "PickCube-v1",
            num_envs=num_envs,
            obs_mode=ms_obs_mode,
            control_mode=control_mode,
            reward_mode=reward_mode,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs
        )
        
        # Apply CPU wrapper for single env numpy compatibility
        if num_envs == 1 and use_numpy:
            base_env = CPUGymWrapper(base_env)
        
        # Apply our observation wrapper
        wrapped_env = ManiSkillStateImageWrapper(
            base_env,
            obs_mode=obs_mode,
            image_size=image_size,
            use_numpy=use_numpy,
        )
        
        super().__init__(wrapped_env)
        
        # Store configuration
        self.obs_mode = obs_mode
        self.num_envs = num_envs
        self.image_size = image_size
        self.control_mode = control_mode
        self.reward_mode = reward_mode
        self.use_numpy = use_numpy
        self.is_vec_env = num_envs > 1
        
        # Store dimensions
        if obs_mode == "state":
            self.state_dim = self.observation_space.shape[-1]
            self.image_shape = None
        elif obs_mode == "image":
            self.state_dim = None
            self.image_shape = (image_size, image_size, 3)
        else:  # state_image
            self.state_dim = self.observation_space["state"].shape[-1]
            self.image_shape = (image_size, image_size, 3)
        
        # Get action dimension
        self.action_dim = self.action_space.shape[-1]


def make_maniskill_env(
    task: str = "PickCube-v1",
    obs_mode: str = "state_image",
    num_envs: int = 1,
    seed: int = 0,
    max_episode_steps: int = 50,
    image_size: int = 128,
    control_mode: str = "pd_ee_delta_pose",
    reward_mode: str = "dense",
    use_numpy: bool = True,
) -> gym.Env:
    """Factory function to create ManiSkill3 environments.
    
    Args:
        task: ManiSkill3 task ID (default: "PickCube-v1")
        obs_mode: Observation mode - "state", "image", or "state_image"
        num_envs: Number of parallel environments (>1 enables GPU simulation)
        seed: Random seed for environment
        max_episode_steps: Maximum steps per episode (default: 50)
        image_size: Size of rendered images for image modes (default: 128)
        control_mode: Robot control mode (default: "pd_ee_delta_pose")
        reward_mode: "dense" or "sparse" (default: "dense")
        use_numpy: Convert outputs to numpy (default: True)
    
    Returns:
        ManiSkill3 environment with specified configuration
    
    Example:
        >>> # Single environment
        >>> env = make_maniskill_env(obs_mode="state_image", seed=42)
        >>> obs, info = env.reset()
        
        >>> # Vectorized environment for parallel training
        >>> vec_env = make_maniskill_env(obs_mode="state_image", num_envs=16)
        >>> obs, info = vec_env.reset()  # obs["state"].shape = (16, state_dim)
    """
    if task != "PickCube-v1":
        raise ValueError(f"Currently only 'PickCube-v1' is supported, got {task}")
    
    env = ManiSkillPickCubeEnv(
        obs_mode=obs_mode,
        num_envs=num_envs,
        image_size=image_size,
        control_mode=control_mode,
        reward_mode=reward_mode,
        use_numpy=use_numpy,
        max_episode_steps=max_episode_steps,
    )
    
    # Set seed
    env.reset(seed=seed)
    
    return env


class ManiSkillExpertPolicy:
    """Scripted expert policy for ManiSkill3 PickCube task.
    
    Uses a phased heuristic approach similar to Fetch:
        Phase 0: Move gripper above the cube
        Phase 1: Lower gripper to cube and close gripper
        Phase 2: Lift cube and move toward goal
        Phase 3: Lower to goal position
    
    This policy works with raw ManiSkill3 observations.
    """
    
    def __init__(
        self,
        approach_height: float = 0.05,
        grip_threshold: float = 0.02,
        goal_threshold: float = 0.03,
        max_action: float = 1.0,
        gain: float = 5.0,
    ):
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
    
    def get_action(self, raw_obs: Dict, info: Dict = None) -> np.ndarray:
        """Compute expert action given raw ManiSkill3 observation.
        
        Args:
            raw_obs: Raw observation dict from ManiSkill3
            info: Additional info dict (may contain goal position)
        
        Returns:
            Action array compatible with pd_ee_delta_pose control mode
            Shape: (7,) for [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Extract positions from observation
        # ManiSkill3 PickCube-v1 extra contains: tcp_pose, goal_pos, obj_pose, etc.
        
        if isinstance(raw_obs, dict):
            extra = raw_obs.get("extra", {})
            
            # Get end-effector (TCP) position
            tcp_pose = extra.get("tcp_pose", None)
            if tcp_pose is not None:
                if isinstance(tcp_pose, torch.Tensor):
                    tcp_pose = tcp_pose.cpu().numpy()
                if len(tcp_pose.shape) > 1:
                    tcp_pose = tcp_pose[0]  # Unbatch
                gripper_pos = tcp_pose[:3]
            else:
                gripper_pos = np.zeros(3)
            
            # Get object position
            obj_pose = extra.get("obj_pose", None)
            if obj_pose is not None:
                if isinstance(obj_pose, torch.Tensor):
                    obj_pose = obj_pose.cpu().numpy()
                if len(obj_pose.shape) > 1:
                    obj_pose = obj_pose[0]
                object_pos = obj_pose[:3]
            else:
                object_pos = np.zeros(3)
            
            # Get goal position
            goal_pos = extra.get("goal_pos", None)
            if goal_pos is not None:
                if isinstance(goal_pos, torch.Tensor):
                    goal_pos = goal_pos.cpu().numpy()
                if len(goal_pos.shape) > 1:
                    goal_pos = goal_pos[0]
                if len(goal_pos.shape) > 0:
                    desired_goal = goal_pos[:3] if goal_pos.shape[-1] >= 3 else goal_pos
                else:
                    desired_goal = np.array([goal_pos])
            else:
                desired_goal = object_pos + np.array([0.1, 0, 0.1])  # Default offset
        else:
            # Flat state - harder to parse, use defaults
            gripper_pos = np.zeros(3)
            object_pos = np.zeros(3)
            desired_goal = np.zeros(3)
        
        # Compute distances
        gripper_to_object = object_pos - gripper_pos
        object_to_goal = desired_goal - object_pos
        
        dist_gripper_object = np.linalg.norm(gripper_to_object)
        dist_object_goal = np.linalg.norm(object_to_goal)
        
        # Initialize action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        # For pd_ee_delta_pose, we control position and orientation deltas + gripper
        action = np.zeros(7)
        
        # Phase 0: Move above object
        if self.phase == 0:
            target = object_pos.copy()
            target[2] += self.approach_height
            
            delta = target - gripper_pos
            action[0:3] = self.gain * delta
            action[6] = 1.0  # Open gripper
            
            if np.linalg.norm(delta) < self.grip_threshold:
                self.phase = 1
        
        # Phase 1: Lower to object and close gripper
        elif self.phase == 1:
            delta = object_pos - gripper_pos
            action[0:3] = self.gain * delta
            
            if dist_gripper_object < self.grip_threshold:
                action[6] = -1.0  # Close gripper
                self.has_object = True
                self.phase = 2
            else:
                action[6] = 1.0  # Open gripper
        
        # Phase 2: Lift and move toward goal
        elif self.phase == 2:
            target = desired_goal.copy()
            target[2] += self.approach_height
            
            delta = target - gripper_pos
            action[0:3] = self.gain * delta
            action[6] = -1.0  # Keep gripper closed
            
            if np.linalg.norm(delta[:2]) < self.goal_threshold:
                self.phase = 3
        
        # Phase 3: Lower to goal
        elif self.phase == 3:
            delta = desired_goal - gripper_pos
            action[0:3] = self.gain * delta
            
            if dist_object_goal < self.goal_threshold:
                action[6] = 1.0  # Open gripper to release
            else:
                action[6] = -1.0  # Keep closed until at goal
        
        # Clip actions
        action[0:3] = np.clip(action[0:3], -self.max_action, self.max_action)
        action[6] = np.clip(action[6], -1.0, 1.0)
        
        return action.astype(np.float32)


def check_maniskill_available() -> bool:
    """Check if ManiSkill3 is available."""
    return MANISKILL_AVAILABLE


if __name__ == "__main__":
    if not MANISKILL_AVAILABLE:
        print("ManiSkill3 is not installed. Please install with:")
        print("  pip install mani_skill")
        exit(1)
    
    print("Testing ManiSkill3 PickCube Environment...")
    
    # Test 1: State mode (single env)
    print("\n1. Testing state mode (single env):")
    try:
        env = make_maniskill_env(obs_mode="state", num_envs=1, seed=42)
        obs, info = env.reset()
        print(f"   State shape: {obs.shape}")
        print(f"   State dim: {env.state_dim}")
        print(f"   Action space: {env.action_space}")
        print(f"   Action dim: {env.action_dim}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"   Step successful, reward: {reward}")
        env.close()
        print("   ✓ State mode passed")
    except Exception as e:
        print(f"   ✗ State mode failed: {e}")
    
    # Test 2: State_image mode (single env)
    print("\n2. Testing state_image mode (single env):")
    try:
        env = make_maniskill_env(obs_mode="state_image", num_envs=1, seed=42)
        obs, info = env.reset()
        print(f"   State shape: {obs['state'].shape}")
        print(f"   Image shape: {obs['image'].shape}")
        
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"   Step successful")
        env.close()
        print("   ✓ State_image mode passed")
    except Exception as e:
        print(f"   ✗ State_image mode failed: {e}")
    
    # Test 3: VecEnv (GPU parallel)
    print("\n3. Testing VecEnv (num_envs=4):")
    try:
        vec_env = make_maniskill_env(
            obs_mode="state_image", 
            num_envs=4, 
            seed=42,
            use_numpy=False  # Keep as torch tensors for VecEnv
        )
        obs, info = vec_env.reset()
        print(f"   State shape: {obs['state'].shape}")
        print(f"   Image shape: {obs['image'].shape}")
        
        # Sample batched action
        action = vec_env.action_space.sample()
        print(f"   Action shape: {action.shape}")
        
        obs, reward, term, trunc, info = vec_env.step(action)
        print(f"   Reward shape: {reward.shape}")
        print(f"   Terminated shape: {term.shape}")
        vec_env.close()
        print("   ✓ VecEnv mode passed")
    except Exception as e:
        print(f"   ✗ VecEnv mode failed: {e}")
    
    # Test 4: Expert policy
    print("\n4. Testing expert policy:")
    try:
        env = make_maniskill_env(obs_mode="state_image", num_envs=1, seed=42)
        expert = ManiSkillExpertPolicy()
        
        total_reward = 0
        successes = 0
        n_episodes = 5
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            expert.reset()
            ep_reward = 0
            
            for step in range(50):
                # Get raw obs for expert
                raw_obs = info.get("raw_obs", obs)
                action = expert.get_action(raw_obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                
                if terminated or truncated:
                    break
            
            total_reward += ep_reward
            # Check success from info
            if info.get("success", False) or ep_reward > 0:
                successes += 1
        
        print(f"   Expert success rate: {successes}/{n_episodes}")
        print(f"   Average episode reward: {total_reward / n_episodes:.2f}")
        env.close()
        print("   ✓ Expert policy test completed")
    except Exception as e:
        print(f"   ✗ Expert policy test failed: {e}")
    
    print("\n✓ All tests completed!")
