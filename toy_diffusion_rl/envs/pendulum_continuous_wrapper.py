"""
Pendulum Continuous Control Wrapper

Wraps the Gymnasium Pendulum-v1 environment with additional 
functionality for expert data collection and unified interface.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class PendulumContinuousWrapper(gym.Wrapper):
    """Wrapper for Gymnasium Pendulum-v1 environment.
    
    Provides:
    - Consistent interface with other toy environments
    - Expert action generation (energy-based swing-up controller)
    - Expert dataset collection
    
    Args:
        max_episode_steps: Maximum steps per episode (default: 200)
        render_mode: Render mode for visualization
    """
    
    def __init__(
        self,
        max_episode_steps: int = 200,
        render_mode: Optional[str] = None
    ):
        env = gym.make("Pendulum-v1", render_mode=render_mode)
        super().__init__(env)
        
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        
        # Normalize action space to [-1, 1]
        self._action_scale = 2.0  # Pendulum uses [-2, 2]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        self.steps = 0
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.astype(np.float32), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step with normalized action."""
        # Scale action from [-1, 1] to [-2, 2]
        scaled_action = np.clip(action, -1.0, 1.0) * self._action_scale
        
        obs, reward, terminated, truncated, info = self.env.step(scaled_action)
        self.steps += 1
        
        # Normalize reward to be more comparable
        reward = reward / 16.0  # Pendulum reward is in [-16.27..., 0]
        
        truncated = truncated or (self.steps >= self.max_episode_steps)
        
        return obs.astype(np.float32), float(reward), terminated, truncated, info
    
    def get_expert_action(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Get expert action using energy-based swing-up controller.
        
        The controller uses energy shaping to swing up and balance the pendulum.
        
        Args:
            state: State [cos(theta), sin(theta), theta_dot]. If None, gets from env.
            
        Returns:
            Expert action in [-1, 1]
        """
        if state is None:
            # Get state from environment's unwrapped state
            theta = self.env.unwrapped.state[0]
            theta_dot = self.env.unwrapped.state[1]
        else:
            # Extract theta from observation [cos(theta), sin(theta), theta_dot]
            cos_theta, sin_theta, theta_dot = state
            theta = np.arctan2(sin_theta, cos_theta)
        
        # Physical parameters of pendulum
        g = 10.0
        m = 1.0
        l = 1.0
        max_torque = 2.0
        
        # Energy-based controller
        # E = 0.5 * m * l^2 * theta_dot^2 - m * g * l * cos(theta)
        # Target energy at upright: E_target = m * g * l = 10
        E = 0.5 * m * l**2 * theta_dot**2 - m * g * l * np.cos(theta)
        E_target = m * g * l
        
        # Energy error
        E_error = E - E_target
        
        # Compute control: pump energy if below target, stabilize if near upright
        if np.abs(np.cos(theta) - (-1)) < 0.2 and np.abs(theta_dot) < 1.0:
            # Near upright: use PD control
            kp = 10.0
            kd = 2.0
            # Desired angle is pi (upright), but theta=0 is down
            # cos(theta) = -1 means upright
            angle_error = np.arctan2(sin_theta, cos_theta)  # Angle from upright
            action = -kp * angle_error - kd * theta_dot
        else:
            # Swing up: energy pumping
            k = 0.5
            action = k * E_error * np.sign(theta_dot * np.cos(theta))
        
        # Normalize to [-1, 1]
        action = np.clip(action / max_torque, -1.0, 1.0)
        
        return np.array([action], dtype=np.float32)
    
    def collect_expert_dataset(
        self, 
        num_episodes: int = 100, 
        noise_std: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """Collect expert demonstrations.
        
        Args:
            num_episodes: Number of episodes to collect
            noise_std: Standard deviation of noise added to expert actions
            
        Returns:
            Dataset dict with keys: states, actions, rewards, next_states, dones
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for _ in range(num_episodes):
            state, _ = self.reset()
            done = False
            
            while not done:
                # Get expert action with noise
                action = self.get_expert_action(state)
                action = action + np.random.normal(0, noise_std, size=action.shape)
                action = np.clip(action, -1.0, 1.0)
                
                next_state, reward, terminated, truncated, _ = self.step(action)
                done = terminated or truncated
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                state = next_state
        
        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32)
        }
