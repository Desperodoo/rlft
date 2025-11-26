"""
2D Point Mass Environment

A simple continuous control environment where a point mass 
must navigate to a goal position.

State: (x, y, vx, vy) - position and velocity
Action: (ax, ay) - acceleration, bounded in [-1, 1]
Reward: -distance_to_goal - action_penalty
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class PointMass2DEnv(gym.Env):
    """2D Point Mass Navigation Environment.
    
    The agent controls a point mass in 2D space and must navigate to a goal.
    
    Args:
        goal_position: Fixed goal position (x, y). If None, goal is at origin.
        max_speed: Maximum velocity magnitude
        dt: Time step for physics simulation
        max_episode_steps: Maximum steps per episode
        action_cost: Coefficient for action penalty in reward
        goal_threshold: Distance threshold to consider goal reached
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        goal_position: Optional[Tuple[float, float]] = None,
        max_speed: float = 2.0,
        dt: float = 0.1,
        max_episode_steps: int = 200,
        action_cost: float = 0.01,
        goal_threshold: float = 0.1,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.goal_position = np.array(goal_position if goal_position else [0.0, 0.0])
        self.max_speed = max_speed
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.action_cost = action_cost
        self.goal_threshold = goal_threshold
        self.render_mode = render_mode
        
        # State: [x, y, vx, vy]
        self.observation_space = spaces.Box(
            low=np.array([-5.0, -5.0, -max_speed, -max_speed]),
            high=np.array([5.0, 5.0, max_speed, max_speed]),
            dtype=np.float32
        )
        
        # Action: [ax, ay] acceleration
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        self.state = None
        self.steps = 0
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options (can contain 'initial_state')
            
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        if options and "initial_state" in options:
            self.state = np.array(options["initial_state"], dtype=np.float32)
        else:
            # Random initial position, zero velocity
            x = self.np_random.uniform(-3.0, 3.0)
            y = self.np_random.uniform(-3.0, 3.0)
            self.state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        
        self.steps = 0
        
        return self.state.copy(), {"goal_position": self.goal_position.copy()}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Acceleration [ax, ay]
            
        Returns:
            observation: New state
            reward: Reward signal
            terminated: Whether episode ended due to success
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        action = np.clip(action, -1.0, 1.0)
        
        # Extract current state
        x, y, vx, vy = self.state
        ax, ay = action
        
        # Physics update (simple Euler integration)
        vx_new = np.clip(vx + ax * self.dt, -self.max_speed, self.max_speed)
        vy_new = np.clip(vy + ay * self.dt, -self.max_speed, self.max_speed)
        x_new = np.clip(x + vx_new * self.dt, -5.0, 5.0)
        y_new = np.clip(y + vy_new * self.dt, -5.0, 5.0)
        
        self.state = np.array([x_new, y_new, vx_new, vy_new], dtype=np.float32)
        self.steps += 1
        
        # Calculate reward
        distance = np.linalg.norm(self.state[:2] - self.goal_position)
        action_penalty = self.action_cost * np.sum(action ** 2)
        reward = -distance - action_penalty
        
        # Check termination conditions
        terminated = distance < self.goal_threshold
        truncated = self.steps >= self.max_episode_steps
        
        info = {
            "distance_to_goal": distance,
            "goal_reached": terminated,
            "goal_position": self.goal_position.copy()
        }
        
        if terminated:
            reward += 10.0  # Bonus for reaching goal
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (placeholder for visualization)."""
        if self.render_mode == "human":
            print(f"State: {self.state}, Goal: {self.goal_position}")
        return None
    
    def get_expert_action(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Get expert action using simple proportional control.
        
        This can be used to generate expert demonstrations for offline RL.
        
        Args:
            state: State to compute action for. If None, uses current state.
            
        Returns:
            Expert action
        """
        if state is None:
            state = self.state
            
        x, y, vx, vy = state
        gx, gy = self.goal_position
        
        # Simple proportional-derivative control
        kp = 2.0  # Proportional gain
        kd = 1.0  # Derivative gain
        
        ax = kp * (gx - x) - kd * vx
        ay = kp * (gy - y) - kd * vy
        
        # Clip to action space
        action = np.clip([ax, ay], -1.0, 1.0)
        
        return action.astype(np.float32)
    
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
