"""
Diffusion Policy Trainer

Training utilities for Diffusion Policy including:
- Dataset loading
- Training loop
- Evaluation
- Logging
"""

import numpy as np
import torch
from typing import Dict, Optional, Callable
from tqdm import tqdm

from .agent import DiffusionPolicyAgent
from ...common.replay_buffer import ReplayBuffer


class DiffusionPolicyTrainer:
    """Trainer for Diffusion Policy.
    
    Handles the training loop, evaluation, and logging for
    behavior cloning with diffusion models.
    
    Args:
        agent: DiffusionPolicyAgent instance
        env: Gymnasium environment for evaluation
        replay_buffer: Buffer containing expert demonstrations
        eval_episodes: Number of episodes for evaluation
        log_interval: Steps between logging
        eval_interval: Steps between evaluations
    """
    
    def __init__(
        self,
        agent: DiffusionPolicyAgent,
        env,
        replay_buffer: ReplayBuffer,
        eval_episodes: int = 10,
        log_interval: int = 100,
        eval_interval: int = 1000
    ):
        self.agent = agent
        self.env = env
        self.buffer = replay_buffer
        self.eval_episodes = eval_episodes
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        
        # Initialize EMA
        self.agent._init_ema()
        
        # Tracking
        self.total_steps = 0
        self.best_return = float("-inf")
        
    def train(
        self,
        total_steps: int,
        batch_size: int = 256,
        callback: Optional[Callable] = None
    ) -> Dict[str, list]:
        """Run training loop.
        
        Args:
            total_steps: Total training steps
            batch_size: Batch size for training
            callback: Optional callback function called each step
            
        Returns:
            Dictionary of training history
        """
        history = {
            "loss": [],
            "eval_return": [],
            "eval_step": []
        }
        
        pbar = tqdm(range(total_steps), desc="Training Diffusion Policy")
        
        for step in pbar:
            # Sample batch and train
            batch = self.buffer.sample(batch_size)
            metrics = self.agent.train_step(batch)
            
            self.total_steps += 1
            history["loss"].append(metrics["loss"])
            
            # Logging
            if step % self.log_interval == 0:
                avg_loss = np.mean(history["loss"][-self.log_interval:])
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # Evaluation
            if step % self.eval_interval == 0:
                eval_return = self.evaluate()
                history["eval_return"].append(eval_return)
                history["eval_step"].append(step)
                
                if eval_return > self.best_return:
                    self.best_return = eval_return
                    
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "eval_return": f"{eval_return:.2f}",
                    "best": f"{self.best_return:.2f}"
                })
            
            if callback:
                callback(step, metrics)
                
        return history
    
    def evaluate(self, num_episodes: Optional[int] = None) -> float:
        """Evaluate the policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Average episode return
        """
        num_episodes = num_episodes or self.eval_episodes
        returns = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_return = 0
            done = False
            
            while not done:
                action = self.agent.sample_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_return += reward
                done = terminated or truncated
                
            returns.append(episode_return)
            
        return np.mean(returns)
    
    def collect_expert_data(
        self,
        num_episodes: int = 100,
        noise_std: float = 0.1
    ):
        """Collect expert demonstrations and add to buffer.
        
        Args:
            num_episodes: Number of episodes to collect
            noise_std: Noise standard deviation for exploration
        """
        if hasattr(self.env, "collect_expert_dataset"):
            dataset = self.env.collect_expert_dataset(num_episodes, noise_std)
            self.buffer.load_dataset(dataset)
        else:
            raise NotImplementedError(
                "Environment must implement collect_expert_dataset method"
            )
