# src/agents/base_agent.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import torch


class BaseAgent(ABC):
    """
    抽象基类，定义强化学习智能体的标准接口。

    所有具体智能体（如 DQNAgent、PPOAgent、SACAgent）都应继承此类，
    并实现其抽象方法。
    """

    def __init__(self, env: gym.Env, device: torch.device = torch.device("cpu")) -> None:
        """
        初始化智能体。

        Args:
            env (gym.Env): 与智能体交互的 Gymnasium 环境。
            device (torch.device): 模型运行的设备（CPU/GPU）。
        """
        self.env = env
        self.device = device
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    @abstractmethod
    def select_action(self, obs: Any, deterministic: bool = False) -> Any:
        """
        根据当前观测选择动作。

        Args:
            obs (Any): 当前环境观测（可能为 numpy array、dict 等）。
            deterministic (bool): 是否使用确定性策略（例如评估时设为 True）。

        Returns:
            action (Any): 选择的动作，格式需与 env.action_space 兼容。
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> Dict[str, float]:
        """
        使用一个批次的经验更新智能体参数。

        Args:

        Returns:
            metrics (Dict[str, float]): 训练指标字典，如 loss、q_value 等，
                用于日志记录。
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, save_info: Dict[str, Any]) -> None:
        """
        保存模型权重到指定路径。

        Args:
            save_info (str): 保存信息。
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """
        从指定路径加载模型权重。

        Args:
            path (str): 模型权重路径。
        """
        raise NotImplementedError

    def train(self, num_steps: int) -> None:
        """
        （可选）高层训练循环封装。通常在 train.py 中实现更灵活的训练逻辑，
        因此此处可留空或提供默认实现。

        Args:
            num_steps (int): 训练步数。
        """
        pass

    def eval(self, num_episodes: int = 10) -> Tuple[float, float]:
        """
        （可选）评估智能体性能。

        Args:
            num_episodes (int): 评估轮次。

        Returns:
            mean_reward (float): 平均回报。
            std_reward (float): 回报标准差。
        """
        total_rewards = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action = self.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            total_rewards.append(episode_reward)
        import numpy as np
        return float(np.mean(total_rewards)), float(np.std(total_rewards))