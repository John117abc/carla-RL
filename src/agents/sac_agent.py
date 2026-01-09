# src/agents/sac_agent.py

import os
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent
from src.models.actor_critic import ActorNetwork, CriticNetwork
from src.utils.replay_buffer import ReplayBuffer


class SACAgent(BaseAgent):
    """
    Soft Actor-Critic (SAC) 智能体实现。
    支持自动熵调节（automatic entropy tuning）。
    """

    def __init__(
        self,
        env: gym.Env,
        device: torch.device = torch.device("cpu"),
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        alpha: float = "auto",  # 可设为 float 或 "auto"
        target_entropy: float = None,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__(env, device)

        assert isinstance(self.action_space, gym.spaces.Box), "SAC only supports continuous action spaces."

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # 网络初始化
        obs_dim = np.prod(self.observation_space.shape)
        action_dim = np.prod(self.action_space.shape)

        self.actor = ActorNetwork(self.observation_space, self.action_space, hidden_dim=hidden_dim).to(self.device)
        self.critic = CriticNetwork(self.observation_space, self.action_space, hidden_dim=hidden_dim).to(self.device)
        self.critic_target = CriticNetwork(self.observation_space, self.action_space, hidden_dim=hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 熵相关设置
        if alpha == "auto":
            if target_entropy is None:
                # 默认目标熵：-dim(A)
                self.target_entropy = -np.prod(self.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
            self.log_alpha = None  # 不训练

        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size, self.observation_space, self.action_space, device=self.device)

    def select_action(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        """
        根据观测选择动作。
        训练时使用随机策略，评估时使用确定性策略（均值）。
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if deterministic:
                _, _, action_mean = self.actor(obs_tensor)
                action = action_mean
            else:
                action, _, _ = self.actor(obs_tensor)
            action = action.cpu().numpy().flatten()

        # 确保动作在合法范围内（虽然 tanh 已约束，但可再 clip）
        return np.clip(action, self.action_space.low, self.action_space.high)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        SAC 更新步骤：先更新 critic，再更新 actor 和 alpha。
        """
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"].unsqueeze(1)
        next_obs = batch["next_obs"]
        done = batch["done"].unsqueeze(1)

        metrics = {}

        # --- Critic 更新 ---
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = reward + (1 - done.float()) * self.gamma * q_next

        q1, q2 = self.critic(obs, action)
        critic_loss = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics["critic_loss"] = critic_loss.item()
        metrics["q1_mean"] = q1.mean().item()
        metrics["q_target_mean"] = q_target.mean().item()

        # --- Actor 更新 ---
        action_curr, log_pi, _ = self.actor(obs)
        q1_curr, q2_curr = self.critic(obs, action_curr)
        q_curr = torch.min(q1_curr, q2_curr)
        actor_loss = (self.alpha * log_pi - q_curr).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics["actor_loss"] = actor_loss.item()
        metrics["entropy"] = -log_pi.mean().item()

        # --- Alpha (温度系数) 更新（如果启用自动调节）---
        if self.log_alpha is not None:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            metrics["alpha_loss"] = alpha_loss.item()
            metrics["alpha"] = self.alpha

        # --- 软更新 target critic ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return metrics

    def save(self, path: str) -> None:
        """保存所有模型和优化器状态。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict() if self.log_alpha is not None else None,
        }, path)

    def load(self, path: str) -> None:
        """从路径加载模型。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        if self.log_alpha is not None and "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"]
            if checkpoint["alpha_optimizer_state_dict"] is not None:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
            self.alpha = self.log_alpha.exp().item()