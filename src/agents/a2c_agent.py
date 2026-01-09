# src/agents/a2c_agent.py

import os
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from .base_agent import BaseAgent
from src.models.actor_critic import ActorNetwork, CriticNetwork

# 训练演示
# agent = A2CAgent(env, device=device)
# obs, _ = env.reset()
#
# for step in range(total_steps):
#     action = agent.select_action(obs)
#     next_obs, reward, terminated, truncated, _ = env.step(action)
#     done = terminated or truncated
#
#     # 构造 batch（单步）
#     batch = {
#         "obs": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device),
#         "action": torch.as_tensor(action, dtype=torch.float32).unsqueeze(0).to(device),
#         "reward": torch.as_tensor([reward], dtype=torch.float32).to(device),
#         "next_obs": torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device),
#         "done": torch.as_tensor([done], dtype=torch.bool).to(device),
#     }
#
#     metrics = agent.update(batch)
#     logger.log(metrics)
#
#     obs = next_obs
#     if done:
#         obs, _ = env.reset()

class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic (A2C) 智能体。
    同步 on-policy 算法，使用 GAE 或单步 TD 误差估计优势。
    默认使用单步优势：A(s,a) = r + γV(s') - V(s)
    """

    def __init__(
        self,
        env: gym.Env,
        device: torch.device = torch.device("cpu"),
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_gae: bool = False,
        gae_lambda: float = 0.95,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__(env, device)

        assert isinstance(self.action_space, gym.spaces.Box), "A2智能体需要连续的动作空间。"

        # 网络
        self.actor = ActorNetwork(self.observation_space, self.action_space, hidden_dim=hidden_dim).to(self.device)
        self.critic = CriticNetwork(self.observation_space, self.action_space, hidden_dim=hidden_dim).to(self.device)

        # 优化器（可共享或分离）
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 超参数
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        # 用于 GAE 的缓存（如果启用）
        self._rollout_cache = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }

    def select_action(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        """
        根据观测选择动作。
        训练时返回随机动作和 log_prob；评估时返回均值。
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if deterministic:
                _, _, action_mean = self.actor(obs_tensor)
                action = action_mean
            else:
                action, log_prob, _ = self.actor(obs_tensor)
            action = action.cpu().numpy().flatten()
        return np.clip(action, self.action_space.low, self.action_space.high)

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 A2C 损失：策略梯度 + 价值损失 + 熵正则。
        使用单步 TD 优势：A = r + γV(s') - V(s)
        """
        # 当前状态价值
        values = self.critic(obs).squeeze(-1)  # [B]
        # 下一状态价值（用于 bootstrap）
        with torch.no_grad():
            next_values = self.critic(next_obs).squeeze(-1)  # [B]
            target_values = rewards + self.gamma * next_values * (1 - dones.float())
            advantages = target_values - values  # [B]

        # 重新评估动作
        _, log_probs, entropy = self.actor.evaluate_actions(obs, actions)

        # 策略损失（最大化 E[logπ(a|s) * A]）
        actor_loss = -(log_probs * advantages.detach()).mean()

        # 价值损失
        critic_loss = nn.functional.mse_loss(values, target_values)

        # 总损失
        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy.mean()

        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.mean().item(),
            "advantage_mean": advantages.mean().item(),
            "value_mean": values.mean().item(),
        }
        return loss, metrics

    def update(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        执行一次 A2C 更新。
        batch 应包含: obs, action, reward, next_obs, done
        """
        obs = batch["obs"]
        actions = batch["action"]
        rewards = batch["reward"]
        next_obs = batch["next_obs"]
        dones = batch["done"]

        loss, metrics = self.compute_loss(obs, actions, rewards, next_obs, dones)

        # 优化 Actor
        self.actor_optimizer.zero_grad()
        # 注意：loss 已包含 actor 和 critic 部分，但通常分开优化更稳定
        # 这里我们拆开（更清晰）
        values = self.critic(obs).squeeze(-1)
        with torch.no_grad():
            next_values = self.critic(next_obs).squeeze(-1)
            target_values = rewards + self.gamma * next_values * (1 - dones.float())
            advantages = target_values - values

        _, log_probs, entropy = self.actor.evaluate_actions(obs, actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = nn.functional.mse_loss(values, target_values)

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        metrics.update({
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.mean().item(),
        })
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    def eval(self, num_episodes: int = 10) -> Tuple[float, float]:
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
        return float(np.mean(total_rewards)), float(np.std(total_rewards))