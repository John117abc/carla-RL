# src/agents/ppo_agent.py

import os
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from .base_agent import BaseAgent
from src.models.actor_critic import ActorNetwork, CriticNetwork


class RolloutBuffer:
    """
    On-policy 滚动缓冲区，用于 PPO。
    存储一个 episode 或多个 rollout 的轨迹。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(torch.as_tensor(obs, dtype=torch.float32))
        self.actions.append(torch.as_tensor(action, dtype=torch.float32))
        self.log_probs.append(log_prob.clone())
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.clone())

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        使用 GAE 计算优势函数和回报。
        """
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = torch.cat(self.values).cpu().numpy()
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        # Bootstrap 最后一步
        next_value = last_value
        advantage = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_nonterminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_nonterminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]

            delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
            advantage = delta + gamma * gae_lambda * next_nonterminal * advantage
            advantages[t] = advantage
            returns[t] = advantages[t] + values[t]

        self.returns = torch.as_tensor(returns, dtype=torch.float32)
        self.advantages = torch.as_tensor(advantages, dtype=torch.float32)

    def get(self):
        return (
            torch.stack(self.obs),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            self.returns,
            self.advantages,
        )


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) 智能体，支持连续动作空间。
    """

    def __init__(
        self,
        env: gym.Env,
        device: torch.device = torch.device("cpu"),
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__(env, device)

        assert isinstance(self.action_space, gym.spaces.Box), "PPO agent expects continuous action space."

        obs_dim = np.prod(self.observation_space.shape)
        action_dim = np.prod(self.action_space.shape)

        # 网络
        self.actor = ActorNetwork(self.observation_space, self.action_space, hidden_dim=hidden_dim).to(self.device)
        self.critic = CriticNetwork(self.observation_space, self.action_space, hidden_dim=hidden_dim).to(self.device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs

        # 缓冲区
        self.buffer = RolloutBuffer()

    def select_action(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        """
        选择动作，并在训练时记录 log_prob 和 value（用于后续更新）。
        注意：此方法在训练时需配合 `store_transition` 使用。
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, _ = self.actor(obs_tensor)
            value = self.critic(obs_tensor)

            action = action.cpu().numpy().flatten()
            log_prob = log_prob.item()
            value = value.item()

        if deterministic:
            # 使用均值作为确定性动作
            _, _, action_mean = self.actor(obs_tensor)
            action = action_mean.cpu().numpy().flatten()

        # 保存 log_prob 和 value 供 buffer 使用（仅在 rollout 时）
        self._last_log_prob = log_prob
        self._last_value = value

        return np.clip(action, self.action_space.low, self.action_space.high)

    def store_transition(self, obs, action, reward, done, next_obs=None):
        """
        将 transition 存入 buffer。
        注意：PPO 需要完整轨迹，因此在每一步调用。
        """
        self.buffer.add(obs, action, self._last_log_prob, reward, done, self._last_value)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        PPO 不使用外部 batch，而是从内部 buffer 更新。
        因此该方法实际由 train.py 调用 `agent.learn()` 触发。
        为兼容 BaseAgent 接口，我们重载逻辑到 learn()，此处抛出提示。
        """
        raise NotImplementedError("PPO uses on-policy learning. Call `agent.learn()` instead of `update(batch)`.")

    def learn(self) -> Dict[str, float]:
        """
        执行 PPO 更新：计算 GAE，然后进行多轮策略和价值网络优化。
        """
        # 获取最后一步的 value（用于 bootstrap）
        last_obs = self.buffer.obs[-1].unsqueeze(0).to(self.device)
        with torch.no_grad():
            last_value = self.critic(last_obs).item()

        # 计算 GAE 和 returns
        self.buffer.compute_returns_and_advantages(
            last_value, gamma=self.gamma, gae_lambda=self.gae_lambda
        )

        obs, actions, old_log_probs, returns, advantages = self.buffer.get()
        obs, actions, old_log_probs, returns, advantages = (
            obs.to(self.device),
            actions.to(self.device),
            old_log_probs.to(self.device),
            returns.to(self.device),
            advantages.to(self.device),
        )

        # 归一化优势（可选但推荐）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.n_epochs):
            # 重新评估旧动作
            new_actions, new_log_probs, entropy = self.actor(obs)
            new_values = self.critic(obs).squeeze(-1)

            # 策略损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 熵正则（鼓励探索）
            entropy_loss = -entropy.mean()

            # 价值损失
            critic_loss = nn.MSELoss()(new_values, returns)

            # 总损失
            loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss

            # 优化 Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # 优化 Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()

        # 清空 buffer
        self.buffer.reset()

        metrics = {
            "actor_loss": total_actor_loss / self.n_epochs,
            "critic_loss": total_critic_loss / self.n_epochs,
            "entropy": total_entropy / self.n_epochs,
        }
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
        # 复用基类的 eval，但确保使用 deterministic=True
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