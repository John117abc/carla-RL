# src/agents/a2c_agent.py

import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple

from .base_agent import BaseAgent
from src.models.actor_critic import ActorNetwork, CriticNetwork
from src.utils import save_checkpoint,load_checkpoint

class OcpAgent(BaseAgent):
    """
    ocp智能体，复现《Integrated Decision and Control: Toward  Interpretable and Computationally  Efficient Driving Intelligence》
    这篇论文中的Dynamic Optimal Tracking-Offline Training算法
    """

    def __init__(
        self,
        rl_config: Dict[str, Any],
        env: gym.Env,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(env, device)

        assert isinstance(self.action_space, gym.spaces.Box), "A2智能体需要连续的动作空间。"

        # 读取配置参数
        rl_algorithm = "OCP"
        self.base_config = rl_config['rl']
        self.ocp_config = rl_config['rl'][rl_algorithm]

        # 网络
        self.actor = ActorNetwork(self.observation_space['ocp_obs'], self.action_space, hidden_dim=self.ocp_config['hidden_dim']).to(self.device)
        self.critic = CriticNetwork(self.observation_space['ocp_obs'],hidden_dim=self.ocp_config['hidden_dim']).to(self.device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.ocp_config['lr_actor'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.ocp_config['lr_critic'])

        # 超参数
        self.init_penalty = self.ocp_config['init_penalty']
        self.max_penalty = self.ocp_config['max_penalty']
        self.other_car_min_distance = self.ocp_config['other_car_min_distance']
        self.road_min_distance = self.ocp_config['road_min_distance']

        # 正定矩阵
        self.Q_matrix = np.diag([0.04, 0.04, 0.01, 0.01, 0.1, 0.02])
        self.R_matrix = np.diag([0.1, 0.005])
        self.M_matrix = np.diag([1,1,0,0,0,0])
        # 严格使用s^ref = [δp, δφ, δv ]状态时候的Q
        self.Q_matrix_ref = np.diag([0.04,0.1,0.01])

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
        values = self.critic(obs)
        # 下一状态价值（用于 bootstrap）
        with torch.no_grad():
            next_values = self.critic(next_obs)
            target_values = rewards + self.gamma * next_values * (1 - dones.float())
            advantages = target_values - values  # [B]

        # 重新评估动作
        log_probs, entropy = self.actor.evaluate_actions(obs, actions)

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
        values = self.critic(obs)
        with torch.no_grad():
            next_values = self.critic(next_obs)
            target_values = rewards + self.gamma * next_values * (1 - dones.float())
            advantages = target_values - values

        log_probs, entropy = self.actor.evaluate_actions(obs, actions)
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

    def save(self,
             rl_config:Dict[str, Any],
             global_step:int,
             episode:int,
             map_name:str) -> None:
        actor_model = self.actor
        critic_model = self.critic
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        model = {'actor': actor_model, 'critic': critic_model}
        optimizer = {'actor_optim': actor_optimizer, 'critic_optim': critic_optimizer}
        extra_info = {'config': rl_config, 'global_step': global_step}
        met = {'episode': episode}
        save_checkpoint(
            model=model,
            model_name='a2c-v1.0',
            optimizer=optimizer,
            extra_info=extra_info,
            metrics=met,
            env_name=map_name
        )

    def load(self, path: str) -> None:
        load_checkpoint(
            model={'actor': self.actor, 'critic': self.critic},
            filepath=path,
            optimizer={'actor_optim': self.actor_optimizer, 'critic_optim': self.critic_optimizer},
            device=self.device
        )

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