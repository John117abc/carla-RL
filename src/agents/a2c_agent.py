# src/agents/a2c_agent.py

import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, List, Union

from .base_agent import BaseAgent
from src.models.advantage_actor_critic import ActorNetwork,CriticNetwork
from src.utils import save_checkpoint,load_checkpoint,get_logger
from src.buffer import StochasticBuffer

logger = get_logger('a2c_agent')

class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic (A2C) 智能体。
    同步 on-policy 算法，使用 GAE 或单步 TD 误差估计优势。
    默认使用单步优势：A(s,a) = r + γV(s') - V(s)
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
        rl_algorithm = "A2C"
        self.base_config = rl_config['rl']
        self.a2c_config = rl_config['rl'][rl_algorithm]

        # 网络
        self.actor = ActorNetwork(self.observation_space['measurements'], self.action_space, hidden_dim=self.a2c_config['hidden_dim']).to(self.device)
        self.critic = CriticNetwork(self.observation_space['measurements'],hidden_dim=self.a2c_config['hidden_dim']).to(self.device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.a2c_config['lr_actor'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.a2c_config['lr_critic'])

        # 超参数
        self.gamma = self.a2c_config['gamma']
        self.ent_coef = self.a2c_config['ent_coef']
        self.vf_coef = self.a2c_config['vf_coef']
        self.max_grad_norm = self.a2c_config['max_grad_norm']

        # 采样数量
        self.batch_size = self.a2c_config['batch_size']

        # 初始化缓冲区
        self.buffer = StochasticBuffer(min_start_train = self.a2c_config['min_start_train'],
                                              total_capacity = self.a2c_config['total_capacity'])

        # 记录历史日志数据值
        self.globe_eps = 0
        self.history_loss = []
        self.global_step = 0


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

    def update(self) -> Dict[str, float]:
        """
        执行一次 A2C 更新。
        """
        batch = self.buffer.sample_batch(self.batch_size)
        obs, actions, rewards, next_obs, dones ,_ = zip(*np.asarray(batch,dtype=object).reshape(-1, 6))
        rewards = np.array([r['total_reward'] for r in rewards])

        # 转为tensor
        obs = torch.from_numpy(np.asarray(obs)).to(self.device).float()
        actions = torch.from_numpy(np.asarray(actions)).to(self.device).float()
        rewards = torch.from_numpy(np.asarray(rewards)).to(self.device).float()
        next_obs = torch.from_numpy(np.asarray(next_obs)).to(self.device).float()
        dones = torch.from_numpy(np.asarray(dones)).to(self.device).float()

        loss, metrics = self.compute_loss(obs, actions, rewards, next_obs, dones)

        # 优化 Actor
        self.actor_optimizer.zero_grad()
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
            "actor_loss": actor_loss.detach().item(),
            "critic_loss": critic_loss.detach().item(),
            "entropy": entropy.mean().item(),
        })
        return metrics

    def save(self,save_info: Dict[str, Any]) -> None:
        """
        保存模型参数
        :param save_info: 参数数据
        """
        actor_model = self.actor
        critic_model = self.critic
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer

        self.global_step += save_info['global_step']
        self.globe_eps += save_info['episode']
        self.history_loss.append(save_info['history_loss'])

        model = {'actor': actor_model, 'critic': critic_model}
        optimizer = {'actor_optim': actor_optimizer, 'critic_optim': critic_optimizer}
        extra_info = {'config': save_info['rl_config'], 'global_step': self.global_step,'history':self.history_loss,
                      'meas_normalizer':save_info['meas_normalizer'],'globe_eps':self.globe_eps}
        met = {'episode': self.globe_eps}
        save_checkpoint(
            model=model,
            model_name='ocp-v1.0',
            optimizer=optimizer,
            extra_info=extra_info,
            metrics=met,
            env_name=save_info['map']
        )

    def load(self, path: str) -> None:
        checkpoint = load_checkpoint(
            model={'actor': self.actor, 'critic': self.critic},
            filepath=path,
            optimizer={'actor_optim': self.actor_optimizer, 'critic_optim': self.critic_optimizer},
            device=self.device
        )
        self.globe_eps = checkpoint['globe_eps']
        self.history_loss = checkpoint['history']
        self.global_step = checkpoint['global_step']
        return checkpoint

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

    def unpack_observation(self, obs: Union[List, np.ndarray], batched: bool = False):
        """
        解包 observation 数据，支持单样本和批量样本

        Args:
            obs:
                - 若 batched=False: [ego, other, road, ref]
                  其中 ego/other/road/ref 为 array-like (list, np.ndarray)
                - 若 batched=True: [sample_1, sample_2, ..., sample_B]
                  其中 sample_i = [ego_i, other_i, road_i, ref_i]
            batched: 是否为批量数据

        Returns:
            state_all: [D] if not batched, or [B, D] if batched
            state_ego, state_other, state_road, state_ref: 各部分张量（在 self.device 上）
        """

        def to_tensor(data, is_batch=False):
            """将 list 或 np.ndarray 转为 tensor"""
            if isinstance(data, torch.Tensor):
                tensor = data
            else:
                # 先转为 numpy array避免 list of ndarray
                arr = np.array(data, dtype=np.float32)
                tensor = torch.from_numpy(arr)
            return tensor.to(self.device)

        if batched:
            ego_list = [sample[0] for sample in obs]
            other_list = [sample[1] for sample in obs]
            road_list = [sample[2] for sample in obs]
            ref_list = [sample[3] for sample in obs]

            state_ego = to_tensor(ego_list)
            state_other = to_tensor(other_list)
            state_road = to_tensor(road_list)
            state_ref = to_tensor(ref_list)

            B = state_ego.shape[0]
            state_ego_flat = state_ego.view(B, -1)
            state_other_flat = state_other.view(B, -1)
            state_road_flat = state_road.view(B, -1)
            state_ref_flat = state_ref.view(B, -1)

            state_all = torch.cat([
                state_ego_flat,
                state_other_flat,
                state_road_flat,
                state_ref_flat
            ], dim=1)  # [B, D]

        else:

            state_ego = to_tensor(obs[0])
            state_other = to_tensor(obs[1])
            state_road = to_tensor(obs[2])
            state_ref = to_tensor(obs[3])

            state_all = torch.cat([
                state_ego.view(-1),
                state_other.view(-1),
                state_road.view(-1),
                state_ref.view(-1)
            ], dim=0)

        return state_all, state_ego, state_other, state_road, state_ref