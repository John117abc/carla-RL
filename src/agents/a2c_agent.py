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
from src.buffer import TrajectoryBuffer

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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.a2c_config['lr_actor'],betas=(0.9, 0.999))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.a2c_config['lr_critic'],betas=(0.9, 0.999))

        # 超参数
        self.gamma = self.a2c_config['gamma']
        self.ent_coef = self.a2c_config['ent_coef']
        self.vf_coef = self.a2c_config['vf_coef']
        self.max_grad_norm = self.a2c_config['max_grad_norm']

        # 采样数量
        self.batch_size = self.a2c_config['batch_size']
        self.total_capacity = self.a2c_config['total_capacity']
        self.min_start_train = self.a2c_config['min_start_train']

        # 初始化缓冲区
        self.buffer = []

        # 记录历史日志数据值
        self.globe_eps = 0
        self.history_loss = []
        self.global_step = 0


    def select_action(self, obs: Any, deterministic: bool = False):
        """
        根据观测选择动作。
        训练时返回随机动作和 log_prob；评估时返回均值。
        """
        with torch.no_grad():
            # 转为tensor
            obs_tensor = torch.from_numpy(obs).to(self.device).float()
            if deterministic:
                action_scaled, log_prob, action_mean = self.actor(obs_tensor)
                action = action_mean
            else:
                action, log_prob, _ = self.actor(obs_tensor)
            action = action.cpu().numpy().flatten()
            log_prob = log_prob.cpu().numpy().flatten()
        return np.clip(action, self.action_space.low, self.action_space.high),log_prob

    def calculate_value(self, obs: Any):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).to(self.device).float()
            return self.critic(obs_tensor).cpu().detach().numpy()

    def update(self) -> Dict[str, float]:
        """
        标准 A2C 单步更新。
        输入应为一个 rollout batch: [B, ...]
        """
        obs_list,act_list,rew_list,info,done_list,values_list = zip(*self.n_step_batch())

        # 转 tensor
        obs = torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(act_list), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(np.asarray(rew_list), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(done_list, dtype=torch.bool, device=self.device)
        values = torch.as_tensor(np.array(values_list), dtype=torch.float32, device=self.device)

        # 计算 n-step TD 回报
        returns = self.compute_nstep_returns(rew_list, done_list, values_list, self.gamma, 5)
        advantages = torch.as_tensor(np.array(returns - values_list), dtype=torch.float32, device=self.device)  # A_t = R_t^{(n)} - V(s_t)
        target_values = self.critic(obs)

        # 动作评估
        log_probs, entropy = self.actor.evaluate_actions(obs, actions)
        # 损失
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.functional.mse_loss(values, target_values)
        entropy_mean = entropy.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_mean.item(),
            "advantage_mean": advantages.mean().item(),
            "value_mean": values.mean().item(),
        }

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
        self.globe_eps += self.base_config['save_freq']
        self.history_loss.append(save_info['history_loss'])

        model = {'actor': actor_model, 'critic': critic_model}
        optimizer = {'actor_optim': actor_optimizer, 'critic_optim': critic_optimizer}
        extra_info = {'config': save_info['rl_config'], 'global_step': self.global_step,'history':self.history_loss,
                      'meas_normalizer':save_info['meas_normalizer'],'globe_eps':self.globe_eps}
        met = {'episode': self.globe_eps}
        save_checkpoint(
            model=model,
            model_name='a2c-v1.0',
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
                action,_ = self.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            total_rewards.append(episode_reward)
        return float(np.mean(total_rewards)), float(np.std(total_rewards))

    # 存储记录
    def store_transition(self, state,action,reward,info,done,value):
        self.buffer.append((state,action,reward,info,done,value))
        if len(self.buffer) > self.total_capacity:
            self.buffer.pop(0)

    # 是否能够开始训练
    def should_start_training(self):
        return len(self.buffer) > self.min_start_train

    def n_step_batch(self):
        if self.should_start_training():
            if len(self.buffer) < self.batch_size:
                return self.buffer[len(self.buffer) - self.min_start_train :len(self.buffer)]
            else:
                return self.buffer[len(self.buffer) - self.batch_size :len(self.buffer)]
        else:
            return None

    def calculate_returns(self,batch_buffer):
        returns = []
        G = 0.0
        # 如果最后一步不是终止状态，用 critic 估计 V(last_state)
        if not batch_buffer[-1][4]:  # 第4个元素是 done
            last_state = torch.FloatTensor(batch_buffer[-1][0]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                G = self.critic(last_state).item()

        # 从后往前计算
        for i in reversed(range(len(batch_buffer))):
            reward = batch_buffer[i][2]
            G = reward['total_reward'] + self.gamma * G
            returns.insert(0, G)
        return returns

    def clean_mem(self):
        self.buffer.clear()

    def compute_nstep_returns(self,rewards, dones, values, gamma, n_steps):
        """
        计算 n-step TD 回报 R_t^{(n)} = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})
        输入:
            rewards: [T]       (T = ROLLOUT_STEPS)
            dones:   [T]       (bool, 表示是否终止)
            values:  [T]       (V(s_t))
            gamma: discount factor
            n_steps: n
        输出:
            returns: [T]       (每个 t 对应的 n-step 回报，若超出则用更短的)
        """
        T = len(rewards)
        returns = np.zeros(T)
        # 将 next_value 加到末尾，方便索引
        extended_values = values
        extended_rewards = np.concatenate([rewards, np.zeros(n_steps)])
        extended_dones = np.concatenate([dones, np.ones(n_steps)])

        for t in range(T):
            R = 0.0
            discount = 1.0
            # 最多往前看 n_steps 步，但不能超过 T
            steps = min(n_steps, T - t)
            for i in range(steps):
                R += discount * extended_rewards[t + i]
                discount *= gamma
                if extended_dones[t + i]:  # 如果提前终止，不再加后续奖励或 bootstrap
                    break
            else:
                # 如果没有 break（即未提前终止），加上 bootstrap 项
                if not extended_dones[t + steps - 1]:
                    R += discount * extended_values[t + steps]
            returns[t] = R
        return returns

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