# src/agents/a2c_agent.py

import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Tuple,List, Union

from .base_agent import BaseAgent
from src.models.advantage_actor_critic import ActorNetwork, CriticNetwork
from src.utils import save_checkpoint,load_checkpoint
from src.buffer import TrajectoryBuffer
from src.utils import get_logger,RunningNormalizer

logger = get_logger('ocp_agent')

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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.ocp_config['lr_actor'],betas=(0.9, 0.999))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.ocp_config['lr_critic'],betas=(0.9, 0.999))

        # 损失函数
        self.loss_func = nn.MSELoss()

        # 超参数
        self.init_penalty = self.ocp_config['init_penalty']
        self.max_penalty = self.ocp_config['max_penalty']
        self.amplifier_c = self.ocp_config['amplifier_c']
        self.amplifier_m = self.ocp_config['amplifier_m']
        self.other_car_min_distance = self.ocp_config['other_car_min_distance']
        self.road_min_distance = self.ocp_config['road_min_distance']
        self.gamma = self.ocp_config['gamma']

        # 正定矩阵
        self.Q_matrix = np.diag([0.04, 0.01, 0.1, 0.01, 0.1, 0.05])
        self.R_matrix = np.diag([0.1, 0.005])
        self.M_matrix = np.diag([1,1,0,0,0,0])
        # 严格使用s^ref = [δp, δφ, δv ]状态时候的Q
        self.Q_matrix_ref = np.diag([0.04,0.01,0.01])

        # 采样数量
        self.batch_size = self.ocp_config['batch_size']

        # 初始化缓冲区
        self.buffer = TrajectoryBuffer(min_start_train = self.ocp_config['min_start_train'],
                                       total_capacity = self.ocp_config['total_capacity'])

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
            # 转为tensor
            obs_tensor = torch.from_numpy(obs[0]).to(self.device).float()
            if deterministic:
                _, _, action_mean = self.actor(obs_tensor)
                action = action_mean
            else:
                action, log_prob, _ = self.actor(obs_tensor)
            action = action.cpu().numpy().flatten()
        return np.clip(action, self.action_space.low, self.action_space.high)

    def update(self):
        # 采样 N 条完整轨迹
        trajectories = self.buffer.sample_batch(self.batch_size)

        actor_losses = []
        critic_targets = []
        initial_states = []

        for traj in trajectories:
            # traj 包含: init_s, states, actions, refs, etc.
            total_cost = 0
            total_constraint = 0
            discount = 1.0
            s_all, s_ego, s_other, s_road, s_ref = self.unpack_observation(traj.states,True)

            action, log_prob, _ = self.actor(s_all)

            # 跟踪消耗
            tracking_diff = s_ego - s_ref
            tracking_diff[4] = (tracking_diff[4] + 180) % 360 - 180  # 角度制 wrap
            Q_diag = torch.from_numpy(np.diag(self.Q_matrix)).to(self.device).float()
            tracking = (tracking_diff * Q_diag * tracking_diff).sum()

            # 控制消耗，
            R_diag = torch.from_numpy(np.diag(self.R_matrix)).to(self.device).float()
            control = (action * R_diag * action).sum()

            total_cost += discount * (tracking + control)

            # 自车-周车
            rel_pos_car = s_ego.unsqueeze(1)[:,:,0:2] - s_other[:,:,0:2]
            M_xy = torch.from_numpy(np.diag(self.M_matrix)[:2]).to(self.device).float()
            dist_sq_car = (rel_pos_car * M_xy * rel_pos_car).sum()
            g_car = dist_sq_car - self.other_car_min_distance ** 2
            ge_car = torch.relu(-g_car)

            # 道路约束
            rel_pos_road = s_ego[:,0:2] - s_road[:,0:2]
            dist_sq_road = (rel_pos_road * M_xy * rel_pos_road).sum()
            g_road = dist_sq_road - self.road_min_distance ** 2
            ge_road = torch.relu(-g_road)

            total_constraint += discount * (ge_car + ge_road)

            # discount *= self.gamma

            actor_loss = total_cost + self.init_penalty * total_constraint
            actor_losses.append(actor_loss)

            initial_states.append(traj.initial_state)
            critic_targets.append(total_cost)

        # Actor更新
        actor_loss = torch.stack(actor_losses).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic更新
        initial_state = torch.from_numpy(initial_states[0][0]).to(self.device).float()
        critic_pred = self.critic(initial_state).squeeze()
        critic_target = torch.tensor(critic_targets, device=self.device)
        critic_loss = F.mse_loss(critic_pred, critic_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return {
            "actor_loss": actor_loss.detach().item(),
            "critic_loss": critic_loss.detach().item()
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
                      'ocp_normalizer':save_info['ocp_normalizer'],'globe_eps':self.globe_eps}
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

    def update_penalty(self,step_count:int = 0):
        """
        更新惩罚参数
        """
        if step_count % self.amplifier_m == 0:
            self.init_penalty = min(self.init_penalty * self.amplifier_c,self.max_penalty)

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


    def compute_total_cost_and_constraint(self,states,action):
        """
        计算这条轨迹的效用值和约束
        :param states: 观察状态
        :param action: 动作
        :return: 效用值，约束值
        """
        # 解包状态
        state_ego =  np.asarray([sample[0] for sample in states])
        state_other = np.asarray([sample[1] for sample in states])
        state_road = np.asarray([sample[2] for sample in states])
        state_ref = np.asarray([sample[3] for sample in states])
        action = np.asarray(action)

        # 计算 cost components
        tracking_error = ((state_ref - state_ego) @ self.Q_matrix) * (state_ref - state_ego)
        control_energy = (action @ self.R_matrix) * action
        l_current = tracking_error.mean() + control_energy.mean()

        # 计算约束项
        diff = np.expand_dims(state_ego, axis=1)- state_other
        dist_sq = np.sum(((diff @ self.M_matrix)**2),axis=-1)
        ge_car = np.maximum(0.0,self.other_car_min_distance ** 2 - dist_sq).mean()
        ge_road = np.maximum(0.0,-np.sum((((state_ego - state_road) @ self.M_matrix)**2),axis=-1)+ self.road_min_distance ** 2)
        constraint = self.init_penalty * (ge_car.mean() + ge_road.mean())
        return l_current,constraint