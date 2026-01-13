# src/agents/a2c_agent.py

import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple,List, Union

from .base_agent import BaseAgent
from src.models.advantage_actor_critic import ActorNetwork, CriticNetwork
from src.utils import save_checkpoint,load_checkpoint,get_logger
from src.buffer import Trajectory,TrajectoryBuffer
from src.envs.carla_env import CarlaEnv

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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.ocp_config['lr_actor'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.ocp_config['lr_critic'])

        # 损失函数
        self.loss_func = nn.MSELoss()

        # 超参数
        self.init_penalty = self.ocp_config['init_penalty']
        self.max_penalty = self.ocp_config['max_penalty']
        self.amplifier_c = self.ocp_config['amplifier_c']
        self.amplifier_m = self.ocp_config['amplifier_m']
        self.other_car_min_distance = self.ocp_config['other_car_min_distance']
        self.road_min_distance = self.ocp_config['road_min_distance']

        # 正定矩阵
        self.Q_matrix = np.diag([0.04, 0.04, 0.1, 0.01, 0.1, 0.02])
        self.R_matrix = np.diag([0.1, 0.005])
        self.M_matrix = np.diag([1,1,0,0,0,0])
        # 严格使用s^ref = [δp, δφ, δv ]状态时候的Q
        self.Q_matrix_ref = np.diag([0.04,0.1,0.01])

        # 采样数量
        self.batch_size = self.ocp_config['batch_size']

        # 初始化缓冲区
        self.buffer = TrajectoryBuffer(min_start_train = self.ocp_config['min_start_train'],
                                       total_capacity = self.ocp_config['total_capacity'])

    def select_action(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        """
        根据观测选择动作。
        训练时返回随机动作和 log_prob；评估时返回均值。
        """
        with torch.no_grad():
            obs_tensor, _, _, _, _ = self.unpack_observation(obs)
            if deterministic:
                _, _, action_mean = self.actor(obs_tensor)
                action = action_mean
            else:
                action, log_prob, _ = self.actor(obs_tensor)
            action = action.cpu().numpy().flatten()
            # logger.info(f'动作打印：{np.clip(action, self.action_space.low, self.action_space.high)}')
        return np.clip(action, self.action_space.low, self.action_space.high)

    def update(self):
        """
        更新 Actor 和 Critic
        """
        # 从 buffer 采样
        batch = self.buffer.sample_batch(self.batch_size)
        batch = np.asarray(batch,dtype=object).reshape(-1, 6)
        states = batch[:, 0]

        # 解包状态
        state_all,state_ego,state_other,state_road,state_ref = self.unpack_observation(states,True)

        # 获取 Q, R, M 矩阵
        Q = torch.from_numpy(self.Q_matrix).to(self.device).float()
        R = torch.from_numpy(self.R_matrix).to(self.device).float()
        M = torch.from_numpy(self.M_matrix).to(self.device).float()

        # 计算当前策略下的 action
        action_new, log_prob, _ = self.actor(state_all)

        # 计算 cost components
        tracking_error = ((state_ref - state_ego) @ Q) * (state_ref - state_ego)
        control_energy = (action_new @ R) * action_new
        l_current = tracking_error.mean() + control_energy.mean()

        # 计算约束项
        diff = state_ego.unsqueeze(1) - state_other
        dist_sq = (diff @ M).pow(2).sum(dim=-1)
        ge_car = torch.relu(self.other_car_min_distance ** 2 - dist_sq).mean()
        ge_road = torch.relu(-((state_ego - state_road) @ M).pow(2).sum(dim=-1) + self.road_min_distance ** 2)
        constraint = self.init_penalty * (ge_car.mean() + ge_road.mean())

        # Actor Loss
        actor_loss = l_current + self.init_penalty * constraint
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            v_target = l_current

        v_pred = self.critic(state_all).mean()
        critic_loss = self.loss_func(v_pred, v_target.detach())

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
        model = {'actor': actor_model, 'critic': critic_model}
        optimizer = {'actor_optim': actor_optimizer, 'critic_optim': critic_optimizer}
        extra_info = {'config': save_info['rl_config'], 'global_step': save_info['global_step'],'history':save_info['history_loss']}
        met = {'episode': save_info['episode']}
        save_checkpoint(
            model=model,
            model_name='ocp-v1.0',
            optimizer=optimizer,
            extra_info=extra_info,
            metrics=met,
            env_name=save_info['map']
        )

    def load(self, path: str) -> None:
        # 再创建优化器，不加载critic
        checkpoint = load_checkpoint(
            model={'actor': self.actor},
            filepath=path,
            optimizer={'actor_optim': self.actor_optimizer},
            device=self.device
        )
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


    def collect_trajectory(self,env:CarlaEnv, horizon=25):
        """
        收集轨迹
        :param env: 环境
        :param horizon: 收集轨迹中的步数
        :return: 轨迹
        """
        states, actions, rewards, infos = [], [], [], []
        state,_ = env.reset()
        initial_state = state.copy()
        for t in range(horizon):
            action = self.select_action(state)
            actions.append(action)
            states.append(state)
            next_state, reward_dict,  terminated, truncated, info = env.step(action)
            done = info['collision'] or info['off_route'] or info['TimeLimit.truncated']
            rewards.append(reward_dict)
            infos.append(info)

            state = next_state
            if done:
                break

        # 计算 total_cost 和 total_constraint
        total_cost, total_constraint = self.compute_total_cost_and_constraint(states, actions)

        return Trajectory(
            initial_state=initial_state,
            states=states,
            actions=actions,
            rewards=rewards,
            infos=infos,
            total_cost=total_cost,
            total_constraint=total_constraint,
            path_id=env.current_path_id,
            horizon=len(states)
        )

    def compute_total_cost_and_constraint(self,states,action):
        """
        计算这条轨迹的效用值和约束
        :param states:
        :param action:
        :return:
        """
        # 解包状态
        state_all, state_ego, state_other, state_road, state_ref = self.unpack_observation(states, True)

        # 获取 Q, R, M 矩阵
        Q = torch.from_numpy(self.Q_matrix).to(self.device).float()
        R = torch.from_numpy(self.R_matrix).to(self.device).float()
        M = torch.from_numpy(self.M_matrix).to(self.device).float()

        # 计算 cost components
        tracking_error = ((state_ref - state_ego) @ Q) * (state_ref - state_ego)
        control_energy = (action @ R) * action
        l_current = tracking_error.mean() + control_energy.mean()

        # 计算约束项
        diff = state_ego.unsqueeze(1) - state_other
        dist_sq = (diff @ M).pow(2).sum(dim=-1)
        ge_car = torch.relu(self.other_car_min_distance ** 2 - dist_sq).mean()
        ge_road = torch.relu(-((state_ego - state_road) @ M).pow(2).sum(dim=-1) + self.road_min_distance ** 2)
        constraint = self.init_penalty * (ge_car.mean() + ge_road.mean())

        return l_current,constraint
