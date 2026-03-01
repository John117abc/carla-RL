# src/agents/a2c_agent.py

import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import statistics

from typing import Dict, Any, Tuple,List, Union


from .base_agent import BaseAgent
from src.models.advantage_actor_critic import ActorNetwork, CriticNetwork
from src.utils import save_checkpoint,load_checkpoint
from src.buffer import TrajectoryBuffer
from src.utils import get_logger

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
        self.Q_matrix = np.diag([0.04, 0.04, 0.01, 0.01, 0.1, 0.02])
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

    def select_action(self, obs: Any, deterministic: bool = False):
        """
        根据观测选择动作。
        训练时返回随机动作和 log_prob；评估时返回均值。
        """
        with torch.no_grad():
            # 转为tensor
            obs_tensor = torch.from_numpy(obs[0]).to(self.device).float()
            if deterministic:
                action_scaled, log_prob, action_mean = self.actor(obs_tensor)
                action = action_mean
            else:
                action, log_prob, _ = self.actor(obs_tensor)
            action = action.cpu().numpy().flatten()
            log_prob = log_prob.cpu().numpy().flatten()
        return np.clip(action, self.action_space.low, self.action_space.high),log_prob

    def update_one_step(self,one_step_stata):
        s_all = torch.from_numpy(np.array(one_step_stata[0], dtype=np.float32)).to(self.device)
        s_ego = torch.from_numpy(np.array(one_step_stata[1][0], dtype=np.float32)).to(self.device)
        s_other = torch.from_numpy(np.array(one_step_stata[1][1], dtype=np.float32)).to(self.device)
        s_road = torch.from_numpy(np.array(one_step_stata[1][2], dtype=np.float32)).to(self.device)
        s_ref = torch.from_numpy(np.array(one_step_stata[1][3], dtype=np.float32)).to(self.device)

        action, _, _ = self.actor(s_all)

        # 跟踪消耗
        tracking_diff = s_ego - s_ref
        Q_diag = torch.from_numpy(np.diag(self.Q_matrix)).to(self.device).float()
        tracking = (tracking_diff * Q_diag * tracking_diff).sum()

        # 控制消耗
        R_diag = torch.from_numpy(np.diag(self.R_matrix)).to(self.device).float()
        control = (action * R_diag * action).sum()

        total_cost = (tracking + control)

        rel_pos_car = s_ego - s_other
        M_xy = torch.from_numpy(np.diag(self.M_matrix)).to(self.device).float()
        dist_sq_car = (rel_pos_car * M_xy * rel_pos_car).sum()
        g_car = dist_sq_car - self.other_car_min_distance ** 2
        ge_car = torch.relu(-g_car)

        # 道路约束
        rel_pos_road = s_ego - s_road
        dist_sq_road = (rel_pos_road * M_xy * rel_pos_road).sum()
        g_road = dist_sq_road - self.road_min_distance ** 2
        ge_road = torch.relu(-g_road)

        total_constraint = (ge_car + ge_road)

        actor_loss = total_cost + self.init_penalty * total_constraint

        # Actor更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic更新
        critic_pred = self.critic(s_all).squeeze()
        critic_target = total_cost.detach()
        critic_loss = F.mse_loss(critic_pred, critic_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return {
            "actor_loss": actor_loss.detach().item(),
            "critic_loss": critic_loss.detach().item()
        }

    def update(self):
        # 1. 采样 N 条完整轨迹
        trajectories = self.buffer.sample_batch(self.batch_size)

        all_states = []
        all_targets = []  # 存储每个状态对应的“剩余代价”
        all_actions = []

        actor_loss_sum = 0

        for traj in trajectories:
            # 解包数据 (假设 traj.states 是 [T, Dim])
            s_all, s_ego, s_other, s_road, s_ref = self.unpack_observation(traj.states, True)
            T = s_all.shape[0]  # 轨迹长度

            # 获取动作 (重新通过 Actor 生成，确保计算图连通，或者用 traj.actions)
            # 为了训练 Actor，我们需要梯度流过 action，所以必须 re-forward
            actions, _, _ = self.actor(s_all)  # [T, Action_Dim]

            # --- 计算每一步的即时代价 (Instant Cost) ---
            tracking_diff = s_ego - s_ref
            # 建议将 Q_diag, R_diag, M_xy 注册为 buffer 并在 init 时转到 device，避免循环内反复创建
            Q_diag = torch.from_numpy(np.diag(self.Q_matrix)).to(self.device).float()
            R_diag = torch.from_numpy(np.diag(self.R_matrix)).to(self.device).float()
            M_xy = torch.from_numpy(np.diag(self.M_matrix)).to(self.device).float()

            tracking = (tracking_diff * Q_diag * tracking_diff).sum(dim=1)  # [T]
            control = (actions * R_diag * actions).sum(dim=1)  # [T]
            instant_cost = tracking + control  # [T]

            # --- 计算每一步的即时约束 ---
            rel_pos_car = s_ego.unsqueeze(1) - s_other  # [T, N_other, Dim]
            # 注意：这里假设 s_other 包含所有周围车辆，需要聚合 (例如取最小距离)
            # 原代码直接 sum，这里保留原逻辑，但需注意维度
            dist_sq_car = (rel_pos_car * M_xy * rel_pos_car).sum(dim=-1)  # [T, N_other]
            dist_sq_car_min = dist_sq_car.min(dim=1)[0]  # 取最近的車 [T]

            g_car = dist_sq_car_min - self.other_car_min_distance ** 2
            ge_car = F.relu(-g_car)  # [T]

            rel_pos_road = s_ego - s_road
            dist_sq_road = (rel_pos_road * M_xy * rel_pos_road).sum(dim=1)  # [T]
            g_road = dist_sq_road - self.road_min_distance ** 2
            ge_road = F.relu(-g_road)  # [T]

            instant_constraint = ge_car + ge_road  # [T]

            # --- 计算剩余代价 (Cost-to-Go) ---
            # Target for Critic at time t: sum(cost[k] for k in t...T)
            # 使用 cumsum 倒序计算
            total_cost_traj = instant_cost + self.init_penalty * instant_constraint  # [T]

            # flip -> cumsum -> flip 得到每个时刻的剩余和
            remaining_cost = torch.flip(torch.cumsum(torch.flip(total_cost_traj, dims=[0]), dim=0), dims=[0])

            # --- 构建 Actor Loss ---
            # 论文中 Actor 是最小化整条轨迹的总代价
            # 我们这里对整条轨迹的总 Loss 求平均，避免梯度随轨迹长度爆炸
            trajectory_total_loss = total_cost_traj.sum()
            actor_loss_sum += trajectory_total_loss

            # --- 构建 Critic 训练数据 ---
            # 状态：s_all [T, Dim]
            # 目标：remaining_cost [T] (从 t 到结束的总代价)
            all_states.append(s_all)
            all_targets.append(remaining_cost)
            all_actions.append(actions)

        # 2. 合并 Batch
        states_batch = torch.cat(all_states, dim=0)  # [B*T, Dim]
        targets_batch = torch.cat(all_targets, dim=0)  # [B*T]
        # actions_batch = torch.cat(all_actions, dim=0)   # 如果 Critic 需要 action

        # 归一化 Loss (除以轨迹总步数)，防止长轨迹梯度爆炸
        num_total_steps = states_batch.shape[0]
        actor_loss = actor_loss_sum / num_total_steps

        # 3. Actor 更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 【关键】梯度裁剪，防止长时序累积梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 4. Critic 更新
        # 预测：V(s)
        critic_pred = self.critic(states_batch).squeeze()  # [B*T]

        # 目标：剩余代价 (已 detach)
        critic_target = targets_batch.detach()

        # 可选：对 Target 进行归一化或缩放，使其与 Pred 量级匹配
        # 如果训练不稳定，可以打印一下 target 的均值看看是否过大

        critic_loss = F.mse_loss(critic_pred, critic_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "avg_trajectory_cost": (targets_batch[0::int(num_total_steps / self.batch_size)]).mean().item()  # 粗略估算首状态代价
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
        self.history_loss.extend(save_info['history_loss'])

        model = {'actor': actor_model, 'critic': critic_model}
        optimizer = {'actor_optim': actor_optimizer, 'critic_optim': critic_optimizer}
        extra_info = {'config': save_info['rl_config'],
                      'global_step': self.global_step,
                      'history':self.history_loss,
                      'globe_eps':self.globe_eps}

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

    def eval(self, num_episodes: int = 10,action_repeat: int = 5) -> Tuple[float, float]:
        total_rewards = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False
            step = 0
            action = None
            while not done:
                # if step % action_repeat == 0:
                action,_ = self.select_action(obs['ocp_obs'])
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                # 不计算环境步，按照只有碰撞才停止
                done = terminated
                step +=1
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
        # constraint = self.init_penalty * (ge_car.mean() + ge_road.mean())
        constraint = ge_car.mean() + ge_road.mean()
        return l_current,constraint