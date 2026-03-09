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
from src.models.bicycle import BicycleModel
from src.utils import save_checkpoint,load_checkpoint
from src.buffer import TrajectoryBuffer
from src.utils import get_logger
from src.carla_utils import predict_ref_next_torch,predict_other_next,predict_road_torch

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
        self.dt = self.ocp_config['dt']
        self.actor = ActorNetwork(self.observation_space['ocp_obs'], self.action_space, hidden_dim=self.ocp_config['hidden_dim']).to(self.device)
        self.critic = CriticNetwork(self.observation_space['ocp_obs'],hidden_dim=self.ocp_config['hidden_dim']).to(self.device)
        self.dynamics_model = BicycleModel(dt=self.dt, L=2.9).to(self.device)

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

        # 预测步数
        self.horizon = self.ocp_config['horizon']

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


    def update(self):
        # 1. 从 Buffer 采样 N 个【初始状态】 (Initial States)
        # 形状: [Batch_Size, State_Dim]
        # 这些 s_0 包含了 ego, other, road, ref 的初始信息
        batch_s0 = self.buffer.sample_batch(self.batch_size)

        actor_loss_sum = 0
        critic_inputs = []
        critic_targets = []

        # # 动态调整 GEP 惩罚系数 rho
        # if self.global_step % self.m == 0:
        #     self.rho *= self.c
        # self.global_step += 1

        for s_0 in batch_s0:
            # --- 核心步骤：在计算图中进行轨迹推演 (Rollout) ---
            # 我们需要一个可微的动力学模型 (Differentiable Dynamics Model)
            # 这个模型通常是简单的自行车模型 (Bicycle Model)，可以用 PyTorch 轻松实现

            s_all, s_ego, s_other, s_road, s_ref = self.unpack_observation(s_0.states, True)

            current_s = [s_all, s_ego, s_other, s_road, s_ref]

            # 初始化
            trajectory_actions = []
            trajectory_states = []  # 存储每一步完整的 state_all (Tensor)

            # 【关键修改】：初始状态 s_init 应该是已经准备好的 Tensor (cat 好的)
            # 假设 s_init 是 t=0 时的 state_all
            current_state_all = s_all.clone()  # 确保是可导的副本
            current_ego_state = s_ego.clone()
            current_other_state = s_other.clone()
            # 因为每条轨迹的参考路径从第一次规划就确定了，所以取s_0.infos[0]
            current_ref_xy = torch.tensor(s_0.infos[0]['ref_path_xy'],dtype=torch.float32).to(self.device)
            current_static_road_xy = torch.tensor([item['static_road_xy'][0] for item in s_0.infos],dtype=torch.float32).to(self.device)

            # 如果需要固定的参考线/道路信息作为全局上下文，可以在循环外定义
            # 如果是随时间变化的参考线，需要有一个函数 get_ref(t) 来获取，而不是直接读数组
            # 假设 ref_trajectory 是一个预先定义好的 Tensor [horizon+1, dim]
            # road_info 同理
            dt = torch.tensor(self.dt)
            for t in range(self.horizon):
                # --- A. Actor 生成动作 ---
                # 输入必须是 Tensor
                action, _, _ = self.actor(current_state_all[0])
                trajectory_actions.append(action)

                # --- B. 动力学推演 (核心梯度路径) ---

                # 1. 自车动力学 (确保输出是 Tensor 且保留梯度)
                next_ego_state = self.dynamics_model(current_ego_state[0], action)

                # 2. 周车动力学
                next_other_state = predict_other_next(current_other_state[0], t, dt)
                # 如果 update_other_context 返回的是 list，记得转 tensor: torch.tensor(..., device=self.device)
                if isinstance(next_other_state, list):
                    next_other_state = torch.stack(next_other_state).to(self.device).view(-1)

                # 3. 获取下一时刻的道路/参考信息
                ego_x = next_ego_state[:,0][0].unsqueeze(0)
                ego_y = next_ego_state[:,1][0].unsqueeze(0)
                next_road_state = predict_road_torch(ego_x,ego_y,current_static_road_xy)
                next_ref_state = predict_ref_next_torch(ego_x,ego_y, current_ref_xy.unsqueeze(0))
                # --- C. 构建下一时刻的完整状态 (State Construction) ---
                # 展平所有部分
                state_ego_flat = next_ego_state.view(-1)
                state_other_flat = next_other_state.view(-1)
                state_road_flat = next_road_state.view(-1)
                state_ref_flat = next_ref_state.view(-1)

                # 拼接成新的 state_all (这是传递给下一步 actor 的输入)
                next_state_all = torch.cat([
                    state_ego_flat,
                    state_other_flat,
                    state_road_flat,
                    state_ref_flat
                ], dim=0)

                # --- D. 更新变量，进入下一轮 ---
                trajectory_states.append(next_state_all)

                # 更新当前状态指针
                current_state_all = next_state_all.unsqueeze(0)
                current_ego_state = next_ego_state.unsqueeze(0)
                current_other_state = next_other_state.unsqueeze(0)

            # 循环结束后，trajectory_states 包含了 t=1 到 t=horizon 的状态
            # trajectory_actions 包含了 t=0 到 t=horizon-1 的动作

                # --- 此时，我们得到了一条由【当前网络参数】生成的完整轨迹 ---
            # 这条轨迹是“活”的，梯度可以贯穿始终

            # 2. 计算代价 (Cost)
            # 将列表转为 Tensor
            states_traj = torch.stack(trajectory_states[:-1])  # [T, Dim]
            actions_traj = torch.stack(trajectory_actions)  # [T, Dim]

            # 计算 tracking, control, constraint
            # 注意：所有计算必须使用 torch 操作
            # 跟踪消耗
            # tracking_diff = s_ref - states_traj[1]
            # Q_diag = torch.from_numpy(np.diag(self.Q_matrix)).to(self.device).float()
            # tracking = (tracking_diff * Q_diag * tracking_diff).sum()
            #
            instant_cost = self.compute_instant_cost(states_traj, actions_traj)
            instant_penalty = self.compute_constraints(states_traj)  # 记得平方 **2

            # GEP 目标函数: J = sum(cost) + rho * sum(penalty)
            total_loss = instant_cost.sum() + self.init_penalty * instant_penalty.sum()

            actor_loss_sum += total_loss

            # 3. 准备 Critic 数据
            # Critic 输入：初始状态 s_0
            # Critic 目标：这条生成轨迹的总代价 (detach)
            critic_inputs.append(s_all.clone()[0])
            critic_targets.append(total_loss.detach())

        # --- 优化步骤 ---
        # Actor Update
        actor_loss = actor_loss_sum / self.batch_size
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Critic Update
        inputs_batch = torch.stack(critic_inputs)
        targets_batch = torch.stack(critic_targets)
        pred = self.critic(inputs_batch).squeeze()
        critic_loss = F.mse_loss(pred, targets_batch)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()
        }

    def compute_instant_cost(self,states_traj,actions_traj):
        """
        计算控制消耗
        :param states_traj: 状态信息
        :param actions_traj: 动作信息
        :return:
        """
        # 还原自车，周车，道路，参考信息
        s_ego, s_other, s_road, s_ref = self.unpack_tensor(data=states_traj)

        # 跟踪消耗
        tracking_diff = s_ref - s_ego
        Q_diag = torch.from_numpy(np.diag(self.Q_matrix)).to(self.device).float()
        tracking = (tracking_diff * Q_diag * tracking_diff).sum()

        # 控制消耗，
        R_diag = torch.from_numpy(np.diag(self.R_matrix)).to(self.device).float()
        control = (actions_traj * R_diag * actions_traj)

        total_cost = (tracking + control)

        return total_cost

    def compute_constraints(self,states_traj):
        # 还原自车，周车，道路，参考信息
        s_ego, s_other, s_road, s_ref = self.unpack_tensor(data=states_traj)

        rel_pos_car = s_ego.unsqueeze(1) - s_other
        M_xy = torch.from_numpy(np.diag(self.M_matrix)).to(self.device).float()
        dist_sq_car = (rel_pos_car * M_xy * rel_pos_car).sum()
        g_car = dist_sq_car - self.other_car_min_distance ** 2
        ge_car = torch.relu(-g_car)

        # 道路约束
        rel_pos_road = s_ego - s_road
        dist_sq_road = (rel_pos_road * M_xy * rel_pos_road)
        g_road = dist_sq_road - self.road_min_distance ** 2
        ge_road = torch.relu(-g_road)

        total_constraint = (ge_car + ge_road)

        return total_constraint

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


    def unpack_tensor(self,data: torch.Tensor):
        """
        解包形状为 [N, 66] 的 Tensor。

        参数:
            data (torch.Tensor): 输入 tensor，形状应为 [N, 66]

        返回:
            tuple: (ego_state, neighbor_states, road_state, ref_state)
                - ego_state: [N, 6]
                - neighbor_states: [N, 8, 6]
                - road_state: [N, 6] (根据索引 54:60 计算得出)
                - ref_state: [N, 6] (根据索引 60:66 计算得出)

        """
        if data.dim() != 2 or data.shape[1] != 66:
            raise ValueError(f"输入 Tensor 形状必须为 [N, 66]，当前形状为 {data.shape}")

        # 1. Ego 状态: 索引 [0, 6) -> 长度 6
        ego_state = data[:, 0:6]  # Shape: [N, 6]

        # 2. 周车状态: 索引 [6, 54) -> 长度 48 -> Reshape 为 [N, 8, 6]
        neighbor_raw = data[:, 6:54]
        other_states = neighbor_raw.view(neighbor_raw.shape[0], 8, 6)  # Shape: [N, 8, 6]

        # 3. 道路状态: 索引 [54, 60) -> 长度 6
        road_state = data[:, 54:60]  # Shape: [N, 6]

        # 4. 参考状态: 索引 [60, 66) -> 长度 6
        # 用户描述形状为 [6]，但 66-59=7。此处按索引切片。
        ref_state = data[:, 60:66]  # Shape: [N, 6]

        return ego_state, other_states, road_state, ref_state