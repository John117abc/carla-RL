# src/agents/a2c_agent.py

import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from typing import Dict, Any, Tuple,List, Union



from .base_agent import BaseAgent
from src.models.actor_critic import ActorNet,CriticNet
from src.utils import save_checkpoint,load_checkpoint
from src.buffer import TrajectoryBuffer
from src.utils import get_logger
from ..carla_utils import bicycle_model
from ..carla_utils.ocp_setup import npc_model

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
        self.actor = ActorNet(np.prod(self.observation_space['ocp_obs'].shape), hidden_dim=self.ocp_config['hidden_dim']).to(self.device)
        self.critic = CriticNet(np.prod(self.observation_space['ocp_obs'].shape),hidden_dim=self.ocp_config['hidden_dim']).to(self.device)

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
        # self.buffer = TrajectoryBuffer(min_start_train = self.ocp_config['min_start_train'],
        #                                total_capacity = self.ocp_config['total_capacity'])

        self.buffer = []

        self.predict_step = self.ocp_config['predict_step']
        self.step_time = self.ocp_config['step_time']

        # 记录历史日志数据值
        self.globe_eps = 0
        self.history_loss = []
        self.global_step = 0

    def sample_batch(self,data_list, batch_size):
        """
        从列表中随机取样 batch_size 个样本。

        参数:
            data_list (list): 原始数据列表
            batch_size (int): 需要取样的样本数量

        返回:
            list: 包含随机样品的列表

        注意:
            如果 batch_size 大于列表长度，将返回整个列表（不放回抽样）。
            如果需要放回抽样（即允许重复），请设置 replace=True。
        """
        if batch_size <= 0:
            return []

        # 不放回抽样（默认）
        if batch_size >= len(data_list):
            return data_list[:]  # 返回副本

        return random.sample(data_list, batch_size)

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

    def online_update(self, all_state, ref_road, static_road):
        s_all, s_ego, s_other, s_road, s_ref = self.unpack_observation(all_state)

        # 初始化累积变量 (不需要 requires_grad=True，因为它们是累加结果，最后总 loss 才需要图)
        # 注意：total_cost_traj 需要保留图以便 actor backward，所以不能 detach
        total_cost_traj = torch.tensor(0.0, device=self.device)
        # 用于记录每一步的 cost，方便调试，但不直接参与最终的 actor backward 累加逻辑（除非你要做每步优化）
        instant_costs = []

        # 临时状态变量，用于滚动预测
        curr_s_ego = s_ego.clone()
        curr_s_other = s_other.clone()
        curr_s_all = s_all.clone()  # 关键：需要一个随时间变化的状态输入

        ref_road_tensor = torch.tensor(ref_road,dtype=torch.float32).to(self.device).clone()

        # 预加载矩阵到 device (优化性能)
        if not hasattr(self, 'Q_diag_buf'):
            self.Q_diag_buf = torch.from_numpy(np.diag(self.Q_matrix)).to(self.device).float()
            self.R_diag_buf = torch.from_numpy(np.diag(self.R_matrix)).to(self.device).float()
            self.M_xy_buf = torch.from_numpy(np.diag(self.M_matrix)).to(self.device).float()

        # --- 轨迹 rollout ---
        for t in range(self.predict_step):
            # 1. Actor 基于【当前时刻】的状态输出动作
            # 假设 s_all 的结构允许切片或更新，如果 s_all 是固定结构，可能需要重新打包
            # 这里假设 curr_s_all 已经包含了最新的 ego/other 信息
            action = self.actor(curr_s_all)

            # 2. 动力学模型推演下一状态
            # 注意维度处理，确保 model 接收正确的 shape
            next_ego = bicycle_model(curr_s_ego.view(1, 1, -1), action.view(1, 1, -1), self.step_time)
            next_other = npc_model(curr_s_other.view(1, 1, curr_s_other.shape[0], -1), self.step_time).view(8,-1)

            curr_ref_state = s_ref + t
            curr_road_state = s_road  + t

            # 3. 计算即时 Cost
            tracking_diff = next_ego - curr_ref_state
            tracking = (tracking_diff * self.Q_diag_buf * tracking_diff).sum()
            control = (action * self.R_diag_buf * action).sum()

            # 约束成本
            rel_pos_road = next_ego - curr_road_state
            dist_sq_road = (rel_pos_road * self.M_xy_buf * rel_pos_road).sum(dim=-1)
            g_road = dist_sq_road - self.road_min_distance ** 2
            ge_road = F.relu(-g_road).sum()

            step_cost = tracking + control + self.init_penalty * ge_road

            # 累加到总代价 (这一步构建了从 action -> total_cost 的计算图)
            total_cost_traj = total_cost_traj + step_cost
            instant_costs.append(step_cost.detach().item())  # 仅记录数值

            # 4. 【关键】更新状态用于下一步
            curr_s_ego = next_ego.detach()  # 滚动预测时通常 detach，防止图太长爆炸，除非你需要端到端梯度
            curr_s_other = next_other.detach()

            # 重新构建 curr_s_all (将新的 ego/other 填入观察向量)
            # 这一步至关重要，否则 actor 永远看到初始状态
            curr_s_all = self.pack_observation(curr_s_ego, curr_s_other, s_road, s_ref)

            # ------------------ Train Critic -------------------------
        # Critic 的目标：预测初始状态 s_all 下的总代价 total_cost_traj
        # 注意：Critic 输入是初始状态 s_all，输出是一个标量估值
        pred_value = self.critic(s_all).squeeze()  # 去掉多余维度

        # Target 是 rollout 出来的真实总代价 (detach 掉，不让梯度流向 actor)
        target_value = total_cost_traj.detach()

        critic_loss = F.mse_loss(pred_value, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------ Train Actor --------------------------
        # Actor 的目标：最小化 total_cost_traj
        self.actor_optimizer.zero_grad()

        # 此时 total_cost_traj 包含完整的从 action -> cost 的图
        total_cost_traj.backward()

        # 可选：梯度裁剪，防止爆炸
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

        self.actor_optimizer.step()

        return {
            "actor_loss": total_cost_traj.item(),
            "critic_loss": critic_loss.item(),
            "step_costs": instant_costs
        }

    # 辅助函数：你需要实现这个来更新观察向量
    def pack_observation(self, ego, other, road, ref):
        # 根据你 unpack_observation 的逆过程，重新组装 tensor
        # 必须保证形状和 s_all 一致
        return torch.cat([ego.view(-1), other.view(-1), road, ref], dim=-1)  # 示例逻辑

    def update(self):
        # 0. 预处理常量 (移出循环，避免重复创建和潜在的图问题)
        # 建议在 __init__ 中完成此步，这里为了演示放在这里
        if not hasattr(self, '_diag_tensors'):
            self.Q_diag_tensor = torch.from_numpy(np.diag(self.Q_matrix).copy()).to(self.device).float()
            self.R_diag_tensor = torch.from_numpy(np.diag(self.R_matrix).copy()).to(self.device).float()
            self.M_xy_tensor = torch.from_numpy(np.diag(self.M_matrix).copy()).to(self.device).float()
            self._diag_tensors = True

        Q_diag = self.Q_diag_tensor
        R_diag = self.R_diag_tensor
        M_xy = self.M_xy_tensor

        # 调试：检查权重矩阵是否为0
        if R_diag.sum() == 0:
            print("ERROR: R_matrix is all zeros. Actor loss will be 0 w.r.t actions.")

        trajectories = self.sample_batch(self.buffer,self.batch_size)
        all_states = []
        all_targets = []

        # 使用列表收集 loss，最后 sum，比循环累加 tensor 更稳健
        trajectory_losses = []

        for traj in trajectories:
            s_ego, s_other, s_road, s_ref = self.unpack_tensor(traj)
            full_state = traj  # 假设这是 (Batch, Time, State_Dim)

            # 展平到 (Batch*Time, State_Dim) 以便统计
            # 前向传播
            actions = self.actor(traj).squeeze()  # [T, Action_Dim]

            # 1. 控制代价 (直接依赖 actions)
            # 确保 R_diag 广播正确
            control = (actions * R_diag * actions).sum(dim=1)  # [T]

            # 2. 追踪代价 (依赖 s_ego, s_ref，通常不直接依赖 actions，除非 s_ego 是预测出来的)
            # 注意：如果 s_ego 是从 buffer 采样的真实状态，它对 actions 的梯度为 0。
            # 只有 control 项提供梯度给 Actor。这是正常的 On-Policy/Off-Policy AC 算法逻辑吗？
            # 如果是 Off-Policy (如 SAC/DDPG)，通常 Loss = Q(s, a)。
            # 如果是这种直接最小化 Cost 的结构 (类似 MPC 或 直接策略搜索)，
            # 必须确保 Cost 函数里有项是直接关于 action 的。这里 control 项就是。

            tracking_diff = s_ego - s_ref
            tracking = (tracking_diff * Q_diag * tracking_diff).sum(dim=1)  # [T] (梯度为0 w.r.t action)

            instant_cost = tracking + control

            # 3. 约束代价
            rel_pos_car = s_ego.unsqueeze(1) - s_other
            dist_sq_car = (rel_pos_car * M_xy * rel_pos_car).sum(dim=-1)
            dist_sq_car_min = dist_sq_car.min(dim=1)[0]
            g_car = dist_sq_car_min - self.other_car_min_distance ** 2
            ge_car = F.relu(-g_car)

            rel_pos_road = s_ego - s_road
            dist_sq_road = (rel_pos_road * M_xy * rel_pos_road).sum(dim=1)
            g_road = dist_sq_road - self.road_min_distance ** 2
            ge_road = F.relu(-g_road)

            instant_constraint = ge_car + ge_road

            total_cost_traj = instant_cost + self.init_penalty * instant_constraint

            # 累加该轨迹的总 loss
            trajectory_losses.append(total_cost_traj.sum())

            # Critic 数据准备 (保持不变)
            remaining_cost = torch.flip(torch.cumsum(torch.flip(total_cost_traj, dims=[0]), dim=0), dims=[0])
            all_states.append(traj.squeeze())
            all_targets.append(remaining_cost)

        # 合并 Actor Loss
        if len(trajectory_losses) == 0:
            return {"actor_loss": 0, "critic_loss": 0}

        actor_loss_sum = torch.stack(trajectory_losses).mean()  # 或者 sum()，看你的 batch 定义

        # --- Actor Update ---
        self.actor_optimizer.zero_grad()
        actor_loss_sum.backward()

        # 【调试关键】检查梯度
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # 先裁剪再检查或直接检查

        self.actor_optimizer.step()

        # --- Critic Update (保持不变，略) ---
        states_batch = torch.cat(all_states, dim=0)
        targets_batch = torch.cat(all_targets, dim=0).detach()

        critic_pred = self.critic(states_batch).squeeze()
        critic_loss = F.mse_loss(critic_pred, targets_batch)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss_sum.item(),
            "critic_loss": critic_loss.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
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

    def unpack_tensor(self, data: torch.Tensor):
        """
        解包形状为 [B, N, 66] 的 Tensor。

        参数:
            data (torch.Tensor): 输入 tensor，形状应为 [B, N, 66]
                - B: Batch size (批次大小)
                - N: Number of scenarios/agents per batch (每批次的场景/智能体数量)

        返回:
            tuple: (ego_state, neighbor_states, road_state, ref_state)
                - ego_state: [B, N, 6]
                - neighbor_states: [B, N, 8, 6]
                - road_state: [B, N, 6]
                - ref_state: [B, N, 6]
        """
        # 1. 验证输入维度：现在应该是 3 维 [B, N, 66]
        if data.dim() != 3 or data.shape[2] != 66:
            raise ValueError(f"输入 Tensor 形状必须为 [B, N, 66]，当前形状为 {data.shape}")

        # 获取 B 和 N 以便后续 reshape 使用
        B, N = data.shape[0], data.shape[1]

        # 2. Ego 状态: 索引 [0, 6)
        # 切片操作会自动保留前两个维度 [B, N]
        ego_state = data[:, :, 0:6]  # Shape: [B, N, 6]

        # 3. 周车状态: 索引 [6, 54) -> 长度 48
        # 原始形状: [B, N, 48] -> 目标形状: [B, N, 8, 6]
        neighbor_raw = data[:, :, 6:54]
        other_states = neighbor_raw.view(B, N, 8, 6)  # Shape: [B, N, 8, 6]

        # 4. 道路状态: 索引 [54, 60)
        road_state = data[:, :, 54:60]  # Shape: [B, N, 6]

        # 5. 参考状态: 索引 [60, 66)
        ref_state = data[:, :, 60:66]  # Shape: [B, N, 6]

        return ego_state.squeeze(), other_states.squeeze(), road_state.squeeze(), ref_state.squeeze()

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
            return state_all, state_ego, state_other, state_road, state_ref
        else:
            state_ego = to_tensor(obs[0:6])
            state_other = to_tensor(obs[6:54])
            state_road = to_tensor(obs[54:60])
            state_ref = to_tensor(obs[60:66])

            state_all = torch.cat([
                state_ego,
                state_other,
                state_road,
                state_ref
            ], dim=0)
            return state_all, state_ego, state_other.view(8, -1), state_road, state_ref




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
        # constraint = ge_car.mean() + ge_road.mean()
        return l_current,constraint