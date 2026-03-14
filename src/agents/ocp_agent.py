# src/agents/ocp_agent.py
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Union
from .base_agent import BaseAgent
from src.models.actor_critic import ActorNet, CriticNet
from src.models.bicycle import BicycleModel
from src.utils import save_checkpoint, load_checkpoint
from src.buffer import StochasticBuffer
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
        assert isinstance(self.action_space, gym.spaces.Box), "OCP智能体需要连续的动作空间。"

        # 读取配置参数
        rl_algorithm = "OCP"
        self.base_config = rl_config['rl']
        self.ocp_config = rl_config['rl'][rl_algorithm]

        # 网络
        self.dt = self.ocp_config['dt']
        self.actor = ActorNet(state_dim=np.prod(self.observation_space['ocp_obs'].shape),
                              hidden_dim=self.ocp_config['hidden_dim']).to(self.device)
        self.critic = CriticNet(state_dim=np.prod(self.observation_space['ocp_obs'].shape),
                                hidden_dim=self.ocp_config['hidden_dim']).to(self.device)
        self.dynamics_model = BicycleModel(dt=self.dt, L=2.9)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.ocp_config['lr_actor'],
                                          betas=(0.9, 0.999), eps=1e-8)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.ocp_config['lr_critic'],
                                           betas=(0.9, 0.999), eps=1e-8)

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

        # 论文核心：误差状态权重矩阵（对应δp, δφ, δv）
        self.q_lat = 0.04  # 横向跟踪误差权重
        self.q_head = 0.1  # 航向误差权重
        self.q_speed = 0.01  # 速度误差权重

        # 控制权重矩阵
        self.R_matrix = np.diag([0.005, 0.1])

        # 采样数量
        self.batch_size = self.ocp_config['batch_size']
        self.horizon = self.ocp_config['horizon']

        # 初始化缓冲区
        self.buffer = StochasticBuffer(min_start_train=self.ocp_config['min_start_train'],
                                       total_capacity=self.ocp_config['total_capacity'])

        # 记录历史日志数据值
        self.globe_eps = 0
        self.history_loss = []
        self.global_step = 0
        self.gep_iteration = 0  # 对应论文算法1的外层迭代次数 i

    def select_action(self, obs: Any, deterministic: bool = False):
        """
        根据观测选择动作，强制正向加速度，打破死亡循环,确保：
        1. 正确处理 list 格式的观测；
        2. 动作缩放/裁剪到环境期望的 [-1,1]；
        3. 动作维度顺序正确（[加速度, 转向角]）；
        4. 训练初期强制正向探索。
        """
        with torch.no_grad():
            # 1. 正确解析观测（处理 list 格式）
            if isinstance(obs, list) and len(obs) == 2:
                obs_np = np.array(obs[0], dtype=np.float32)
            elif isinstance(obs, np.ndarray):
                obs_np = obs
            else:
                obs_np = np.array(obs, dtype=np.float32).flatten()

            # 2. 检查并修复观测维度
            if obs_np.ndim > 1:
                obs_np = obs_np.flatten()
            if obs_np.shape[0] != np.prod(self.observation_space['ocp_obs'].shape):
                logger.warning(
                    f"⚠️ 观测维度异常: {obs_np.shape}, 期望: {np.prod(self.observation_space['ocp_obs'].shape)}")
                obs_np = np.zeros(np.prod(self.observation_space['ocp_obs'].shape), dtype=np.float32)

            # 3. 转为 tensor 并输入 Actor
            obs_tensor = torch.from_numpy(obs_np).to(self.device).float()
            raw_action = self.actor(obs_tensor.unsqueeze(0)).squeeze(0)

            # 4. 正确缩放动作到环境期望的 [-1,1]
            action_low = torch.tensor(self.action_space.low).to(self.device)
            action_high = torch.tensor(self.action_space.high).to(self.device)
            action = torch.clamp(raw_action, action_low, action_high)
            action = action.cpu().numpy().flatten()

            # 5. 定向探索，打破不动的死亡循环
            if not deterministic:
                if self.gep_iteration < 5000:
                    # 前5000次迭代：强制正向加速度探索
                    accel_noise = np.random.normal(0.5, 0.2, size=1)  # 均值0.5，保证正向
                    steer_noise = np.random.normal(0, 0.1, size=1)  # 转向小噪声
                    noise = np.concatenate([accel_noise, steer_noise])
                    action = np.clip(action + noise, -1.0, 1.0)
                else:
                    # 后期：正常高斯探索
                    noise = np.random.normal(0, 0.1, size=action.shape)
                    action = np.clip(action + noise, -1.0, 1.0)

            return action, np.zeros(1, dtype=np.float32)

    def update(self):
        """
        完全对齐论文《Integrated Decision and Control》算法1：Dynamic Optimal Tracking-Offline Training
        核心逻辑：
        1. 每次迭代都执行：策略评估 (PEV) -> 更新 Critic（贝尔曼方程拟合代价-to-go）
        2. 仅当 iteration % m == 0 时执行：放大惩罚因子 ρ -> 策略改进 (PIM) -> 更新 Actor
        """
        # ------------------------------------------
        # 1. 数据准备：从 Buffer 采样 Batch
        # ------------------------------------------
        batch_data = self.buffer.sample_batch(self.batch_size)
        if len(batch_data) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "actor_updated": False}

        # 解析 Batch 数据
        states_list = []
        infos = []
        for item in batch_data:
            state, action, reward, _, done, info = item
            obs_vector = state[0] if isinstance(state, (list, tuple)) else state
            states_list.append(np.asarray(obs_vector, dtype=np.float32))
            infos.append(info)

        # 转为 Tensor：初始状态 s_0
        states_np = np.stack(states_list, axis=0)
        state_tensor = torch.from_numpy(states_np).to(self.device).float()
        state_tensor.requires_grad_(True)

        # 预处理参考线和道路边界
        max_len_ref = max(len(info.get('ref_path_xy', [])) for info in infos)
        max_len_left = max(len(info.get('static_road_left', [])) for info in infos)
        max_len_right = max(len(info.get('static_road_right', [])) for info in infos)

        ref_batch_list, left_batch_list, right_batch_list, cleaned_refs = [], [], [], []
        max_len = max(len(info.get('ref_path_xy', [])) for info in infos)

        for info in infos:
            # 处理参考线 Ref
            r_raw = np.atleast_2d(np.asarray(info.get('ref_path_xy', []), dtype=np.float32)).reshape(-1, 2)
            if len(r_raw) < max_len_ref:
                r_raw = np.vstack([r_raw, np.zeros((max_len_ref - len(r_raw), 2), dtype=np.float32)])
            ref_batch_list.append(r_raw)

            # 处理左边界 Left
            l_raw = np.atleast_2d(np.asarray(info.get('static_road_left', []), dtype=np.float32)).reshape(-1, 2)
            if len(l_raw) < max_len_left:
                l_raw = np.vstack([l_raw, np.zeros((max_len_left - len(l_raw), 2), dtype=np.float32)])
            left_batch_list.append(l_raw)

            # 处理右边界 Right
            r_raw_road = np.atleast_2d(np.asarray(info.get('static_road_right', []), dtype=np.float32)).reshape(-1, 2)
            if len(r_raw_road) < max_len_right:
                r_raw_road = np.vstack([r_raw_road, np.zeros((max_len_right - len(r_raw_road), 2), dtype=np.float32)])
            right_batch_list.append(r_raw_road)

            # 生成带速度和航向的参考线 [x, y, v, phi]
            raw_points = np.atleast_2d(np.asarray(info.get('ref_path_xy', []), dtype=np.float32)).reshape(-1, 2)
            if len(raw_points) == 0:
                processed_line = np.zeros((max_len, 4), dtype=np.float32)
            else:
                N = len(raw_points)
                xs, ys = raw_points[:, 0], raw_points[:, 1]
                dx = np.diff(xs, prepend=xs[0])
                dy = np.diff(ys, prepend=ys[0])
                phis = np.arctan2(dy, dx)
                vs = np.full(N, 5.0, dtype=np.float32)
                processed_line = np.stack([xs, ys, vs, phis], axis=1)
                if N < max_len:
                    processed_line = np.vstack([processed_line, np.zeros((max_len - N, 4), dtype=np.float32)])
            cleaned_refs.append(processed_line)

        # 转为 Tensor
        current_ref_xy = torch.from_numpy(np.stack(cleaned_refs)).to(self.device).float()
        current_road_left_xy = torch.from_numpy(np.stack(left_batch_list)).to(self.device).float()
        current_road_right_xy = torch.from_numpy(np.stack(right_batch_list)).to(self.device).float()

        # 2. 有限时域前向推演
        trajectory_actions = []
        trajectory_states = []

        # 解包初始状态
        ego_state, other_states, road_state, ref_state = self.unpack_tensor(state_tensor.unsqueeze(1))

        # 初始化当前状态
        current_ego = ego_state.clone()
        current_other = other_states.clone()
        current_all = state_tensor.clone()

        for t in range(self.horizon):
            # Actor 输出动作
            action = self.actor(current_all)
            trajectory_actions.append(action)

            # 动力学模型推演
            next_ego_state = self.dynamics_model(current_ego, action)

            # 其他车辆预测
            next_other_state = self.predict_other_next_batch(current_other, self.dt)

            # 获取下一刻参考点和道路观测
            ego_x = next_ego_state[:, 0, 0]
            ego_y = next_ego_state[:, 0, 1]
            next_ref_state = self.get_ref_observation_torch(ego_x, ego_y, current_ref_xy)
            next_road_state = self.get_road_observation_torch(ego_x, ego_y, current_road_left_xy,
                                                              current_road_right_xy)

            # 组装下一刻完整状态
            next_state_flat = torch.cat([
                next_ego_state.view(next_ego_state.shape[0], -1),
                next_other_state.view(next_other_state.shape[0], -1),
                next_road_state,
                next_ref_state
            ], dim=1)

            trajectory_states.append(next_state_flat)

            # 更新状态
            current_ego = next_ego_state
            current_other = next_other_state
            current_all = next_state_flat

        # 拼接轨迹数据
        states_traj = torch.stack(trajectory_states).transpose(0, 1)  # [Batch, Horizon, 66]
        actions_traj = torch.stack(trajectory_actions).transpose(0, 1)  # [Batch, Horizon, 2]

        # 解包轨迹
        s_ego, s_other, s_road, s_ref = self.unpack_tensor(states_traj)
        s_ego = s_ego.squeeze(2)
        s_other = s_other.squeeze(2)
        s_road = s_road.squeeze(2)
        s_ref = s_ref.squeeze(2)

        # 3.计算论文定义的误差状态与成本
        dx = s_ref[..., 0] - s_ego[..., 0]
        dy = s_ref[..., 1] - s_ego[..., 1]
        psi_ref = s_ref[..., 4]
        psi_ego = s_ego[..., 4]
        v_ref = s_ref[..., 2]
        v_ego = s_ego[..., 2]

        # 论文核心：横向跟踪误差δp（唯一惩罚的位置误差）
        lat_err = -dx * torch.sin(psi_ref) + dy * torch.cos(psi_ref)
        # 航向误差δφ
        head_err = torch.atan2(torch.sin(psi_ego - psi_ref), torch.cos(psi_ego - psi_ref))
        # 速度误差δv
        speed_err = v_ref - v_ego

        # 计算每一步的即时成本 l(s_t, u_t)
        r_weights = torch.from_numpy(np.diag(self.R_matrix).copy()).to(self.device).float().view(1, 1, -1)
        step_l = (
                self.q_lat * (lat_err ** 2)
                + self.q_head * (head_err ** 2)
                + self.q_speed * (speed_err ** 2)
                + (r_weights * (actions_traj ** 2)).sum(dim=-1)
        )

        # 计算每一步的约束违反 φ(s_t, u_t)
        # 1. 跟踪误差约束
        track_threshold = 1.0
        g_track = torch.abs(lat_err) + 0.5 * torch.abs(head_err) - track_threshold
        violation_track = F.relu(g_track)

        # 2. 车辆碰撞约束
        violation_car = torch.zeros_like(violation_track)
        if s_other is not None and s_other.numel() > 0:
            rel_pos_car = s_ego.unsqueeze(2) - s_other
            dist_sq_car = (rel_pos_car[..., :2] ** 2).sum(dim=-1)
            g_car = self.other_car_min_distance ** 2 - dist_sq_car
            violation_car = F.relu(g_car).max(dim=-1)[0]

        # 3. 道路边界约束
        ego_xy = s_ego[..., :2]
        min_road_dist = self.calc_ego_to_road_dist(ego_xy, current_road_left_xy, current_road_right_xy)
        g_road = self.road_min_distance - min_road_dist
        violation_road = F.relu(g_road)

        # 总约束违反
        step_phi = violation_track + violation_car + violation_road

        # 速度奖励（鼓励前进）
        speed_reward = 0.05 * torch.clamp(v_ego, 0, 10)

        # 增广成本：l + ρ·φ - 速度奖励
        step_aug_l = step_l + self.init_penalty * step_phi - speed_reward

        # 累积成本（用于Actor更新）
        instant_cost = step_l.mean(dim=0).sum() - speed_reward.mean(dim=0).sum()
        instant_penalty = step_phi.mean(dim=0).sum()

        # 4. 策略评估 (PEV)：贝尔曼方程拟合代价-to-go
        # 准备全轨迹状态
        all_states = torch.cat([state_tensor.unsqueeze(1), states_traj], dim=1)  # [Batch, Horizon+1, 66]

        # 从后往前递推贝尔曼目标值
        targets = torch.zeros_like(step_aug_l)
        targets[:, -1] = step_aug_l[:, -1]
        for t in reversed(range(self.horizon - 1)):
            # Critic更新时，next_state用detach()，不影响Actor计算图
            next_state = all_states[:, t + 1].detach()
            next_value = self.critic(next_state).squeeze()
            targets[:, t] = step_aug_l[:, t] + self.gamma * next_value

        # 拟合Critic（输入也用detach，彻底隔离计算图）
        critic_inputs = all_states[:, :-1].detach().reshape(-1, all_states.shape[-1])
        critic_targets = targets.detach().reshape(-1, 1)

        pred = self.critic(critic_inputs)
        critic_loss = F.mse_loss(pred, critic_targets)

        self.critic_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        critic_loss.backward()  # 不需要retain_graph了，因为全用了detach
        self.critic_optimizer.step()

        # 5. 策略改进 (PIM)：仅当 i mod m == 0 时执行
        self.gep_iteration += 1
        actor_updated = False
        actor_loss = torch.tensor(0.0)

        if self.gep_iteration % self.amplifier_m == 0:
            # 5.1 放大惩罚因子
            old_penalty = self.init_penalty
            self.init_penalty = min(self.init_penalty * self.amplifier_c, self.max_penalty)
            logger.info(
                f"[GEP] 迭代 {self.gep_iteration}: 放大惩罚因子 ρ: {old_penalty:.4f} -> {self.init_penalty:.4f}")

            # 5.2 【关键】用新ρ重新计算增广代价（确保计算图完整）
            # 重新计算一遍损失，保证梯度能从Actor一路回传到状态
            # （虽然重复计算，但保证了计算图的独立性和完整性）
            lat_err_new = -dx * torch.sin(psi_ref) + dy * torch.cos(psi_ref)
            head_err_new = torch.atan2(torch.sin(psi_ego - psi_ref), torch.cos(psi_ego - psi_ref))
            speed_err_new = v_ref - v_ego

            step_l_new = (
                    self.q_lat * (lat_err_new ** 2)
                    + self.q_head * (head_err_new ** 2)
                    + self.q_speed * (speed_err_new ** 2)
                    + (r_weights * (actions_traj ** 2)).sum(dim=-1)
            )

            # 重新计算约束违反
            g_track_new = torch.abs(lat_err_new) + 0.5 * torch.abs(head_err_new) - track_threshold
            violation_track_new = F.relu(g_track_new)

            violation_car_new = torch.zeros_like(violation_track_new)
            if s_other is not None and s_other.numel() > 0:
                rel_pos_car_new = s_ego.unsqueeze(2) - s_other
                dist_sq_car_new = (rel_pos_car_new[..., :2] ** 2).sum(dim=-1)
                g_car_new = self.other_car_min_distance ** 2 - dist_sq_car_new
                violation_car_new = F.relu(g_car_new).max(dim=-1)[0]

            min_road_dist_new = self.calc_ego_to_road_dist(ego_xy, current_road_left_xy, current_road_right_xy)
            g_road_new = self.road_min_distance - min_road_dist_new
            violation_road_new = F.relu(g_road_new)

            step_phi_new = violation_track_new + violation_car_new + violation_road_new
            speed_reward_new = 0.05 * torch.clamp(v_ego, 0, 10)

            # Actor损失：用新ρ计算，计算图完全独立
            actor_loss = (
                    step_l_new.mean(dim=0).sum()
                    + self.init_penalty * step_phi_new.mean(dim=0).sum()
                    - speed_reward_new.mean(dim=0).sum()
            )

            # 5.3 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()  # 不需要retain_graph，因为是全新的计算图
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()

            # 5.4 清空buffer（on-policy）
            self.buffer.clear()
            logger.info(f"[GEP] 迭代 {self.gep_iteration}: Actor已更新 {actor_loss.item() if torch.is_tensor(actor_loss) else 0.0}，已清空Buffer")

            actor_updated = True

        # 6. 返回日志
        return {
            "actor_loss": actor_loss.item() if torch.is_tensor(actor_loss) else 0.0,
            "critic_loss": critic_loss.item(),
            "penalty": self.init_penalty,
            "gep_iteration": self.gep_iteration,
            "actor_updated": actor_updated
        }

    def predict_other_next_batch(self, other_states, dt):
        """批量预测其他车辆下一时刻状态 (匀速模型)"""
        if other_states.dim() == 3:
            other_states = other_states.unsqueeze(1)
        if other_states.dim() != 4 or other_states.shape[3] != 6:
            raise ValueError(
                f"输入 Tensor 形状必须为 [B, N, N_npc, 6]，当前形状为 {other_states.shape}")

        B, N, n_npc, _ = other_states.shape
        x = other_states[:, :, :, 0]
        y = other_states[:, :, :, 1]
        vx = other_states[:, :, :, 2]
        vy = other_states[:, :, :, 3]
        psi = other_states[:, :, :, 4]
        omega = other_states[:, :, :, 5]

        x_next = x + dt * vx
        y_next = y + dt * vy
        psi_next = psi
        vx_next = vx
        vy_next = vy
        omega_next = omega

        next_other_states = torch.stack([x_next, y_next, vx_next, vy_next, psi_next, omega_next], dim=3)
        return next_other_states

    def calc_ego_to_road_dist(self, ego_xy: torch.Tensor, road_left_xy: torch.Tensor,
                              road_right_xy: torch.Tensor) -> torch.Tensor:
        """计算自车到道路左右边界的最小距离"""
        batch_size, horizon, _ = ego_xy.shape
        road_n = road_left_xy.shape[1]

        ego_xy_expand = ego_xy.unsqueeze(2)
        left_expand = road_left_xy.unsqueeze(1)
        right_expand = road_right_xy.unsqueeze(1)

        dist_left = torch.norm(ego_xy_expand - left_expand, dim=-1)
        dist_right = torch.norm(ego_xy_expand - right_expand, dim=-1)

        min_dist_left = dist_left.min(dim=-1)[0]
        min_dist_right = dist_right.min(dim=-1)[0]
        min_dist_to_road = torch.min(min_dist_left, min_dist_right)

        return min_dist_to_road

    def get_ref_observation_torch(self, ego_x: torch.Tensor, ego_y: torch.Tensor, ref_xy: torch.Tensor):
        """获取自车位置对应的最近参考点状态"""
        batch_size = ego_x.shape[0]
        ego_xy = torch.stack([ego_x, ego_y], dim=1)
        ego_xy_expand = ego_xy.unsqueeze(1)

        ref_xy_2d = ref_xy[..., :2]
        dists = torch.norm(ref_xy_2d - ego_xy_expand, dim=-1)
        min_dist_idx = torch.argmin(dists, dim=1)

        batch_idx = torch.arange(batch_size, device=ref_xy.device)
        closest_ref = ref_xy[batch_idx, min_dist_idx]

        dx = closest_ref[:, 0] - ego_x
        dy = closest_ref[:, 1] - ego_y

        ref_obs = torch.cat([closest_ref, dx.unsqueeze(1), dy.unsqueeze(1)], dim=1)
        return ref_obs.detach()

    def get_road_observation_torch(self, ego_x, ego_y, road_left_xy, road_right_xy):
        """批量获取道路边界观测"""

        def get_nearest_dist(ego_x, ego_y, road_map):
            ego_pos = torch.stack([ego_x, ego_y], dim=1).unsqueeze(-1)
            road_pos = road_map.transpose(1, 2)
            diff = road_pos - ego_pos
            dist_sq = torch.sum(diff ** 2, dim=1)
            min_dist_sq, _ = torch.min(dist_sq, dim=1)
            return torch.sqrt(min_dist_sq + 1e-6)

        dist_left = get_nearest_dist(ego_x, ego_y, road_left_xy)
        dist_right = get_nearest_dist(ego_x, ego_y, road_right_xy)
        zeros = torch.zeros_like(dist_left)
        road_obs = torch.stack([dist_left, dist_right, zeros, zeros, zeros, zeros], dim=1)
        return road_obs

    def save(self, save_info: Dict[str, Any]) -> None:
        """保存模型参数"""
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
                      'history': self.history_loss,
                      'globe_eps': self.globe_eps}
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

    def eval(self, num_episodes: int = 10, action_repeat: int = 5) -> Tuple[float, float]:
        total_rewards = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False
            step = 0
            while not done:
                action, _ = self.select_action(obs['ocp_obs'], deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated
                step += 1
            total_rewards.append(episode_reward)
        return float(np.mean(total_rewards)), float(np.std(total_rewards))

    def unpack_tensor(self, data: torch.Tensor):
        """解包形状为 [B, N, 66] 的 Tensor"""
        if data.dim() != 3 or data.shape[2] != 66:
            raise ValueError(f"输入 Tensor 形状必须为 [B, N, 66]，当前形状为 {data.shape}")

        B, N = data.shape[0], data.shape[1]
        ego_state = data[:, :, 0:6]
        neighbor_raw = data[:, :, 6:54]
        other_states = neighbor_raw.view(B, N, 8, 6)
        road_state = data[:, :, 54:60]
        ref_state = data[:, :, 60:66]

        return ego_state, other_states, road_state, ref_state