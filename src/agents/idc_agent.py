# src/agents/idc_agent.py
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
from src.buffer import StochasticBuffer,PERBuffer
from src.utils import get_logger
from src.carla_utils import rect_min_dist_sq


logger = get_logger('idc_agent')


class OcpAgent(BaseAgent):
    """
    严格对齐论文《Integrated Decision and Control》Algorithm 2 (GEP) 实现
    """

    def __init__(
            self,
            rl_config: Dict[str, Any],
            env: gym.Env,
            device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(env, device)
        assert isinstance(self.action_space, gym.spaces.Box), "IDC智能体需要连续的动作空间。"

        # 读取配置
        rl_algorithm = "IDC"
        self.base_config = rl_config['rl']
        self.idc_config = rl_config['rl'][rl_algorithm]

        # 严格对齐论文的状态维度定义
        self.DIM_EGO = 6  # 自车 [x,y,v_lon,v_lat,phi,omega]
        self.DIM_OTHER = self.env.env_cfg['idc'][
                             'others'] * 4  # 8车×4维
        self.DIM_REF_ERROR = 3  # 跟踪误差 [δ_p, δ_φ, δ_v]
        self.TOTAL_STATE_DIM = self.DIM_EGO + self.DIM_OTHER + self.DIM_REF_ERROR

        # 道路维度
        self.DIM_ROAD = self.env.env_cfg['idc'][
                            'num_points'] * 4  # 道路80维

        self.road_state_buffer = None  # 用于存储当前的道路边缘信息

        # 核心参数初始化
        self.dt = self.idc_config['dt']
        self.horizon = self.idc_config['horizon']
        self.batch_size = self.idc_config['batch_size']

        # 网络初始化
        self.actor = ActorNet(
            state_dim=self.TOTAL_STATE_DIM,
            hidden_dim=self.idc_config['hidden_dim']
        ).to(self.device)
        self.critic = CriticNet(
            state_dim=self.TOTAL_STATE_DIM,
            hidden_dim=self.idc_config['hidden_dim']
        ).to(self.device)
        self.dynamics_model = BicycleModel(dt=self.dt, L=2.9)

        # 优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.idc_config['lr_actor'],
            betas=(0.9, 0.999)
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.idc_config['lr_critic'],
            betas=(0.9, 0.999)
        )

        # __init__ 中，修改成本函数权重
        # 极大地提高横向追踪权重，让网络不惜代价也要回到中心线
        self.q_lat = 0.30  # 从 0.04 提高 7.5 倍
        self.q_head = 0.02  # 从 0.1 降低，让位置控制主导方向
        self.q_speed = 0.01  # 保持不变

        # 放宽转向惩罚，让纠偏过程不被控制成本过分压抑
        # 但保留一个最低限度的代价，避免完全无成本的暴力转向
        self.R_matrix = np.diag([0.005, 0.02])  # 从 0.005, 0.08 修改

        # 车辆信息
        self.HALF_L = 2.25  # 车长的一半
        self.HALF_W = 1.0  # 车宽的一半

        # GEP算法超参数（严格对齐论文收敛逻辑）
        self.init_penalty = self.idc_config['init_penalty']
        self.max_penalty = self.idc_config['max_penalty']
        self.amplifier_c = self.idc_config['amplifier_c']
        self.amplifier_m = self.idc_config['amplifier_m']
        self.other_car_min_distance = self.idc_config['other_car_min_distance']
        self.road_min_distance = self.idc_config['road_min_distance']
        self.gamma = self.idc_config['gamma']

        # 缓冲区
        self.buffer = PERBuffer(capacity=100000, min_start_train=256)

        # 训练状态
        self.global_step = 0
        self.gep_iteration = 0
        self.history_loss = []
        self.globe_eps = 0

        # 参考速度固定为5m/s（18km/h），但环境实际参考速度由env.ego_ref_speed提供
        self.ref_vlon = self.env.ego_ref_speed

        # 预测轨迹
        self.predict_traj = None

        # 惩罚因子放大阈值（设为0，让惩罚因子尽早启动，增加阻尼）
        self.penalty_growth_threshold = 0.0

        # 校验配置一致性
        if self.DIM_OTHER < 0:
            raise ValueError("env_cfg['idc']['others'] 必须为非负整数，当前值导致 DIM_OTHER < 0")

    def _calc_ref_error_from_state(self, ego_state: torch.Tensor, ref_path_tensor: torch.Tensor) -> torch.Tensor:
        B = ego_state.shape[0]
        if ref_path_tensor.shape[0] == 1 and B > 1:
            ref_path_tensor = ref_path_tensor.repeat(B, 1, 1)

        ego_xy = ego_state[..., :2]  # [B,1,2]
        ego_phi = ego_state[..., 4]  # [B,1]
        ego_vlon = ego_state[..., 2]  # [B,1]

        # 最近点
        dist = torch.norm(ego_xy.unsqueeze(2) - ref_path_tensor.unsqueeze(1), dim=-1)  # [B,1,N]
        min_dist, closest_idx = torch.min(dist, dim=-1)  # [B,1]
        closest_idx = closest_idx.squeeze(1)  # [B]

        # 前视距离
        v_lon = ego_vlon.squeeze(1)  # [B]
        L_min = 2.0
        k = 0.5
        L_look = torch.clamp(L_min + k * v_lon, min=L_min, max=20.0)

        # 获取前视点
        look_xy, look_phi, look_idx = self._advance_along_path(
            ref_path_tensor, closest_idx, L_look
        )  # [B,2], [B], [B]

        # 横向误差
        ego_xy_sq = ego_xy.squeeze(1)  # [B,2]
        dx = look_xy[:, 0] - ego_xy_sq[:, 0]
        dy = look_xy[:, 1] - ego_xy_sq[:, 1]
        cross = dy * torch.cos(look_phi) - dx * torch.sin(look_phi)
        delta_p = cross

        # 航向误差
        delta_phi = ego_phi.squeeze(1) - look_phi
        delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))

        # 速度误差
        delta_v = v_lon - self.ref_vlon

        error = torch.stack([delta_p, delta_phi, delta_v], dim=-1).unsqueeze(1)
        # 保留日志
        if B > 0:
            _ego = ego_xy_sq[0].detach().cpu().numpy()
            _look = look_xy[0].detach().cpu().numpy()
            logger.debug(
                f"[LOOKAHEAD] v_lon={v_lon[0].item():.2f}, L={L_look[0].item():.2f}m, "
                f"look=({_look[0]:.3f},{_look[1]:.3f}), phi={look_phi[0].item():.4f}, dp={delta_p[0].item():.4f}"
            )
        return error

    def _forward_horizon(self,
                         state_tensor: torch.Tensor,
                         ref_path_tensor: torch.Tensor,
                         road_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        严格对齐论文 IDC 公式与 GEP 算法，
        """
        B = state_tensor.shape[0]
        ego_state, other_states, ref_error = self.unpack_tensor(state_tensor.unsqueeze(1))
        current_ego = ego_state.clone()
        current_other = other_states.clone()
        current_ref_error = self._calc_ref_error_from_state(current_ego, ref_path_tensor)

        # 【修复】道路状态解析：80维 -> 20左点 + 20右点 (自车坐标系)
        if road_state is not None:
            road_left = road_state[..., :int(self.DIM_ROAD / 2)].contiguous().view(B, 1, 20, 2)
            road_right = road_state[..., int(self.DIM_ROAD / 2):].contiguous().view(B, 1, 20, 2)
        else:
            road_left = torch.zeros(B, 1, 20, 2, device=self.device)
            road_right = torch.zeros(B, 1, 20, 2, device=self.device)

        # 安全距离平方
        safe_veh_sq = self.other_car_min_distance ** 2
        safe_road_sq = self.road_min_distance ** 2

        step_l_list = []
        step_phi_list = []
        trajectory_states = []

        for t in range(self.horizon):
            current_state = torch.cat([
                current_ego.view(-1, self.DIM_EGO),
                current_other.view(-1, self.DIM_OTHER),
                current_ref_error.view(-1, self.DIM_REF_ERROR)
            ], dim=1)

            norm_action = self.actor(
                current_state)  # [B, 2]
            a_phy = norm_action[
                        :, 0:1] * 2.25 - 0.75  # [-1,1] → [-3, 1.5] m/s²
            # 转向映射：正值→右转，负值→左转，直接乘以0.4
            delta_phy = norm_action[
                            :, 1:2] * 0.4  # [-1,1] → [-0.4,0.4] rad
            phy_action = torch.cat([a_phy, delta_phy], dim=1)

            next_ego = self.dynamics_model(current_ego, phy_action)
            next_other = self.predict_other_next_batch(current_other, self.dt)
            next_ref_error = self._calc_ref_error_from_state(next_ego, ref_path_tensor)

            next_state = torch.cat([
                next_ego.view(-1, self.DIM_EGO),
                next_other.view(-1, self.DIM_OTHER),
                next_ref_error.view(-1, self.DIM_REF_ERROR)
            ], dim=1)
            trajectory_states.append(next_state)

            # 1. 计算IDC成本项 step_l
            lat_err_t = next_ref_error[..., 0].squeeze(1)
            head_err_t = next_ref_error[..., 1].squeeze(1)
            speed_err_t = next_ref_error[..., 2].squeeze(1)
            # 【防御】限制误差范围，防止梯度爆炸
            # lat_err_t = torch.clamp(lat_err_t, -10.0, 10.0)
            # head_err_t = torch.clamp(head_err_t, -3.14, 3.14)
            # speed_err_t = torch.clamp(speed_err_t, -10.0, 10.0)
            err_cost = self.q_lat * (lat_err_t ** 2) + self.q_head * (head_err_t ** 2) + self.q_speed * (
                        speed_err_t ** 2)

            r_weights = torch.tensor(self.R_matrix.diagonal().copy(), device=self.device).float()
            control_cost = torch.sum((phy_action ** 2) * r_weights, dim=1)

            # 2. 补全约束违反量 step_phi 并计算周车原始违反量 (用于平滑惩罚)
            ego_xy = next_ego[..., :2]  # [B,1,2] (仍然保留，如果需要其他地方用，但双圆不再用它)
            phi_violation = torch.zeros(B, device=self.device)
            penalty_npc_raw = torch.zeros(B, device=self.device)

            # ============= 周车安全距离约束 (双圆覆盖模型) =============
            if self.DIM_OTHER > 0:
                # --- 1. 构造自车双圆中心 ---
                # 使用 next_ego [B,1,6]
                dist_ego = self.HALF_L * 1.0  # 圆心偏移 (约1.35m)
                ego_cos = torch.cos(next_ego[..., 4])  # phi
                ego_sin = torch.sin(next_ego[..., 4])
                ego_x = next_ego[..., 0]
                ego_y = next_ego[..., 1]

                ego_front_x = ego_x + dist_ego * ego_cos
                ego_front_y = ego_y + dist_ego * ego_sin
                ego_rear_x = ego_x - dist_ego * ego_cos
                ego_rear_y = ego_y - dist_ego * ego_sin

                ego_front = torch.stack([ego_front_x, ego_front_y], dim=-1)  # [B,1,2]
                ego_rear = torch.stack([ego_rear_x, ego_rear_y], dim=-1)  # [B,1,2]

                # --- 2. 构造周车双圆中心 ---
                # next_other 形状 [B,1,N,4] (x,y,phi,v_lon)
                other_x = next_other[..., 0]
                other_y = next_other[..., 1]
                other_phi = next_other[..., 2]  # 第三维是 phi
                other_v = next_other[..., 3]

                other_cos = torch.cos(other_phi)
                other_sin = torch.sin(other_phi)
                dist_other = self.HALF_L * 1.0  # 同样的偏移量

                other_front_x = other_x + dist_other * other_cos
                other_front_y = other_y + dist_other * other_sin
                other_rear_x = other_x - dist_other * other_cos
                other_rear_y = other_y - dist_other * other_sin

                other_front = torch.stack([other_front_x, other_front_y], dim=-1)  # [B,1,N,2]
                other_rear = torch.stack([other_rear_x, other_rear_y], dim=-1)  # [B,1,N,2]

                # --- 3. 计算四个圆对距离平方 ---
                # 扩展维度以便广播: 自车 [B,1,1,2] vs 周车 [B,1,N,2]
                ego_front_exp = ego_front.unsqueeze(2)  # [B,1,1,2]
                ego_rear_exp = ego_rear.unsqueeze(2)

                d_ff = torch.sum((ego_front_exp - other_front) ** 2, dim=-1)  # [B,1,N]
                d_fr = torch.sum((ego_front_exp - other_rear) ** 2, dim=-1)
                d_rf = torch.sum((ego_rear_exp - other_front) ** 2, dim=-1)
                d_rr = torch.sum((ego_rear_exp - other_rear) ** 2, dim=-1)

                min_dist_sq, _ = torch.min(torch.stack([d_ff, d_fr, d_rf, d_rr], dim=-1), dim=-1)  # [B,1,N]

                # --- 4. 过滤占位车辆 ---
                other_norm = torch.norm(torch.stack([other_x, other_y], dim=-1), dim=-1)  # [B,1,N]
                invalid_mask = other_norm < 1e-3
                min_dist_sq = torch.where(invalid_mask, torch.full_like(min_dist_sq, 1e9), min_dist_sq)
                if (min_dist_sq < 1e9).any():
                    logger.debug(f"检测到车辆接近！最近平方距离: {min_dist_sq.min().item():.2f}")

                # --- 5. 计算安全距离阈值 ---
                circle_radius = self.HALF_W * 0.65  # 每个圆的半径 (~0.9m)
                # 两个圆之间的最小中心距 = 2*半径 + 预设间隙
                safe_center_dist = 2.0 * circle_radius + self.other_car_min_distance  # 米
                safe_center_dist_sq = safe_center_dist ** 2

                # --- 6. 违反量 ---
                veh_violation_sq = torch.clamp(safe_center_dist_sq - min_dist_sq, min=0.0, max=10.0)
                phi_violation += (veh_violation_sq ** 2).sum(dim=[1, 2])

                # 原始违反量 (用于平滑惩罚)
                veh_violation_raw = torch.clamp(safe_center_dist - torch.sqrt(min_dist_sq), min=0.0)
                penalty_npc_raw = veh_violation_raw.sum(dim=[1, 2])  # [B]

            # 道路边缘安全距离约束 (road_left/right 已在自车坐标系)
            dist_left_sq = torch.sum((ego_xy.unsqueeze(2) - road_left) ** 2, dim=-1)
            dist_right_sq = torch.sum((ego_xy.unsqueeze(2) - road_right) ** 2, dim=-1)
            min_left_sq, _ = torch.min(dist_left_sq, dim=-1)
            min_right_sq, _ = torch.min(dist_right_sq, dim=-1)
            left_violation = torch.clamp(torch.maximum(safe_road_sq - min_left_sq, torch.zeros_like(min_left_sq)),
                                         max=10.0).squeeze(1)
            right_violation = torch.clamp(torch.maximum(safe_road_sq - min_right_sq, torch.zeros_like(min_right_sq)),
                                          max=10.0).squeeze(1)
            phi_violation += (left_violation ** 2 + right_violation ** 2)

            # 【防御】截断总惩罚项
            step_phi = torch.clamp(phi_violation, max=50.0)
            # ─── 移植自 loss.py 的平滑惩罚 ───
            # 获取当前时刻的转向角 delta_phy (已在前面计算)
            delta_phy = phy_action[:, 1]  # [B]

            # 条件：周车约束满足 (penalty_npc_raw < 0.05) 且转向角有小幅动作 (|delta| > 0.01 rad)
            condition = (penalty_npc_raw < 0.05) & (torch.abs(delta_phy) > 0.01)
            extra_steer_penalty = torch.where(condition, torch.abs(delta_phy) * 2.0, torch.zeros_like(delta_phy))
            # ──────────────────────────────
            step_l = torch.clamp(err_cost + control_cost + extra_steer_penalty, max=100.0)
            step_l_list.append(step_l)
            step_phi_list.append(step_phi)

            current_ego = next_ego
            current_other = next_other
            current_ref_error = next_ref_error

        step_l = torch.stack(step_l_list).transpose(0, 1)
        step_phi = torch.stack(step_phi_list).transpose(0, 1)
        states_traj = torch.stack(trajectory_states).transpose(0, 1)

        return step_l, step_phi, states_traj

    def select_action(self, obs: Any, deterministic: bool = False):
        """
        动作链路：归一化动作 → 物理量映射 → 安全护盾 (对齐论文 Eq.24-25)
        """
        with torch.no_grad():
            if isinstance(obs, list):
                obs_np = np.array(obs, dtype=np.float32).flatten()
            elif isinstance(obs, np.ndarray):
                obs_np = obs.flatten()
            else:
                obs_np = np.array(obs, dtype=np.float32).flatten()

                # 【防御】检查输入是否包含 nan/inf，防止污染网络
            if np.any(np.isnan(obs_np)) or np.any(np.isinf(obs_np)):
                logger.warning("输入观测包含 nan/inf，返回安全零动作")
                return np.array([0.0, 0.0], dtype=np.float32), np.zeros(1, dtype=np.float32)

                # 【加固】严格校验维度，防止静默错位导致策略崩溃
            if obs_np.shape[0] != self.TOTAL_STATE_DIM:
                raise ValueError(
                    f"观测维度异常: {obs_np.shape[0]} (期望{self.TOTAL_STATE_DIM})。"
                    f"请检查环境 idc_obs 是否严格遵循论文格式：[ego(6) + others*4 + ref_err(3)] 且为自车相对坐标。"
                )

            if deterministic:
                obs_tensor = torch.from_numpy(obs_np).to(self.device).float()
                norm_action = self.actor(obs_tensor.unsqueeze(0)).squeeze(0)
                norm_action = norm_action.cpu().numpy().flatten()
            else:
                obs_tensor = torch.from_numpy(obs_np).to(self.device).float()
                norm_action = self.actor(obs_tensor.unsqueeze(0)).squeeze(0)
                norm_action = norm_action.cpu().numpy().flatten()
                # 标准高斯噪声探索 (论文未指定硬编码，使用标准噪声更利于梯度收敛)
                noise = np.random.normal(0, [0.1, 0.05], size=norm_action.shape)
                norm_action = np.clip(norm_action + noise, -1.0, 1.0)

            a_phy = np.interp(norm_action[0], [-1, 1], [-3.0, 1.5])
            # 转向映射：正值→右转，负值→左转，直接乘以0.4
            norm_steer = norm_action[1]
            delta_phy = norm_steer * 0.4   # [-1,1] → [-0.4,0.4] rad

            # ---- 调试日志 ----
            logger.debug(
                f"[ACTION] norm_steer={norm_steer:.4f} -> delta_phy={delta_phy:.4f} rad "
                f"(正=右转? norm_steer>0 => 右转; norm_steer<0 => 左转)"
            )

            phy_action = np.array([a_phy, delta_phy], dtype=np.float32)

            return phy_action, np.zeros(1, dtype=np.float32)

    def update(self, ref_path_tensor: torch.Tensor = None,
               road_state_tensor: torch.Tensor = None):
        """
        严格对齐论文 Algorithm 2 (GEP) 训练逻辑
        """
        batch_data = self.buffer.sample_batch(self.batch_size)
        if len(batch_data) == 0 or ref_path_tensor is None:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "penalty": self.init_penalty,
                "gep_iteration": self.gep_iteration,
                "actor_updated": False
            }

        states_list = []
        road_list = []
        for item in batch_data:
            state, _, _, _, _, info = item
            state_np = np.array(state, dtype=np.float32).flatten()
            # 【防御】检查状态是否包含 nan/inf
            if np.any(np.isnan(state_np)) or np.any(np.isinf(state_np)):
                logger.warning("Buffer 状态包含 nan/inf，跳过该样本")
                continue
                # 【加固】严格校验维度
            if state_np.shape[0] != self.TOTAL_STATE_DIM:
                raise ValueError(
                    f"Buffer 状态维度异常: {state_np.shape[0]} (期望{self.TOTAL_STATE_DIM})。"
                    f"请检查环境 idc_obs 输出格式。"
                )
            states_list.append(state_np)

            road_np = info['road_state']
            if road_np is not None and (np.any(np.isnan(road_np)) or np.any(np.isinf(road_np))):
                logger.warning("Buffer 道路状态包含 nan/inf，跳过该样本")
                continue
            road_list.append(road_np)

        if len(states_list) == 0:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "penalty": self.init_penalty,
                "gep_iteration": self.gep_iteration,
                "actor_updated": False
            }

        state_tensor = torch.from_numpy(np.stack(states_list)).to(self.device).float()
        road_tensor = torch.from_numpy(np.stack(road_list)).to(self.device).float()

        # 1. Critic更新 (策略评估) - 严格对齐 Eq.7: 目标仅为成本项 J_actor，不含惩罚
        with torch.no_grad():
            step_l, step_phi, states_traj = self._forward_horizon(state_tensor, ref_path_tensor, road_tensor)
            # 有限时域累计成本 (无折扣 γ=1，对齐论文 IDC)
            # 【防御】截断目标值，防止梯度爆炸
            targets = torch.clamp(torch.flip(torch.cumsum(torch.flip(step_l, [1]), dim=1), [1]), max=1000.0)
            # ----- 优先经验回放：实时刷新采样批次的优先级 -----
            # 重新计算本批次的约束违反/性能指标（你已算出的 step_phi 可以复用）
            # step_phi: [B, horizon] 是每个样本每步的违规平方和
            violation_per_sample = step_phi.sum(dim=1)  # 每个样本总违规
            new_pri = violation_per_sample.cpu().numpy().astype(np.float64)
            # 确保最小值为正
            new_pri = np.maximum(new_pri, 1e-6)
            # 打印观察
            if new_pri.min() > 1e8:
                logger.debug(f"DEBUG: violation range [{new_pri.min():.4f}, {new_pri.max():.4f}], mean={new_pri.mean():.4f}")

        # 构造 (experience, new_priority) 列表
        experiences_and_priorities = []
        max_violation = violation_per_sample.max().item() + 1e-5
        for i, item in enumerate(batch_data):
            # 将违反程度映射到优先级 0.1~10
            priority = 0.1 + 9.9 * (violation_per_sample[i].item() / max_violation)
            experiences_and_priorities.append((item, priority))

        all_states = torch.cat([state_tensor.unsqueeze(1), states_traj], dim=1)
        critic_inputs = all_states[:, :-1].reshape(-1, self.TOTAL_STATE_DIM)
        critic_targets = targets.reshape(-1, 1)
        pred = self.critic(critic_inputs)
        critic_loss = F.mse_loss(pred, critic_targets)

        self.critic_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Actor更新 (策略改进) - 严格对齐 Eq.9: 成本项 + 惩罚因子×约束违反项
        step_l_actor, step_phi_actor, _ = self._forward_horizon(state_tensor, ref_path_tensor, road_tensor)
        actor_loss = step_l_actor.mean() + self.init_penalty * step_phi_actor.mean()

        self.actor_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        actor_loss.backward()
        self.actor_optimizer.step()
        actor_updated = True
        self.gep_iteration += 1  # 记录策略改进次数

        # 3. GEP惩罚因子放大 (每 m 次策略改进后执行，严格对齐 Algorithm 2)
        if self.gep_iteration % self.amplifier_m == 0:
            # 只要有违反，立即放大，不再要求门槛
            if violation_per_sample.sum() > 1e-6:
                old_penalty = self.init_penalty
                self.init_penalty = min(self.init_penalty * self.amplifier_c, self.max_penalty)
                logger.debug(f"[GEP] 立即放大 ρ: {old_penalty:.2f} → {self.init_penalty:.2f}")

        self.predict_traj = states_traj.cpu().detach().numpy()

        # 更新缓冲区
        # 将每个样本的违规量作为新优先级
        new_pri = violation_per_sample.cpu().numpy() + 1e-5  # 确保>0
        self.buffer.update_last_batch_priorities(new_pri)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "penalty": self.init_penalty,
            "gep_iteration": self.gep_iteration,
            "actor_updated": actor_updated
        }

    def predict_other_next_batch(self, other_states: torch.Tensor, dt: float) -> torch.Tensor:
        """
        周车状态: [B, 1, N, 4]  → 4 = [x, y, phi(rad), v_lon]
        """
        if other_states.dim() != 4 or other_states.shape[3] != 4:
            raise ValueError(f"周车状态维度必须为 [B,1,N,4]，当前={other_states.shape}")
        if other_states.shape[2] == 0:
            return torch.zeros_like(other_states)

        x, y, phi, v = (other_states[..., 0], other_states[..., 1],
                        other_states[..., 2], other_states[..., 3])
        x_next = x + dt * v * torch.cos(phi)
        y_next = y + dt * v * torch.sin(phi)
        # phi 和 v 保持不变（恒速恒向）
        return torch.stack([x_next, y_next, phi, v], dim=-1)

    def _advance_along_path(
            self,
            path_xy: torch.Tensor,  # [N,2] or [B,N,2]
            start_idx: torch.Tensor,  # [B] or [B,1]
            dist_forward: torch.Tensor  # [B] or [B,1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        安全沿路径前进，返回前视点坐标、航向和索引（均在[0,N-1]内）。
        """
        # 统一形状：path -> [B,N,2]
        if path_xy.dim() == 2:
            path_xy = path_xy.unsqueeze(0)  # [1,N,2]
        B, N, _ = path_xy.shape
        # 如果传入的 B 与 start_idx 的 B 不一致，则扩展 path
        if B == 1 and start_idx.shape[0] > 1:
            path_xy = path_xy.repeat(start_idx.shape[0], 1, 1)
            B = start_idx.shape[0]

        start_idx = start_idx.view(B).clamp(0, N - 1).long()  # [B]
        dist_forward = dist_forward.view(B).clamp(min=1e-6)  # [B]

        # 路径段向量和长度
        segs = path_xy[:, 1:] - path_xy[:, :-1]  # [B, N-1, 2]
        seg_len = torch.norm(segs, dim=-1)  # [B, N-1]
        seg_len = seg_len.clamp(min=1e-6)

        # 从路径起点开始的累积距离 (每段的终点)
        cum_from_start = torch.cat([
            torch.zeros(B, 1, device=path_xy.device),
            torch.cumsum(seg_len, dim=1)
        ], dim=1)  # [B, N]  索引0..N-1，cum_from_start[i] 表示点 i 的距离起点距离

        # 前视目标距离 = 起点距离 + 前视量
        start_dist = cum_from_start[torch.arange(B), start_idx]  # [B]
        target_dist = start_dist + dist_forward  # [B]

        # 若目标超出路径总长，则直接取最后一个点
        total_len = cum_from_start[:, -1]  # [B]
        overrun = target_dist >= total_len

        # 对于未超出的，找到第一个 cum_from_start >= target_dist 的点索引
        # 使用搜索排序：在 cum_from_start 上逐 batch 查找
        # 由于 cum_from_start 单调递增，可以用 torch.searchsorted
        look_idx = torch.searchsorted(cum_from_start, target_dist.unsqueeze(1)).squeeze(1)  # [B]
        # 防止超出 N-1
        look_idx = look_idx.clamp(0, N - 1)
        # 对于超出的，强制为 N-1
        look_idx = torch.where(overrun, torch.full_like(look_idx, N - 1), look_idx)

        # 获取目标点坐标
        look_xy = path_xy[torch.arange(B), look_idx]  # [B,2]

        # 计算目标点航向：使用 look_idx 与 look_idx+1 (若为 N-1 则用 N-2 和 N-1)
        next_idx = torch.where(look_idx < N - 1, look_idx + 1, look_idx)
        prev_idx = torch.where(look_idx > 0, look_idx - 1, look_idx)
        # 如果 look_idx == N-1 且不是 overrun，则用前一段的航向；否则直接用下一点
        # 为简化且鲁棒，统一使用：点 look_idx 到 next_idx 的方向（如果 next_idx == look_idx 则回退到 prev_idx 方向）
        dxy = path_xy[torch.arange(B), next_idx] - path_xy[torch.arange(B), look_idx]
        # 如果两点重合（比如已在终点），则使用 prev_idx 方向
        dxy_len = torch.norm(dxy, dim=-1, keepdim=True)
        use_prev = (dxy_len < 1e-6).squeeze(-1)
        dxy = torch.where(
            use_prev.unsqueeze(-1),
            path_xy[torch.arange(B), look_idx] - path_xy[torch.arange(B), prev_idx],
            dxy
        )
        dxy_len = torch.norm(dxy, dim=-1, keepdim=True).clamp(min=1e-6)
        dxy = dxy / dxy_len
        look_phi = torch.atan2(dxy[..., 1], dxy[..., 0])  # [B]

        return look_xy, look_phi, look_idx

    def unpack_tensor(self, data: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if data.dim() != 3 or data.shape[2] != self.TOTAL_STATE_DIM:
            raise ValueError(f"输入张量必须为 [B,N,{self.TOTAL_STATE_DIM}]，当前={data.shape}")
        B, N = data.shape[0], data.shape[1]
        ego_state = data[:, :, 0:self.DIM_EGO]

        # 【修复】处理 others=0 时的维度对齐问题
        if self.DIM_OTHER > 0:
            other_raw = data[:, :, self.DIM_EGO:self.DIM_EGO + self.DIM_OTHER]
            other_states = other_raw.view(B, N, self.env.env_cfg['idc']['others'], 4)
        else:
            other_states = torch.empty(B, N, 0, 4, device=data.device)

        ref_error = data[:, :, self.DIM_EGO + self.DIM_OTHER:self.DIM_EGO + self.DIM_OTHER + self.DIM_REF_ERROR]
        return ego_state, other_states, ref_error

    def save(self, save_info: Dict[str, Any]) -> None:
        model = {'actor': self.actor, 'critic': self.critic}
        optimizer = {'actor_optim': self.actor_optimizer, 'critic_optim': self.critic_optimizer}
        extra_info = {
            'config': save_info['rl_config'],
            'global_step': self.global_step,
            'history': save_info['history_loss'],
            'globe_eps': self.globe_eps + self.base_config['save_freq'],
            'state_dim': self.TOTAL_STATE_DIM,
            'punish_factor': self.init_penalty,
            'gep_iteration': self.gep_iteration,
            'buffer_data': save_info['buffer_data']
        }
        metrics = {'episode': extra_info['globe_eps']}
        save_checkpoint(model=model, model_name='idc-v1.0', optimizer=optimizer,
                        extra_info=extra_info, metrics=metrics, env_name=save_info['map'])
        self.globe_eps = extra_info['globe_eps']
        self.global_step = extra_info['global_step']
        self.history_loss = extra_info['history']
        self.init_penalty = extra_info['punish_factor']

    def load(self, path: str) -> Dict[str, Any]:
        checkpoint = load_checkpoint(
            model={'actor': self.actor, 'critic': self.critic},
            filepath=path,
            optimizer={'actor_optim': self.actor_optimizer, 'critic_optim': self.critic_optimizer},
            device=self.device
        )
        loaded_dim = checkpoint.get('state_dim', self.TOTAL_STATE_DIM)
        if loaded_dim != self.TOTAL_STATE_DIM:
            logger.warning(f"加载模型维度{loaded_dim}与当前{self.TOTAL_STATE_DIM}不一致")
        self.globe_eps = checkpoint['globe_eps']
        self.history_loss = checkpoint['history']
        self.global_step = checkpoint['global_step']
        self.init_penalty = checkpoint['punish_factor']
        self.gep_iteration = checkpoint['gep_iteration']
        self.buffer.load_buffer_data(checkpoint['buffer_data'])
        return checkpoint

    def eval(self, num_episodes: int = 10) -> Tuple[float, float]:
        total_rewards = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                obs_idc = obs.get('idc_obs', obs)
                action, _ = self.select_action(obs_idc, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            total_rewards.append(episode_reward)
        mean_reward = float(np.mean(total_rewards))
        std_reward = float(np.std(total_rewards))
        logger.info(f"评估完成：{num_episodes}轮，平均奖励={mean_reward:.2f}，标准差={std_reward:.2f}")
        return mean_reward, std_reward
