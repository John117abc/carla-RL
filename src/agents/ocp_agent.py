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
    严格对齐论文《Integrated Decision and Control》Algorithm 2 (GEP) 实现
    """

    def __init__(
            self,
            rl_config: Dict[str, Any],
            env: gym.Env,
            device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(env, device)
        assert isinstance(self.action_space, gym.spaces.Box), "OCP智能体需要连续的动作空间。"

        # 读取配置
        rl_algorithm = "OCP"
        self.base_config = rl_config['rl']
        self.ocp_config = rl_config['rl'][rl_algorithm]

        # 严格对齐论文的状态维度定义
        self.DIM_EGO = 6  # 自车 [x,y,v_lon,v_lat,phi,omega]
        self.DIM_OTHER = self.env.env_cfg['ocp']['others'] * 4  # 8车×4维
        self.DIM_REF_ERROR = 3  # 跟踪误差 [δ_p, δ_φ, δ_v]
        self.TOTAL_STATE_DIM = self.DIM_EGO + self.DIM_OTHER + self.DIM_REF_ERROR

        # 道路维度
        self.DIM_ROAD = self.env.env_cfg['ocp']['num_points'] * 4  # 道路80维

        self.road_state_buffer = None  # 用于存储当前的道路边缘信息

        # 核心参数初始化
        self.dt = self.ocp_config['dt']
        self.horizon = self.ocp_config['horizon']
        self.batch_size = self.ocp_config['batch_size']

        # 【修复】状态归一化标准差，统一量纲防止梯度震荡
        self.state_std = {
            'ego_pos': 10.0, 'ego_vel': 5.0, 'ego_ang': 1.0,
            'other_pos': 10.0, 'other_vel': 5.0,
            'lat': 1.0, 'head': 1.0, 'speed': 5.0,
            'road': 10.0
        }

        # 网络初始化
        self.actor = ActorNet(
            state_dim=self.TOTAL_STATE_DIM,
            hidden_dim=self.ocp_config['hidden_dim']
        ).to(self.device)
        self.critic = CriticNet(
            state_dim=self.TOTAL_STATE_DIM,
            hidden_dim=self.ocp_config['hidden_dim']
        ).to(self.device)
        self.dynamics_model = BicycleModel(dt=self.dt, L=2.9)

        # 优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.ocp_config['lr_actor'],
            betas=(0.9, 0.999)
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.ocp_config['lr_critic'],
            betas=(0.9, 0.999)
        )

        # 论文核心权重（严格对齐原文 Table III & Eq. 1）
        # 【优化】提升横向误差权重，确保车道保持优先级
        self.q_lat = 0.1  # 横向误差权重 (原0.04过小)
        self.q_head = 0.1  # 航向误差权重
        self.q_speed = 0.01  # 速度误差权重
        self.R_matrix = np.diag([0.005, 0.1])  # 控制权重 [加速度, 转向角]

        # GEP算法超参数（严格对齐论文收敛逻辑）
        self.init_penalty = self.ocp_config['init_penalty']
        self.max_penalty = self.ocp_config['max_penalty']
        self.amplifier_c = self.ocp_config['amplifier_c']
        self.amplifier_m = self.ocp_config['amplifier_m']
        self.other_car_min_distance = self.ocp_config['other_car_min_distance']
        self.road_min_distance = self.ocp_config['road_min_distance']
        self.gamma = self.ocp_config['gamma']

        # 缓冲区
        self.buffer = StochasticBuffer(
            min_start_train=self.ocp_config['min_start_train'],
            total_capacity=self.ocp_config['total_capacity']
        )

        # 训练状态
        self.global_step = 0
        self.gep_iteration = 0
        self.history_loss = []
        self.globe_eps = 0

        # 参考速度固定为5m/s（18km/h）
        self.ref_vlon = self.env.ego_ref_speed

        # 预测轨迹
        self.predict_traj = None

    def _calc_ref_error_from_state(self, ego_state: torch.Tensor, ref_path_tensor: torch.Tensor) -> torch.Tensor:
        """
        严格对齐论文 Section IV-B1 的参考误差计算（统一使用自车坐标系）
        :param ego_state: [B, 1, 6] 自车状态 (自车坐标系, x=0, y=0, phi=0)
        :param ref_path_tensor: [1, N, 2] 参考路径xy (自车坐标系)
        :return: [B, 1, 3] 跟踪误差 [δ_p, δ_φ, δ_v]
        """
        B = ego_state.shape[0]
        if ref_path_tensor.shape[0] == 1 and B > 1:
            ref_path_tensor = ref_path_tensor.repeat(B, 1, 1)

        # 自车坐标系下，自车位置恒为原点，航向恒为0
        ego_xy = torch.zeros(B, 1, 2, device=self.device)
        ego_phi = torch.zeros(B, 1, device=self.device)
        ego_vlon = ego_state[..., 2]  # [B, 1]

        # 找最近参考点
        dist = torch.norm(ego_xy.unsqueeze(2) - ref_path_tensor.unsqueeze(1), dim=-1)  # [B, 1, N]
        min_dist, closest_idx = torch.min(dist, dim=-1)  # [B, 1]
        ref_idx = torch.clamp(closest_idx, max=ref_path_tensor.shape[1] - 1)

        ref_xy = torch.gather(ref_path_tensor, 1, ref_idx.unsqueeze(-1).repeat(1, 1, 2))  # [B, 1, 2]
        next_ref_idx = torch.clamp(ref_idx + 1, max=ref_path_tensor.shape[1] - 1)
        next_ref_xy = torch.gather(ref_path_tensor, 1, next_ref_idx.unsqueeze(-1).repeat(1, 1, 2))
        delta_xy = next_ref_xy - ref_xy
        ref_phi = torch.atan2(delta_xy[..., 1], delta_xy[..., 0])

        # 计算带符号横向误差δ_p (论文定义：左侧为正)
        # 在自车坐标系中，ref_xy 即为相对位移
        dx = ref_xy[..., 0]
        dy = ref_xy[..., 1]
        cross = dx * torch.sin(ref_phi) - dy * torch.cos(ref_phi)
        delta_p = min_dist * torch.sign(cross)

        # 航向误差δ_φ（归一化到[-π, π]）
        # 【修复】在自车坐标系下，自车航向为0，参考路径航向即为偏航角
        delta_phi = -ref_phi
        delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))

        # 速度误差δ_v
        delta_v = ego_vlon - self.ref_vlon

        # 【新增】误差归一化，统一量纲提升网络收敛稳定性
        delta_p = delta_p / self.state_std['lat']
        delta_phi = delta_phi / self.state_std['head']
        delta_v = delta_v / self.state_std['speed']

        return torch.stack([delta_p, delta_phi, delta_v], dim=-1)

    def _forward_horizon(self,
                         state_tensor: torch.Tensor,
                         ref_path_tensor: torch.Tensor,
                         road_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        严格对齐论文 OCP 公式与 GEP 算法，统一使用自车坐标系
        """
        B = state_tensor.shape[0]
        ego_state, other_states, ref_error = self.unpack_tensor(state_tensor.unsqueeze(1))
        current_ego = ego_state.clone()
        current_other = other_states.clone()
        current_ref_error = self._calc_ref_error_from_state(current_ego, ref_path_tensor)

        # 【修复】道路状态解析：80维 -> 20左点 + 20右点 (自车坐标系)
        if road_state is not None:
            road_left = road_state[..., :int(self.DIM_ROAD/2)].contiguous().view(B, 1, 20, 2)
            road_right = road_state[..., int(self.DIM_ROAD/2):].contiguous().view(B, 1, 20, 2)
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
            # 【修复】输入网络前进行状态归一化，防止量纲差异导致梯度消失
            ego_norm = current_ego / self.state_std['ego_pos']
            other_norm = current_other / self.state_std['other_pos']
            
            current_state = torch.cat([
                ego_norm.view(-1, self.DIM_EGO),
                other_norm.view(-1, self.DIM_OTHER),
                current_ref_error.view(-1, self.DIM_REF_ERROR)
            ], dim=1)

            norm_action = self.actor(current_state)  # [B, 2]
            a_phy = norm_action[:, 0:1] * 2.25 - 0.75  # [-1,1] → [-3, 1.5] m/s²
            delta_phy = norm_action[:, 1:2] * 0.4  # [-1,1] → [-0.4, 0.4] rad
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

            # 1. 计算OCP成本项 step_l
            lat_err_t = next_ref_error[..., 0].squeeze(1)
            head_err_t = next_ref_error[..., 1].squeeze(1)
            speed_err_t = next_ref_error[..., 2].squeeze(1)
            # 误差已归一化，权重可保持平衡
            err_cost = self.q_lat * (lat_err_t ** 2) + self.q_head * (head_err_t ** 2) + self.q_speed * (speed_err_t ** 2)

            r_weights = torch.tensor([0.005, 0.1], device=self.device).float()
            control_cost = torch.sum((phy_action ** 2) * r_weights, dim=1)
            step_l = err_cost + control_cost

            # 2. 补全约束违反量 step_phi (严格对齐论文 Eq.9: 惩罚项需平方)
            # 在自车坐标系中，ego_xy 恒为 [0,0]
            ego_xy = torch.zeros(B, 1, 2, device=self.device)
            phi_violation = torch.zeros(B, device=self.device)

            # 周车安全距离约束 (other_states 已在自车坐标系)
            other_xy = next_other[..., :2]  # [B, 1, 8, 2]
            dist_veh_sq = torch.sum((ego_xy.unsqueeze(2) - other_xy) ** 2, dim=-1)
            veh_violation = torch.maximum(safe_veh_sq - dist_veh_sq, torch.zeros_like(dist_veh_sq))
            phi_violation += (veh_violation ** 2).sum(dim=[1, 2])  # 【修复】必须平方

            # 道路边缘安全距离约束 (road_left/right 已在自车坐标系)
            dist_left_sq = torch.sum((ego_xy.unsqueeze(2) - road_left) ** 2, dim=-1)
            dist_right_sq = torch.sum((ego_xy.unsqueeze(2) - road_right) ** 2, dim=-1)
            min_left_sq, _ = torch.min(dist_left_sq, dim=-1)
            min_right_sq, _ = torch.min(dist_right_sq, dim=-1)
            left_violation = torch.maximum(safe_road_sq - min_left_sq, torch.zeros_like(min_left_sq))
            right_violation = torch.maximum(safe_road_sq - min_right_sq, torch.zeros_like(min_right_sq))
            phi_violation += (left_violation ** 2 + right_violation ** 2)  # 【修复】必须平方

            step_l_list.append(step_l)
            step_phi_list.append(phi_violation)

            current_ego = next_ego
            current_other = next_other
            current_ref_error = next_ref_error

        step_l_tensor = torch.stack(step_l_list, dim=1)  # [B, H]
        step_phi_tensor = torch.stack(step_phi_list, dim=1)  # [B, H]
        trajectory_states_tensor = torch.stack(trajectory_states, dim=1)  # [B, H, D]

        return step_l_tensor, step_phi_tensor, trajectory_states_tensor

    def select_action(self, obs: Dict[str, Any], deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        严格对齐论文 Algorithm 2 的 Action Selection 步骤
        """
        obs_np = obs['ocp_obs']
        ref_path_np = obs['ref_path_locations']
        road_state_np = obs['road_state']

        # 【修复】推理时同样需要归一化
        ego_state = torch.tensor(obs_np[..., :self.DIM_EGO], dtype=torch.float32, device=self.device).unsqueeze(0)
        other_states = torch.tensor(obs_np[..., self.DIM_EGO:self.DIM_EGO + self.DIM_OTHER], dtype=torch.float32, device=self.device).unsqueeze(0)
        ref_error = torch.tensor(obs_np[..., self.DIM_EGO + self.DIM_OTHER:], dtype=torch.float32, device=self.device).unsqueeze(0)

        ego_norm = ego_state / self.state_std['ego_pos']
        other_norm = other_states / self.state_std['other_pos']
        current_state = torch.cat([
            ego_norm.view(-1, self.DIM_EGO),
            other_norm.view(-1, self.DIM_OTHER),
            ref_error.view(-1, self.DIM_REF_ERROR)
        ], dim=1)

        with torch.no_grad():
            norm_action = self.actor(current_state)
            if not deterministic:
                norm_action += torch.randn_like(norm_action) * self.ocp_config['action_noise']

        a_phy = norm_action[0, 0].item() * 2.25 - 0.75
        delta_phy = norm_action[0, 1].item() * 0.4

        action = np.array([a_phy, delta_phy], dtype=np.float32)
        info = {'ref_path': ref_path_np, 'road_state': road_state_np}
        return action, info

    def update(self, ref_path_tensor: torch.Tensor = None, road_state_tensor: torch.Tensor = None) -> Dict[str, float]:
        """
        严格对齐论文 Algorithm 2 的 GEP 更新步骤
        """
        if self.globe_eps < self.ocp_config['min_start_train']:
            return {}

        self.globe_eps += 1
        self.gep_iteration += 1

        # 1. 采样
        state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor, info = self.buffer.sample(self.batch_size)
        ref_path_tensor = info['ref_path']
        road_state_tensor = info['road_state']

        # 【修复】训练时统一归一化
        ego_state = state_tensor[..., :self.DIM_EGO]
        other_states = state_tensor[..., self.DIM_EGO:self.DIM_EGO + self.DIM_OTHER]
        ego_norm = ego_state / self.state_std['ego_pos']
        other_norm = other_states / self.state_std['other_pos']
        state_tensor_norm = torch.cat([
            ego_norm.view(-1, self.DIM_EGO),
            other_norm.view(-1, self.DIM_OTHER),
            state_tensor[..., self.DIM_EGO + self.DIM_OTHER:].view(-1, self.DIM_REF_ERROR)
        ], dim=1)

        road_state_tensor = road_state_tensor / self.state_std['road']

        # 2. OCP Rollout
        step_l, step_phi, trajectory_states = self._forward_horizon(state_tensor_norm, ref_path_tensor, road_state_tensor)

        # 3. GEP 惩罚更新 (严格对齐论文 Eq. 10 & 11)
        if self.gep_iteration % self.amplifier_m == 0:
            self.init_penalty = min(self.init_penalty * self.amplifier_c, self.max_penalty)

        # 4. Critic 更新 (严格对齐论文 Eq. 12)
        targets = torch.flip(torch.cumsum(torch.flip(step_l + self.init_penalty * step_phi, [1]), dim=1), [1])
        targets = targets.detach()
        critic_pred = self.critic(state_tensor_norm)
        critic_loss = F.mse_loss(critic_pred, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 5. Actor 更新 (严格对齐论文 Eq. 13)
        actor_loss = step_l.mean() + self.init_penalty * step_phi.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 6. 记录与保存
        self.history_loss.append(actor_loss.item())
        if self.globe_eps % self.ocp_config['save_interval'] == 0:
            save_checkpoint(self.actor, self.critic, self.actor_optimizer, self.critic_optimizer, self.globe_eps, self.ocp_config['save_dir'])

        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(), 'penalty': self.init_penalty}

    def predict_other_next_batch(self, other_states: torch.Tensor, dt: float) -> torch.Tensor:
        """
        严格对齐论文 Section IV-B2 的周车运动学预测
        """
        B = other_states.shape[0]
        next_other = other_states.clone()
        for i in range(other_states.shape[2]):
            other_xy = other_states[..., i, :2]
            other_vlon = other_states[..., i, 2]
            other_vlat = other_states[..., i, 3]
            next_other[..., i, :2] = other_xy + torch.stack([other_vlon * dt, other_vlat * dt], dim=-1)
        return next_other

    def unpack_tensor(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        严格对齐论文 Section IV-B 的状态解包
        """
        ego_state = state_tensor[..., :self.DIM_EGO]
        other_states = state_tensor[..., self.DIM_EGO:self.DIM_EGO + self.DIM_OTHER]
        ref_error = state_tensor[..., self.DIM_EGO + self.DIM_OTHER:]
        return ego_state, other_states, ref_error
