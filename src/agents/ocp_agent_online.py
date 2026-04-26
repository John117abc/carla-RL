# src/agents/ocp_agent_online.py
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
from src.utils import get_logger

logger = get_logger('ocp_agent_online')


class OcpAgentOnline(BaseAgent):
    """
    严格对齐论文《Integrated Decision and Control》GEP 算法的在线模式实现
    采用单步推演即时更新策略，不依赖大规模经验回放缓冲区
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

        # 核心参数初始化
        self.dt = self.ocp_config['dt']
        self.horizon = self.ocp_config['horizon']
        self.batch_size = self.ocp_config['batch_size']

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
        self.q_lat = 0.04  # 横向误差权重
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

        # 训练状态
        self.global_step = 0
        self.gep_iteration = 0
        self.history_loss = []
        self.globe_eps = 0

        # 参考速度固定为5m/s（18km/h）
        self.ref_vlon = self.env.ego_ref_speed

        # 预测轨迹（用于可视化）
        self.predict_traj = None

        # 校验配置一致性
        if self.DIM_OTHER < 0:
            raise ValueError("env_cfg['ocp']['others'] 必须为非负整数，当前值导致 DIM_OTHER < 0")

    def _calc_ref_error_from_state(self, ego_state: torch.Tensor, ref_path: torch.Tensor) -> torch.Tensor:
        """
        严格对齐论文 Section IV-B1 的参考误差计算（统一使用自车坐标系）
        """
        B = ego_state.shape[0]
        if ref_path.shape[0] == 1 and B > 1:
            ref_path = ref_path.repeat(B, 1, 1)

        ego_xy = ego_state[..., :2]  # [B,1,2]
        ego_phi = ego_state[..., 4]  # [B,1]
        ego_vlon = ego_state[..., 2]  # [B,1]

        # 找最近参考点
        dist = torch.norm(ego_xy.unsqueeze(2) - ref_path.unsqueeze(1), dim=-1)  # [B,1,N]
        min_dist, closest_idx = torch.min(dist, dim=-1)  # [B,1]
        ref_idx = torch.clamp(closest_idx + self.env.carla_cfg['world']['ref_offset'], max=ref_path.shape[1] - 1)

        ref_xy = torch.gather(ref_path, 1, ref_idx.unsqueeze(-1).repeat(1, 1, 2))  # [B,1,2]
        next_ref_idx = torch.clamp(ref_idx + 1, max=ref_path.shape[1] - 1)
        next_ref_xy = torch.gather(ref_path, 1, next_ref_idx.unsqueeze(-1).repeat(1, 1, 2))
        delta_xy = next_ref_xy - ref_xy
        ref_phi = torch.atan2(delta_xy[..., 1], delta_xy[..., 0])  # [B,1]

        # 计算带符号横向误差δ_p
        dx = ref_xy[..., 0] - ego_xy[..., 0]
        dy = ref_xy[..., 1] - ego_xy[..., 1]
        cross = dx * torch.sin(ref_phi) - dy * torch.cos(ref_phi)
        delta_p = min_dist * torch.sign(cross)

        # 航向误差δ_φ（归一化）
        delta_phi = ego_phi - ref_phi
        delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))

        # 速度误差δ_v
        delta_v = ego_vlon - self.ref_vlon

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

        # 道路状态解析
        if road_state is not None:
            road_left = road_state[..., :int(self.DIM_ROAD / 2)].contiguous().view(B, 1, int(self.DIM_ROAD / 4), 2)
            road_right = road_state[..., int(self.DIM_ROAD / 2):].contiguous().view(B, 1, int(self.DIM_ROAD / 4), 2)
        else:
            road_left = torch.zeros(B, 1, 20, 2, device=self.device)
            road_right = torch.zeros(B, 1, 20, 2, device=self.device)

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
            err_cost = self.q_lat * (lat_err_t ** 2) + self.q_head * (head_err_t ** 2) + self.q_speed * (speed_err_t ** 2)

            r_weights = torch.tensor([0.005, 0.1], device=self.device).float()
            control_cost = torch.sum((phy_action ** 2) * r_weights, dim=1)
            step_l = err_cost + control_cost

            # 2. 补全约束违反量 step_phi
            ego_xy = next_ego[..., :2]  # [B, 1, 2]
            phi_violation = torch.zeros(B, device=self.device)

            if self.DIM_OTHER > 0:
                other_xy = next_other[..., :2]  # [B, 1, 8, 2]
                dist_veh_sq = torch.sum((ego_xy.unsqueeze(2) - other_xy) ** 2, dim=-1)
                veh_violation = torch.maximum(safe_veh_sq - dist_veh_sq, torch.zeros_like(dist_veh_sq))
                phi_violation += veh_violation.sum(dim=[1, 2])

            dist_left_sq = torch.sum((ego_xy.unsqueeze(2) - road_left) ** 2, dim=-1)
            dist_right_sq = torch.sum((ego_xy.unsqueeze(2) - road_right) ** 2, dim=-1)
            min_left_sq, _ = torch.min(dist_left_sq, dim=-1)
            min_right_sq, _ = torch.min(dist_right_sq, dim=-1)
            left_violation = torch.maximum(safe_road_sq - min_left_sq, torch.zeros_like(min_left_sq)).squeeze(1)
            right_violation = torch.maximum(safe_road_sq - min_right_sq, torch.zeros_like(min_right_sq)).squeeze(1)
            phi_violation += left_violation + right_violation

            step_phi = phi_violation
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
        动作链路：归一化动作 → 物理量映射 → 标准高斯探索
        """
        with torch.no_grad():
            if isinstance(obs, list):
                obs_np = np.array(obs, dtype=np.float32).flatten()
            elif isinstance(obs, np.ndarray):
                obs_np = obs.flatten()
            else:
                obs_np = np.array(obs, dtype=np.float32).flatten()

            # 【严格校验】防止静默错位导致策略崩溃
            if obs_np.shape[0] != self.TOTAL_STATE_DIM:
                raise ValueError(
                    f"观测维度异常: {obs_np.shape[0]} (期望{self.TOTAL_STATE_DIM})。"
                    f"请检查环境 ocp_obs 是否严格遵循论文格式：[ego(6) + others*4 + ref_err(3)] 且为自车相对坐标。"
                )

            obs_tensor = torch.from_numpy(obs_np).to(self.device).float()
            norm_action = self.actor(obs_tensor.unsqueeze(0)).squeeze(0)
            norm_action = norm_action.cpu().numpy().flatten()

            if not deterministic:
                # 标准高斯噪声探索，利于梯度收敛
                noise = np.random.normal(0, [0.1, 0.05], size=norm_action.shape)
                norm_action = np.clip(norm_action + noise, -1.0, 1.0)

            a_phy = np.interp(norm_action[0], [-1, 1], [-3.0, 1.5])
            delta_phy = np.interp(norm_action[1], [-1, 1], [-0.4, 0.4])
            phy_action = np.array([a_phy, delta_phy], dtype=np.float32)

            return phy_action, np.zeros(1, dtype=np.float32)

    def update(self, state: np.ndarray, road_state: np.ndarray, ref_path_tensor: torch.Tensor):
        """
        严格对齐论文 Algorithm 2 (GEP) 在线单步更新逻辑
        """
        # 维度校验
        if state.shape[0] != self.TOTAL_STATE_DIM:
            raise ValueError(f"输入状态维度异常: {state.shape[0]} (期望{self.TOTAL_STATE_DIM})")
        
        state_tensor = torch.from_numpy(state).to(self.device).float().unsqueeze(0)  # [1, DIM]
        road_tensor = torch.from_numpy(road_state).to(self.device).float().unsqueeze(0) if road_state is not None else None

        # 1. Critic更新 (策略评估) - 严格对齐 Eq.7: 目标仅为成本项 J_actor，不含惩罚
        with torch.no_grad():
            step_l, step_phi, states_traj = self._forward_horizon(state_tensor, ref_path_tensor, road_tensor)
            # 有限时域累计成本 (无折扣 γ=1，对齐论文 OCP)
            targets = torch.flip(torch.cumsum(torch.flip(step_l, [1]), dim=1), [1])

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
        self.gep_iteration += 1

        # 3. GEP惩罚因子放大 (每 m 次策略改进后执行，严格对齐 Algorithm 2)
        if self.gep_iteration % self.amplifier_m == 0:
            old_penalty = self.init_penalty
            self.init_penalty = min(self.init_penalty * self.amplifier_c, self.max_penalty)
            logger.info(f"[GEP] 惩罚因子更新: {old_penalty:.4f} → {self.init_penalty:.4f}")

        self.predict_traj = states_traj.cpu().detach().numpy()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "penalty": self.init_penalty,
            "gep_iteration": self.gep_iteration,
            "actor_updated": actor_updated
        }

    def predict_other_next_batch(self, other_states: torch.Tensor, dt: float) -> torch.Tensor:
        """周车匀速预测模型，维度严格对齐"""
        if other_states.dim() != 4 or other_states.shape[3] != 4:
            raise ValueError(f"周车状态维度必须为 [B,1,N_others,4]，当前={other_states.shape}")
        if other_states.shape[2] == 0:
            return torch.zeros_like(other_states)
        
        x, y, vx, vy = other_states[..., 0], other_states[..., 1], other_states[..., 2], other_states[..., 3]
        x_next = x + dt * vx
        y_next = y + dt * vy
        return torch.stack([x_next, y_next, vx, vy], dim=-1)

    def unpack_tensor(self, data: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if data.dim() != 3 or data.shape[2] != self.TOTAL_STATE_DIM:
            raise ValueError(f"输入张量必须为 [B,N,{self.TOTAL_STATE_DIM}]，当前={data.shape}")
        B, N = data.shape[0], data.shape[1]
        ego_state = data[:, :, 0:self.DIM_EGO]
        other_raw = data[:, :, self.DIM_EGO:self.DIM_EGO + self.DIM_OTHER]
        other_states = other_raw.view(B, N, self.env.env_cfg['ocp']['others'], 4)
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
            'buffer_data': save_info.get('buffer_data', {})
        }
        metrics = {'episode': extra_info['globe_eps']}
        save_checkpoint(model=model, model_name='ocp-online-v1.0', optimizer=optimizer,
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
        return checkpoint

    def eval(self, num_episodes: int = 10) -> Tuple[float, float]:
        total_rewards = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                obs_ocp = obs.get('ocp_obs', obs)
                action, _ = self.select_action(obs_ocp, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            total_rewards.append(episode_reward)
        mean_reward = float(np.mean(total_rewards))
        std_reward = float(np.std(total_rewards))
        logger.info(f"评估完成：{num_episodes}轮，平均奖励={mean_reward:.2f}，标准差={std_reward:.2f}")
        return mean_reward, std_reward
