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


class OcpAgentOnline(BaseAgent):
    """
    严格对齐论文《Integrated Decision and Control》算法1实现
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

        # 论文核心权重（严格对齐原文）
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

        # 【关键修复】参考速度固定为5m/s（18km/h），先让车能动，再提速
        self.ref_vlon = self.env.ego_ref_speed

    def _calc_ref_error_from_state(self, ego_state: torch.Tensor, ref_path: torch.Tensor) -> torch.Tensor:
        """
        严格对齐ocp_setup.py的误差计算，修复维度匹配问题
        :param ego_state: [B, 1, 6] 自车状态
        :param ref_path: [1, N, 2] 参考路径xy
        :return: [B, 1, 3] 跟踪误差 [δ_p, δ_φ, δ_v]
        """
        print(f"[调试] 自车初始xy (自车坐标系): {ego_state[0, 0, :2]}")  # 应该接近 [0, 0]
        print(f"[调试] 参考路径前3个点: {ref_path[0, :3]}")  # 应该在自车前方
        B = ego_state.shape[0]
        # 参考路径广播到batch维度
        if ref_path.shape[0] == 1 and B > 1:
            ref_path = ref_path.repeat(B, 1, 1)

        # 解包自车状态
        ego_xy = ego_state[..., :2]  # [B,1,2]
        ego_phi = ego_state[..., 4]  # [B,1]
        ego_vlon = ego_state[..., 2]  # [B,1]

        # 找最近参考点
        dist = torch.norm(ego_xy.unsqueeze(2) - ref_path.unsqueeze(1), dim=-1)  # [B,1,N]
        min_dist, closest_idx = torch.min(dist, dim=-1)  # [B,1]
        ref_idx = torch.clamp(closest_idx + self.env.carla_cfg['world']['ref_offset'], max=ref_path.shape[1] - 1)  # 前瞻2个点

        # 批量获取参考点和航向
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

        # 速度误差δ_v（参考速度5m/s，先让车能动）
        delta_v = ego_vlon - self.ref_vlon

        # 拼接误差
        ref_error = torch.stack([delta_p, delta_phi, delta_v], dim=-1)
        return ref_error

    def _forward_horizon(self,
                         state_tensor: torch.Tensor,
                         ref_path_tensor: torch.Tensor,
                         road_state: torch.Tensor = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        严格对齐论文OCP公式与GEP算法，修复所有维度不匹配问题
        :param state_tensor: 初始状态 [B, 121]
        :param ref_path_tensor: 参考路径 [1, N, 2]
        :return: step_l 每步成本, step_phi 每步约束违反量, states_traj 推演状态轨迹
        """
        B = state_tensor.shape[0]
        # 初始状态解包
        ego_state, other_states, ref_error = self.unpack_tensor(state_tensor.unsqueeze(1))
        current_ego = ego_state.clone()
        current_other = other_states.clone()
        current_ref_error = self._calc_ref_error_from_state(current_ego, ref_path_tensor)

        # 【道路状态仅在这里使用，用于约束计算】
        if road_state is not None:
            road_left = road_state[..., :int(self.DIM_ROAD/2)].contiguous().view(B, 1, int(self.DIM_ROAD/4), 2)
            road_right = road_state[..., int(self.DIM_ROAD/2):].contiguous().view(B, 1, int(self.DIM_ROAD/4), 2)
        else:
            # 若没有道路信息，约束违反量设为0
            road_left = torch.zeros(B, 1, 20, 2, device=self.device)
            road_right = torch.zeros(B, 1, 20, 2, device=self.device)


        # 安全距离平方（避免开根号，提升计算效率）
        safe_veh_sq = self.other_car_min_distance ** 2
        safe_road_sq = self.road_min_distance ** 2

        step_l_list = []
        step_phi_list = []
        trajectory_states = []

        for t in range(self.horizon):
            # 拼接完整121维状态
            current_state = torch.cat([
                current_ego.view(-1, self.DIM_EGO),
                current_other.view(-1, self.DIM_OTHER),
                current_ref_error.view(-1, self.DIM_REF_ERROR)
            ], dim=1)

            # 网络输出归一化动作 → 映射到物理量
            norm_action = self.actor(current_state)  # [B, 2]
            a_phy = norm_action[:, 0:1] * 2.25 - 0.75  # [-1,1] → [-3, 1.5] m/s²
            delta_phy = norm_action[:, 1:2] * 0.4  # [-1,1] → [-0.4, 0.4] rad
            phy_action = torch.cat([a_phy, delta_phy], dim=1)  # [B, 2]

            # 动力学推演
            next_ego = self.dynamics_model(current_ego, phy_action)
            next_other = self.predict_other_next_batch(current_other, self.dt)
            next_ref_error = self._calc_ref_error_from_state(next_ego, ref_path_tensor)

            # 拼接下一状态
            next_state = torch.cat([
                next_ego.view(-1, self.DIM_EGO),
                next_other.view(-1, self.DIM_OTHER),
                next_ref_error.view(-1, self.DIM_REF_ERROR)
            ], dim=1)
            trajectory_states.append(next_state)

            # --------------------------
            # 1. 计算OCP成本项 step_l（修复维度匹配）
            # --------------------------
            # 跟踪误差成本：[B]
            lat_err_t = next_ref_error[..., 0].squeeze(1)  # [B,1] -> [B]
            head_err_t = next_ref_error[..., 1].squeeze(1)
            speed_err_t = next_ref_error[..., 2].squeeze(1)
            err_cost = self.q_lat * (lat_err_t ** 2) + \
                       self.q_head * (head_err_t ** 2) + \
                       self.q_speed * (speed_err_t ** 2)  # [B]

            # 控制量成本：确保结果为 [B]，使用逐元素相乘再求和
            # 论文公式：u^T * R * u
            r_weights = torch.tensor([0.005, 0.1], device=self.device).float()  # 直接取对角元素 [2]
            control_cost = torch.sum((phy_action ** 2) * r_weights, dim=1)  # [B,2] * [2] -> [B,2] -> sum -> [B]

            # 单步总成本：[B]
            step_l = err_cost + control_cost

            # --------------------------
            # 2. 补全约束违反量 step_phi（修复维度广播）
            # --------------------------
            ego_xy = next_ego[..., :2]  # [B, 1, 2]
            phi_violation = torch.zeros(B, device=self.device)  # [B]

            # --- 2.1 周车安全距离约束 ---
            other_xy = next_other[..., :2]  # [B, 1, 8, 2]
            # 计算自车与每个周车的距离平方：[B, 1, 8]
            dist_veh_sq = torch.sum((ego_xy.unsqueeze(2) - other_xy) ** 2, dim=-1)
            # 约束违反量：max(0, 安全距离 - 实际距离)，求和所有周车 -> [B]
            veh_violation = torch.maximum(safe_veh_sq - dist_veh_sq, torch.zeros_like(dist_veh_sq))
            phi_violation += veh_violation.sum(dim=[1, 2])  # 求和第1、2维 -> [B]

            # --- 2.2 道路边缘安全距离约束 ---
            # 计算自车到左、右边缘所有点的距离平方：[B, 1, 20]
            dist_left_sq = torch.sum((ego_xy.unsqueeze(2) - road_left) ** 2, dim=-1)
            dist_right_sq = torch.sum((ego_xy.unsqueeze(2) - road_right) ** 2, dim=-1)
            # 取最近的边缘点距离：[B, 1]
            min_left_sq, _ = torch.min(dist_left_sq, dim=-1)
            min_right_sq, _ = torch.min(dist_right_sq, dim=-1)
            # 约束违反量：[B]
            left_violation = torch.maximum(safe_road_sq - min_left_sq, torch.zeros_like(min_left_sq)).squeeze(1)
            right_violation = torch.maximum(safe_road_sq - min_right_sq, torch.zeros_like(min_right_sq)).squeeze(1)
            phi_violation += left_violation + right_violation

            # 单步总约束违反量：[B]
            step_phi = phi_violation

            # 加入列表
            step_l_list.append(step_l)
            step_phi_list.append(step_phi)

            # 更新状态
            current_ego = next_ego
            current_other = next_other
            current_ref_error = next_ref_error

        # 拼接时域结果：[B, horizon]
        step_l = torch.stack(step_l_list).transpose(0, 1)
        step_phi = torch.stack(step_phi_list).transpose(0, 1)
        states_traj = torch.stack(trajectory_states).transpose(0, 1)

        return step_l, step_phi, states_traj

    def select_action(self, obs: Any, deterministic: bool = False):
        """
         动作链路：归一化动作 → 物理量映射，强制起步加速
        """
        with torch.no_grad():
            # 1. 统一观测格式
            if isinstance(obs, list):
                obs_np = np.array(obs, dtype=np.float32).flatten()
            elif isinstance(obs, np.ndarray):
                obs_np = obs.flatten()
            else:
                obs_np = np.array(obs, dtype=np.float32).flatten()

            # 2. 维度校验
            if obs_np.shape[0] != self.TOTAL_STATE_DIM:
                logger.warning(f"观测维度异常: {obs_np.shape[0]} (期望{self.TOTAL_STATE_DIM})，自动修正")
                obs_121 = np.zeros(self.TOTAL_STATE_DIM, dtype=np.float32)
                valid_len = min(len(obs_np), self.TOTAL_STATE_DIM)
                obs_121[:valid_len] = obs_np[:valid_len]
                obs_np = obs_121

            # --------------------------
            # 前n步完全强制随机正加速，打破刹车死循环
            # --------------------------
            if not deterministic and not self.buffer.should_start_training():
                # 强制正加速度：归一化加速度 [0.2, 1.0] → 物理量 [0, 1.5]m/s²，绝对不会刹车
                a_norm = np.random.uniform(0.2, 1.0, size=1)
                # 小范围转向，避免转圈
                delta_norm = np.random.uniform(-0.2, 0.2, size=1)
                norm_action = np.concatenate([a_norm, delta_norm])
            else:
                # 网络推理
                obs_tensor = torch.from_numpy(obs_np).to(self.device).float()
                norm_action = self.actor(obs_tensor.unsqueeze(0)).squeeze(0)
                norm_action = norm_action.cpu().numpy().flatten()

                # 推理阶段加小噪声
                if not deterministic:
                    noise = np.random.normal(0, [0.1, 0.05], size=norm_action.shape)
                    norm_action = np.clip(norm_action + noise, -1.0, 1.0)

            # --------------------------
            # 归一化动作 → 论文物理量 映射，100%对齐链路
            # --------------------------
            # 归一化加速度 [-1,1] → 物理加速度 [-3, 1.5] m/s²
            a_phy = np.interp(norm_action[0], [-1, 1], [-3.0, 1.5])
            # 归一化转向 [-1,1] → 物理转向 [-0.4, 0.4] rad
            delta_phy = np.interp(norm_action[1], [-1, 1], [-0.4, 0.4])
            # 最终输出物理动作，传给环境
            phy_action = np.array([a_phy, delta_phy], dtype=np.float32)

            return phy_action, np.zeros(1, dtype=np.float32)

    def update(self, ref_path_tensor,state,road_state):
        """
        严格对齐论文算法1 GEP训练逻辑
        每步执行策略评估(Critic更新)，每步执行策略改进(Actor更新)，每m步放大惩罚因子
        """
        # 解析批量状态，严格校验维度
        state_tensor = torch.from_numpy(state).to(self.device).float()
        road_tensor = torch.from_numpy(road_state).to(self.device).float()

        # --------------------------
        # 1. Critic更新（策略评估，每步执行，对齐论文有限时域OCP）
        # --------------------------
        with torch.no_grad():
            step_l, step_phi, states_traj = self._forward_horizon(state_tensor, ref_path_tensor,road_tensor)
            step_aug_l = step_l + self.init_penalty * step_phi
            # 有限时域累计成本（无折扣γ=1，和论文OCP完全等价）
            targets = torch.flip(torch.cumsum(torch.flip(step_aug_l, [1]), dim=1), [1])

        # 拟合Critic：V(s) ≈ 累计未来成本
        all_states = torch.cat([state_tensor.unsqueeze(1), states_traj], dim=1)
        critic_inputs = all_states[:, :-1].reshape(-1, self.TOTAL_STATE_DIM)
        critic_targets = targets.reshape(-1, 1)
        pred = self.critic(critic_inputs)
        critic_loss = F.mse_loss(pred, critic_targets)

        self.critic_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        critic_loss.backward()
        self.critic_optimizer.step()

        # --------------------------
        # 2. Actor更新（策略改进，每步执行，对齐GEP算法）
        # --------------------------
        # 重新前向推演，保留完整计算图
        step_l_actor, step_phi_actor, _ = self._forward_horizon(state_tensor, ref_path_tensor,road_tensor)
        # GEP总损失：成本项 + 惩罚因子×约束违反项
        actor_loss = step_l_actor.mean() + self.init_penalty * step_phi_actor.mean()

        self.actor_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # 放大梯度裁剪阈值，避免梯度消失
        actor_loss.backward()
        self.actor_optimizer.step()
        actor_updated = True

        # --------------------------
        # 3. GEP惩罚因子放大（每m步执行，严格对齐论文）
        # --------------------------
        if self.global_step % self.amplifier_m == 0 and self.global_step > 0:
            old_penalty = self.init_penalty
            self.init_penalty = min(self.init_penalty * self.amplifier_c, self.max_penalty)
            self.gep_iteration += 1
            logger.info(f"[GEP] 惩罚因子更新: {old_penalty:.4f} → {self.init_penalty:.4f}")

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
            raise ValueError(f"周车状态维度必须为 [B,1,8,4]，当前={other_states.shape}")
        x = other_states[..., 0]
        y = other_states[..., 1]
        vx = other_states[..., 2]
        vy = other_states[..., 3]
        x_next = x + dt * vx
        y_next = y + dt * vy
        next_other = torch.stack([x_next, y_next, vx, vy], dim=-1)
        return next_other

    def unpack_tensor(self, data: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """解包121维状态，严格对齐维度"""
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
            'buffer_data': save_info['buffer_data']
        }
        metrics = {'episode': extra_info['globe_eps']}
        save_checkpoint(
            model=model, model_name='ocp-v1.0', optimizer=optimizer,
            extra_info=extra_info, metrics=metrics, env_name=save_info['map']
        )
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

        # 获取缓冲区数据
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