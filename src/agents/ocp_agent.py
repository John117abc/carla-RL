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
    OCP智能体，复现《Integrated Decision and Control: Toward Interpretable and Computationally Efficient Driving Intelligence》
    论文中的Dynamic Optimal Tracking-Offline Training算法
    输入维度严格对齐：121维 = ego(6) + other(8×4=32) + ref_error(3) + road(80)
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


        # 核心维度定义（严格对齐用户输入）
        self.DIM_EGO = 6          # 自车状态维度 [x,y,vx,vy,psi,omega]
        self.DIM_OTHER = self.env.env_cfg['ocp']['others'] * 4       # 其他车辆状态 8×4
        self.DIM_REF_ERROR = 3    # 参考误差 [横向误差δp, 航向误差δφ, 速度误差δv]
        self.DIM_ROAD = self.env.env_cfg['ocp']['num_points'] * 4        # 道路边界 80维 [左1x,左1y,...左20x,左20y,右1x,右1y,...右20x,右20y]
        self.TOTAL_STATE_DIM = self.DIM_EGO + self.DIM_OTHER + self.DIM_REF_ERROR + self.DIM_ROAD  # 121维

        # 网络初始化（输入维度改为121）
        self.dt = self.ocp_config['dt']
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
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.ocp_config['lr_critic'],
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 损失函数
        self.loss_func = nn.MSELoss()

        # 超参数（论文+工程配置）
        self.init_penalty = self.ocp_config['init_penalty']
        self.max_penalty = self.ocp_config['max_penalty']
        self.amplifier_c = self.ocp_config['amplifier_c']
        self.amplifier_m = self.ocp_config['amplifier_m']
        self.other_car_min_distance = self.ocp_config['other_car_min_distance']
        self.road_min_distance = self.ocp_config['road_min_distance']
        self.gamma = self.ocp_config['gamma']

        # 论文核心：误差状态权重矩阵（对应δp, δφ, δv）
        self.q_lat = 0.04    # 横向跟踪误差权重（δp）
        self.q_head = 0.1    # 航向误差权重（δφ）
        self.q_speed = 0.01  # 速度误差权重（δv）

        # 控制权重矩阵（加速度、转向角）
        self.R_matrix = np.diag([0.005, 0.1])

        # 采样/时域参数
        self.batch_size = self.ocp_config['batch_size']
        self.horizon = self.ocp_config['horizon']

        # 缓冲区初始化
        self.buffer = StochasticBuffer(
            min_start_train=self.ocp_config['min_start_train'],
            total_capacity=self.ocp_config['total_capacity']
        )

        # 日志/迭代记录
        self.globe_eps = 0
        self.history_loss = []
        self.global_step = 0
        self.gep_iteration = 0  # 论文算法1外层迭代次数i

    def select_action(self, obs: Any, deterministic: bool = False):
        """
        根据121维观测选择动作，严格保证：
        1. 维度正确（121维输入）；
        2. 动作缩放到[-1,1]（加速度、转向角）；
        3. 训练初期强制正向加速度探索，打破死亡循环；
        4. 兼容多格式输入（list/np.ndarray）。
        """
        with torch.no_grad():
            # 1. 统一观测格式为numpy数组
            if isinstance(obs, list):
                obs_np = np.array(obs, dtype=np.float32).flatten()
            elif isinstance(obs, np.ndarray):
                obs_np = obs.flatten()
            else:
                obs_np = np.array(obs, dtype=np.float32).flatten()

            # 2. 严格校验维度（强制补0/截断到121维）
            if obs_np.shape[0] != self.TOTAL_STATE_DIM:
                logger.warning(
                    f"观测维度异常: {obs_np.shape[0]} (期望121)，自动修正"
                )
                obs_121 = np.zeros(self.TOTAL_STATE_DIM, dtype=np.float32)
                valid_len = min(len(obs_np), self.TOTAL_STATE_DIM)
                obs_121[:valid_len] = obs_np[:valid_len]
                obs_np = obs_121

            # 3. 转换为tensor输入Actor网络
            obs_tensor = torch.from_numpy(obs_np).to(self.device).float()
            raw_action = self.actor(obs_tensor.unsqueeze(0)).squeeze(0)

            # 4. 动作裁剪到环境要求的[-1,1]
            action_low = torch.tensor(self.action_space.low).to(self.device)
            action_high = torch.tensor(self.action_space.high).to(self.device)
            action = torch.clamp(raw_action, action_low, action_high)
            action = action.cpu().numpy().flatten()

            # 5. 定向探索（训练初期强制正向加速度）
            if not deterministic:
                if self.global_step < 5000:
                    # 前5000步：加速度均值0.5（正向），转向小噪声
                    accel_noise = np.random.normal(0.5, 0.2, size=1)
                    steer_noise = np.random.normal(0, 0.1, size=1)
                    noise = np.concatenate([accel_noise, steer_noise])
                else:
                    # 后期：正常高斯探索
                    noise = np.random.normal(0, 0.1, size=action.shape)
                action = np.clip(action + noise, -1.0, 1.0)

            return action, np.zeros(1, dtype=np.float32)

    def update(self):
        # 1. 缓冲区采样校验
        batch_data = self.buffer.sample_batch(self.batch_size)
        if len(batch_data) == 0:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "penalty": self.init_penalty,
                "gep_iteration": self.global_step,
                "actor_updated": False
            }

        # 2. 解析批量状态，严格对齐121维
        states_list = []
        for item in batch_data:
            state, _, _, _, _, _ = item
            state_np = np.array(state, dtype=np.float32).flatten()
            if state_np.shape[0] != self.TOTAL_STATE_DIM:
                state_121 = np.zeros(self.TOTAL_STATE_DIM, dtype=np.float32)
                valid_len = min(len(state_np), self.TOTAL_STATE_DIM)
                state_121[:valid_len] = state_np[:valid_len]
                state_np = state_121
            states_list.append(state_np)
        state_tensor = torch.from_numpy(np.stack(states_list)).to(self.device).float()

        # 3. 前向推演，用于Critic更新
        step_l, step_phi, states_traj = self._forward_horizon(state_tensor)
        step_aug_l = step_l + self.init_penalty * step_phi

        # 4. 策略评估（Critic更新）：严格对齐有限时域OCP，移除γ折扣！
        all_states = torch.cat([state_tensor.unsqueeze(1), states_traj], dim=1)
        # 有限时域OCP的价值目标：从t步到结束的累计增广成本，无折扣，和论文完全等价
        targets = torch.zeros_like(step_aug_l)
        targets[:, -1] = step_aug_l[:, -1]
        for t in reversed(range(self.horizon - 1)):
            targets[:, t] = step_aug_l[:, t] + targets[:, t + 1]

        # Critic参数更新
        critic_inputs = all_states[:, :-1].detach().reshape(-1, self.TOTAL_STATE_DIM)
        critic_targets = targets.detach().reshape(-1, 1)
        pred = self.critic(critic_inputs)
        critic_loss = F.mse_loss(pred, critic_targets)

        self.critic_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        critic_loss.backward()
        self.critic_optimizer.step()

        # 5. 策略改进（Actor更新）：严格对齐论文GEP逻辑，每m步执行
        actor_updated = False
        actor_loss = torch.tensor(0.0, device=self.device)
        if self.global_step % self.amplifier_m == 0:
            # 【论文GEP核心】先优化当前ρ下的目标，再放大ρ！
            # 重新前向推演，重建Actor的完整计算图
            step_l_actor, step_phi_actor, _ = self._forward_horizon(state_tensor)
            # 计算论文定义的无约束目标J_p
            actor_loss = step_l_actor.mean() + self.init_penalty * step_phi_actor.mean()

            # 更新Actor参数
            self.actor_optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            actor_loss.backward()
            self.actor_optimizer.step()

            # 【论文GEP逻辑】优化完成后，再放大惩罚因子ρ
            old_penalty = self.init_penalty
            self.init_penalty = min(self.init_penalty * self.amplifier_c, self.max_penalty)
            self.gep_iteration += 1
            actor_updated = True

        # 返回日志数据
        return {
            "actor_loss": actor_loss.item() if torch.is_tensor(actor_loss) else 0.0,
            "critic_loss": critic_loss.item(),
            "penalty": self.init_penalty,
            "gep_iteration": self.gep_iteration,
            "actor_updated": actor_updated
        }

    def _forward_horizon(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        封装horizon前向推演逻辑，每次调用都重建完整计算图
        返回：step_l(跟踪+控制成本)、step_phi(约束违反)、states_traj(轨迹状态)
        """
        B = state_tensor.shape[0]
        # 初始状态解包（严格对齐121维定义）
        ego_state, other_states, ref_error, road_state = self.unpack_tensor(state_tensor.unsqueeze(1))
        current_ego = ego_state.clone()  # [B, 1, 6]
        current_other = other_states.clone()  # [B, 1, 8, 4]
        current_ref_error = ref_error.clone()  # [B, 1, 3]
        current_road = road_state.clone()  # [B, 1, 80]

        step_l_list = []
        step_phi_list = []
        trajectory_states = []

        for t in range(self.horizon):
            # 拼接当前完整121维状态
            current_state = torch.cat([
                current_ego.view(-1, self.DIM_EGO),
                current_other.view(-1, self.DIM_OTHER),
                current_ref_error.view(-1, self.DIM_REF_ERROR),
                current_road.view(-1, self.DIM_ROAD)
            ], dim=1)

            # Actor输出动作，保留完整计算图
            action = self.actor(current_state)

            # 动力学推演（用修复后的bicycle模型）
            next_ego = self.dynamics_model(current_ego, action)
            next_other = self.predict_other_next_batch(current_other, self.dt)

            # --------------------------
            # 修复点1：参考误差更新，移除多余的 unsqueeze(1)
            # --------------------------
            lat_err = current_ref_error[..., 0]  # [B, 1]
            head_err = current_ref_error[..., 1]  # [B, 1]
            speed_err = current_ref_error[..., 2]  # [B, 1]

            # 模拟跟踪误差的真实变化，stack 后直接是 [B, 1, 3]
            next_ref_error = torch.stack([
                lat_err * 0.95,
                head_err * 0.9,
                speed_err * 0.98
            ], dim=-1)  # ✅ 维度保持 [B, 1, 3]

            next_road = current_road.clone()

            # 拼接下一状态
            next_state = torch.cat([
                next_ego.view(-1, self.DIM_EGO),
                next_other.view(-1, self.DIM_OTHER),
                next_ref_error.view(-1, self.DIM_REF_ERROR),
                next_road.view(-1, self.DIM_ROAD)
            ], dim=1)
            trajectory_states.append(next_state)

            # --------------------------
            # 1. 计算论文定义的即时成本l(s_t,u_t)
            # --------------------------
            lat_err_t = next_ref_error[..., 0].squeeze(1)  # [B]
            head_err_t = next_ref_error[..., 1].squeeze(1)  # [B]
            speed_err_t = next_ref_error[..., 2].squeeze(1)  # [B]

            # 跟踪误差成本（严格对齐论文权重）
            err_cost = self.q_lat * (lat_err_t ** 2) + \
                       self.q_head * (head_err_t ** 2) + \
                       self.q_speed * (speed_err_t ** 2)

            # 控制成本
            r_weights = torch.from_numpy(self.R_matrix.diagonal().copy()).to(self.device).float()
            control_cost = (action ** 2) @ r_weights  # [B]

            step_l = err_cost + control_cost

            # --------------------------
            # 2. 计算逐状态约束违反φ(s_t)（仅安全约束，和论文一致）
            # --------------------------
            # 车辆碰撞约束
            violation_car = torch.zeros_like(step_l)
            if other_states.numel() > 0:
                ego_xy_t = next_ego[..., :2].squeeze(1)  # [B, 2]
                other_xy_t = next_other[..., :2].squeeze(1)  # [B, 8, 2]
                rel_pos = ego_xy_t.unsqueeze(1) - other_xy_t  # [B, 8, 2]
                dist_sq = (rel_pos ** 2).sum(dim=-1)  # [B, 8]
                g_car = self.other_car_min_distance ** 2 - dist_sq
                violation_car = F.relu(g_car).max(dim=-1)[0]  # [B]

            # 道路边界约束
            violation_road = torch.zeros_like(step_l)
            if next_road.numel() > 0:
                num_points = self.env.env_cfg['ocp']['num_points']
                road_flat = next_road.squeeze(1)  # [B, 80]
                road_left = road_flat[:, :num_points * 2].view(-1, num_points, 2)  # [B, 20, 2]
                road_right = road_flat[:, num_points * 2:].view(-1, num_points, 2)  # [B, 20, 2]
                ego_xy_t = next_ego[..., :2].squeeze(1).unsqueeze(1)  # [B, 1, 2]

                dist_left = torch.norm(ego_xy_t - road_left, dim=-1).min(dim=-1)[0]  # [B]
                dist_right = torch.norm(ego_xy_t - road_right, dim=-1).min(dim=-1)[0]  # [B]
                min_road_dist = torch.min(dist_left, dist_right)
                g_road = self.road_min_distance - min_road_dist
                violation_road = F.relu(g_road)

            # 总约束违反
            step_phi = violation_car + violation_road

            # --------------------------
            # 修复点2：强制确保维度是 [B]，再加入列表
            # --------------------------
            step_l = step_l.squeeze()
            step_phi = step_phi.squeeze()

            step_l_list.append(step_l)
            step_phi_list.append(step_phi)

            # 更新当前状态
            current_ego = next_ego
            current_other = next_other
            current_ref_error = next_ref_error
            current_road = next_road

        # 整理输出
        step_l = torch.stack(step_l_list).transpose(0, 1)  # [B, Horizon]
        step_phi = torch.stack(step_phi_list).transpose(0, 1)  # [B, Horizon]
        states_traj = torch.stack(trajectory_states).transpose(0, 1)  # [B, Horizon, 121]
        return step_l, step_phi, states_traj

    def predict_other_next_batch(self, other_states: torch.Tensor, dt: float) -> torch.Tensor:
        """
        批量预测其他车辆下一状态（匀速模型）
        输入维度：[B, 1, 8, 4] → 输出维度：[B, 1, 8, 4]
        其他车辆状态定义：[x, y, vx, vy]
        """
        if other_states.dim() != 4 or other_states.shape[3] != 4:
            raise ValueError(
                f"其他车辆状态维度必须为 [B,1,8,4]，当前={other_states.shape}"
            )

        # 匀速模型：x' = x + vx*dt, y' = y + vy*dt
        x = other_states[..., 0]
        y = other_states[..., 1]
        vx = other_states[..., 2]
        vy = other_states[..., 3]

        x_next = x + dt * vx
        y_next = y + dt * vy
        vx_next = vx  # 速度不变
        vy_next = vy  # 速度不变

        next_other = torch.stack([x_next, y_next, vx_next, vy_next], dim=-1)
        return next_other

    def unpack_tensor(self, data: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        严格解包121维状态张量
        输入维度：[B, N, 121] → 输出：
        - ego_state: [B, N, 6]
        - other_states: [B, N, 8, 4]
        - ref_error: [B, N, 3]
        - road_state: [B, N, 80]
        """
        if data.dim() != 3 or data.shape[2] != self.TOTAL_STATE_DIM:
            raise ValueError(
                f"输入张量必须为 [B,N,121]，当前={data.shape}"
            )

        B, N = data.shape[0], data.shape[1]

        # 解包各部分
        ego_state = data[:, :, 0:self.DIM_EGO]  # 0-5
        other_raw = data[:, :, self.DIM_EGO:self.DIM_EGO+self.DIM_OTHER]  # 6-37
        other_states = other_raw.view(B, N, self.env.env_cfg['ocp']['others'], 4)  # 8×4=32
        ref_error = data[:, :, self.DIM_EGO+self.DIM_OTHER:self.DIM_EGO+self.DIM_OTHER+self.DIM_REF_ERROR]  # 38-40
        road_state = data[:, :, self.DIM_EGO+self.DIM_OTHER+self.DIM_REF_ERROR:]  # 41-120

        return ego_state, other_states, ref_error, road_state

    def save(self, save_info: Dict[str, Any]) -> None:
        """保存模型参数（包含维度信息）"""
        # 整理保存数据
        model = {
            'actor': self.actor,
            'critic': self.critic
        }
        optimizer = {
            'actor_optim': self.actor_optimizer,
            'critic_optim': self.critic_optimizer
        }
        extra_info = {
            'config': save_info['rl_config'],
            'global_step': self.global_step + save_info['global_step'],
            'history': self.history_loss + save_info['history_loss'],
            'globe_eps': self.globe_eps + self.base_config['save_freq'],
            'state_dim': self.TOTAL_STATE_DIM,  # 保存维度信息
            'punish_factor': self.init_penalty,  # 惩罚因子
            'gep_iteration': self.gep_iteration   # GEP迭代次数
        }
        metrics = {
            'episode': extra_info['globe_eps']
        }

        # 保存 checkpoint
        save_checkpoint(
            model=model,
            model_name='ocp-v1.0',
            optimizer=optimizer,
            extra_info=extra_info,
            metrics=metrics,
            env_name=save_info['map']
        )

        # 更新本地日志
        self.globe_eps = extra_info['globe_eps']
        self.global_step = extra_info['global_step']
        self.history_loss = extra_info['history']
        self.init_penalty = extra_info['punish_factor']
        self.gep_iteration = extra_info['gep_iteration']


    def load(self, path: str) -> Dict[str, Any]:
        """加载模型参数（兼容维度校验）"""
        checkpoint = load_checkpoint(
            model={'actor': self.actor, 'critic': self.critic},
            filepath=path,
            optimizer={'actor_optim': self.actor_optimizer, 'critic_optim': self.critic_optimizer},
            device=self.device
        )

        # 校验维度一致性
        loaded_dim = checkpoint.get('state_dim', 121)
        if loaded_dim != self.TOTAL_STATE_DIM:
            logger.warning(
                f"加载的模型维度({loaded_dim})与当前配置({self.TOTAL_STATE_DIM})不一致"
            )

        # 更新本地状态
        self.globe_eps = checkpoint['globe_eps']
        self.history_loss = checkpoint['history']
        self.global_step = checkpoint['global_step']
        self.init_penalty = checkpoint['punish_factor']
        self.gep_iteration = checkpoint['gep_iteration']

        return checkpoint

    def eval(self, num_episodes: int = 10, action_repeat: int = 5) -> Tuple[float, float]:
        """评估智能体性能（确定性动作）"""
        total_rewards = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                # 仅使用ocp_obs维度
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