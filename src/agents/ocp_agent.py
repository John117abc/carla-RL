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
        """
        严格对齐论文算法1：Dynamic Optimal Tracking-Offline Training
        核心流程：PEV（策略评估）→ 每m次迭代执行PIM（策略改进）
        维度完全对齐121维输入，移除冗余的参考线/道路预处理
        """
        # 1. 缓冲区采样（无数据则直接返回）
        batch_data = self.buffer.sample_batch(self.batch_size)
        if len(batch_data) == 0:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "penalty": self.init_penalty,
                "gep_iteration": self.global_step,
                "actor_updated": False
            }

        # 2. 解析批量数据（严格提取121维状态）
        states_list = []
        for item in batch_data:
            state, _, _, _, _, _ = item
            # 强制转换为121维
            state_np = np.array(state, dtype=np.float32).flatten()
            if state_np.shape[0] != self.TOTAL_STATE_DIM:
                state_121 = np.zeros(self.TOTAL_STATE_DIM, dtype=np.float32)
                valid_len = min(len(state_np), self.TOTAL_STATE_DIM)
                state_121[:valid_len] = state_np[:valid_len]
                state_np = state_121
            states_list.append(state_np)

        # 转换为tensor（[B, 121]）
        state_tensor = torch.from_numpy(np.stack(states_list)).to(self.device).float()
        state_tensor.requires_grad_(True)

        # 3. 有限时域前向推演（Horizon步）
        trajectory_actions = []  # [Horizon, B, 2]
        trajectory_states = []   # [Horizon, B, 121]

        # 初始状态解包
        ego_state, other_states, ref_error, road_state = self.unpack_tensor(state_tensor.unsqueeze(1))
        current_ego = ego_state.clone()          # [B, 1, 6]
        current_other = other_states.clone()     # [B, 1, 8, 4]
        current_ref_error = ref_error.clone()    # [B, 1, 3]
        current_road = road_state.clone()        # [B, 1, 80]

        # 时域推演循环
        for t in range(self.horizon):
            # 3.1 拼接当前完整状态（121维）
            current_state = torch.cat([
                current_ego.view(-1, self.DIM_EGO),
                current_other.view(-1, self.DIM_OTHER),
                current_ref_error.view(-1, self.DIM_REF_ERROR),
                current_road.view(-1, self.DIM_ROAD)
            ], dim=1)  # [B, 121]

            # 3.2 Actor输出动作
            action = self.actor(current_state)
            trajectory_actions.append(action)

            # 3.3 动力学模型推演自车下一状态
            next_ego = self.dynamics_model(current_ego, action)  # [B, 1, 6]

            # 3.4 匀速模型推演其他车辆下一状态
            next_other = self.predict_other_next_batch(current_other, self.dt)  # [B, 1, 8, 4]

            # 3.5 参考误差更新（论文核心：δp, δφ, δv）
            # 从自车和参考误差计算新的误差状态
            lat_err = current_ref_error[..., 0]    # 横向误差δp
            head_err = current_ref_error[..., 1]   # 航向误差δφ
            speed_err = current_ref_error[..., 2]  # 速度误差δv
            # 误差衰减（模拟跟踪误差的自然变化）
            next_ref_error = torch.stack([
                lat_err * 0.95,
                head_err * 0.9,
                speed_err * 0.98
            ], dim=-1).unsqueeze(1)  # [B, 1, 3]

            # 3.6 道路状态保持不变（静态道路）
            next_road = current_road.clone()

            # 3.7 拼接下一状态
            next_state = torch.cat([
                next_ego.view(-1, self.DIM_EGO),
                next_other.view(-1, self.DIM_OTHER),
                next_ref_error.view(-1, self.DIM_REF_ERROR),
                next_road.view(-1, self.DIM_ROAD)
            ], dim=1)  # [B, 121]
            trajectory_states.append(next_state)

            # 3.8 更新当前状态
            current_ego = next_ego
            current_other = next_other
            current_ref_error = next_ref_error
            current_road = next_road

        # 4. 轨迹数据整理（维度转换：[B, Horizon, DIM]）
        states_traj = torch.stack(trajectory_states).transpose(0, 1)  # [B, Horizon, 121]
        actions_traj = torch.stack(trajectory_actions).transpose(0, 1)  # [B, Horizon, 2]

        # 5. 解包轨迹状态（提取误差和约束相关维度）
        _, _, ref_error_traj, road_traj = self.unpack_tensor(states_traj)
        lat_err = ref_error_traj[..., 0]    # [B, Horizon] 横向误差δp
        head_err = ref_error_traj[..., 1]   # [B, Horizon] 航向误差δφ
        speed_err = ref_error_traj[..., 2]  # [B, Horizon] 速度误差δv

        # 6. 计算论文定义的即时成本 l(s_t, u_t)
        # 6.1 误差成本（δp², δφ², δv²）
        err_cost = self.q_lat * (lat_err ** 2) + \
                   self.q_head * (head_err ** 2) + \
                   self.q_speed * (speed_err ** 2)

        # 6.2 控制成本（R矩阵加权）
        r_weights = torch.from_numpy(self.R_matrix.diagonal().copy()).to(self.device).float()  # [2]
        control_cost = (actions_traj ** 2) @ r_weights  # [B, Horizon]

        # 6.3 基础即时成本
        step_l = err_cost + control_cost  # [B, Horizon]

        # 7. 约束违反计算 φ(s_t, u_t)
        # 7.1 跟踪误差约束（|δp| + 0.5|δφ| > 1.0 视为违反）
        track_threshold = 1.0
        g_track = torch.abs(lat_err) + 0.5 * torch.abs(head_err) - track_threshold
        violation_track = F.relu(g_track)  # 负值置0（无违反）

        # 7.2 车辆碰撞约束（维度对齐 [B, H]）
        violation_car = torch.zeros_like(violation_track)  # [B, H]
        if other_states.numel() > 0:
            # 解析自车/其他车辆xy，确保维度匹配时域H
            # current_ego 维度：[B, 1, 6] → 扩展到 [B, H, 2]
            ego_xy = current_ego[..., :2].repeat(1, self.horizon, 1)  # [B, H, 2]
            # current_other 维度：[B, 1, 8, 6] → 调整为 [B, H, 8, 2]
            other_xy = current_other[..., :2].view(-1, 1, self.env.env_cfg['ocp']['others'], 2).repeat(1, self.horizon, 1, 1)  # [B, H, 8, 2]

            # 计算自车与其他车辆的相对位置（广播匹配）
            rel_pos = ego_xy.unsqueeze(2) - other_xy  # [B, H, 8, 2]
            dist_sq = (rel_pos ** 2).sum(dim=-1)  # [B, H, 8]

            # 碰撞约束：最小距离平方 - 实际距离平方，取relu后最大值（对8辆车取最大违反）
            g_car = self.other_car_min_distance ** 2 - dist_sq  # [B, H, 8]
            violation_car = F.relu(g_car).max(dim=-1)[0]  # [B, H]（max后保留[B,H]）

        # 7.3 道路边界约束（从80维道路数据计算）
        violation_road = torch.zeros_like(violation_track)
        if road_traj.numel() > 0:
            # 解析道路边界（左20点+右20点，各xy）
            road_left = road_traj[..., :self.env.env_cfg['ocp']['num_points'] * 2].view(-1, self.horizon, self.env.env_cfg['ocp']['num_points'], 2)  # [B, H, 20, 2]
            road_right = road_traj[..., self.env.env_cfg['ocp']['num_points'] * 2:].view(-1, self.horizon, self.env.env_cfg['ocp']['num_points'], 2)  # [B, H, 20, 2]

            # 修正1：调整自车xy轨迹维度，对齐road_left的B/H维度
            ego_xy = current_ego[..., :2].squeeze(1)  # [B, 2] → 先提取自车xy，去掉多余维度
            ego_xy_traj = ego_xy.unsqueeze(1).repeat(1, self.horizon, 1)  # [B, H, 2] → 扩展到horizon维度

            # 修正2：插入维度用于广播（匹配road_left的20个边界点维度）
            ego_xy_expand = ego_xy_traj.unsqueeze(2)  # [B, H, 1, 2]

            # 修正3：计算自车到边界所有点的距离，再取最小值
            dist_left_all = torch.norm(ego_xy_expand - road_left, dim=-1)  # [B, H, 20]
            dist_left = dist_left_all.min(dim=-1)[0]  # [B, H] → 对20个边界点取最小

            dist_right_all = torch.norm(ego_xy_expand - road_right, dim=-1)  # [B, H, 20]
            dist_right = dist_right_all.min(dim=-1)[0]  # [B, H]

            min_road_dist = torch.min(dist_left, dist_right)
            g_road = self.road_min_distance - min_road_dist
            violation_road = F.relu(g_road)

        # 7.4 总约束违反
        step_phi = violation_track + violation_car + violation_road  # [B, Horizon]

        # # 7.5 速度奖励（鼓励正向速度）
        # ego_speed = torch.norm(current_ego[..., 2:4], dim=-1).squeeze(1)  # [B]
        # speed_reward = 0.05 * torch.clamp(ego_speed, 0, 10).unsqueeze(1)  # [B, 1]

        # 7.6 增广成本（论文核心：l + ρ·φ - 速度奖励）
        # step_aug_l = step_l + self.init_penalty * step_phi - speed_reward  # [B, Horizon]
        step_aug_l = step_l + self.init_penalty * step_phi

        # 8. 策略评估（PEV）：拟合Critic（贝尔曼方程）
        # 8.1 整理轨迹状态和目标值
        all_states = torch.cat([
            state_tensor.unsqueeze(1),  # 初始状态 [B,1,121]
            states_traj                 # 推演状态 [B,H,121]
        ], dim=1)  # [B, H+1, 121]

        # 8.2 从后往前计算贝尔曼目标值
        targets = torch.zeros_like(step_aug_l)  # [B, H]
        targets[:, -1] = step_aug_l[:, -1]
        for t in reversed(range(self.horizon - 1)):
            next_state = all_states[:, t + 1].detach()  # 隔离计算图
            next_value = self.critic(next_state).squeeze()
            targets[:, t] = step_aug_l[:, t] + self.gamma * next_value

        # 8.3 训练Critic
        critic_inputs = all_states[:, :-1].detach().reshape(-1, self.TOTAL_STATE_DIM)  # [B*H, 121]
        critic_targets = targets.detach().reshape(-1, 1)  # [B*H, 1]
        pred = self.critic(critic_inputs)
        critic_loss = F.mse_loss(pred, critic_targets)

        # 8.4 更新Critic参数
        self.critic_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        critic_loss.backward()
        self.critic_optimizer.step()

        # 9. 策略改进（PIM）：每m次迭代执行一次
        actor_updated = False
        actor_loss = torch.tensor(0.0, device=self.device)

        if self.global_step % self.amplifier_m == 0:
            # 9.1 放大惩罚因子ρ（论文算法1步骤3）
            old_penalty = self.init_penalty
            self.init_penalty = min(self.init_penalty * self.amplifier_c, self.max_penalty)
            logger.info(
                f"[GEP] 迭代 {self.gep_iteration}: 惩罚因子ρ更新 {old_penalty:.4f} → {self.init_penalty:.4f}"
            )

            # 9.2 重新计算增广成本（保证计算图完整）
            # actor_loss = (
            #     step_l.mean() +
            #     self.init_penalty * step_phi.mean() -
            #     speed_reward.mean()
            # )

            actor_loss = (
                step_l.mean() +
                self.init_penalty * step_phi.mean()
            )

            # 9.3 更新Actor参数
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()

            # 9.4 On-Policy：清空缓冲区
            # self.buffer.clear()
            logger.info(
                f"[GEP] 迭代 {self.gep_iteration}: Actor更新完成，损失={actor_loss.item():.4f}，缓冲区已清空"
            )
            self.gep_iteration += 1
            actor_updated = True

        # 11. 返回日志数据
        return {
            "actor_loss": actor_loss.item() if torch.is_tensor(actor_loss) else 0.0,
            "critic_loss": critic_loss.item(),
            "penalty": self.init_penalty,
            "gep_iteration": self.global_step,
            "actor_updated": actor_updated
        }

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