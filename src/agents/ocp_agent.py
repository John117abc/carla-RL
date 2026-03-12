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
from src.models.actor_critic import ActorNet,CriticNet
from src.models.bicycle import BicycleModel
from src.utils import save_checkpoint,load_checkpoint
from src.buffer import StochasticBuffer
from src.utils import get_logger
from src.carla_utils import get_ref_observation_torch,predict_other_next,get_road_observation_torch

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
        self.actor = ActorNet(state_dim = np.prod(self.observation_space['ocp_obs'].shape), hidden_dim=self.ocp_config['hidden_dim']).to(self.device)
        self.critic = CriticNet(state_dim = np.prod(self.observation_space['ocp_obs'].shape), hidden_dim=self.ocp_config['hidden_dim']).to(self.device)
        self.dynamics_model = BicycleModel(dt=self.dt, L=2.9)
        self.actor.parameters(),

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.ocp_config['lr_actor'],betas=(0.9, 0.999),eps = 1e-8)
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
        # self.Q_matrix = np.diag([0.04, 0.04, 0.01, 0.01, 0.1, 0.02])
        self.Q_matrix = np.diag([10.0, 10.0, 2.0, 2.0, 50.0, 5.0])
        # self.R_matrix = np.diag([0.1, 0.005])
        self.R_matrix = np.diag([0.5, 5.0])
        self.M_matrix = np.diag([1,1,0,0,0,0])
        # 严格使用s^ref = [δp, δφ, δv ]状态时候的Q
        self.Q_matrix_ref = np.diag([0.04,0.01,0.01])

        # 采样数量
        self.batch_size = self.ocp_config['batch_size']

        # 预测步数
        self.horizon = self.ocp_config['horizon']

        # 初始化缓冲区
        self.buffer = StochasticBuffer(min_start_train = self.ocp_config['min_start_train'],
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
            action = self.actor(obs_tensor.unsqueeze(0)).squeeze(0)
            action = action.cpu().numpy().flatten()
        return np.clip(action, self.action_space.low, self.action_space.high), np.zeros(1, dtype=np.float32)


    def update(self):
        # 1. 采样 Batch (假设 buffer.sample_batch 返回的是 numpy 数组列表或字典，需转为 Tensor)
        # 假设 batch_s0 是一个列表，每个元素是 (state, action, reward, next_state, done, info)
        batch_data = self.buffer.sample_batch(self.batch_size)

        if len(batch_data) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "actor_grad_norm": 0.0}

        # --- 数据预处理：构建初始 Batch Tensor ---
        # 将列表转换为统一的 Batch Tensor
        # 假设 state 是 numpy array [dim] 或 list
        states_list = []
        infos = []

        for item in batch_data:
            state, action, reward, _, done, info = item
            obs_vector = state[0] if isinstance(state, (list, tuple)) else state
            states_list.append(np.asarray(obs_vector, dtype=np.float32))
            infos.append(info)

        # [Batch, Dim]
        states_np = np.stack(states_list, axis=0)
        state_tensor = torch.from_numpy(states_np).to(self.device).float()

        # 需要梯度的状态副本 (用于 Actor 优化)
        state_tensor.requires_grad_(True)

        # 解包初始状态 (假设 unpack_tensor 支持 Batch 输入)
        # ego_state: [Batch, 1, 6] 或 [Batch, 6] -> 需调整为 [Batch, 1, 6] 以匹配 loss.py 风格
        ego_state, other_states, road_state, ref_state = self.unpack_tensor(state_tensor.unsqueeze(0))

        # 确保维度为 [Batch, 1, Dim_Ego], [Batch, N_npc, Dim_Other] 等
        if ego_state.dim() == 2:
            ego_state = ego_state.unsqueeze(1)  # [B, 6] -> [B, 1, 6]

            # 找出所有序列的最大长度 (Ref, Left, Right 可能长度都不同)
        max_len_ref = max(len(info.get('ref_path_xy', [])) for info in infos)
        max_len_left = max(len(info.get('road_left_xy', [])) for info in infos)
        max_len_right = max(len(info.get('road_right_xy', [])) for info in infos)

        ref_batch_list = []
        left_batch_list = []
        right_batch_list = []

        cleaned_refs = []
        max_len = 0

        for info in infos:
            # 获取当前样本的参考线长度
            current_len = len(info.get('ref_path_xy', []))

            # 更新最大值
            if current_len > max_len:
                max_len = current_len

        for info in infos:
            # 1. 处理参考线 (Ref)
            r_raw = np.atleast_2d(np.asarray(info.get('ref_path_xy', []), dtype=np.float32)).reshape(-1, 2)
            if len(r_raw) < max_len_ref:
                pad = np.zeros((max_len_ref - len(r_raw), 2), dtype=np.float32)
                r_raw = np.vstack([r_raw, pad])
            ref_batch_list.append(r_raw)

            # 2. 处理左边界 (Left)
            l_raw = np.atleast_2d(np.asarray(info.get('static_road_left', []), dtype=np.float32)).reshape(-1, 2)
            if len(l_raw) < max_len_left:
                pad = np.zeros((max_len_left - len(l_raw), 2), dtype=np.float32)
                l_raw = np.vstack([l_raw, pad])
            left_batch_list.append(l_raw)

            # 3. 处理右边界 (Right)
            r_raw_road = np.atleast_2d(np.asarray(info.get('static_road_right', []), dtype=np.float32)).reshape(-1, 2)
            if len(r_raw_road) < max_len_right:
                pad = np.zeros((max_len_right - len(r_raw_road), 2), dtype=np.float32)
                r_raw_road = np.vstack([r_raw_road, pad])
            right_batch_list.append(r_raw_road)

            raw_points = np.atleast_2d(np.asarray(info.get('ref_path_xy', []), dtype=np.float32)).reshape(-1, 2)

            if len(raw_points) == 0:
                # 处理空路径情况
                processed_line = np.zeros((max_len, 4), dtype=np.float32)
            else:
                N = len(raw_points)
                xs = raw_points[:, 0]
                ys = raw_points[:, 1]

                # 1. 计算航向角 phi (atan2(dy, dx))
                # 使用差分计算切线方向
                dx = np.diff(xs, prepend=xs[0])
                dy = np.diff(ys, prepend=ys[0])
                phis = np.arctan2(dy, dx)

                # 2. 计算期望速度 v_lon
                # 策略 A: 恒定速度 (例如 5.0 m/s)
                vs = np.full(N, 5.0, dtype=np.float32)

                # 策略 B: 根据曲率或预设限速调整 (这里简化为常数)
                # vs = ...

                # 3. 堆叠成 [N, 4] -> [x, y, v, phi]
                processed_line = np.stack([xs, ys, vs, phis], axis=1)

                # Padding 到 max_len
                if N < max_len:
                    pad = np.zeros((max_len - N, 4), dtype=np.float32)
                    processed_line = np.vstack([processed_line, pad])

            cleaned_refs.append(processed_line)

        # 转为 Tensor [Batch, Max_Len, 2]
        current_ref_xy = torch.from_numpy(np.stack(cleaned_refs)).to(self.device).float()
        current_road_left_xy = torch.from_numpy(np.stack(left_batch_list)).to(self.device).float()
        current_road_right_xy = torch.from_numpy(np.stack(right_batch_list)).to(self.device).float()


        # --- 预测循环 (模仿 loss.py 的 run_step 逻辑) ---
        trajectory_actions = []
        trajectory_states = []  # 存储每一步的完整状态用于计算 Loss

        # 初始化累积 Loss (虽然通常最后一起算，但这里为了结构清晰先收集)
        # 我们采用 loss.py 的方式：在循环外一次性计算所有 step 的 loss

        current_ego = ego_state.clone()  # [B, 1, 6]
        current_other = other_states.clone()  # [B, N, 6]
        current_all = state_tensor.clone()
        # 预分配参考状态索引 (可选优化，这里在循环内动态计算)

        for t in range(self.horizon):
            # 1. Actor 输出动作 [Batch, 2]
            # 注意：input 可能需要展平或保持特定维度，视 actor 定义而定
            # 假设 actor 接受 [B, Dim_Flat] 或 [B, 1, Dim]
            # input_state_flat = current_ego.view(current_ego.size(0), -1)
            # 如果需要融合其他状态，需在此处 concat，参照 unpack_tensor 的逆过程或 actor 输入要求
            # 这里假设 actor 只需要 ego_state 或者你已经把全部状态拼好传入了
            # 修正：通常 MPC/RL 的 actor 需要观测向量。
            # 让我们重构 input_tensor 类似于 loss.py 的 batch_states
            # 这里简化处理：假设 self.actor 能够处理当前的状态表示

            action = self.actor(current_all)  # [B, 2]
            trajectory_actions.append(action)

            # 2. 动力学推演 (Batch 化)
            # dynamics_model 需支持 [B, 1, 6] 输入
            next_ego_state = self.dynamics_model.forward(current_ego, action)

            # 1. 获取自车位置 (Batch,)
            ego_x = next_ego_state[:, 0, 0]
            ego_y = next_ego_state[:, 0, 1]

            # 2. 预测其他车 (Batch, N, 6)
            # 确保 self.dt 是 tensor 或 float
            next_other_state = self.predict_other_next_batch(current_other, self.dt)

            # 3. 获取动态参考点 (Batch, 4)
            # current_ref_xy 形状应为 [Batch, Path_Len, 4] (包含 x,y,v,yaw)
            # 如果只有 xy，请调整 get_ref_observation_torch 或预处理数据
            next_ref_state = self.get_ref_observation_torch(ego_x, ego_y, current_ref_xy)

            # 4. 获取道路观测 (Batch, 2)

            # 调用新版函数，传入左右两个张量
            next_road_state = self.get_road_observation_torch(
                ego_x,
                ego_y,
                current_road_left_xy,
                current_road_right_xy
            )

            # 4. 组装完整状态用于下一步输入和 Loss 计算
            # 格式: [Ego, Other, Road, Ref]
            # 注意维度对齐：Other 可能是 [B, N, 6] -> flatten
            next_state_flat = torch.cat([
                next_ego_state.view(next_ego_state.size(0), -1),
                next_other_state.view(next_other_state.size(0), -1),
                next_road_state,
                next_ref_state
            ], dim=1)

            trajectory_states.append(next_state_flat)

            # 更新状态
            current_ego = next_ego_state
            current_other = next_other_state
            current_all = next_state_flat
        # --- 计算 Loss (模仿 loss.py 的 get_oneStepLoss 累加逻辑) ---
        # Stack: [Horizon, Batch, Dim] -> 转置为 [Batch, Horizon, Dim] 方便计算
        states_traj = torch.stack(trajectory_states).transpose(0, 1)  # [B, H, Dim_State]
        actions_traj = torch.stack(trajectory_actions).transpose(0, 1)  # [B, H, 2]

        # 调用你原有的损失计算函数，但需确保它支持 Batch 输入
        # 原代码: instant_cost = self.compute_instant_cost(states_traj, actions_traj)
        # 需确保 compute_instant_cost 内部是向量化操作 (没有 python for 循环遍历 batch)
        instant_cost = self.compute_instant_cost(states_traj, actions_traj)  # 返回 [B] 或 Scalar
        instant_penalty = self.compute_constraints(states_traj)  # 返回 [B] 或 Scalar

        # print(f'instant_penalty约束：{instant_penalty}')

        # 如果返回的是 [B]，则求 mean；如果是标量直接用
        if instant_cost.dim() > 0:
            actor_loss = (instant_cost + self.init_penalty * instant_penalty).mean()
        else:
            actor_loss = instant_cost + self.init_penalty * instant_penalty

        # print(f'# 永远不变: {actor_loss}") ')

        # --- Critic 准备 ---
        # Critic 输入：初始状态 [B, Dim]
        # Critic 目标：Rollout 的总 Loss (detach)
        critic_inputs = state_tensor  # [B, Dim]
        critic_targets = (instant_cost + self.init_penalty * instant_penalty).detach()  # [B]

        # --- 反向传播 ---
        # 1. Actor Update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        # print(f"Actor Grad Norm: {actor_grad_norm}")
        # print(f"Instant Cost: {instant_cost.item()}")
        # print(f"Instant Penalty: {instant_penalty.item()}")
        # print(f"Actions Mean: {actions_traj.mean().item()}")
        self.actor_optimizer.step()

        # 2. Critic Update
        pred = self.critic(critic_inputs)  # [B, 1]
        # 确保 target 维度匹配
        if critic_targets.dim() == 0:
            # 如果是标量，扩展为 [Batch, 1]
            critic_targets = critic_targets.unsqueeze(0).unsqueeze(0).expand(pred.shape[0], 1)
        elif critic_targets.dim() == 1:
            # 如果是 [Batch]，变为 [Batch, 1]
            critic_targets = critic_targets.unsqueeze(1)

        critic_loss = F.mse_loss(pred, critic_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_grad_norm": float(actor_grad_norm.item()) if torch.is_tensor(actor_grad_norm) else float(
                actor_grad_norm)
        }

    # 辅助函数提示：你需要确保以下函数支持 Batch 输入
    # 1. self.predict_other_next_batch(other_states, dt)
    # 2. self.get_road_observation_torch(ego_x[B], ego_y[B], road_maps[B, L, 2])
    # 3. self.get_ref_observation_torch(ego_x[B], ego_y[B], ref_paths[B, L, 2])
    # 4. self.compute_instant_cost(states[B, H, D], actions[B, H, 2]) -> Tensor[B] or Scalar
    # 5. self.compute_constraints(states[B, H, D]) -> Tensor[B] or Scalar

    def predict_other_next_batch(self, other_states, dt):
        """
        批量预测其他车辆下一时刻状态 (匀速模型)
        支持输入形状: [B, N, N_npc, 6]

        Parameters:
        ---
        other_states : torch.Tensor
                       形状: [B, N, N_npc, 6]
                       - B: Batch size
                       - N: 场景数/假设数/主车数 (新增维度)
                       - N_npc: 每场景下的NPC数量
                       - 6: 状态维度 [x, y, vx, vy, psi, omega]

                       兼容处理: 如果输入是 [B, N_npc, 6] (即 N=1 的压缩形式)，
                       可以尝试自动unsqueeze，但为了严谨，本函数主要适配 4D 输入。

        dt : float 或 torch.Tensor (支持广播)

        Returns:
        ---
        next_other_states : torch.Tensor [B, N, N_npc, 6]
        """

        # 1. 维度检查与预处理
        if other_states.dim() == 3:
            # 可选兼容：如果用户传入 [B, N_npc, 6]，自动视为 N=1
            # 将其转换为 [B, 1, N_npc, 6]
            other_states = other_states.unsqueeze(1)

        if other_states.dim() != 4 or other_states.shape[3] != 6:
            raise ValueError(
                f"输入 Tensor 形状必须为 [B, N, N_npc, 6] (或兼容的 [B, N_npc, 6])，当前形状为 {other_states.shape}")

        B, N, n_npc, _ = other_states.shape

        # 2. 提取分量
        # 注意索引方式：[:, :, :, index] 对应 [B, N, N_npc, Feature]
        x = other_states[:, :, :, 0]  # [B, N, N_npc]
        y = other_states[:, :, :, 1]
        vx = other_states[:, :, :, 2]
        vy = other_states[:, :, :, 3]
        psi = other_states[:, :, :, 4]
        omega = other_states[:, :, :, 5]

        # 3. 运动学预测 (匀速模型)
        # 假设 vx, vy 为全局坐标系下的速度
        # dt 会自动广播到 [B, N, N_npc]
        x_next = x + dt * vx
        y_next = y + dt * vy

        # 假设航向角、速度大小不变 (匀速直线)
        psi_next = psi
        vx_next = vx
        vy_next = vy
        omega_next = omega  # 保持角速度不变

        # 4. 堆叠回 [B, N, N_npc, 6]
        # dim=3 表示在第4个维度上堆叠 (0-indexed: 0,1,2,3)
        next_other_states = torch.stack([x_next, y_next, vx_next, vy_next, psi_next, omega_next], dim=3)

        return next_other_states

    def get_ref_observation_torch(self, ego_x, ego_y, ref_paths):
        """
        批量获取当前自车位置对应的最近参考点状态
        严格遵循公式: [px, py, v_lon, 0, phi, 0]

        Parameters:
        ---
        ego_x : torch.Tensor [Batch]
        ego_y : torch.Tensor [Batch]
        ref_paths : torch.Tensor [Batch, Path_Len, 4]
                    必须包含: [x, y, v_lon, phi]
                    (如果只有 x,y，需要在外部预处理补充 v 和 phi)

        Returns:
        ---
        ref_obs : torch.Tensor [Batch, 6]
                  格式: [ref_x, ref_y, ref_v_lon, 0, ref_phi, 0]
        """
        batch_size = ego_x.shape[0]

        # --- 步骤 1: 找到最近的参考点索引 ---

        # 构建自车位置 [B, 2, 1]
        ego_pos = torch.stack([ego_x, ego_y], dim=1).unsqueeze(-1)

        # 提取参考线位置部分 [B, L, 2] -> [B, 2, L]
        # 假设 ref_paths 最后维是 [x, y, v, phi]
        ref_pos_xy = ref_paths[:, :, :2].transpose(1, 2)

        # 计算距离并找到最近点索引 [B]
        diff = ref_pos_xy - ego_pos
        dist_sq = torch.sum(diff ** 2, dim=1)
        nearest_idx = torch.argmin(dist_sq, dim=1)  # [B]

        # --- 步骤 2: 提取最近点的 [x, y, v, phi] ---

        # 扩展索引以 gather 所有 4 个维度: [B, 1, 4]
        idx_expanded = nearest_idx.view(-1, 1, 1).expand(-1, 1, ref_paths.shape[2])

        # 获取最近点状态 [B, 4] -> [x, y, v, phi]
        nearest_state_4d = torch.gather(ref_paths, 1, idx_expanded).squeeze(1)

        ref_x = nearest_state_4d[:, 0]  # [B]
        ref_y = nearest_state_4d[:, 1]  # [B]
        ref_v = nearest_state_4d[:, 2]  # [B]
        ref_phi = nearest_state_4d[:, 3]  # [B]

        # --- 步骤 3: 组装成公式要求的 6 维向量 ---
        # 公式: [px, py, v_lon, 0, phi, 0]

        zero_tensor = torch.zeros_like(ref_x)  # [B] 全 0

        ref_obs = torch.stack([
            ref_x,  # 0: px
            ref_y,  # 1: py
            ref_v,  # 2: v_lon
            zero_tensor,  # 3: 0 (固定值)
            ref_phi,  # 4: phi (yaw)
            zero_tensor  # 5: 0 (固定值，对应 omega 或类似)
        ], dim=1)  # [B, 6]

        return ref_obs

    def get_road_observation_torch(self, ego_x, ego_y, road_left_xy, road_right_xy):
        """
        批量获取道路边界观测
        Parameters:
        ---
        ego_x, ego_y : [Batch]
        road_left_xy : [Batch, N_left, 2]
        road_right_xy: [Batch, N_right, 2]

        Returns:
        ---
        road_obs : [Batch, 4]
                   格式: [dist_to_left, dist_to_right, left_nearest_x, right_nearest_x]
                   或者仅返回距离用于约束计算
        """

        def get_nearest_dist(ego_x, ego_y, road_map):
            # road_map: [B, L, 2]
            ego_pos = torch.stack([ego_x, ego_y], dim=1).unsqueeze(-1)  # [B, 2, 1]
            road_pos = road_map.transpose(1, 2)  # [B, 2, L]

            diff = road_pos - ego_pos
            dist_sq = torch.sum(diff ** 2, dim=1)  # [B, L]

            # 找到最近距离的平方
            min_dist_sq, _ = torch.min(dist_sq, dim=1)  # [B]

            # 返回实际距离 (加一个小值防止除零或梯度爆炸，可选)
            return torch.sqrt(min_dist_sq + 1e-6)

        # 计算到左边界的最近距离
        dist_left = get_nearest_dist(ego_x, ego_y, road_left_xy)

        # 计算到右边界的最近距离
        dist_right = get_nearest_dist(ego_x, ego_y, road_right_xy)


        # 组合观测：[距离左边界，距离右边界]
        zeros = torch.zeros_like(dist_left)
        # 你也可以在这里计算横向误差：(dist_right - dist_left) / 2 等
        road_obs = torch.stack([dist_left, dist_right,zeros,zeros,zeros,zeros], dim=1)  # [Batch, 2]

        return road_obs

    def compute_instant_cost(self, states_traj, actions_traj):
        """
        APM 框架下的即时成本 (主要关注控制平滑度)
        跟踪误差已移至 constraint 中处理
        """
        s_ego, _, _, s_ref = self.unpack_tensor(data=states_traj)

        # 统一维度 [B, T, Dim]
        if s_ego.dim() == 2:
            s_ego = s_ego.unsqueeze(1)
            s_ref = s_ref.unsqueeze(1)
            actions_traj = actions_traj.unsqueeze(1) if actions_traj.dim() == 2 else actions_traj

        batch_size, time_horizon, _ = s_ego.shape

        # --- 1. 准备权重 (修复广播问题) ---
        # Q 用于状态偏差 (如果还想保留一点跟踪惩罚作为软目标，可以留小权重)
        # 但在严格 APM 中，这里可以只放控制权重，或者 Q 设得很小
        q_weights = torch.from_numpy(np.diag(self.Q_matrix).copy()).to(self.device).float()
        r_weights = torch.from_numpy(np.diag(self.R_matrix).copy()).to(self.device).float()

        # Reshape 为 [1, 1, Dim] 以便正确广播
        q_weights = q_weights.view(1, 1, -1)
        r_weights = r_weights.view(1, 1, -1)

        # --- 2. 状态跟踪成本 (可选：设为很小，主要靠约束) ---
        # 如果完全依赖约束，这部分可以注释掉或权重减半
        diff = s_ref - s_ego
        # 角度特殊处理
        diff_psi = torch.atan2(torch.sin(diff[..., 4]), torch.cos(diff[..., 4]))
        diff_processed = diff.clone()
        diff_processed[..., 4] = diff_psi

        # 加权平方和
        state_cost = (q_weights * (diff_processed ** 2)).sum(dim=-1).mean()

        # --- 3. 控制消耗成本 (主要优化目标) ---
        control_cost = (r_weights * (actions_traj ** 2)).sum(dim=-1).mean()

        # 总成本
        return state_cost + control_cost

    def compute_constraints(self, states_traj):
        """
        计算约束违反值 h(x)。
        根据 Guan 论文，应将跟踪误差和安全距离都作为约束。
        返回标量或 [Batch,] 张量，表示违反程度。
        """
        s_ego, s_other, s_road, s_ref = self.unpack_tensor(data=states_traj)

        # 确保维度为 [B, T, Dim]
        if s_ego.dim() == 2:
            s_ego = s_ego.unsqueeze(1)
            s_ref = s_ref.unsqueeze(1)
            if s_other is not None and s_other.dim() == 2: s_other = s_other.unsqueeze(1)
            if s_road is not None and s_road.dim() == 2: s_road = s_road.unsqueeze(1)

        constraints_list = []

        # --- 1. 跟踪误差约束 (最关键！防止转圈) ---
        # 定义：横向误差 + 航向误差 必须小于阈值
        # 计算横向误差 (Lateral Error)
        dx = s_ref[..., 0] - s_ego[..., 0]
        dy = s_ref[..., 1] - s_ego[..., 1]
        psi_ref = s_ref[..., 4]

        # 投影到参考系法向量
        lat_err = -dx * torch.sin(psi_ref) + dy * torch.cos(psi_ref)

        # 计算航向误差
        psi_ego = s_ego[..., 4]
        head_err = torch.atan2(torch.sin(psi_ego - psi_ref), torch.cos(psi_ego - psi_ref))

        # 综合跟踪误差指标 (加权)
        tracking_err = torch.abs(lat_err) + 0.5 * torch.abs(head_err)

        # 约束公式：h_track = tracking_err - threshold
        # 如果 err > threshold, h > 0 (违规); 否则 h <= 0 (合规)
        track_threshold = 0.5  # 米 + 弧度混合阈值，可根据实际情况调整
        g_track = tracking_err - track_threshold

        # 使用 ReLU 获取违反量 (只惩罚违规部分)
        violation_track = F.softplus(g_track * 5.0).mean()
        constraints_list.append(violation_track * 10.0)  # 乘以系数放大梯度信号

        # --- 2. 车辆碰撞约束 (保持原有逻辑，但修复维度) ---
        if s_other is not None and s_other.numel() > 0:
            # s_other shape: [B, T, N_other, Dim]
            # s_ego shape: [B, T, 1, Dim] (需要 unsqueeze 以广播)
            rel_pos_car = s_ego.unsqueeze(2) - s_other

            # 仅取 x, y 计算距离 (索引 0, 1)
            M_xy = torch.from_numpy(np.diag(self.M_matrix)[:2].copy()).to(self.device).float().view(1, 1, 1, -1)
            dist_sq_car = (rel_pos_car[..., :2] * M_xy * rel_pos_car[..., :2]).sum(dim=-1)  # [B, T, N]

            g_car = dist_sq_car - self.other_car_min_distance ** 2
            # 只要有一辆车距离过近，就算违规
            violation_car = F.softplus(-g_car).max(dim=-1)[0].mean()
            constraints_list.append(violation_car)

        # --- 3. 道路边界约束 (视 s_road 定义而定，若 s_road 是中心线则类似跟踪误差) ---
        # 假设 s_road 是左右边界点，逻辑同上，此处简化略过，重点在跟踪误差

        total_constraint = sum(constraints_list)
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

        return ego_state, other_states, road_state, ref_state
