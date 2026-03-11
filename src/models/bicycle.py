import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BicycleModel(nn.Module):
    def __init__(self, dt=0.1, L=2.9, max_steer=0.5, max_accel=3.0, min_accel=-5.0, max_steer_rate=0.5):
        """
        改进版自行车模型：状态包含 vx, vy, omega

        dt: 时间步长 (秒)
        L: 车辆轴距 (米)
        max_steer: 最大转向角 (rad)
        max_accel/min_accel: 加减速限制 (m/s^2)
        max_steer_rate: 最大转角速度 (rad/s)，用于限制 omega 的变化率或作为动作约束参考
        """
        super(BicycleModel, self).__init__()
        self.dt = dt
        self.L = L
        self.max_steer = max_steer
        self.max_accel = max_accel
        self.min_accel = min_accel

        # 注册 buffer 以支持 GPU
        self.register_buffer('dt_tensor', torch.tensor(dt))
        self.register_buffer('L_tensor', torch.tensor(L))
        self.register_buffer('pi_tensor', torch.tensor(math.pi))

    def forward(self, state, action):
        """
        可微动力学推演一步: s_{t+1} = f(s_t, u_t)

        Parameters:
        ---
        state : torch.Tensor [Batch, 6]
                格式: [x, y, vx, vy, psi, omega]
                x, y: 全局坐标 (m)
                vx, vy: 全局坐标系下的速度分量 (m/s)
                psi: 航向角 (rad)
                omega: 横摆角速度 (rad/s)

        action : torch.Tensor [Batch, 2]
                 格式: [steer_norm, accel_norm]（与 ActorNet 保持一致）
                 steer_norm / accel_norm: 归一化动作，范围通常在 [-1, 1]

        Returns:
        ---
        next_state : torch.Tensor [Batch, 6]
                     格式: [x, y, vx, vy, psi, omega]
        """
        # 确保输入维度为 [Batch, Dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        x = state[:, 0]
        y = state[:, 1]
        vx = state[:, 2]
        vy = state[:, 3]
        psi = state[:, 4]
        omega = state[:, 5]

        # 环境观测里 yaw 常为角度制；动力学内部统一用弧度制
        psi = torch.where(torch.abs(psi) > self.pi_tensor, psi * self.pi_tensor / 180.0, psi)

        steer_norm = torch.tanh(action[:, 0])
        accel_norm = torch.tanh(action[:, 1])

        # 用平滑仿射映射替代 clamp，避免动作饱和区梯度为 0
        steer = steer_norm * self.max_steer
        accel = self.min_accel + 0.5 * (accel_norm + 1.0) * (self.max_accel - self.min_accel)

        v_mag = torch.hypot(vx, vy)
        v_mag = torch.sqrt(v_mag * v_mag + 1e-4)
        tan_steers = torch.tan(steer)
        omega_kin = (v_mag * tan_steers) / self.L_tensor

        # --- 位置更新 ---
        x_next = x + vx * self.dt_tensor
        y_next = y + vy * self.dt_tensor

        v_mag_next_raw = v_mag + accel * self.dt_tensor
        # softplus 避免 clamp 造成的梯度截断
        v_mag_next = F.softplus(v_mag_next_raw, beta=5.0)

        psi_next = psi + omega_kin * self.dt_tensor

        vx_next = v_mag_next * torch.cos(psi_next)
        vy_next = v_mag_next * torch.sin(psi_next)

        omega_next = omega_kin

        next_state = torch.stack([x_next, y_next, vx_next, vy_next, psi_next, omega_next], dim=1)
        return next_state
