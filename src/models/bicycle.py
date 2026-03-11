import torch
import torch.nn as nn
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
                 格式: [accel, steer]
                 accel: 纵向加速度指令 (m/s^2) (沿车身坐标系)
                 steer: 前轮转角指令 (rad)

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

        accel_cmd = action[:, 1]
        steer_cmd = action[:, 0]

        accel = torch.clamp(accel_cmd, self.min_accel, self.max_accel)
        steer = torch.clamp(steer_cmd, -self.max_steer, self.max_steer)

        v_mag = torch.hypot(vx, vy)
        v_mag = torch.clamp(v_mag, min=1e-2)
        tan_steers = torch.tan(steer)
        omega_kin = (v_mag * tan_steers) / self.L_tensor

        # --- 位置更新 ---
        x_next = x + vx * self.dt_tensor
        y_next = y + vy * self.dt_tensor

        v_mag_next = v_mag + accel * self.dt_tensor
        # 防止速度为负 (倒车逻辑需单独处理，这里假设向前)
        v_mag_next = torch.clamp(v_mag_next, min=0.0)

        psi_next = psi + omega_kin * self.dt_tensor

        vx_next = v_mag_next * torch.cos(psi_next)
        vy_next = v_mag_next * torch.sin(psi_next)

        omega_next = omega_kin

        next_state = torch.stack([x_next, y_next, vx_next, vy_next, psi_next, omega_next], dim=1)
        return next_state