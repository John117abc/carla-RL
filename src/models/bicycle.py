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

        # --- 1. 解包状态 ---
        x = state[:, 0]
        y = state[:, 1]
        vx = state[:, 2]
        vy = state[:, 3]
        psi = state[:, 4]
        omega = state[:, 5]

        # --- 2. 解包控制并做物理约束 ---
        accel_cmd = action[:, 1]
        steer_cmd = action[:, 0]

        # 物理限制 (Clamp 是可微的)
        accel = torch.clamp(accel_cmd, self.min_accel, self.max_accel)
        steer = torch.clamp(steer_cmd, -self.max_steer, self.max_steer)

        # --- 3. 中间变量计算 ---
        # 计算当前合速度 v (用于自行车模型公式)
        # v = sqrt(vx^2 + vy^2)
        # 注意：如果 vx, vy 很小，添加 epsilon 防止梯度爆炸
        v_mag = torch.sqrt(vx ** 2 + vy ** 2 + 1e-6)

        # 计算理论横摆角速度 omega_kin (基于自行车模型运动学)
        # omega = v * tan(delta) / L
        tan_steers = torch.tan(steer)
        omega_kin = (v_mag * tan_steers) / self.L_tensor

        # --- 4. 状态积分 (欧拉法) ---

        # A. 位置更新 (使用当前的 vx, vy)
        # x_{t+1} = x_t + vx * dt
        # y_{t+1} = y_t + vy * dt
        x_next = x + vx * self.dt_tensor
        y_next = y + vy * self.dt_tensor

        # B. 速度更新 (vx, vy)
        # 加速度 accel 是沿车身纵向的。我们需要将其转换到全局坐标系。
        # 假设侧向加速度主要由向心力提供，或者简化为仅由纵向加速度改变速度矢量方向？
        # 更精确的做法：
        # a_global_x = accel * cos(psi) - (v^2/L * tan(delta)) * sin(psi) ?
        # 为了保持简单且可微，我们通常假设：
        # 1. 纵向加速度 accel 改变速度大小。
        # 2. 转向改变速度方向 (通过 omega 旋转速度矢量)。

        # 方法：先计算下一时刻的速度大小 v_next_mag，再结合下一时刻的航向 psi_next 分解？
        # 或者：直接在全局系下施加力。
        # 这里采用一种常用且稳定的近似：
        # 将加速度投影到全局坐标，并考虑转向带来的速度矢量旋转效应。

        # 简单的运动学更新策略：
        # 1. 更新速度大小: v_new = v_old + accel * dt
        # 2. 更新航向: psi_new = psi_old + omega_kin * dt
        # 3. 假设车辆始终沿车头方向运动 (无侧滑)，则:
        #    vx_new = v_new * cos(psi_new)
        #    vy_new = v_new * sin(psi_new)
        # 这种假设在低速和正常驾驶下非常有效，且完全可微。

        v_mag_next = v_mag + accel * self.dt_tensor
        # 防止速度为负 (倒车逻辑需单独处理，这里假设主要向前，或允许负值代表倒车)
        # 如果允许倒车，v_mag_next 可以为负，cos/sin 会自动处理方向。

        # C. 航向角更新 (使用计算出的 omega_kin)
        psi_next = psi + omega_kin * self.dt_tensor
        # 角度归一化 [-pi, pi]
        psi_next = torch.remainder(psi_next + self.pi_tensor, 2 * self.pi_tensor) - self.pi_tensor

        # D. 速度分量更新 (基于新的速度大小和新航向)
        # 这隐含了侧向速度的产生是由于航向变化引起的
        vx_next = v_mag_next * torch.cos(psi_next)
        vy_next = v_mag_next * torch.sin(psi_next)

        # E. 横摆角速度更新
        # 这里的 omega 直接取运动学计算值 omega_kin，作为状态传递给下一步
        # 这样保证了状态的一致性
        omega_next = omega_kin

        # --- 5. 打包返回 ---
        # [x, y, vx, vy, psi, omega]
        next_state = torch.stack([x_next, y_next, vx_next, vy_next, psi_next, omega_next], dim=1)

        return next_state