import torch
import torch.nn as nn

class BicycleModel(nn.Module):
    """
    严格对齐论文的线性轮胎自行车模型
    输入动作：[加速度a, 前轮转角δ]，范围[-1,1]，内部自动缩放
    """
    def __init__(self, dt=0.1, L=2.9):
        super().__init__()
        self.dt = dt
        self.L = L  # 轴距
        self.register_buffer('dt_tensor', torch.tensor(dt, dtype=torch.float32))
        self.register_buffer('L_tensor', torch.tensor(L, dtype=torch.float32))

        # 【关键修复】动作缩放范围，和论文完全对齐
        self.accel_scale = 2.25  # [-1,1] → [-3, 1.5] m/s²
        self.accel_bias = -0.75
        self.steer_scale = 0.4  # [-1,1] → [-0.4, 0.4] rad

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        :param state: 自车状态 [B, 1, 6] → [x,y,v_lon,v_lat,phi,omega]
        :param action: 网络输出动作 [B, 2] → [a_norm, δ_norm]
        :return: 下一时刻状态，和输入维度完全一致
        """
        original_shape = state.shape
        B = original_shape[0]
        s_flat = state.view(-1, 6)  # [B, 6]
        a_flat = action.view(-1, 2)  # [B, 2]

        # 解包状态（严格对齐论文定义）
        p_x = s_flat[:, 0]
        p_y = s_flat[:, 1]
        v_lon = s_flat[:, 2]
        v_lat = s_flat[:, 3]
        phi = s_flat[:, 4]
        omega = s_flat[:, 5]

        # 【关键修复】动作缩放，正确映射到物理范围
        a_norm = a_flat[:, 0]
        delta_norm = a_flat[:, 1]
        a = a_norm * self.accel_scale + self.accel_bias  # [-1,1] → [-3, 1.5] m/s²
        delta = delta_norm * self.steer_scale  # [-1,1] → [-0.4, 0.4] rad

        # --------------------------
        # 自行车模型动力学更新
        # --------------------------
        # 1. 纵向速度更新
        v_lon_next = v_lon + a * self.dt_tensor
        v_lon_next = torch.clamp(v_lon_next, min=0.01, max=20.0)  # 允许0速，不倒车

        # 2. 横摆角速度更新
        v_safe = torch.clamp(v_lon, min=0.5)  # 避免低速除零
        omega_next = (v_safe / self.L_tensor) * torch.tan(delta)

        # 3. 航向角更新（归一化）
        phi_next = phi + omega_next * self.dt_tensor
        phi_next = torch.atan2(torch.sin(phi_next), torch.cos(phi_next))

        # 4. 位置更新
        v_avg = (v_lon + v_lon_next) * 0.5
        p_x_next = p_x + v_avg * torch.cos(phi) * self.dt_tensor
        p_y_next = p_y + v_avg * torch.sin(phi) * self.dt_tensor

        # 5. 横向速度更新（小侧偏角假设）
        v_lat_next = v_lat + (omega_next * v_safe - (v_lat * v_safe / self.L_tensor) * torch.tan(delta)) * self.dt_tensor
        v_lat_next = torch.clamp(v_lat_next, min=-5.0, max=5.0)

        # 打包下一状态，严格和输入顺序一致
        next_state_flat = torch.stack([
            p_x_next, p_y_next, v_lon_next, v_lat_next, phi_next, omega_next
        ], dim=1)

        return next_state_flat.view(original_shape)