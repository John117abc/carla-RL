import torch
import torch.nn as nn

class BicycleModel(nn.Module):
    """
    严格对齐论文 Eq. 3 的带线性轮胎假设的动态自行车模型
    输入动作：[加速度a, 前轮转角δ] 物理量
    输出：下一时刻状态 [x, y, v_lon, v_lat, phi, omega]
    """
    def __init__(self, dt=0.1, L=2.9):
        super().__init__()
        self.dt = dt
        self.L = L  # 轴距
        # 论文 Table I 参数
        self.k_f = -155495.0  # 前轮侧偏刚度 [N/rad]
        self.k_r = -155495.0  # 后轮侧偏刚度 [N/rad]
        self.L_f = 1.19       # CG到前轴距离 [m]
        self.L_r = 1.46       # CG到后轴距离 [m]
        self.m = 1520.0       # 质量 [kg]
        self.I_z = 2640.0     # 转动惯量 [kg·m²]
        
        self.register_buffer('dt_tensor', torch.tensor(dt, dtype=torch.float32))
        self.register_buffer('L_tensor', torch.tensor(L, dtype=torch.float32))
        self.register_buffer('k_f_tensor', torch.tensor(self.k_f, dtype=torch.float32))
        self.register_buffer('k_r_tensor', torch.tensor(self.k_r, dtype=torch.float32))
        self.register_buffer('L_f_tensor', torch.tensor(self.L_f, dtype=torch.float32))
        self.register_buffer('L_r_tensor', torch.tensor(self.L_r, dtype=torch.float32))
        self.register_buffer('m_tensor', torch.tensor(self.m, dtype=torch.float32))
        self.register_buffer('I_z_tensor', torch.tensor(self.I_z, dtype=torch.float32))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        original_shape = state.shape
        B = original_shape[0]
        s_flat = state.view(-1, 6)  # [B, 6]
        a_flat = action.view(-1, 2)  # [B, 2]

        p_x, p_y, v_lon, v_lat, phi, omega = s_flat.unbind(dim=-1)
        a, delta = a_flat.unbind(dim=-1)

        # 避免低速除零保护
        v_safe = torch.clamp(v_lon, min=0.5)
        
        # 1. 纵向速度更新 (欧拉离散)
        v_lon_next = v_lon + self.dt_tensor * (a + v_lat * omega)

        # 2. 横向速度更新 (论文 Eq.3 分子分母)
        denom_lat = self.m_tensor * v_lon - self.dt_tensor * (self.k_f_tensor + self.k_r_tensor)
        # 防止分母为0
        denom_lat = torch.where(torch.abs(denom_lat) < 1e-6, 1e-6 * torch.sign(denom_lat), denom_lat)
        num_lat = self.m_tensor * v_lon * v_lat + self.dt_tensor * (
            (self.L_f_tensor * self.k_f_tensor - self.L_r_tensor * self.k_r_tensor) * omega
            - self.k_f_tensor * delta * v_lon
            - self.m_tensor * v_lon ** 2 * omega
        )
        v_lat_next = num_lat / denom_lat + self.dt_tensor * omega

        # 3. 横摆角速度更新 (论文 Eq.3)
        denom_yaw = self.dt_tensor * (self.L_f_tensor ** 2 * self.k_f_tensor + self.L_r_tensor ** 2 * self.k_r_tensor) - self.I_z_tensor * v_lon
        denom_yaw = torch.where(torch.abs(denom_yaw) < 1e-6, 1e-6 * torch.sign(denom_yaw), denom_yaw)
        num_yaw = -self.I_z_tensor * omega * v_lon - self.dt_tensor * (
            (self.L_f_tensor * self.k_f_tensor - self.L_r_tensor * self.k_r_tensor) * v_lat
            - self.L_f_tensor * self.k_f_tensor * delta * v_lon
        )
        omega_next = num_yaw / denom_yaw

        # 4. 航向角更新
        phi_next = phi + omega_next * self.dt_tensor
        phi_next = torch.atan2(torch.sin(phi_next), torch.cos(phi_next))

        # 5. 位置更新 (使用平均速度减少离散化误差)
        v_avg_lon = (v_lon + v_lon_next) * 0.5
        v_avg_lat = (v_lat + v_lat_next) * 0.5
        p_x_next = p_x + self.dt_tensor * (v_avg_lon * torch.cos(phi) - v_avg_lat * torch.sin(phi))
        p_y_next = p_y + self.dt_tensor * (v_avg_lon * torch.sin(phi) + v_avg_lat * torch.cos(phi))

        next_state_flat = torch.stack([p_x_next, p_y_next, v_lon_next, v_lat_next, phi_next, omega_next], dim=1)
        return next_state_flat.view(original_shape)
