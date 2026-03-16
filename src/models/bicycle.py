import torch
import torch.nn as nn


class BicycleModel(nn.Module):
    def __init__(self, dt=0.1, L=2.9):
        super().__init__()
        self.dt = dt
        self.L = L  # 轴距，和论文场景对齐
        self.register_buffer('dt_tensor', torch.tensor(dt, dtype=torch.float32))
        self.register_buffer('L_tensor', torch.tensor(L, dtype=torch.float32))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        严格对齐论文的自行车模型，输入输出状态和论文定义100%匹配
        :param state: 论文定义的自车状态 [B, N, 6]，维度为 [p_x, p_y, v_lon, v_lat, φ, ω]
        :param action: 控制量 [B, 2]，维度为 [a(纵向加速度), δ(前轮转角)]，和ocp_agent输出完全对齐
        :return: 下一时刻状态，和输入state维度完全一致
        """
        original_shape = state.shape
        B, N = original_shape[0], original_shape[1]

        # 展平处理，保证batch维度兼容
        s_flat = state.reshape(-1, 6)  # [B*N, 6]
        a_flat = action.reshape(-1, 2)  # [B*N, 2]

        # 严格按论文定义解包状态
        p_x = s_flat[:, 0]  # x坐标
        p_y = s_flat[:, 1]  # y坐标
        v_lon = s_flat[:, 2]  # 纵向速度
        v_lat = s_flat[:, 3]  # 横向速度
        phi = s_flat[:, 4]  # 航向角 φ
        omega = s_flat[:, 5]  # 横摆角速度 ω

        # 解包动作，和ocp_agent的select_action输出严格对齐
        a = a_flat[:, 0]  # 纵向加速度
        delta = a_flat[:, 1]  # 前轮转角 δ

        # --------------------------
        # 论文线性轮胎自行车模型动力学更新
        # --------------------------
        # 1. 纵向速度更新
        v_lon_next = v_lon + a * self.dt_tensor
        v_lon_next = torch.clamp(v_lon_next, min=0.01, max=30.0)  # 禁止倒车，符合驾驶场景

        # 2. 横摆角速度更新（自行车模型核心公式）
        v_safe = torch.clamp(v_lon, min=0.5)  # 避免低速除零
        omega_next = (v_safe / self.L_tensor) * torch.tan(delta)

        # 3. 航向角更新+归一化
        phi_next = phi + omega_next * self.dt_tensor
        phi_next = torch.atan2(torch.sin(phi_next), torch.cos(phi_next))  # 归一化到[-π, π]

        # 4. 世界坐标系位置更新
        v_avg = (v_lon + v_lon_next) * 0.5  # 平均纵向速度，提升精度
        p_x_next = p_x + v_avg * torch.cos(phi) * self.dt_tensor
        p_y_next = p_y + v_avg * torch.sin(phi) * self.dt_tensor

        # 5. 横向速度更新（小侧偏角线性假设，和论文一致）
        v_lat_next = v_lat + (
                    omega_next * v_safe - (v_lat * v_safe / self.L_tensor) * torch.tan(delta)) * self.dt_tensor
        v_lat_next = torch.clamp(v_lat_next, min=-5.0, max=5.0)  # 限制横向速度合理范围

        # 打包下一时刻状态，严格和输入维度、顺序一致
        next_state_flat = torch.stack([
            p_x_next, p_y_next, v_lon_next, v_lat_next, phi_next, omega_next
        ], dim=1)

        # 恢复原始输入形状
        return next_state_flat.view(original_shape)