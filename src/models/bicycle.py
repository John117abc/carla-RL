import torch
import torch.nn as nn


class BicycleModel(nn.Module):
    def __init__(self, dt=0.1, L=2.5):
        super(BicycleModel, self).__init__()
        self.dt = dt
        self.L = L
        self.register_buffer('dt_tensor', torch.tensor(dt))
        self.register_buffer('L_tensor', torch.tensor(L))

    def forward(self, state, action):
        original_shape = state.shape
        B, N = original_shape[0], original_shape[1]

        s_flat = state.view(-1, 6)
        if action.dim() == 2:
            a_flat = action.unsqueeze(1).expand(-1, N, -1).reshape(-1, 2)
        else:
            a_flat = action.reshape(-1, 2)

        x = s_flat[:, 0]
        y = s_flat[:, 1]
        psi = s_flat[:, 2]
        v = s_flat[:, 3]
        delta = s_flat[:, 4]
        extra = s_flat[:, 5]

        accel = a_flat[:, 0]
        steer = a_flat[:, 1]

        # --- 1. 速度更新 (允许减速到 0，但不要强制为正，除非物理引擎限制) ---
        # 建议：只限制最大值，最小值设为 0 (禁止倒车) 或 -1 (允许微倒车)
        # ❌ 错误：v_next = torch.clamp(v_next, 0.1, 30.0) -> 导致梯度在 v<0.1 时消失
        v_next = v + accel * self.dt_tensor
        # ✅ 修复：允许速度归零，但不要出现巨大的梯度截断
        # 如果确实不能倒车，设为 0.0。如果想让训练更稳定，可以设为 -0.5 允许微小倒车
        v_next = torch.clamp(v_next, 0.01, 30.0)

        # --- 2. 转向计算 (保留你的 v_safe 技巧，这很好) ---
        # 注意：这里用 v 还是 v_next 都可以，关键是用 v_safe 保证梯度
        # 为了物理一致性，通常用当前速度 v 计算角速度，或者用平均速度
        v_safe = torch.clamp(v, min=0.5)
        d_psi = (v_safe / self.L_tensor) * torch.tan(steer)
        psi_next = psi + d_psi * self.dt_tensor

        # --- 3. 位移计算 (关键修复) ---
        # ❌ 原代码：x_next = x + v * cos(psi) * dt
        # 问题：如果 v=0 且 accel>0，这一步 x 不变，虽然 v_next 变了，但第一帧没位移。
        # 这会导致网络认为"加油也没用"。

        # ✅ 修复：使用更新后的速度 v_next (欧拉前向) 或者 平均速度 (梯形积分)
        # 方法 A: 使用 v_next (假设加速度瞬间生效)
        # x_next = x + v_next * torch.cos(psi) * self.dt_tensor

        # 方法 B (推荐): 使用平均速度 (v + v_next) / 2，物理更准确，梯度更平滑
        v_avg = (v + v_next) * 0.5
        x_next = x + v_avg * torch.cos(psi) * self.dt_tensor
        y_next = y + v_avg * torch.sin(psi) * self.dt_tensor

        delta_next = torch.clamp(steer, -0.5, 0.5)

        next_state_flat = torch.stack([
            x_next, y_next, psi_next, v_next, delta_next, extra
        ], dim=1)

        return next_state_flat.view(original_shape)