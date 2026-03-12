import torch
import numpy as np

# ✅ 修改：作为一个普通类或直接使用函数
class BicycleModel:
    def __init__(self, dt=0.1, L=2.5):
        self.dt = dt
        self.L = L  # 轴距

    def forward(self, state, action):
        """
        纯物理公式计算，不涉及任何 nn.Parameter
        state: [B, 1, 6] (x, y, psi, v, delta, ...) 或类似结构
        action: [B, 2] (accel, steer)
        """
        # 确保输入是 tensor
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)

        # 解包状态 (假设 state 结构: x, y, psi, v, ...)
        # 请根据你实际的 state 维度调整索引
        # 假设 state shape: [B, 1, 6], action shape: [B, 2]
        # 这里需要 squeeze 掉中间的维度以匹配运算
        s = state.squeeze(1)  # [B, 6]
        a = action  # [B, 2]

        x = s[:, 0]
        y = s[:, 1]
        psi = s[:, 2]
        v = s[:, 3]
        # delta = s[:, 4] # 如果需要当前舵角

        accel = a[:, 0]
        steer = a[:, 1]

        # --- 物理模型 ---
        # 1. 更新速度
        v_next = v + accel * self.dt
        v_next = torch.clamp(v_next, 1.0, 30.0)  # 限制速度范围

        # 2. 更新航向 (关键修复：处理 v=0 的奇点)
        # 原公式: d_psi = (v / L) * tan(steer)
        # 问题: 当 v=0 时，梯度为 0。
        # 修复: 添加一个极小值 epsilon，或者在低速时使用近似模型
        # 但更物理的做法是：即使 v=0，如果我们有加速度，下一帧 v 就不为 0 了。
        # 这里的梯度断裂是因为 v 是当前帧的，而 steer 影响的是下一帧的 v 方向？
        # 不，自行车模型中 steer 直接影响角速度。如果 v=0，车确实原地转不了弯（除非考虑动力学滑移，但这是运动学模型）。

        # 🔥 真正的解决方案：
        # 在运动学模型中，如果 v=0，转向确实无效。
        # 必须保证初始状态 v != 0，或者在 Loss 中加入对 "未来速度" 的激励。
        # 但为了梯度能传回，我们可以人为地在计算角速度时加一个 epsilon，
        # 或者更简单地：确保环境初始速度不为 0。

        # 临时 Hack (为了让梯度流动): 给 v 加一个极小值用于计算角速度，但不更新实际速度
        v_safe = torch.clamp(v, min=0.1)

        d_psi = (v_safe / self.L) * torch.tan(steer)
        psi_next = psi + d_psi * self.dt

        # 3. 更新位置
        x_next = x + v * torch.cos(psi) * self.dt
        y_next = y + v * torch.sin(psi) * self.dt

        # 4. 更新舵角 (假设动作直接控制舵角变化率或直接设定)
        # 假设 action[:, 1] 是目标舵角或舵角增量，这里简化为直接更新
        delta_next = steer  # 或者 delta + steer * dt

        # 重新打包
        # 保持与原代码一致的输出形状 [B, 1, 6]
        next_state = torch.stack([x_next, y_next, psi_next, v_next, delta_next, s[:, 5]], dim=1).unsqueeze(1)

        return next_state

    # 如果你之前有 batch 处理函数，也改成普通方法
    def predict_batch(self, states, actions):
        return self.forward(states, actions)