import torch


def ellipse_min_dist_sq(
    ego_xy: torch.Tensor,        # [B, 2] 自车中心
    ego_phi: torch.Tensor,       # [B]    自车航向 (rad)
    other_xy: torch.Tensor,      # [B, 2] 周车中心
    other_phi: torch.Tensor,     # [B]    周车航向
    a: float = 2.25,
    b: float = 1.0,
    num_samples: int = 5         # 采样方向数（奇数，含中心方向）
) -> torch.Tensor:
    """
    计算两辆车椭圆之间的最小距离平方（多方向采样，取最小值）。
    返回 [B] 距离平方，重叠时返回 0。
    """
    B = ego_xy.shape[0]
    device = ego_xy.device

    d_vec = other_xy - ego_xy               # [B, 2]
    d_sq = (d_vec ** 2).sum(dim=1)           # [B]
    d = torch.sqrt(d_sq + 1e-8)              # [B]
    u = d_vec / d.unsqueeze(1)               # [B, 2] 中心连线方向

    # 在连线方向附近生成多个采样方向（在 2D 平面内旋转）
    angles = torch.linspace(-0.5, 0.5, num_samples, device=device)  # 度数范围可调
    # 将 u 旋转这些角度：u_rot = R(angle) * u
    # R = [[cos(a), -sin(a)], [sin(a), cos(a)]]
    cos_a = torch.cos(angles)  # [num_samples]
    sin_a = torch.sin(angles)  # [num_samples]
    # 扩展到 [B, num_samples, 2]
    u_exp = u.unsqueeze(1)                # [B, 1, 2]
    cos_a = cos_a.view(1, -1, 1)
    sin_a = sin_a.view(1, -1, 1)

    u_x = u_exp[..., 0:1]  # [B, 1, 1]
    u_y = u_exp[..., 1:2]

    u_rot_x = cos_a * u_x - sin_a * u_y   # [B, num_samples, 1]
    u_rot_y = sin_a * u_x + cos_a * u_y

    u_rot = torch.cat([u_rot_x, u_rot_y], dim=-1)  # [B, num_samples, 2]

    # 计算自车椭圆在各方向上的半径
    def ellipse_radius(u_local_x, u_local_y):
        # u_local_x, u_local_y: [B, num_samples]
        denom = torch.sqrt((a * u_local_y) ** 2 + (b * u_local_x) ** 2 + 1e-8)
        return a * b / denom

    # 自车：航向 phi_ego，旋转矩阵 (cos, sin; -sin, cos)
    cos_e = torch.cos(ego_phi).view(B, 1, 1)
    sin_e = torch.sin(ego_phi).view(B, 1, 1)
    u_ego_x = cos_e * u_rot[..., 0:1] + sin_e * u_rot[..., 1:2]   # [B, num_samples, 1]
    u_ego_y = -sin_e * u_rot[..., 0:1] + cos_e * u_rot[..., 1:2]
    r_ego = ellipse_radius(u_ego_x.squeeze(-1), u_ego_y.squeeze(-1))  # [B, num_samples]

    # 周车：航向 phi_other，对相反方向 -u_rot 计算半径
    cos_o = torch.cos(other_phi).view(B, 1, 1)
    sin_o = torch.sin(other_phi).view(B, 1, 1)
    u_oth_x = -(cos_o * u_rot[..., 0:1] + sin_o * u_rot[..., 1:2])
    u_oth_y = -(-sin_o * u_rot[..., 0:1] + cos_o * u_rot[..., 1:2])
    r_oth = ellipse_radius(u_oth_x.squeeze(-1), u_oth_y.squeeze(-1))  # [B, num_samples]

    # 每个采样方向上的间隙
    gap_samples = d.unsqueeze(1) - (r_ego + r_oth)   # [B, num_samples]
    gap_min, _ = torch.min(gap_samples, dim=1)       # [B]

    dist = torch.clamp(gap_min, min=0.0)
    return dist ** 2


def rect_min_dist_sq(center1, phi1, half_l1, half_w1, center2, phi2, half_l2, half_w2):
    """
    计算两个旋转矩形之间的最小距离平方（单对矩形）。
    参数均为 Tensor，支持批量（形状末尾维对齐）。
    """
    # 保证形状一致： (..., 2) 和 (..., )
    delta = center1 - center2  # (..., 2)
    cos1, sin1 = torch.cos(phi1), torch.sin(phi1)
    cos2, sin2 = torch.cos(phi2), torch.sin(phi2)
    # 每个矩形的两个局部轴（单位向量）
    axes1_x = torch.stack([cos1, sin1], dim=-1)  # (..., 2)
    axes1_y = torch.stack([-sin1, cos1], dim=-1)
    axes2_x = torch.stack([cos2, sin2], dim=-1)
    axes2_y = torch.stack([-sin2, cos2], dim=-1)

    # 候选分离轴：四个轴
    axes = [axes1_x, axes1_y, axes2_x, axes2_y]
    # 矩形半长、半宽
    r1_x, r1_y = half_l1, half_w1
    r2_x, r2_y = half_l2, half_w2

    max_gap = None
    for axis in axes:
        # 投影间隙
        proj_delta = torch.sum(delta * axis, dim=-1)
        # 矩形1投影半径
        abs_ax1_x = torch.abs(torch.sum(axis * axes1_x, dim=-1))
        abs_ax1_y = torch.abs(torch.sum(axis * axes1_y, dim=-1))
        proj_r1 = r1_x * abs_ax1_x + r1_y * abs_ax1_y
        # 矩形2投影半径
        abs_ax2_x = torch.abs(torch.sum(axis * axes2_x, dim=-1))
        abs_ax2_y = torch.abs(torch.sum(axis * axes2_y, dim=-1))
        proj_r2 = r2_x * abs_ax2_x + r2_y * abs_ax2_y
        gap = torch.abs(proj_delta) - (proj_r1 + proj_r2)
        if max_gap is None:
            max_gap = gap
        else:
            max_gap = torch.max(max_gap, gap)
    min_dist = torch.clamp(max_gap, min=0.0)
    return min_dist ** 2


def rect_min_dist_sq_batch(ego_center, ego_phi, ego_half_l, ego_half_w,
                           other_center, other_phi, other_half_l, other_half_w):
    """
    批量计算自车与多个周车之间的最小距离平方。
    参数：
        ego_center: (B, 1, 2) 或 (B, 2)
        ego_phi:    (B, 1) 或 (B,)
        other_center: (B, 1, N, 2)
        other_phi:    (B, 1, N)
    返回：
        min_dist_sq: (B, 1, N)
    """
    # 统一扩展为 (B, 1, N, 2) 和 (B, 1, N)
    B = ego_center.shape[0]
    if ego_center.dim() == 2:
        ego_center = ego_center.unsqueeze(1).unsqueeze(2)  # (B,1,1,2)
    elif ego_center.dim() == 3:
        ego_center = ego_center.unsqueeze(2)  # (B,1,1,2)
    if ego_phi.dim() == 1:
        ego_phi = ego_phi.unsqueeze(1).unsqueeze(2)  # (B,1,1)
    elif ego_phi.dim() == 2:
        ego_phi = ego_phi.unsqueeze(2)  # (B,1,1)
    N = other_center.shape[2]
    # 展平到 (B*N, 2) 和 (B*N,)
    ego_center_flat = ego_center.expand(-1, -1, N, -1).reshape(B * N, 2)
    ego_phi_flat = ego_phi.expand(-1, -1, N).reshape(B * N)
    other_center_flat = other_center.reshape(B * N, 2)
    other_phi_flat = other_phi.reshape(B * N)
    min_dist_sq_flat = rect_min_dist_sq(
        ego_center_flat, ego_phi_flat, ego_half_l, ego_half_w,
        other_center_flat, other_phi_flat, other_half_l, other_half_w
    )
    return min_dist_sq_flat.reshape(B, 1, N)