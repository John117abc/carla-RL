import torch
from typing import Tuple


def get_two_circles(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入 states: [B, 1, 6] 或 [B, N, 6] (x,y,vlat,vlon,phi,omega)
    返回两个圆的中心坐标: front_xy, rear_xy，形状与输入的前两维相同
    """
    # 圆心沿车辆纵轴分布，前后各偏移 dist_from_center
    dist_from_center = self.HALF_L * 0.6  # 可取车长一半的0.6倍，使两圆覆盖全车
    x = states[..., 0]
    y = states[..., 1]
    phi = states[..., 4]  # 航向角
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    front_x = x + dist_from_center * cos_phi
    front_y = y + dist_from_center * sin_phi
    rear_x = x - dist_from_center * cos_phi
    rear_y = y - dist_from_center * sin_phi
    front_xy = torch.stack([front_x, front_y], dim=-1)  # [..., 2]
    rear_xy = torch.stack([rear_x, rear_y], dim=-1)
    return front_xy, rear_xy