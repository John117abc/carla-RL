import numpy as np

import carla
import math
from typing import Tuple

def world_to_ego_coordinate(
        world_x: float,
        world_y: float,
        ego_transform: carla.Transform
) -> Tuple[float, float]:
    """
    将世界坐标系下的坐标转换为自车坐标系
    自车坐标系定义：自车位置为原点，车头朝向为x轴正方向，左侧为y轴正方向
    :param world_x: 世界坐标系x
    :param world_y: 世界坐标系y
    :param ego_transform: 自车的transform
    :return: (ego_x, ego_y) 自车坐标系下的坐标
    """
    # 自车在世界坐标系下的位置
    ego_x_world = ego_transform.location.x
    ego_y_world = ego_transform.location.y

    # 自车航向角（弧度）
    ego_yaw_rad = math.radians(ego_transform.rotation.yaw)

    # 计算相对位移
    dx = world_x - ego_x_world
    dy = world_y - ego_y_world

    # 旋转矩阵：世界坐标系 -> 自车坐标系
    # [x_ego]   [cosθ  sinθ] [dx]
    # [y_ego] = [-sinθ cosθ] [dy]
    ego_x = dx * math.cos(ego_yaw_rad) + dy * math.sin(ego_yaw_rad)
    ego_y = -dx * math.sin(ego_yaw_rad) + dy * math.cos(ego_yaw_rad)

    return ego_x, ego_y


def ego_to_world_coordinate(
        ego_x: float,
        ego_y: float,
        ego_transform: carla.Transform
) -> Tuple[float, float]:
    """
    将自车坐标系下的坐标还原为世界坐标系（world_to_ego_coordinate的反函数）
    自车坐标系定义：自车位置为原点，车头朝向为x轴正方向，左侧为y轴正方向
    :param ego_x: 自车坐标系x
    :param ego_y: 自车坐标系y
    :param ego_transform: 自车的transform（包含世界坐标位置和航向角）
    :return: (world_x, world_y) 世界坐标系下的坐标
    """
    # 1. 获取自车在世界坐标系的位置
    ego_x_world = ego_transform.location.x
    ego_y_world = ego_transform.location.y

    # 2. 获取自车航向角（弧度）
    ego_yaw_rad = math.radians(ego_transform.rotation.yaw)

    # 3. 旋转矩阵：自车坐标系 -> 世界坐标系（原旋转矩阵的逆矩阵/转置矩阵）
    # 原矩阵：世界→自车 = [cosθ  sinθ; -sinθ cosθ]
    # 逆矩阵：自车→世界 = [cosθ -sinθ; sinθ  cosθ]
    dx = ego_x * math.cos(ego_yaw_rad) - ego_y * math.sin(ego_yaw_rad)
    dy = ego_x * math.sin(ego_yaw_rad) + ego_y * math.cos(ego_yaw_rad)

    # 4. 加上自车在世界坐标系的偏移，得到最终世界坐标
    world_x = ego_x_world + dx
    world_y = ego_y_world + dy

    return world_x, world_y

def batch_world_to_ego(path_locations, ego_transform):
    xy_world = np.array([[p.x, p.y] for p in path_locations], dtype=np.float32)
    ego_x = ego_transform.location.x
    ego_y = ego_transform.location.y
    yaw = np.radians(ego_transform.rotation.yaw)
    c, s = np.cos(yaw), np.sin(yaw)

    dx = xy_world[:, 0] - ego_x
    dy = xy_world[:, 1] - ego_y

    # 修复1：使用正确的旋转矩阵（注意第二行符号）
    x_ego = dx * c + dy * s
    y_ego = dx * (-s) + dy * c  # 修复符号问题

    # 修复2：确保横向误差定义正确
    # 在IDC中：y_ego > 0 表示参考路径在车辆左侧（需要右转）
    #          y_ego < 0 表示参考路径在车辆右侧（需要左转）
    return np.stack([x_ego, y_ego], axis=1).tolist()


def velocity_to_global(v_lon:float, yaw:float):
    """
    根据车辆纵向速度和航向角计算全局速度。
    """
    vx = v_lon * np.cos(yaw)
    vy = v_lon * np.sin(yaw)

    return np.array([vx, vy], dtype=float)