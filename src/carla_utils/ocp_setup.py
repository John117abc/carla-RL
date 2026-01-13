# src/carla_utils/ocp_setup.py
import carla
import numpy as np
import torch

import src.envs.sensors
import math
from typing import List, Dict, Union

def get_ocp_observation(ego_vehicle:carla.Vehicle,
                        ego_imu:src.envs.sensors.IMUSensor,
                        other_vehicles: List[carla.Vehicle],
                        path_locations: List[carla.Location]):
    """
    获取文章中的各个参考信息
    :param ego_vehicle: 自车
    :param ego_imu: imu传感器
    :param other_vehicles: 周车
    :param path_locations: 静态路径点
    :return: [自车观察信息，周车观察信息，道路观察信息，静态路径观察信息]
    """

    # 获取自车的观察信息
    ocp_ego = get_ego_observation(ego_vehicle,ego_imu)
    # 获取周车信息
    ocp_other = get_other_observation(ego_vehicle, other_vehicles)
    # 获取道路信息
    ocp_road = get_closest_lane_edge_point(ego_vehicle)
    # 获取静态路径信息
    ocp_ref = get_ref_observation(ego_vehicle,path_locations)
    return [ocp_ego,ocp_other,ocp_road,ocp_ref]


def get_ego_observation(ego_vehicle:carla.Vehicle,ego_imu:src.envs.sensors.IMUSensor):
    """
    获取ocp中自车的观察信息
    :param ego_vehicle: 自车
    :param ego_imu: imu传感器
    :return: 坐标x，坐标y，纵向速度，横向速度，航向角，横摆角速度
    """
    from src.carla_utils.vehicle_control import world_to_vehicle_frame
    # 获取自车的观察信息
    v = ego_vehicle.get_velocity()
    ego_transform = ego_vehicle.get_transform()
    # 获取x，y坐标
    ego_x = ego_transform.location.x
    ego_y = ego_transform.location.y
    # 转换为车辆坐标系的速度
    ego_vx, ego_vy = world_to_vehicle_frame(v, ego_vehicle.get_transform())
    # 航向角
    ego_yaw_deg = ego_transform.rotation.yaw
    # 角速度
    ego_angular_velocity = ego_imu.get_angular_velocity()
    ocp_obs = np.array([ego_x,ego_y,ego_vx,ego_vy,ego_yaw_deg,ego_angular_velocity.z])
    return ocp_obs



def get_other_observation(
    ego_vehicle: carla.Vehicle,
    other_vehicles: List[carla.Vehicle],
    max_num_vehicles: int = 8,
    distance_threshold: float = 50.0
) -> List[float]:
    """
    获取自车周围指定距离内最近的若干辆车的状态信息。
    若车辆数不足 max_num_vehicles，则用零值字典填充至固定长度。

    Returns:
        List[Dict]：长度恒为 max_num_vehicles。每项为字典，若无车则全为0/None。
        字段包括： 坐标x，坐标y，纵向速度，横向速度(0)，航向角，横摆角速度(0)
    """
    from src.carla_utils.vehicle_control import world_to_vehicle_frame

    def _get_zero_vehicle_obs() -> Dict:
        """返回一个“空车辆”的零值观测字典"""
        return [0, 0, 0, 0, 0, 0]


    if not other_vehicles:
        # 直接返回全零填充
        return [_get_zero_vehicle_obs() for _ in range(max_num_vehicles)]

    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location

    nearby_vehicles = []

    for vehicle in other_vehicles:
        if not vehicle.is_alive:
            continue

        other_loc = vehicle.get_transform().location
        distance = ego_location.distance(other_loc)

        if distance > distance_threshold or distance < 1e-3:
            continue

        vx,vy = world_to_vehicle_frame(vehicle.get_velocity(), vehicle.get_transform())

        other_yaw = vehicle.get_transform().rotation.yaw

        nearby_vehicles.append([other_loc.x,other_loc.y,vx,0,other_yaw,0])

    # 按距离排序，取最近的 N 辆
    # nearby_vehicles.sort(key=lambda x: x['distance'])
    valid_vehicles = nearby_vehicles[:max_num_vehicles]

    # 补零至固定长度
    padded_result = []
    for i in range(max_num_vehicles):
        if i < len(valid_vehicles):
            padded_result.append(valid_vehicles[i])
        else:
            padded_result.append(_get_zero_vehicle_obs())

    return np.array(padded_result)


def get_closest_lane_edge_point(ego_vehicle: carla.Vehicle) -> carla.Location:
    """
    获取自车到左右车道边缘中**更近的那个边缘点**的世界坐标。

    返回:
        carla.Location —— 最近的车道边缘点（左或右）
    """
    world = ego_vehicle.get_world()
    map_obj = world.get_map()
    ego_loc = ego_vehicle.get_transform().location

    # 投影到道路中心线
    waypoint = map_obj.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if waypoint is None:
        raise RuntimeError("车辆未在可行驶车道上，无法计算车道边缘。")

    # 获取车道参数
    lane_width = waypoint.lane_width if waypoint.lane_width > 0 else 3.5
    yaw_rad = math.radians(waypoint.transform.rotation.yaw)

    # 计算前进方向和左侧垂直方向
    forward = carla.Vector3D(x=math.cos(yaw_rad), y=math.sin(yaw_rad), z=0.0)
    left_dir = carla.Vector3D(x=-forward.y, y=forward.x, z=0.0)  # 左侧单位向量

    # 归一化（虽然理论上已是单位向量）
    norm = (left_dir.x ** 2 + left_dir.y ** 2) ** 0.5
    if norm > 1e-6:
        left_dir = left_dir / norm

    # 计算左右边缘点
    center = waypoint.transform.location
    offset = lane_width / 2.0

    left_edge = carla.Location(
        x=center.x + left_dir.x * offset,
        y=center.y + left_dir.y * offset,
        z=center.z
    )
    right_edge = carla.Location(
        x=center.x - left_dir.x * offset,
        y=center.y - left_dir.y * offset,
        z=center.z
    )

    # 比较哪个更近，返回最近的
    dist_left = ego_loc.distance(left_edge)
    dist_right = ego_loc.distance(right_edge)

    pos = left_edge if dist_left < dist_right else right_edge

    return np.array([pos.x,pos.y,0,0,0,0])



def get_ref_observation(
    ego_vehicle: carla.Vehicle,
    path_locations: List[carla.Location],
    default_longitudinal_velocity: float = 5.0  # 默认纵向速度（m/s），可根据任务调整
) -> List[float]:
    """
    获取自车到参考路径中最近点的状态信息。

    Args:
        ego_vehicle: 自车
        path_locations: 参考路径点列表（世界坐标）
        default_longitudinal_velocity: 默认纵向速度（因路径点无速度信息，需预设）

    Returns:
        (x, y, longitudinal_velocity, yaw) —— 最近参考点的世界坐标 x, y，
        预设的纵向速度，以及该点的航向角（度）。
        若路径为空，返回 None。
    """
    if not path_locations:
        return None

    ego_location = ego_vehicle.get_transform().location

    min_distance = float('inf')
    ref_location = path_locations[0]  # 初始化

    for loc in path_locations:
        distance = ego_location.distance(loc)
        if distance < min_distance:
            min_distance = distance
            ref_location = loc

    # 假设参考路径的航向角由前后点差分得到
    ref_yaw_deg = 0.0
    if len(path_locations) >= 2:
        # 找到 ref_location 在 path_locations 中的索引
        try:
            idx = path_locations.index(ref_location)
        except ValueError:
            # 如果有重复点，用距离最近的索引
            idx = min(range(len(path_locations)), key=lambda i: ego_location.distance(path_locations[i]))

        # 用后一个点计算方向
        next_idx = min(idx + 1, len(path_locations) - 1)
        delta_x = path_locations[next_idx].x - path_locations[idx].x
        delta_y = path_locations[next_idx].y - path_locations[idx].y

        import math
        ref_yaw_deg = math.degrees(math.atan2(delta_y, delta_x))
    else:
        # 单点路径：使用自车当前航向 或 默认 0
        ref_yaw_deg = ego_vehicle.get_transform().rotation.yaw

    x = ref_location.x
    y = ref_location.y
    longitudinal_velocity = default_longitudinal_velocity
    yaw = ref_yaw_deg

    return np.array([x, y, longitudinal_velocity,0, yaw,0])