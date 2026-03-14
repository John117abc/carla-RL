# src/carla_utils/ocp_setup.py
import carla
import numpy as np
import torch
from numpy import ndarray, dtype
import math
from typing import List, Dict, Union, Any, Optional

import src.envs.sensors
from src.carla_utils.vehicle_control import world_to_vehicle_frame  # 统一导入，避免重复


def get_ocp_observation(
    ego_vehicle: carla.Vehicle,
    ego_imu: src.envs.sensors.IMUSensor,
    other_vehicles: List[carla.Vehicle],
    path_locations: List[carla.Location],
    world_map: carla.Map,
    ego_ref_speed: float
) -> List[ndarray]:
    """
    获取OCP控制所需的全量观测信息
    :param ego_ref_speed: 自车参考速度 m/s
    :param world_map: 世界地图
    :param ego_vehicle: 自车
    :param ego_imu: imu传感器
    :param other_vehicles: 周车列表
    :param path_locations: 静态路径点列表
    :return: [自车观察信息(6维), 周车观察信息(N×6维), 道路观察信息(6维), 静态路径观察信息(6维)]
    """
    # 鲁棒性：确保各子函数返回非None的数组
    ocp_ego = get_ego_observation(ego_vehicle, ego_imu)
    ocp_other = get_other_observation(ego_vehicle, other_vehicles)
    ocp_road = get_road_edge_points(ego_vehicle, world_map)
    ocp_ref = get_ref_observation(ego_vehicle, path_locations, ego_ref_speed)

    # 兜底：若子函数返回None则替换为全零数组
    ocp_road = ocp_road if ocp_road is not None else np.zeros(6)
    ocp_ref = ocp_ref if ocp_ref is not None else np.zeros(6)

    return [ocp_ego, ocp_other, ocp_road, ocp_ref]


def get_ego_observation(
    ego_vehicle: carla.Vehicle,
    ego_imu: Optional[src.envs.sensors.IMUSensor]
) -> ndarray:
    """
    获取OCP中自车的观察信息
    :param ego_vehicle: 自车
    :param ego_imu: imu传感器（允许为空，空则横摆角速度设0）
    :return: 6维数组 [x坐标, y坐标, 纵向速度, 横向速度, 航向角(度), 横摆角速度(rad/s)]
    """
    ego_transform = ego_vehicle.get_transform()
    # 获取x，y坐标（世界坐标系）
    ego_x = ego_transform.location.x
    ego_y = ego_transform.location.y

    # 转换为车辆坐标系的速度（纵向/横向）
    v_world = ego_vehicle.get_velocity()
    ego_vx, ego_vy = world_to_vehicle_frame(v_world, ego_transform)

    # 航向角（度）
    ego_yaw_deg = ego_transform.rotation.yaw

    # 横摆角速度（鲁棒处理：IMU为空则设0）
    ego_angular_velocity_z = 0.0
    if ego_imu is not None and hasattr(ego_imu, 'get_angular_velocity'):
        angular_vel = ego_imu.get_angular_velocity()
        ego_angular_velocity_z = angular_vel.z if isinstance(angular_vel, carla.Vector3D) else 0.0

    # 组装6维数组
    ocp_obs = np.array([ego_x, ego_y, ego_vx, ego_vy, ego_yaw_deg, ego_angular_velocity_z], dtype=np.float32)
    return ocp_obs


def get_other_observation(
    ego_vehicle: carla.Vehicle,
    other_vehicles: List[carla.Vehicle],
    max_num_vehicles: int = 8,
    distance_threshold: float = 50.0
) -> ndarray:
    """
    获取自车周围指定距离内最近的若干辆车的状态信息
    若车辆数不足 max_num_vehicles，则用零值填充至固定长度
    :param ego_vehicle: 自车
    :param other_vehicles: 周车列表
    :param max_num_vehicles: 最大返回车辆数
    :param distance_threshold: 距离阈值（米）
    :return: ndarray [max_num_vehicles, 6]，每行为[ x, y, 纵向速度, 横向速度(0), 航向角, 横摆角速度(0) ]
    """
    def _get_zero_vehicle_obs() -> List[float]:
        """返回一个“空车辆”的零值观测列表"""
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if not other_vehicles:
        # 直接返回全零填充
        zero_array = np.array([_get_zero_vehicle_obs() for _ in range(max_num_vehicles)], dtype=np.float32)
        return zero_array

    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    nearby_vehicles = []

    for vehicle in other_vehicles:
        # 跳过无效车辆
        if not vehicle or not vehicle.is_alive:
            continue

        other_loc = vehicle.get_transform().location
        # 计算自车到周车的距离
        distance = ego_location.distance(other_loc)
        # 过滤过远/过近（自身）的车辆
        if distance > distance_threshold or distance < 1e-3:
            continue

        # 周车速度：仅保留纵向速度，横向速度强制设0（按注释要求）
        v_world = vehicle.get_velocity()
        vx, _ = world_to_vehicle_frame(v_world, vehicle.get_transform())
        other_yaw = vehicle.get_transform().rotation.yaw

        # 保存：[x, y, 纵向速度, 横向速度(0), 航向角, 横摆角速度(0)] + 距离（用于排序）
        nearby_vehicles.append({
            'obs': [other_loc.x, other_loc.y, vx, 0.0, other_yaw, 0.0],
            'distance': distance
        })

    # 按距离排序，取最近的N辆
    nearby_vehicles.sort(key=lambda x: x['distance'])
    valid_obs = [item['obs'] for item in nearby_vehicles[:max_num_vehicles]]

    # 补零至固定长度
    padded_result = valid_obs + [_get_zero_vehicle_obs() for _ in range(max_num_vehicles - len(valid_obs))]
    # 转换为数组，确保类型为float32
    return np.array(padded_result, dtype=np.float32)


def get_closest_lane_edge_point(ego_vehicle: carla.Vehicle) -> ndarray:
    """
    获取自车到左右车道边缘中**更近的那个边缘点**的世界坐标
    :return: ndarray [6,]，格式为[x, y, 0, 0, 0, 0]
    """
    world = ego_vehicle.get_world()
    map_obj = world.get_map()
    ego_loc = ego_vehicle.get_transform().location

    # 投影到道路中心线（鲁棒处理）
    waypoint = map_obj.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if waypoint is None:
        # 未在可行驶车道上，返回全零
        return np.zeros(6, dtype=np.float32)

    # 获取车道参数（兜底默认宽度）
    lane_width = waypoint.lane_width if waypoint.lane_width > 0 else 3.5
    yaw_rad = math.radians(waypoint.transform.rotation.yaw)

    # 计算前进方向和左侧垂直方向
    forward = carla.Vector3D(x=math.cos(yaw_rad), y=math.sin(yaw_rad), z=0.0)
    left_dir = carla.Vector3D(x=-forward.y, y=forward.x, z=0.0)  # 左侧单位向量

    # 归一化（避免零向量）
    norm = math.hypot(left_dir.x, left_dir.y)
    if norm > 1e-6:
        left_dir = carla.Vector3D(x=left_dir.x/norm, y=left_dir.y/norm, z=0.0)

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

    # 比较哪个更近，返回6维数组
    dist_left = ego_loc.distance(left_edge)
    dist_right = ego_loc.distance(right_edge)
    pos = left_edge if dist_left < dist_right else right_edge

    return np.array([pos.x, pos.y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def get_ref_observation(
    ego_vehicle: carla.Vehicle,
    path_locations: List[carla.Location],
    default_longitudinal_velocity: float = 20.0
) -> ndarray:
    """
    获取自车到参考路径中最近点的状态信息
    :param ego_vehicle: 自车
    :param path_locations: 参考路径点列表（世界坐标）
    :param default_longitudinal_velocity: 默认纵向速度（m/s）
    :return: ndarray [6,]，格式为[x, y, 纵向速度, 0, 航向角(度), 0]；空路径返回全零数组
    """
    if not path_locations:
        return np.zeros(6, dtype=np.float32)

    ego_location = ego_vehicle.get_transform().location

    # 找到距离自车最近的路径点（鲁棒处理重复点）
    min_distance = float('inf')
    ref_location = path_locations[0]
    closest_idx = 0
    for idx, loc in enumerate(path_locations):
        distance = ego_location.distance(loc)
        if distance < min_distance:
            min_distance = distance
            ref_location = loc
            closest_idx = idx

    # 参考点索引：向后偏移4个点（兜底避免越界）
    ref_index = min(closest_idx + 4, len(path_locations) - 1)
    ref_location = path_locations[ref_index]

    # 计算参考点航向角（基于前后路径点）
    ref_yaw_deg = 0.0
    if len(path_locations) >= 2:
        # 用当前参考点和下一个点计算航向
        next_idx = min(ref_index + 1, len(path_locations) - 1)
        delta_x = path_locations[next_idx].x - path_locations[ref_index].x
        delta_y = path_locations[next_idx].y - path_locations[ref_index].y

        # 避免除以零
        if math.hypot(delta_x, delta_y) < 1e-6:
            ref_yaw_deg = ego_vehicle.get_transform().rotation.yaw
        else:
            ref_yaw_deg = math.degrees(math.atan2(delta_y, delta_x))
    else:
        # 单点路径：使用自车当前航向
        ref_yaw_deg = ego_vehicle.get_transform().rotation.yaw

    # 组装6维数组
    return np.array([
        ref_location.x, ref_location.y,
        default_longitudinal_velocity, 0.0,
        ref_yaw_deg, 0.0
    ], dtype=np.float32)


def get_road_edge_points(ego_vehicle: carla.Vehicle, world_map: carla.Map) -> ndarray:
    """
    获取当前道路最外侧边缘的近似点（基于最外侧车道中心线偏移）
    :param world_map: 世界地图
    :param ego_vehicle: 自车
    :return: ndarray [6,]，格式为[边缘点x, 边缘点y, 0, 0, 0, 0]
    """
    # 鲁棒获取Waypoint
    vehicle_loc = ego_vehicle.get_transform().location
    wp = world_map.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return np.zeros(6, dtype=np.float32)

    def get_one_road_edge_points(current_wp: carla.Waypoint, direction: str) -> Optional[carla.Location]:
        """向指定方向遍历到最外侧车道，返回边缘点"""
        if direction == 'right':
            # 找最右侧驾驶车道
            while True:
                next_wp = current_wp.get_right_lane()
                if next_wp is None or next_wp.lane_type != carla.LaneType.Driving:
                    break
                current_wp = next_wp
            # 计算右边缘点
            lane_width = current_wp.lane_width if current_wp.lane_width > 0 else 3.5
            right_vec = current_wp.transform.get_right_vector()
            return carla.Location(
                x=current_wp.transform.location.x + right_vec.x * lane_width / 2,
                y=current_wp.transform.location.y + right_vec.y * lane_width / 2,
                z=current_wp.transform.location.z
            )

        elif direction == 'left':
            # 找最左侧驾驶车道
            while True:
                next_wp = current_wp.get_left_lane()
                if next_wp is None or next_wp.lane_type != carla.LaneType.Driving:
                    break
                current_wp = next_wp
            # 计算左边缘点（左 = -右向量）
            lane_width = current_wp.lane_width if current_wp.lane_width > 0 else 3.5
            right_vec = current_wp.transform.get_right_vector()
            return carla.Location(
                x=current_wp.transform.location.x - right_vec.x * lane_width / 2,
                y=current_wp.transform.location.y - right_vec.y * lane_width / 2,
                z=current_wp.transform.location.z
            )
        return None

    # 获取左右两侧道路边缘点（鲁棒处理None）
    right_edge = get_one_road_edge_points(wp, 'right') or vehicle_loc
    left_edge = get_one_road_edge_points(wp, 'left') or vehicle_loc

    # 计算自车到两个边缘的距离，取最近的
    dist_right = vehicle_loc.distance(right_edge)
    dist_left = vehicle_loc.distance(left_edge)
    nearest_road_edge = right_edge if dist_right < dist_left else left_edge

    return np.array([nearest_road_edge.x, nearest_road_edge.y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def get_current_lane_forward_edges(
    vehicle: carla.Vehicle,
    world: carla.World,
    distance: float = 50.0,
    resolution: float = 0.5
) -> tuple[List[carla.Location], List[carla.Location]]:
    """
    获取自车当前所在车道前方指定距离内的【左边缘】和【右边缘】坐标序列
    :param vehicle: 自车
    :param world: Carla世界对象
    :param distance: 前方搜索距离 (米), 默认50米
    :param resolution: 采样步长 (米), 默认0.5米
    :return: tuple(list, list):
        - left_edges: 左侧边缘坐标列表 [carla.Location, ...]
        - right_edges: 右侧边缘坐标列表 [carla.Location, ...]
        (两个列表长度一致，索引 i 对应同一横截面的左右边缘)
    """
    map_obj = world.get_map()
    vehicle_loc = vehicle.get_transform().location

    # 1. 鲁棒获取当前Waypoint
    try:
        current_wp = map_obj.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    except Exception as e:
        print(f"获取Waypoint失败: {e}")
        return [], []

    if current_wp is None:
        return [], []

    left_edges = []
    right_edges = []

    def calculate_edge_locations(center_wp: carla.Waypoint) -> tuple[carla.Location, carla.Location]:
        """
        从中心waypoint计算当前截面的左右边缘点
        :return: (left_edge_loc, right_edge_loc)
        """
        # --- 找左边缘（最左侧驾驶车道）---
        left_most_wp = center_wp
        while True:
            next_left = left_most_wp.get_left_lane()
            if next_left is None or next_left.lane_type != carla.LaneType.Driving:
                break
            left_most_wp = next_left

        lane_width_l = left_most_wp.lane_width if left_most_wp.lane_width > 0 else 3.5
        right_vec_l = left_most_wp.transform.get_right_vector()
        left_edge_loc = carla.Location(
            x=left_most_wp.transform.location.x - right_vec_l.x * lane_width_l / 2.0,
            y=left_most_wp.transform.location.y - right_vec_l.y * lane_width_l / 2.0,
            z=left_most_wp.transform.location.z
        )

        # --- 找右边缘（最右侧驾驶车道）---
        right_most_wp = center_wp
        while True:
            next_right = right_most_wp.get_right_lane()
            if next_right is None or next_right.lane_type != carla.LaneType.Driving:
                break
            right_most_wp = next_right

        lane_width_r = right_most_wp.lane_width if right_most_wp.lane_width > 0 else 3.5
        right_vec_r = right_most_wp.transform.get_right_vector()
        right_edge_loc = carla.Location(
            x=right_most_wp.transform.location.x + right_vec_r.x * lane_width_r / 2.0,
            y=right_most_wp.transform.location.y + right_vec_r.y * lane_width_r / 2.0,
            z=right_most_wp.transform.location.z
        )

        return left_edge_loc, right_edge_loc

    # 2. 向前迭代采样边缘点
    next_wps = current_wp.next(resolution)
    if not next_wps:
        return [], []

    iterator_wp = next_wps[0]
    accumulated_dist = resolution

    while accumulated_dist <= distance:
        # 计算当前截面的左右边缘
        l_loc, r_loc = calculate_edge_locations(iterator_wp)
        left_edges.append(l_loc)
        right_edges.append(r_loc)

        # 向前移动（鲁棒处理分叉路口）
        next_step = iterator_wp.next(resolution)
        if not next_step:
            break
        iterator_wp = next_step[0]
        accumulated_dist += resolution

    return left_edges, right_edges