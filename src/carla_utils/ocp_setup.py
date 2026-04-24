# src/carla_utils/ocp_setup.py
import numpy as np
from typing import List, Optional
import src.envs.env_model.sensors_manager
from src.carla_utils.vehicle_control import world_to_vehicle_frame

import carla
import math
from typing import Tuple


def batch_world_to_ego(path_locations, ego_transform):
    """
    批量将 CARLA Location 列表 转换为 自车坐标系 xy 坐标
    无循环、纯向量化计算，速度极快
    """
    # 1. 一次性把所有坐标转成 numpy 数组（关键提速点）
    xy_world = np.array([[p.x, p.y] for p in path_locations], dtype=np.float32)

    # 2. 自车位置与航向角
    ego_x = ego_transform.location.x
    ego_y = ego_transform.location.y
    yaw = np.radians(ego_transform.rotation.yaw)
    c, s = np.cos(yaw), np.sin(yaw)

    # 3. 相对位移
    dx = xy_world[:, 0] - ego_x
    dy = xy_world[:, 1] - ego_y

    # 4. 旋转矩阵（向量化，一次性转换所有点）
    x_ego = dx * c + dy * s
    y_ego = -dx * s + dy * c

    # 5. 组合成和你原来格式一样的 [[x,y], [x,y], ...]
    return np.stack([x_ego, y_ego], axis=1).tolist()

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


def get_ocp_observation_ego_frame(
        ego_vehicle: carla.Vehicle,
        ego_imu: Optional[src.envs.env_model.sensors_manager.IMUSensor],
        other_vehicles: List[carla.Vehicle],
        path_locations: List[carla.Location],
        ego_ref_speed: float,
        ref_offset: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    获取OCP观测信息并转换为自车坐标系（修复了角度未转换的Bug）
    """
    # 1. 调用原函数获取世界坐标系下的观测数据
    network_state, s_road, s_ref_raw, s_ref_error = get_ocp_observation(
        ego_vehicle, ego_imu, other_vehicles, path_locations, ego_ref_speed, ref_offset
    )

    # 2. 获取自车transform用于坐标转换
    ego_transform = ego_vehicle.get_transform()
    ego_yaw_rad = math.radians(ego_transform.rotation.yaw)  # 获取自车在世界坐标系下的航向角(弧度)

    # --------------------------
    # 转换network_state中的位置和角度数据
    # --------------------------
    # network_state结构：s_ego(6) + s_other(32) + s_ref_error(3)
    s_ego_world = network_state[:6].copy()
    s_other_world = network_state[6:6 + 32].copy().reshape(8, 4)
    s_ref_error_original = network_state[6 + 32:].copy()

    # 【修复1】转换自车状态：在自车坐标系中，自车位置恒为原点，航向恒为0
    s_ego_ego = s_ego_world.copy()
    s_ego_ego[0], s_ego_ego[1] = 0.0, 0.0  # 自车在自身坐标系原点
    s_ego_ego[4] = 0.0  # 自车相对自身的航向角永远是 0

    # 【修复2 & 4】转换周车状态 (位置 + 相对航向角)
    s_other_ego = s_other_world.copy()
    for i in range(s_other_ego.shape[0]):
        # 更严谨的 Dummy (填充空位) 车辆判定: 如果完全是 [0,0,0,0] 则不转换
        is_dummy = (s_other_ego[i, 0] == 0.0 and s_other_ego[i, 1] == 0.0 and
                    s_other_ego[i, 2] == 0.0 and s_other_ego[i, 3] == 0.0)

        if not is_dummy:
            # 转换 xy 坐标
            x, y = world_to_ego_coordinate(
                s_other_ego[i, 0], s_other_ego[i, 1], ego_transform
            )
            s_other_ego[i, 0] = x
            s_other_ego[i, 1] = y

            # 转换航向角：周车绝对航向角 - 自车绝对航向角，并归一化到 [-π, π]
            delta_yaw = s_other_ego[i, 2] - ego_yaw_rad
            s_other_ego[i, 2] = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))

    # 重新组合network_state
    network_state_ego = np.concatenate([
        s_ego_ego,
        s_other_ego.flatten(),
        s_ref_error_original
    ], axis=0)

    # --------------------------
    # 转换s_road中的道路边缘点 (仅位置，无角度，原逻辑基本正确)
    # --------------------------
    s_road_ego = s_road.copy()
    num_points = 20
    # 处理左边缘点
    for i in range(num_points):
        idx = i * 2
        if idx + 1 < len(s_road_ego) and not (s_road_ego[idx] == 0 and s_road_ego[idx + 1] == 0):
            x, y = world_to_ego_coordinate(s_road_ego[idx], s_road_ego[idx + 1], ego_transform)
            s_road_ego[idx] = x
            s_road_ego[idx + 1] = y

    # 处理右边缘点
    right_start_idx = num_points * 2
    for i in range(num_points):
        idx = right_start_idx + i * 2
        if idx + 1 < len(s_road_ego) and not (s_road_ego[idx] == 0 and s_road_ego[idx + 1] == 0):
            x, y = world_to_ego_coordinate(s_road_ego[idx], s_road_ego[idx + 1], ego_transform)
            s_road_ego[idx] = x
            s_road_ego[idx + 1] = y

    # --------------------------
    # 转换s_ref_raw中的参考路径点
    # --------------------------
    s_ref_ego = s_ref_raw.copy()
    # 判断是否为无效全0状态
    if not (s_ref_ego[0] == 0 and s_ref_ego[1] == 0 and s_ref_ego[4] == 0):
        # 转换坐标
        x, y = world_to_ego_coordinate(s_ref_ego[0], s_ref_ego[1], ego_transform)
        s_ref_ego[0] = x
        s_ref_ego[1] = y

        # 【修复3】转换参考航向角并归一化到 [-π, π]
        ref_delta_yaw = s_ref_ego[4] - ego_yaw_rad
        s_ref_ego[4] = math.atan2(math.sin(ref_delta_yaw), math.cos(ref_delta_yaw))

    return network_state_ego, s_road_ego, s_ref_ego, s_ref_error, s_road, s_ref_raw

def get_ocp_observation(
        ego_vehicle: carla.Vehicle,
        ego_imu: Optional[src.envs.env_model.sensors_manager.IMUSensor],
        other_vehicles: List[carla.Vehicle],
        path_locations: List[carla.Location],
        ego_ref_speed: float,
        ref_offset: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    获取完全对齐论文的OCP控制所需全量观测信息
    :param ref_offset: 参考偏移
    :param ego_vehicle: 自车
    :param ego_imu: IMU传感器（允许为空）
    :param other_vehicles: 周车列表
    :param path_locations: 静态路径点列表
    :param world_map: 世界地图
    :param ego_ref_speed: 自车参考速度 (m/s)
    :return: (s_ego, s_other, s_road, s_ref_raw, s_ref_error)
        - s_ego: 自车原始状态 (6,) [x, y, v_lon, v_lat, φ(rad), ω]
        - s_other: 周车状态 (8, 4) [x, y, φ(rad), v_lon] 每车
        - s_road: 多帧道路边缘 (80,) [左x1,左y1,右x1,右y1,...] 前20个点
        - s_ref_raw: 参考路径原始状态 (6,) [x_ref, y_ref, v_ref, 0, φ_ref(rad), 0]
        - s_ref_error: 参考路径误差状态 (3,) [δ_p(带符号), δ_φ, δ_v]
    """
    # 1. 获取自车状态
    s_ego = get_ego_observation(ego_vehicle, ego_imu)

    # 2. 获取周车状态
    s_other = get_other_observation(ego_vehicle, other_vehicles)

    # 3. 获取多帧道路边缘状态
    s_road = get_road_observation_multi_frame(ego_vehicle, ego_vehicle.get_world())

    # 4. 获取参考路径原始状态
    s_ref_raw = get_ref_observation(ego_vehicle, path_locations, ego_ref_speed,ref_offset)

    # 5. 计算参考路径误差状态
    s_ref_error = calc_ref_error(s_ego, s_ref_raw)

    # 网络输入状态：s_ego(6) + s_other(32) + s_ref_error(3) = 41/45维
    network_state = np.concatenate([
        s_ego,
        s_other.flatten(),  # (8,4) -> (32,)
        s_ref_error
    ], axis=0)

    # 单独返回道路信息，仅用于约束计算
    return network_state, s_road, s_ref_raw, s_ref_error


def get_ego_observation(
        ego_vehicle: carla.Vehicle,
        ego_imu: Optional[src.envs.env_model.sensors_manager.IMUSensor]
) -> np.ndarray:
    """
    获取论文定义的自车状态 (6维)
    :param ego_vehicle: 自车
    :param ego_imu: IMU传感器
    :return: [x, y, v_lon, v_lat, φ(rad), ω]
    """
    ego_transform = ego_vehicle.get_transform()

    # 1. 世界坐标系下的重心坐标
    ego_x = ego_transform.location.x
    ego_y = ego_transform.location.y

    # 2. 车辆坐标系下的纵向/横向速度
    v_world = ego_vehicle.get_velocity()
    ego_vlon, ego_vlat = world_to_vehicle_frame(v_world, ego_transform)

    # 3. 航向角 (弧度) - 论文要求
    ego_yaw_rad = math.radians(ego_transform.rotation.yaw)

    # 4. 横摆角速度 (rad/s) - 鲁棒处理
    ego_omega = 0.0
    if ego_imu is not None and hasattr(ego_imu, 'get_angular_velocity'):
        angular_vel = ego_imu.get_angular_velocity()
        if isinstance(angular_vel, carla.Vector3D):
            ego_omega = angular_vel.z

    return np.array([ego_x, ego_y, ego_vlon, ego_vlat, ego_yaw_rad, ego_omega], dtype=np.float32)


def get_other_observation(
        ego_vehicle: carla.Vehicle,
        other_vehicles: List[carla.Vehicle],
        max_num_vehicles: int = 8,
        distance_threshold: float = 50.0
) -> np.ndarray:
    """
    获取论文定义的周车状态 (8×4维)
    论文V-B1: s^other = [p_x^j, p_y^j, φ^j, v_lon^j] 每车
    :param ego_vehicle: 自车
    :param other_vehicles: 周车列表
    :param max_num_vehicles: 最大返回车辆数
    :param distance_threshold: 距离阈值 (米)
    :return: (8, 4) 每车为 [x, y, φ(rad), v_lon]
    """

    def _get_zero_vehicle_obs() -> List[float]:
        return [0.0, 0.0, 0.0, 0.0]

    if not other_vehicles:
        return np.array([_get_zero_vehicle_obs() for _ in range(max_num_vehicles)], dtype=np.float32)

    ego_location = ego_vehicle.get_transform().location
    nearby_vehicles = []

    for vehicle in other_vehicles:
        if not vehicle or not vehicle.is_alive:
            continue

        other_loc = vehicle.get_transform().location
        distance = ego_location.distance(other_loc)

        if distance > distance_threshold or distance < 1e-3:
            continue

        # 论文要求：世界坐标系下的坐标
        other_x = other_loc.x
        other_y = other_loc.y

        # 论文要求：航向角 (弧度)
        other_yaw_rad = math.radians(vehicle.get_transform().rotation.yaw)

        # 论文要求：周车自身坐标系下的纵向速度
        v_world = vehicle.get_velocity()
        v_world_x = v_world.x
        v_world_y = v_world.y
        other_vlon = v_world_x * math.cos(other_yaw_rad) + v_world_y * math.sin(other_yaw_rad)

        nearby_vehicles.append({
            'obs': [other_x, other_y, other_yaw_rad, other_vlon],
            'distance': distance
        })

    # 按距离排序，取最近的N辆
    nearby_vehicles.sort(key=lambda x: x['distance'])
    valid_obs = [item['obs'] for item in nearby_vehicles[:max_num_vehicles]]

    # 补零至固定长度
    padded_result = valid_obs + [_get_zero_vehicle_obs() for _ in range(max_num_vehicles - len(valid_obs))]

    return np.array(padded_result, dtype=np.float32)


def get_ref_observation(
        ego_vehicle: carla.Vehicle,
        path_locations: List[carla.Location],
        default_longitudinal_velocity: float = 20.0,
        ref_offset: int = 10
) -> np.ndarray:
    """
    获取论文定义的参考路径原始状态 (6维)
    论文IV-A公式2: x_ref = [p_x^ref, p_y^ref, v_lon^ref, 0, φ^ref, 0]
    :param ref_offset: 参考偏移
    :param ego_vehicle: 自车
    :param path_locations: 参考路径点列表
    :param default_longitudinal_velocity: 默认参考速度 (m/s)
    :return: [x_ref, y_ref, v_ref, 0, φ_ref(rad), 0]
    """
    if not path_locations:
        return np.zeros(6, dtype=np.float32)

    ego_location = ego_vehicle.get_transform().location

    # 1. 找到距离自车最近的路径点
    min_distance = float('inf')
    closest_idx = 0
    for idx, loc in enumerate(path_locations):
        distance = ego_location.distance(loc)
        if distance < min_distance:
            min_distance = distance
            closest_idx = idx

    # 2. 参考点偏移+4 (舒适性设计，论文未禁止)
    ref_index = min(closest_idx + ref_offset, len(path_locations) - 1)
    ref_location = path_locations[ref_index]

    # 3. 计算参考点航向角 (弧度) - 基于前后路径点差分
    ref_yaw_rad = 0.0
    if len(path_locations) >= 2:
        next_idx = min(ref_index + 1, len(path_locations) - 1)
        delta_x = path_locations[next_idx].x - path_locations[ref_index].x
        delta_y = path_locations[next_idx].y - path_locations[ref_index].y

        if math.hypot(delta_x, delta_y) < 1e-6:
            # 路径点重合时用自车航向
            ref_yaw_rad = math.radians(ego_vehicle.get_transform().rotation.yaw)
        else:
            ref_yaw_rad = math.atan2(delta_y, delta_x)
    else:
        ref_yaw_rad = math.radians(ego_vehicle.get_transform().rotation.yaw)

    return np.array([
        ref_location.x, ref_location.y,
        default_longitudinal_velocity, 0.0,
        ref_yaw_rad, 0.0
    ], dtype=np.float32)


def calc_ref_error(ego_state: np.ndarray, ref_state: np.ndarray) -> np.ndarray:
    """
    计算论文定义的参考路径误差状态 (3维)
    论文V-B1: s^ref = [δ_p, δ_φ, δ_v]
    :param ego_state: 自车状态 (6,)
    :param ref_state: 参考路径原始状态 (6,)
    :return: [δ_p(带符号), δ_φ, δ_v]
        - δ_p: 位置误差 (米)，左正右负
        - δ_φ: 航向角误差 (rad)，归一化到[-π, π]
        - δ_v: 速度误差 (m/s)
    """
    # 1. 计算位置误差 δ_p
    dx = ref_state[0] - ego_state[0]
    dy = ref_state[1] - ego_state[1]
    delta_p = math.hypot(dx, dy)

    # 计算方向符号：自车在参考路径左侧为正，右侧为负
    # 叉积判断：dx*sin(φ_ref) - dy*cos(φ_ref)
    cross = dx * math.sin(ref_state[4]) - dy * math.cos(ref_state[4])
    delta_p = delta_p * math.copysign(1.0, cross)

    # 2. 计算航向角误差 δ_φ
    delta_phi = ego_state[4] - ref_state[4]
    # 归一化到 [-π, π]
    delta_phi = math.atan2(math.sin(delta_phi), math.cos(delta_phi))

    # 3. 计算速度误差 δ_v
    delta_v = ego_state[2] - ref_state[2]

    return np.array([delta_p, delta_phi, delta_v], dtype=np.float32)


def get_road_observation_multi_frame(
        ego_vehicle: carla.Vehicle,
        world: carla.World,
        distance: float = 10.0,
        resolution: float = 0.5,
        num_points: int = 20
) -> np.ndarray:
    """
    获取论文隐含的多帧道路边缘状态 (80维)
    用于OCP有限时域安全约束
    :param ego_vehicle: 自车
    :param world: Carla世界对象
    :param distance: 前方搜索距离 (米)
    :param resolution: 采样步长 (米)
    :param num_points: 返回的边缘点数量
    :return: (80,) [左x1,左y1,左x2,左y2, ..., 右x1,右x1,右x2,右y2]
    """
    map_obj = world.get_map()
    vehicle_loc = ego_vehicle.get_transform().location

    # 1. 获取当前Waypoint
    try:
        current_wp = map_obj.get_waypoint(
            vehicle_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
    except Exception:
        return np.zeros(num_points * 4, dtype=np.float32)

    if current_wp is None:
        return np.zeros(num_points * 4, dtype=np.float32)

    left_edges = []
    right_edges = []

    # 2. 辅助函数：计算当前截面的左右边缘点
    def calculate_edge_locations(center_wp: carla.Waypoint) -> Tuple[carla.Location, carla.Location]:
        # 找最左侧驾驶车道
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

        # 找最右侧驾驶车道
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

    # 3. 向前迭代采样边缘点
    next_wps = current_wp.next(resolution)
    if not next_wps:
        return np.zeros(num_points * 4, dtype=np.float32)

    iterator_wp = next_wps[0]
    accumulated_dist = resolution

    while accumulated_dist <= distance and len(left_edges) < num_points:
        l_loc, r_loc = calculate_edge_locations(iterator_wp)
        left_edges.append(l_loc)
        right_edges.append(r_loc)

        next_step = iterator_wp.next(resolution)
        if not next_step:
            break
        iterator_wp = next_step[0]
        accumulated_dist += resolution

    # 4. 【修改这里！】先把所有左点放前面，再放所有右点
    left_flat = []
    for l in left_edges:
        left_flat.extend([l.x, l.y])  # 左1x,左1y,左2x,左2y...

    right_flat = []
    for r in right_edges:
        right_flat.extend([r.x, r.y]) # 右1x,右1y,右2x,右2y...

    # 最终：左点全部 + 右点全部
    road_obs = left_flat + right_flat

    # 补零至固定长度
    road_obs = road_obs + [0.0] * (num_points * 4 - len(road_obs))

    return np.array(road_obs, dtype=np.float32)


# 保留原函数以保持接口兼容性（但内部调用新实现）
def get_closest_lane_edge_point(ego_vehicle: carla.Vehicle) -> np.ndarray:
    """兼容接口：获取最近车道边缘点（返回6维数组）"""
    world = ego_vehicle.get_world()
    map_obj = world.get_map()
    ego_loc = ego_vehicle.get_transform().location

    waypoint = map_obj.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if waypoint is None:
        return np.zeros(6, dtype=np.float32)

    lane_width = waypoint.lane_width if waypoint.lane_width > 0 else 3.5
    yaw_rad = math.radians(waypoint.transform.rotation.yaw)
    forward = carla.Vector3D(x=math.cos(yaw_rad), y=math.sin(yaw_rad), z=0.0)
    left_dir = carla.Vector3D(x=-forward.y, y=forward.x, z=0.0)

    norm = math.hypot(left_dir.x, left_dir.y)
    if norm > 1e-6:
        left_dir = carla.Vector3D(x=left_dir.x / norm, y=left_dir.y / norm, z=0.0)

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

    dist_left = ego_loc.distance(left_edge)
    dist_right = ego_loc.distance(right_edge)
    pos = left_edge if dist_left < dist_right else right_edge

    return np.array([pos.x, pos.y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def get_road_edge_points(ego_vehicle: carla.Vehicle, world_map: carla.Map) -> np.ndarray:
    """兼容接口：获取道路边缘点（返回6维数组）"""
    return get_closest_lane_edge_point(ego_vehicle)


def get_current_lane_forward_edges(
        vehicle: carla.Vehicle,
        world: carla.World,
        distance: float = 50.0,
        resolution: float = 0.5
) -> Tuple[List[carla.Location], List[carla.Location]]:
    """兼容接口：获取前方道路边缘点列表"""
    map_obj = world.get_map()
    vehicle_loc = vehicle.get_transform().location

    try:
        current_wp = map_obj.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    except Exception:
        return [], []

    if current_wp is None:
        return [], []

    left_edges = []
    right_edges = []

    def calculate_edge_locations(center_wp: carla.Waypoint) -> Tuple[carla.Location, carla.Location]:
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

    next_wps = current_wp.next(resolution)
    if not next_wps:
        return [], []

    iterator_wp = next_wps[0]
    accumulated_dist = resolution

    while accumulated_dist <= distance:
        l_loc, r_loc = calculate_edge_locations(iterator_wp)
        left_edges.append(l_loc)
        right_edges.append(r_loc)

        next_step = iterator_wp.next(resolution)
        if not next_step:
            break
        iterator_wp = next_step[0]
        accumulated_dist += resolution

    return left_edges, right_edges
