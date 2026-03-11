# src/carla_utils/ocp_setup.py
import carla
import numpy as np
import torch
from numpy import ndarray, dtype

import src.envs.sensors
import math
from typing import List, Dict, Union, Any, Optional


def get_ocp_observation(ego_vehicle:carla.Vehicle,
                        ego_imu:src.envs.sensors.IMUSensor,
                        other_vehicles: List[carla.Vehicle],
                        path_locations: List[carla.Location],
                        world_map: carla.Map):
    """
    获取文章中的各个参考信息
    :param world_map: 世界地图
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
    ocp_road = get_road_edge_points(ego_vehicle, world_map)       # 道路边缘
    # ocp_road = get_closest_lane_edge_point(ego_vehicle)     # 车道边缘
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
) -> list[dict] | ndarray[Any, dtype[Any]]:
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


def get_closest_lane_edge_point(ego_vehicle: carla.Vehicle) -> ndarray[Any, dtype[Any]]:
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
    default_longitudinal_velocity: float =20.0  # 默认纵向速度（m/s），可根据任务调整
) -> ndarray[Any, dtype[Any]] | None:
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
    # 为了舒适性+4因为直线距离转弯很不舒服
    ref_index = path_locations.index(ref_location) + 4
    if ref_index > len(path_locations):
        ref_index = path_locations.index(ref_location)
    ref_location = path_locations[ref_index]
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



def get_road_edge_points(ego_vehicle: carla.Vehicle,world_map: carla.Map):
    """
    获取当前道路最外侧边缘的近似点（基于最外侧车道中心线偏移）
    :param world_map: 世界map
    :param ego_vehicle: 自车
    :return: carla.Location，表示道路边缘上离自车最近的点（近似）
    """
    # 获取车辆当前位置的 waypoint
    vehicle_loc = ego_vehicle.get_location()
    wp = world_map.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

    def get_one_road_edge_points(current_wp,direction):
        # 向指定方向遍历到最外侧车道
        if direction == 'right':
            while True:
                next_wp = current_wp.get_right_lane()
                if next_wp is None or next_wp.lane_type != carla.LaneType.Driving:
                    break
                current_wp = next_wp
            # 最右侧车道的右边界 ≈ 道路右边缘
            lane_width = current_wp.lane_width
            # 从车道中心向右偏移 lane_width/2 得到边缘
            right_vec = current_wp.transform.get_right_vector()
            edge_location = current_wp.transform.location + carla.Location(
                x=right_vec.x * lane_width / 2,
                y=right_vec.y * lane_width / 2,
                z=0.0
            )
            return edge_location

        elif direction == 'left':
            while True:
                next_wp = current_wp.get_left_lane()
                if next_wp is None or next_wp.lane_type != carla.LaneType.Driving:
                    break
                current_wp = next_wp
            # 最左侧车道的左边界 ≈ 道路左边缘
            lane_width = current_wp.lane_width
            left_vec = current_wp.transform.get_right_vector()  # 注意：左 = -right
            edge_location = current_wp.transform.location - carla.Location(
                x=left_vec.x * lane_width / 2,
                y=left_vec.y * lane_width / 2,
                z=0.0
            )
            return edge_location
        return None

    # 获取左右两侧道路边缘点
    right_edge = get_one_road_edge_points(wp, 'right')
    left_edge = get_one_road_edge_points(wp, 'left')

    # 计算自车到两个边缘的距离，取最近的
    dist_right = vehicle_loc.distance(right_edge)
    dist_left = vehicle_loc.distance(left_edge)

    if dist_right < dist_left:
        nearest_road_edge = right_edge
    else:
        nearest_road_edge = left_edge

    return [nearest_road_edge.x,nearest_road_edge.y,0,0,0,0]


def predict_other_next(initial_obs_states, dt):
    """
    简单预测周围车辆状态 (Constant Velocity Model).
    这不是一个可学习的模型，只是一个几何计算。

    Args:
        initial_obs_states: [Batch, N_obs, 6] (x, y, vx, vy, psi, omega) - 来自 s_0
        t: 当前时间步索引
        dt: 时间步长

    Returns:
        current_obs_states: [Batch, N_obs, 6]
    """
    # 提取初始速度和航向
    # 假设在短时域内，他车速度和航向不变
    vx = initial_obs_states[:,:,2]
    vy = initial_obs_states[:,:,3]

    x_init = initial_obs_states[:,:, 0]
    y_init = initial_obs_states[:,:,1]

    x_curr = x_init + vx * dt
    y_curr = y_init + vy * dt

    # 其他状态 (vx, vy, psi, omega) 假设保持不变
    psi_init = initial_obs_states[:,:, 4]
    omega_init = initial_obs_states[:,:,5]
    psi_curr = psi_init + omega_init * dt

    # 重组状态
    current_obs_states = torch.stack([x_curr, y_curr, vx, vy, psi_curr, omega_init], dim=-1)

    return current_obs_states


def get_ref_observation_torch(
        ego_x: torch.Tensor,  # [B] 自车 x 坐标
        ego_y: torch.Tensor,  # [B] 自车 y 坐标
        path_locations: torch.Tensor,  # [N, 2] 或 [B, N, 2] 参考路径点 (x, y)
        default_longitudinal_velocity: float = 20.0,
        lookahead_offset: int = 4
) -> torch.Tensor:
    """
    【Tensor 版本】获取自车到参考路径的状态信息，保持梯度流通
    使用软注意力替代 argmin，所有操作可微分

    Args:
        ego_x: 自车 x 坐标 [B]
        ego_y: 自车 y 坐标 [B]
        path_locations: 参考路径点 [N, 2] 或 [B, N, 2]
        default_longitudinal_velocity: 默认纵向速度 (m/s)
        lookahead_offset: 前瞻偏移量

    Returns:
        torch.Tensor [B, 6]: [x, y, longitudinal_velocity, 0, yaw_deg, 0]
    """
    if path_locations is None or path_locations.shape[-2] < 2:
        return None

    B = ego_x.shape[0]
    device = ego_x.device

    # ---------------------------------------------------------
    # 1. 维度标准化
    # ---------------------------------------------------------
    if path_locations.dim() == 2:
        # [N, 2] -> [B, N, 2]
        path_locations = path_locations.unsqueeze(0).expand(B, -1, -1).clone()
    elif path_locations.dim() == 3 and path_locations.shape[0] != B:
        path_locations = path_locations.unsqueeze(0).expand(B, -1, -1).clone()

    N = path_locations.shape[1]  # 路径点数

    # ---------------------------------------------------------
    # 2. 计算自车到所有路径点的距离 [B, N]
    # ---------------------------------------------------------
    ego_coords = torch.stack([ego_x, ego_y], dim=-1).unsqueeze(1)  # [B, 1, 2]
    diff = path_locations - ego_coords  # [B, N, 2]
    dists = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-6)  # [B, N]

    # ---------------------------------------------------------
    # 3. ✅ 软注意力：用距离的倒数作为权重（可微！）
    # ---------------------------------------------------------
    inverse_dists = 1.0 / (dists + 1e-6)  # [B, N]
    weights = torch.softmax(inverse_dists, dim=-1)  # [B, N]

    # ---------------------------------------------------------
    # 4. ✅ 软前瞻偏移：将权重向前偏移 lookahead_offset
    # ---------------------------------------------------------
    offset_weights = torch.zeros_like(weights)  # [B, N]

    # 将每个位置 i 的权重移到 i + lookahead_offset
    for i in range(N):
        target_i = min(i + lookahead_offset, N - 1)
        offset_weights[:, target_i] = offset_weights[:, target_i] + weights[:, i]

    # 归一化偏移后的权重
    offset_weights = offset_weights / (offset_weights.sum(dim=-1, keepdim=True) + 1e-6)

    # ---------------------------------------------------------
    # 5. ✅ 加权平均获取目标点坐标（可微！）
    # ---------------------------------------------------------
    target_pts = torch.sum(offset_weights.unsqueeze(-1) * path_locations, dim=1)  # [B, 2]

    x = target_pts[:, 0]  # [B]
    y = target_pts[:, 1]  # [B]

    # ---------------------------------------------------------
    # 6. ✅ 软方式计算航向角（可微！）
    # ---------------------------------------------------------
    # 创建下一个位置的偏移权重（lookahead_offset + 1）
    next_offset_weights = torch.zeros_like(weights)
    for i in range(N):
        target_i = min(i + lookahead_offset + 1, N - 1)
        next_offset_weights[:, target_i] = next_offset_weights[:, target_i] + weights[:, i]

    next_offset_weights = next_offset_weights / (next_offset_weights.sum(dim=-1, keepdim=True) + 1e-6)

    next_pts = torch.sum(next_offset_weights.unsqueeze(-1) * path_locations, dim=1)  # [B, 2]

    dx = next_pts[:, 0] - x  # [B]
    dy = next_pts[:, 1] - y  # [B]

    # ✅ 使用 torch.atan2 而不是 math.atan2
    yaw_rad = torch.atan2(dy, dx)
    # yaw_deg = yaw_rad * (180.0 / 3.141592653589793)  # ✅ 使用 Python float 常量（没问题，乘法可微）
    yaw_output = yaw_rad
    # ---------------------------------------------------------
    # 7. 构建输出 [B, 6]
    # ---------------------------------------------------------
    long_vel_tensor = torch.full_like(x, default_longitudinal_velocity)
    zeros = torch.zeros_like(x)

    # ✅ 返回 torch.Tensor 而不是 numpy array
    result = torch.stack([x, y, long_vel_tensor, zeros, yaw_output, zeros], dim=-1)  # [B, 6]

    return result


def get_current_lane_forward_edges(vehicle, world, distance=50.0, resolution=0.5):
    """
    获取自车当前所在车道前方指定距离内的【左边缘】和【右边缘】坐标序列。

    参数:
        vehicle: carla.Vehicle 对象 (自车)
        world: carla.World 对象
        distance: 前方搜索距离 (米), 默认50米
        resolution: 采样步长 (米), 默认0.5米

    返回:
        tuple(list, list):
            - left_edges: 左侧边缘坐标列表 [carla.Location, ...]
            - right_edges: 右侧边缘坐标列表 [carla.Location, ...]
            (两个列表长度一致，索引 i 对应同一横截面的左右边缘)
    """
    map = world.get_map()
    vehicle_loc = vehicle.get_location()

    # 1. 获取自车当前位置的中心 Waypoint
    # 注意：这里修正为 project_to_road=True (根据你之前的报错信息)
    try:
        current_wp = map.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    except Exception as e:
        print(f"获取 Waypoint 失败: {e}")
        return [], []

    if current_wp is None:
        return [], []

    left_edges = []
    right_edges = []

    # 辅助函数：给定一个中心 waypoint，计算其所在路段的左右边缘坐标
    def calculate_edge_locations(center_wp):
        """
        从 center_wp 开始，向左/右找到最外侧车道，并计算边缘点坐标。
        返回 (left_edge_loc, right_edge_loc)
        """
        # --- 找左边缘 ---
        left_most_wp = center_wp
        while True:
            next_left = left_most_wp.get_left_lane()
            # 确保下一个点是驾驶车道且属于同一条路 (防止跨到对面车道或人行道)
            if next_left is None or next_left.lane_type != carla.LaneType.Driving:
                break
            # 可选：检查 road_id 是否相同，防止跨越隔离带进入对向车道
            # if next_left.road_id != center_wp.road_id: break
            left_most_wp = next_left

        # 计算左边缘坐标：最左车道中心 - (车道宽/2) * 右向量
        # 注意：get_right_vector() 指向车道右侧，所以左侧是 减去 该向量
        lane_width_l = left_most_wp.lane_width
        right_vec_l = left_most_wp.transform.get_right_vector()
        left_edge_loc = left_most_wp.transform.location - carla.Location(
            x=right_vec_l.x * lane_width_l / 2.0,
            y=right_vec_l.y * lane_width_l / 2.0,
            z=left_most_wp.transform.location.z  # 保持Z轴高度
        )

        # --- 找右边缘 ---
        right_most_wp = center_wp
        while True:
            next_right = right_most_wp.get_right_lane()
            if next_right is None or next_right.lane_type != carla.LaneType.Driving:
                break
            right_most_wp = next_right

        # 计算右边缘坐标：最右车道中心 + (车道宽/2) * 右向量
        lane_width_r = right_most_wp.lane_width
        right_vec_r = right_most_wp.transform.get_right_vector()
        right_edge_loc = right_most_wp.transform.location + carla.Location(
            x=right_vec_r.x * lane_width_r / 2.0,
            y=right_vec_r.y * lane_width_r / 2.0,
            z=right_most_wp.transform.location.z
        )

        return left_edge_loc, right_edge_loc

    # 2. 主循环：向前迭代
    # 先处理当前点（如果需要包含车头正下方的边缘，取消下面注释）
    # l_loc, r_loc = calculate_edge_locations(current_wp)
    # left_edges.append(l_loc)
    # right_edges.append(r_loc)

    # 获取第一步
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

        # 向前移动
        next_step = iterator_wp.next(resolution)
        if not next_step:
            break

        # 分叉路口处理：取第一个分支（通常为主路）
        # 如需更精确，可在此处比较向量夹角
        iterator_wp = next_step[0]
        accumulated_dist += resolution

    return left_edges, right_edges


def get_road_observation_torch(
        ego_x: torch.Tensor,  # [B]
        ego_y: torch.Tensor,  # [B]
        static_road_xy: torch.Tensor,  # [N, 2, 2] 或 [B, N, 2, 2]
):
    """
    【Agent 侧】使用软注意力获取最近道路边缘点信息，保持梯度流通
    返回格式兼容原函数：[edge_x, edge_y, 0, 0, 0, 0]
    """
    B = ego_x.shape[0]
    device = ego_x.device

    # ---------------------------------------------------------
    # 1. 维度处理
    # ---------------------------------------------------------
    if static_road_xy.dim() == 3:
        # [N, 2, 2] -> [B, N, 2, 2]
        static_road_xy = static_road_xy.unsqueeze(0).expand(B, -1, -1, -1).clone()

    N = static_road_xy.shape[1]  # 道路点数

    # ---------------------------------------------------------
    # 2. 计算自车到所有道路点的距离 [B, N]
    # ---------------------------------------------------------
    ego_coords = torch.stack([ego_x, ego_y], dim=-1).unsqueeze(1)  # [B, 1, 2]

    # 取左右边缘的中点作为道路中心参考
    road_center = (static_road_xy[:, :, 0, :] + static_road_xy[:, :, 1, :]) / 2  # [B, N, 2]

    diff = road_center - ego_coords  # [B, N, 2]
    dists = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-6)  # [B, N]

    # ---------------------------------------------------------
    # 3. ✅ 软注意力：用距离的倒数作为权重（可微！）
    # ---------------------------------------------------------
    inverse_dists = 1.0 / (dists + 1e-6)  # [B, N]
    weights = torch.softmax(inverse_dists, dim=-1)  # [B, N]

    # ---------------------------------------------------------
    # 4. ✅ 加权平均获取最近边缘点（可微！）
    # ---------------------------------------------------------
    # 分别获取左右边缘的加权平均
    left_edge = torch.sum(weights.unsqueeze(-1) * static_road_xy[:, :, 0, :], dim=1)  # [B, 2]
    right_edge = torch.sum(weights.unsqueeze(-1) * static_road_xy[:, :, 1, :], dim=1)  # [B, 2]

    # 计算自车到左右边缘的距离
    ego_coords_2d = torch.stack([ego_x, ego_y], dim=-1)  # [B, 2]
    dist_to_left = torch.sqrt(torch.sum((left_edge - ego_coords_2d) ** 2, dim=-1) + 1e-6)  # [B]
    dist_to_right = torch.sqrt(torch.sum((right_edge - ego_coords_2d) ** 2, dim=-1) + 1e-6)  # [B]

    # ✅ 软选择：用 sigmoid 权重融合左右边缘（可微！）
    # 距离近的边缘权重更大
    edge_select_weight = torch.sigmoid(dist_to_right - dist_to_left)  # [B]
    # dist_to_right - dist_to_left > 0 说明左边更近，weight > 0.5，更多选择左边缘

    # 加权融合左右边缘
    nearest_edge = edge_select_weight.unsqueeze(-1) * left_edge + \
                   (1.0 - edge_select_weight).unsqueeze(-1) * right_edge  # [B, 2]

    edge_x = nearest_edge[:, 0]  # [B]
    edge_y = nearest_edge[:, 1]  # [B]

    # ---------------------------------------------------------
    # 5. 构建输出 [B, 6]（兼容原函数格式）
    # ---------------------------------------------------------
    zeros = torch.zeros_like(edge_x)

    result = torch.stack([
        edge_x,  # 道路边缘 x 坐标
        edge_y,  # 道路边缘 y 坐标
        zeros,  # 占位
        zeros,  # 占位
        zeros,  # 占位
        zeros  # 占位
    ], dim=-1)  # [B, 6]

    return result
