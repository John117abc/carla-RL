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


def predict_other_next(initial_obs_states, t, dt):
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
    vx = initial_obs_states[:, 2]
    vy = initial_obs_states[:, 3]

    # 简单线性外推: pos(t) = pos(0) + v * t_total
    # 注意：这里直接用总时间 t*dt 从初始位置推算，比一步步累加误差更小
    total_time = t * dt

    x_init = initial_obs_states[:, 0]
    y_init = initial_obs_states[:, 1]

    x_curr = x_init + vx * total_time
    y_curr = y_init + vy * total_time

    # 其他状态 (vx, vy, psi, omega) 假设保持不变
    # 如果需要考虑他车也在转向，可以加上 omega * total_time 到 psi 上
    psi_init = initial_obs_states[:, 4]
    omega_init = initial_obs_states[:, 5]
    psi_curr = psi_init + omega_init * total_time

    # 重组状态
    current_obs_states = torch.stack([x_curr, y_curr, vx, vy, psi_curr, omega_init], dim=-1)

    return current_obs_states


def predict_ref_next_torch(
        ego_x: torch.Tensor,  # [B] 或 [B, 1]
        ego_y: torch.Tensor,  # [B] 或 [B, 1]
        path_locations: torch.Tensor,  # [B, N, 2] (N 是路径点数)
        default_longitudinal_velocity: float = 20.0,
        lookahead_offset: int = 4  # 对应原代码中的 "+4"
) -> Optional[torch.Tensor]:
    """
    [PyTorch 可导版] 复现 get_ref_observation 逻辑。

    逻辑流程：
    1. 找到距离自车最近的参考点索引 (closest_idx)。
    2. 应用前瞻偏移：target_idx = closest_idx + 4。
    3. 边界处理：如果 target_idx 超出路径长度，则回退到有效范围内。
    4. 计算 target_idx 处的航向角 (利用 target_idx 和 target_idx+1)。
    5. 返回 [x, y, vel, 0, yaw_deg, 0]。

    Args:
        ego_x: [B] 自车 x 坐标
        ego_y: [B] 自车 y 坐标
        path_locations: [B, N, 2] 参考路径点
        default_longitudinal_velocity: 默认纵向速度
        lookahead_offset: 前瞻步数 (默认为 4)

    Returns:
        torch.Tensor: [B, 6]
    """
    if path_locations is None or path_locations.shape[1] < 2:
        return None

    B, N, _ = path_locations.shape
    device = path_locations.device

    # 确保 ego 坐标形状为 [B, 1] 以便广播
    if ego_x.dim() == 1:
        ego_x = ego_x.unsqueeze(-1)
        ego_y = ego_y.unsqueeze(-1)

    ego_pos = torch.cat([ego_x, ego_y], dim=-1).unsqueeze(1)  # [B, 1, 2]

    # ---------------------------------------------------------
    # 步骤 1: 寻找最近点 (对应原代码的 for 循环 min_distance)
    # ---------------------------------------------------------
    diff = path_locations - ego_pos  # [B, N, 2]
    dists_sq = torch.sum(diff ** 2, dim=-1)  # [B, N]

    # 获取最近点的索引 [B]
    closest_idx = torch.argmin(dists_sq, dim=1)

    # ---------------------------------------------------------
    # 步骤 2: 应用前瞻偏移 (+4) (对应原代码 ref_index = ... + 4)
    # ---------------------------------------------------------
    target_idx = closest_idx + lookahead_offset

    # 边界处理：如果 target_idx >= N，则强制设为 N-1 (最后一个点)
    # 原代码逻辑：if ref_index > len: ref_index = original_idx (这里简化为 clamp 到最大值，效果类似且更平滑)
    # 为了严格复现 "如果越界则回退到最近点索引" 的逻辑稍微复杂，通常工程上直接 clamp 到 N-1 即可
    # 这里采用 clamp: max(0, min(target_idx, N-1))
    target_idx = torch.clamp(target_idx, 0, N - 1)

    # ---------------------------------------------------------
    # 步骤 3: 获取目标点坐标 (x, y)
    # ---------------------------------------------------------
    # 扩展索引形状用于 gather: [B, 1, 1]
    idx_expanded = target_idx.view(B, 1, 1).expand(-1, -1, 2)

    # 获取目标点: [B, 1, 2] -> [B, 2]
    target_pts = torch.gather(path_locations, 1, idx_expanded).squeeze(1)
    x = target_pts[:, 0]
    y = target_pts[:, 1]

    # ---------------------------------------------------------
    # 步骤 4: 计算航向角 (对应原代码 delta_x, delta_y, atan2)
    # ---------------------------------------------------------
    # 需要获取 target_idx 和 target_idx + 1 的点
    # 计算 next_idx，同样需要处理边界 (如果是最后一个点，next_idx 保持为最后一个点，或者指向前一个？)
    # 原代码：next_idx = min(idx + 1, len - 1)。即如果是最后一个点，next_idx = idx (导致 dx=0, dy=0, atan2=0 或不定)
    # 为了防止 atan2(0,0)，如果 next_idx == idx，我们通常指向前一个点，或者保持原逻辑。
    # 这里严格复现原代码逻辑：next_idx = min(idx + 1, N - 1)

    next_idx = target_idx + 1
    next_idx = torch.clamp(next_idx, 0, N - 1)  # 相当于 min(..., N-1)

    # 获取下一点坐标
    next_idx_expanded = next_idx.view(B, 1, 1).expand(-1, -1, 2)
    next_pts = torch.gather(path_locations, 1, next_idx_expanded).squeeze(1)

    # 计算差分
    dx = next_pts[:, 0] - x
    dy = next_pts[:, 1] - y

    # 计算弧度
    yaw_rad = torch.atan2(dy, dx)

    # 转换为角度 (degrees)
    yaw_deg = yaw_rad * (180.0 / math.pi)

    # 特殊处理：如果 dx=0 且 dy=0 (即 target 是最后一个点，next 也是它自己)，yaw 会是 0。
    # 这符合原代码逻辑 (min(idx+1, len-1) 会导致重合)。

    # ---------------------------------------------------------
    # 步骤 5: 构建输出
    # ---------------------------------------------------------
    long_vel_tensor = torch.full_like(x, default_longitudinal_velocity)
    zeros = torch.zeros_like(x)

    # 返回格式: [x, y, longitudinal_velocity, 0, yaw, 0]
    result = torch.stack([
        x,
        y,
        long_vel_tensor,
        zeros,
        yaw_deg,
        zeros
    ], dim=-1)

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


import torch

import torch


def predict_road_torch(
        ego_x: torch.Tensor,  # [B]
        ego_y: torch.Tensor,  # [B]
        static_road_xy: torch.Tensor  # [N, 2, 2] (N个点, 2个边缘, 2个坐标xy)
):
    """
    【逻辑 B】获取道路边缘上离自车最近的全局点
    遍历整条路径的所有点（左/右边缘），找到距离自车最近的那个点。

    :return: torch.Tensor [B, 6], 格式为 [x, y, 0, 0, 0, 0]
    """

    B = ego_x.shape[0]
    device = ego_x.device

    # 1. 维度对齐与扩展
    # static_road_xy 原始形状: [N, 2, 2]
    # 我们需要将其扩展为 [B, N, 2, 2] 以便与 Batch 中的每个 ego 进行广播计算
    if static_road_xy.dim() == 3:
        # [N, 2, 2] -> [1, N, 2, 2] -> [B, N, 2, 2]
        # expand 不会复制数据内存，非常高效
        road_expanded = static_road_xy.unsqueeze(0).expand(B, -1, -1, -1)
    else:
        raise ValueError(f"Expected static_road_xy dim 3, got {static_road_xy.dim()}")

    # 2. 构造自车坐标 Tensor [B, 1, 1, 2]
    # stack: [B, 2] -> unsqueeze: [B, 1, 2] -> unsqueeze: [B, 1, 1, 2]
    # 这样做的目的是为了让它能和 [B, N, 2, 2] 进行广播减法
    ego_xy = torch.stack([ego_x, ego_y], dim=-1).view(B, 1, 1, 2)

    # 3. 计算所有点到自车的距离
    # road_expanded: [B, N, 2, 2]
    # ego_xy:      [B, 1, 1, 2]
    # diff:        [B, N, 2, 2] (广播后)
    diff = road_expanded - ego_xy

    # 计算平方和 (x差^2 + y差^2)，保持维度 [B, N, 2] (分别对应左、右边缘的距离)
    # dim=-1 表示在最后一个维度 (xy) 上求和
    dist_squared = torch.sum(diff ** 2, dim=-1)

    # 开根号得到欧几里得距离 [B, N, 2]
    dists = torch.sqrt(dist_squared)

    # 4. 寻找全局最小值
    # 我们需要在 N (点数) 和 2 (左右边缘) 这两个维度上找最小值
    # 先将后两维展平：[B, N*2]
    dists_flat = dists.view(B, -1)

    # 找到最小值的索引 (0 到 N*2-1)
    # min_vals: [B], min_indices: [B]
    min_vals, min_indices = torch.min(dists_flat, dim=1)

    # 5. 根据索引还原坐标
    # 我们需要从原始的 road_expanded [B, N, 2, 2] 中取出对应的点
    # 首先将展平的索引转换回 (point_idx, edge_idx)
    N = static_road_xy.shape[0]

    point_indices = min_indices // 2  # 整数除法，得到是第几个点
    edge_indices = min_indices % 2  # 取余，得到是左边缘(1)还是右边缘(0)

    # 构建用于 gather 的索引
    # 我们需要提取 [B, :, point_indices, edge_indices, :]
    # PyTorch 的高级索引方式：
    batch_range = torch.arange(B, device=device)

    # 提取最近的点的坐标 [B, 2] (x, y)
    nearest_points = road_expanded[
        batch_range,
        point_indices,
        edge_indices,
        :
    ]

    # 6. 构造返回值 [B, 6]
    # 格式: [x, y, 0, 0, 0, 0]
    zeros = torch.zeros((B, 4), device=device)
    result = torch.cat([nearest_points, zeros], dim=1)

    return result
