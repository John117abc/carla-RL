"""
CARLA 路径规划模块。
使用 CARLA 内置的 GlobalRoutePlanner 生成从起点到终点的全局路径，
并提供当前位置到下一个路径点的距离、方向等信息，辅助 RL 智能体导航。
"""

import carla
import numpy as np
import sys
import os
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

import carla

class RoutePlanner:
    """
    全局路径规划器封装类。
    支持动态设置目标点，并提供当前车辆位置到路径的投影信息。
    """

    def __init__(self, world: carla.World, sampling_resolution: float = 2.0):
        """
        初始化路径规划器。

        参数：
            world (carla.World): CARLA 世界对象。
            sampling_resolution (float): 路径采样间隔（米），值越小路径越精细，默认 2.0 米。
        """
        self.world = world
        self.map = world.get_map()
        self.sampling_resolution = sampling_resolution

        # 初始化全局路径规划器
        from carla_agents.navigation.global_route_planner import GlobalRoutePlanner
        self._grp = GlobalRoutePlanner(self.map, sampling_resolution)

        self._route: List[carla.Location] = []  # 存储当前路径点列表
        self._target_waypoint: Optional[carla.Location] = None  # 目标终点
        self._current_index = 0  # 当前最近路径点索引

    def set_destination(self, start_location: carla.Location, end_location: carla.Location) -> bool:
        """
        设置新的起点和终点，重新规划路径。

        参数：
            start_location (carla.Location): 起始位置（通常为主车当前位置）。
            end_location (carla.Location): 目标终点位置。

        返回：
            bool: 是否成功规划路径。
        """
        try:
            # CARLA 的 GlobalRoutePlanner 接受 Location 或 Waypoint
            route = self._grp.trace_route(start_location, end_location)
            # route 是 [(Waypoint, RoadOption), ...]，我们只取位置
            self._route = [wp.transform.location for wp, _ in route]
            self._target_waypoint = end_location
            self._current_index = 0
            logger.info(f"路径规划成功：共 {len(self._route)} 个路径点。")
            return True
        except Exception as e:
            logger.error(f"路径规划失败: {e}")
            self._route = []
            return False

    def get_next_waypoint(self, vehicle_location: carla.Location, lookahead: int = 1) -> Optional[carla.Location]:
        """
        获取前方第 `lookahead` 个路径点（用于计算方向或目标）。

        参数：
            vehicle_location (carla.Location): 车辆当前位置。
            lookahead (int): 向前看几个路径点（默认 1，即下一个点）。

        返回：
            carla.Location 或 None: 下一个路径点位置。
        """
        if not self._route:
            return None

        # 找到离车辆最近的路径点索引（简单线性搜索，路径不长时足够快）
        min_dist = float('inf')
        closest_index = 0
        for i, loc in enumerate(self._route):
            dist = np.sqrt((loc.x - vehicle_location.x)**2 +
                           (loc.y - vehicle_location.y)**2 +
                           (loc.z - vehicle_location.z)**2)
            if dist < min_dist:
                min_dist = dist
                closest_index = i

        self._current_index = closest_index
        next_index = min(closest_index + lookahead, len(self._route) - 1)
        return self._route[next_index]

    def get_distance_to_end(self, vehicle_location: carla.Location) -> float:
        """
        计算车辆当前位置到路径终点的剩余路径距离（近似为直线距离）。

        更精确的做法是累加路径段长度，但此处为效率考虑使用直线距离。

        参数：
            vehicle_location (carla.Location): 车辆当前位置。

        返回：
            float: 到终点的近似距离（米）。
        """
        if not self._target_waypoint:
            return float('inf')
        dx = self._target_waypoint.x - vehicle_location.x
        dy = self._target_waypoint.y - vehicle_location.y
        return np.sqrt(dx**2 + dy**2)

    def is_near_end(self, vehicle_location: carla.Location, threshold: float = 5.0) -> bool:
        """
        判断车辆是否接近路径终点。

        参数：
            vehicle_location (carla.Location): 车辆当前位置。
            threshold (float): 距离阈值（米），默认 5 米。

        返回：
            bool: 是否到达终点附近。
        """
        return self.get_distance_to_end(vehicle_location) < threshold

    def get_route(self) -> List[carla.Location]:
        """
        获取当前完整路径（用于可视化或调试）。
        """
        return self._route.copy()

    def get_progress(self, vehicle_location: carla.Location) -> float:
        """
        （可选）计算路径完成进度（0.0 ～ 1.0）。
        简单实现：基于已走过路径点数量。

        返回：
            float: 进度比例。
        """
        if not self._route:
            return 0.0
        total_points = len(self._route)
        progress = min(self._current_index / max(1, total_points - 1), 1.0)
        return progress