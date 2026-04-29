import logging
import time
import carla
import random
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple, Optional, Union, List

from src.utils import get_logger
from src.carla_utils.vehicle_control import PIDLongitudinalController, world_to_vehicle_frame

logger = get_logger('vehicle_manager', level=logging.INFO)


class VehicleManager:
    def __init__(self, world, client, config):
        self.world = world
        self.client = client
        self.config = config
        self.ego_vehicle = None
        self.npc_vehicles = []

        # 纵向 PID 控制器（暂不使用，保留用于未来可选扩展）
        self.pid_lon = PIDLongitudinalController(K_P=1.0, K_I=0.05, K_D=0.1, dt=0.1)

        # 转向平滑用状态（用于低通滤波）
        self.steer_smooth = None

    def _try_spawn_with_offset(self, vehicle_bp, base_transform, background_vehicles, occupancy_radius=2.5, max_offset=10.0):
        """
        在 base_transform 附近沿道路前进方向偏移尝试生成车辆。
        返回 (vehicle, transform_used) 或 (None, None)。
        """
        offsets = [0.0]
        d = 2.0
        while d <= max_offset:
            offsets.append(d)
            offsets.append(-d)
            d += 2.0

        forward = base_transform.get_forward_vector()
        for off in offsets:
            loc_off = base_transform.location + forward * off
            # 如果该位置不在道路上，尝试投影到道路，但不强制
            transform_off = carla.Transform(loc_off, base_transform.rotation)

            # 占用检查：只需要检查附近是否存在任何背景车辆（role_name != 'hero'）
            occupied = False
            for v in background_vehicles:
                if v.get_location().distance(loc_off) < occupancy_radius:
                    occupied = True
                    break
            if occupied:
                continue

            # 尝试生成
            vehicle = self.world.try_spawn_actor(vehicle_bp, transform_off)
            if vehicle is not None:
                return vehicle, transform_off

        return None, None

    def spawn_ego_vehicle(
            self,
            spawn_point_index: Optional[Union[int, List[carla.Location], Dict[str, float]]] = None,
            yaw=0
    ) -> carla.Vehicle:
        """
        在世界中生成主控车辆（ego vehicle）。

        参数：
            spawn_point_index (Optional[Union[int, List[carla.Location], Dict[str, float]]]):
                - 若为 int：使用地图预设 spawn points 中对应索引的位置。
                - 若为 List[carla.Location]：从这些自定义位置中随机选择一个。
                - 若为 Dict[str, float]：形如 {"x": 100, "y": 200, "z": 0.6}，表示一个自定义位置。
                - 若为 None：从所有地图预设 spawn points 中随机选择。

        返回：
            carla.Vehicle: 成功生成的主车对象。

        异常：
            RuntimeError: 当无法在多次尝试后生成车辆时抛出。
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter(self.config["actors"]['ego']["ego_car_type"]))
        vehicle_bp.set_attribute('role_name', 'hero')

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("当前地图没有可用的出生点！")

        # 构建候选 spawn point 列表（Transform）
        candidate_points: List[carla.Transform] = []

        if spawn_point_index is None:
            # 随机选择一个预设 spawn point
            candidate_points = [sp for sp in spawn_points]
        elif isinstance(spawn_point_index, int):
            # 指定预设索引
            if spawn_point_index < 0 or spawn_point_index >= len(spawn_points):
                raise ValueError(f"出生点索引 {spawn_point_index} 超出范围（共 {len(spawn_points)} 个）")
            candidate_points = [spawn_points[spawn_point_index]]
        elif isinstance(spawn_point_index, list):
            # 列表形式的自定义位置
            if not spawn_point_index:
                raise ValueError("自定义出生点列表不能为空")
            for loc in spawn_point_index:
                if not isinstance(loc, carla.Location):
                    raise TypeError(f"列表中的项必须是 carla.Location，但得到的是 {type(loc)}")
                candidate_points.append(carla.Transform(loc, carla.Rotation(yaw=yaw)))
        elif isinstance(spawn_point_index, dict):
            # 字典形式：{"x": ..., "y": ..., "z": ...}
            required_keys = {"x", "y", "z"}
            if not required_keys.issubset(spawn_point_index.keys()):
                missing = required_keys - spawn_point_index.keys()
                raise ValueError(f"缺少必要键：{missing}")
            try:
                location = carla.Location(
                    x=spawn_point_index["x"],
                    y=spawn_point_index["y"],
                    z=spawn_point_index["z"]
                )
                candidate_points.append(carla.Transform(location, carla.Rotation(yaw=yaw)))
            except Exception as e:
                raise ValueError(f"无法创建 Location：{e}")
        else:
            raise TypeError(f"不支持的 spawn_point_index 类型: {type(spawn_point_index)}")

        # 提前获取背景车辆（排除 hero），以便高效地进行占用检查
        all_actors = self.world.get_actors().filter('vehicle.*')
        background_vehicles = [v for v in all_actors if v.attributes.get('role_name') != 'hero']

        max_attempts = 20
        desired_transform = None  # 记录最近一次尝试的候选点，用于兜底传送

        for attempt in range(max_attempts):
            spawn_point = random.choice(candidate_points)
            desired_transform = spawn_point

            # 【核心改进】在一个候选点周围沿道路方向偏移尝试生成
            vehicle, used_transform = self._try_spawn_with_offset(
                vehicle_bp, spawn_point, background_vehicles,
                occupancy_radius=2.5, max_offset=10.0
            )

            if vehicle is not None:
                self.ego_vehicle = vehicle
                logger.info(
                    f"主车已在偏移位置生成，偏移={used_transform.location.distance(spawn_point.location):.2f}m，"
                    f"位置：x={used_transform.location.x:.2f}, y={used_transform.location.y:.2f}"
                )
                return self.ego_vehicle
            else:
                logger.info(f"出生点 {spawn_point.location} 及其周围偏移位置均被占用，尝试第 {attempt+1} 次")

        # 所有重试均失败，启动兜底传送
        logger.warning("所有出生点尝试均失败，将尝试从任意空闲点生成，并通过 teleport 传送到期望位置")
        fallback_vehicle = None
        fallback_spawn = None

        # 寻找一个未被占用的空闲出生点
        for sp in spawn_points:
            occupied = any(v.get_location().distance(sp.location) < 2.5 for v in background_vehicles)
            if not occupied:
                fallback_vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
                if fallback_vehicle is not None:
                    fallback_spawn = sp
                    break

        if fallback_vehicle is None:
            msg = "无法在任何空闲出生点生成主车，背景车辆可能过于密集"
            logger.error(msg)
            raise RuntimeError(msg)

        # 选定目标传送位置
        if desired_transform is None:
            # 极端情况：如果连尝试都没执行，则把第一个候选点作为目标（一般不会）
            desired_transform = candidate_points[0]
        fallback_vehicle.set_transform(desired_transform, teleport=True)
        self.ego_vehicle = fallback_vehicle

        logger.info(
            f"兜底传送成功，ego 从 {fallback_spawn.location} 传送到 {desired_transform.location}"
        )
        return self.ego_vehicle

    def spawn_npcs(self, tm_port, sync_mode):
        """
        生成背景交通车辆
        """
        actors = []
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [bp for bp in blueprints if int(bp.get_attribute('number_of_wheels')) == 4]

        num_vehicles = self.config["actors"]['others']["num_vehicles"]

        spawn_points = self.world.get_map().get_spawn_points()
        if len(spawn_points) < num_vehicles:
            logger.warning(
                f'出生点数量 ({len(spawn_points)}) 少于请求车辆数 ({num_vehicles})，将生成 {len(spawn_points)} 辆。')
            num_vehicles = len(spawn_points)

        for i in range(num_vehicles):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'background')

            # 修改索引以避免越界
            spawn_point = spawn_points[i % len(spawn_points)]

            v = self.world.try_spawn_actor(blueprint, spawn_point)
            if v is not None:  # 检查是否成功生成
                v.set_autopilot(True, tm_port)
                actors.append(v)  # 只有在成功生成后才添加到列表

        if sync_mode:
            self.world.tick()  # 确保车辆完全激活
        logger.info(f"成功生成 {len(actors)} 辆背景交通车辆（TM 端口: {tm_port}）。")

        return actors

    def apply_control(self, action, action_space):
        if self.ego_vehicle is None:
            raise RuntimeError("环境没有重置. 请先 reset()。")

        if not isinstance(action_space, gym.spaces.Discrete):
            # 接收论文物理动作：[a (m/s²), δ (rad)]
            a_phy = float(np.clip(action[0], -3.0, 1.5))       # 物理加速度
            delta_phy = float(np.clip(action[1], -0.4, 0.4))   # 物理前轮转角

            # ---------- 纵向控制：简单线性映射（恢复之前可工作的方案）----------
            if a_phy > 0.1:
                # 正加速：[0.1, 1.5] → 油门[0.1, 1.0]
                throttle_val = np.interp(a_phy, [0.1, 1.5], [0.1, 1.0])
                brake_val = 0.0
            elif a_phy < -0.1:
                # 刹车：[-3.0, -0.1] → 刹车[0.1, 1.0]
                throttle_val = 0.0
                brake_val = np.interp(abs(a_phy), [0.1, 3.0], [0.1, 1.0])
            else:
                # 滑行
                throttle_val = 0.0
                brake_val = 0.0

            # ---------- 转向映射与平滑 ----------
            steer_val_raw = np.interp(delta_phy, [-0.4, 0.4], [-0.67, 0.67])
            steer_val_raw = float(np.clip(steer_val_raw, -1.0, 1.0))

            if self.steer_smooth is None:
                self.steer_smooth = steer_val_raw
            else:
                self.steer_smooth = 0.8 * self.steer_smooth + 0.2 * steer_val_raw
            steer_val = float(np.clip(self.steer_smooth, -1.0, 1.0))

            reverse_flag = False

            logger.debug(
                f"[VEHICLE_CTRL] a_phy={a_phy:.3f}, delta_phy={delta_phy:.3f}, "
                f"油门={throttle_val:.3f}, 刹车={brake_val:.3f}, steer_filtered={steer_val:.3f}"
            )
        else:
            throttle_val = 0.0
            brake_val = 0.0
            steer_val = 0.0
            reverse_flag = False

        # 应用控制
        ctrl = carla.VehicleControl()
        ctrl.throttle = throttle_val
        ctrl.brake = brake_val
        ctrl.steer = steer_val
        ctrl.reverse = reverse_flag
        ctrl.hand_brake = False
        self.ego_vehicle.apply_control(ctrl)

    def get_vehicle_state(self):
        # 原_carla_env.py中_get_vehicle_state的完整逻辑
        pass

    def cleanup(self):
        # 清理周车
        self.cleanup_ego()
        self.cleanup_ego()

    def cleanup_npc(self):
        # 清理周车
        for actor in self.npc_vehicles:
            try:
                if actor.is_alive:
                    actor.destroy()
                    print(f"销毁额外Actor: {actor.id}")
            except Exception as e:
                print(f"销毁Actor {actor.id} 失败: {e}")
        self.npc_vehicles = []

    def cleanup_ego(self):
        if self.ego_vehicle is not None and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

    def get_surrounding_vehicles(self, max_distance=50.0):
        """
        获取 ego 车辆周围一定范围内的所有 SUMO 背景车辆（排除 ego 自身）
        """
        if self.ego_vehicle is None:
            self.npc_vehicles = []

        # 获取世界中所有车辆
        all_vehicles = self.world.get_actors().filter('vehicle.*')

        surrounding = []
        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location

        for v in all_vehicles:
            # 跳过 ego 车辆（通过 role_name 或 id 判断）
            if v.id == self.ego_vehicle.id or v.attributes.get('role_name') == 'hero':
                continue

            dist = ego_location.distance(v.get_location())
            if dist <= max_distance:
                surrounding.append(v)

        self.npc_vehicles = surrounding

    def cleanup_finished_vehicles(self, sumo_simulation, synchronization):
        pass
