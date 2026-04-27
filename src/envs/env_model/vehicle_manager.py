import carla
import random
import numpy as np
import gymnasium as gym

from src.utils import get_logger
from typing import Dict, Any, Tuple, Optional, Union, List

logger = get_logger('vehicle_manager')
class VehicleManager:
    def __init__(self, world, client, config):
        self.world = world
        self.client = client
        self.config = config
        self.ego_vehicle = None
        self.npc_vehicles = []

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

        # 尝试生成车辆（最多重试20次）
        max_attempts = 20
        for attempt in range(max_attempts):
            spawn_point = random.choice(candidate_points)
            self.ego_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.ego_vehicle is not None:
                break
            logger.info(f"出生点被占用，重试第 {attempt + 1} 次...")

        if self.ego_vehicle is None:
            msg = f"无法初始化自车，最后尝试位置：x={spawn_point.location.x:.2f}, y={spawn_point.location.y:.2f}"
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info(f"主车已生成，位置：x={spawn_point.location.x:.2f}, y={spawn_point.location.y:.2f}")
        return self.ego_vehicle

    def spawn_npcs(self,tm_port,sync_mode):
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

    def apply_control(self, action,action_space):
        if self.ego_vehicle is None:
            raise RuntimeError("环境没有重置. 请先 reset()。")

        if not isinstance(action_space, gym.spaces.Discrete):
            # 接收论文物理动作：[a (m/s²), δ (rad)]
            a_phy = float(np.clip(action[0], -3.0, 1.5))  # 物理加速度
            delta_phy = float(np.clip(action[1], -0.4, 0.4))  # 物理前轮转角

            # 1. 物理加速度 → Carla油门/刹车
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

            # 2. 物理前轮转角 → Carla转向
            # Carla steer [-1,1] 对应最大转角约0.6rad，我们用0.4rad对应0.67的steer
            steer_val = np.interp(delta_phy, [-0.4, 0.4], [-0.67, 0.67])
            steer_val = float(np.clip(steer_val, -1.0, 1.0))

            # 强制禁止倒车，论文场景不需要
            reverse_flag = False

            # 诊断日志（debug 级别，默认不输出，可通过设置 logger 级别开启）
            logger.debug(
                f"动作映射 - 输入 [a={a_phy:.4f}, δ={delta_phy:.4f}] -> "
                f"油门={throttle_val:.4f}, 刹车={brake_val:.4f}, steer={steer_val:.4f}"
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

    def cleanup_finished_vehicles(self,sumo_simulation,synchronization):
        pass
        """清理 SUMO 中已销毁（到达终点或离开路网）的车辆"""
        # try:
        #     # 获取当前真正在 SUMO 中存活的车辆 ID 列表
        #     active_sumo_vehicles = set(sumo_simulation.traci.vehicle.getIDList())
        #     sumo2carla = synchronization._sumo2carla
        #
        #     # 必须用 list() 包裹，因为我们要在遍历时修改字典本身
        #     for sumo_veh_id in list(sumo2carla.keys()):
        #         if sumo_veh_id not in active_sumo_vehicles:
        #             # 说明该车辆已经到达终点，被 SUMO 移除了
        #             carla_actor_id = sumo2carla[sumo_veh_id]
        #             carla_actor = self.world.get_actor(carla_actor_id)
        #
        #             # 1. 销毁 CARLA 中的物理残留车辆
        #             if carla_actor is not None and carla_actor.is_alive:
        #                 carla_actor.destroy()
        #                 logger.debug(f"销毁已到达终点的残留 CARLA 车辆：{carla_actor_id}")
        #
        #             # 2. 【核心修复】从同步字典中彻底移除，防止 bridge_helper 接着同步它报错
        #             del sumo2carla[sumo_veh_id]
        #
        #             # 顺手把 _carla2sumo 映射也清了（双向安全）
        #             if hasattr(synchronization,
        #                        '_carla2sumo') and carla_actor_id in synchronization._carla2sumo:
        #                 del synchronization._carla2sumo[carla_actor_id]
        #
        # except Exception as e:
        #     logger.warning(f"清理 SUMO 车辆时发生异常: {e}")```
