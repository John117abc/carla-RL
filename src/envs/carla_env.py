# src/envs/carla_env.py

import gymnasium as gym
import numpy as np
import carla
import random
import time
import math
import traceback
from typing import Dict, Any, Tuple, Optional, Union, List
from gymnasium import spaces
from src.envs.sensors import CameraSensor, CollisionSensor, LaneInvasionSensor,ObstacleSensor,IMUSensor
from src.carla_utils import get_compass, world_to_vehicle_frame, RoutePlanner, get_ocp_observation, \
    get_current_lane_forward_edges, draw_text_at_location, draw_lines_between_points, draw_points, \
    get_ocp_observation_ego_frame, ego_to_world_coordinate, batch_world_to_ego
from src.utils import get_logger, RunningNormalizer, unpack_ocp_numpy
from src.configs.constant import (LAYERS_TO_REMOVE_1,
                                  LAYERS_TO_REMOVE_2,
                                  LAYERS_TO_REMOVE_3,
                                  LAYERS_TO_REMOVE_4,
                                  BIRTH_POINT,END_POINT,BIRTH_YAW)

# SUMO 相关导入
import traci
from sumo_sync.sumo_integration.sumo_simulation import SumoSimulation
from sumo_sync.sumo_integration.carla_simulation import CarlaSimulation
from sumo_sync.sumo_integration.bridge_helper import BridgeHelper
from sumo_sync.run_synchronization import SimulationSynchronization

logger = get_logger(name='carla_env')

class CarlaEnv(gym.Env):
    """
    CARLA 环境封装，兼容 Gymnasium 0.28+。
    支持连续/离散动作空间，多模态观测。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        carla_config: Dict[str, Any],
        env_config: Dict[str, Any],
        sumo_config: Dict[str, Any] = None,
        render_mode: Optional[str] = None,
        is_eval: bool = False
    ):
        super().__init__()
        self.carla_cfg = carla_config
        self.env_cfg = env_config
        self.render_mode = render_mode

        # 是否启用sumo控制交通
        self.enable_sumo = env_config['traffic']['enable_sumo']

        # 初始化 CARLA 客户端
        self.carla_host = self.carla_cfg["host"]
        self.carla_port = self.carla_cfg["port"]
        self.client = carla.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(self.carla_cfg["timeout"])
        self.world = self.client.load_world(self.env_cfg["world"]["map"])

        # 初始化 SUMO 仿真
        if self.enable_sumo:
            self.simulation_step_length = self.carla_cfg["fixed_delta_seconds"]  # 确保与 CARLA 同步
            self.cfg_file = sumo_config['default']['sumo_config_file']
            self.sumo_gui = sumo_config['default']['sumo_gui']
            self.sumo_host = sumo_config['default']['sumo_host']
            self.sumo_port = sumo_config['default']['sumo_port']

            self.carla_simulation = CarlaSimulation(host=self.carla_host,
                                                    port=self.carla_port,
                                                    step_length=self.simulation_step_length)

            self.sumo_simulation = SumoSimulation(self.cfg_file,
                                                  self.simulation_step_length,
                                                  host=self.sumo_host,
                                                  port=self.sumo_port,
                                                  sumo_gui=self.sumo_gui,
                                                  client_order=1)

            self.synchronization = SimulationSynchronization(self.sumo_simulation,
                                                             self.carla_simulation,
                                                             'none',  # 交通信号灯管理('carla', 'sumo', 'none')
                                                             False,  # 是否同步车颜色
                                                             False)

        else:
            # 交通管理器（即使不用背景车也建议初始化，避免端口冲突）
            self.tm_port = self.carla_cfg["world"]["traffic_manager"]["port"]
            self.synchronous_mode = self.carla_cfg["world"]["traffic_manager"]["synchronous_mode"]
            # 同步模式设置
            if self.carla_cfg["sync_mode"]:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.carla_cfg["fixed_delta_seconds"]
                self.world.apply_settings(settings)
            try:
                self.tm = self.client.get_trafficmanager(self.tm_port)
                self.tm.set_synchronous_mode(self.synchronous_mode)
                self.tm.set_global_distance_to_leading_vehicle(2.0)
                self.tm.set_hybrid_physics_mode(
                    self.carla_cfg["world"]["traffic_manager"]["hybrid_physics_mode"]
                )
            except RuntimeError as e:
                logger.error(f"⚠️ TrafficManager 初始化失败（可能端口被占用）: {e}")


            # 天气
        weather = carla.WeatherParameters(
            cloudiness=self.carla_cfg["world"]["weather"]["cloudiness"],
            precipitation=self.carla_cfg["world"]["weather"]["precipitation"],
            sun_altitude_angle=self.carla_cfg["world"]["weather"]["sun_altitude_angle"],
        )
        self.world.set_weather(weather)

        self.blueprint_library = self.world.get_blueprint_library()

        # 传感器与状态
        self.camera_sensor: Optional[CameraSensor] = None
        self.collision_sensor: Optional[CollisionSensor] = None
        self.lane_invasion_sensor: Optional[LaneInvasionSensor] = None
        self.obstacle_sensor: Optional[ObstacleSensor] = None
        self.imu_sensor: Optional[IMUSensor] = None
        self.vehicle = None
        self.step_count = 0

        # 归一化处理
        meas_dim = self._get_measurements_dim()
        self.meas_normalizer = RunningNormalizer(shape=(meas_dim,))

        # 控制是否更新归一化统计量（评估 时不更新）
        self._is_eval = is_eval

        # ocp观察模式是否为debug模式
        self._ocp_debug = True

        # 周车
        self.actors = []

        # 自车参考速度
        self.ego_ref_speed = self.env_cfg['actors']['ego']['ref_speed']

        # 动作空间
        action_type = self.env_cfg["action"]["type"]
        if action_type == "continuous":
            low = np.array([
                self.env_cfg["action"]["continuous"]["throttle_range"][0],
                self.env_cfg["action"]["continuous"]["steer_range"][0]
            ], dtype=np.float32)
            high = np.array([
                self.env_cfg["action"]["continuous"]["throttle_range"][1],
                self.env_cfg["action"]["continuous"]["steer_range"][1]
            ], dtype=np.float32)
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        elif action_type == "discrete":
            n_actions = len(self.env_cfg["action"]["discrete"]["actions"])
            self.action_space = gym.spaces.Discrete(n_actions)
        else:
            raise ValueError(f"不支持的操作类型: {action_type}")

        # 观测空间
        obs_type = self.env_cfg["obs_type"]
        obs_spaces = {}

        if "image" in obs_type:
            img_shape = (
                self.env_cfg["image"]["height"],
                self.env_cfg["image"]["width"],
                self.env_cfg["image"]["channels"]
            )
            obs_spaces["image"] = gym.spaces.Box(
                low=0, high=255, shape=img_shape, dtype=np.uint8
            )

        if "measurements" in obs_type:
            n_meas = self._get_measurements_dim()
            obs_spaces["measurements"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_meas,), dtype=np.float32
            )

        if "ocp_obs" in obs_type:
            n_meas = self._get_ocp_dim()
            obs_spaces["ocp_obs"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_meas,), dtype=np.float32
            )

        self.observation_space = gym.spaces.Dict(obs_spaces)

        # 卸载地图层级
        self._init_map_layers()

        # 初始化路径规划器
        self.route_planner = None
        self.path_locations = None
        self.ref_path_xy = None
        self.ref_path_xy_raw = None
        # 路径id
        self.current_path_id = 0

        # 静态道路坐标
        self.static_road_xy = None

        # 保存上一帧在参考线上的投影点索引（加速搜索，避免每帧从头找）
        self.last_ref_idx = 0
        # 保存上一帧的视角（可选，用于极轻微的平滑，防止参考线本身有抖动）
        self.prev_spectator_transform = None

    def _init_notice_str_world(self):
        location = self.vehicle.get_location()
        # 在车顶上方 5 米处显示文字
        self.world.debug.draw_string(
            location + carla.Location(z=5.0),
            self.carla_cfg["world"]["notice_world"],
            life_time=60.0,  # 设为0则永久
            color=carla.Color(255, 0, 0),  # 红色
            draw_shadow=True
        )

    def _get_ocp_dim(self):
        """
                获取ocp观察下的维度大小
                :return: 维度大小
                """
        dim = 0
        for key in self.env_cfg["ocp"]["include"]:
            if key == "ego":
                dim += 6
            elif key == "others":
                dim += (self.env_cfg["ocp"]["others"] * 4)
            elif key == "ref_error":
                dim += 3
            elif key == "road":
                dim += (self.env_cfg["ocp"]["num_points"] * 4)

        return dim

    def _get_measurements_dim(self):
        """
        获取measurements观察下的维度大小
        :return: 维度大小
        """
        dim = 0
        for key in self.env_cfg["measurements"]["include"]:
            if key == "speed":
                dim += 2
            elif key == "steer":
                dim += 1
            elif key == "compass":
                dim += 1
            elif key == "gps":
                dim += 2
            elif key == "imu":
                dim += 3  # acc, ang_v, yaw_rate
            else:
                dim += 1
        return dim

    def _init_map_layers(self):
        match self.carla_cfg["world"]["map_layer"]:
            case 1:
                remove_layer = LAYERS_TO_REMOVE_1
            case 2:
                remove_layer = LAYERS_TO_REMOVE_2
            case 3:
                remove_layer = LAYERS_TO_REMOVE_3
            case 4:
                remove_layer = LAYERS_TO_REMOVE_4
            case _:
                remove_layer = []
        # 批量卸载
        for layer in remove_layer:
            self.world.unload_map_layer(layer)
            logger.info(f"卸载层级: {layer}")

    def _load_map_layers(self):
        match self.carla_cfg["world"]["map_layer"]:
            case 1:
                remove_layer = LAYERS_TO_REMOVE_1
            case 2:
                remove_layer = LAYERS_TO_REMOVE_2
            case 3:
                remove_layer = LAYERS_TO_REMOVE_3
            case 4:
                remove_layer = LAYERS_TO_REMOVE_4
            case _:
                remove_layer = []
        # 批量加载
        for layer in remove_layer:
            self.world.load_map_layer(layer)
            logger.info(f"加载层级: {layer}")


    def _spawn_ego_vehicle(
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
        vehicle_bp = random.choice(blueprint_library.filter(self.env_cfg["actors"]['ego']["ego_car_type"]))
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
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is not None:
                break
            logger.info(f"出生点被占用，重试第 {attempt + 1} 次...")

        if self.vehicle is None:
            msg = f"无法初始化自车，最后尝试位置：x={spawn_point.location.x:.2f}, y={spawn_point.location.y:.2f}"
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info(f"主车已生成，位置：x={spawn_point.location.x:.2f}, y={spawn_point.location.y:.2f}")
        return self.vehicle

    def _setup_sensors(self):
        # 先清理旧传感器
        if self.camera_sensor:
            self.camera_sensor.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
        if self.obstacle_sensor:
            self.obstacle_sensor.destroy()
        if self.imu_sensor:
            self.imu_sensor.destroy()

        # 创建新传感器
        if "image" in self.env_cfg["obs_type"]:
            self.camera_sensor = CameraSensor(
                self.vehicle,
                self.world,
                width=self.env_cfg["image"]["width"],
                height=self.env_cfg["image"]["height"],
                fov=self.env_cfg["image"]["fov"],
            )
        self.collision_sensor = CollisionSensor(self.vehicle)
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle)
        self.obstacle_sensor = ObstacleSensor(self.vehicle)
        self.imu_sensor = IMUSensor(self.vehicle)
        # 等待传感器数据就绪，避免首次观测为空
        for _ in range(5):  # 最多等待 5 帧
            if self.carla_cfg["sync_mode"]:
                self.world.tick()
            else:
                time.sleep(0.05)
            if ("image" not in self.env_cfg["obs_type"]) or (self.camera_sensor.get_data() is not None):
                break

    def _get_observation(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        obs = {}
        if "image" in self.env_cfg["obs_type"]:
            img = self.camera_sensor.get_data()
            if img is None:
                img = np.zeros(
                    (self.env_cfg["image"]["height"], self.env_cfg["image"]["width"], 3),
                    dtype=np.uint8
                )
            obs["image"] = img
        # 如果是sumo直接观察周车
        self.actors = self._get_surrounding_vehicles()
        if "measurements" in self.env_cfg["obs_type"]:
            measurements = []
            v = self.vehicle.get_velocity()
            c = self.vehicle.get_control()
            # 转换为车辆坐标系
            vx,vy = world_to_vehicle_frame(v,self.vehicle.get_transform())
            compass = get_compass(self.vehicle)  # 弧度
            loc = self.vehicle.get_location()
            if self.imu_sensor.get_acceleration() is None:
                acc = 0.0
            else:
                acc = math.sqrt(self.imu_sensor.get_acceleration().x ** 2 + self.imu_sensor.get_acceleration().y ** 2 + self.imu_sensor.get_acceleration().z ** 2)

            if self.imu_sensor.get_angular_velocity() is None:
                ang_v = 0.0
            else:
                ang_v = math.sqrt(
                    self.imu_sensor.get_angular_velocity().x ** 2 + self.imu_sensor.get_angular_velocity().y ** 2 + self.imu_sensor.get_angular_velocity().z ** 2)

            for key in self.env_cfg["measurements"]["include"]:
                if key == "speed":
                    measurements.extend([vx, vy])
                elif key == "steer":
                    measurements.append(float(c.steer))
                elif key == "compass":
                    measurements.append(float(compass))
                elif key == "gps":
                    measurements.extend([float(loc.x), float(loc.y)])
                elif key == "imu":
                    measurements.extend([acc, ang_v, self.imu_sensor.get_yaw_rate()])
                else:
                    measurements.append(0.0)

            # 归一化 measurements
            meas_array = np.array(measurements, dtype=np.float32)
            if not self._is_eval:
                self.meas_normalizer.update(meas_array[None, :])
            meas_normalized = self.meas_normalizer.normalize(meas_array)

            obs["measurements"] = meas_normalized
        if "ocp_obs" in self.env_cfg["obs_type"]:
            # 获取ocp观察信息
            network_state, s_road_ego, s_ref_raw_ego, s_ref_error,s_road,s_ref_raw = get_ocp_observation_ego_frame(self.vehicle,self.imu_sensor,self.actors,self.path_locations,self.ego_ref_speed)
            obs["ocp_obs"] = network_state
            obs["s_road"] = s_road_ego
            obs["s_ref_raw"] = s_ref_raw_ego
            obs["s_ref_raw"] = s_ref_error
            # # 如果是debug模式，在训练页面上显示和各个点的连线
            if self._ocp_debug:
                self._debug_ocp(obs["ocp_obs"],s_ref_raw,s_road)
        if len(obs) == 1:
            return list(obs.values())[0]
        return obs

    def _debug_ocp(self,ocp_obs,s_ref_raw,s_road):
        ocp_obs_np = np.array(ocp_obs, dtype=np.float32).flatten().reshape([1,1,-1])
        ego_state, other_states, ref_error = unpack_ocp_numpy(ocp_obs_np,self.env_cfg['ocp']['num_points'],self.env_cfg['ocp']['others'])

        # 显示当前步数和最大步数
        step_num_text = f'now step:{self.step_count},\n max limit step:{self.env_cfg["termination"]["max_episode_steps"]}'
        # 显示ego文字
        world_ego_x,world_ego_y = ego_to_world_coordinate(ego_state[0][0][0],ego_state[0][0][1],self.vehicle.get_transform())
        draw_text_at_location(world=self.world,text=step_num_text,
                              location=np.array([world_ego_x ,world_ego_y + 20], dtype=np.float32),
                              display_time=0.002,
                              color=carla.Color(0, 0, 255))


        ego_text = f'v_lon:{ego_state[0][0][2]:.2f} \n v_lat:{ego_state[0][0][3]:.2f} \n φ:{ego_state[0][0][4]:.2f} \n 0:{ego_state[0][0][5]:.2f}'

        # 显示ego文字
        draw_text_at_location(world=self.world,text=ego_text,
                              location=np.array([world_ego_x,world_ego_y], dtype=np.float32),
                              display_time=0.002,
                              color=carla.Color(0, 0, 255))

        road_left = s_road[..., :self.env_cfg['ocp']['num_points'] * 2].reshape(self.env_cfg['ocp'][
                                                                                        'num_points'],
                                                                                    2)
        road_right = s_road[..., self.env_cfg['ocp']['num_points'] * 2:].reshape(self.env_cfg['ocp'][
                                                                                         'num_points'],
                                                                                     2)

        # 左车道
        draw_points(world=self.world,points=road_left,display_time=0.2, color=carla.Color(0, 255, 0),size=0.05)

        # 右车道
        draw_points(world=self.world, points=road_right, display_time=0.2, color=carla.Color(0, 255, 0),size=0.05)


        # 参考路径绘制
        draw_points(world=self.world, points=s_ref_raw[0:2].reshape(1,-1), display_time=0.2, color=carla.Color(0, 0, 255),size=0.05)

        # 绘制误差
        ref_error_text = f'e_p:{ref_error[0][0][0]:.2f} \n e_φ:{ref_error[0][0][1]:.2f} \n e_v:{ref_error[0][0][2]:.2f}\n'
        draw_text_at_location(world=self.world,text=ref_error_text,
                              location=np.array([world_ego_x,world_ego_y + 10.0], dtype=np.float32),
                              display_time=0.002,
                              color=carla.Color(0, 0, 255))

    def _compute_reward(self,lane_inv,collision,obstacle) -> dict[str, Any]:
        w = self.env_cfg["reward_weights"]
        r = 0.0

        v = self.vehicle.get_velocity()
        speed = np.linalg.norm([v.x, v.y, v.z])
        if speed < 2.0:
            speed_reward = -w["low_speed_penalty"]  # e.g., 2.0
        else:
            speed_reward = w["speed"] * min(speed, 10.0)

        r += speed_reward

        centering_reward = w["centering"] * lane_inv * 5.0
        r -= centering_reward

        collision_reward = 0.0
        if collision > self.env_cfg["termination"]["collision_threshold"]:
            collision_reward = w["collision"]
            r += w["collision"]

        obstacle_reward = 0.0
        if obstacle:
            obstacle_reward = w["obstacle"]
            r += obstacle_reward

        # todo 缺少高性能样本计算高性能标准：高速、保持在道路上、在正确车道

        # 航向对齐简化奖励
        r += w["angle"] * (1.0 - abs(self.vehicle.get_transform().rotation.yaw) / 180.0)

        return {
            'total_reward':float(r),
            'speed_reward':speed_reward,
            'centering_reward':centering_reward,
            'collision_reward':collision_reward,
            'obstacle_reward':obstacle_reward,
            'high_speed_reward':0.0,
            'right_lane_reward':0.0
        }

    def _check_termination(self,lane_inv,collision,obstacle) -> Tuple[bool, bool, Dict[str, Any]]:
        # 计算速度信息存到info中
        vx,vy = world_to_vehicle_frame(self.vehicle.get_velocity(),self.vehicle.get_transform())
        v = math.sqrt(vx**2 + vy**2)

        info = {"collision": False,
                "off_route": False,
                "TimeLimit.truncated": False,
                'speed': v}

        # 碰撞actors和障碍物公用一个终止条件
        if collision > self.env_cfg["termination"]["collision_threshold"] or obstacle:
            info["collision"] = True

        if self.step_count >= self.env_cfg["termination"]["max_episode_steps"]:
            info["TimeLimit.truncated"] = True

        return info["collision"] ,info["TimeLimit.truncated"] ,info

    def _spawn_background_traffic(self):
        """
        生成背景交通车辆
        """
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [bp for bp in blueprints if int(bp.get_attribute('number_of_wheels')) == 4]

        num_vehicles = self.env_cfg["actors"]['others']["num_vehicles"]

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
                v.set_autopilot(True, self.tm_port)
                self.actors.append(v)  # 只有在成功生成后才添加到列表

        if self.carla_cfg["sync_mode"]:
            self.world.tick()  # 确保车辆完全激活
        logger.info(f"成功生成 {len(self.actors)} 辆背景交通车辆（TM 端口: {self.tm_port}）。")

    def _get_surrounding_vehicles(self, max_distance=50.0):
        """
        获取 ego 车辆周围一定范围内的所有 SUMO 背景车辆（排除 ego 自身）
        """
        if self.vehicle is None:
            return []

        # 获取世界中所有车辆
        all_vehicles = self.world.get_actors().filter('vehicle.*')

        surrounding = []
        ego_transform = self.vehicle.get_transform()
        ego_location = ego_transform.location

        for v in all_vehicles:
            # 跳过 ego 车辆（通过 role_name 或 id 判断）
            if v.id == self.vehicle.id or v.attributes.get('role_name') == 'hero':
                continue

            dist = ego_location.distance(v.get_location())
            if dist <= max_distance:
                surrounding.append(v)

        return surrounding

    def _place_spectator_above_vehicle(self):
        """
        完全还原你最初的原版代码
        仅修复：车辆Z轴震动导致的抖动 + 上下坡显示异常
        兼容纯2D参考线 [x,y]，无任何报错
        """
        if self.vehicle is None or len(self.dense_ref_path) < 2:
            return

        # 1. 获取车辆当前位置
        vehicle_loc = self.vehicle.get_location()
        vehicle_xy = np.array([vehicle_loc.x, vehicle_loc.y])

        # 2. 找到车辆在密集参考线上的「最近投影点」（用上一帧索引加速）
        search_window = 100
        start_idx = max(0, self.last_ref_idx - search_window)
        end_idx = min(len(self.dense_ref_path), self.last_ref_idx + search_window)
        search_path = self.dense_ref_path[start_idx:end_idx]

        # 计算距离，找到最近点
        dists = np.hypot(search_path[:, 0] - vehicle_xy[0], search_path[:, 1] - vehicle_xy[1])
        local_min_idx = np.argmin(dists)
        closest_idx = start_idx + local_min_idx
        self.last_ref_idx = closest_idx  # 更新索引供下一帧使用

        # 3. 计算参考线在该点的「切线方向」（保证视角平行于道路）
        lookahead_idx = min(closest_idx + 5, len(self.dense_ref_path) - 1)
        lookbehind_idx = max(closest_idx - 1, 0)

        forward_pt = self.dense_ref_path[lookahead_idx]
        backward_pt = self.dense_ref_path[lookbehind_idx]

        # 计算切线角度（yaw）
        dx = forward_pt[0] - backward_pt[0]
        dy = forward_pt[1] - backward_pt[1]
        ref_yaw = np.degrees(np.arctan2(dy, dx))

        # 4. 设置相机位置：参考线投影点正上方
        x_offset = 10.0
        z_height = 50.0

        # 把x_offset投影到参考线方向上
        offset_x = x_offset * np.cos(np.radians(ref_yaw))
        offset_y = x_offset * np.sin(np.radians(ref_yaw))

        # ===================== 唯一修改：平滑车辆Z轴，解决抖动+上下坡 =====================
        # 初始化平滑Z值
        if not hasattr(self, 'smoothed_z'):
            self.smoothed_z = vehicle_loc.z
        # 一阶低通滤波：消除车辆Z轴抖动，保留上下坡大趋势
        alpha = 0.1  # 平滑系数，越小越稳
        self.smoothed_z = self.smoothed_z * (1 - alpha) + vehicle_loc.z * alpha

        target_location = carla.Location(
            x=float(self.dense_ref_path[closest_idx, 0] + offset_x),
            y=float(self.dense_ref_path[closest_idx, 1] + offset_y),
            z=float(self.smoothed_z + z_height)  # 用平滑后的Z，不抖+上下坡正常
        )

        # 5. 设置相机朝向（原版完全不变）
        target_rotation = carla.Rotation(
            pitch=-90.0,
            yaw=float(ref_yaw),
            roll=0.0
        )
        target_transform = carla.Transform(target_location, target_rotation)

        # 6. 极轻微平滑（原版完全不变）
        if self.prev_spectator_transform is None:
            final_transform = target_transform
        else:
            lerp_factor = 0.5
            prev_loc = self.prev_spectator_transform.location
            final_loc = carla.Location(
                x=prev_loc.x + (target_location.x - prev_loc.x) * lerp_factor,
                y=prev_loc.y + (target_location.y - prev_loc.y) * lerp_factor,
                z=prev_loc.z + (target_location.z - prev_loc.z) * lerp_factor
            )

            prev_rot = self.prev_spectator_transform.rotation
            delta_yaw = target_rotation.yaw - prev_rot.yaw
            if delta_yaw > 180: delta_yaw -= 360
            if delta_yaw < -180: delta_yaw += 360
            final_yaw = prev_rot.yaw + delta_yaw * lerp_factor

            final_rot = carla.Rotation(
                pitch=prev_rot.pitch + (target_rotation.pitch - prev_rot.pitch) * lerp_factor,
                yaw=final_yaw,
                roll=0.0
            )
            final_transform = carla.Transform(final_loc, final_rot)

        # 7. 应用视角（原版完全不变）
        self.world.get_spectator().set_transform(final_transform)
        self.prev_spectator_transform = final_transform

    def step(self, action):
        if self.vehicle is None:
            raise RuntimeError("环境没有重置. 请先 reset()。")

        # 初始化控制量默认值，避免分支变量未定义
        throttle_val = 0.0
        brake_val = 0.0
        steer_val = 0.0
        reverse_flag = False

        if not isinstance(self.action_space, gym.spaces.Discrete):
            # ========== 【最终对齐】接收论文物理动作：[a (m/s²), δ (rad)] ==========
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
        self.vehicle.apply_control(ctrl)

        # ========== 仿真推进逻辑（修复模运算括号）==========
        if self.carla_cfg["sync_mode"]:
            if self.enable_sumo:
                self.synchronization.tick()
                self.world.tick()
                # 【修复1】给 step_count+1 加括号，保证模运算优先级正确
                if (self.step_count + 1) % 1000 == 0:
                    self._cleanup_finished_vehicles()
            else:
                self.world.tick()
        else:
            if self.enable_sumo:
                self.synchronization.tick()
                # 【修复1】同上
                if (self.step_count + 1) % 1000 == 0:
                    self._cleanup_finished_vehicles()
            else:
                self.world.wait_for_tick()

        # ========== 在此处调用视角跟随函数（仿真推进后，车辆位置已更新）==========
        self._place_spectator_above_vehicle()

        self.step_count += 1

        # ========== 传感器、观测、奖励、终止判断（保持不变）==========
        lane_inv = self.lane_invasion_sensor.get_count()
        collision = self.collision_sensor.get_intensity()
        obstacle = self.obstacle_sensor.is_obstacle_ahead(self.env_cfg["termination"]["obstacle_threshold"])
        obs = self._get_observation()
        reward = self._compute_reward(lane_inv, collision, obstacle)
        terminated, truncated, info = self._check_termination(lane_inv, collision, obstacle)
        info.update(reward)
        info['road_state'] = obs["s_road"]  # 单独存道路信息
        info['ref_path_xy'] = self.ref_path_xy
        left_pts, right_pts = get_current_lane_forward_edges(self.vehicle, self.world)
        info['static_road_left'] = [[item.x, item.y] for item in left_pts]
        info['static_road_right'] = [[item.x, item.y] for item in right_pts]

        return obs, reward['total_reward'], terminated, truncated, info

    def _destroy_all_sensors(self):
        for sensor in [self.camera_sensor, self.collision_sensor,
                       self.lane_invasion_sensor, self.obstacle_sensor, self.imu_sensor]:
            if sensor is not None:
                sensor.destroy()
        # 重置引用
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.obstacle_sensor = None
        self.imu_sensor = None

    def _destroy_all_npc_vehicles(self):
        # 获取所有车辆
        actors = self.world.get_actors()
        vehicles = [a for a in actors if "vehicle" in a.type_id]

        # 排除自车（如果还存在）
        npc_vehicles = [v for v in vehicles if v.id != self.vehicle.id] if self.vehicle else vehicles

        destroyed_count = 0
        for v in npc_vehicles:
            try:
                v.destroy()
                destroyed_count += 1
            except Exception as e:
                # 车辆可能已被自动销毁（如碰撞后），忽略错误
                pass  # 或者 logger.debug(f"跳过已销毁车辆 {v.id}: {e}")

        logger.info(f"已销毁 {destroyed_count} 辆周车（共尝试 {len(npc_vehicles)} 辆）")

    def _cleanup_finished_vehicles(self):
        """
        安全清理已完成行程的 SUMO 背景车辆及其 CARLA 映射。
        适用于长期 RL 训练，防止内存溢出和同步崩溃。
        """
        try:
            # 获取当前所有 SUMO 车辆 ID
            sumo_vehicle_ids = traci.vehicle.getIDList()
        except Exception as e:
            print(f"[Cleanup] Failed to get SUMO vehicle list: {e}")
            return

        # 主车标识前缀
        EGO_PREFIXES = ("ego")

        for veh_id in sumo_vehicle_ids:
            # 跳过主车
            if any(veh_id.startswith(prefix) for prefix in EGO_PREFIXES):
                continue

            try:
                # 判断车辆是否“已完成”
                # 检查是否在最后一条 edge 上（通用，兼容旧版 SUMO）
                route = traci.vehicle.getRoute(veh_id)
                current_edge = traci.vehicle.getRoadID(veh_id)

                if not route or not current_edge:
                    # 车辆可能已 teleport 到无效 edge（如 ':xxx'），视为可清理
                    should_remove = True
                else:
                    # 检查当前 edge 是否是 route 的最后一个
                    last_edge = route[-1]
                    second_last_edge = route[-2] if len(route) >= 2 else last_edge
                    should_remove = (current_edge == last_edge or current_edge == second_last_edge)

                # 检查速度是否接近 0（防止未到终点但堵死）
                if should_remove:
                    speed = traci.vehicle.getSpeed(veh_id)
                    if speed > 1.0:  # 还在移动，暂不删
                        should_remove = False

                # 执行安全移除
                if should_remove:
                    # 1. 销毁 CARLA 对应的 actor
                    if hasattr(self, 'sumo2carla_ids') and veh_id in self.sumo2carla_ids:
                        carla_id = self.sumo2carla_ids[veh_id]
                        try:
                            carla_actor = self.carla_world.get_actor(carla_id)
                            if carla_actor is not None and carla_actor.is_alive:
                                carla_actor.destroy()
                        except Exception as e:
                            print(f"[Cleanup] Failed to destroy CARLA actor {carla_id}: {e}")

                        # 清理双向映射
                        self.sumo2carla_ids.pop(veh_id, None)
                        if hasattr(self, 'carla2sumo_ids'):
                            self.carla2sumo_ids.pop(carla_id, None)

                    # 从 SUMO 移除车辆
                    try:
                        traci.vehicle.remove(veh_id, reason=traci.constants.REMOVE_ARRIVED)
                        traci.vehicle.unsubscribe(veh_id)  # 取消订阅，减少开销
                        print(f"[Cleanup] Removed finished vehicle: {veh_id}")
                    except traci.exceptions.TraCIException:
                        pass  # 可能已被移除

            except traci.exceptions.TraCIException as e:
                # 车辆可能在检查后瞬间被 SUMO 删除（race condition）
                print(f"[Cleanup] Vehicle {veh_id} disappeared during cleanup: {e}")
                continue
            except Exception as e:
                print(f"[Cleanup] Unexpected error for {veh_id}: {e}")
                continue

    def _interpolate_ref_path(self, sparse_path, interval=0.5):
        """
        对稀疏参考线进行插值加密
        :param sparse_path: 原始稀疏参考线 [[x1,y1], [x2,y2], ...]
        :param interval: 插值后点与点的间隔（米），越小越密集，0.5~1.0米推荐
        :return: 加密后的参考线 array(N, 2)
        """
        if len(sparse_path) < 2:
            return np.array(sparse_path)

        sparse_array = np.array(sparse_path)
        # 计算原始路径上每个点之间的累积距离
        diffs = np.diff(sparse_array, axis=0)
        seg_dists = np.hypot(diffs[:, 0], diffs[:, 1])
        cum_dists = np.concatenate(([0], np.cumsum(seg_dists)))
        total_dist = cum_dists[-1]

        # 生成新的密集距离点
        num_new_points = int(np.ceil(total_dist / interval))
        new_dists = np.linspace(0, total_dist, num_new_points)

        # 对x和y分别进行线性插值
        x_new = np.interp(new_dists, cum_dists, sparse_array[:, 0])
        y_new = np.interp(new_dists, cum_dists, sparse_array[:, 1])

        return np.column_stack((x_new, y_new))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        try:
            # 销毁所有旧资源
            # 清除周车
            if not self.enable_sumo:
                for actor in self.actors:
                    if actor is not None and actor.is_alive:
                        actor.destroy()
                # 生成周车
                self.actors = []
                self._spawn_background_traffic()
            else:
                pass    # 由sumo控制周车生命周期
            # 销毁自车
            if self.vehicle is not None:
                self._destroy_all_sensors()
                self.vehicle.destroy()
                self.vehicle = None

            # 生成自车
            if self.env_cfg["actors"]['ego']["random_place"]:
                self._spawn_ego_vehicle()
            else:
                self._spawn_ego_vehicle(BIRTH_POINT[self.env_cfg["world"]["map"]],BIRTH_YAW[self.env_cfg["world"]["map"]])


            self._setup_sensors()

            # 重置计数器
            self.step_count = 0

            # 规划静态路径
            self.route_planner = RoutePlanner(self.world, self.carla_cfg["world"]["sampling_resolution"])

            self.route_plane(END_POINT[self.env_cfg["world"]["map"]])

            # 对参考线进行插值加密（解决参考线稀疏问题）
            self.dense_ref_path = self._interpolate_ref_path(self.ref_path_xy_raw)

            obs = self._get_observation()
            info = {}  # 可扩展
            info['road_state'] = obs["s_road"]  # 单独存道路信息
            info['ref_path_locations'] = self.ref_path_xy
            # 重置观察者视角
            self._place_spectator_above_vehicle()

            # 路径id+1
            self.current_path_id +=1

            # 初始化提示文字，有便与区分不同环境训练
            self._init_notice_str_world()

            # 确保同步模式正确
            if self.carla_cfg["sync_mode"]:
                settings = self.world.get_settings()
                if not settings.synchronous_mode or settings.fixed_delta_seconds != self.carla_cfg["fixed_delta_seconds"]:
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = self.carla_cfg["fixed_delta_seconds"]
                    self.world.apply_settings(settings)

            # tick 一次确保状态稳定
            if not self.enable_sumo and self.carla_cfg["sync_mode"]:
                self.tm = self.client.get_trafficmanager(self.tm_port)
                self.tm.set_random_device_seed(22)  # 重置随机性
                self.world.tick()
            elif self.enable_sumo and self.carla_cfg["sync_mode"]:
                # # 关闭旧的同步器
                # if hasattr(self, 'synchronization'):
                #     self.synchronization.close()
                # # 重新创建 SUMO 同步器
                # self.carla_simulation = CarlaSimulation(
                #     host=self.carla_host,
                #     port=self.carla_port,
                #     step_length=self.simulation_step_length
                # )
                # self.sumo_simulation = SumoSimulation(
                #     self.cfg_file,
                #     self.simulation_step_length,
                #     host=self.sumo_host,
                #     port=self.sumo_port,
                #     sumo_gui=self.sumo_gui,
                #     client_order=1
                # )
                # self.synchronization = SimulationSynchronization(
                #     self.sumo_simulation,
                #     self.carla_simulation,
                #     'none',
                #     False,
                #     False,
                #     False
                # )
                self.synchronization.tick()

        except Exception as e:
            logger.error(f"重置环境失败: {e}")
            self.close()
            raise
        return obs, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            if self.camera_sensor and self.camera_sensor.get_data() is not None:
                return self.camera_sensor.get_data()
            else:
                return np.zeros(
                    (self.env_cfg["image"]["height"], self.env_cfg["image"]["width"], 3),
                    dtype=np.uint8
                )
        return None

    def close(self):
        """补充地图层级恢复 + 全量Actor清理的close函数"""
        try:
            # 恢复所有卸载的地图层级
            self._load_map_layers()
            # 销毁车辆和传感器
            if self.vehicle is not None:
                self._destroy_all_sensors()
                self.vehicle.destroy()
                self.vehicle = None

            # if self.enable_sumo:
            # self.synchronization.close()

            # 清理所有额外生成的Actor（如行人、其他车辆等
            # 如果你有存储所有Actor的列表（比如self.actors），补充这部分清理
            if hasattr(self, 'actors') and self.actors:
                for actor in self.actors:
                    try:
                        if actor.is_alive:
                            actor.destroy()
                            print(f"销毁额外Actor: {actor.id}")
                    except Exception as e:
                        print(f"销毁Actor {actor.id} 失败: {e}")
                self.actors = []

            # 恢复异步模式
            if self.carla_cfg["sync_mode"]:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)

            # 重置世界
            if hasattr(self, 'client') and self.client is not None:
                self.client.reload_world()
            print("Carla资源清理完成")

        except Exception as e:
            print(f"close函数执行出错: {e}")
            traceback.print_exc()

    def route_plane(self,end_location:carla.Location):
        start_location = self.vehicle.get_transform().location
        end_location = end_location
        logger.info("开始进行路径规划")
        self.route_planner.set_destination(start_location,end_location)
        path_locations = self.route_planner.get_route()
        # 可视化路径
        for i, loc in enumerate(path_locations):
            self.world.debug.draw_point(loc, size=0.01, color=carla.Color(0, 255, 0), life_time=60.0)
            if i > 0:
                self.world.debug.draw_line(
                    path_locations[i - 1], loc,
                    thickness=0.1,
                    color=carla.Color(255, 0, 0),
                    life_time=60.0
                )
        self.path_locations = path_locations
        self.ref_path_xy = batch_world_to_ego(path_locations, self.vehicle.get_transform())
        self.ref_path_xy_raw = [[item.x, item.y] for item in path_locations]
        logger.info(f"路径规划成功！已规划{len(path_locations)}个坐标点")

    def get_random_driving_action(self):
        """
        生成用于探索的随机驾驶动作。
        返回: [steer, accel]
        范围: 均为 [-1, 1]

        逻辑优化:
        - 转向: 完全随机 [-1, 1]
        - 加速: 70% 概率给正向油门 (0.3 ~ 1.0)，30% 概率刹车或滑行 (-1.0 ~ 0.2)
          (这样能保证车大概率是向前动的，避免原地不动)
        """

        # # 1. 转向角 (Steering): -1 (左满) 到 1 (右满)
        # steer = np.random.uniform(-1.0, 1.0)
        #
        # # 2. 加速度/油门 (Acceleration): -1 (急刹) 到 1 (地板油)
        # # 策略：大部分时间给油，让车动起来
        # if np.random.rand() < 0.7:
        #     # 70% 概率：给一个明显的油门 (0.3 到 1.0)，确保克服静摩擦力
        #     accel = np.random.uniform(0.3, 1.0)
        # else:
        #     # 30% 概率：刹车或松油门 (-1.0 到 0.2)
        #     accel = np.random.uniform(-1.0, 0.2)

        # return np.array([accel,steer], dtype=np.float32)

        # 强制正加速度：归一化加速度 [0.2, 1.0] → 物理量 [0, 1.5]m/s²，绝对不会刹车
        a_norm = np.random.uniform(0.2, 1.0, size=1)
        # 小范围转向，避免转圈
        delta_norm = np.random.uniform(-0.2, 0.2, size=1)
        norm_action = np.concatenate([a_norm, delta_norm])

    @property
    def is_eval(self):
        return self._is_eval

