# src/envs/carla_env.py

import gymnasium as gym
import numpy as np
import carla
import random
import time
import math
from typing import Dict, Any, Tuple, Optional, Union

from src.envs.sensors import CameraSensor, CollisionSensor, LaneInvasionSensor,ObstacleSensor,IMUSensor
from src.carla_utils import get_compass,world_to_vehicle_frame,RoutePlanner,get_ocp_observation
from src.utils import get_logger,RunningNormalizer
from src.configs.constant import (LAYERS_TO_REMOVE_1,
                                  LAYERS_TO_REMOVE_2,
                                  LAYERS_TO_REMOVE_3,
                                  LAYERS_TO_REMOVE_4,
                                  BIRTH_POINT)

logger = get_logger(name='carla_env')

class CarlaEnv(gym.Env):
    """
    CARLA 环境封装，兼容 Gymnasium 0.28+。
    支持连续/离散动作空间，多模态观测（图像 + 测量值）。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        carla_config: Dict[str, Any],
        env_config: Dict[str, Any],
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.carla_cfg = carla_config
        self.env_cfg = env_config
        self.render_mode = render_mode

        # 初始化 CARLA 客户端
        self.client = carla.Client(self.carla_cfg["host"], self.carla_cfg["port"])
        self.client.set_timeout(self.carla_cfg["timeout"])
        self.world = self.client.load_world(self.env_cfg["world"]["map"])

        # 同步模式设置
        if self.carla_cfg["sync_mode"]:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.carla_cfg["fixed_delta_seconds"]
            self.world.apply_settings(settings)

        # 交通管理器（即使不用背景车也建议初始化，避免端口冲突）
        self.tm_port = self.carla_cfg["world"]["traffic_manager"]["port"]
        self.synchronous_mode = self.carla_cfg["world"]["traffic_manager"]["synchronous_mode"]
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

        # 归一化
        self.ocp_normalizer = RunningNormalizer(shape=(66,))

        # 控制是否更新归一化统计量（评估 时不更新）
        self._is_eval = False

        # ocp观察模式是否为debug模式
        self._ocp_debug = False

        # 周车
        self.actors = []

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
            # 有自车，周车，道路，参考4个维度数据，每个维度有6项
            # 而且周车观察8辆车，所以维度是66
            obs_spaces["ocp_obs"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(66,), dtype=np.float32
            )

        if len(obs_spaces) == 1:
            self.observation_space = list(obs_spaces.values())[0]
        else:
            self.observation_space = gym.spaces.Dict(obs_spaces)

        # 卸载地图层级
        self._init_map_layers()

        # 初始化路径规划器
        self.route_planner = None
        self.path_locations = None

        # 路径id
        self.current_path_id = 0

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


    def _spawn_ego_vehicle(self,spawn_point_index: Optional[int] = None) -> carla.Vehicle:
        """
        在世界中生成主控车辆（ego vehicle）。

        参数：
            spawn_point_index (Optional[int]): 指定出生点索引。若为 None，则随机选择。
        返回：
            carla.Vehicle: 生成的主车对象。
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter(self.env_cfg["actors"]["ego_car"]))
        vehicle_bp.set_attribute('role_name', 'ego')

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("当前地图没有可用的出生点！")
        if spawn_point_index is not None:
            if spawn_point_index >= len(spawn_points):
                raise ValueError(f"出生点索引 {spawn_point_index} 超出范围（共 {len(spawn_points)} 个）")
            spawn_point = spawn_points[spawn_point_index]
        else:
            spawn_point = random.choice(spawn_points)

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.vehicle is None:
            logger.info(f'出生点被占用，换一个')
            spawn_point = spawn_points[spawn_point_index+1]
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

        if self.vehicle is None:
            logger.info(f'出生点还是被占用，随机选一个')
            # 重试20次
            for count in range(20):
                spawn_point = random.choice(spawn_points)
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
                if self.vehicle is not None:
                    break

        if self.vehicle is None:
            logger.error(f'无法初始化自车:x:{spawn_point.location.x} y:{spawn_point.location.y}')
            RuntimeError(f'无法初始化自车:x:{spawn_point.location.x} y:{spawn_point.location.y}')

        self.vehicle.set_autopilot(False,tm_port=self.tm_port)
        logger.info(f"主车已生成，位置：{spawn_point.location}")

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
            ocp_obs = get_ocp_observation(self.vehicle,self.imu_sensor,self.actors,self.path_locations,self.world.get_map())
            state_ego_flat = np.asarray(ocp_obs[0])
            state_other_flat = np.asarray(ocp_obs[1]).flatten()
            state_road_flat = np.asarray(ocp_obs[2])
            state_ref_flat = np.asarray(ocp_obs[3])
            state_all = np.concatenate([state_ego_flat,state_other_flat,state_road_flat,state_ref_flat], axis=0)
            if not self._is_eval:
                self.ocp_normalizer.update(state_all[None, :])
            obs["ocp_obs"] = [state_all,ocp_obs]
            # 如果是debug模式，在训练页面上显示和各个点的连线
            if self._ocp_debug:
                self._debug_ocp(ocp_obs)
        if len(obs) == 1:
            return list(obs.values())[0]
        return obs

    def _debug_ocp(self,ocp_obs):
        """
        如果启动ocp的debug模式，会在地图上显示自车与ref的连线
        周车的连线，道路边缘的连线
        """
        ego_location = self.vehicle.get_location()
        road_location = ocp_obs[2]
        # 自车与道路边缘的连线
        self.world.debug.draw_line(
            ego_location, carla.Location(x=road_location[0],y=road_location[1]),
            thickness=0.05,
            color=carla.Color(255, 0, 0),
            life_time=10.0
        )

        # 自车与参考路径的连线
        ref_location = ocp_obs[3]
        self.world.debug.draw_line(
            ego_location, carla.Location(x=ref_location[0],y=ref_location[1]),
            thickness=0.05,
            color=carla.Color(0, 255, 0),
            life_time=10.0
        )

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

        num_vehicles = self.env_cfg["actors"]["num_vehicles"]

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

    def _place_spectator_above_vehicle(self):
        """将观察者摄像头放置在车辆后上方，便于查看"""
        if self.vehicle is None:
            return

        vehicle_transform = self.vehicle.get_transform()
        offset = carla.Location(x=-5.0, y=-20.0, z=15.0)
        spectator_transform = carla.Transform(
            vehicle_transform.location + offset,
            carla.Rotation(pitch=-30.0, yaw=120.0, roll=0.0)
        )
        self.world.get_spectator().set_transform(spectator_transform)

    def step(self, action):
        if self.vehicle is None:
            raise RuntimeError("环境没有重置. 请先 reset()。")

        # 解析动作
        if isinstance(self.action_space, gym.spaces.Discrete):
            throttle, steer = self.env_cfg["action"]["discrete"]["actions"][action]
        else:
            throttle, steer = float(action[0]), float(action[1])
            throttle = np.clip(throttle, -1.0, 1.0)
            steer = np.clip(steer, -1.0, 1.0)

        # 应用控制
        ctrl = carla.VehicleControl()
        if throttle >= 0:
            ctrl.throttle = float(throttle)
            ctrl.brake = 0.0
        else:
            ctrl.throttle = 0.0
            ctrl.brake = float(-throttle)
        ctrl.steer = float(steer)
        self.vehicle.apply_control(ctrl)

        # 推进仿真
        if self.carla_cfg["sync_mode"]:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        self.step_count += 1

        # 获取传感器状态
        lane_inv = self.lane_invasion_sensor.get_count()
        collision = self.collision_sensor.get_intensity()
        obstacle = self.obstacle_sensor.is_obstacle_ahead(self.env_cfg["termination"]["obstacle_threshold"])
        # 获取观察状态
        obs = self._get_observation()
        reward = self._compute_reward(lane_inv,collision,obstacle)
        terminated, truncated, info = self._check_termination(lane_inv,collision,obstacle)

        return obs, reward, terminated, truncated, info

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

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        try:
            # 销毁所有旧资源
            if self.vehicle is not None:
                self._destroy_all_sensors()
                self.vehicle.destroy()
                self.vehicle = None

            # 清除周车
            for actor in self.actors:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            self.actors = []

            # 生成新车
            if self.env_cfg["actors"]["random_place"]:
                self._spawn_ego_vehicle()
            else:
                self._spawn_ego_vehicle(BIRTH_POINT[self.env_cfg["world"]["map"]])

            self._setup_sensors()
            # 生成周车
            self._spawn_background_traffic()

            # 重置计数器
            self.step_count = 0

            # 规划静态路径
            self.route_planner = RoutePlanner(self.world, self.carla_cfg["world"]["sampling_resolution"])
            self.route_plane(end_x = -229.562531,end_y = -15.150213,end_z = 0.300000)

            obs = self._get_observation()
            info = {}  # 可扩展
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
            if self.carla_cfg["sync_mode"]:
                self.world.tick()

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
        # 销毁所有旧资源
        if self.vehicle is not None:
            self._destroy_all_sensors()
            self.vehicle.destroy()
            self.vehicle = None

        # 恢复异步模式
        if self.carla_cfg["sync_mode"]:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

    def route_plane(self,end_x,end_y,end_z):
        start_location = self.vehicle.get_transform().location
        end_location = carla.Location(x=end_x,y=end_y,z=end_z)
        logger.info("开始进行路径规划")
        self.route_planner.set_destination(start_location,end_location)
        path_locations = self.route_planner.get_route()
        # 可视化路径
        for i, loc in enumerate(path_locations):
            self.world.debug.draw_point(loc, size=0.1, color=carla.Color(0, 255, 0), life_time=240.0)
            if i > 0:
                self.world.debug.draw_line(
                    path_locations[i - 1], loc,
                    thickness=0.05,
                    color=carla.Color(255, 0, 0),
                    life_time=240.0
                )
        self.path_locations = path_locations
        logger.info(f"路径规划成功！已规划{len(path_locations)}个坐标点")

    @property
    def is_eval(self):
        return self._is_eval

