# src/envs/carla_env.py

import gymnasium as gym
import numpy as np
import carla
import random
import time
from typing import Dict, Any, Tuple, Optional, Union

from src.envs.sensors import CameraSensor, CollisionSensor, LaneInvasionSensor
from src.carla_utils.vehicle_control import get_compass


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

        # === 初始化 CARLA 客户端 ===
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
            self.tm.set_hybrid_physics_mode(
                self.carla_cfg["world"]["traffic_manager"]["hybrid_physics_mode"]
            )
        except RuntimeError as e:
            print(f"⚠️ TrafficManager 初始化失败（可能端口被占用）: {e}")
            # 继续运行（不影响主车）

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
        self.vehicle = None
        self.step_count = 0

        # === 动作空间 ===
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

        # === 观测空间 ===
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
            meas_keys = self.env_cfg["measurements"]["include"]
            n_meas = 0
            for key in meas_keys:
                if key == "gps":
                    n_meas += 2
                elif key == "imu":
                    n_meas += 3
                else:
                    n_meas += 1
            obs_spaces["measurements"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_meas,), dtype=np.float32
            )

        if len(obs_spaces) == 1:
            self.observation_space = list(obs_spaces.values())[0]
        else:
            self.observation_space = gym.spaces.Dict(obs_spaces)

        # 注意：不在 __init__ 中调用 reset()！由用户显式调用

    def _spawn_ego_vehicle(self):
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("没有可用的重生点。")
        transform = random.choice(spawn_points)
        vehicle_bp = self.blueprint_library.filter("vehicle.tesla.model3")[0]
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
        if self.vehicle is None:
            raise RuntimeError("生成自我载具失败。")
        self.vehicle.set_autopilot(False,tm_port=self.tm_port)
        if self.carla_cfg["sync_mode"]:
            self.world.tick()

    def _setup_sensors(self):
        # 先清理旧传感器（安全）
        if self.camera_sensor:
            self.camera_sensor.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()

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

        # 等待传感器数据就绪（关键！避免首次观测为空）
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
            speed = np.linalg.norm([v.x, v.y, v.z])
            compass = get_compass(self.vehicle)  # radians
            loc = self.vehicle.get_location()

            for key in self.env_cfg["measurements"]["include"]:
                if key == "speed":
                    measurements.append(float(speed))
                elif key == "steer":
                    measurements.append(float(c.steer))
                elif key == "compass":
                    measurements.append(float(compass))
                elif key == "gps":
                    measurements.extend([float(loc.x), float(loc.y)])
                elif key == "imu":
                    measurements.extend([0.0, 0.0, 0.0])  # 占位
                else:
                    measurements.append(0.0)

            obs["measurements"] = np.array(measurements, dtype=np.float32)

        if len(obs) == 1:
            return list(obs.values())[0]
        return obs

    def _compute_reward(self) -> float:
        w = self.env_cfg["reward_weights"]
        r = 0.0

        v = self.vehicle.get_velocity()
        speed = np.linalg.norm([v.x, v.y, v.z])
        r += w["speed"] * min(speed, 10.0)

        lane_inv = self.lane_invasion_sensor.get_count()
        r -= w["centering"] * lane_inv * 0.5

        collision = self.collision_sensor.get_intensity()
        if collision > self.env_cfg["termination"]["collision_threshold"]:
            r += w["collision"]

        # 航向对齐简化奖励
        r += w["angle"] * (1.0 - abs(self.vehicle.get_transform().rotation.yaw) / 180.0)

        return float(r)

    def _check_termination(self) -> Tuple[bool, bool, Dict[str, Any]]:
        info = {"collision": False, "off_route": False, "TimeLimit.truncated": False}

        if self.collision_sensor.get_intensity() > self.env_cfg["termination"]["collision_threshold"]:
            info["collision"] = True
            return True, False, info

        if self.step_count >= self.env_cfg["termination"]["max_episode_steps"]:
            info["TimeLimit.truncated"] = True
            return False, True, info

        return False, False, info

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

        # 获取状态
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated, truncated, info = self._check_termination()

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # 清理旧车辆与传感器
        if self.vehicle is not None:
            if self.camera_sensor:
                self.camera_sensor.destroy()
            if self.collision_sensor:
                self.collision_sensor.destroy()
            if self.lane_invasion_sensor:
                self.lane_invasion_sensor.destroy()
            self.vehicle.destroy()

        # 生成新车
        self._spawn_ego_vehicle()
        self._setup_sensors()

        # 重置计数器
        self.step_count = 0

        obs = self._get_observation()
        info = {}  # 可扩展
        # 重置观察者视角
        self._place_spectator_above_vehicle()
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
        if self.vehicle is not None:
            if self.camera_sensor:
                self.camera_sensor.destroy()
            if self.collision_sensor:
                self.collision_sensor.destroy()
            if self.lane_invasion_sensor:
                self.lane_invasion_sensor.destroy()
            self.vehicle.destroy()

        # 恢复异步模式
        if self.carla_cfg["sync_mode"]:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

    def _place_spectator_above_vehicle(self):
        """将观察者摄像头放置在车辆后上方，便于查看"""
        if self.vehicle is None:
            return

        vehicle_transform = self.vehicle.get_transform()
        # 设置 spectator 位置：在车后 6 米，高 4 米，朝向车辆
        offset = carla.Location(x=-6.0, y=0.0, z=4.0)
        spectator_transform = carla.Transform(
            vehicle_transform.location + offset,
            carla.Rotation(pitch=-20.0, yaw=vehicle_transform.rotation.yaw, roll=0.0)
        )
        self.world.get_spectator().set_transform(spectator_transform)


