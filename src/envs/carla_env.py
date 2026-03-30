# src/envs/carla_env_bak.py

import gymnasium as gym
import numpy as np
import carla
from typing import Dict, Any, Optional
from src.carla_utils import RoutePlanner, batch_world_to_ego, remove_only_visible_traffic_signs

from src.utils import get_logger, RunningNormalizer
from src.configs.constant import (LAYERS_TO_REMOVE_1,
                                  LAYERS_TO_REMOVE_2,
                                  LAYERS_TO_REMOVE_3,
                                  LAYERS_TO_REMOVE_4,
                                  BIRTH_POINT,END_POINT,BIRTH_YAW)

from src.envs.env_model.vehicle_manager import VehicleManager
from src.envs.env_model.sensors_manager import SensorManager
from src.envs.env_model.observation_processor import ObservationProcessor
from src.envs.env_model.sumo_integration import SumoIntegration
from src.envs.env_model.debug_visualizer import DebugVisualizer
from src.envs.env_model.reward_calculator import RewardCalculator
from src.envs.env_model.termination_checker import TerminationChecker

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
        # 1. 基础参数初始化（最核心的配置/模式变量）
        self._init_basic_params(carla_config, env_config, sumo_config, render_mode, is_eval)
        # 2. CARLA客户端&世界初始化
        self._init_carla_client()
        # 6. 归一化器初始化
        self._init_normalizer()
        # 8. 观测空间构建
        self._init_observation_space()

        self._init_module()

        # 3. 交通仿真初始化（SUMO / 原生TM 二选一）
        self._init_traffic_simulation()
        # 4. CARLA世界环境配置（天气）
        self._init_carla_world_env()
        # 7. 动作空间构建
        self._init_action_space()
        # 9. 地图层级初始化
        self._init_map_layers()

    def _init_basic_params(
            self,
            carla_config: Dict[str, Any],
            env_config: Dict[str, Any],
            sumo_config: Dict[str, Any] = None,
            render_mode: Optional[str] = None,
            is_eval: bool = False
    ):
        """初始化基础配置和模式参数（无业务逻辑，仅变量赋值）"""
        # 配置文件
        self.carla_cfg = carla_config
        self.env_cfg = env_config
        self.sumo_cfg = sumo_config
        # 渲染/评估模式
        self.render_mode = render_mode
        self._is_eval = is_eval
        # 核心开关
        self.enable_sumo = env_config['traffic']['enable_sumo']
        # OCP调试模式
        self._ocp_debug = True
        # 自车参考速度
        self.ego_ref_speed = self.env_cfg['actors']['ego']['ref_speed']

        # 计数器
        self.step_count = 0

        # 路径规划相关变量
        self.route_planner = None
        self.path_locations = None
        self.ref_path_xy = None
        self.ref_path_xy_raw = None
        self.current_path_id = 0
        self.static_road_xy = None

        # 参考线投影加速变量
        self.last_ref_idx = 0

        # 视角平滑相关变量
        self.prev_spectator_transform = None

        # 参考线插值后密集路径（后续reset中生成）
        self.dense_ref_path = None

        # Z轴平滑相关变量（后续_place_spectator_above_vehicle中初始化）
        self.smoothed_z = None

    def _init_carla_client(self):
        """初始化CARLA客户端并加载指定地图世界"""
        self.client = carla.Client(
            self.carla_cfg["host"],
            self.carla_cfg["port"]
        )
        self.client.set_timeout(self.carla_cfg["timeout"])
        self.world = self.client.load_world(self.env_cfg["world"]["map"])
        # 蓝图库（后续生成车辆/传感器需要，提前初始化）
        self.blueprint_library = self.world.get_blueprint_library()

    def _init_traffic_simulation(self):
        """初始化交通仿真：启用SUMO则初始化SUMO桥接，否则初始化CARLA原生TM"""
        if self.enable_sumo:
            # SUMO仿真初始化
            self.simulation_step_length = self.carla_cfg["fixed_delta_seconds"]
            self.sumo_integration.init_sumo(self.simulation_step_length,self.carla_cfg["host"],self.carla_cfg["port"])
        else:
            # CARLA原生TrafficManager初始化（原逻辑完全保留）
            self.tm_port = self.carla_cfg["world"]["traffic_manager"]["port"]
            self.synchronous_mode = self.carla_cfg["world"]["traffic_manager"]["synchronous_mode"]

            # 同步模式设置
            if self.carla_cfg["sync_mode"]:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.carla_cfg["fixed_delta_seconds"]
                self.world.apply_settings(settings)

            # TM初始化+异常处理
            try:
                self.tm = self.client.get_trafficmanager(self.tm_port)
                self.tm.set_synchronous_mode(self.synchronous_mode)
                self.tm.set_global_distance_to_leading_vehicle(2.0)
                self.tm.set_hybrid_physics_mode(
                    self.carla_cfg["world"]["traffic_manager"]["hybrid_physics_mode"]
                )
            except RuntimeError as e:
                logger.error(f"⚠️ TrafficManager 初始化失败（可能端口被占用）: {e}")

    def _init_carla_world_env(self):
        """初始化CARLA世界环境（目前仅天气，后续可扩展时间/光照等）"""
        weather = carla.WeatherParameters(
            cloudiness=self.carla_cfg["world"]["weather"]["cloudiness"],
            precipitation=self.carla_cfg["world"]["weather"]["precipitation"],
            sun_altitude_angle=self.carla_cfg["world"]["weather"]["sun_altitude_angle"],
        )
        self.world.set_weather(weather)

    def _init_normalizer(self):
        """初始化观测数据归一化器（仅measurements）"""
        meas_dim = self._get_measurements_dim()
        self.meas_normalizer = RunningNormalizer(shape=(meas_dim,))

    def _init_action_space(self):
        """构建Gym动作空间（连续/离散二选一，按配置）"""
        action_type = self.env_cfg["action"]["type"]
        if action_type == "continuous":
            # 连续动作空间：油门+转向
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
            # 离散动作空间：按配置的动作数量
            n_actions = len(self.env_cfg["action"]["discrete"]["actions"])
            self.action_space = gym.spaces.Discrete(n_actions)
        else:
            raise ValueError(f"不支持的操作类型: {action_type}")

    def _init_observation_space(self):
        """构建Gym观测空间（多模态Dict，按配置的obs_type）"""
        obs_type = self.env_cfg["obs_type"]
        obs_spaces = {}

        # 图像观测
        if "image" in obs_type:
            img_shape = (
                self.env_cfg["image"]["height"],
                self.env_cfg["image"]["width"],
                self.env_cfg["image"]["channels"]
            )
            obs_spaces["image"] = gym.spaces.Box(
                low=0, high=255, shape=img_shape, dtype=np.uint8
            )

        # 数值测量观测
        if "measurements" in obs_type:
            n_meas = self._get_measurements_dim()
            obs_spaces["measurements"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_meas,), dtype=np.float32
            )

        # OCP观测
        if "ocp_obs" in obs_type:
            n_meas = self._get_ocp_dim()
            obs_spaces["ocp_obs"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_meas,), dtype=np.float32
            )

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _init_notice_str_world(self):
        location = self.vehicle_manager.ego_vehicle.get_location()
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
        remove_layer = []
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

        remove_only_visible_traffic_signs(self.world)

    def _load_map_layers(self):
        load_layer = []
        match self.carla_cfg["world"]["map_layer"]:
            case 1:
                load_layer = LAYERS_TO_REMOVE_1
            case 2:
                load_layer = LAYERS_TO_REMOVE_2
            case 3:
                load_layer = LAYERS_TO_REMOVE_3
            case 4:
                load_layer = LAYERS_TO_REMOVE_4
            case _:
                load_layer = []
        # 批量加载
        for layer in load_layer:
            self.world.load_map_layer(layer)
            logger.info(f"加载层级: {layer}")


    def _init_module(self):
        # 车辆模块初始化
        self.vehicle_manager = VehicleManager(world=self.world,client=self.client,config=self.env_cfg)
        # 传感器模块初始化
        self.sensor_manager = SensorManager(world=self.world,config=self.env_cfg)
        # sumo交通模块初始化
        self.sumo_integration = SumoIntegration(carla_client=self.client,carla_world=self.world,sumo_config=self.sumo_cfg,vehicle_manager=self.vehicle_manager)
        # 观察模块初始化
        self.observation_processor = ObservationProcessor(vehicle_manager=self.vehicle_manager,
                                                          sensor_manager=self.sensor_manager,
                                                          world=self.world,
                                                          config=self.env_cfg,
                                                          normalizer=self.meas_normalizer)
        # debug模块初始化
        self.debug_visualizer = DebugVisualizer(world=self.world,vehicle_manager=self.vehicle_manager,config=self.env_cfg)
        # 奖励计算模块初始化
        self.reward_calculator = RewardCalculator(vehicle_manager=self.vehicle_manager,config=self.env_cfg)
        # 终止模块初始化
        self.termination_checker = TerminationChecker(vehicle_manager=self.vehicle_manager,sensor_manager=self.sensor_manager,config=self.env_cfg)

    def _cleanup(self):
        # 销毁所有旧资源
        # 清除周车
        if not self.enable_sumo:
            self.vehicle_manager.cleanup_npc()
        else:
            pass  # 由sumo控制周车生命周期
        # 销毁自车
        self.vehicle_manager.cleanup_ego()
        self.sensor_manager.cleanup()


    def _init_episode_state(self):
        # 规划静态路径
        ego_vehicle = self.vehicle_manager.ego_vehicle
        self.route_planner = RoutePlanner(self.world, self.carla_cfg["world"]["sampling_resolution"])

        self.path_locations = self.route_planner.route_plane(BIRTH_POINT[self.env_cfg["world"]["map"]][0],
                                                             END_POINT[self.env_cfg["world"]["map"]],240)
        self.ref_path_xy = batch_world_to_ego(self.path_locations, ego_vehicle.get_transform())
        self.ref_path_xy_raw = np.array([[item.x, item.y] for item in self.path_locations],dtype=np.float32)
        logger.info(f"路径规划成功！已规划{len(self.path_locations)}个坐标点")


    def step(self, action):
        # 1. 执行动作 → 委托 VehicleManager
        self.vehicle_manager.apply_control(action,self.action_space)
        # 2. 推进仿真步（SUMO/原生模式）
        if self.enable_sumo:
            self.sumo_integration.sync_step()

        self.step_count += 1
        # 3. 获取观测 → 委托 ObservationProcessor
        input_params = {'path_locations':self.path_locations,'ego_ref_speed':self.ego_ref_speed,'ref_offset':self.carla_cfg['world']['ref_offset']}
        obs = self.observation_processor.get_observation(self.is_eval,input_params)
        # 4. 计算奖励 → 委托 RewardCalculator
        lane_inv = self.sensor_manager.lane_invasion_sensor.get_count()
        collision = self.sensor_manager.collision_sensor.get_intensity()
        obstacle = self.sensor_manager.obstacle_sensor.is_obstacle_ahead(self.env_cfg["termination"]["obstacle_threshold"])
        reward = self.reward_calculator.compute_reward(lane_inv,collision,obstacle)
        # 5. 检查终止 → 委托 TerminationChecker
        terminated, truncated, info = self.termination_checker.check_termination(collision,obstacle,self.step_count)
        info.update(reward)
        # 6. 调试可视化（可选）→ 委托 DebugVisualizer
        if self._ocp_debug:
            self.debug_visualizer.debug_ocp(obs['ocp_obs'],obs['s_ref_raw'],obs['s_road_raw'],self.step_count,self.carla_cfg["fixed_delta_seconds"])
            self.last_ref_idx ,self.prev_spectator_transform= self.debug_visualizer.update_spectator(self.ref_path_xy_raw,self.last_ref_idx,self.prev_spectator_transform)
        # 仿真推进一步，解决debug闪烁问题
        self.world.tick()
        return obs, reward['total_reward'], terminated, truncated, info

    def pause_simulation(self):
        """同时暂停CARLA和SUMO仿真"""
        if self.enable_sumo:
            self.sumo_simulation.pause()
        if self.carla_cfg["sync_mode"]:
            self.world.tick()  # 完成当前步后暂停
        logger.info("仿真已暂停")

    def resume_simulation(self):
        """同时恢复CARLA和SUMO仿真"""
        if self.enable_sumo:
            self.sumo_simulation.resume()
        logger.info("仿真已恢复")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        sync_mode = self.carla_cfg["sync_mode"]
        # 1. 清理上一轮残留资源
        self._cleanup()
        # 2. 生成自车 → 委托 VehicleManager
        if self.env_cfg["actors"]['ego']["random_place"]:
            self.vehicle_manager.spawn_ego_vehicle()
        else:
            self.vehicle_manager.spawn_ego_vehicle(BIRTH_POINT[self.env_cfg["world"]["map"]], BIRTH_YAW[self.env_cfg["world"]["map"]])
        # 3. 初始化传感器 → 委托 SensorManager
        self.sensor_manager.setup_sensors(self.vehicle_manager.ego_vehicle,sync_mode)
        # 4. 生成NPC/同步SUMO → 委托对应模块
        if self.enable_sumo:
            self.sumo_integration.sync_reset()
        else:
            self.vehicle_manager.spawn_npcs(self.tm,sync_mode)
        # 5. 初始化路径规划
        self._init_episode_state()
        # 6. 获取初始观测 → 委托 ObservationProcessor
        input_params = {'path_locations':self.path_locations,'ego_ref_speed':self.ego_ref_speed,'ref_offset':self.carla_cfg['world']['ref_offset']}
        obs = self.observation_processor.get_observation(self._is_eval,input_params)
        obs['ref_path_locations'] = self.ref_path_xy
        self.step_count = 0
        return obs

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            if self.sensor_manager.camera_sensor and self.sensor_manager.camera_sensor.get_data() is not None:
                return self.sensor_manager.camera_sensor.get_data()
            else:
                return np.zeros(
                    (self.env_cfg["image"]["height"], self.env_cfg["image"]["width"], 3),
                    dtype=np.uint8
                )
        return None

    def close(self):
        # 恢复所有卸载的地图层级
        self._load_map_layers()
        # 1. 销毁传感器 → SensorManager
        self.sensor_manager.cleanup()
        # 2. 销毁车辆 → VehicleManager
        self.vehicle_manager.cleanup()
        # 3. 关闭SUMO → SumoIntegration
        if self.enable_sumo:
            self.sumo_integration.cleanup()

        # 恢复异步模式
        if self.carla_cfg["sync_mode"]:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        # 重置世界
        if self.client is not None:
            self.client.reload_world()


    @property
    def is_eval(self):
        return self._is_eval

