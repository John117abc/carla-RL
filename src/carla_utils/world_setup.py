"""
CARLA 世界初始化与场景配置模块。
负责连接客户端、加载地图、设置天气、生成主车（ego vehicle）及可选的背景交通车辆。
"""

import carla
import random
from typing import Optional, Tuple, List
from src.utils import get_logger

# 配置日志
logger = get_logger()
import socket

def find_free_port():
    """找到一个本地可用的空闲端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def connect_to_carla(host: str = 'localhost', port: int = 2000, timeout: float = 10.0) -> carla.Client:
    """
    连接到 CARLA 仿真服务器。

    参数：
        host (str): CARLA 服务器主机地址，默认为 'localhost'。
        port (int): 端口号，默认为 2000。
        timeout (float): 连接超时时间（秒）。

    返回：
        carla.Client: 已连接的 CARLA 客户端对象。
    """
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    logger.info(f"已连接到 CARLA 服务器：{host}:{port}")
    return client


def setup_world(client: carla.Client, map_name: str = 'Town05',
                synchronous_mode: bool = True,
                delta_seconds: float = 0.05) -> carla.World:
    """
    加载指定地图并配置世界参数（如同步模式、固定时间步长）。

    参数：
        client (carla.Client): 已连接的 CARLA 客户端。
        map_name (str): 要加载的地图名称，例如 'Town01', 'Town05'。
        synchronous_mode (bool): 是否启用同步模式（推荐用于 RL 训练）。
        delta_seconds (float): 固定仿真步长时间（秒），仅在同步模式下生效。

    返回：
        carla.World: 配置完成的 CARLA 世界对象。
    """
    world = client.load_world(map_name)
    settings = world.get_settings()

    if synchronous_mode:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta_seconds
        logger.info(f"启用同步模式，固定时间步长：{delta_seconds} 秒")
    else:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None

    world.apply_settings(settings)
    logger.info(f"已加载地图：{map_name}")
    return world


def set_weather(world: carla.World, weather_preset: str = 'ClearNoon') -> None:
    """
    设置 CARLA 世界的天气。

    支持的预设包括：
        'ClearNoon', 'CloudyNoon', 'WetNoon', 'HardRainNoon',
        'ClearSunset', 'CloudySunset', 'WetSunset', 'HardRainSunset'

    参数：
        world (carla.World): 目标世界对象。
        weather_preset (str): 天气预设名称。
    """
    if hasattr(carla.WeatherParameters, weather_preset):
        weather = getattr(carla.WeatherParameters, weather_preset)
        world.set_weather(weather)
        logger.info(f"天气已设置为：{weather_preset}")
    else:
        available = [attr for attr in dir(carla.WeatherParameters) if not attr.startswith('_')]
        logger.warning(f"未知天气预设 '{weather_preset}'。可用选项：{available}")


def spawn_ego_vehicle(world: carla.World,
                      blueprint_filter: str = 'vehicle.tesla.model3',
                      spawn_point_index: Optional[int] = None) -> carla.Vehicle:
    """
    在世界中生成主控车辆（ego vehicle）。

    参数：
        world (carla.World): 目标世界。
        blueprint_filter (str): 车辆蓝图过滤器，例如 'vehicle.*', 'vehicle.tesla.model3'。
        spawn_point_index (Optional[int]): 指定出生点索引。若为 None，则随机选择。

    返回：
        carla.Vehicle: 生成的主车对象。
    """
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = random.choice(blueprint_library.filter(blueprint_filter))
    vehicle_bp.set_attribute('role_name', 'ego')

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("当前地图没有可用的出生点！")

    if spawn_point_index is not None:
        if spawn_point_index >= len(spawn_points):
            raise ValueError(f"出生点索引 {spawn_point_index} 超出范围（共 {len(spawn_points)} 个）")
        spawn_point = spawn_points[spawn_point_index]
    else:
        spawn_point = random.choice(spawn_points)

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        raise RuntimeError("无法生成主车！可能出生点被占用。")

    logger.info(f"主车已生成，位置：{spawn_point.location}")
    return vehicle


def spawn_background_traffic(client: carla.Client,
                             world: carla.World,
                             num_vehicles: int = 30,
                             safe_spawn: bool = True,
                             tm_port: int = 8005) -> List[carla.Actor]:
    """
    生成背景交通车辆，并指定 TrafficManager 端口以避免绑定冲突。

    参数新增：
        tm_port (int): TrafficManager 使用的端口号，默认 8005（避开默认 8000）
    """
    # 获取或创建指定端口的 TrafficManager
    traffic_manager = client.get_trafficmanager(tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    traffic_manager.set_synchronous_mode(True)

    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprints = [bp for bp in blueprints if int(bp.get_attribute('number_of_wheels')) == 4]

    spawn_points = world.get_map().get_spawn_points()
    if len(spawn_points) < num_vehicles:
        logger.warning(
            f"出生点数量 ({len(spawn_points)}) 少于请求车辆数 ({num_vehicles})，将生成 {len(spawn_points)} 辆。")
        num_vehicles = len(spawn_points)

    if safe_spawn:
        batch = []
        for i in range(num_vehicles):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'background')
            spawn_point = spawn_points[i]  # 按顺序取，避免重复
            batch.append(
                carla.command.SpawnActor(blueprint, spawn_point)
                .then(carla.command.SetAutopilot(True, tm_port))  # 关键：传入 tm_port
            )

        responses = client.apply_batch_sync(batch, True)  # ← 这里不会再报 bind error
        actors = []
        for response in responses:
            if response.error:
                logger.debug(f"背景车辆生成失败：{response.error}")
            else:
                actors.append(world.get_actor(response.actor_id))
        logger.info(f"成功生成 {len(actors)} 辆背景交通车辆（TM 端口: {tm_port}）。")
        return actors

    else:
        # 非安全模式（不推荐用于训练）
        actors = []
        for _ in range(num_vehicles):
            blueprint = random.choice(blueprints)
            spawn_point = random.choice(spawn_points)
            vehicle = world.try_spawn_actor(blueprint, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True, tm_port)  # 👈 同样传入 tm_port
                actors.append(vehicle)
        logger.info(f"（非安全模式）生成 {len(actors)} 辆背景车辆（TM 端口: {tm_port}）。")
        return actors


def cleanup_world(world: carla.World) -> None:
    """
    清理世界中的所有动态对象（车辆、行人等），防止内存泄漏或状态残留。

    参数：
        world (carla.World): 要清理的世界。
    """
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    sensors = actors.filter('sensor.*')

    # 销毁传感器
    for sensor in sensors:
        sensor.destroy()

    # 销毁车辆
    for vehicle in vehicles:
        vehicle.destroy()

    logger.info("已清理世界中的所有车辆和传感器。")


# 示例用法（可删除）
if __name__ == "__main__":
    client = connect_to_carla()
    world = setup_world(client, map_name='Town07_Opt', synchronous_mode=True)
    set_weather(world, 'ClearNoon')
    ego_vehicle = spawn_ego_vehicle(world)
    background_vehicles = spawn_background_traffic(client, world, num_vehicles=20)

    # 模拟几帧
    for _ in range(10):
        world.tick()

    cleanup_world(world)