import carla
import numpy as np
from typing import Optional
from src.utils import get_logger

logger = get_logger('sensors')

class CameraSensor:
    """
    RGB 相机传感器，返回 (H, W, C) 的 uint8 RGB 图像。
    """

    def __init__(
        self,
        vehicle: carla.Actor,
        world: carla.World,
        width: int = 1920,
        height: int = 1080,
        fov: float = 100.0,
        location: carla.Location = carla.Location(x=1.5, z=1.4),
        rotation: carla.Rotation = carla.Rotation(),
    ):
        self.vehicle = vehicle
        self.world = world
        self.data: Optional[np.ndarray] = None  # 最新图像 (H, W, 3), RGB, uint8

        # 创建蓝图
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        bp.set_attribute("sensor_tick", "0.0")  # 每帧都触发

        # 安装位置
        transform = carla.Transform(location, rotation)

        # 生成传感器并绑定回调
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.sensor.listen(self._callback)

    def _callback(self, image: carla.Image):
        """CARLA 回调函数，将原始数据转为 RGB numpy 数组"""
        # CARLA raw_data 是 BGRA 格式（每个像素 4 字节）
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))  # (H, W, 4)
        bgr = array[:, :, :3]  # 去掉 alpha
        rgb = bgr[:, :, ::-1]  # BGR → RGB
        self.data = rgb.copy()  # 确保主线程安全（避免引用）

    def get_data(self) -> Optional[np.ndarray]:
        """获取最新图像，若无则返回 None"""
        return self.data

    def destroy(self):
        """销毁传感器"""
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
        self.data = None


class CollisionSensor:
    """
    碰撞传感器，记录最近一次碰撞的冲量强度（L2 范数）。
    调用 get_intensity() 后自动清零
    """

    def __init__(self, vehicle: carla.Actor):
        self.vehicle = vehicle
        self.intensity: float = 0.0  # 冲量大小

        world = vehicle.get_world()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._callback)

    def _callback(self, event: carla.CollisionEvent):
        """记录碰撞冲量强度"""
        impulse = event.normal_impulse
        self.intensity = (impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2) ** 0.5

    def get_intensity(self) -> float:
        """获取并重置碰撞强度"""
        val = self.intensity
        self.intensity = 0.0
        return val

    def destroy(self):
        """销毁传感器"""
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


class LaneInvasionSensor:
    """
    车道入侵传感器，记录自上次 get_count() 调用以来的入侵次数。
    """

    def __init__(self, vehicle: carla.Actor):
        self.vehicle = vehicle
        self.count: int = 0

        world = vehicle.get_world()
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._callback)

    def _callback(self, event: carla.LaneInvasionEvent):
        """每次入侵计数 +1"""
        self.count += 1

    def get_count(self) -> int:
        """获取并重置入侵次数"""
        val = self.count
        self.count = 0
        return val

    def destroy(self):
        """销毁传感器"""
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


class ObstacleSensor:
    """
    障碍物传感器（基于 ray-cast），检测车辆前方最近的障碍物。
    调用 get_distance() 可获取最近障碍物距离（米），若无障碍物返回 None。
    """

    def __init__(self, vehicle: carla.Actor, distance: float = 50.0):
        """
        :param vehicle: 绑定的车辆
        :param distance: 最大探测距离（米）
        """
        self.vehicle = vehicle
        self.distance = distance
        self._last_distance: float | None = None
        self._last_actor: carla.Actor | None = None

        world = vehicle.get_world()
        bp = world.get_blueprint_library().find("sensor.other.obstacle")
        bp.set_attribute("distance", str(distance))      # 最大探测距离
        bp.set_attribute("hit_radius", "0.6")            # 射线半径
        bp.set_attribute("only_dynamics", "false")       # false 表示也检测静态物体（如路灯、建筑）
        bp.set_attribute("debug_linetrace", "true")     # 设为 true 可在仿真中看到射线（调试用）

        # 通常安装在车头中心，稍微往前一点
        transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.sensor.listen(self._callback)

    def _callback(self, event: carla.ObstacleDetectionEvent):
        """当检测到障碍物时触发"""
        self._last_distance = event.distance  # 单位：米
        self._last_actor = event.other_actor

    def get_distance(self) -> float | None:
        """
        获取最近障碍物的距离（米）
        :return: 距离（float）或 None（无障碍物）
        """
        return self._last_distance

    def get_obstacle_actor(self) -> carla.Actor | None:
        """获取最近障碍物的 Actor 对象"""
        return self._last_actor

    def is_obstacle_ahead(self, threshold: float = 2.0) -> bool:
        """
        判断前方是否有障碍物在指定距离内
        :param threshold: 距离阈值（米）
        :return: bool
        """
        dist = self.get_distance()
        return dist is not None and dist < threshold

    def destroy(self):
        """销毁传感器"""
        if hasattr(self, 'sensor') and self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


class IMUSensor:
    """
    IMU 传感器（惯性测量单元），提供加速度、角速度和时间戳。
    调用 get_angular_velocity() 可获取当前角速度（含 yaw_rate = gyroscope.z）。
    所有数据均为最新一帧的瞬时值。
    """

    def __init__(self, vehicle: carla.Actor):
        """
        :param vehicle: 绑定的车辆
        """
        self.vehicle = vehicle
        self._last_accelerometer: Optional[carla.Vector3D] = None  # m/s²
        self._last_gyroscope: Optional[carla.Vector3D] = None      # rad/s
        self._last_timestamp: Optional[float] = None               # seconds (simulation time)

        world = vehicle.get_world()
        bp = world.get_blueprint_library().find("sensor.other.imu")

        # 安装在车辆重心附近
        transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.sensor.listen(self._callback)

    def _callback(self, imu_data: carla.IMUMeasurement):
        """当 IMU 数据更新时触发"""
        self._last_accelerometer = imu_data.accelerometer  # Vector3D (x, y, z) in m/s²
        self._last_gyroscope = imu_data.gyroscope          # Vector3D (x, y, z) in rad/s
        self._last_timestamp = imu_data.timestamp          # simulation time in seconds

    def get_acceleration(self) -> Optional[carla.Vector3D]:
        """
        获取最新加速度（单位：m/s²）
        :return: carla.Vector3D 或 None
        """
        return self._last_accelerometer

    def get_angular_velocity(self) -> Optional[carla.Vector3D]:
        """
        获取最新角速度（单位：rad/s）
        其中：
            - x: roll rate（横滚角速度）
            - y: pitch rate（俯仰角速度）
            - z: yaw rate（偏航角速度，即航向角变化率）
        :return: carla.Vector3D 或 None
        """
        return self._last_gyroscope

    def get_yaw_rate(self) -> Optional[float]:
        """
        获取 yaw 角速度（即 yaw_rate，单位：rad/s）
        等价于 get_angular_velocity().z
        :return: float 或 None
        """
        if self._last_gyroscope is not None:
            return self._last_gyroscope.z
        return None

    def get_timestamp(self) -> Optional[float]:
        """
        获取最新 IMU 数据的时间戳（仿真时间，秒）
        :return: float 或 None
        """
        return self._last_timestamp

    def destroy(self):
        """销毁传感器"""
        if hasattr(self, 'sensor') and self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()