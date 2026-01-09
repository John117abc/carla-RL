import carla
import numpy as np
from typing import Optional


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
    调用 get_intensity() 后自动清零，适合 step-wise 奖励计算。
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