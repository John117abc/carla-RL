import numpy as np
import math
import carla
from src.carla_utils import get_compass, world_to_vehicle_frame, get_ocp_observation,resample_path_equal_distance

class ObservationProcessor:
    def __init__(self, vehicle_manager, sensor_manager,world, config, normalizer):
        self.vehicle_manager = vehicle_manager
        self.sensor_manager = sensor_manager
        self.world = world
        self.config = config
        self.normalizer = normalizer

    def get_observation(self,is_eval,input_params):
        obs = {}
        if "image" in self.config["obs_type"]:
            img = self.sensor_manager.camera_sensor.get_data()
            if img is None:
                img = np.zeros(
                    (self.config["image"]["height"], self.config["image"]["width"], 3),
                    dtype=np.uint8
                )
            obs["image"] = img
        if "measurements" in self.config["obs_type"]:
            measurements = []
            v = self.vehicle_manager.ego_vehicle.get_velocity()
            c = self.vehicle_manager.ego_vehicle.get_control()
            # 转换为车辆坐标系
            vx, vy = world_to_vehicle_frame(v, self.vehicle_manager.ego_vehicle.get_transform())
            compass = get_compass(self.vehicle_manager.ego_vehicle)  # 弧度
            loc = self.vehicle_manager.ego_vehicle.get_location()
            if self.sensor_manager.imu_sensor.get_acceleration() is None:
                acc = 0.0
            else:
                acc = math.sqrt(
                    self.sensor_manager.imu_sensor.get_acceleration().x ** 2 + self.sensor_manager.imu_sensor.get_acceleration().y ** 2 + self.sensor_manager.imu_sensor.get_acceleration().z ** 2)

            if self.sensor_manager.imu_sensor.get_angular_velocity() is None:
                ang_v = 0.0
            else:
                ang_v = math.sqrt(
                    self.sensor_manager.imu_sensor.get_angular_velocity().x ** 2 + self.sensor_manager.imu_sensor.get_angular_velocity().y ** 2 + self.sensor_manager.imu_sensor.get_angular_velocity().z ** 2)

            for key in self.config["measurements"]["include"]:
                if key == "speed":
                    measurements.extend([vx, vy])
                elif key == "steer":
                    measurements.append(float(c.steer))
                elif key == "compass":
                    measurements.append(float(compass))
                elif key == "gps":
                    measurements.extend([float(loc.x), float(loc.y)])
                elif key == "imu":
                    measurements.extend([acc, ang_v, self.sensor_manager.imu_sensor.get_yaw_rate()])
                else:
                    measurements.append(0.0)

            # 归一化 measurements
            meas_array = np.array(measurements, dtype=np.float32)
            if not is_eval:
                self.normalizer.update(meas_array[None, :])
            meas_normalized = self.normalizer.normalize(meas_array)

            obs["measurements"] = meas_normalized
        if "ocp_obs" in self.config["obs_type"]:
            # 获取ocp观察信息
            v = self.vehicle_manager.ego_vehicle.get_velocity()
            speed = np.linalg.norm([v.x, v.y])
            # 速度越快，间距越大 (0.8 秒 * 速度)
            spacing = np.clip(speed * 0.8, 1.0, 5.0)
            if input_params.get('path_locations') is not None:
                raw_path = np.array([[p.x, p.y] for p in input_params['path_locations']])
                # 暂时无法直接用上述函数，因为世界坐标下距离是真实的，可以直接重采样。
                resampled = resample_path_equal_distance(raw_path, spacing)
                # 用新的 carla.Location 替换
                input_params['path_locations'] = [carla.Location(x=pt[0], y=pt[1], z=0) for pt in resampled]
            # 观察周车
            self.vehicle_manager.get_surrounding_vehicles()
            network_state, s_road, s_ref_raw, s_ref_error, s_others = get_ocp_observation(
                self.vehicle_manager.ego_vehicle,
                self.sensor_manager.imu_sensor,
                self.vehicle_manager.npc_vehicles,
                input_params['path_locations'],
                input_params['ego_ref_speed'],
                input_params['ref_offset'],
                input_params['perceived_distance']
            )

            obs["ocp_obs"] = network_state
            obs["s_road_raw"] = s_road
            obs["s_ref_raw"] = s_ref_raw
            obs["s_ref_error"] = s_ref_error
            obs["s_others"] = s_others
            obs["ref_path_locations"] = np.array([[p.x, p.y] for p in input_params['path_locations']], dtype=np.float32)
        if len(obs) == 1:
            return list(obs.values())[0]
        return obs
