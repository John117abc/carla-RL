import math
from src.carla_utils import world_to_vehicle_frame
from src.utils import get_logger

logger = get_logger(name='termination_checker')
class TerminationChecker:
    def __init__(self, vehicle_manager, sensor_manager, config):
        self.vehicle_manager = vehicle_manager
        self.sensor_manager = sensor_manager
        self.config = config

    def check_termination(self,collision,obstacle,step_count):
        ego_vehicle = self.vehicle_manager.ego_vehicle
        # 计算速度信息存到info中
        vx, vy = world_to_vehicle_frame(ego_vehicle.get_velocity(), ego_vehicle.get_transform())
        v = math.sqrt(vx ** 2 + vy ** 2)

        info = {"collision": False,
                "off_route": False,
                "TimeLimit.truncated": False,
                'speed': v}

        # 碰撞actors和障碍物公用一个终止条件
        if collision > self.config["termination"]["collision_threshold"] or obstacle:
            info["collision"] = True

        if step_count >= self.config["termination"]["max_episode_steps"]:
            info["TimeLimit.truncated"] = True

        return info["collision"], info["TimeLimit.truncated"], info