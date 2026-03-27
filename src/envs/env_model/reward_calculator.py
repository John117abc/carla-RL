import numpy as np

class RewardCalculator:
    def __init__(self, vehicle_manager, config):
        self.vehicle_manager = vehicle_manager
        self.config = config

    def compute_reward(self,lane_inv,collision,obstacle):
        ego_vehicle = self.vehicle_manager.ego_vehicle
        w = self.config["reward_weights"]
        r = 0.0

        v = ego_vehicle.get_velocity()
        speed = np.linalg.norm([v.x, v.y, v.z])
        if speed < 2.0:
            speed_reward = -w["low_speed_penalty"]  # e.g., 2.0
        else:
            speed_reward = w["speed"] * min(speed, 10.0)

        r += speed_reward

        centering_reward = w["centering"] * lane_inv * 5.0
        r -= centering_reward

        collision_reward = 0.0
        if collision > self.config["termination"]["collision_threshold"]:
            collision_reward = w["collision"]
            r += w["collision"]

        obstacle_reward = 0.0
        if obstacle:
            obstacle_reward = w["obstacle"]
            r += obstacle_reward

        # todo 缺少高性能样本计算高性能标准：高速、保持在道路上、在正确车道

        # 航向对齐简化奖励
        r += w["angle"] * (1.0 - abs(ego_vehicle.get_transform().rotation.yaw) / 180.0)

        return {
            'total_reward': float(r),
            'speed_reward': speed_reward,
            'centering_reward': centering_reward,
            'collision_reward': collision_reward,
            'obstacle_reward': obstacle_reward,
            'high_speed_reward': 0.0,
            'right_lane_reward': 0.0
        }