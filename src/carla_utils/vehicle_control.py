"""
CARLA 强化学习环境中的车辆控制工具模块。
将强化学习智能体输出的归一化动作转换为 CARLA 的 VehicleControl 控制指令。
"""

import carla
import math
import numpy as np


def action_to_control(action: np.ndarray, vehicle: carla.Vehicle = None) -> carla.VehicleControl:
    """
    将智能体输出的归一化动作转换为 CARLA 车辆控制指令。

    预期的动作空间（连续型）：
        action[0]: 油门/刹车信号，范围 [-1, 1]
            - 大于 0：表示油门（0 到 1）
            - 小于 0：表示刹车（0 到 1，取负值的绝对值）
        action[1]: 转向信号，范围 [-1, 1]（-1 为左转，1 为右转）

    可选扩展（本实现未使用）：
        - 手刹（hand_brake）、倒车（reverse）、手动换挡等可根据需要添加。

    参数：
        action (np.ndarray): 形状为 (2,) 的数组，数值应在 [-1, 1] 范围内。
        vehicle (carla.Vehicle, 可选): 用于高级控制逻辑（例如基于速度的策略）。

    返回：
        carla.VehicleControl: 应用于车辆的控制指令。
    """
    if not isinstance(action, np.ndarray):
        action = np.array(action)

    # 为安全起见，将动作限制在 [-1, 1] 范围内
    action = np.clip(action, -1.0, 1.0)

    # 解析油门和刹车
    throttle_brake = action[0]
    if throttle_brake >= 0:
        throttle = float(throttle_brake)
        brake = 0.0
    else:
        throttle = 0.0
        brake = float(-throttle_brake)

    # 解析转向
    steer = float(action[1])

    # 构造控制指令
    control = carla.VehicleControl()
    control.throttle = throttle
    control.brake = brake
    control.steer = steer
    control.hand_brake = False
    control.reverse = False

    return control


def get_vehicle_speed(vehicle: carla.Vehicle) -> float:
    """
    计算车辆的当前速度（单位：米/秒）。

    参数：
        vehicle (carla.Vehicle): CARLA 中的车辆角色对象。

    返回：
        float: 车辆当前速度（m/s）。
    """
    velocity = vehicle.get_velocity()
    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed


def smooth_control(prev_control: carla.VehicleControl, new_action: np.ndarray,
                   steer_smooth: float = 0.8, throttle_smooth: float = 0.8) -> carla.VehicleControl:
    """
    对控制指令进行指数平滑处理，减少车辆行为的突兀感。
    在评估阶段或规则回退策略中非常有用。

    参数：
        prev_control (carla.VehicleControl): 上一时刻的控制指令。
        new_action (np.ndarray): 智能体新输出的原始动作。
        steer_smooth (float): 转向平滑系数 [0,1]；值越大越平滑。
        throttle_smooth (float): 油门/刹车平滑系数。

    返回：
        carla.VehicleControl: 平滑后的控制指令。
    """
    new_control = action_to_control(new_action)

    # 平滑转向
    smoothed_steer = steer_smooth * prev_control.steer + (1 - steer_smooth) * new_control.steer

    # 分别平滑油门和刹车
    smoothed_throttle = throttle_smooth * prev_control.throttle + (1 - throttle_smooth) * new_control.throttle
    smoothed_brake = throttle_smooth * prev_control.brake + (1 - throttle_smooth) * new_control.brake

    control = carla.VehicleControl()
    control.throttle = float(np.clip(smoothed_throttle, 0.0, 1.0))
    control.brake = float(np.clip(smoothed_brake, 0.0, 1.0))
    control.steer = float(np.clip(smoothed_steer, -1.0, 1.0))
    control.hand_brake = False
    control.reverse = False

    return control


# 可选组件：基于 PID 的纵向控制器（用于参考或混合控制策略）
class PIDLongitudinalController:
    """简单的 PID 控制器，用于跟踪目标速度。"""
    def __init__(self, K_P=1.0, K_I=0.05, K_D=0.1, dt=0.05):
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_sum = 0.0      # 积分项累计误差
        self._last_error = 0.0     # 上一次误差

    def run_step(self, target_speed: float, current_speed: float) -> float:
        """
        根据目标速度和当前速度，计算油门/刹车指令。
        返回值范围为 [-1, 1]：正值表示油门，负值表示刹车。
        """
        error = target_speed - current_speed
        self._error_sum += error * self._dt
        derivative = (error - self._last_error) / self._dt if self._dt > 0 else 0.0

        output = self._k_p * error + self._k_i * self._error_sum + self._k_d * derivative
        self._last_error = error

        # 限制输出在 [-1, 1] 范围内
        return np.clip(output, -1.0, 1.0)

def get_compass(vehicle: carla.Vehicle) -> float:
    """
    获取车辆的罗盘方向（弧度），0 表示正北，顺时针增加。
    返回值范围: [0, 2π)
    """
    yaw_deg = vehicle.get_transform().rotation.yaw
    # 转换为弧度，并归一化到 [0, 2π)
    compass_rad = math.radians(yaw_deg) % (2 * math.pi)
    return compass_rad