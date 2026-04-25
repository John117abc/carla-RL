import numpy as np
import carla
import traceback
from src.carla_utils import draw_points,draw_text_at_location, ego_to_world_coordinate, draw_predicted_trajectory
from src.utils import get_logger, unpack_ocp_numpy

logger = get_logger(name='debug_visualizer')
class DebugVisualizer:
    def __init__(self, world, vehicle_manager, config):
        self.world = world
        self.vehicle_manager = vehicle_manager
        self.config = config

    def debug_ocp(self,ocp_obs, s_ref_raw, s_road,step_count,fixed_delta_seconds,predict_traj):
        ocp_obs_np = np.array(ocp_obs, dtype=np.float32).flatten().reshape([1, 1, -1])
        ego_state, other_states, ref_error = unpack_ocp_numpy(ocp_obs_np, self.config['ocp']['num_points'],
                                                              self.config['ocp']['others'])

        # 显示当前步数和最大步数
        step_num_text = f'now step:{step_count},\n max limit step:{self.config["termination"]["max_episode_steps"]}'
        # 显示ego文字
        world_ego_x, world_ego_y = ego_to_world_coordinate(ego_state[0][0][0], ego_state[0][0][1],
                                                           self.vehicle_manager.ego_vehicle.get_transform())

        # CARLA+SUMO固定步长0.05s，显示时间统一适配，杜绝闪烁
        SYNC_STEP = fixed_delta_seconds/5  # 必须和fixed_delta_seconds一致

        # 1. 步数文字
        draw_text_at_location(
            world=self.world,
            text=step_num_text,
            location=np.array([world_ego_x, world_ego_y + 20], dtype=np.float32),
            display_time=SYNC_STEP,  # 0.3s，覆盖6帧，无断层
            color=carla.Color(0, 0, 255)
        )

        # 2. 自车状态文字
        ego_text = f'v_lon:{ego_state[0][0][2]:.2f} \n v_lat:{ego_state[0][0][3]:.2f} \n φ:{ego_state[0][0][4]:.2f} \n 0:{ego_state[0][0][5]:.2f}'
        draw_text_at_location(
            world=self.world,
            text=ego_text,
            location=np.array([world_ego_x, world_ego_y], dtype=np.float32),
            display_time=SYNC_STEP,
            color=carla.Color(0, 0, 255)
        )

        road_left = s_road[..., :self.config['ocp']['num_points'] * 2].reshape(self.config['ocp']['num_points'], 2)
        road_right = s_road[..., self.config['ocp']['num_points'] * 2:].reshape(self.config['ocp']['num_points'], 2)

        # 左车道
        draw_points(
            world=self.world,
            points=road_left,
            display_time=SYNC_STEP * 20,
            color=carla.Color(0, 255, 0),
            size=0.05
        )

        # 右车道
        draw_points(
            world=self.world,
            points=road_right,
            display_time=SYNC_STEP * 20,
            color=carla.Color(0, 255, 0),
            size=0.05
        )

        # 参考路径绘制
        draw_points(
            world=self.world,
            points=s_ref_raw[0:2].reshape(1, -1),
            display_time=SYNC_STEP * 20,
            color=carla.Color(0, 0, 255),
            size=0.05
        )

        # 绘制误差文字
        ref_error_text = f'e_p:{ref_error[0][0][0]:.2f} \n e_φ:{ref_error[0][0][1]:.2f} \n e_v:{ref_error[0][0][2]:.2f}\n'
        draw_text_at_location(
            world=self.world,
            text=ref_error_text,
            location=np.array([world_ego_x, world_ego_y + 10.0], dtype=np.float32),
            display_time=SYNC_STEP,
            color=carla.Color(0, 0, 255)
        )

        # 绘制预测点
        # 【新增】可视化 OCP 预测轨迹
        if predict_traj is not None:
            try:
                traj_xy = predict_traj[0, :, :2]  # [horizon, 2] 自车坐标系 xy

                # 转换至世界坐标系
                ego_transform = self.vehicle_manager.ego_vehicle.get_transform()
                world_points = []
                for x, y in traj_xy:
                    wx, wy = ego_to_world_coordinate(float(x), float(y), ego_transform)
                    world_points.append(carla.Location(wx, wy, ego_transform.location.z))

                if len(world_points) > 1:
                    draw_predicted_trajectory(self.world,world_points,0.1,carla.Color(255, 255, 0),0.1)
            except Exception as e:
                logger.error(f"OCP 轨迹可视化失败: {e}")
                traceback.print_exc()





    def update_spectator(self,ref_path,last_ref_idx,prev_spectator_transform):
        """
        车辆Z轴震动导致的抖动 + 上下坡显示异常
        兼容纯2D参考线 [x,y]
        """
        ego_vehicle = self.vehicle_manager.ego_vehicle
        if ego_vehicle is None or len(ref_path) < 2:
            return

        # 1. 获取车辆当前位置
        vehicle_loc = ego_vehicle.get_location()
        vehicle_xy = np.array([vehicle_loc.x, vehicle_loc.y])

        # 2. 找到车辆在密集参考线上的「最近投影点」（用上一帧索引加速）
        search_window = 100
        start_idx = max(0, last_ref_idx - search_window)
        end_idx = min(len(ref_path), last_ref_idx + search_window)
        search_path = ref_path[start_idx:end_idx]

        # 计算距离，找到最近点
        dists = np.hypot(search_path[:, 0] - vehicle_xy[0], search_path[:, 1] - vehicle_xy[1])
        local_min_idx = np.argmin(dists)
        closest_idx = start_idx + local_min_idx
        last_ref_idx = closest_idx  # 更新索引供下一帧使用

        # 3. 计算参考线在该点的「切线方向」（保证视角平行于道路）
        lookahead_idx = min(closest_idx + 5, len(ref_path) - 1)
        lookbehind_idx = max(closest_idx - 1, 0)

        forward_pt = ref_path[lookahead_idx]
        backward_pt = ref_path[lookbehind_idx]

        # 计算切线角度（yaw）
        dx = forward_pt[0] - backward_pt[0]
        dy = forward_pt[1] - backward_pt[1]
        ref_yaw = np.degrees(np.arctan2(dy, dx))

        # 4. 设置相机位置：参考线投影点正上方
        x_offset = 10.0
        z_height = 50.0

        # 把x_offset投影到参考线方向上
        offset_x = x_offset * np.cos(np.radians(ref_yaw))
        offset_y = x_offset * np.sin(np.radians(ref_yaw))

        # ===================== 唯一修改：平滑车辆Z轴，解决抖动+上下坡 =====================
        # 初始化平滑Z值
        if not hasattr(self, 'smoothed_z'):
            self.smoothed_z = vehicle_loc.z
        # 一阶低通滤波：消除车辆Z轴抖动，保留上下坡大趋势
        alpha = 0.1  # 平滑系数，越小越稳
        self.smoothed_z = self.smoothed_z * (1 - alpha) + vehicle_loc.z * alpha

        target_location = carla.Location(
            x=float(ref_path[closest_idx, 0] + offset_x),
            y=float(ref_path[closest_idx, 1] + offset_y),
            z=float(self.smoothed_z + z_height)  # 用平滑后的Z，不抖+上下坡正常
        )

        # 5. 设置相机朝向（原版完全不变）
        target_rotation = carla.Rotation(
            pitch=-90.0,
            yaw=float(ref_yaw),
            roll=0.0
        )
        target_transform = carla.Transform(target_location, target_rotation)

        # 6. 极轻微平滑
        if prev_spectator_transform is None:
            final_transform = target_transform
        else:
            lerp_factor = 0.5
            prev_loc = prev_spectator_transform.location
            final_loc = carla.Location(
                x=prev_loc.x + (target_location.x - prev_loc.x) * lerp_factor,
                y=prev_loc.y + (target_location.y - prev_loc.y) * lerp_factor,
                z=prev_loc.z + (target_location.z - prev_loc.z) * lerp_factor
            )

            prev_rot = prev_spectator_transform.rotation
            delta_yaw = target_rotation.yaw - prev_rot.yaw
            if delta_yaw > 180: delta_yaw -= 360
            if delta_yaw < -180: delta_yaw += 360
            final_yaw = prev_rot.yaw + delta_yaw * lerp_factor

            final_rot = carla.Rotation(
                pitch=prev_rot.pitch + (target_rotation.pitch - prev_rot.pitch) * lerp_factor,
                yaw=final_yaw,
                roll=0.0
            )
            final_transform = carla.Transform(final_loc, final_rot)

        # 7. 应用视角
        self.world.get_spectator().set_transform(final_transform)
        prev_spectator_transform = final_transform

        return last_ref_idx, prev_spectator_transform