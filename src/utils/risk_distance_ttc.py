import numpy as np

from .geometry import velocity_to_global

class RiskDistanceTTC:
    """
    输入自车和周车状态，返回：
        1. 有碰撞风险车辆中的最小距离
        2. 有碰撞风险车辆中的最小 TTC

    状态格式：
        state = np.array([[x, y, v_lat, v_lon, yaw, w]])

    数据格式：
        x = vehicle_transform_carla.location.x
        y = vehicle_transform_carla.location.y
        v_lat = 0.0
        v_lon = traci.vehicle.getSpeed(vehicle_id_sumo)
        yaw = vehicle_transform_carla.rotation.yaw * np.pi / 180.0
        w = 0.0

    说明：
        2. 正前方车辆无论相对速度方向如何，都认为有碰撞风险；
        3. 旁边车道车辆只有相对速度方向进入扩张碰撞盒，才认为有风险；
        4. 没有风险时返回 max_dist, max_ttc。
    """

    def __init__(
        self,
        vehicle_length: float =5.0,
        vehicle_width:float=2.2,
        max_dist:float=18.0,
        max_ttc:float=10.0,
        eps:float=1e-8,
        front_lateral_threshold=None,
    ):
        """
        vehicle_length, vehicle_width: 定义了车辆碰撞盒（bounding box）的尺寸。代码将所有车辆都简化为一个固定大小的矩形。
        max_dist, max_ttc: 预设的最大距离和最大 TTC 值。当环境中没有检测到任何碰撞风险时，主函数会返回这两个值，作为“安全”的信号。
        eps: 一个极小的数值（如 1e-8），用于浮点数比较，避免因计算精度问题导致的除零错误或逻辑错误。
        front_lateral_threshold: 判断一辆车是否属于“正前方”的横向距离阈值。默认值是 vehicle_width / 2，即自车半宽。
            这意味着只要目标车辆的中心在自车中心线左右各一个半宽的范围内，就被视为正前方车辆。
        """
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width

        self.half_l = vehicle_length / 2.0
        self.half_w = vehicle_width / 2.0

        self.max_dist = max_dist
        self.max_ttc = max_ttc
        self.eps = eps

        # 判断“正前方”的横向阈值
        # 默认取两车半宽之和
        if front_lateral_threshold is None:
            self.front_lateral_threshold = vehicle_width/2
        else:
            self.front_lateral_threshold = front_lateral_threshold

    def compute_longitudinal_ttc(
        self,
        longitudinal_dist:float,  # 自车前端到目标车后端的纵向距离
        ego_v:np.ndarray,
        npc_v:np.ndarray,
        ego_heading:np.ndarray,        # 自车航向的单位矢量（长度是1）
    ):
        """
        对正前方车辆，计算沿自车航向方向的 TTC。

        如果自车相对前车不闭合，则 TTC = max_ttc。
        """

        rel_v = ego_v - npc_v       # 相对速度矢量
        closing_speed = float(np.dot(rel_v, ego_heading))       # 相对速度在自车航向上的投影

        if closing_speed > self.eps:
            ttc = longitudinal_dist / closing_speed
        else:
            ttc = self.max_ttc

        return min(ttc, self.max_ttc)

    def ray_intersect_expanded_box_ttc(
            self,
            ego_x:float,
            ego_y:float,
            npc_x:float,
            npc_y:float,
            rel_vx:float,
            rel_vy:float,
    ):
        """
        判断自车中心沿相对速度方向是否会进入扩张后的周车盒子。
        """

        expanded_left = npc_x - 2.0 * self.half_l
        expanded_right = npc_x + 2.0 * self.half_l
        expanded_bottom = npc_y - 2.0 * self.half_w
        expanded_top = npc_y + 2.0 * self.half_w

        # 当前已经接触或重叠
        if (
                expanded_left <= ego_x <= expanded_right
                and expanded_bottom <= ego_y <= expanded_top
        ):
            return 0.0, True

        # 相对速度接近 0
        if abs(rel_vx) < self.eps and abs(rel_vy) < self.eps:
            return self.max_ttc, False

        # x 方向进入/离开时间
        if abs(rel_vx) < self.eps:
            if ego_x < expanded_left or ego_x > expanded_right:
                return self.max_ttc, False
            tx_min = -float("inf")
            tx_max = float("inf")
        else:
            tx1 = (expanded_left - ego_x) / rel_vx
            tx2 = (expanded_right - ego_x) / rel_vx
            tx_min = min(tx1, tx2)
            tx_max = max(tx1, tx2)

        # y 方向进入/离开时间
        if abs(rel_vy) < self.eps:
            if ego_y < expanded_bottom or ego_y > expanded_top:
                return self.max_ttc, False
            ty_min = -float("inf")
            ty_max = float("inf")
        else:
            ty1 = (expanded_bottom - ego_y) / rel_vy
            ty2 = (expanded_top - ego_y) / rel_vy
            ty_min = min(ty1, ty2)
            ty_max = max(ty1, ty2)

        t_enter = max(tx_min, ty_min)
        t_exit = min(tx_max, ty_max)

        if t_enter > t_exit:
            return self.max_ttc, False

        if t_exit < 0.0:
            return self.max_ttc, False

        ttc = max(t_enter, 0.0)

        return min(ttc, self.max_ttc), True

    def get_min_risk_distance_ttc(
            self,
            ego_state:np.ndarray,
            npc_states:np.ndarray
    ):
        """
        主函数。

        return:
            min_risk_dist : 有碰撞风险车辆中的最小距离
            min_risk_ttc  : 有碰撞风险车辆中的最小 TTC

        如果没有碰撞风险：
            return max_dist, max_ttc
        """

        if ego_state is None or len(ego_state) == 0:
            return self.max_dist, self.max_ttc

        if npc_states is None or len(npc_states) < 3:
            return self.max_dist, self.max_ttc

        ego_x = float(ego_state[0, 0])
        ego_y = float(ego_state[0, 1])
        ego_v_lon = float(ego_state[0, 3])
        ego_yaw = float(ego_state[0, 4])

        ego_v = velocity_to_global(ego_v_lon, ego_yaw)

        ego_heading = np.array([
            np.cos(ego_yaw),
            np.sin(ego_yaw),
        ])

        # ego 左侧法向方向，用于计算横向偏差
        ego_lateral = np.array([
            -np.sin(ego_yaw),
            np.cos(ego_yaw),
        ])

        front_states = npc_states[:3]

        min_risk_dist = float("inf")
        min_risk_ttc = float("inf")
        has_risk = False

        for npc in front_states:
            if npc is None or len(npc) == 0:
                continue

            npc_x = float(npc[0, 0])
            npc_y = float(npc[0, 1])

            # 全 0 表示未检测到车辆
            if npc_x == 0.0 and npc_y == 0.0:
                continue

            rel_pos = np.array([
                npc_x - ego_x,
                npc_y - ego_y,
            ])

            # 纵向投影：是否在自车前方
            longitudinal_proj = float(np.dot(rel_pos, ego_heading))

            if longitudinal_proj < 0.0:
                continue

            # 横向投影：是否在正前方区域
            lateral_proj = float(np.dot(rel_pos, ego_lateral))

            is_direct_front = abs(lateral_proj) <= self.front_lateral_threshold

            npc_v_lon = float(npc[0, 3])
            npc_yaw = float(npc[0, 4])

            npc_v = velocity_to_global(npc_v_lon, npc_yaw)

            box_dist = self.compute_box_distance(
                ego_x,
                ego_y,
                npc_x,
                npc_y,
            )

            # 情况 1：正前方车辆，无论相对速度方向，都认为有碰撞风险
            if is_direct_front:
                has_risk = True

                # 正前方距离用于 TTC，避免横向距离干扰
                longitudinal_dist = longitudinal_proj - self.vehicle_length
                longitudinal_dist = max(0.0, longitudinal_dist)

                ttc = self.compute_longitudinal_ttc(
                    longitudinal_dist=longitudinal_dist,
                    ego_v=ego_v,
                    npc_v=npc_v,
                    ego_heading=ego_heading,
                )

                min_risk_dist = min(min_risk_dist, box_dist)
                min_risk_ttc = min(min_risk_ttc, ttc)

                continue

            # 情况 2：非正前方车辆，用相对速度方向判断是否会碰撞
            rel_v = ego_v - npc_v
            rel_vx = rel_v[0]
            rel_vy = rel_v[1]

            ttc, will_collide = self.ray_intersect_expanded_box_ttc(
                ego_x=ego_x,
                ego_y=ego_y,
                npc_x=npc_x,
                npc_y=npc_y,
                rel_vx=rel_vx,
                rel_vy=rel_vy,
            )

            if not will_collide:
                continue

            has_risk = True

            min_risk_dist = min(min_risk_dist, box_dist)
            min_risk_ttc = min(min_risk_ttc, ttc)

        if not has_risk:
            return self.max_dist, self.max_ttc

        min_risk_dist = min(min_risk_dist, self.max_dist)
        min_risk_ttc = min(min_risk_ttc, self.max_ttc)

        return min_risk_dist, min_risk_ttc