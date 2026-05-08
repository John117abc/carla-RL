import carla
import math
import numpy as np

# ====================== 画线函数（终极修复） ======================
def draw_lines_between_points(world, points, display_time=5.0, color=carla.Color(255, 0, 0), thickness=0.15):
    debug = world.debug

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]

        # 【核心修复】严格判断：只有不是Location时才转换，是Location直接用
        # 修复：避免 carla.Location(Location) 报错
        if not isinstance(p1, carla.Location):
            p1 = carla.Location(float(p1[0]), float(p1[1]), float(p1[2]) if len(p1)>=3 else 0.0)
        if not isinstance(p2, carla.Location):
            p2 = carla.Location(float(p2[0]), float(p2[1]), float(p2[2]) if len(p2)>=3 else 0.0)

        # 严格按位置传参（CARLA 0.9.16 强制要求）
        debug.draw_line(
            p1,          # 起点
            p2,          # 终点
            thickness,   # 粗细
            color,       # 颜色
            display_time # 显示时间
        )


# ====================== 只画点，不连线 ======================
def draw_points(world, points, display_time=5.0, color=None, size=0.2):
    """
    只绘制一堆点，不连线
    :param world: CARLA world
    :param points: 点列表 [x,y,z] 或 carla.Location
    :param display_time: 显示时间
    :param color: 点颜色
    :param size: 点大小
    """
    if color is None:
        color = carla.Color(0, 255, 0)  # 点默认绿色

    debug = world.debug

    for p in points:
        # 安全转换坐标
        if not isinstance(p, carla.Location):
            p = carla.Location(float(p[0]), float(p[1]), float(p[2]) if len(p) >= 3 else 5.0)

        # 绘制点（无连线，纯点）
        # draw_point 参数：位置，大小，颜色，显示时间
        debug.draw_point(
            p,
            size,
            color,
            display_time
        )


def draw_text_at_location(world, text, location, display_time=5.0, color=None, size=0.5):
    """
    适配 CARLA 0.9.16 终极修复版
    """
    if color is None:
        color = carla.Color(255, 255, 255)  # 白色

    debug = world.debug

    # 安全处理坐标
    if isinstance(location, (list, tuple, np.ndarray)):
        x = float(location[0])
        y = float(location[1])
        z = float(location[2]) if len(location) >= 3 else 0.0
        loc = carla.Location(x, y, z)
    else:
        loc = location

    # --------------------------
    # 关键修复：严格按 CARLA 官方参数顺序传！
    # --------------------------
    debug.draw_string(
        loc,                # 1. 位置
        text,               # 2. 文字
        False,              # 3. 阴影（必须传）
        color,              # 4. 颜色
        display_time,       # 5. 显示时间
        True                # 6. 持久化（必须传）
    )


def draw_predicted_trajectory(world, points, display_time=5.0, color=None, thickness=0.2):
    """
    绘制预测轨迹连线（封装函数，统一颜色与显示逻辑）
    :param world: CARLA World
    :param points: 世界坐标系下的点列表 [carla.Location]
    :param display_time: 显示持续时间
    :param color: 轨迹颜色
    :param thickness: 线条粗细
    """
    if color is None:
        color = carla.Color(0, 255, 0)  # 默认绿色轨迹
    draw_lines_between_points(world, points, display_time, color, thickness)



def draw_all_vehicles_ellipses(world, ego_vehicle, other_vehicles, a=3.2, b=1.5, life_time=0.1):
    """绘制自车和所有周车的椭圆"""
    # 自车（红色）
    draw_vehicle_ellipse(world, ego_vehicle.get_transform(), a, b,
                         color=carla.Color(255, 0, 0), life_time=life_time)

    # 周车（绿色）
    for v in other_vehicles:
        if v is not None and v.is_alive:
            draw_vehicle_ellipse(world, v.get_transform(), a, b,
                                 color=carla.Color(0, 255, 0), life_time=life_time)

def draw_all_vehicles_double_circles(world, ego_vehicle, other_vehicles, a=2.25, b=1.0, life_time=0.1):
    debug = world.debug
    # 绘制自车
    draw_vehicle_circles(debug, ego_vehicle,
                         color=carla.Color(0, 255, 0),
                         circle_radius=a * 0.65,
                         dist_from_center=b * 1.0,
                         life_time=life_time)
    # 绘制每辆NPC
    for npc in other_vehicles:
        if npc.is_alive:
            draw_vehicle_circles(debug, npc,
                                 color=carla.Color(255, 100, 0),
                                 circle_radius=a * 0.65,
                                 dist_from_center=b * 1.0,
                                 life_time=life_time)


def draw_vehicle_ellipse(
    world: carla.World,
    transform: carla.Transform,
    a: float = 2.25,          # 半长轴 (车长一半)
    b: float = 1.0,           # 半短轴 (车宽一半)
    color: carla.Color = carla.Color(255, 0, 255),
    life_time: float = 0.1,   # 每帧绘制持续时间
    num_points: int = 36      # 分段数，越大越圆滑
):
    """
    在 CARLA 世界中绘制一辆车的椭圆包络。
    :param world: CARLA world 对象
    :param transform: 车辆中心的世界位姿 (位置+航向)
    :param a: 半长轴 (m)
    :param b: 半短轴 (m)
    :param color: 颜色
    :param life_time: 线条持续时间 (秒)，用于循环调用时自动消失
    :param num_points: 椭圆离散点数
    """
    if world is None:
        return

    # 1. 在车辆局部坐标系生成椭圆轮廓点 (顺时针)
    theta = np.linspace(0, 2 * math.pi, num_points, endpoint=False)
    local_x = a * np.cos(theta)
    local_y = b * np.sin(theta)
    local_pts = np.stack([local_x, local_y], axis=1)  # (N, 2)

    # 2. 获取车辆航向的旋转矩阵
    yaw_rad = math.radians(transform.rotation.yaw)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    # 3. 世界坐标系: p_world = R * p_local + translation
    world_pts = []
    for lx, ly in local_pts:
        wx = transform.location.x + cos_y * lx - sin_y * ly
        wy = transform.location.y + sin_y * lx + cos_y * ly
        world_pts.append(carla.Location(x=wx, y=wy, z=transform.location.z + 5.0))  # 略微抬高避免穿地

    # 4. 绘制线段连接相邻点
    debug = world.debug
    for i in range(len(world_pts)):
        p1 = world_pts[i]
        p2 = world_pts[(i + 1) % len(world_pts)]
        debug.draw_line(p1, p2, thickness=0.1, color=color, life_time=life_time)

    # 也可以画一个方向箭头表示车头
    # head_x = transform.location.x + cos_y * a
    # head_y = transform.location.y + sin_y * a
    # debug.draw_point(carla.Location(x=head_x, y=head_y, z=transform.location.z + 0.5),
    #                  size=0.1, color=carla.Color(255, 0, 0), life_time=life_time)

def draw_circle(debug_helper, center, radius, color=carla.Color(255,255,255),
                life_time=0.1, num_segments=16):
    """
    用线段逼近圆，适配 Carla DebugHelper::draw_line 的完整签名。
    """
    pts = []
    for i in range(num_segments + 1):
        angle = 2 * math.pi * i / num_segments
        x = center.x + radius * math.cos(angle)
        y = center.y + radius * math.sin(angle)
        pts.append(carla.Location(x, y, center.z))
    for i in range(num_segments):
        debug_helper.draw_line(
            pts[i], pts[i+1],
            thickness=0.1,
            color=color,
            life_time=life_time,
            persistent_lines=False
        )

def draw_vehicle_circles(debug_helper, vehicle, color=carla.Color(255, 0, 0),
                         circle_radius=1.0, dist_from_center=1.35, life_time=0.1):
    """
    绘制车辆的双圆覆盖模型（前后两个圆 + 圆心连线）。
    """
    transform = vehicle.get_transform()
    loc = transform.location
    yaw_rad = math.radians(transform.rotation.yaw)
    cos = math.cos(yaw_rad)
    sin = math.sin(yaw_rad)

    front_center = carla.Location(
        loc.x + dist_from_center * cos,
        loc.y + dist_from_center * sin,
        loc.z + 0.3
    )
    rear_center = carla.Location(
        loc.x - dist_from_center * cos,
        loc.y - dist_from_center * sin,
        loc.z + 0.3
    )

    # 画前后圆
    draw_circle(debug_helper, front_center, circle_radius, color, life_time)
    draw_circle(debug_helper, rear_center, circle_radius, color, life_time)

    # 画圆心连线（车辆纵轴方向）
    debug_helper.draw_line(
        front_center, rear_center,
        thickness=0.1,
        color=color,
        life_time=life_time,
        persistent_lines=False
    )