import carla
import numpy as np

# ====================== 画线函数（终极修复） ======================
def draw_lines_between_points(world, points, display_time=5.0, color=None, thickness=0.15):
    if color is None:
        color = carla.Color(255, 0, 0)

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
            p = carla.Location(float(p[0]), float(p[1]), float(p[2]) if len(p) >= 3 else 0.0)

        # 绘制点（无连线，纯点）
        # draw_point 参数：位置，大小，颜色，显示时间
        debug.draw_point(
            p,
            size,
            color,
            display_time
        )

# ====================== 显示文字函数（终极修复） ======================
def draw_text_at_location(world, text, location, display_time=5.0, color=None):
    if color is None:
        color = carla.Color(255, 255, 255)

    debug = world.debug

    # 【核心修复】严格判断：只有不是Location时才转换
    if not isinstance(location, carla.Location):
        location = carla.Location(float(location[0]), float(location[1]), float(location[2]) if len(location)>=3 else 0.0)

    # 严格按位置传参（CARLA 0.9.16 强制要求）
    debug.draw_string(
        location,    # 坐标
        text,        # 文字
        False,       # 阴影
        color,       # 颜色
        display_time # 显示时间
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