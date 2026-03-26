import carla
import random
import time
import math


class TrafficGenerator:
    def __init__(self, world, blueprint_library):
        self.world = world
        self.bp_lib = blueprint_library
        self.spawned_vehicles = []

        # 获取常见的车辆蓝图 (这里以轿车为例，你可以扩展)
        self.vehicle_blueprints = [bp for bp in blueprint_library.filter('vehicle.*')]
        # 过滤掉太大的车或特殊的车，只保留普通轿车/卡车等
        self.vehicle_blueprints = [bp for bp in self.vehicle_blueprints
                                   if bp.has_attribute('number_of_wheels') and int(
                bp.get_attribute('number_of_wheels').as_int()) == 4]

        if not self.vehicle_blueprints:
            raise Exception("未找到可用的车辆蓝图，请检查 CARLA 资源库。")

    def get_all_lane_ids_in_section(self, waypoint_list):
        """
        获取给定路点列表中所有唯一的车道 ID。
        waypoint_list: 一个包含起始路点和结束路点的列表，或者是一个路段的采样路点。
        为了简化，我们假设用户传入的是代表起点区域和终点区域的两个路点列表，或者单个路点代表该路段。
        这里我们优化为：传入一个起始路点 (start_wp) 和一个结束路点 (end_wp)，
        函数会尝试获取该路段附近的所有车道。

        更通用的做法：传入 start_waypoint 和 end_waypoint，
        我们分别获取它们所在路段的所有相邻车道。
        """
        # 这个函数逻辑稍作调整：用户通常传入的是“起点路点”和“终点路点”
        # 我们需要找到这两个路点所在“路段”的所有车道
        return self._get_sibling_lanes(waypoint_list)

    def _get_sibling_lanes(self, base_waypoints):
        """
        基于给定的路点，获取同一路段的所有兄弟车道（即所有平行车道）。
        参数 base_waypoints 可以是一个路点列表，我们会合并所有找到的车道。
        """
        lane_ids = set()
        if not isinstance(base_waypoints, list):
            base_waypoints = [base_waypoints]

        for wp in base_waypoints:
            # 获取当前车道
            current_lane_id = wp.lane_id
            road_id = wp.road_id

            # 向左查找
            left_wp = wp
            while left_wp.get_left_lane():
                left_wp = left_wp.get_left_lane()
                if left_wp.road_id == road_id:  # 确保还在同一条路上
                    lane_ids.add((road_id, left_wp.lane_id))

            # 向右查找
            right_wp = wp
            while right_wp.get_right_lane():
                right_wp = right_wp.get_right_lane()
                if right_wp.road_id == road_id:
                    lane_ids.add((road_id, right_wp.lane_id))

            # 添加自己
            lane_ids.add((road_id, current_lane_id))

        return list(lane_ids)

    def generate_traffic_flow(self, start_waypoints, end_waypoints, vehicles_per_hour, duration_seconds=None):
        """
        生成车流。

        参数:
        start_waypoints: [carla.Waypoint] 起点路点列表 (代表起点路段的所有可能位置)
        end_waypoints: [carla.Waypoint] 终点路点列表 (代表终点路段的所有可能位置)
        vehicles_per_hour: int 每小时车流量 (辆/小时)
        duration_seconds: float 运行持续时间 (秒)，如果为 None 则无限运行直到手动停止
        """

        # 1. 预处理：获取所有可用的起点车道和终点车道组合
        # 为了简化，我们从传入的路点中提取所有可能的 (road_id, lane_id)
        start_lanes = self._get_sibling_lanes(start_waypoints)
        end_lanes = self._get_sibling_lanes(end_waypoints)

        if not start_lanes or not end_lanes:
            print("错误：无法提取有效的起点或终点车道。")
            return

        print(f"--- 开始生成车流 ---")
        print(f"起点车道数量: {len(start_lanes)}, 终点车道数量: {len(end_lanes)}")
        print(f"设定流量: {vehicles_per_hour} 辆/小时")

        # 2. 计算生成间隔 (秒)
        # 间隔 = 3600秒 / 车辆数
        spawn_interval = 3600.0 / vehicles_per_hour if vehicles_per_hour > 0 else 0

        if spawn_interval < 0.1:
            print(
                f"警告：流量过大 ({vehicles_per_hour})，生成间隔过小 ({spawn_interval:.2f}s)，可能导致车辆重叠或仿真卡顿。")
            spawn_interval = 0.1  # 设置最小安全间隔

        start_time = time.time()
        last_spawn_time = start_time - spawn_interval  # 初始化以便立即生成第一辆

        try:
            while True:
                current_time = time.time()

                # 检查是否达到持续时间
                if duration_seconds and (current_time - start_time) >= duration_seconds:
                    print("达到指定持续时间，停止生成。")
                    break

                # 检查是否到了生成时间
                if current_time - last_spawn_time >= spawn_interval:

                    # A. 随机选择一个起点车道
                    start_road, start_lane_id = random.choice(start_lanes)
                    # 在选定的车道上找一个具体的生成点 (在原始传入路点附近微调，避免完全重合)
                    # 这里简单起见，直接使用传入的某个起点点的变换，但修正车道ID
                    base_start_wp = random.choice(start_waypoints)
                    # 重新获取该道路和车道上的精确路点
                    actual_start_wp = self.world.get_map().get_waypoint_xodr(start_road, start_lane_id, base_start_wp.s)

                    if not actual_start_wp:
                        # 如果 XODR 查找失败，退回到原始路点 (可能车道号不连续)
                        actual_start_wp = base_start_wp

                        # B. 随机选择一个终点车道
                    end_road, end_lane_id = random.choice(end_lanes)
                    base_end_wp = random.choice(end_waypoints)
                    actual_end_wp = self.world.get_map().get_waypoint_xodr(end_road, end_lane_id, base_end_wp.s)
                    if not actual_end_wp:
                        actual_end_wp = base_end_wp

                    # C. 检查起点是否拥堵 (防重叠)
                    # 检查前方 5 米内是否有车
                    is_blocked = False
                    for other in self.world.get_actors().filter('vehicle.*'):
                        if other.id != 0:  # 忽略自车等
                            dist = actual_start_wp.transform.location.distance(other.get_location())
                            if dist < 8.0:  # 安全距离
                                is_blocked = True
                                break

                    if not is_blocked:
                        self._spawn_vehicle_with_route(actual_start_wp, actual_end_wp)
                        last_spawn_time = current_time
                    else:
                        print("起点拥堵，跳过本次生成。")

                # 让出 CPU，保持仿真流畅
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n用户中断，停止生成。")
        finally:
            print(f"总共生成了 {len(self.spawned_vehicles)} 辆车。")

    def _spawn_vehicle_with_route(self, start_wp, end_wp):
        """生成单辆车并设置导航"""
        # 1. 选择车辆类型
        bp = random.choice(self.vehicle_blueprints)

        # 2. 设置颜色等属性 (可选)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # 3. 生成变换 (稍微偏移一点 s 值，避免正好压在路点上导致物理抖动)
        transform = carla.Transform(start_wp.transform.location + carla.Location(z=1.0), start_wp.transform.rotation)

        # 4. 生成车辆
        vehicle = self.world.try_spawn_actor(bp, transform)

        if vehicle:
            self.spawned_vehicles.append(vehicle)

            # 5. 设置自动驾驶/导航
            # 方法：使用 carla.Agent 或直接设置 Route
            # 这里使用简单的设置目的地方式，CARLA 内置的导航器会自动处理
            # 注意：这需要车辆有自动驾驶能力或者我们手动控制。
            # 最简单的方法是给车安装一个 "WalkerAIController" 类似的控制器，但车辆通常用 "VehicleAIController" (如果存在)
            # 或者，我们直接利用 CARLA 的导航网格 (如果地图支持) 或者手动设置速度方向。

            # 【推荐方案】：使用 CARLA 的 Navigation 模块 (需要地图有导航网格)
            # 如果没有导航网格，我们需要手动计算路径点。
            # 这里演示最通用的方法：设置一个目标点，并使用简单的 PID 或 内置行为树 (如果有)
            # 由于原生 CARLA Python API 没有直接的 "go_to" 函数给普通车辆，
            # 我们通常需要通过设置油门/刹车/转向来控制，或者使用第三方的 autopilot。

            # 开启自动驾驶模式 (Autopilot) 是最简单的，但它不会严格遵循我们指定的终点，只是随机开。
            # 若要严格遵循终点，我们需要计算路径。

            # --- 严格路径规划方案 ---
            try:
                # 计算从 start 到 end 的路径
                path = self._compute_path(start_wp, end_wp)
                if path:
                    # 这里需要一个控制器来跟随路径。
                    # 为了代码简洁且不依赖外部库 (如 agents)，我们开启 Autopilot 并尝试设置目的地
                    # 注意：CARLA 的 Autopilot 在较新版本中支持设置目的地 (Destination)
                    vehicle.set_autopilot(True)

                    # 尝试使用 navigation 模块设置目的地 (CARLA 0.9.10+ 支持较好)
                    # 如果地图没有构建导航网格，这步可能会失败或无效
                    nav = self.world.get_navigation()
                    if nav:
                        route = [end_wp.transform]
                        nav.set_destination(vehicle, route)
                        print(f"车辆 {vehicle.id} 已生成并设置导航至终点。")
                    else:
                        print(f"车辆 {vehicle.id} 已生成 (自动驾驶模式)，但地图可能不支持精确导航到特定终点。")
                else:
                    print("无法计算路径，车辆将以自动驾驶模式随机行驶。")
                    vehicle.set_autopilot(True)

            except Exception as e:
                print(f"导航设置失败: {e}，车辆将开启随机自动驾驶。")
                vehicle.set_autopilot(True)
        else:
            print("生成车辆失败 (可能位置重叠)。")

    def _compute_path(self, start_wp, end_wp):
        """计算两点之间的路径点列表"""
        try:
            # 使用 CARLA 内置的路径规划器 (Global Route Planner 需要导入 agents 模块，这里用简易版)
            # 如果不想引入复杂的 agents 库，可以使用 world.get_map().to_carla_location() 等
            # 这里假设用户环境可能有 agents 库，如果没有，fallback 到直线或空
            from carla import GlobalRoutePlanner

            # 注意：GlobalRoutePlanner 通常需要基于 map 的拓扑图，初始化较慢
            # 为简化代码，这里仅返回一个包含终点的路由，依赖车辆自身的寻路能力 (如果开启了 autopilot 且地图支持)
            # 真正的完整路径规划代码较长，此处简化为返回端点
            return [start_wp, end_wp]
        except ImportError:
            return None

    def destroy_all(self):
        """销毁所有生成的车辆"""
        print("正在销毁所有生成的车辆...")
        for vehicle in self.spawned_vehicles:
            if vehicle:
                vehicle.destroy()
        self.spawned_vehicles.clear()


# ==========================================
# 使用示例 (请在您的主仿真循环中调用)
# ==========================================
"""
假设您已经连接了 client 和 world
client = carla.Client('localhost', 2000)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# 1. 定义起点和终点 (通过坐标或路点)
# 例如：起点在 Road 10, 终点在 Road 20
# 您需要先获取具体的 Waypoint 对象
map = world.get_map()
# 获取起点路点 (示例坐标，请替换为您地图的实际坐标)
start_location = carla.Location(x=100.0, y=0.0, z=1.0) 
start_wp = map.get_waypoint(start_location)

# 获取终点路点
end_location = carla.Location(x=300.0, y=50.0, z=1.0)
end_wp = map.get_waypoint(end_location)

# 如果您想指定“所有车道”，只需传入该路段的一个代表性路点即可，
# 上面的类会自动查找该路段的所有平行车道。
start_waypoints_list = [start_wp]
end_waypoints_list = [end_wp]

# 2. 创建生成器
generator = TrafficGenerator(world, blueprint_library)

# 3. 启动生成
# 参数：起点列表，终点列表，每小时车流量 (例如 600 辆/小时), 持续时间 (秒，可选)
try:
    generator.generate_traffic_flow(start_waypoints_list, end_waypoints_list, vehicles_per_hour=600, duration_seconds=60)
except Exception as e:
    print(f"发生错误: {e}")
finally:
    generator.destroy_all()
"""