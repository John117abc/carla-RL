from src.utils import get_logger
from traci import TraCIException
from sumo_sync.sumo_integration.sumo_simulation import SumoSimulation
from sumo_sync.sumo_integration.carla_simulation import CarlaSimulation
from sumo_sync.sumo_integration.bridge_helper import BridgeHelper
from sumo_sync.run_synchronization import SimulationSynchronization


logger = get_logger(name='sumo_integration')

class SumoIntegration:
    def __init__(self, carla_client, carla_world,vehicle_manager ,sumo_config):
        self.carla_client = carla_client
        self.carla_world = carla_world
        self.vehicle_manager = vehicle_manager
        self.sumo_config = sumo_config
        self.carla_simulation = None
        self.sumo_simulation = None
        self.synchronization = None
        self.synchronization = None


    def init_sumo(self,fixed_delta_seconds,carla_host,carla_port):
        simulation_step_length = fixed_delta_seconds # 确保与 CARLA 同步
        cfg_file = self.sumo_config['default']['sumo_config_file']
        sumo_gui = self.sumo_config['default']['sumo_gui']
        sumo_host = self.sumo_config['default']['sumo_host']
        sumo_port = self.sumo_config['default']['sumo_port']

        self.carla_simulation = CarlaSimulation(host=carla_host,
                                                port=carla_port,
                                                step_length=simulation_step_length)

        self.sumo_simulation = SumoSimulation(cfg_file,
                                              simulation_step_length,
                                              host=sumo_host,
                                              port=sumo_port,
                                              sumo_gui=sumo_gui,
                                              client_order=1)

        self.synchronization = SimulationSynchronization(self.sumo_simulation,
                                                         self.carla_simulation,
                                                         'none',  # 交通信号灯管理('carla', 'sumo', 'none')
                                                         False,  # 是否同步车颜色
                                                         False)

        # 重写桥接的车辆生成回调
        def _on_sumo_vehicle_spawn(carla_vehicle):
            # 空值校验，防止车辆未生成完成就调用属性
            if not carla_vehicle:
                return
            if carla_vehicle.attributes.get('role_name') != 'hero':
                carla_vehicle.set_simulate_physics(False)
                logger.debug(f"关闭SUMO周车物理：{carla_vehicle.id}")

        # 绑定回调
        self.carla_simulation.on_vehicle_spawned = _on_sumo_vehicle_spawn

    def sync_step(self):
        self.sumo_simulation.tick()  # 1. SUMO执行一步

        # 手动把自车从同步列表中移除
        if hasattr(self.synchronization, 'carla2sumo_ids'):
            ego_id = self.vehicle_manager.ego_vehicle.id  # 自车ID：carla0
            if ego_id in self.synchronization.carla2sumo_ids:
                del self.synchronization.carla2sumo_ids[ego_id]
                logger.debug(f"已从同步映射中移除自车：{ego_id}")

        # 在同步之前，必须先清理掉已经到达终点的僵尸车辆！
        self.vehicle_manager.cleanup_finished_vehicles(self.sumo_simulation,self.synchronization)

        # 防护性调用桥接
        try:
            self.synchronization.tick()  # 2. 桥接同步（现在绝对不会碰自车和死车）
        except (AttributeError, TraCIException) as e:
            logger.warning(f"同步跳过异常：{str(e)}")
            pass


    def sync_reset(self):
        # 在重置时，tick 之前必须先清理上一轮残留的幽灵车！
        self.vehicle_manager.cleanup_finished_vehicles(self.sumo_simulation,self.synchronization)

        # 手动把新生成的自车从同步列表中移除（防止自车被桥接接管）
        if hasattr(self.synchronization, 'carla2sumo_ids') and self.vehicle_manager.ego_vehicle:
            ego_id = self.vehicle_manager.ego_vehicle.id
            if ego_id in self.synchronization.carla2sumo_ids:
                del self.synchronization.carla2sumo_ids[ego_id]

        self.synchronization.tick()

    def cleanup(self):
        pass