"""
数据采集与环境测试脚本。
用于验证 CarlaEnv 是否能正常 reset/step，并打印观测、动作、奖励等信息。
可选保存图像或状态日志。
"""


from src.utils import (load_config,get_logger,
                       setup_code_environment)
from src.agents import OcpAgent
from src.envs.carla_env import CarlaEnv
from src.utils import Plotter

logger = get_logger('eval_idc')
def main():
    logger.info('开始读取配置文件...')
    carla_config = load_config('configs/carla.yaml')['word_01']
    env_config = load_config('configs/env.yaml')
    sys_config = load_config('configs/sys.yaml')
    rl_config = load_config('configs/rl.yaml')
    eval_config = rl_config['rl']
    # 启用sumo控制交通
    sumo_config = None
    if env_config['traffic']['enable_sumo']:
        sumo_config = load_config('configs/sumo.yaml')
    device = setup_code_environment(sys_config)
    logger.info("🚀 正在初始化 CARLA 环境...")
    env = CarlaEnv(
        render_mode=None,
        carla_config=carla_config,
        train_config=rl_config,
        sumo_config=sumo_config,
        env_config=env_config
    )
    try:
        agent = OcpAgent(env=env, rl_config=rl_config, device=device)

        if eval_config['continue_idc']:
            logger.info("开始读取智能体参数...")
            checkpoint = agent.load(eval_config["model_path_idc"])

        logger.info("环境创建成功！")
        logger.info(f"观测空间: {env.observation_space}")
        logger.info(f"动作空间: {env.action_space}")

        if eval_config['continue_idc']:
            logger.info("开始读取智能体参数...")
            checkpoint = agent.load(eval_config["model_path_idc"])
            logger.info('开始绘制图像')
            plotter = Plotter()
            plotter.load_history(checkpoint['history'])
            plotter.plot_training_metrics()

        num_eval = eval_config['num_eps_eval']
        agent.eval(num_eval)

    except Exception as e:
        logger.error(f"环境运行出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        logger.info("\n正在关闭环境...")
        env.close()
        logger.info("测试结束.")

if __name__ == "__main__":
    main()
