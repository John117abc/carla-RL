"""
在线模式训练入口脚本。
严格对齐 OCP 在线单步更新逻辑，不依赖大规模经验回放缓冲区。
"""

import os
import numpy as np
import cv2
import torch

from src.utils import (load_config, get_logger, setup_code_environment)
from src.agents import OcpAgentOnline
from src.envs.carla_env import CarlaEnv
from src.carla_utils.ocp_setup import batch_world_to_ego

logger = get_logger('train_ocp_online')


def save_image(obs, step: int, save_dir: str = "debug_images"):
    """保存观测中的图像（假设 obs 是 dict 且包含 'image'）"""
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(obs, dict) and 'image' in obs:
        img = obs['image']
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(f"{save_dir}/step_{step:04d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    elif isinstance(obs, np.ndarray) and obs.ndim == 3:
        img = (obs * 255).astype(np.uint8)
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        cv2.imwrite(f"{save_dir}/step_{step:04d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    logger.info('开始读取配置文件...')
    carla_config = load_config('configs/carla.yaml')['word_01']
    env_config = load_config('configs/env.yaml')
    sys_config = load_config('configs/sys.yaml')
    rl_config = load_config('configs/rl.yaml')
    train_config = rl_config['rl']
    
    sumo_config = None
    if env_config['traffic']['enable_sumo']:
        sumo_config = load_config('configs/sumo.yaml')
        
    device = setup_code_environment(sys_config)
    history = []
    logger.info("🚀 正在初始化 CARLA 环境...")
    
    env = CarlaEnv(
        render_mode=None,
        carla_config=carla_config,
        sumo_config=sumo_config,
        env_config=env_config
    )
    
    try:
        agent = OcpAgentOnline(env=env, rl_config=rl_config, device=device)
        env.agent = agent  # 绑定智能体用于可视化
        
        if train_config['continue_ocp']:
            logger.info("开始读取智能体参数...")
            checkpoint = agent.load(train_config["model_path_ocp"])

        logger.info("环境创建成功！")
        logger.info(f"观测空间: {env.observation_space}")
        logger.info(f"动作空间: {env.action_space}")

        num_episodes = train_config["num_episodes"]
        episode = 0
        while episode < num_episodes:
            logger.info(f"\n开始第 {episode + 1} 轮测试...")
            obs = env.reset()
            state = obs['ocp_obs']
            ref_path_locations = obs['ref_path_locations']  # carla.Location 列表
            
            # 【修复】正确转换参考路径为 numpy 数组
            ref_path_np = np.array([[loc.x, loc.y] for loc in ref_path_locations], dtype=np.float32)
            ref_path_tensor = torch.from_numpy(ref_path_np).unsqueeze(0).to(device)
            
            # 【修复】校验参考路径维度
            if ref_path_tensor.shape != (1, ref_path_np.shape[0], 2):
                raise ValueError(f"参考路径维度异常: {ref_path_tensor.shape} (期望 [1, N, 2])")

            total_reward = 0.0
            done = False
            while not done:
                action, _ = agent.select_action(state, train_config['continue_ocp'])
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = next_obs['ocp_obs']
                
                # 【修复】正确累加奖励
                total_reward += reward
                
                # 获取道路信息
                road_state_np = obs.get('s_road', None)
                if road_state_np is not None:
                    road_state_np = np.asarray(road_state_np, dtype=np.float32).flatten()
                    # 【加固】校验道路状态维度
                    if road_state_np.shape[0] != agent.DIM_ROAD:
                        raise ValueError(f"道路状态维度异常: {road_state_np.shape[0]} (期望{agent.DIM_ROAD})")
                
                # 【修复】正确判断 episode 结束
                done = terminated or truncated
                
                # 更新 buffer 记录（仅用于日志，不参与在线训练）
                info['road_state'] = road_state_np if road_state_np is not None else np.zeros(agent.DIM_ROAD, dtype=np.float32)
                agent.buffer.handle_new_experience((state, action, reward, info['road_state'], done, info))

                # 在线单步更新
                loss = agent.update(state, info['road_state'], ref_path_tensor)

                state = next_state

                # 打印关键信息
                if agent.global_step % train_config["log_interval"] == 0:
                    if 'speed' in info:
                        logger.info(f"    速度: {info['speed']:.2f} km/h, 动作: {action}")

                    if loss is not None:
                        logger.info(
                            f"训练损失: actor_loss:{loss['actor_loss']:.5f}, critic_loss:{loss['critic_loss']:.5f}, "
                            f"惩罚参数 ρ:{loss['penalty']:.5f}, GEP迭代次数:{loss['gep_iteration']}, "
                            f"Actor是否更新:{loss['actor_updated']}"
                        )
                        history.append(loss)

                if done:
                    logger.info(f"  Episode 结束, 总奖励: {total_reward:.2f}")
                    break
                    
                agent.global_step += 1

            episode += 1

            # 保存模型
            if episode % train_config["save_freq"] == 0:
                logger.info(f"开始保存模型：  Step {agent.global_step}: total={total_reward:.2f}")
                save_info = {
                    'rl_config': rl_config,
                    'global_step': agent.global_step,
                    'map': env_config['world']['map'],
                    'history_loss': history.copy(),
                    'buffer_data': agent.buffer.get_save_buffer_data()
                }
                agent.save(save_info)

    except Exception as e:
        logger.error(f"环境运行出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        logger.info("\n正在关闭环境...")
        env.close()
        logger.info("测试结束。")


def get_obs_shape(obs):
    """辅助函数：递归打印观测结构"""
    if isinstance(obs, dict):
        return {k: get_obs_shape(v) for k, v in obs.items()}
    elif isinstance(obs, np.ndarray):
        return obs.shape
    else:
        return type(obs)


if __name__ == "__main__":
    main()
