"""
数据采集与环境测试脚本。
用于验证 CarlaEnv 是否能正常 reset/step，并打印观测、动作、奖励等信息。
可选保存图像或状态日志。
"""

import os
import numpy as np
import cv2
from src.utils import (load_config,get_logger,
                       setup_code_environment)
from src.agents import OcpAgent
from src.buffer import Trajectory
# 添加项目源码路径
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import gymnasium as gym
from src.envs.carla_env import CarlaEnv

logger = get_logger('train_ocp')


def save_image(obs, step: int, save_dir: str = "debug_images"):
    """保存观测中的图像（假设 obs 是 dict 且包含 'image'）"""
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(obs, dict) and 'image' in obs:
        img = obs['image']
        # 如果是 (C, H, W)，转为 (H, W, C)
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
    device = setup_code_environment(sys_config)
    history = []
    logger.info("🚀 正在初始化 CARLA 环境...")
    env = CarlaEnv(
        render_mode=None,
        carla_config=carla_config,
        env_config=env_config
    )
    try:
        agent = OcpAgent(env=env, rl_config=rl_config, device=device)
        if train_config['continue_ocp']:
            logger.info("开始读取智能体参数...")
            checkpoint = agent.load(train_config["model_path_ocp"])
            if not env.is_eval:
                # 读取归一化参数
                env.ocp_normalizer.load_state_dict(checkpoint['ocp_normalizer'])

        logger.info("环境创建成功！")
        logger.info(f"观测空间: {env.observation_space}")
        logger.info(f"动作空间: {env.action_space}")

        num_episodes = train_config["num_episodes"]
        global_step = 0
        episode = 0
        while episode < num_episodes:
            logger.info(f"\n开始第 {episode + 1} 轮测试...")
            state, info = env.reset()
            state = state['ocp_obs']
            logger.info(f"初始观测类型: {type(state)}, 形状/结构: {get_obs_shape(state)}")
            total_reward = 0.0
            done = False
            states, actions, rewards, infos ,log_probs= [], [], [], [],[]
            initial_state = state.copy()
            while not done:
                action,log_prob = agent.select_action(state)
                next_obs, reward, _, _, info = env.step(action)
                next_state = next_obs['ocp_obs']
                done = info['collision'] or info['off_route'] or info['TimeLimit.truncated']
                total_reward += reward
                # 数据加入buffer
                actions.append(action)
                states.append(state[1])
                rewards.append(reward)
                log_probs.append(log_prob)
                infos.append(info)
                state = next_state

                # 更新惩罚参数
                agent.update_penalty(env.step_count)
                # 打印关键信息
                if global_step % train_config["log_interval"] == 0:
                    logger.info(f"  Step {global_step}: reward={reward:.3f}, total={total_reward:.2f}")
                    if 'speed' in info:
                        logger.info(f"    速度: {info['speed']:.2f} km/h")

                if done:
                    logger.info(f"  Episode 结束 (info={info})")
                    break
                global_step += 1
            # 计算 total_cost 和 total_constraint
            total_cost, total_constraint = agent.compute_total_cost_and_constraint(states, actions)
            trajectory = Trajectory(initial_state=initial_state,
                                    states=states,actions=actions,
                                    rewards=rewards,
                                    infos=infos,
                                    total_cost=total_cost,
                                    total_constraint=total_constraint,
                                    path_id=env.current_path_id,
                                    horizon=len(states),
                                    log_probs = log_probs)
            # 加入buffer
            agent.buffer.handle_new_trajectory(trajectory)

            # 更新参数
            loss = None
            if agent.buffer.should_start_training():
                loss = agent.update()
            logger.info(f"第 {episode} 轮完成，总奖励: {total_reward:.2f}")

            if loss is not None:
                logger.info(f"训练损失: actor_loss:{loss['actor_loss']:.5f},critic_loss:{loss['critic_loss']:.5f},"
                            f"惩罚参数：{agent.init_penalty:.5f}")
                loss.update({
                    'global_step': global_step
                })
                history.append(loss)

            episode += 1

            # 保存模型
            if episode % train_config["save_freq"] == 0:
                logger.info(f"开始保存模型：  Step {global_step}: reward={reward['total_reward']:.3f}, total={total_reward:.2f}")
                save_info = {
                    'rl_config':rl_config,
                    'global_step':global_step,
                    'map':env_config['world']['map'],
                    'history_loss':history,
                    'ocp_normalizer': env.ocp_normalizer.state_dict()
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