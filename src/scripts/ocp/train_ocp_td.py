"""
数据采集与环境测试脚本。
用于验证 CarlaEnv 是否能正常 reset/step，并打印观测、动作、奖励等信息。
可选保存图像或状态日志。
"""

import os
import numpy as np
import cv2
from src.utils import (load_config,get_logger,
                       setup_code_environment,
                       average_ocp_list)
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
    action_repeat = env_config['world']['action_repeat']
    # 启用sumo控制交通
    sumo_config = None
    if env_config['traffic']['enable_sumo']:
        sumo_config = load_config('configs/sumo.yaml')

    history = []
    logger.info("🚀 正在初始化 CARLA 环境...")
    env = CarlaEnv(
        render_mode=None,
        carla_config=carla_config,
        sumo_config=sumo_config,
        env_config=env_config
    )
    try:
        agent = OcpAgent(env=env, rl_config=rl_config, device=device)
        if train_config['continue_ocp']:
            logger.info("开始读取智能体参数...")
            checkpoint = agent.load(train_config["model_path_ocp"])

        logger.info("环境创建成功！")
        logger.info(f"观测空间: {env.observation_space}")
        logger.info(f"动作空间: {env.action_space}")

        num_episodes = train_config["num_episodes"]
        global_step = 0
        episode = 0
        while episode < num_episodes:
            state, info = env.reset()
            state = state['ocp_obs']
            total_reward = 0.0
            done = False
            states, actions, rewards, infos, log_probs = [], [], [], [], []
            initial_state = state.copy()
            episode_step = 0
            global_step += 1  # global_step代表“决策步”数
            loss = None
            while not done:
                # 决策时刻
                if episode_step % action_repeat == 0:
                    action, log_prob = agent.select_action(state)

                    # 初始化累计 reward 和临时 info 列表
                    accumulated_reward = 0.0
                    temp_infos = []
                    accumulated_state = []
                    # 执行 action_repeat 次动作
                    for _ in range(action_repeat):
                        accumulated_state.append(state[1])
                        if done:
                            break
                        next_obs, reward, _, _, info = env.step(action)
                        next_state = next_obs['ocp_obs']
                        done = info['collision'] or info['off_route'] or info['TimeLimit.truncated']

                        accumulated_reward += reward
                        temp_infos.append(info)
                        total_reward += reward
                        episode_step += 1  # 环境 step

                        # 更新参数
                        if agent.buffer.should_start_training():
                            loss = agent.update()

                        # 记录原始 step 日志
                        if (global_step * action_repeat + episode_step) % train_config["log_interval"] == 0:
                            logger.info(f"  EnvStep {episode_step}: reward={reward:.3f}")

                    #收集一个“决策周期”的 transition
                    states.append(average_ocp_list(accumulated_state))
                    actions.append(action)
                    rewards.append(accumulated_reward)
                    log_probs.append(log_prob)
                    # 合并 info：可以用最后一步，或自定义（如是否有 collision）
                    final_info = temp_infos[-1] if temp_infos else info
                    infos.append(final_info)

                    # 更新状态
                    state = next_state

                    # 更新惩罚参数
                    agent.update_penalty(env.step_count)  # env.step_count 是环境总步数

                    global_step += 1  # 决策步 +1

                    # 检查是否结束
                    if done:
                        logger.info(f"  Episode 结束 (info={final_info})")
                        break

                else:
                    raise RuntimeError("逻辑错误：非决策步不应在此循环中")

            # 构建 trajectory（现在 states/actions/rewards 都是决策步级别的）
            total_cost, total_constraint = agent.compute_total_cost_and_constraint(states, actions)
            trajectory = Trajectory(
                initial_state=initial_state,
                states=states,
                actions=actions,
                rewards=rewards,
                infos=infos,
                total_cost=total_cost,
                total_constraint=total_constraint,
                path_id=env.current_path_id,
                horizon=len(states),
                log_probs=log_probs
            )
            agent.buffer.handle_new_trajectory(trajectory)

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
                logger.info(f"开始保存模型：  Step {global_step}: total={total_reward:.2f}")
                save_info = {
                    'rl_config':rl_config,
                    'global_step':global_step,
                    'map':env_config['world']['map'],
                    'history_loss':history.copy(),
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