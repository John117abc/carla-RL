"""
数据采集与环境测试脚本。
用于验证 CarlaEnv 是否能正常 reset/step，并打印观测、动作、奖励等信息。
可选保存图像或状态日志。
"""

import os
import torch
import numpy as np
import cv2
import sys
from src.utils import (load_config,get_logger,
                       setup_code_environment,
                       load_checkpoint)
from src.agents import A2CAgent

# === 添加项目源码路径 ===
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import gymnasium as gym
from src.envs.carla_env import CarlaEnv  # 假设你的环境类在这里

logger = get_logger()


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
    carla_config = load_config('configs/carla.yaml')
    env_config = load_config('configs/env.yaml')
    sys_config = load_config('configs/sys.yaml')
    rl_config = load_config('configs/rl.yaml')
    train_config = rl_config['rl']
    device = setup_code_environment(sys_config)
    history = []
    try:
        logger.info("🚀 正在初始化 CARLA 环境...")
        env = CarlaEnv(
            render_mode=None,
            carla_config=carla_config,
            env_config=env_config
        )
        agent = A2CAgent(env=env, rl_config=rl_config, device=device)
        if train_config['continue_a2c']:
            logger.info("开始读取智能体参数...")
            agent.load(train_config["model_path_a2c"])

        logger.info("✅ 环境创建成功！")
        logger.info(f"观测空间: {env.observation_space}")
        logger.info(f"动作空间: {env.action_space}")

        num_episodes = train_config["num_episodes"]
        global_step = 0
        episode = 0
        while episode < num_episodes:
            logger.info(f"\n▶️  开始第 {episode + 1} 轮测试...")
            obs, info = env.reset()
            obs = obs['measurements']
            logger.info(f"初始观测类型: {type(obs)}, 形状/结构: {get_obs_shape(obs)}")
            total_reward = 0.0
            done = False
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, _, _, info = env.step(action)
                next_obs = next_obs['measurements']
                done = info['collision'] or info['off_route'] or info['TimeLimit.truncated']
                total_reward += reward

                # 构造 batch（单步）
                batch = {
                    "obs": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device),
                    "action": torch.as_tensor(action, dtype=torch.float32).unsqueeze(0).to(device),
                    "reward": torch.as_tensor([reward], dtype=torch.float32).to(device),
                    "next_obs": torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device),
                    "done": torch.as_tensor([done], dtype=torch.bool).to(device),
                }

                metrics = agent.update(batch)
                obs = next_obs

                global_step+=1
                # 打印关键信息
                if global_step % train_config["log_interval"] == 0:
                    logger.info(f"  Step {global_step}: reward={reward:.3f}, total={total_reward:.2f}")
                    if 'speed' in info:
                        logger.info(f"    速度: {info['speed']:.2f} km/h")
                    # 记录日志
                    history.append(metrics)

                # 可选：保存图像（调试用）
                # save_image(obs, now_step)

                if done:
                    logger.info(f"  ⏹️  Episode 结束 (info={info})")
                    break
            episode += 1

            logger.info(f"✅ 第 {episode} 轮完成，总奖励: {total_reward:.2f}")

            # 保存模型
            if episode % train_config["save_freq"] == 0:
                logger.info(f"开始保存模型：  Step {global_step}: reward={reward:.3f}, total={total_reward:.2f}")
                agent.save(rl_config,global_step,episode,env_config['world']['map'])

    except Exception as e:
        logger.error(f"❌ 环境运行出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        logger.info("\n🧹 正在关闭环境...")
        env.close()
        logger.info("👋 测试结束。")


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