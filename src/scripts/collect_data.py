"""
数据采集与环境测试脚本。
用于验证 CarlaEnv 是否能正常 reset/step，并打印观测、动作、奖励等信息。
可选保存图像或状态日志。
"""

import sys
import os
import time
import numpy as np
import cv2
from src.utils import load_config,get_logger

# # === 添加 CARLA 路径（请根据你的实际路径修改）===
# CARLA_ROOT = "/home/user/CARLA_0.9.16"  # 👈 请替换为你的 CARLA 根目录
# sys.path.append(os.path.join(CARLA_ROOT, 'PythonAPI'))
# sys.path.append(os.path.join(CARLA_ROOT, 'PythonAPI/carla'))

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
    logger.info("🚀 正在初始化 CARLA 环境...")
    env = CarlaEnv(  # 或直接使用类：CarlaEnv()
        render_mode=None,  # 设为 'human' 可显示 CARLA 视窗（但会变慢）
        carla_config=carla_config,
        env_config=env_config
    )
    try:
        logger.info("✅ 环境创建成功！")
        logger.info(f"观测空间: {env.observation_space}")
        logger.info(f"动作空间: {env.action_space}")

        num_episodes = 3
        for ep in range(num_episodes):
            logger.info(f"\n▶️  开始第 {ep + 1} 轮测试...")
            obs, info = env.reset()
            logger.info(f"初始观测类型: {type(obs)}, 形状/结构: {get_obs_shape(obs)}")
            total_reward = 0.0

            for step in range(1000):  # 每轮最多 200 步
                # 随机动作（也可替换为固定动作，如 [0.0, 1.0] 表示直行加速）
                action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                # 打印关键信息（每 20 步）
                if step % 20 == 0:
                    logger.info(f"  Step {step}: reward={reward:.3f}, total={total_reward:.2f}")
                    if 'speed' in info:
                        logger.info(f"    速度: {info['speed']:.2f} km/h")

                # 可选：保存图像（调试用）
                # save_image(obs, step)

                if terminated or truncated:
                    logger.info(f"  ⏹️  Episode 结束 (terminated={terminated}, truncated={truncated})")
                    break

            logger.info(f"✅ 第 {ep + 1} 轮完成，总奖励: {total_reward:.2f}")

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