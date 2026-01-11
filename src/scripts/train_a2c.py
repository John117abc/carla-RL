"""
数据采集与环境测试脚本。
用于验证 CarlaEnv 是否能正常 reset/step，并打印观测、动作、奖励等信息。
可选保存图像或状态日志。
"""

import os
import torch
import numpy as np
import cv2
from src.utils import load_config,get_logger,setup_code_environment,checkpoint
from src.agents import A2CAgent

# # === 添加 CARLA 路径（请根据你的实际路径修改）===
# CARLA_ROOT = "/home/user/CARLA_0.9.16"  # 请替换为你的 CARLA 根目录
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
    sys_config = load_config('configs/sys.yaml')
    rl_config = load_config('configs/rl.yaml')
    train_config = load_config('configs/train.yaml')
    device = setup_code_environment(sys_config)
    train_param = train_config["train"]
    logger.info("🚀 正在初始化 CARLA 环境...")
    env = CarlaEnv(  # 或直接使用类：CarlaEnv()
        render_mode=None,  # 设为 'human' 可显示 CARLA 视窗（但会变慢）
        carla_config=carla_config,
        env_config=env_config
    )
    agent = A2CAgent(env=env,rl_config=rl_config, device=device)
    obs, _ = env.reset()
    history = []
    try:
        logger.info("✅ 环境创建成功！")
        logger.info(f"观测空间: {env.observation_space}")
        logger.info(f"动作空间: {env.action_space}")

        num_episodes = train_param["num_episodes"]
        max_step = train_param["max_step"]
        for ep in range(num_episodes):
            logger.info(f"\n▶️  开始第 {ep + 1} 轮测试...")
            obs, info = env.reset()
            logger.info(f"初始观测类型: {type(obs)}, 形状/结构: {get_obs_shape(obs)}")
            total_reward = 0.0
            now_step = 0
            done = False
            while now_step <= max_step and not done:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
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

                now_step+=1
                # 打印关键信息
                if now_step % train_param["log_interval"] == 0:
                    logger.info(f"  Step {now_step}: reward={reward:.3f}, total={total_reward:.2f}")
                    if 'speed' in info:
                        logger.info(f"    速度: {info['speed']:.2f} km/h")
                    # 记录日志
                    history.append(metrics)

                # 可选：保存图像（调试用）
                # save_image(obs, now_step)

                if done:
                    logger.info(f"  ⏹️  Episode 结束 (terminated={terminated}, truncated={truncated})")
                    break
            # 保存模型
            # if ep % train_param["save_freq"] == 0:
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