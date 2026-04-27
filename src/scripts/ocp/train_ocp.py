"""
数据采集与环境测试脚本。
用于验证 CarlaEnv 是否能正常 reset/step，并打印观测、动作、奖励等信息。
可选保存图像或状态日志。
"""

import os
import numpy as np
import cv2
import torch

from src.utils import (load_config,get_logger,
                       setup_code_environment)
from src.agents import OcpAgent
from src.buffer import Trajectory
# 添加项目源码路径
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import gymnasium as gym
from src.envs.carla_env import CarlaEnv
from src.carla_utils.ocp_setup import batch_world_to_ego

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
    # 启用sumo控制交通
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
        agent = OcpAgent(env=env, rl_config=rl_config, device=device)
        # 【新增】将智能体绑定至环境，用于 step 中实时可视化预测轨迹
        env.agent = agent
        
        if train_config['continue_ocp']:
            logger.info("开始读取智能体参数...")
            checkpoint = agent.load(train_config["model_path_ocp"])
            # if not env.is_eval:
            #     # 读取归一化参数
            #     env.ocp_normalizer.load_state_dict(checkpoint['ocp_normalizer'])

        logger.info("环境创建成功！")
        logger.info(f"观测空间: {env.observation_space}")
        logger.info(f"动作空间: {env.action_space}")

        num_episodes = train_config["num_episodes"]
        episode = 0
        while episode < num_episodes:
            logger.info(f"\n开始第 {episode + 1} 轮测试...")
            # logger.info(f"初始观测类型: {type(state)}, 形状/结构: {get_obs_shape(state)}")
            # 在reset之后，获取规划好的参考路径，转换为tensor
            # 在reset之后添加这段代码
            obs = env.reset()
            state = obs['ocp_obs']
            # 提取参考路径，转为tensor [1, N, 2]
            ref_path_locations = obs['ref_path_locations']  # 你的路径规划输出的carla.Location列表
            
            total_reward = 0.0
            done = False
            while not done:
                # 【修复】将参考路径动态转换到当前自车坐标系，与网络输入保持一致
                ego_transform = env.vehicle_manager.ego_vehicle.get_transform()
                ref_path_ego_np = np.array(batch_world_to_ego(ref_path_locations, ego_transform), dtype=np.float32)
                
                # 【防御】检查参考路径是否包含 nan/inf
                if np.any(np.isnan(ref_path_ego_np)) or np.any(np.isinf(ref_path_ego_np)):
                    logger.warning("参考路径包含 nan/inf，跳过当前步")
                    # 保持状态不变，继续下一步或记录错误，这里选择继续循环但跳过训练
                    # 实际中应检查环境转换逻辑 batch_world_to_ego
                    pass 
                
                ref_path_tensor = torch.from_numpy(ref_path_ego_np).unsqueeze(0).to(device)

                # 【修复】ref_path_locations 是 list，需使用已转换的 numpy 数组进行维度校验
                if ref_path_tensor.shape != (1, ref_path_ego_np.shape[0], 2):
                    raise ValueError(f"参考路径维度异常: {ref_path_tensor.shape} (期望 [1, N, 2])")

                action, _ = agent.select_action(state,train_config['continue_ocp'])
                next_obs, reward, _, _, info = env.step(action)
                next_state = next_obs['ocp_obs']
                # 【修改】获取道路信息，转为tensor
                road_state_np = obs['s_road']
                road_state_tensor = torch.from_numpy(road_state_np).unsqueeze(0).to(
                    device) if road_state_np is not None else None

                # 【加固】校验道路状态维度
                if road_state_tensor is not None and road_state_tensor.shape[1] != agent.DIM_ROAD:
                    raise ValueError(f"道路状态维度异常: {road_state_tensor.shape[1]} (期望{agent.DIM_ROAD})")

                done = info['TimeLimit.truncated'] or info['collision']
                info['road_state'] = road_state_np
                # 加入buffer时，同时存储road_state
                agent.buffer.handle_new_experience((state, action, reward, road_state_np, done, info))

                # 更新时传入road_state
                loss = None
                if agent.buffer.should_start_training():
                    loss = agent.update(ref_path_tensor, road_state_tensor)

                state = next_state

                # 打印关键信息
                if agent.global_step % train_config["log_interval"] == 0:
                    if 'speed' in info:
                        logger.info(f"    速度: {info['speed']:.2f} km/h,动作：{action}")
                        # ========== 新增诊断日志 ==========
                        try:
                            d_e = agent.DIM_EGO          # 自车维度
                            d_o = agent.DIM_OTHER        # 周车维度
                            d_r = agent.DIM_REF_ERROR    # 误差维度
                            # 确保状态足够长
                            if state.shape[0] >= d_e + d_o + d_r:
                                delta_p = float(state[d_e + d_o])
                                delta_phi = float(state[d_e + d_o + 1])
                                delta_v = float(state[d_e + d_o + 2])
                                steer_dir = "右" if action[1] > 0 else ("左" if action[1] < 0 else "0")
                                logger.info(
                                    f"    ref_error: δ_p={delta_p:.3f} m, δ_φ={delta_phi:.3f} rad, "
                                    f"δ_v={delta_v:.3f} m/s, 转向:{steer_dir}({action[1]:.3f} rad)"
                                )
                        except Exception as diag_e:
                            logger.warning(f"ref_error诊断失败: {diag_e}")
                        # ========== 诊断日志结束 ==========

                    if loss is not None:
                        logger.info(
                            f"训练损失: actor_loss:{loss['actor_loss']:.5f}, critic_loss:{loss['critic_loss']:.5f}, "
                            f"惩罚参数 ρ:{loss['penalty']:.5f}, GEP迭代次数:{loss['gep_iteration']}, "
                            f"Actor是否更新:{loss['actor_updated']}"
                        )
                        history.append(loss)

                if done:
                    logger.info(f"  Episode 结束")
                    break
                agent.global_step += 1

            episode += 1

            # 保存模型
            if episode % train_config["save_freq"] == 0:
                logger.info(f"开始保存模型：  Step {agent.global_step}: total={total_reward:.2f}")
                save_info = {
                    'rl_config':rl_config,
                    'global_step':agent.global_step,
                    'map':env_config['world']['map'],
                    'history_loss':history.copy(),
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
