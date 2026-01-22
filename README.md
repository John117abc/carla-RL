# carla-RL  
**基于CARLA与SUMO的联合仿真强化学习训练框架**  
兼容 Gymnasium 环境与 Stable-Baselines3，支持多场景配置化训练

---

## 📌 项目简介

本项目旨在构建一个高效、灵活的强化学习训练平台，通过整合 **CARLA 高保真驾驶仿真** 与 **SUMO 交通流仿真**，实现交通场景的精细化建模与智能体决策训练。核心特点包括：

- **兼容主流生态**：基于 Gymnasium 标准环境接口，无缝对接 Stable-Baselines3 强化学习算法库  
- **多场景支持**：支持 CARLA 多种地图与天气条件，可配置 SUMO 交通流（车辆、路口、信号灯）  
- **模块化设计**：通过 `src` 目录下的模块化组件（环境、模型、缓存等）实现功能扩展  
- **强化学习训练**：提供从环境初始化、仿真同步到模型训练的完整流程  

---

## 🛠 功能特点

| 特性 | 说明 |
|------|------|
| ✅ 联合仿真 | CARLA 负责车辆动力学与传感器仿真，SUMO 负责宏观交通流管理，通过自定义同步模块保障时间步一致 |
| ✅ 配置驱动 | 通过 `configs` 目录下的配置文件灵活定义环境参数（地图、天气）、SUMO 场景（`.net` 文件、`.rou` 文件）、训练超参 |
| ✅ 算法支持 | 开箱即用的 Stable-Baselines3 算法（如 PPO、A2C、SAC），支持自定义策略网络 |
| ✅ 缓存机制 | `buffer` 模块提供经验回放缓冲区，适用于需要回放的算法（如 DQN） |
| ✅ 实用工具 | `utils` 目录提供状态预处理、动作映射、日志记录等辅助功能 |

---

## 🧰 环境搭建

### 1. 系统要求
- Linux/macOS（推荐 Ubuntu 20.04+）
- Python 3.8+
- [CARLA Simulator 0.9.16+](https://carla.org/)
- [SUMO 1.16+](https://sumo.dlr.de/docs/Installing/index.html)

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/John117abc/carla-RL.git
cd carla-RL

# 创建并激活Python虚拟环境（推荐）
python -m venv rl-env
source rl-env/bin/activate  # Windows: rl-env\Scripts\activate

# 安装Python依赖
pip install -r requirements.txt
# 或手动安装关键包
pip install gymnasium stable-baselines3 numpy pysumo
```
### 3.⚡快速入门
```bash
1. 启动仿真器
bash

# 终端1：启动CARLA服务器
./CarlaUE4.sh  # Linux/macOS
# 或
CarlaUE4.exe    # Windows

# 终端2：启动SUMO仿真
sumo -c src/envs/carla_sumo_env/town06_opt_env/Town06_Opt.sumocfg

# 训练脚本目录
src/scripts

carla-RL/
├── carla_agents/        # CARLA智能体代码，从carlaAPI中迁移过来的，实现路径规划
├── carla_note/          # carla的一些基础使用教程
├── src/                 # 核心源码目录
│   ├── agents/          # 智能体实现
│   ├── buffer/          # 经验回放缓冲区
│   ├── carla_utils/     # CARLA仿真工具
│   ├── configs/         # 配置文件（环境/训练）
│   ├── envs/            # Gymnasium环境封装
│   ├── models/          # 神经网络模型
│   ├── scripts/         # 训练/评估脚本
│   ├── sumo_sync/       # SUMO同步逻辑
│   └── utils/           # 工具函数（日志/可视化）
├── sumo_sync/           # SUMO网络文件
├── requirements.txt     # 依赖列表
├── LICENSE              # 开源协议
└── README.md            # 项目说明
```
### 4.🚀使用示例

```bash
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

import gymnasium as gym
from src.envs.carla_env import CarlaEnv

logger = get_logger('train_ocp')

def main():
    logger.info('开始读取配置文件...')
    carla_config = load_config('configs/carla.yaml')['word_01']
    env_config = load_config('configs/env.yaml')
    sys_config = load_config('configs/sys.yaml')
    rl_config = load_config('configs/rl.yaml')
    train_config = rl_config['rl']
    device = setup_code_environment(sys_config)
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
            # if not env.is_eval:
            #     # 读取归一化参数
            #     env.ocp_normalizer.load_state_dict(checkpoint['ocp_normalizer'])

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
                logger.info(f"开始保存模型：  Step {global_step}: total={total_reward:.2f}")
                save_info = {
                    'rl_config':rl_config,
                    'global_step':global_step,
                    'map':env_config['world']['map'],
                    'history_loss':history.copy(),
                    # 'ocp_normalizer': env.ocp_normalizer.state_dict()
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
```

### 5.🌟 贡献指南

    提交 Issue：报告 Bug 或提出功能建议
    Fork 项目：在 GitHub 上 Fork 本项目
    提交 PR：完善文档 / 添加新功能
    交流：通过 GitHub Issues 讨论

    欢迎 Star 支持！⭐️

📜 开源协议
本项目采用 MIT License 开源，允许商业和非商业使用，无需额外授权。
使用时请保留原作者信息（John117abc）。