# src/utils/replay_buffer.py

import numpy as np
import torch


class ReplayBuffer:
    """
    简单的经验回放缓冲区，支持字典形式采样。
    适用于 off-policy 算法（如 SAC、DDPG）。
    """

    def __init__(self, capacity, obs_space, act_space, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_space.shape
        self.act_shape = act_space.shape

        # 预分配内存
        self.obs_buf = np.empty((capacity,) + self.obs_shape, dtype=np.float32)
        self.next_obs_buf = np.empty((capacity,) + self.obs_shape, dtype=np.float32)
        self.acts_buf = np.empty((capacity,) + self.act_shape, dtype=np.float32)
        self.rews_buf = np.empty(capacity, dtype=np.float32)
        self.done_buf = np.empty(capacity, dtype=bool)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """添加一条经验"""
        self.obs_buf[self.ptr] = np.array(obs, dtype=np.float32)
        self.next_obs_buf[self.ptr] = np.array(next_obs, dtype=np.float32)
        self.acts_buf[self.ptr] = np.array(action, dtype=np.float32)
        self.rews_buf[self.ptr] = float(reward)
        self.done_buf[self.ptr] = bool(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """随机采样一个批次"""
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(
            obs=torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32, device=self.device),
            next_obs=torch.as_tensor(self.next_obs_buf[idxs], dtype=torch.float32, device=self.device),
            action=torch.as_tensor(self.acts_buf[idxs], dtype=torch.float32, device=self.device),
            reward=torch.as_tensor(self.rews_buf[idxs], dtype=torch.float32, device=self.device),
            done=torch.as_tensor(self.done_buf[idxs], dtype=torch.bool, device=self.device),
        )
        return batch

    def __len__(self):
        return self.size