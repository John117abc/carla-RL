import numpy as np
import torch
from src.utils import get_logger
logger = get_logger('per_buffer')

class SumTree:
    """SumTree 数据结构，用于高效按优先级采样"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)  # 存储经验
        self.data_pointer = 0          # 环形缓冲区指针
        self.n_entries = 0             # 当前存储的样本数

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)


    def get_leaf(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self):
        return self.tree[0]


class PERBuffer:
    """
    优先经验回放缓冲区，与 StochasticBuffer 接口兼容
    经验结构为 (state, action, reward, value, done, info)
    """
    def __init__(self, capacity=100000, min_start_train=256,
                 alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.min_start_train = min_start_train
        self.alpha = alpha          # 优先级指数 (0: 均匀采样, 1: 完全按优先级)
        self.beta = beta            # 重要性采样修正系数
        self.beta_increment = beta_increment
        self.max_priority = 1.0     # 记录最大优先级，用于新样本
        self.last_indices = None    # 最近一次采样使用的索引

    def _calc_initial_priority(self, experience):
        """根据经验内容估算初始优先级"""
        _, _, reward, _, _, info = experience
        # 安全关键样本给予极高初始优先级
        if info.get('collision', False) or info.get('collision_reward', 0) < -0.5:
            return 10.0
        elif info.get('collision_reward', 0) < 0:
            return 5.0
        elif info.get('centering_reward', 1.0) < 0.5:  # 偏离道路
            return 3.0
        else:
            return 1.0

    def handle_new_experience(self, experience):
        """自动计算初始优先级并存储"""
        priority = self._calc_initial_priority(experience)
        self.add(experience, priority)

    def add_safety_trajectory(self, transitions):
        """强制以最高优先级存储整条轨迹（用于碰撞/危险场景）"""
        for exp in transitions:
            self.add(exp, 10.0)

    def add(self, experience, priority):
        """手动添加经验，优先级会被 alpha 次方处理"""
        p = max(priority, 1e-6) ** self.alpha
        self.tree.add(p, experience)
        self.max_priority = max(self.max_priority, p)

    def sample_batch(self, batch_size):
        """
        采样一批经验，返回与原 StochasticBuffer 兼容的列表。
        同时内部保存本次采样的索引，供后续 update_last_batch_priorities 使用。
        """
        if self.tree.n_entries < batch_size:
            batch_size = self.tree.n_entries
        if batch_size == 0:
            return []

        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        # 保存本次采样索引，供 update_last_batch_priorities 使用
        self.last_indices = idxs
        return batch

    def update_last_batch_priorities(self, new_priorities):
        """
        根据 OCP 计算出的违规量（或其他指标）更新最近一批样本的优先级。
        new_priorities: list/array of float, 长度与上次采样的 batch_size 相同
        """
        p_array = np.array(new_priorities, dtype=np.float64)
        if len(np.unique(p_array)) > 1:
            logger.debug(f"经验缓冲区更新 update: {len(p_array)} priorities, unique values: {len(np.unique(p_array))}")

        if self.last_indices is None:
            return
        p_array = np.array(new_priorities, dtype=np.float64)
        p_array = np.maximum(p_array, 1e-6) ** self.alpha
        for idx, p in zip(self.last_indices, p_array):
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def should_start_training(self):
        return self.tree.n_entries >= self.min_start_train

    def clear(self):
        """重置缓冲区"""
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self.last_indices = None

    def get_save_buffer_data(self):
        """获取用于保存的数据"""
        return {
            'tree_data': self.tree.tree.copy(),
            'data': self.tree.data.copy(),
            'data_pointer': self.tree.data_pointer,
            'n_entries': self.tree.n_entries,
            'max_priority': self.max_priority,
            'beta': self.beta,
            'capacity': self.capacity,
            'min_start_train': self.min_start_train
        }

    def load_buffer_data(self, data):
        """从数据恢复缓冲区"""
        self.tree.tree = data['tree_data'].copy()
        self.tree.data = data['data'].copy()
        self.tree.data_pointer = data['data_pointer']
        self.tree.n_entries = data['n_entries']
        self.max_priority = data.get('max_priority', 1.0)
        self.beta = data.get('beta', self.beta)
        self.capacity = data.get('capacity', self.capacity)
        self.min_start_train = data.get('min_start_train', self.min_start_train)

    def __len__(self):
        return self.tree.n_entries