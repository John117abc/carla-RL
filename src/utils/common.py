import numpy as np
from pathlib import Path

def normalize_Kinematics_obs(obs):
    """
    归一化Kinematics观察到的结果
    :param obs:
    :return:
    """
    # obs: (10, 7) or (70,)
    obs = obs.reshape(-1, 7)
    obs[:, 1] /= 100.0  # x
    obs[:, 2] /= 100.0  # y
    obs[:, 3] /= 20.0   # vx
    obs[:, 4] /= 20.0   # vy
    # presence, cos_h, sin_h 已经归一化了
    return obs.flatten()


def get_project_root() -> Path:

    """返回项目根目录"""

    return Path(__file__).parent.parent.resolve()



class RunningNormalizer:
    """
    归一化
    """
    def __init__(self, shape, eps=1e-8, clip_range=(-5.0, 5.0)):
        self.eps = eps
        self.clip_range = clip_range
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        """x: (..., *shape)"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        if self.clip_range is not None:
            x_norm = np.clip(x_norm, self.clip_range[0], self.clip_range[1])
        return x_norm

    def denormalize(self, x_norm):
        """将归一化后的 x_norm 反归一化回原始尺度"""
        x = x_norm * np.sqrt(self.var + self.eps) + self.mean
        return x

    def state_dict(self):
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count
        }

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.var = np.asarray(state["var"], dtype=np.float32)
        self.count = state["count"]


import numpy as np


def normalize_ocp__scenario_relative(data):
    """
    基于自车的局部观测归一化（观测半径 50m）

    输入 data 结构:
      - data[0]: ego [x, y, vx, vy, yaw, acc]
      - data[1]: 8 agents, each [x, y, vx, vy, yaw, acc]
      - data[2]: road_edge [x, y, 0, 0, 0, 0]
      - data[3]: ref_path [x, y, vx, vy, yaw, acc]

    输出：所有位置转为 (ego_x, ego_y) 为中心的相对坐标，并归一化到 [-1, 1]
    """
    # 物理量程（可根据你的场景调整）
    PHYSICAL_RANGES = {
        'rel_x': (-50.0, 50.0),  # 相对x：-50 ～ 50 米
        'rel_y': (-50.0, 50.0),  # 相对y：-50 ～ 50 米
        'vx': (-60.0, 60.0),  # 速度 x (m/s)
        'vy': (-60.0, 60.0),  # 速度 y (m/s)
        'yaw': (-np.pi, np.pi),  # 航向角 (rad)
        'acc': (-10.0, 10.0)  # 加速度 (m/s²)
    }

    keys = ['rel_x', 'rel_y', 'vx', 'vy', 'yaw', 'acc']

    def norm(val, key):
        low, high = PHYSICAL_RANGES[key]
        # Min-Max 到 [-1, 1]
        normalized = 2 * (val - low) / (high - low) - 1
        return np.clip(normalized, -1.0, 1.0)

    # 提取自车信息
    ego = np.array(data[0], dtype=np.float32)
    ego_x, ego_y = ego[0], ego[1]

    # 辅助函数：将一个六维向量转为相对坐标并归一化
    def process_entity(entity):
        ent = np.array(entity, dtype=np.float32)
        # 位置转相对坐标
        rel_x = ent[0] - ego_x
        rel_y = ent[1] - ego_y
        # 构建新向量 [rel_x, rel_y, vx, vy, yaw, acc]
        new_vec = np.array([rel_x, rel_y, ent[2], ent[3], ent[4], ent[5]])
        # 各维度归一化
        normalized = np.array([norm(new_vec[i], keys[i]) for i in range(6)])
        return normalized

    # 处理自车：自车相对位置为 (0, 0)
    normalized_ego = np.array([norm(0.0, 'rel_x'), norm(0.0, 'rel_y'),
                               norm(ego[2], 'vx'), norm(ego[3], 'vy'),
                               norm(ego[4], 'yaw'), norm(ego[5], 'acc')])

    # 处理8个周车
    agents = data[1]
    normalized_agents = []
    for agent in agents:
        if np.allclose(agent[:2], [0, 0]) and np.allclose(agent[2:], 0):
            # 如果是“空”代理（全0），可选择保留全0 或 归一化后的“无效”值
            # 这里我们仍按规则处理（0-ego_x 可能很大，但会被 clip 到边界）
            normalized_agents.append(process_entity(agent).tolist())
        else:
            normalized_agents.append(process_entity(agent).tolist())

    # 处理道路边缘
    road_edge = data[2]
    normalized_road_edge = process_entity(road_edge)

    # 处理参考路径
    ref_path = data[3]
    normalized_ref_path = process_entity(ref_path)

    return [
        normalized_ego.tolist(),
        normalized_agents,
        normalized_road_edge.tolist(),
        normalized_ref_path.tolist()
    ]


def average_ocp_list(data_list):
    """
    对多个 ocp_data 样本求平均，返回平均后的 data。

    输入:
        data_list: List[data], 每个 data 结构为 [ego, agents, road_edge, ref_path]
                   - ego: (6,)
                   - agents: list of 8 × (6,)
                   - road_edge: (6,)
                   - ref_path: (6,)

    输出:
        avg_data: 平均后的 data，结构相同
    """
    if not data_list:
        raise ValueError("data_list is empty")

    num_samples = len(data_list)

    # 转换为 numpy 数组便于批量操作
    egos = np.array([d[0] for d in data_list])  # (N, 6)
    agents = np.array([d[1] for d in data_list])  # (N, 8, 6)
    road_edges = np.array([d[2] for d in data_list])  # (N, 6)
    ref_paths = np.array([d[3] for d in data_list])  # (N, 6)

    # 计算平均值（沿第0维，即样本维度）
    avg_ego = np.mean(egos, axis=0).tolist()
    avg_agents = np.mean(agents, axis=0).tolist()  # (8, 6) -> list of 8 lists
    avg_road_edge = np.mean(road_edges, axis=0).tolist()
    avg_ref_path = np.mean(ref_paths, axis=0).tolist()

    return [
        avg_ego,
        avg_agents,
        avg_road_edge,
        avg_ref_path
    ]