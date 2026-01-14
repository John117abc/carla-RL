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