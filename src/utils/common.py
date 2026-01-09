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