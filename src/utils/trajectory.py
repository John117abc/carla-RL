import numpy as np

def resample_path_equal_distance(path_xy: np.ndarray, spacing: float) -> np.ndarray:
    """
    对 2D 路径进行等距重采样。
    :param path_xy: [N, 2] 原始路径点
    :param spacing: 期望点间距 (米)
    :return: [M, 2] 重采样后的路径
    """
    if len(path_xy) < 2:
        return path_xy
    # 计算每段长度
    segs = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cum_dist = np.insert(np.cumsum(segs), 0, 0)
    total_len = cum_dist[-1]
    if total_len < spacing:
        return path_xy

    sample_dists = np.arange(0, total_len, spacing)
    # 使用线性插值
    resampled_x = np.interp(sample_dists, cum_dist, path_xy[:, 0])
    resampled_y = np.interp(sample_dists, cum_dist, path_xy[:, 1])
    return np.stack([resampled_x, resampled_y], axis=1)