# utils/checkpoint.py
import os
import torch
import numpy as np
from datetime import datetime
from utils import get_logger



def save_checkpoint(
        model,
        model_name: str,
        env_name: str,
        file_dir: str = "checkpoints",
        metrics: dict = None,
        extension: str = "pth",
        optimizer=None,
        extra_info=None
):
    from utils import get_project_root
    """
    保存模型检查点，自动按日期创建子文件夹（如 checkpoints/20251223/...）

    Args:
        model: 要保存的模型（nn.Module）
        model_name: 模型名称，如 'ppo', 'dqn'
        env_name: 环境名称，如 'highway-v0'
        file_dir: 基础保存目录，如 "checkpoints"
        metrics: 性能指标，如 {'reward': 234.5}
        extension: 文件扩展名，默认 'pth'
        optimizer: 可选，优化器状态
        epoch: 当前 epoch
        extra_info: 其他自定义信息
    """
    # 获取根目录信息
    file_dir = get_project_root() / file_dir

    # 1. 获取当前日期和时间
    today = datetime.now().strftime("%Y%m%d")  # 20251223
    time_now = datetime.now().strftime("%H%M%S")  # 145723

    # 2. 构建日期子目录路径
    dated_dir = os.path.join(file_dir, today)
    os.makedirs(dated_dir, exist_ok=True)

    # 3. 构建文件名（不含路径）
    base_name = f"{model_name}_{env_name}_{time_now}"
    if metrics:
        metric_strs = []
        for k, v in metrics.items():
            if isinstance(v, float):
                metric_strs.append(f"{k}={v:.1f}")
            else:
                metric_strs.append(f"{k}={v}")
        base_name += "_" + "_".join(metric_strs)

    file_name = f"{base_name}.{extension}"
    file_path = os.path.join(dated_dir, file_name)

    # 4. 构建 checkpoint 字典
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
    }
    if extra_info:
        clean_info = to_serializable(extra_info)
        checkpoint.update(clean_info)

    # 5. 保存
    torch.save(checkpoint, file_path)
    get_logger().info(f'训练文件保存在：{file_path}')


def load_checkpoint(model, filepath, optimizer=None, device=None):
    """
    加载模型检查点。

    Args:
        model: 要加载权重的模型（nn.Module）
        filepath: 检查点文件路径
        optimizer: 可选，用于恢复优化器状态（评估时通常不需要）
        device: 指定加载到哪个设备（如 'cpu' 或 'cuda'）

    Returns:
        checkpoint: 完整的 checkpoint 字典（包含 epoch、extra_info 等）
    """
    from utils import get_project_root

    filepath = get_project_root() / filepath

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    # 自动选择设备（如果未指定）
    map_location = device if device is not None else lambda storage, loc: storage
    checkpoint = torch.load(filepath, map_location=map_location)

    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])

    # 如果提供了 optimizer 且 checkpoint 中有优化器状态，则加载
    if optimizer and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def generate_model_filename(
        model_name: str,
        env_name: str,
        metrics: dict = None,
        extension: str = "pth"
) -> str:
    """
    生成唯一的模型保存文件名。

    Args:
        model_name (str): 模型名称，如 'ppo', 'dqn', 'sac'
        env_name (str): 环境名称，如 'highway-v0'
        metrics (dict, optional): 性能指标，如 {'reward': 234.5, 'steps': 50000}
        extension (str): 文件扩展名，默认 'pth'

    Returns:
        str: 完整路径的文件名，如 'checkpoints/ppo_highway-v0_20251223-145723_reward=234.5.pth'
    """
    # 时间戳：年月日-时分秒（便于排序）
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 构建基础名
    base_name = f"{model_name}_{env_name}_{timestamp}"

    # 添加指标（保留1位小数，避免过长）
    if metrics:
        metric_strs = []
        for k, v in metrics.items():
            if isinstance(v, float):
                metric_strs.append(f"{k}={v:.1f}")
            else:
                metric_strs.append(f"{k}={v}")
        base_name += "_" + "_".join(metric_strs)

    filename = f"{base_name}.{extension}"
    return filename


# 在 save_checkpoint 中，对 history 或 metrics 做清洗
def to_serializable(obj):
    """递归将 numpy 标量转为 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_serializable(x) for x in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # 或保留为 tensor（如果需要）
    else:
        return obj



def aaa():
    from utils import get_project_root
    return get_project_root() / 'checkpoints'

if __name__ == '__main__':
    print(aaa())