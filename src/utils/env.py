import random
import numpy as np
import torch
import warnings
from pathlib import Path

def setup_code_environment(config):
    """
    设置训练环境，确保可复现性和稳定性。

    Args:
        config: 配置对象，需包含以下字段（示例）：
            - seed: int
            - device: str ("cpu" or "cuda")
            - cudnn_deterministic: bool (可选)
            - cudnn_benchmark: bool (可选)
            - output_dir: str (可选，用于创建目录)
    """
    # 1. 设置随机种子
    seed = getattr(config, 'seed', 22)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 所有 GPU
    np.random.seed(seed)
    random.seed(seed)

    # 2. CuDNN 设置
    cudnn_deterministic = getattr(config, 'cudnn_deterministic', True)
    cudnn_benchmark = getattr(config, 'cudnn_benchmark', False)

    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark

    if cudnn_deterministic:
        print("⚠️  CUDNN deterministic enabled — may slow down training.")
    if cudnn_benchmark:
        print("⚡ CUDNN benchmark enabled — faster but non-deterministic.")

    # 3. 设备检查
    device_str = getattr(config, 'device', 'cuda')
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU.")
        config.device = 'cpu'
    else:
        config.device = device_str

    # 4. 创建输出目录（如果配置中有）
    output_dir = getattr(config, 'output_dir', None)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")

    # 5. 抑制烦人的警告
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning)

    print(f"✅ Environment set up. Seed={seed}, Device={config.device}")