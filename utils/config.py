# utils/configs.py
import os
from omegaconf import OmegaConf

def load_config(config_path: str):
    from utils import get_project_root

    # 目录拼接
    config_path = get_project_root() / config_path
    """加载 YAML 配置文件"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)
    return config

def load_config_json(config_path: str):
    from utils import get_project_root
    # 目录拼接
    config_path = get_project_root() / config_path
    """加载 YAML 配置文件"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config)


# 可选：支持命令行覆盖
def merge_cli_args(config, cli_args):
    # 将 argparse 的命名（如 --lr）转为 OmegaConf 支持的 dot 表示（如 train.lr）
    # 或直接使用 OmegaConf.from_cli()
    pass