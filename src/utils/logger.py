# utils/logger.py

import os
import logging
from datetime import datetime
from pathlib import Path
from src.utils.common import get_project_root

file_dir = get_project_root() / 'logs'


def get_logger(
        name: str = __name__,
        log_dir: str = "logs",
        level: int = logging.INFO,
        use_console: bool = True,
        use_file: bool = True
):
    """
    获取 logger，将同一天、同一模块的日志写入同一个文件。

    日志文件路径示例：logs/20260112/train.log

    Args:
        name: logger 名称（如 'train', 'eval'）
        log_dir: 基础日志目录（相对于项目根目录）
        level: 日志级别
        use_console: 是否输出到控制台
        use_file: 是否写入文件

    Returns:
        logging.Logger 实例
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 已配置，直接返回

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- 控制台输出 ---
    if use_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # --- 文件输出：按天 + 模块 聚合 ---
    if use_file:
        # 1. 构建日志目录：logs/YYYYMMDD/
        project_root = Path(get_project_root())
        base_log_dir = project_root / log_dir
        today_str = datetime.now().strftime("%Y%m%d")
        dated_log_dir = base_log_dir / today_str
        os.makedirs(dated_log_dir, exist_ok=True)

        # 2. 文件名：模块名.log （如 train.log）
        clean_name = name.split('.')[-1] if '.' in name else name
        log_filename = f"{clean_name}.log"
        log_filepath = dated_log_dir / log_filename

        # 3. 使用 FileHandler（默认 mode='a' 追加写入）
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 可选：首次创建时提示
        if not hasattr(logger, '_file_logged'):
            print(f"[Logger] 日志追加到: {log_filepath}")
            logger._file_logged = True

    return logger