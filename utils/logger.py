# utils/logger.py
import logging
import os

def get_logger(name=__name__, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:  # 避免重复添加 handler
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 可选：输出到文件
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger