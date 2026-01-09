# src/utils/transforms.py

import numpy as np
import torch
from PIL import Image
from typing import Any, Callable, List, Optional, Union

# 使用示例
# # 在 env.reset() 或 env.step() 中
# from src.utils.transforms import carla_transform
#
# class CarlaEnv(gym.Env):
#     def __init__(self):
#         self.transform = carla_transform(
#             resize_size=(84, 84),
#             crop_region=(400, 900),
#             grayscale=False
#         )
#         self.observation_space = gym.spaces.Box(0, 1, (3, 84, 84), dtype=np.float32)
#
#     def _process_image(self, raw_img: np.ndarray) -> torch.Tensor:
#         # raw_img: (1080, 1920, 3) uint8
#         obs = self.transform(raw_img)  # [3, 84, 84]
#         return obs.numpy()  # 或保持为 tensor，根据你的 agent 设计

class Compose:
    """组合多个变换"""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img: Any) -> Any:
        for t in self.transforms:
            img = t(img)
        return img


class Resize:
    """调整图像尺寸"""

    def __init__(self, size: Union[int, tuple]):
        """
        Args:
            size: (height, width) 或 int（正方形）
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        img: (H, W, C) uint8 in [0, 255]
        returns: (H', W', C) uint8
        """
        h, w = self.size
        pil_img = Image.fromarray(img)
        resized = pil_img.resize((w, h), Image.BILINEAR)  # 注意：PIL 是 (width, height)
        return np.array(resized)


class CenterCrop:
    """中心裁剪"""

    def __init__(self, crop_size: Union[int, tuple]):
        if isinstance(crop_size, int):
            self.crop_h = self.crop_w = crop_size
        else:
            self.crop_h, self.crop_w = crop_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        img: (H, W, C)
        """
        h, w = img.shape[:2]
        top = (h - self.crop_h) // 2
        left = (w - self.crop_w) // 2
        return img[top:top + self.crop_h, left:left + self.crop_w]


class CropRegion:
    """自定义区域裁剪（适合 CARLA：保留道路区域）"""

    def __init__(self, top: int, bottom: int, left: int = 0, right: int = None):
        """
        Args:
            top: 裁剪起始行
            bottom: 裁剪结束行
            left/right: 列范围（默认全宽）
        """
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        right = self.right if self.right is not None else w
        return img[self.top:self.bottom, self.left:right]


class ToTensor:
    """将 HWC uint8 [0,255] 转为 CHW float [0,1]"""

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        elif img.ndim == 2:
            img = img[None, :, :]  # HW -> CHW (C=1)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
        return torch.as_tensor(img, dtype=torch.float32).div_(255.0)


class Normalize:
    """归一化到 [-1, 1] 或自定义均值/标准差"""

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


class Grayscale:
    """转灰度图（保留通道维度）"""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.shape[-1] == 1:
            return img
        # 使用标准 RGB 权重
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        return gray.astype(np.uint8)[..., None]  # (H, W, 1)


# ========================
# 预设常用组合
# ========================

def carla_transform(
        resize_size: tuple = (84, 84),
        crop_region: Optional[tuple] = (400, 900),  # (top, bottom)
        grayscale: bool = False,
        normalize: bool = False,
) -> Compose:
    """
    CARLA 图像预处理流水线。

    示例输入: (1080, 1920, 3) uint8

    Args:
        resize_size: 最终输出尺寸 (H, W)
        crop_region: (top, bottom)，如 (400, 900) 表示只保留第 400～900 行
        grayscale: 是否转灰度
        normalize: 是否归一化到 [-1,1]（用于某些算法）
    """
    transforms = []

    # 1. 裁剪（可选）
    if crop_region is not None:
        top, bottom = crop_region
        transforms.append(CropRegion(top=top, bottom=bottom))

    # 2. 转灰度（可选）
    if grayscale:
        transforms.append(Grayscale())

    # 3. 缩放
    transforms.append(Resize(resize_size))

    # 4. 转 Tensor [0,1]
    transforms.append(ToTensor())

    # 5. 归一化（可选）
    if normalize:
        # ImageNet 均值/标准差（若用预训练模型）或简单 [-1,1]
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5] if not grayscale else [0.5],
                                    std=[0.5, 0.5, 0.5] if not grayscale else [0.5]))

    return Compose(transforms)


# ========================
# 使用示例
# ========================
if __name__ == "__main__":
    # 模拟 CARLA 原始图像
    raw_img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

    # 创建变换
    transform = carla_transform(
        resize_size=(84, 84),
        crop_region=(400, 900),
        grayscale=False,
        normalize=False
    )

    # 应用变换
    processed = transform(raw_img)  # torch.Tensor of shape [3, 84, 84], dtype=float32, range [0,1]

    print("Output shape:", processed.shape)
    print("Value range:", processed.min().item(), "to", processed.max().item())