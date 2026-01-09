# src/models/cnn_encoder.py

import torch
import torch.nn as nn
import numpy as np

# 使用示例
# from src.models.cnn_encoder import CNNEncoder
# import torch
#
# # 创建编码器（输入 3 通道，输出 256 维）
# encoder = CNNEncoder(input_channels=3, output_dim=256)
#
# # 模拟一批 84x84 的 RGB 图像
# images = torch.randint(0, 256, (4, 3, 84, 84), dtype=torch.uint8)  # [B, C, H, W]
#
# # 编码
# features = encoder(images)  # shape: [4, 256]
#
# print(features.shape)  # torch.Size([4, 256])

class CNNEncoder(nn.Module):
    """
    简单的 CNN 编码器，将图像编码为固定维度的特征向量。

    默认结构（受 Nature DQN 启发）：
        Conv2d(3, 32, kernel_size=8, stride=4)
        Conv2d(32, 64, kernel_size=4, stride=2)
        Conv2d(64, 64, kernel_size=3, stride=1)
        Flatten()
        Linear(out_features=512)

    支持自定义输入通道数和输出维度。
    """

    def __init__(
            self,
            input_channels: int = 3,
            output_dim: int = 512,
            use_layer_norm: bool = True,
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm

        # 卷积层（固定结构，适合 84x84 或类似尺寸）
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # 动态计算卷积输出的展平维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 84, 84)  # 假设输入为 84x84
            conv_out = self.conv_layers(dummy_input)
            self.conv_output_size = int(np.prod(conv_out.shape[1:]))

        # 投影到目标输出维度
        self.fc = nn.Linear(self.conv_output_size, output_dim)

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像张量，shape [B, C, H, W]，建议归一化到 [0, 1] 或 [-1, 1]
        Returns:
            feature: [B, output_dim]
        """
        # 如果输入是 uint8（0-255），自动转换为 float 并归一化
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)  # flatten
        h = self.fc(h)

        if self.layer_norm is not None:
            h = self.layer_norm(h)

        return torch.relu(h)  # 可选：加 ReLU 保证非负