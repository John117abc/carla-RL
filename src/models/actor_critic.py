import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.init as init


class ActorNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, output_dim: int = 2):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

        self.hidden_act = nn.LeakyReLU(0.01)

        # 【核心修复】应用正交初始化
        self._init_weights()

    def _init_weights(self):
        # 隐藏层：正交初始化，gain 适配 LeakyReLU
        init.orthogonal_(self.l1.weight, gain=init.calculate_gain('leaky_relu', 0.01))
        init.zeros_(self.l1.bias)

        init.orthogonal_(self.l2.weight, gain=init.calculate_gain('leaky_relu', 0.01))
        init.zeros_(self.l2.bias)

        # 输出层：正交初始化，但 gain 设得很小 (0.01)
        # 目的：让初始输出接近 0，避免一开始就 Tanh 饱和
        init.orthogonal_(self.l3.weight, gain=0.01)
        init.zeros_(self.l3.bias)

    def forward(self, x):
        x = self.hidden_act(self.l1(x))
        x = self.hidden_act(self.l2(x))
        return torch.tanh(self.l3(x))


class CriticNet(nn.Module):
    """
    AC 算法的 Critic 网络 (适用于 A2C/PPO 等估计 V(s) 的场景)
    注意：如果是 DDPG/TD3，需要同时输入 state 和 action
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, output_dim: int = 1):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.l1, self.l2, self.l3]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        :param x: 状态输入 (batch, state_dim)
        :return: 价值估计 (batch, 1), 范围 (-inf, +inf)
        """
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        x = self.l3(x)
        # Value 可以是任意实数，代表累积奖励，可能是很大的负数（惩罚）
        return x