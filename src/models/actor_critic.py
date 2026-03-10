import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    """
    AC 算法的 Actor 网络
    输出：[转向角，加速度]，范围 [-1, 1]
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, output_dim: int = 2):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

        # 权重初始化有助于训练稳定性
        self._init_weights()

    def _init_weights(self):
        for m in [self.l1, self.l2, self.l3]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        :param x: 状态输入 (batch, state_dim)
        :return: 动作 (batch, 2), 范围 [-1, 1]
        """
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        x = self.l3(x)  # 先线性输出
        return F.tanh(x)  # 再压缩到 [-1, 1]


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