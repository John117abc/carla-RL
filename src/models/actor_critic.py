import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)

        # 【修复】正确的初始化方式
        nn.init.uniform_(self.fc3.weight, -1e-3, 1e-3)
        # 直接给bias赋值，而不是用constant_
        with torch.no_grad():
            self.fc3.bias[0] = 0.5  # 加速度通道初始偏置0.5
            self.fc3.bias[1] = 0.0  # 转向通道初始偏置0

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        x = torch.tanh(x)
        return x


class CriticNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, output_dim: int = 1):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        x = self.l3(x)
        return x