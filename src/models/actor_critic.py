import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)

        # 【关键修复】最后一层初始化为极小值，让初始输出接近 0
        nn.init.uniform_(self.fc3.weight, -1e-4, 1e-4)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        # 【关键修复】去掉 Tanh 前的 *0.1，改为直接输出
        # 初始时 x 接近 0，tanh(x) 也接近 0，车会慢慢动起来
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
