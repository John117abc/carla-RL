from torch import nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """
    AC/A2C算法的神经网络，使用共享主干参数，来获得policy和value
    """
    def __init__(self, state_dim, hidden_dim, action_dim=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)      # 均值
        self.log_std_head = nn.Linear(hidden_dim, action_dim) # 标准差
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        value = self.value_head(h)
        return mu, log_std, value


class ActorNet(nn.Module):
    """
    ac/a2c算法的actor网络
    """
    def __init__(self,state_dim : int,hidden_dim: int = 256,output_dim: int = 2):
        super().__init__()
        self.l1 = nn.Linear(state_dim,hidden_dim)
        self.l2 = nn.Linear(hidden_dim,hidden_dim)
        self.l3 = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        """
        向前传播
        :param x: 输入参数
        :return: 转向角，加速度 (映射到-1，1)
        """
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        x = self.l3(x)
        return F.tanh(x)

class CriticNet(nn.Module):
    """
    ac/a2c算法的critic网络
    """
    def __init__(self,state_dim : int,hidden_dim: int = 256,output_dim: int = 1):
        super().__init__()
        self.l1 = nn.Linear(state_dim,hidden_dim)
        self.l2 = nn.Linear(hidden_dim,hidden_dim)
        self.l3 = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        """
        向前传播
        :param x: 输入参数
        :return: 评估值
        """
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        return F.elu(self.l3(x))
