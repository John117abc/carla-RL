from torch import nn
import torch.nn.functional as F
class ActorNet(nn.Module):
    """
    ac算法的actor网络
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
    ac算法的critic网络
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
