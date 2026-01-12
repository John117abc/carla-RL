import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from src.utils import get_logger


logger = get_logger('actor_critic')

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """构建多层感知机（MLP）"""
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    """
    A2C 高斯策略网络（适用于连续动作空间）
    输出动作均值和对数标准差。
    注意：A2C 通常不强制限制动作范围（由环境处理），但也可保留缩放。
    """

    def __init__(self, observation_space, action_space, hidden_dim=256):
        super().__init__()
        obs_dim = np.prod(observation_space.shape)
        action_dim = np.prod(action_space.shape)

        self.net = mlp([obs_dim, hidden_dim, hidden_dim], activation=nn.ReLU)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 共享 log_std

        # 如果环境要求动作在 [low, high]，可保留缩放（可选）
        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)

        # 初始化
        self.mean_layer.weight.data.uniform_(-1e-3, 1e-3)
        self.mean_layer.bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean_layer(h)

        log_std = self.log_std.clamp(-20, 2)
        std = log_std.exp().expand_as(mean)

        dist = Normal(mean, std)
        action = dist.rsample()
        action = torch.clamp(action, -10, 10)

        action_tanh = torch.tanh(action)
        action_scaled = action_tanh * self.action_scale.to(obs.device) + self.action_bias.to(obs.device)

        # 数值稳定的 log_prob 计算
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (action - F.softplus(2 * action))).sum(dim=-1, keepdim=True)

        # logger.info(f"mean:{mean}")
        # logger.info(f"std:{std}")
        # logger.info(f"action:{action}")
        # logger.info(f"log_prob:{log_prob}")

        return action_scaled, log_prob, mean

    def evaluate_actions(self, obs, actions):
        """
        给定 obs 和已执行的 actions，计算 log_prob 和 entropy。
        注意：actions 是环境中的原始动作（已缩放），需反变换回 tanh 前的高斯变量。
        """
        h = self.net(obs)
        mean = self.mean_layer(h)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        # 反缩放：从 [low, high] -> [-1, 1]
        normalized_actions = (actions - self.action_bias.to(obs.device)) / self.action_scale.to(obs.device)
        # 反 tanh
        unsquashed = torch.atanh(torch.clamp(normalized_actions, -0.999999, 0.999999))

        log_prob = dist.log_prob(unsquashed).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - normalized_actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class CriticNetwork(nn.Module):
    """
    A2C 的 Critic：学习状态价值函数 V(s)
    输入：obs
    输出：标量 V(s)
    """

    def __init__(self, observation_space, hidden_dim=256):
        super().__init__()
        obs_dim = np.prod(observation_space.shape)
        self.net = mlp([obs_dim, hidden_dim, hidden_dim, 1], activation=nn.ReLU)

        # 初始化输出层（可选）
        self.net[-2].weight.data.uniform_(-3e-3, 3e-3)
        self.net[-2].bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)  # shape: [B]