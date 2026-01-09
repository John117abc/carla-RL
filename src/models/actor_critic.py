# src/models/actor_critic.py

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, TransformedDistribution

# 在 src/models/actor_critic.py 的 ActorNetwork 类中添加：

def evaluate_actions(self, obs, actions):
    """
    给定 obs 和 actions，返回 log_prob 和 entropy。
    用于训练时重计算概率。
    """
    h = self.net(obs)
    mean = self.mean_layer(h)
    log_std = self.log_std_layer(h)
    log_std = torch.clamp(log_std, min=-20, max=2)
    std = log_std.exp()

    normal = Normal(mean, std)
    # 反 tanh 得到原始高斯变量
    unsquashed_actions = torch.atanh(torch.clamp(actions, -0.999999, 0.999999))
    # 缩放回 [-1,1] 再反变换
    normalized_actions = (actions - self.action_bias.to(obs.device)) / self.action_scale.to(obs.device)
    unsquashed = torch.atanh(torch.clamp(normalized_actions, -0.999999, 0.999999))

    log_prob = normal.log_prob(unsquashed).sum(dim=-1, keepdim=True)
    log_prob -= torch.log(1 - normalized_actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
    entropy = normal.entropy().sum(dim=-1, keepdim=True)

    return log_prob, entropy, mean


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """构建多层感知机（MLP）"""
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    """
    高斯策略网络，输出动作均值和对数标准差。
    使用 Tanh 变换将动作限制在 [-1, 1]，再缩放到 action_space 范围。
    """

    def __init__(self, observation_space, action_space, hidden_dim=256):
        super().__init__()
        obs_dim = np.prod(observation_space.shape)
        action_dim = np.prod(action_space.shape)

        self.net = mlp([obs_dim, hidden_dim, hidden_dim], activation=nn.ReLU)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # 保存动作空间边界（用于反 tanh 缩放）
        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)

        # 初始化最后一层权重较小，避免训练初期过大动作
        self.mean_layer.weight.data.uniform_(-1e-3, 1e-3)
        self.mean_layer.bias.data.uniform_(-1e-3, 1e-3)
        self.log_std_layer.weight.data.uniform_(-1e-3, 1e-3)
        self.log_std_layer.bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, min=-20, max=2)  # 防止数值不稳定
        std = log_std.exp()

        normal = Normal(mean, std)
        # 使用 reparameterization trick
        x_t = normal.rsample()  # 从 N(mean, std) 采样
        y_t = torch.tanh(x_t)   # 压缩到 (-1, 1)

        # 计算 log_prob（考虑 tanh 的雅可比行列式）
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)  # tanh 变换的 log det
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # 缩放到实际动作范围
        action = y_t * self.action_scale.to(obs.device) + self.action_bias.to(obs.device)

        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class CriticNetwork(nn.Module):
    """
    双 Q 网络（Twin Q），用于减少 Q 值过估计。
    输出两个独立的 Q 值。
    """

    def __init__(self, observation_space, action_space, hidden_dim=256):
        super().__init__()
        obs_dim = np.prod(observation_space.shape)
        action_dim = np.prod(action_space.shape)

        # Q1
        self.q1_net = mlp([obs_dim + action_dim, hidden_dim, hidden_dim, 1], activation=nn.ReLU)
        # Q2
        self.q2_net = mlp([obs_dim + action_dim, hidden_dim, hidden_dim, 1], activation=nn.ReLU)

        # 初始化输出层
        for net in [self.q1_net, self.q2_net]:
            net[-2].weight.data.uniform_(-3e-3, 3e-3)
            net[-2].bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, action):
        xu = torch.cat([obs, action], dim=-1)
        q1 = self.q1_net(xu).squeeze(-1)
        q2 = self.q2_net(xu).squeeze(-1)
        return q1, q2