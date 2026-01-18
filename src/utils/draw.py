import matplotlib as mpl
mpl.rcParams.update({
    'font.sans-serif': ['Noto Sans CJK JP'],  # 或 WenQuanYi Zen Hei + 避免 \u2212
    'axes.unicode_minus': False,
    'mathtext.fontset': 'stix'
})
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, history=None):
        """
        初始化 Plotter 对象。

        :param history: dict，包含训练历史数据，默认为None。期望的键包括'episode', 'total_reward', 'avg_return', 'avg_loss'。
        """
        self.history = history if history is not None else {'episode': [], 'reward': [], 'loss_critic': []}

    def load_history(self, history):
        """
        加载训练历史数据。

        :param history: dict，包含训练历史数据。
        """
        self.history = history

    def plot_training_metrics(self):
        """绘制训练过程中各个指标的变化情况"""
        # 绘制总奖励随轮次变化的图表
        plt.figure(figsize=(10, 6))
        actor_loss = [s['actor_loss'] for s in self.history[0]]
        critic_loss = [s['critic_loss'] for s in self.history[0]]
        global_step = [s['global_step'] for s in self.history[0]]
        plt.plot(global_step, actor_loss, label='actor 损失', color='green')
        plt.xlabel('step')
        plt.ylabel('actor_loss')
        plt.title('cator损失表')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(global_step, critic_loss, label='critic 损失', color='blue')
        plt.xlabel('step')
        plt.ylabel('critic_loss')
        plt.title('critic损失表')
        plt.grid(True)
        plt.legend()
        plt.show()

        # # 绘制步骤数和平均损失随轮次变化的双轴图表
        # fig, ax1 = plt.subplots(figsize=(10, 6))
        #
        # color = 'tab:blue'
        # ax1.set_xlabel('Episode')
        # ax1.set_ylabel('Avg_return', color=color)
        # ax1.plot(self.history['episode'], self.history['loss_critic'], color=color, label='returns')
        # ax1.tick_params(axis='y', labelcolor=color)
        #
        # # ax2 = ax1.twinx()
        # # color = 'tab:red'
        # # ax2.set_ylabel('Avg Loss', color=color)
        # # ax2.plot(self.history['episode'], self.history['avg_loss'], color=color, label='Loss')
        # # ax2.tick_params(axis='y', labelcolor=color)
        #
        # plt.title('Returns and Loss over Episodes')
        # fig.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        # plt.show()


# 示例用法：
if __name__ == "__main__":
    # 假设这是从checkpoint中加载得到的历史记录
    training_history = {
        'episode': [1, 2, 3, 4, 5],
        'total_reward': [100, 150, 200, 250, 300],
        'avg_return': [50, 60, 70, 80, 90],
        'avg_loss': [0.3, 0.25, 0.2, 0.15, 0.1]
    }

    plotter = Plotter()
    plotter.load_history(training_history)
    plotter.plot_training_metrics()