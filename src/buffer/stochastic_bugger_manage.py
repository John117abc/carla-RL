import numpy as np
import random
import heapq
import itertools
from collections import defaultdict


class PriorityBuffer:
    """基础优先级缓冲区，使用优先级队列实现"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []  # 使用堆实现优先级队列 [(priority, experience)]
        self.sum_priorities = 0.0
        self._counter = itertools.count()

    def add(self, experience, priority):
        count = next(self._counter)
        # 如果缓冲区未满，直接添加
        if len(self.buffer) < self.capacity:
            heapq.heappush(self.buffer, (priority, count,experience))
            self.sum_priorities += priority
            return True
        # 如果新样本优先级高于最低优先级样本，替换
        elif priority > self.buffer[0][0]:
            removed = heapq.heappop(self.buffer)
            self.sum_priorities -= removed[0]
            heapq.heappush(self.buffer, (priority, self.count,experience))
            self.sum_priorities += priority
            return True
        return False

    def can_replace_low_priority(self, priority):
        return len(self.buffer) == self.capacity and priority > self.buffer[0][0]

    def replace_lowest_priority(self, experience, priority):
        count = next(self._counter)
        if self.can_replace_low_priority(priority):
            removed = heapq.heappop(self.buffer)
            self.sum_priorities -= removed[0]
            heapq.heappush(self.buffer, (priority, count,experience))
            self.sum_priorities += priority
            return True
        return False

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        if batch_size == 0:
            return []

        # 从缓冲区中按优先级权重采样
        priorities = np.array([item[0] for item in self.buffer])
        probabilities = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)

        batch = []
        for idx in indices:
            batch.append(self.buffer[idx][2])

        return batch

    def temporarily_expand(self):
        self.capacity = int(self.capacity * 1.2)  # 临时增加20%容量

    def __len__(self):
        return len(self.buffer)


class SafetyCriticalBuffer(PriorityBuffer):
    """存储与安全约束违反或接近违反的样本"""

    def __init__(self, capacity=250000):
        super().__init__(capacity)
        self.collision_history = []  # 记录最近的碰撞情况

    def calculate_priority(self, experience):
        """
        计算安全关键样本的优先级
        experience结构: (state, action, reward, done, info)
        """
        state, action, reward, value, done, info = experience

        # 从info中获取安全相关指标
        collision = info.get('collision', False)
        collision_reward = reward['collision_reward']
        on_road_reward = reward['centering_reward']
        speed = info.get('speed', 0.0)

        # 基础安全优先级
        safety_priority = 0.1  # 默认最低优先级

        # 1. 碰撞或即将碰撞
        if collision or collision_reward < -0.5:  # 假设负碰撞奖励表示危险
            safety_priority = 10.0
        # 2. 轻微碰撞风险
        elif collision_reward < 0:
            safety_priority = 7.0
        # 3. 脱离道路
        elif on_road_reward < 0.5:
            safety_priority = 5.0
        # 4. 高速情况下的安全裕度
        elif speed > 25.0:  # 假设25m/s是高速阈值
            safety_priority = 3.0

        # 记录碰撞历史，用于长期安全评估
        if collision:
            self.collision_history.append(1)
        else:
            self.collision_history.append(0)

        # 保留最近100次记录
        if len(self.collision_history) > 100:
            self.collision_history.pop(0)

        # 如果近期碰撞频繁，提高所有安全相关样本的优先级
        recent_collision_rate = sum(self.collision_history) / max(1, len(self.collision_history))
        safety_priority *= (1 + recent_collision_rate * 2)

        return max(0.1, min(10.0, safety_priority))  # 限制在0.1-10.0之间


class PerformanceBuffer(PriorityBuffer):
    """存储有助于跟踪性能优化的样本"""

    def __init__(self, capacity=300000):
        super().__init__(capacity)
        self.best_performance = -float('inf')
        self.performance_history = []

    def calculate_priority(self, experience):
        """
        基于跟踪性能计算优先级
        """
        state, action, reward, value, done, info = experience

        # 从info中获取性能相关指标
        high_speed_reward = reward['high_speed_reward']
        on_road_reward = reward['centering_reward']
        right_lane_reward = reward['right_lane_reward']
        speed = info.get('speed', 0.0)

        # 计算综合性能得分
        performance_score = (
                0.4 * high_speed_reward +
                0.3 * on_road_reward +
                0.3 * right_lane_reward
        )

        # 更新最佳性能记录
        self.best_performance = max(self.best_performance, performance_score)
        self.performance_history.append(performance_score)

        # 保留最近100次记录
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

        # 计算近期平均性能
        recent_avg_performance = sum(self.performance_history) / max(1, len(self.performance_history))

        # 性能好的样本优先级高
        if performance_score > recent_avg_performance:
            priority = 5.0 + (performance_score - recent_avg_performance) * 10.0
        else:
            priority = 1.0 + performance_score * 2.0

        # 高速且保持在道路上、正确车道的样本获得额外奖励
        if speed > 20.0 and on_road_reward > 0.9 and right_lane_reward > 0.9:
            priority *= 1.5

        return max(0.1, min(10.0, priority))


class DiversityBuffer(PriorityBuffer):
    """存储多样化场景的样本，确保策略泛化能力"""

    def __init__(self, capacity=250000):
        super().__init__(capacity)
        # 使用离散化方法创建场景指纹
        self.scenario_counts = defaultdict(int)  # 记录不同场景的出现频率
        self.total_samples = 0

    def calculate_priority(self, experience):
        """
        基于场景多样性和稀有性计算优先级
        """
        state, action, reward, value, done, info = experience

        # 创建场景指纹
        scenario_fingerprint = self._create_scenario_fingerprint(info,reward)

        # 更新场景计数
        self.scenario_counts[scenario_fingerprint] += 1
        self.total_samples += 1

        # 计算场景出现频率
        scenario_freq = self.scenario_counts[scenario_fingerprint] / max(1, self.total_samples)

        # 稀有场景获得更高优先级
        base_priority = max(0.1, 1.0 / (scenario_freq * 50 + 0.01))

        # 考虑动作多样性 - 不常见动作组合获得更高优先级
        action_priority = self._evaluate_action_diversity(info.get('action', [0, 0]))

        # 考虑状态多样性 - 边界状态获得更高优先级
        state_priority = self._evaluate_state_diversity(state)

        return max(0.1, min(10.0, base_priority + action_priority + state_priority))

    def _create_scenario_fingerprint(self, info,reward):
        """创建场景指纹，用于识别不同场景"""
        speed = info.get('speed', 0.0)
        on_road = reward['centering_reward'] > 0.5
        right_lane = reward['right_lane_reward'] > 0.5

        # 速度级别
        if speed < 10.0:
            speed_level = "low"
        elif speed < 20.0:
            speed_level = "medium"
        else:
            speed_level = "high"

        # 道路位置
        position_type = "on_road" if on_road else "off_road"
        lane_type = "right" if right_lane else "other"

        # 碰撞风险
        collision_risk = "high" if info.get('collision', False) or reward['collision_reward'] < -0.5 else "low"

        return f"{speed_level}_{position_type}_{lane_type}_{collision_risk}"

    def _evaluate_action_diversity(self, action):
        """评估动作多样性"""
        # 假设action是[steering, acceleration]
        steering, acceleration = action[0], action[1]

        # 极端动作（急转弯、急加速/刹车）通常更稀有
        extreme_action_score = 0
        if abs(steering) > 0.7:
            extreme_action_score += 1.0
        if acceleration < -0.5 or acceleration > 0.7:
            extreme_action_score += 1.0

        return extreme_action_score

    def _evaluate_state_diversity(self, state):
        """评估状态多样性 - 简化版本"""
        # 在highway-env中，state通常是车辆观察向量
        # 我们假设state包含周围车辆的相对位置和速度
        if state is None or len(state) < 5:
            return 0.5

        # 计算状态向量的范数，极端值表示特殊状态
        state_norm = np.linalg.norm(state)
        if state_norm > 5.0 or state_norm < 1.0:
            return 1.0

        return 0.5


class CurriculumBuffer(PriorityBuffer):
    """实现课程学习的缓冲区，从简单到复杂"""

    def __init__(self, capacity=200000):
        super().__init__(capacity)
        self.min_difficulty = 0.0
        self.max_difficulty = 10.0
        self.current_difficulty_threshold = 2.0  # 初始难度阈值
        self.difficulty_history = []

    def calculate_priority(self, experience):
        """
        基于当前学习阶段和样本难度计算优先级
        """
        state, action, reward, value, done, info = experience

        # 评估样本难度
        difficulty = self._assess_difficulty(info,reward)

        # 根据当前学习阶段调整优先级
        if difficulty <= self.current_difficulty_threshold:
            # 当前阶段可处理的难度：高优先级
            priority = max(5.0, 8.0 - difficulty)
        else:
            # 更高难度的样本：保留但优先级较低
            priority = max(0.1, 2.0 / (difficulty - self.current_difficulty_threshold + 0.1))

        # 逐渐提高难度阈值
        self._update_difficulty_threshold(difficulty)

        return max(0.1, min(10.0, priority))

    def _assess_difficulty(self, info,reward):
        """评估样本难度"""
        difficulty = 0.0

        # 1. 速度 - 高速更难控制
        speed = info.get('speed', 0.0)
        difficulty += min(speed / 10.0, 3.0)  # 假设速度单位为m/s

        # 2. 碰撞风险
        collision_reward = reward['collision_reward']
        if collision_reward < -0.5:
            difficulty += 3.0
        elif collision_reward < 0:
            difficulty += 1.5

        # 3. 道路位置 - 偏离道路增加难度
        on_road_reward = reward['centering_reward']
        if on_road_reward < 0.5:
            difficulty += 2.0

        # 4. 车道选择 - 不在右车道增加难度
        right_lane_reward = reward['right_lane_reward']
        if right_lane_reward < 0.5:
            difficulty += 1.0

        return min(difficulty, self.max_difficulty)

    def _update_difficulty_threshold(self, difficulty):
        """根据训练进展更新难度阈值"""
        self.difficulty_history.append(difficulty)

        # 保留最近1000条记录
        if len(self.difficulty_history) > 1000:
            self.difficulty_history.pop(0)

        # 如果近期样本平均难度超过当前阈值，提高阈值
        if len(self.difficulty_history) > 100:
            avg_difficulty = sum(self.difficulty_history[-100:]) / 100.0
            if avg_difficulty > self.current_difficulty_threshold:
                self.current_difficulty_threshold = min(
                    self.max_difficulty,
                    self.current_difficulty_threshold + 0.1
                )


class StochasticBufferManager:
    def __init__(self,
                 min_start_train: int = 256,
                 total_capacity: int = 1000000):
        """
        随机取样缓冲区
        :param min_start_train: 最小启动训练样本数
        :param total_capacity: 缓冲区容量
        """
        # 4个专用Buffers
        self.buffers = [
            SafetyCriticalBuffer(capacity=total_capacity*0.25),  # 安全关键样本
            PerformanceBuffer(capacity=total_capacity*0.3),  # 性能优化样本
            DiversityBuffer(capacity=total_capacity*0.25),  # 多样性样本
            CurriculumBuffer(capacity=total_capacity*0.2)  # 课程学习样本
        ]
        self.total_capacity = total_capacity
        self.min_start_train = min_start_train  # 最小训练启动样本量
        self.gep_penalty_factor = 1.0  # GEP惩罚因子初始值

    def should_start_training(self):
        """判断是否达到训练启动条件"""
        total_samples = sum(len(buf) for buf in self.buffers)
        return total_samples >= self.min_start_train

    def handle_new_experience(self, experience):
        """处理新生成的经验数据"""
        # 1. 分类到合适的Buffer
        buffer_idx = self._classify_experience(experience)

        # 2. 计算GEP-aware优先级
        priority = self._calculate_gep_priority(experience, buffer_idx)

        # 3. 添加到对应Buffer
        added = self.buffers[buffer_idx].add(experience, priority)

        # 4. 如果添加失败(空间不足)，尝试其他策略
        if not added:
            self._handle_buffer_full(buffer_idx, experience, priority)

    def _classify_experience(self, experience):
        """根据经验特性分类到最合适的缓冲区"""
        state, action, reward, value, done, info = experience

        # 优先考虑安全关键样本
        if self._is_safety_critical(experience):
            return 0  # SafetyCriticalBuffer

        # 检查是否为良好性能样本
        if self._is_high_performance(experience):
            return 1  # PerformanceBuffer

        # 检查是否为稀有或多样化场景
        if self._is_diverse_scenario(experience):
            return 2  # DiversityBuffer

        # 默认归入课程学习缓冲区
        return 3  # CurriculumBuffer

    def _is_safety_critical(self, experience):
        """判断是否为安全关键样本"""
        _, _, reward, _, _, info = experience
        collision = info.get('collision', False)
        collision_reward = reward['collision_reward']
        on_road_reward = reward['centering_reward']

        # 直接安全违反
        if collision or collision_reward < -0.5:
            return True

        # 严重脱离道路
        if on_road_reward < 0.3:
            return True

        return False

    def _is_high_performance(self, experience):
        """判断是否为高性能样本"""
        _, _, reward, _, _, info = experience
        high_speed_reward = reward['high_speed_reward']
        on_road_reward = reward['centering_reward']
        right_lane_reward = reward['right_lane_reward']

        # 高性能标准：高速、保持在道路上、在正确车道
        return (high_speed_reward > 0.7 and
                on_road_reward > 0.9 and
                right_lane_reward > 0.9)

    def _is_diverse_scenario(self, experience):
        """判断是否为多样化场景"""
        _, _, reward, _, _, info = experience

        # 创建场景指纹
        scenario_fingerprint = self._create_scenario_fingerprint(info,reward)

        # 检查是否为稀有场景
        for i, buf in enumerate(self.buffers):
            if isinstance(buf, DiversityBuffer):
                # 如果该场景出现次数少于总样本的1%，认为是稀有场景
                count = buf.scenario_counts.get(scenario_fingerprint, 0)
                if count < max(10, buf.total_samples * 0.01):
                    return True
        return False

    def _create_scenario_fingerprint(self, info,reward):
        """创建场景指纹，用于识别不同场景 - 简化版"""
        speed = info.get('speed', 0.0)
        on_road = reward['centering_reward'] > 0.5
        right_lane = reward['right_lane_reward'] > 0.5
        collision = info.get('collision', False)

        # 速度级别
        if speed < 10.0:
            speed_level = "low"
        elif speed < 20.0:
            speed_level = "medium"
        else:
            speed_level = "high"

        return f"{speed_level}_{'on' if on_road else 'off'}_{'right' if right_lane else 'other'}_{'crash' if collision else 'safe'}"

    def _calculate_gep_priority(self, experience, buffer_idx):
        """计算GEP-aware优先级，适配highway-env"""
        state, action, reward, value, done, info = experience

        base_priority = 1.0

        # 安全关键缓冲区：基于约束违反程度
        if buffer_idx == 0:  # SafetyCriticalBuffer
            collision_reward = reward['collision_reward']
            on_road_reward = reward['centering_reward']

            # 计算安全违反程度
            safety_violation = 0.0
            if collision_reward < 0:
                safety_violation += -collision_reward
            if on_road_reward < 1.0:
                safety_violation += (1.0 - on_road_reward) * 0.5

            return base_priority + self.gep_penalty_factor * safety_violation * 5.0

        # 性能缓冲区：基于综合奖励
        elif buffer_idx == 1:  # PerformanceBuffer
            high_speed_reward = reward['high_speed_reward']
            on_road_reward = reward['centering_reward']
            right_lane_reward = reward['right_lane_reward']

            # 综合性能得分
            performance_score = (
                    0.5 * high_speed_reward +
                    0.3 * on_road_reward +
                    0.2 * right_lane_reward
            )
            return base_priority + performance_score * 5.0

        # 多样性缓冲区：固定较高优先级
        elif buffer_idx == 2:  # DiversityBuffer
            return base_priority + 3.0

        # 课程学习缓冲区：基于难度适应性
        else:  # CurriculumBuffer
            difficulty = self.buffers[3]._assess_difficulty(info,reward)
            if difficulty <= self.buffers[3].current_difficulty_threshold:
                return base_priority + 4.0  # 当前可处理的难度
            else:
                return base_priority + 1.0  # 暂时难以处理的样本

    def _handle_buffer_full(self, buffer_idx, experience, priority):
        """处理缓冲区满载情况"""
        # 策略1: 降级低优先级样本
        if self.buffers[buffer_idx].can_replace_low_priority(priority):
            self.buffers[buffer_idx].replace_lowest_priority(experience, priority)
            return

        # 策略2: 尝试其他Buffer
        for i, buf in enumerate(self.buffers):
            if i != buffer_idx and buf.can_replace_low_priority(priority):
                buf.replace_lowest_priority(experience, priority)
                return

        # 策略3: 动态调整GEP惩罚因子（安全关键样本特别处理）
        if buffer_idx == 0:  # SafetyCriticalBuffer
            self.gep_penalty_factor *= 1.01  # 逐渐增加惩罚

        # 策略4: 动态扩展(临时增加容量)
        if self._can_temporarily_expand():
            self.buffers[buffer_idx].temporarily_expand()
            self.buffers[buffer_idx].add(experience, priority)

    def _can_temporarily_expand(self):
        """检查是否可以临时扩展缓冲区"""
        total_used = sum(len(buf) for buf in self.buffers)
        return total_used < self.total_capacity * 1.2  # 允许临时超过总容量20%

    def sample_batch(self, batch_size):
        """从所有缓冲区采样批次"""
        # 按比例从各缓冲区采样
        buffer_sizes = [len(buf) for buf in self.buffers]
        total_size = sum(buffer_sizes)

        if total_size == 0:
            return []

        # todo 由于缺少其它样本的判断，先取部分
        weights = [0.7, 0.0, 0.3, 0.0]  # Safety, Performance, Diversity, Curriculum
        samples = []

        # 确保每个缓冲区有足够样本
        for i, buf in enumerate(self.buffers):
            if len(buf) > 0:
                buf_batch_size = max(1, int(batch_size * weights[i]))
                buf_samples = buf.sample(buf_batch_size)
                samples.extend(buf_samples)

        # 确保返回正确批次大小
        if len(samples) > batch_size:
            return random.sample(samples, batch_size)
        return samples