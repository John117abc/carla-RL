import numpy as np
import random
import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass,field
from typing import List, Any

@dataclass
class Trajectory:
    initial_state: Any          # 包含 path info 的初始状态（用于 Critic 输入）
    states: List[np.ndarray]           # [s0, s1, ..., sT]
    actions: List[np.ndarray]          # [a0, a1, ..., a_{T-1}]
    rewards: List[dict]         # 每一步的 reward dict
    infos: List[dict]           # 每一步的 info
    total_cost: float           # 整条轨迹的 tracking + control cost
    total_constraint: float     # 整条轨迹的约束违反量（ge_car + ge_road）
    path_id: int                # 路径 ID（用于 diversity）
    horizon: int                # T
    dones: List[bool]           # 终止
    next_states: List[np.ndarray]       # 当前状态的下一个状态
    log_probs: List[np.ndarray]        # 选择动作的log概率

    dones: List[bool] = field(default_factory=list)
    next_states: List[np.ndarray] = field(default_factory=list)
    total_cost: float = 9999
    total_constraint: float = 9999
    path_id: int = -1
    horizon: int = -1
    log_probs: List[np.ndarray] = field(default_factory=list)

class TrajectoryPriorityBuffer:
    """基于轨迹的优先级缓冲区"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []  # [(priority, count, trajectory)]
        self.sum_priorities = 0.0
        self._counter = itertools.count()

    def add(self, trajectory: Trajectory, priority: float) -> bool:
        count = next(self._counter)
        if len(self.buffer) < self.capacity:
            heapq.heappush(self.buffer, (priority, count, trajectory))
            self.sum_priorities += priority
            return True
        elif priority > self.buffer[0][0]:
            removed = heapq.heappop(self.buffer)
            self.sum_priorities -= removed[0]
            heapq.heappush(self.buffer, (priority, count, trajectory))
            self.sum_priorities += priority
            return True
        return False

    def can_replace_low_priority(self, priority: float) -> bool:
        return len(self.buffer) == self.capacity and priority > self.buffer[0][0]

    def replace_lowest_priority(self, trajectory: Trajectory, priority: float) -> bool:
        count = next(self._counter)
        if self.can_replace_low_priority(priority):
            removed = heapq.heappop(self.buffer)
            self.sum_priorities -= removed[0]
            heapq.heappush(self.buffer, (priority, count, trajectory))
            self.sum_priorities += priority
            return True
        return False

    def sample(self, batch_size: int) -> List[Trajectory]:
        if len(self.buffer) == 0:
            return []
        n = min(batch_size, len(self.buffer))
        priorities = np.array([item[0] for item in self.buffer])
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), n, p=probs, replace=False)
        return [self.buffer[i][2] for i in indices]

    def temporarily_expand(self):
        self.capacity = int(self.capacity * 1.2)

    def __len__(self):
        return len(self.buffer)


class SafetyCriticalTrajectoryBuffer(TrajectoryPriorityBuffer):
    def calculate_priority(self, traj: Trajectory) -> float:
        # 整条轨迹的最大碰撞风险
        max_collision_risk = max(
            info.get('collision', False) or (r['collision_reward'] < -0.5)
            for r, info in zip(traj.rewards, traj.infos)
        )

        total_constraint = traj.total_constraint
        avg_speed = np.mean([info.get('speed', 0.0) for info in traj.infos])

        priority = 0.1
        if max_collision_risk:
            priority = 10.0
        elif total_constraint > 0.5:
            priority = 7.0
        elif avg_speed > 25.0:
            priority = 3.0

        return max(0.1, min(10.0, priority))


class PerformanceTrajectoryBuffer(TrajectoryPriorityBuffer):
    def __init__(self, capacity=3000):
        super().__init__(capacity)
        self.best_performance = float('inf')  # 越小越好（cost）
        self.performance_history = []

    def calculate_priority(self, traj: Trajectory) -> float:
        # 性能指标：总 cost 越低越好，约束越小越好
        total_cost = traj.total_cost
        total_constraint = traj.total_constraint

        # 归一化性能得分（可选：加权组合）
        performance_score = total_cost + 5.0 * total_constraint  # 惩罚约束违反

        # 更新历史
        self.best_performance = min(self.best_performance, performance_score)
        self.performance_history.append(performance_score)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

        recent_avg = np.mean(self.performance_history[-20:]) if self.performance_history else performance_score

        # 性能越好（score 越低），优先级越高
        if performance_score < recent_avg:
            priority = 8.0 - (recent_avg - performance_score) * 2.0
        else:
            priority = 3.0 - (performance_score - recent_avg) * 0.5

        # 额外奖励：全程高速 + 无碰撞 + 在右车道
        speeds = [info.get('speed', 0.0) for info in traj.infos]
        on_road_rewards = [r['centering_reward'] for r in traj.rewards]
        right_lane_rewards = [r['right_lane_reward'] for r in traj.rewards]
        collisions = [info.get('collision', False) for info in traj.infos]

        if (np.mean(speeds) > 20.0 and
            np.mean(on_road_rewards) > 0.9 and
            np.mean(right_lane_rewards) > 0.9 and
            not any(collisions)):
            priority *= 1.3

        return max(0.1, min(10.0, priority))


class DiversityTrajectoryBuffer(TrajectoryPriorityBuffer):
    def __init__(self, capacity=2500):
        super().__init__(capacity)
        self.path_counts = defaultdict(int)
        self.total_trajectories = 0
        self.scenario_counts = defaultdict(int)

    def calculate_priority(self, traj: Trajectory) -> float:
        path_id = traj.path_id
        self.path_counts[path_id] += 1
        self.total_trajectories += 1

        # 路径稀有性：出现越少，优先级越高
        path_freq = self.path_counts[path_id] / max(1, self.total_trajectories)
        path_priority = max(0.1, 1.0 / (path_freq * 20 + 0.01))

        # 场景指纹
        scenario_fp = self._create_scenario_fingerprint(traj)
        self.scenario_counts[scenario_fp] += 1
        scenario_freq = self.scenario_counts[scenario_fp] / max(1, self.total_trajectories)
        scenario_priority = max(0.1, 1.0 / (scenario_freq * 30 + 0.01))

        # 动作/状态极端性
        action_diversity = self._evaluate_action_diversity(traj.actions)
        state_diversity = self._evaluate_state_diversity(traj.states)

        total_priority = path_priority + scenario_priority + action_diversity + state_diversity
        return max(0.1, min(10.0, total_priority))

    def _create_scenario_fingerprint(self, traj: Trajectory) -> str:
        avg_speed = np.mean([info.get('speed', 0.0) for info in traj.infos])
        avg_on_road = np.mean([r['centering_reward'] for r in traj.rewards])
        avg_right_lane = np.mean([r['right_lane_reward'] for r in traj.rewards])
        has_collision = any(info.get('collision', False) for info in traj.infos)

        speed_level = "high" if avg_speed > 20 else ("medium" if avg_speed > 10 else "low")
        road_status = "on" if avg_on_road > 0.7 else "off"
        lane_status = "right" if avg_right_lane > 0.7 else "other"
        crash = "crash" if has_collision else "safe"

        return f"{speed_level}_{road_status}_{lane_status}_{crash}"

    def _evaluate_action_diversity(self, actions: List[Any]) -> float:
        if not actions:
            return 0.0
        actions = np.array(actions)
        steering = actions[:, 0]
        accel = actions[:, 1]

        extreme = 0.0
        if np.any(np.abs(steering) > 0.7):
            extreme += 1.0
        if np.any(accel < -0.6) or np.any(accel > 0.8):
            extreme += 1.0
        return extreme

    def _evaluate_state_diversity(self, states: List[Any]) -> float:
        if not states or len(states[0]) < 5:
            return 0.5
        norms = [np.linalg.norm(s) for s in states]
        avg_norm = np.mean(norms)
        if avg_norm > 5.0 or avg_norm < 1.0:
            return 1.0
        return 0.5


class CurriculumTrajectoryBuffer(TrajectoryPriorityBuffer):
    def __init__(self, capacity=2000):
        super().__init__(capacity)
        self.min_difficulty = 0.0
        self.max_difficulty = 10.0
        self.current_difficulty_threshold = 2.0
        self.difficulty_history = []

    def calculate_priority(self, traj: Trajectory) -> float:
        difficulty = self._assess_difficulty(traj)
        self._update_difficulty_threshold(difficulty)

        if difficulty <= self.current_difficulty_threshold:
            # 当前阶段可处理：高优先级
            priority = max(5.0, 8.0 - difficulty)
        else:
            # 超纲样本：低优先级但保留
            priority = max(0.1, 2.0 / (difficulty - self.current_difficulty_threshold + 0.1))

        return max(0.1, min(10.0, priority))

    def _assess_difficulty(self, traj: Trajectory) -> float:
        difficulty = 0.0

        # 1. 平均速度（高速更难）
        avg_speed = np.mean([info.get('speed', 0.0) for info in traj.infos])
        difficulty += min(avg_speed / 10.0, 3.0)

        # 2. 约束违反总量
        difficulty += min(traj.total_constraint * 2.0, 3.0)

        # 3. 脱离道路程度
        avg_centering = np.mean([r['centering_reward'] for r in traj.rewards])
        if avg_centering < 0.5:
            difficulty += 2.0

        # 4. 车道错误
        avg_right_lane = np.mean([r['right_lane_reward'] for r in traj.rewards])
        if avg_right_lane < 0.5:
            difficulty += 1.0

        # 5. 是否发生碰撞
        if any(info.get('collision', False) for info in traj.infos):
            difficulty += 2.0

        return min(difficulty, self.max_difficulty)

    def _update_difficulty_threshold(self, difficulty: float):
        self.difficulty_history.append(difficulty)
        if len(self.difficulty_history) > 1000:
            self.difficulty_history.pop(0)

        if len(self.difficulty_history) >= 100:
            recent_avg = np.mean(self.difficulty_history[-100:])
            if recent_avg > self.current_difficulty_threshold:
                self.current_difficulty_threshold = min(
                    self.max_difficulty,
                    self.current_difficulty_threshold + 0.05
                )

class TrajectoryBuffer:
    def __init__(self, min_start_train: int = 32, total_capacity: int = 10000):  # 注意：trajectory 数量远少于 step
        self.buffers = [
            SafetyCriticalTrajectoryBuffer(int(total_capacity * 0.25)),
            PerformanceTrajectoryBuffer(int(total_capacity * 0.3)),
            DiversityTrajectoryBuffer(int(total_capacity * 0.25)),
            CurriculumTrajectoryBuffer(int(total_capacity * 0.2))
        ]
        self.total_capacity = total_capacity
        self.min_start_train = min_start_train
        self.gep_penalty_factor = 1.0

    def handle_new_trajectory(self, trajectory: Trajectory):
        idx = self._classify_trajectory(trajectory)
        priority = self._calculate_gep_priority(trajectory, idx)
        added = self.buffers[idx].add(trajectory, priority)
        if not added:
            self._handle_buffer_full(idx, trajectory, priority)

    def _classify_trajectory(self, traj: Trajectory) -> int:
        if any(info.get('collision', False) for info in traj.infos):
            return 0
        if traj.total_cost < 10.0 and traj.total_constraint < 0.1:  # good performance
            return 1
        if self._is_rare_path(traj.path_id):
            return 2
        return 3

    def sample_batch(self, batch_size: int) -> List[Trajectory]:
        # 权重可调整，例如更重视安全轨迹
        weights = [0.7, 0.0, 0.3, 0.0]
        samples = []
        for i, buf in enumerate(self.buffers):
            if len(buf) > 0:
                n = max(1, int(batch_size * weights[i]))
                samples.extend(buf.sample(n))
        if len(samples) > batch_size:
            return random.sample(samples, batch_size)
        return samples

    def should_start_training(self) -> bool:
        return sum(len(buf) for buf in self.buffers) >= self.min_start_train

    def _calculate_gep_priority(self, trajectory: Trajectory, buffer_idx: int) -> float:
        """
        根据轨迹内容和目标缓冲区类型，计算 GEP 风格的优先级。
        GEP 强调：探索性（diversity）、性能提升（competence progress）、安全性（criticality）。
        此处复用各子缓冲区的 calculate_priority 方法，并加入全局 penalty 调整。
        """
        # 获取对应缓冲区的原生优先级
        base_priority = self.buffers[buffer_idx].calculate_priority(trajectory)

        # 可选：根据全局 GEP 策略调整（例如对高约束轨迹施加 penalty）
        constraint_penalty = max(0.0, trajectory.total_constraint - 0.5)
        adjusted_priority = base_priority - self.gep_penalty_factor * constraint_penalty

        # 保证优先级在合理范围内
        return max(0.1, min(10.0, adjusted_priority))

    def _handle_buffer_full(self, buffer_idx: int, trajectory: Trajectory, priority: float):
        """
        当目标缓冲区已满且新轨迹优先级不足以替换最低项时，
        尝试以下策略：
        1. 临时扩容（仅限非安全关键缓冲区）
        2. 降级存入次优缓冲区（例如 diversity -> curriculum）
        3. 直接丢弃（最后手段）
        """
        target_buffer = self.buffers[buffer_idx]

        # 策略1：临时扩容（避免频繁丢弃高价值轨迹）
        if buffer_idx != 0:  # 不对 SafetyCritical 缓冲区扩容（保持严格性）
            original_capacity = target_buffer.capacity
            target_buffer.temporarily_expand()
            added = target_buffer.add(trajectory, priority)
            if added:
                return  # 成功插入，结束
            else:
                # 扩容后仍无法插入？恢复容量（理论上不会发生，但保险起见）
                target_buffer.capacity = original_capacity

        # 策略2：尝试降级存入其他缓冲区（按优先级顺序）
        fallback_order = [2, 1, 3]  # 先 diversity，再 performance，最后 curriculum
        if buffer_idx in fallback_order:
            fallback_order.remove(buffer_idx)

        for idx in fallback_order:
            fallback_priority = self._calculate_gep_priority(trajectory, idx)
            if self.buffers[idx].add(trajectory, fallback_priority):
                return  # 成功降级存储

        # 彻底丢弃路径
        # print(f"Trajectory (path_id={trajectory.path_id}) dropped: all buffers full and low priority.")

    def _is_rare_path(self, path_id: int) -> bool:
        """
        判断 path_id 是否属于“稀有路径”。
        使用 DiversityBuffer 中的 path_counts 进行判断。
        若该路径出现频率低于阈值（如 < 2%），则视为稀有。
        """
        diversity_buf = self.buffers[2]  # DiversityTrajectoryBuffer
        total = diversity_buf.total_trajectories
        if total == 0:
            return True  # 初始阶段所有路径都算稀有

        count = diversity_buf.path_counts.get(path_id, 0)
        frequency = count / total
        return frequency < 0.02  # 阈值可调，当前设为 2%