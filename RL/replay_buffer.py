import random
from dataclasses import dataclass
import numpy as np
from collections import deque

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: np.ndarray
    transition_type: str = "ongoing"  # "solve" | "deadlock" | "timeout" | "ongoing"
    action_mask: np.ndarray | None = None
    n_steps: int = 1
    priority_boost: float = 1.0


@dataclass
class SolvedPush:
    state: np.ndarray
    action_mask: np.ndarray
    action: int
    mc_return: float

class _SumTree:

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.n_entries = 0
        self._write = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s < self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        idx = self._write + self.capacity - 1
        self.data[self._write] = data
        self.update(idx, priority)
        self._write = (self._write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx: int, new_priority: float) -> None:
        delta = new_priority - self.tree[tree_idx]
        self.tree[tree_idx] = new_priority
        if tree_idx != 0:
            self._propagate(tree_idx, delta)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "tree": self.tree.copy(),
            "data": self.data.copy(),
            "n_entries": self.n_entries,
            "write": self._write,
        }

    def load_state_dict(self, d: dict) -> None:
        assert d["capacity"] == self.capacity, (
            f"Buffer capacity mismatch: saved={d['capacity']}, current={self.capacity}"
        )
        self.tree = d["tree"].copy()
        self.data = d["data"].copy()
        self.n_entries = d["n_entries"]
        self._write = d["write"]

class SILBuffer:

    def __init__(self, capacity: int = 1_000):
        self._episodes = deque(maxlen=capacity)
        self._capacity = capacity

    def push_episode(self, episode: list[SolvedPush]) -> None:
        if episode:
            self._episodes.append(tuple(episode))

    def sample(self, batch_size: int, segment_length: int = 8) -> list[SolvedPush]:
        if segment_length < 1:
            raise ValueError("segment_length must be at least 1")
        if batch_size <= 0 or not self._episodes:
            return []

        episode_count = min(len(self), batch_size)
        episode_indices = random.sample(range(len(self)), episode_count)
        samples = []
        for episode_idx in episode_indices:
            episode = self._episodes[episode_idx]
            length = min(len(episode), segment_length, batch_size - len(samples))
            start = random.randrange(len(episode) - length + 1)
            samples.extend(episode[start:start + length])
            if len(samples) == batch_size:
                break
        return samples

    def __len__(self) -> int:
        return len(self._episodes)

    def state_dict(self) -> dict:
        return {"episodes": list(self._episodes)}

    def load_state_dict(self, d: dict) -> None:
        self._episodes = deque(d["episodes"], maxlen=self._capacity)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 500_000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self._beta_frame = 1

        self._epsilon = 1e-5
        self._max_priority = 1.0

        self._tree = _SumTree(capacity)

    @property
    def beta(self) -> float:
        progress = self._beta_frame / self.beta_frames
        return min(1.0, self.beta_start + progress * (1.0 - self.beta_start))

    def push(self, state, action, reward, next_state, done, next_action_mask,
             transition_type: str = "ongoing",
             priority_boost: float = 1.0, *, action_mask, n_steps: int = 1) -> None:
        t = Transition(state, action, reward, next_state, done, next_action_mask,
                       transition_type, action_mask, n_steps, priority_boost)
        priority = self._max_priority * priority_boost
        self._tree.add(priority, t)

    def sample(self, batch_size: int):
        total = self._tree.total
        if batch_size < 1 or total <= 0:
            raise ValueError("PER sampling requires a positive batch size and non-empty buffer")

        batch, tree_indices, priorities = [], [], []
        segment = total / batch_size
        upper_bound = np.nextafter(total, 0.0)
        self._beta_frame += 1

        for i in range(batch_size):
            s = min(segment * (i + random.random()), upper_bound)
            idx, priority, data = self._tree.get(s)

            batch.append(data)
            tree_indices.append(idx)
            priorities.append(priority)

        priorities_arr = np.array(priorities, dtype=np.float64)
        probs = priorities_arr / total
        n = self._tree.n_entries
        weights = (n * probs) ** (-self.beta)
        weights /= weights.max()

        return batch, tree_indices, weights.astype(np.float32)

    def update_priorities(self, tree_indices, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + self._epsilon) ** self.alpha
        self._max_priority = max(self._max_priority, float(priorities.max()))
        for idx, p in zip(tree_indices, priorities):
            transition = self._tree.data[idx - self._tree.capacity + 1]
            self._tree.update(idx, float(p) * transition.priority_boost)

    def __len__(self) -> int:
        return self._tree.n_entries

    def state_dict(self) -> dict:
        return {
            "tree": self._tree.state_dict(),
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_frames": self.beta_frames,
            "beta_frame": self._beta_frame,
            "max_priority": self._max_priority,
        }

    def load_state_dict(self, d: dict) -> bool:

        self._tree.load_state_dict(d["tree"])
        self.alpha = d["alpha"]
        self.beta_start = d["beta_start"]
        self.beta_frames = d["beta_frames"]
        self._beta_frame   = d["beta_frame"]
        self._max_priority = d["max_priority"]
        return True

    def buffer_stats(self) -> dict:

        counts = {"solve": 0, "deadlock": 0, "timeout": 0, "ongoing": 0, "unknown": 0}
        n = self._tree.n_entries
        solve_priority = 0.0
        for i in range(n):
            t = self._tree.data[i]
            if t is None:
                continue
            ttype = getattr(t, "transition_type", "unknown")
            if ttype == "solve":
                solve_priority += self._tree.tree[self._tree.capacity - 1 + i]
            if ttype in counts:
                counts[ttype] += 1
            else:
                counts["unknown"] += 1

        total = sum(counts.values())
        denom = max(total, 1)
        tree_total = self._tree.total
        return {
            "total":        total,
            "solve":        counts["solve"],
            "deadlock":     counts["deadlock"],
            "timeout":      counts["timeout"],
            "ongoing":      counts["ongoing"],
            "unknown":      counts["unknown"],
            "solve_pct":          counts["solve"]    / denom * 100,
            "deadlock_pct":       counts["deadlock"] / denom * 100,
            "timeout_pct":        counts["timeout"]  / denom * 100,
            "ongoing_pct":        counts["ongoing"]  / denom * 100,
            "solve_priority_pct": 100 * solve_priority / tree_total if tree_total else 0.0,
        }
