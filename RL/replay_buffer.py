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

class _SumTree:

    def __init__(self, capacity: int):
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
        if s <= self.tree[left]:
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
        self._propagate(tree_idx, delta)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

    def state_dict(self) -> dict:
        return {
            "capacity":  self.capacity,
            "tree":      self.tree.copy(),
            "data":      self.data.copy(),
            "n_entries": self.n_entries,
            "write":     self._write,
        }

    def load_state_dict(self, d: dict) -> None:
        assert d["capacity"] == self.capacity, (
            f"Buffer capacity mismatch: saved={d['capacity']}, current={self.capacity}"
        )
        self.tree     = d["tree"].copy()
        self.data     = d["data"].copy()
        self.n_entries = d["n_entries"]
        self._write   = d["write"]

class SILBuffer:

    def __init__(self, capacity: int = 50_000):
        self._buf = deque(maxlen=capacity)
        self._capacity = capacity

    def push(self, state: np.ndarray, action: int, mc_return: float) -> None:
        self._buf.append((state, action, mc_return))

    def sample(self, batch_size: int):
        k = min(batch_size, len(self._buf))
        return random.choices(self._buf, k=k)

    def __len__(self) -> int:
        return len(self._buf)

    def state_dict(self) -> dict:
        return {"buf": list(self._buf), "capacity": self._capacity}

    def load_state_dict(self, d: dict) -> None:
        self._buf = deque(d["buf"], maxlen=self._capacity)


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
             priority_boost: float = 1.0) -> None:
        t = Transition(state, action, reward, next_state, done, next_action_mask,
                       transition_type)
        priority = (self._max_priority ** self.alpha) * priority_boost
        self._tree.add(priority, t)

    def sample(self, batch_size: int):
        batch, tree_indices, priorities = [], [], []
        total = self._tree.total
        segment = total / batch_size
        self._beta_frame += 1

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.clip(random.uniform(lo, hi), 1e-8, total - 1e-8)
            idx, priority, data = self._tree.get(s)

            if data is None:
                s = random.uniform(0, total)
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
            self._tree.update(idx, float(p))

    def __len__(self) -> int:
        return self._tree.n_entries

    def state_dict(self) -> dict:
        return {
            "tree":         self._tree.state_dict(),
            "alpha":        self.alpha,
            "beta_start":   self.beta_start,
            "beta_frames":  self.beta_frames,
            "beta_frame":   self._beta_frame,
            "max_priority": self._max_priority,
        }

    def load_state_dict(self, d: dict) -> None:
        self._beta_frame   = d["beta_frame"]
        self._max_priority = d["max_priority"]
        self._tree.load_state_dict(d["tree"])

    def buffer_stats(self) -> dict:

        counts = {"solve": 0, "deadlock": 0, "timeout": 0, "ongoing": 0, "unknown": 0}
        n = self._tree.n_entries
        for i in range(n):
            t = self._tree.data[i]
            if t is None:
                continue
            ttype = getattr(t, "transition_type", "unknown")
            if ttype in counts:
                counts[ttype] += 1
            else:
                counts["unknown"] += 1

        total = max(sum(counts.values()), 1)
        return {
            "total":        total,
            "solve":        counts["solve"],
            "deadlock":     counts["deadlock"],
            "timeout":      counts["timeout"],
            "ongoing":      counts["ongoing"],
            "unknown":      counts["unknown"],
            "solve_pct":    counts["solve"]    / total * 100,
            "deadlock_pct": counts["deadlock"] / total * 100,
            "timeout_pct":  counts["timeout"]  / total * 100,
            "ongoing_pct":  counts["ongoing"]  / total * 100,
        }