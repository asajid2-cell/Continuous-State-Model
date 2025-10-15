"""
Prioritized replay buffer for surprise-driven reinforcement of rare events.

Phase A goal: define an interface and minimal implementation so the training
loop can schedule sampling without yet optimizing for large scale.
"""

from __future__ import annotations

import bisect
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ReplayItem:
    """Payload stored in replay."""

    payload: Tuple
    priority: float


class PrioritizedReplay:
    """
    Naive prioritized replay supporting proportional sampling.

    For small capacities the list-based approach is sufficient.
    We'll swap in a sum-tree once profiling shows it is necessary.
    """

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self._items: List[ReplayItem] = []

    def add(self, payload: Tuple, priority: float) -> None:
        """Insert a new payload with the given priority."""

        priority = max(priority, 1e-6)
        item = ReplayItem(payload=payload, priority=priority ** self.alpha)
        if len(self._items) >= self.capacity:
            self._items.pop(0)
        self._items.append(item)

    def sample(self, k: int) -> List[ReplayItem]:
        """Sample k items proportional to their priority."""

        if not self._items:
            return []

        total = sum(item.priority for item in self._items)
        samples: List[ReplayItem] = []
        for _ in range(min(k, len(self._items))):
            r = random.uniform(0, total)
            cdf = 0.0
            for item in self._items:
                cdf += item.priority
                if cdf >= r:
                    samples.append(item)
                    break
        return samples

    def __len__(self) -> int:
        return len(self._items)

