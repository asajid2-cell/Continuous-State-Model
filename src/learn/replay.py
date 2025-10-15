"""
Prioritized replay buffer with importance sampling and without replacement sampling.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass
class ReplayItem:
    """Payload stored in replay."""

    payload: Any
    priority: float


class PrioritizedReplay:
    """Naive prioritized replay supporting proportional sampling and importance weights."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.0) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self._items: List[ReplayItem] = []

    def add(self, payload: Any, priority: float) -> None:
        priority = max(priority, 1e-6) ** self.alpha
        item = ReplayItem(payload=payload, priority=priority)
        if len(self._items) >= self.capacity:
            self._items.pop(0)
        self._items.append(item)

    def sample(self, k: int, with_replacement: bool = False) -> List[Tuple[ReplayItem, float]]:
        if not self._items:
            return []

        results: List[Tuple[ReplayItem, float]] = []
        pool: List[ReplayItem] = self._items if with_replacement else list(self._items)
        total_items = len(self._items)

        for _ in range(min(k, total_items)):
            if not pool:
                break
            total_priority = sum(item.priority for item in pool)
            if total_priority <= 0:
                break
            r = random.uniform(0.0, total_priority)
            cumulative = 0.0
            chosen_index = 0
            chosen_item = pool[0]
            for idx, item in enumerate(pool):
                cumulative += item.priority
                if cumulative >= r:
                    chosen_index = idx
                    chosen_item = item
                    break

            probability = chosen_item.priority / total_priority if total_priority > 0 else 0.0
            weight = 1.0
            if self.beta > 0 and probability > 0:
                weight = (total_items * probability) ** (-self.beta)
            results.append((chosen_item, weight))

            if not with_replacement:
                pool.pop(chosen_index)

        return results

    def __len__(self) -> int:
        return len(self._items)
