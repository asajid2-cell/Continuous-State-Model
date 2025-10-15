"""
Evaluation helpers for monitoring residual cross-entropy, drift, and canary health.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class MetricsAccumulator:
    """Tracks running averages of key metrics."""

    total_ce: float = 0.0
    total_tokens: int = 0
    updates: int = 0
    extras: Dict[str, float] = field(default_factory=dict)
    extra_counts: Dict[str, int] = field(default_factory=dict)

    def update_cross_entropy(self, loss: torch.Tensor, tokens: int) -> None:
        self.total_ce += float(loss.item()) * tokens
        self.total_tokens += tokens
        self.updates += 1

    def add_metric(self, name: str, value: float, accumulate: bool = True) -> None:
        if accumulate:
            self.extras[name] = self.extras.get(name, 0.0) + value
            self.extra_counts[name] = self.extra_counts.get(name, 0) + 1
        else:
            self.extras[name] = value
            self.extra_counts[name] = 1

    def summary(self) -> Dict[str, float]:
        average_ce = self.total_ce / max(self.total_tokens, 1)
        metrics = {"avg_cross_entropy": average_ce}
        for name, total in self.extras.items():
            count = max(self.extra_counts.get(name, 1), 1)
            metrics[name] = total / count
        return metrics
