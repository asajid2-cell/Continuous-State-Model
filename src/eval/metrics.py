"""
Evaluation helpers for monitoring residual cross-entropy, drift, and canary health.

Phase A will rely on simple accumulators; richer dashboards will slot in once
telemetry is wired up.
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

    def update_cross_entropy(self, loss: torch.Tensor, tokens: int) -> None:
        """Accumulate residual cross-entropy with token weighting."""

        self.total_ce += float(loss.item()) * tokens
        self.total_tokens += tokens
        self.updates += 1

    def add_metric(self, name: str, value: float) -> None:
        """Store custom metrics such as KL drift or gate suppression rate."""

        self.extras[name] = value

    def summary(self) -> Dict[str, float]:
        """Return current averages for reporting."""

        average_ce = self.total_ce / max(self.total_tokens, 1)
        return {"avg_cross_entropy": average_ce, **self.extras}

