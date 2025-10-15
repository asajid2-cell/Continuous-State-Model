"""
Two-tick buffer storing predictions keyed by timestep with O(1) access.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class BufferEntry:
    """Stored prediction and metadata needed for learning."""

    step_key: int
    predicted_tokens: torch.Tensor
    logits: torch.Tensor
    features: torch.Tensor
    meta: Dict[str, Any]


class TwoTickBuffer:
    """Fixed-capacity buffer keyed by timestep with constant-time pop/store."""

    def __init__(self, capacity: int = 8192) -> None:
        self.capacity = capacity
        self._entries: "OrderedDict[int, BufferEntry]" = OrderedDict()

    def store(self, entry: BufferEntry) -> None:
        if entry.step_key in self._entries:
            del self._entries[entry.step_key]
        elif len(self._entries) >= self.capacity:
            self._entries.popitem(last=False)
        self._entries[entry.step_key] = entry

    def pop(self, step_key: int) -> Optional[BufferEntry]:
        return self._entries.pop(step_key, None)

    def __len__(self) -> int:
        return len(self._entries)
