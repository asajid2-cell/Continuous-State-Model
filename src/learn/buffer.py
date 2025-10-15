"""
Two-tick buffer implementation for aligning predicted frames with delayed reality.

Phase A target: provide a deterministic container that stores predictions keyed
by `t+1` and retrieves them when the real observation arrives two steps later.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import torch


@dataclass
class BufferEntry:
    """Stored prediction and metadata needed for learning."""

    step_key: int
    predicted_tokens: torch.Tensor
    logits: torch.Tensor
    features: torch.Tensor
    meta: Dict[str, str]


class TwoTickBuffer:
    """Simple fixed-capacity ring buffer keyed by timestep."""

    def __init__(self, capacity: int = 8192) -> None:
        self.capacity = capacity
        self._queue: Deque[BufferEntry] = deque(maxlen=capacity)

    def store(self, entry: BufferEntry) -> None:
        """Insert a new prediction entry."""

        self._queue.append(entry)

    def pop(self, step_key: int) -> Optional[BufferEntry]:
        """
        Remove and return the entry matching `step_key`.

        The Phase A prototype uses a linear scan; we can replace this with a hash
        map once we validate the pipeline.
        """

        for idx, entry in enumerate(self._queue):
            if entry.step_key == step_key:
                self._queue.rotate(-idx)
                found = self._queue.popleft()
                self._queue.rotate(idx)
                return found
        return None

    def __len__(self) -> int:
        return len(self._queue)

