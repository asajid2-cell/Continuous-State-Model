"""
Residual projection head for modeling hidden-state deltas.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ResidualSettings:
    """Configuration for the residual projector."""

    rank: int = 32


class ResidualProjector(nn.Module):
    """Low-rank residual projector mapping hidden states to delta estimates."""

    def __init__(self, hidden_size: int, rank: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        latent = self.down(features)
        latent = self.act(latent)
        return self.up(latent)
