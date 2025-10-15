"""
Frozen teacher head that mirrors the predictor in float32 for KL regularization.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .predictor import ForwardPredictor


class TeacherHead(nn.Module):
    """Float32 copy of the predictor decoder used for KL targets."""

    def __init__(self, hidden_size: int, vocab_size: int, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.decoder.weight.data.zero_()
        if weight is not None:
            self.decoder.weight.data.copy_(weight.float())
        self.decoder.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features.float())


def create_teacher_from_predictor(predictor: ForwardPredictor) -> TeacherHead:
    weight = predictor.decoder.weight.detach().cpu()
    hidden_size = predictor.decoder.in_features
    vocab_size = predictor.decoder.out_features
    teacher = TeacherHead(hidden_size=hidden_size, vocab_size=vocab_size, weight=weight)
    teacher.to(predictor.decoder.weight.device, dtype=torch.float32)
    return teacher
