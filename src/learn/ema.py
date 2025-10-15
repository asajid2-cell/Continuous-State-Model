"""
Exponential moving average (EMA) utilities for maintaining a teacher model.

Phase A implementation only needs a lightweight helper to update parameters
between the student and teacher copies of the trunk.
"""

from __future__ import annotations

from typing import Iterable

import torch


@torch.no_grad()
def update_ema(student: torch.nn.Module, teacher: torch.nn.Module, decay: float) -> None:
    """Update teacher parameters with EMA of student parameters."""

    student_params: Iterable[torch.Tensor] = student.parameters()
    teacher_params: Iterable[torch.Tensor] = teacher.parameters()

    for s_param, t_param in zip(student_params, teacher_params):
        t_param.lerp_(s_param, 1.0 - decay)

