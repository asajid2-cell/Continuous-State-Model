"""
EMA utilities for maintaining teacher models.
"""

from __future__ import annotations

from typing import Iterable

import torch


def update_ema(student: torch.nn.Module, teacher: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
            if not teacher_param.requires_grad:
                continue
            source = student_param.detach()
            if teacher_param.dtype != source.dtype:
                source = source.to(teacher_param.dtype)
            teacher_param.data.mul_(decay).add_(source, alpha=1.0 - decay)
