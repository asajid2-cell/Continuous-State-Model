"""
Loss helpers for the dual-stream delta learner.

Phase A only needs wrappers that call into PyTorch so the training loop can wire
terms together without yet tuning coefficients.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")
mse_loss = nn.MSELoss(reduction="mean")
kl_div_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)


def prediction_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross entropy between next-token logits and ground truth tokens."""

    return cross_entropy_loss(logits.view(-1, logits.size(-1)), target.view(-1))


def consistency_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Mean squared error between student and teacher logits."""

    return mse_loss(student_logits, teacher_logits)


def residual_loss(delta: torch.Tensor, projected: torch.Tensor) -> torch.Tensor:
    """Mean squared error between observed delta and linear projection."""

    return mse_loss(delta, projected)


def kl_guard(student_log_probs: torch.Tensor, base_log_probs: torch.Tensor) -> torch.Tensor:
    """KL divergence between student and teacher log probabilities."""

    return kl_div_loss(student_log_probs, base_log_probs)


def aggregate_losses(loss_terms: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum weighted loss components into a scalar."""

    return torch.stack(list(loss_terms.values()), dim=0).sum()
