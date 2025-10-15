"""
Forward predictor head that produces next-token logits or latent features.

During Phase A we only need scaffolding so the training loop can compile.
Actual weight initialization from the trunk decoder matrix will arrive once
the base model loader is in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class PredictorConfig:
    """Hyper-parameters controlling the predictor head."""

    hidden_size: int
    vocab_size: int
    use_variance_head: bool = False


class ForwardPredictor(nn.Module):
    """Simple linear head with optional variance prediction for uncertainty gating."""

    def __init__(self, cfg: PredictorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.variance_head: Optional[nn.Linear] = None
        if cfg.use_variance_head:
            self.variance_head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Produce logits (and optionally variance estimates) from trunk features.

        Returns:
            dict with keys:
              * 'logits': next-token logits tensor.
              * 'variance': optional log variance for uncertainty-based scaling.
        """

        outputs: Dict[str, torch.Tensor] = {"logits": self.decoder(features)}
        if self.variance_head is not None:
            outputs["variance"] = self.variance_head(features)
        return outputs

