"""
Factory helpers for constructing the streaming trunk and predictor from config objects.
"""

from __future__ import annotations

from typing import Optional

import logging
import torch

from .predictor import ForwardPredictor, PredictorConfig
from .trunk import StreamingTrunk, TrunkConfig

logger = logging.getLogger(__name__)

def create_trunk(cfg: TrunkConfig) -> StreamingTrunk:
    """Instantiate and load the streaming trunk."""

    logger.debug("Creating StreamingTrunk for base '%s'", cfg.base_name)
    trunk = StreamingTrunk(cfg)
    trunk.load()
    total_params = sum(p.numel() for p in trunk.base_model.parameters())
    trainable = sum(p.numel() for p in trunk.base_model.parameters() if p.requires_grad)
    logger.info("StreamingTrunk ready (total_params=%s, trainable=%s)", f"{total_params:,}", f"{trainable:,}")
    return trunk


def create_predictor(trunk: StreamingTrunk, use_variance_head: bool = False, device: Optional[str] = None) -> ForwardPredictor:
    """
    Instantiate the forward predictor with dimensions inferred from the trunk.

    Args:
        trunk: Loaded StreamingTrunk instance.
        use_variance_head: Whether to add a variance head for uncertainty gating.
        device: Optional device string; defaults to trunk parameter device.
    """

    base = trunk.base_model
    logger.debug(
        "Creating ForwardPredictor (hidden_size=%s, vocab_size=%s, variance_head=%s)",
        base.config.hidden_size,
        base.config.vocab_size,
        use_variance_head,
    )
    cfg = PredictorConfig(
        hidden_size=base.config.hidden_size,
        vocab_size=base.config.vocab_size,
        use_variance_head=use_variance_head,
    )
    predictor = ForwardPredictor(cfg)

    base_param = next(base.parameters())
    target_device = torch.device(device) if device else base_param.device
    target_dtype = base_param.dtype
    predictor = predictor.to(device=target_device, dtype=target_dtype)
    logger.debug(
        "Predictor moved to device: %s (dtype=%s)",
        predictor.decoder.weight.device,
        predictor.decoder.weight.dtype,
    )
    return predictor
