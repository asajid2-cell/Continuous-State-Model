"""
Entry point for running the Phase A streaming training loop.

Once the trunk loader and stream iterator are available this script will wire
them together and launch the continual learning cycle.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import os
import sys
import copy

# Ensure repo root is on sys.path when running as python scripts/... from VS Code/PowerShell
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.optim import AdamW

from src.config import load_app_config
from src.data.stream import stream_loader_factory
from src.learn.train_loop import PhaseATrainConfig, PhaseATrainer
from src.model.factory import create_predictor, create_trunk

logger = logging.getLogger("delta_stream.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run delta-driven streaming training.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level (DEBUG, INFO, WARNING, ...)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    app_cfg = load_app_config(args.config)

    logger.info("Phase A bring-up starting")
    logger.info("Loading trunk: %s", app_cfg.model.base_name)
    trunk = create_trunk(app_cfg.model)
    device = next(trunk.base_model.parameters()).device
    hidden = trunk.base_model.config.hidden_size
    vocab = trunk.base_model.config.vocab_size
    logger.info("Trunk ready (device=%s, hidden_size=%s, vocab_size=%s)", device, hidden, vocab)

    logger.info("Initializing predictor head (variance_head=%s)", app_cfg.predictor.use_variance_head)
    predictor = create_predictor(trunk, use_variance_head=app_cfg.predictor.use_variance_head, device=str(device))
    predictor_params = sum(p.numel() for p in predictor.parameters())
    trainable_params = sum(p.numel() for p in trunk.base_model.parameters() if p.requires_grad)
    logger.info("Predictor params: %s", f"{predictor_params:,}")
    logger.info("Trainable trunk params (LoRA): %s", f"{trainable_params:,}")

    ema_predictor = None
    if app_cfg.training.consistency_weight > 0:
        ema_predictor = copy.deepcopy(predictor).eval()
        for param in ema_predictor.parameters():
            param.requires_grad_(False)
        logger.info("EMA teacher initialized for predictor consistency loss")

    trunk_params = [p for p in trunk.base_model.parameters() if p.requires_grad]
    predictor_params_list = list(predictor.parameters())
    optimizer = AdamW(
        [
            {"params": trunk_params, "lr": app_cfg.training.adapter_lr},
            {"params": predictor_params_list, "lr": app_cfg.training.predictor_lr},
        ],
        weight_decay=app_cfg.training.weight_decay,
    )
    total_params = sum(p.numel() for p in trunk_params) + sum(p.numel() for p in predictor_params_list)
    logger.info("Optimizer ready with %s parameters", f"{total_params:,}")

    phase_cfg = PhaseATrainConfig(
        kl_weight=app_cfg.training.kl_weight,
        consistency_weight=app_cfg.training.consistency_weight,
        residual_weight=app_cfg.training.residual_weight,
        grad_clip=app_cfg.training.grad_clip,
        ema_decay=app_cfg.training.ema_decay,
        max_steps=app_cfg.training.max_steps,
        grad_accum_steps=app_cfg.data.gradient_accumulation_steps,
    )
    trainer = PhaseATrainer(
        cfg=phase_cfg,
        trunk=trunk,
        predictor=predictor,
        optimizer=optimizer,
        ema_teacher=ema_predictor,
    )

    stream_path = Path(app_cfg.data.stream_path)
    if not stream_path.exists():
        logger.warning("Stream file %s not found. Skipping training run.", stream_path)
        return

    tokenizer = trunk.tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer failed to load; cannot build stream loader.")

    stream_iterable = stream_loader_factory(
        path=str(stream_path),
        tokenizer=tokenizer,
        sequence_length=app_cfg.data.sequence_length,
    )

    metrics = trainer.train(stream_iterable, log_interval=app_cfg.logging.log_interval)
    logger.info("Training summary: %s", metrics)


if __name__ == "__main__":
    main()

