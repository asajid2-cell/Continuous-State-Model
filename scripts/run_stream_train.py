"""
Entry point for running the Phase A streaming training loop.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import os
import sys
import random

import numpy as np
import torch

torch.set_float32_matmul_precision("medium")

# Ensure repo root is on sys.path when running as python scripts/... from VS Code/PowerShell
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_app_config
from src.data.stream import stream_loader_factory
from src.learn.train_loop import PhaseATrainConfig, PhaseATrainer
from src.learn.replay import PrioritizedReplay
from src.model.factory import create_predictor, create_residual, create_teacher, create_trunk

logger = logging.getLogger("delta_stream.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run delta-driven streaming training.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %s", seed)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    app_cfg = load_app_config(args.config)
    set_global_seed(app_cfg.seed)

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

    residual_head = None
    if app_cfg.residual is not None:
        residual_head = create_residual(trunk, rank=app_cfg.residual.rank)
        logger.info("Residual projector initialized (rank=%s)", app_cfg.residual.rank)

    ema_predictor = None
    if app_cfg.training.consistency_weight > 0:
        ema_predictor = create_predictor(trunk, use_variance_head=False, device=str(device))
        ema_predictor.load_state_dict(predictor.state_dict())
        ema_predictor.eval()
        for param in ema_predictor.parameters():
            param.requires_grad_(False)
        logger.info("EMA teacher initialized for predictor consistency loss")

    teacher_head = None
    if app_cfg.training.kl_weight > 0:
        teacher_head = create_teacher(ema_predictor if ema_predictor is not None else predictor)
        logger.info("Float32 teacher head initialized for KL regularization")

    trunk_params = [p for p in trunk.base_model.parameters() if p.requires_grad]
    predictor_params_list = list(predictor.parameters())
    param_groups = [
        {"params": trunk_params, "lr": app_cfg.training.adapter_lr},
        {"params": predictor_params_list, "lr": app_cfg.training.predictor_lr},
    ]
    if residual_head is not None:
        param_groups.append({"params": list(residual_head.parameters()), "lr": app_cfg.training.predictor_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=app_cfg.training.weight_decay, eps=app_cfg.training.adam_eps)

    replay_cfg = app_cfg.training.replay
    replay_buffer = None
    if replay_cfg.interval > 0 and replay_cfg.batch > 0 and replay_cfg.capacity > 0:
        replay_buffer = PrioritizedReplay(capacity=replay_cfg.capacity, alpha=replay_cfg.alpha, beta=replay_cfg.beta)
        logger.info(
            "Replay buffer initialized (capacity=%s, interval=%s, batch=%s, beta=%.2f)",
            replay_cfg.capacity,
            replay_cfg.interval,
            replay_cfg.batch,
            replay_cfg.beta,
        )

    amp_settings = app_cfg.training.amp
    amp_enabled = bool(amp_settings.enabled and torch.cuda.is_available())
    amp_dtype = getattr(torch, amp_settings.dtype, torch.float16)

    phase_cfg = PhaseATrainConfig(
        kl_weight=app_cfg.training.kl_weight,
        consistency_weight=app_cfg.training.consistency_weight,
        residual_weight=app_cfg.training.residual_weight,
        grad_clip=app_cfg.training.grad_clip,
        ema_decay=app_cfg.training.ema_decay,
        max_steps=app_cfg.training.max_steps,
        grad_accum_steps=app_cfg.data.gradient_accumulation_steps,
        replay_interval=replay_cfg.interval,
        replay_batch=replay_cfg.batch,
        use_amp=amp_enabled,
        amp_dtype=amp_dtype,
        use_grad_scaler=amp_settings.use_grad_scaler,
    )

    trainer = PhaseATrainer(
        cfg=phase_cfg,
        trunk=trunk,
        predictor=predictor,
        optimizer=optimizer,
        residual_head=residual_head,
        teacher_head=teacher_head,
        ema_teacher=ema_predictor,
        replay=replay_buffer,
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
