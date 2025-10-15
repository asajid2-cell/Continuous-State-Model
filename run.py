"""
Smoke-test harness for the continual-learning stack.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

torch.set_float32_matmul_precision("medium")

from src.config import AppConfig, load_app_config
from src.data.stream import stream_loader_factory
from src.learn.train_loop import PhaseATrainConfig, PhaseATrainer
from src.learn.replay import PrioritizedReplay
from src.model.factory import create_predictor, create_residual, create_teacher, create_trunk

LOG = logging.getLogger("delta_stream.run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous-state model smoke test")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--chat-prompt", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    LOG.info("Global seed set to %s", seed)


def _resolve_amp_dtype(name: str) -> torch.dtype:
    return getattr(torch, name, torch.float16)


def _summarise_device() -> None:
    if torch.cuda.is_available():
        LOG.info("CUDA device: %s", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        LOG.info("CUDA not available; running on CPU")


def _maybe_chat(trunk, prompt: Optional[str]) -> None:
    if not prompt:
        return
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    reply = trunk.chat(messages, max_new_tokens=128)
    LOG.info("Chat reply: %s", reply)


def _save_checkpoint(path: Path, trunk, predictor, residual_head, optimizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "predictor": predictor.state_dict(),
        "trunk": trunk.base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "residual": residual_head.state_dict() if residual_head is not None else None,
    }
    torch.save(state, path)
    LOG.info("Checkpoint saved to %s", path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    _summarise_device()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    app_cfg: AppConfig = load_app_config(args.config)
    set_global_seed(app_cfg.seed)

    training_cfg = app_cfg.training
    if args.max_steps:
        training_cfg = dataclasses.replace(training_cfg, max_steps=args.max_steps)
        LOG.info("Overriding max_steps to %s for smoke test", args.max_steps)

    LOG.info("Initialising trunk (model=%s)", app_cfg.model.base_name)
    trunk = create_trunk(app_cfg.model)
    device = next(trunk.base_model.parameters()).device
    LOG.info("Hidden size=%s vocab=%s device=%s", trunk.base_model.config.hidden_size, trunk.base_model.config.vocab_size, device)

    LOG.info("Initialising predictor head")
    predictor = create_predictor(trunk, use_variance_head=app_cfg.predictor.use_variance_head, device=str(device))

    residual_head = None
    if app_cfg.residual is not None:
        residual_head = create_residual(trunk, rank=app_cfg.residual.rank)
        LOG.info("Residual projector ready (rank=%s)", app_cfg.residual.rank)

    ema_teacher = None
    if training_cfg.consistency_weight > 0:
        ema_teacher = create_predictor(trunk, use_variance_head=False, device=str(device))
        ema_teacher.load_state_dict(predictor.state_dict())
        ema_teacher.eval()
        for param in ema_teacher.parameters():
            param.requires_grad_(False)
        LOG.info("EMA teacher initialised")

    teacher_head = None
    if training_cfg.kl_weight > 0:
        teacher_head = create_teacher(ema_teacher if ema_teacher is not None else predictor)
        LOG.info("Teacher head initialised for KL regularisation")

    replay_cfg = training_cfg.replay
    replay_buffer = None
    if replay_cfg.interval > 0 and replay_cfg.batch > 0 and replay_cfg.capacity > 0:
        replay_buffer = PrioritizedReplay(capacity=replay_cfg.capacity, alpha=replay_cfg.alpha, beta=replay_cfg.beta)
        LOG.info(
            "Replay buffer ready (capacity=%s, interval=%s, batch=%s, beta=%.2f)",
            replay_cfg.capacity,
            replay_cfg.interval,
            replay_cfg.batch,
            replay_cfg.beta,
        )

    amp_settings = training_cfg.amp
    amp_enabled = bool(amp_settings.enabled and torch.cuda.is_available())
    amp_dtype = _resolve_amp_dtype(amp_settings.dtype)

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in trunk.base_model.parameters() if p.requires_grad], "lr": training_cfg.adapter_lr},
            {"params": list(predictor.parameters()), "lr": training_cfg.predictor_lr},
            *( [ {"params": list(residual_head.parameters()), "lr": training_cfg.predictor_lr} ] if residual_head else [] ),
        ],
        weight_decay=training_cfg.weight_decay,
        eps=training_cfg.adam_eps,
    )

    phase_cfg = PhaseATrainConfig(
        kl_weight=training_cfg.kl_weight,
        consistency_weight=training_cfg.consistency_weight,
        residual_weight=training_cfg.residual_weight,
        grad_clip=training_cfg.grad_clip,
        ema_decay=training_cfg.ema_decay,
        max_steps=training_cfg.max_steps,
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
        ema_teacher=ema_teacher,
        replay=replay_buffer,
    )

    stream_iterable = stream_loader_factory(
        path=str(app_cfg.data.stream_path),
        tokenizer=trunk.tokenizer,
        sequence_length=app_cfg.data.sequence_length,
    )

    start = time.time()
    metrics = trainer.train(stream_iterable, log_interval=app_cfg.logging.log_interval)
    elapsed = time.time() - start
    LOG.info("Smoke test finished in %.2fs", elapsed)
    LOG.info("Metrics: %s", metrics)

    if args.checkpoint_dir:
        checkpoint_path = args.checkpoint_dir / "last.pt"
        _save_checkpoint(checkpoint_path, trunk, predictor, residual_head, optimizer)

    _maybe_chat(trunk, args.chat_prompt)


if __name__ == "__main__":
    main()
