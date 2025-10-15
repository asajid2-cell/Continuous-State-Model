"""
Run a quick integration check of the continual-learning stack.

Loads the config, instantiates the model, runs a short stream training pass, and
optionally performs a chat smoke test.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path

import torch

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
    parser.add_argument("--max-steps", type=int, default=32, help="Override training.max_steps for the smoke test")
    parser.add_argument("--chat-prompt", type=str, default=None, help="Optional prompt to run after training")
    return parser.parse_args()


def _resolve_amp(dtype_name: str) -> torch.dtype:
    try:
        return getattr(torch, dtype_name)
    except AttributeError:
        LOG.warning("Unknown AMP dtype '%s', defaulting to float16", dtype_name)
        return torch.float16


def _summarize_devices() -> None:
    if torch.cuda.is_available():
        LOG.info("CUDA device: %s", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        LOG.info("CUDA not available; running on CPU")


def _maybe_chat(trunk, prompt: str | None) -> None:
    if not prompt:
        return
    LOG.info("Running chat prompt: %s", prompt)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    reply = trunk.chat(messages, max_new_tokens=128)
    LOG.info("Chat reply: %s", reply)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    _summarize_devices()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    LOG.info("Loading config from %s", args.config)
    app_cfg: AppConfig = load_app_config(args.config)

    training_cfg = app_cfg.training
    if args.max_steps is not None and args.max_steps > 0:
        training_cfg = replace(training_cfg, max_steps=args.max_steps)
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
        teacher_source = ema_teacher if ema_teacher is not None else predictor
        teacher_head = create_teacher(teacher_source)
        LOG.info("Teacher head initialised for KL regularisation")

    replay_cfg = training_cfg.replay
    replay = None
    if replay_cfg.interval > 0 and replay_cfg.batch > 0 and replay_cfg.capacity > 0:
        replay = PrioritizedReplay(capacity=replay_cfg.capacity, alpha=replay_cfg.alpha)
        LOG.info("Replay buffer ready (capacity=%s, interval=%s, batch=%s)", replay_cfg.capacity, replay_cfg.interval, replay_cfg.batch)

    amp_settings = training_cfg.amp
    amp_enabled = bool(amp_settings.enabled and torch.cuda.is_available())
    amp_dtype = _resolve_amp(amp_settings.dtype)

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
        replay=replay,
    )

    stream_iterable = stream_loader_factory(
        path=str(app_cfg.data.stream_path),
        tokenizer=trunk.tokenizer,
        sequence_length=app_cfg.data.sequence_length,
    )

    start = time.time()
    metrics = trainer.train(stream_iterable, log_interval=app_cfg.logging.log_interval)
    elapsed = time.time() - start
    LOG.info("Smoke test finished in %.2fs | metrics=%s", elapsed, metrics)

    _maybe_chat(trunk, args.chat_prompt)


if __name__ == "__main__":
    main()
