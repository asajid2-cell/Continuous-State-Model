"""
Configuration loader for the delta-driven dual-stream project.

Reads YAML config files and instantiates strongly typed dataclasses used across
the training stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.model.trunk import LoraSettings, TrunkConfig


@dataclass
class ResidualSettings:
    """Configuration for the residual projector head."""

    rank: int = 32


@dataclass
class PredictorSettings:
    """Configuration toggles for the forward predictor head."""

    use_variance_head: bool = False


@dataclass
class DataSettings:
    """Streaming data pipeline configuration."""

    stream_path: str
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int


@dataclass
class ReplaySettings:
    """Prioritized replay configuration."""

    interval: int = 0
    batch: int = 0
    capacity: int = 0
    alpha: float = 0.6


@dataclass
class TrainingSettings:
    """Optimization hyper-parameters for adapters and predictor."""

    adapter_lr: float
    predictor_lr: float
    weight_decay: float
    grad_clip: float
    ema_decay: float
    kl_weight: float
    consistency_weight: float
    residual_weight: float
    max_steps: int = 0
    replay: ReplaySettings = field(default_factory=ReplaySettings)


@dataclass
class LoggingSettings:
    """Telemetry configuration."""

    project: str
    run_name: str
    log_interval: int
    eval_interval: int


@dataclass
class AppConfig:
    """Top-level configuration bundle."""

    model: TrunkConfig
    residual: Optional[ResidualSettings]
    predictor: PredictorSettings
    data: DataSettings
    training: TrainingSettings
    logging: LoggingSettings


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_app_config(path: Path) -> AppConfig:
    """Parse a YAML config file into an AppConfig instance."""

    raw = _load_yaml(path)

    model_cfg = raw.get("model", {})
    lora_cfg = model_cfg.get("lora")
    lora = LoraSettings(**lora_cfg) if lora_cfg else None
    trunk_cfg = TrunkConfig(
        base_name=model_cfg["base_name"],
        revision=model_cfg.get("revision", "main"),
        quantization=model_cfg.get("quantization"),
        tokenizer_name=model_cfg.get("tokenizer_name"),
        padding_side=model_cfg.get("padding_side"),
        device=model_cfg.get("device"),
        device_map=model_cfg.get("device_map"),
        torch_dtype=model_cfg.get("torch_dtype"),
        cache_dir=model_cfg.get("cache_dir"),
        lora=lora,
    )
    residual_cfg = model_cfg.get("residual")
    residual_settings = ResidualSettings(**residual_cfg) if residual_cfg else None

    predictor_cfg = raw.get("predictor", {})
    predictor_settings = PredictorSettings(
        use_variance_head=predictor_cfg.get("use_variance_head", False),
    )

    data_cfg = raw.get("data", {})
    data_settings = DataSettings(
        stream_path=data_cfg["stream_path"],
        sequence_length=int(data_cfg["sequence_length"]),
        micro_batch_size=int(data_cfg["micro_batch_size"]),
        gradient_accumulation_steps=int(data_cfg["gradient_accumulation_steps"]),
    )

    training_cfg = raw.get("training", {})
    replay_cfg = training_cfg.get("replay", {})
    replay_settings = ReplaySettings(
        interval=int(replay_cfg.get("interval", 0)),
        batch=int(replay_cfg.get("batch", 0)),
        capacity=int(replay_cfg.get("capacity", 0)),
        alpha=float(replay_cfg.get("alpha", 0.6)),
    )
    training_settings = TrainingSettings(
        adapter_lr=float(training_cfg["adapter_lr"]),
        predictor_lr=float(training_cfg["predictor_lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
        grad_clip=float(training_cfg["grad_clip"]),
        ema_decay=float(training_cfg["ema_decay"]),
        kl_weight=float(training_cfg["kl_weight"]),
        consistency_weight=float(training_cfg["consistency_weight"]),
        residual_weight=float(training_cfg["residual_weight"]),
        max_steps=int(training_cfg.get("max_steps", 0)),
        replay=replay_settings,
    )

    logging_cfg = raw.get("logging", {})
    logging_settings = LoggingSettings(
        project=logging_cfg["project"],
        run_name=logging_cfg["run_name"],
        log_interval=int(logging_cfg["log_interval"]),
        eval_interval=int(logging_cfg["eval_interval"]),
    )

    return AppConfig(
        model=trunk_cfg,
        residual=residual_settings,
        predictor=predictor_settings,
        data=data_settings,
        training=training_settings,
        logging=logging_settings,
    )
