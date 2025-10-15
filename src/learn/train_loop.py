"""
Prototype training loop for Phase A of the delta-driven learner.

Includes residual projection, EMA consistency, prioritized replay, and AMP support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import logging
import math

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from src.data.stream import StreamRecord
from src.eval.metrics import MetricsAccumulator
from src.learn.buffer import BufferEntry, TwoTickBuffer
from src.learn import losses
from src.learn.ema import update_ema
from src.learn.replay import PrioritizedReplay
from src.model.trunk import StreamingTrunk
from src.model.predictor import ForwardPredictor

logger = logging.getLogger(__name__)


@dataclass
class PhaseATrainConfig:
    """Configuration for the Phase A loop."""

    kl_weight: float
    consistency_weight: float
    residual_weight: float
    grad_clip: float
    ema_decay: float
    max_steps: int
    grad_accum_steps: int = 1
    replay_interval: int = 0
    replay_batch: int = 0
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float16
    use_grad_scaler: bool = True


class PhaseATrainer:
    """High-level wrapper around the online learning cycle."""

    def __init__(
        self,
        cfg: PhaseATrainConfig,
        trunk: StreamingTrunk,
        predictor: ForwardPredictor,
        optimizer: torch.optim.Optimizer,
        residual_head: Optional[nn.Module] = None,
        teacher_head: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        ema_teacher: Optional[ForwardPredictor] = None,
        replay: Optional[PrioritizedReplay] = None,
        frozen_base: Optional[StreamingTrunk] = None,
    ) -> None:
        self.cfg = cfg
        self.trunk = trunk
        self.predictor = predictor
        self.residual_head = residual_head
        self.teacher_head = teacher_head
        self.optimizer = optimizer
        self.buffer = TwoTickBuffer()
        self.metrics = MetricsAccumulator()
        self.device = device or trunk.primary_device
        self.ema_teacher = ema_teacher
        if self.ema_teacher is not None:
            self.ema_teacher.eval()
        self.replay = replay
        self.frozen_base = frozen_base
        self.step = 0
        self._accum_counter = 0
        self._trainable_params = list(self.trunk.base_model.parameters()) + list(self.predictor.parameters())
        if self.residual_head is not None:
            self._trainable_params += list(self.residual_head.parameters())

        self.use_amp = cfg.use_amp and torch.cuda.is_available()
        self.amp_dtype = cfg.amp_dtype if isinstance(cfg.amp_dtype, torch.dtype) else torch.float16
        self.scaler = GradScaler(enabled=self.use_amp and cfg.use_grad_scaler)

        if self.teacher_head is not None:
            self._sync_teacher(self.ema_teacher if self.ema_teacher is not None else self.predictor)

    def train(self, stream: Iterable[StreamRecord], log_interval: int = 100) -> Dict[str, float]:
        logger.info("Starting Phase A training loop")
        self.optimizer.zero_grad(set_to_none=True)
        for record in stream:
            if self.cfg.max_steps and self.step >= self.cfg.max_steps:
                logger.info("Reached max_steps=%s; stopping stream iteration", self.cfg.max_steps)
                break

            if not record.tokens:
                logger.debug("Skipping empty token record at t=%s", record.t)
                continue

            self._apply_buffer_updates(record)
            self._enqueue_prediction(record)

            self.step += 1
            self._run_replay_updates()

            if log_interval and self.step % log_interval == 0:
                summary = self.metrics.summary()
                logger.info("Step %s: avg CE %.4f", self.step, summary.get("avg_cross_entropy", 0.0))

        logger.info("Training loop finished at step %s", self.step)
        return self.metrics.summary()

    def _apply_buffer_updates(self, record: StreamRecord) -> None:
        buffer_entry = self.buffer.pop(record.t)
        if buffer_entry is None:
            logger.debug("No buffered prediction for t=%s", record.t)
            return

        prev_features = buffer_entry.features.to(
            device=self.device,
            dtype=self.predictor.decoder.weight.dtype,
        )
        target_token = torch.tensor([record.tokens[0]], device=self.device)

        need_hidden = self.residual_head is not None or (self.replay is not None)
        current_hidden: Optional[torch.Tensor] = None

        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            logits_prev = self.predictor(prev_features)
            if need_hidden:
                input_ids = torch.tensor([record.tokens], device=self.device, dtype=torch.long)
                _, aux_cur = self.trunk(input_ids=input_ids, attention_mask=None, use_cache=False)
                hidden_states = aux_cur["hidden_states"]
                current_hidden = hidden_states[:, -1, :]

        logits_prev_tensor = logits_prev.float().clamp(-20, 20)
        prediction_loss = losses.prediction_loss(logits_prev_tensor, target_token)
        if not torch.isfinite(prediction_loss):
            logger.warning("Non-finite prediction loss at t=%s", record.t)
            return
        loss_terms: Dict[str, torch.Tensor] = {"prediction": prediction_loss}

        if current_hidden is not None:
            current_hidden = current_hidden.to(self.device, dtype=prev_features.dtype)

        if self.teacher_head is not None:
            with torch.no_grad():
                teacher_logits = self.teacher_head(prev_features.float()).clamp(-20, 20)
            student_log_probs = nn.functional.log_softmax(logits_prev_tensor, dim=-1)
            teacher_log_probs = nn.functional.log_softmax(teacher_logits, dim=-1)
            kl = losses.kl_guard(student_log_probs, teacher_log_probs)
            loss_terms["kl"] = self.cfg.kl_weight * kl
        elif self.frozen_base is not None:
            with torch.no_grad():
                base_outputs = self.frozen_base.base_model(
                    input_ids=torch.tensor([record.tokens], device=self.device),
                    output_hidden_states=False,
                    use_cache=False,
                )
                base_log_probs = nn.functional.log_softmax(base_outputs.logits[:, -1, :], dim=-1)
            student_log_probs = nn.functional.log_softmax(logits_prev_tensor, dim=-1)
            kl = losses.kl_guard(student_log_probs, base_log_probs)
            loss_terms["kl"] = self.cfg.kl_weight * kl

        if self.residual_head is not None and current_hidden is not None:
            delta = (current_hidden - prev_features).float()
            predicted_delta = self.residual_head(prev_features).float()
            residual_loss = losses.residual_loss(delta, predicted_delta)
            if torch.isfinite(residual_loss):
                loss_terms["residual"] = self.cfg.residual_weight * residual_loss
            else:
                logger.warning("Non-finite residual loss at t=%s", record.t)

        if self.ema_teacher is not None:
            with torch.no_grad():
                ema_logits = self.ema_teacher(prev_features)["logits"].detach().float().clamp(-20, 20)
            consistency = losses.consistency_loss(logits_prev_tensor, ema_logits)
            loss_terms["consistency"] = self.cfg.consistency_weight * consistency

        total_loss = torch.stack(list(loss_terms.values())).sum()
        gate = self._compute_uncertainty_gate(logits_prev.detach())
        loss_for_backprop = (gate * total_loss) / max(self.cfg.grad_accum_steps, 1)

        self._backward(loss_for_backprop)
        self._accum_counter += 1
        if self._accum_counter % max(self.cfg.grad_accum_steps, 1) == 0:
            self._optimizer_step()
            self._accum_counter = 0

        if self.replay is not None and current_hidden is not None:
            delta_priority = float((current_hidden.float() - prev_features.float()).norm().cpu())
            payload = {
                "features": prev_features.detach().to("cpu", torch.float32),
                "target": target_token.detach().to("cpu", torch.long),
                "actual_hidden": current_hidden.detach().to("cpu", torch.float32),
            }
            self.replay.add(payload, priority=delta_priority)

        self.metrics.update_cross_entropy(prediction_loss.detach(), tokens=1)
        self.metrics.add_metric("gate", float(gate.detach()))
        logger.debug(
            "Applied update for t=%s | loss=%.4f gate=%.3f buffer_size=%s",
            record.t,
            float(total_loss.detach()),
            float(gate.detach()),
            len(self.buffer),
        )

    def _enqueue_prediction(self, record: StreamRecord) -> None:
        input_ids = torch.tensor([record.tokens], device=self.device, dtype=torch.long)
        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            logits, aux = self.trunk(input_ids=input_ids, attention_mask=None, use_cache=True)
            hidden_states = aux["hidden_states"]
            last_hidden = hidden_states[:, -1, :]
            pred_outputs = self.predictor(last_hidden)
            logits_pred = pred_outputs["logits"]
        target_dtype = self.predictor.decoder.weight.dtype
        last_hidden = last_hidden.to(device=self.device, dtype=target_dtype)

        predicted_tokens = torch.argmax(logits_pred, dim=-1)
        entry = BufferEntry(
            step_key=record.t + 1,
            predicted_tokens=predicted_tokens.detach(),
            logits=logits_pred.detach(),
            features=last_hidden.detach(),
            meta=record.meta,
        )
        self.buffer.store(entry)
        logger.debug("Stored prediction for t+1=%s | seq_len=%s", entry.step_key, input_ids.size(-1))

    def _run_replay_updates(self) -> None:
        if self.replay is None or self.cfg.replay_interval <= 0 or self.cfg.replay_batch <= 0:
            return
        if self.step % self.cfg.replay_interval != 0:
            return

        samples = self.replay.sample(self.cfg.replay_batch)
        if not samples:
            return

        for item in samples:
            payload = item.payload
            features = payload["features"].to(self.device, dtype=self.predictor.decoder.weight.dtype)
            target = payload["target"].to(self.device)

            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                logits = self.predictor(features)["logits"]
            loss_terms: Dict[str, torch.Tensor] = {
                "prediction": losses.prediction_loss(logits.float().clamp(-20, 20), target)
            }

            if self.residual_head is not None and "actual_hidden" in payload:
                actual_hidden = payload["actual_hidden"].to(self.device, dtype=features.dtype)
                pred_delta = self.residual_head(features).float()
                delta = (actual_hidden - features).float()
                loss_terms["residual"] = self.cfg.residual_weight * losses.residual_loss(delta, pred_delta)

            total_loss = torch.stack(list(loss_terms.values())).sum()
            gate = self._compute_uncertainty_gate(logits.detach())
            self._backward(gate * total_loss)
            self._optimizer_step()

    def _backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._trainable_params, self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._trainable_params, self.cfg.grad_clip)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema_teacher is not None:
            update_ema(self.predictor, self.ema_teacher, self.cfg.ema_decay)
        self._sync_teacher(self.ema_teacher if self.ema_teacher is not None else self.predictor)

    def _sync_teacher(self, source: Optional[ForwardPredictor]) -> None:
        if self.teacher_head is None or source is None:
            return
        with torch.no_grad():
            self.teacher_head.decoder.weight.data.copy_(
                source.decoder.weight.detach().float().to(self.teacher_head.decoder.weight.device)
            )

    @staticmethod
    def _compute_uncertainty_gate(logits: torch.Tensor) -> torch.Tensor:
        logits_fp32 = logits.float().clamp(-20, 20)
        probs = nn.functional.softmax(logits_fp32, dim=-1)
        entropy = -(probs * nn.functional.log_softmax(logits_fp32, dim=-1)).sum(dim=-1)
        max_entropy = math.log(logits.size(-1))
        gate = 1.0 - (entropy / max_entropy)
        return torch.clamp(gate.mean(), min=0.1, max=1.0)
