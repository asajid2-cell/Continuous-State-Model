"""
Prototype training loop for Phase A of the delta-driven learner.

The loop currently sketches out the control flow without executing real model
updates; it will be fleshed out once the streaming data loader and trunk wrapper
are complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import logging
import math

import torch
from torch import nn

from src.data.stream import StreamRecord
from src.eval.metrics import MetricsAccumulator
from src.learn.buffer import BufferEntry, TwoTickBuffer
from src.learn import losses
from src.learn.ema import update_ema
from src.model.trunk import StreamingTrunk
from src.model.predictor import ForwardPredictor

logger = logging.getLogger(__name__)


@dataclass
class PhaseATrainConfig:
    """Configuration for the Phase A prototype loop."""

    kl_weight: float
    consistency_weight: float
    residual_weight: float
    grad_clip: float
    ema_decay: float
    max_steps: int
    grad_accum_steps: int = 1


class PhaseATrainer:
    """High-level wrapper around the online learning cycle."""

    def __init__(
        self,
        cfg: PhaseATrainConfig,
        trunk: StreamingTrunk,
        predictor: ForwardPredictor,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        ema_teacher: Optional[ForwardPredictor] = None,
        frozen_base: Optional[StreamingTrunk] = None,
    ) -> None:
        self.cfg = cfg
        self.trunk = trunk
        self.predictor = predictor
        self.optimizer = optimizer
        self.buffer = TwoTickBuffer()
        self.metrics = MetricsAccumulator()
        self.device = device or trunk.primary_device
        self.ema_teacher = ema_teacher
        if self.ema_teacher is not None:
            self.ema_teacher.eval()
        self.frozen_base = frozen_base
        self.step = 0
        self._accum_counter = 0

    def train(self, stream: Iterable[StreamRecord], log_interval: int = 100) -> Dict[str, float]:
        """
        Run a single pass over the stream.

        Returns aggregate statistics so callers can inspect loss composition.
        """

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
            if log_interval and self.step % log_interval == 0:
                logger.info("Step %s: avg CE %.4f", self.step, self.metrics.summary().get("avg_cross_entropy", 0.0))

        logger.info("Training loop finished at step %s", self.step)
        return self.metrics.summary()

    def _apply_buffer_updates(self, record: StreamRecord) -> None:
        """If a prediction exists for timestep `record.t`, apply loss and update parameters."""

        buffer_entry = self.buffer.pop(record.t)
        if buffer_entry is None:
            logger.debug("No buffered prediction for t=%s", record.t)
            return

        target_token = torch.tensor([record.tokens[0]], device=self.device)
        prediction_loss = losses.prediction_loss(buffer_entry.logits, target_token)
        loss_terms: Dict[str, torch.Tensor] = {"prediction": prediction_loss}

        if self.frozen_base is not None:
            with torch.no_grad():
                base_outputs = self.frozen_base.base_model(
                    input_ids=torch.tensor([record.tokens], device=self.device),
                    output_hidden_states=False,
                    use_cache=False,
                )
                base_log_probs = nn.functional.log_softmax(base_outputs.logits[:, -1, :], dim=-1)
            student_log_probs = nn.functional.log_softmax(buffer_entry.logits, dim=-1)
            kl = losses.kl_guard(student_log_probs, base_log_probs)
            loss_terms["kl"] = self.cfg.kl_weight * kl

        # Consistency loss against EMA teacher logits.
        if self.ema_teacher is not None and hasattr(buffer_entry, "features"):
            with torch.no_grad():
                teacher_input = buffer_entry.features.to(
                    device=self.device,
                    dtype=self.predictor.decoder.weight.dtype,
                )
                teacher_logits = self.ema_teacher(teacher_input)["logits"].detach()
            student_logits = buffer_entry.logits
            consistency = losses.consistency_loss(student_logits, teacher_logits)
            loss_terms["consistency"] = self.cfg.consistency_weight * consistency

        total_loss = torch.stack(list(loss_terms.values())).sum()

        gate = self._compute_uncertainty_gate(buffer_entry.logits)
        scaled_loss = gate * total_loss
        loss_for_backprop = scaled_loss / max(self.cfg.grad_accum_steps, 1)
        loss_for_backprop.backward()
        self._accum_counter += 1

        if self._accum_counter % max(self.cfg.grad_accum_steps, 1) == 0:
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.trunk.base_model.parameters()) + list(self.predictor.parameters()),
                    self.cfg.grad_clip,
                )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema_teacher is not None:
                update_ema(self.predictor, self.ema_teacher, self.cfg.ema_decay)
            self._accum_counter = 0
        elif self.ema_teacher is not None:
            update_ema(self.predictor, self.ema_teacher, self.cfg.ema_decay)

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
        """Run the trunk & predictor to produce the next-step prediction and store it."""

        input_ids = torch.tensor([record.tokens], device=self.device, dtype=torch.long)
        logits, aux = self.trunk(
            input_ids=input_ids,
            attention_mask=None,
            use_cache=True,
        )
        hidden_states = aux["hidden_states"]  # batch, seq, hidden
        target_dtype = self.predictor.decoder.weight.dtype
        last_hidden = hidden_states[:, -1, :].to(device=self.device, dtype=target_dtype)

        pred_outputs = self.predictor(last_hidden)
        logits = pred_outputs["logits"]
        predicted_tokens = torch.argmax(logits, dim=-1)

        entry = BufferEntry(
            step_key=record.t + 1,
            predicted_tokens=predicted_tokens.detach(),
            logits=logits,
            features=last_hidden.detach(),
            meta=record.meta,
        )
        self.buffer.store(entry)
        logger.debug(
            "Stored prediction for t+1=%s | seq_len=%s",
            entry.step_key,
            input_ids.size(-1),
        )

    @staticmethod
    def _compute_uncertainty_gate(logits: torch.Tensor) -> torch.Tensor:
        """Simple entropy-based gate in range (0, 1]."""

        probs = nn.functional.softmax(logits, dim=-1)
        entropy = -(probs * nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1)
        max_entropy = math.log(logits.size(-1))
        gate = 1.0 - (entropy / max_entropy)
        return torch.clamp(gate.mean(), min=0.1, max=1.0)
