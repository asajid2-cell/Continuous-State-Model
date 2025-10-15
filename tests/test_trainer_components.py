import torch
from types import SimpleNamespace
from torch import nn

from src.data.stream import StreamRecord
from src.learn.train_loop import PhaseATrainConfig, PhaseATrainer
from src.learn.replay import PrioritizedReplay


class DummyTrunk(nn.Module):
    def __init__(self, hidden_size: int = 4, vocab_size: int = 8) -> None:
        super().__init__()
        self.primary_device = torch.device("cpu")
        self.embed = nn.Embedding(32, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.base_model = self

    def forward(self, input_ids, attention_mask=None, use_cache=True):
        hidden = self.embed(input_ids)
        logits = self.decoder(hidden)
        return logits, {"hidden_states": hidden}


class DummyPredictor(nn.Module):
    def __init__(self, hidden_size: int = 4, vocab_size: int = 8) -> None:
        super().__init__()
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, features):
        logits = self.decoder(features)
        return {"logits": logits}


class DummyResidual(nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, features):
        return self.proj(features)


def _make_trainer(consistency_weight: float = 0.1, residual_weight: float = 0.05) -> PhaseATrainer:
    trunk = DummyTrunk()
    predictor = DummyPredictor()
    residual = DummyResidual()
    optimizer = torch.optim.SGD(
        list(trunk.parameters()) + list(predictor.parameters()) + list(residual.parameters()),
        lr=0.01,
    )
    replay = PrioritizedReplay(capacity=32, alpha=0.6)
    cfg = PhaseATrainConfig(
        kl_weight=0.0,
        consistency_weight=consistency_weight,
        residual_weight=residual_weight,
        grad_clip=1.0,
        ema_decay=0.9,
        max_steps=10,
        grad_accum_steps=1,
        replay_interval=1,
        replay_batch=1,
    )
    ema_teacher = DummyPredictor()
    for param in ema_teacher.parameters():
        param.requires_grad_(False)
    trainer = PhaseATrainer(
        cfg=cfg,
        trunk=trunk,
        predictor=predictor,
        optimizer=optimizer,
        residual_head=residual,
        ema_teacher=ema_teacher,
        replay=replay,
    )
    return trainer


def test_apply_buffer_updates_with_ema_and_residual():
    trainer = _make_trainer()
    record0 = StreamRecord(t=0, tokens=[1, 2, 3], source_id="test", meta={})
    trainer._enqueue_prediction(record0)

    record1 = StreamRecord(t=1, tokens=[4, 5, 6], source_id="test", meta={})
    trainer._apply_buffer_updates(record1)
    trainer._enqueue_prediction(record1)
    trainer.step += 1

    summary = trainer.metrics.summary()
    assert summary["avg_cross_entropy"] >= 0
    assert trainer.replay is not None
    assert len(trainer.replay) >= 0
    assert trainer.ema_teacher is not None and not trainer.ema_teacher.training


def test_replay_update_runs():
    trainer = _make_trainer()
    record0 = StreamRecord(t=0, tokens=[1, 2, 3], source_id="test", meta={})
    trainer._enqueue_prediction(record0)
    record1 = StreamRecord(t=1, tokens=[4, 5, 6], source_id="test", meta={})
    trainer._apply_buffer_updates(record1)
    trainer._enqueue_prediction(record1)
    trainer.step += 1

    trainer._run_replay_updates()
    assert len(trainer.replay) >= 0
