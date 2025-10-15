"""
Lightweight tests for the JSONL stream loader.

Uses a dummy tokenizer so the test runs without downloading large assets.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from transformers import PreTrainedTokenizerBase

from src.data.stream import JsonlStreamLoader


class DummyTokenizer:
    """Minimal callable that mimics the Hugging Face tokenizer interface."""

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = True,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        tokens = [len(token) for token in text.strip().split()]
        if max_length is not None:
            tokens = tokens[:max_length]
        return {"input_ids": tokens}


def test_jsonl_stream_loader_infers_timesteps(tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/stream_sample.jsonl")
    loader = JsonlStreamLoader(
        fixture_path,
        tokenizer=cast(PreTrainedTokenizerBase, DummyTokenizer()),
        sequence_length=16,
    )

    records = list(loader)
    assert len(records) == 2
    assert records[0].t == 0
    assert records[1].t == 1  # second line lacked explicit timestep
    assert records[0].source_id == "synthetic"
    assert records[0].meta["lang"] == "en"
    assert records[0].tokens  # non-empty token list

