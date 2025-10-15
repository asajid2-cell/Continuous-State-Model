"""
Streaming data utilities for the dual-stream delta learner.

Phase A goal: provide a deterministic iterator that yields tokenized
chunks alongside timestamps so the two-tick buffer can align predictions
with delayed ground truth entries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class StreamRecord:
    """Container for a single streaming example."""

    t: int
    tokens: List[int]
    source_id: str
    meta: Dict[str, str]


class JsonlStreamLoader:
    """
    Streaming JSONL loader with tokenizer integration.

    Reads newline-delimited JSON objects with fields:
      * t: optional timestep (int)
      * text: raw text to tokenize
      * source_id: identifier for diagnostics
      * meta: arbitrary metadata dict
    """

    def __init__(self, path: Path, tokenizer: PreTrainedTokenizerBase, sequence_length: int) -> None:
        self._path = path
        self._tokenizer = tokenizer
        self._sequence_length = sequence_length
        self._counter = 0

    def __iter__(self) -> Iterator[StreamRecord]:
        logger.info("Streaming JSONL from %s", self._path)
        with self._path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    logger.debug("Skipping empty line at %s:%s", self._path, lineno)
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.error("Failed to parse JSON at %s:%s (%s)", self._path, lineno, exc)
                    raise
                text = payload.get("text", "")
                tokenized = self._tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self._sequence_length,
                )
                tokens = tokenized["input_ids"]
                logger.debug("Line %s tokenized to %s tokens", lineno, len(tokens))

                t_val = payload.get("t")
                if t_val is None:
                    t_val = self._counter
                self._counter = max(self._counter + 1, int(t_val) + 1)

                meta = payload.get("meta") or {}
                if not isinstance(meta, dict):
                    logger.error("Meta field must be dict at %s:%s", self._path, lineno)
                    raise ValueError(f"Expected 'meta' to be a dict, received {type(meta)}")

                record = StreamRecord(
                    t=int(t_val),
                    tokens=tokens,
                    source_id=payload.get("source_id", "unknown"),
                    meta=meta,
                )
                logger.debug("Yielding record t=%s source=%s", record.t, record.source_id)
                yield record


def stream_loader_factory(path: Optional[str], tokenizer: PreTrainedTokenizerBase, sequence_length: int) -> Iterable[StreamRecord]:
    """
    Helper that returns an iterable over `StreamRecord` instances.

    Providing a factory keeps higher-level code decoupled from the underlying
    storage backend.
    """

    if not path:
        raise ValueError("Stream path must be provided.")

    loader = JsonlStreamLoader(Path(path), tokenizer=tokenizer, sequence_length=sequence_length)
    return loader
