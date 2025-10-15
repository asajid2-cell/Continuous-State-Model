"""
Utility for downloading and preparing the base Llama checkpoint.

The script relies on `huggingface_hub` so users can authenticate with tokens
when required. Phase A only needs to cache weights locally; quantization and
adapter injection will happen inside the model wrapper.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from huggingface_hub import snapshot_download

logger = logging.getLogger("delta_stream.download")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download base LLM weights for delta learner.")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--local-dir", type=Path, default=None)
    parser.add_argument("--allow-pattern", type=str, action="append", default=None,
                        help="Optional glob patterns to restrict downloaded files.")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (or rely on env var).")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    target_dir = args.local_dir or args.cache_dir / args.model.replace("/", "_")
    logger.info("Downloading %s (revision=%s) to %s", args.model, args.revision, target_dir)

    snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        allow_patterns=args.allow_pattern,
        token=args.token,
    )
    logger.info("Download complete. Cached at %s", target_dir)


if __name__ == "__main__":
    main()

