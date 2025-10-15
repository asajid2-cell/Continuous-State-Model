"""
Placeholder script for evaluating canary prompts against the adaptive model.

Phase A scope: define CLI and reporting skeleton so the evaluation pipeline can
integrate smoothly once the training loop produces checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate canary prompts for drift detection.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--canary-path", type=Path, default=Path("data/canaries.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Canary evaluation placeholder")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  canaries : {args.canary_path}")
    # TODO: load model, run prompts, compute metrics defined in metrics.py.


if __name__ == "__main__":
    main()


