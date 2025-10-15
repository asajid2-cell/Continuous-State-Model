"""
Interactive and one-off chat utility for the delta-driven dual-stream project.
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
from typing import List, Dict

from src.config import load_app_config
from src.model.factory import create_trunk

logger = logging.getLogger("delta_stream.chat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the base Llama model.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--prompt", type=str, default=None, help="Single-turn prompt for generation.")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="System message for chat template.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive chat session.")
    return parser.parse_args()


def build_initial_messages(system_prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    return messages


def run_single_turn(trunk, args: argparse.Namespace) -> None:
    messages = build_initial_messages(args.system)
    prompt = args.prompt or ""
    messages.append({"role": "user", "content": prompt})
    logger.info("Running single-turn generation")
    reply = trunk.chat(
        messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(reply)


def run_interactive(trunk, args: argparse.Namespace) -> None:
    messages = build_initial_messages(args.system)
    print("Interactive chat session started. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("user> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        reply = trunk.chat(
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"assistant> {reply}")
        messages.append({"role": "assistant", "content": reply})
        logger.debug("Conversation length: %s messages", len(messages))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    app_cfg = load_app_config(args.config)
    logger.info("Loading trunk for chat interface")
    trunk = create_trunk(app_cfg.model)

    if args.interactive or args.prompt is None:
        run_interactive(trunk, args)
    else:
        run_single_turn(trunk, args)


if __name__ == "__main__":
    main()


