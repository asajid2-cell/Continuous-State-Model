# Continuous-State Model (Delta-Driven Dual Stream)

This project explores a lean continual-learning architecture that lets a language model learn from the gap between what it predicts and what actually happens. The codebase is tuned for an RTX 3070 Ti (8 GB VRAM) but automatically falls back to CPU if CUDA or bitsandbytes is unavailable.

## What you get
- **Streaming TinyLlama trunk** with LoRA adapters and optional EMA teacher for stability.
- **Two-tick delta learning loop** that buffers predictions, compares them with delayed reality, and applies gated updates on the fly.
- **Chat CLI** for quick qualitative checks alongside the online training script.
- **Blueprint reference** (`dual_stream_delta_learning_plan_8gb.docx`) that details the architecture roadmap.

## Hardware & prerequisites
- Windows 11 or WSL2 with Python 3.10+ (Python 3.12 works as well).
- NVIDIA RTX 3070 Ti (8 GB). CPU-only runs are supported but slower.
- Hugging Face account with access to TinyLlama weights.

## Quick start (PowerShell in VS Code)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python scripts\setup_env.py      # installs deps, retries with --user if needed, prints CUDA status
```

Log in to Hugging Face (once per machine):
```powershell
huggingface-cli login
```
Or load a saved token:
```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '=') { $name,$value = $_.Split('='); Set-Item env:$name $value }
}
```

## Download the base model
```powershell
python scripts\download_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --log-level INFO
```
The script caches weights under `models/`. Re-run with `--log-level DEBUG` for detailed network traces.

## Prepare a stream
- Use `data/stream.jsonl` as the template (one JSON object per line with `t`, `text`, optional metadata).
- Update `configs/base.yaml:data.stream_path` if you keep data elsewhere.

## Run the Phase A training loop
```powershell
python scripts\run_stream_train.py --config configs/base.yaml --log-level DEBUG
```
- INFO logs confirm device, hidden size, LoRA parameter counts, and rolling cross-entropy.
- If CUDA isn’t detected, the script logs a fallback to CPU/float32 automatically.
- Monitor VRAM usage in a second terminal: `nvidia-smi -l 1`.

## Talk to the model
Single prompt:
```powershell
python scripts\chat.py --config configs/base.yaml --prompt "Summarize the delta learner." --max-new-tokens 128
```
Interactive session:
```powershell
python scripts\chat.py --config configs/base.yaml --interactive --log-level DEBUG
```

## Current roadmap
1. **Residual & consistency losses** – residual head and EMA teacher now scaffolded in code; tuning weights and tests are next.
2. **Replay buffer** – CPU-backed prioritized replay utilities exist; integrate them into the loop to reinforce surprising samples.
3. **Metrics & tests** – add unit tests for buffer/EMA flows and light dashboards (TensorBoard or VS Code notebooks).

## Troubleshooting
- **Permission denied during pip install** – `scripts/setup_env.py` reruns installs with `--user` automatically.
- **bitsandbytes import warning** – expected on Windows; the model loads in FP16/FP32 mode instead.
- **Hugging Face cache symlink warning** – enable Windows Developer Mode or ignore (caching still works).

Questions or contributions? Open an issue or PR. The goal is to refine a continuous-state learner that improves while it runs—no giant retrains required.
