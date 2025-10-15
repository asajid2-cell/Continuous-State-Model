from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REQS_FILE = Path("requirements.txt")


def run_pip(args: list[str]) -> subprocess.CompletedProcess[int]:
    print("Running:", " ".join(args))
    return subprocess.run(args)


def install_requirements() -> int:
    if not REQS_FILE.exists():
        print(f"requirements file not found: {REQS_FILE}")
        return 1

    print("Installing Python dependencies...")
    base_cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQS_FILE)]
    proc = run_pip(base_cmd)

    if proc.returncode == 0:
        return 0

    print("pip install failed (possible permissions issue). Retrying with --user...")
    user_cmd = base_cmd + ["--user"]
    proc_user = run_pip(user_cmd)
    if proc_user.returncode != 0:
        print("pip install still failed", file=sys.stderr)
    return proc_user.returncode


def check_cuda() -> dict[str, str | bool]:
    info: dict[str, str | bool] = {
        "torch_installed": False,
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
    }
    try:
        import torch

        info["torch_installed"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def main() -> None:
    code = install_requirements()
    if code != 0:
        print("Skipping CUDA check because installation failed.")
        sys.exit(code)

    info = check_cuda()
    print("CUDA diagnostics:")
    print(json.dumps(info, indent=2))

    if not info["torch_installed"]:
        print("PyTorch is not installed; check pip output.")
    elif info["cuda_available"]:
        print("CUDA is available. You're ready to run GPU workloads.")
    else:
        print("CUDA not detected. Training will run on CPU.")


if __name__ == "__main__":
    main()
