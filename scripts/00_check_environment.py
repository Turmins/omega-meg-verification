"""Report interpreter, package and accelerator availability."""

from __future__ import annotations

import importlib.util
import platform
import sys

from _bootstrap import ROOT


def main() -> int:
    packages = ["numpy", "scipy", "pandas", "sklearn", "torch", "matplotlib", "yaml", "threadpoolctl", "pytest"]
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    for name in packages:
        print(f"{name}: {'available' if importlib.util.find_spec(name) else 'missing'}")
    try:
        import torch
        print(f"cuda: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("cuda: unavailable because torch is missing")
    print(f"extension: {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
