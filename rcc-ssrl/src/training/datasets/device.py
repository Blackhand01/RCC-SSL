"""Device selection utilities."""
from __future__ import annotations

import os
import time

import torch

__all__ = ["device_from_env"]


def device_from_env(allow_cpu: bool = False) -> torch.device:
    """
    Resolve the preferred torch.device by respecting CUDA availability and the
    configuration/env escape hatches for CPU-only dry runs.
    """
    wait_secs = float(os.environ.get("DEVICE_WAIT_FOR_CUDA", 10))
    if not torch.cuda.is_available():
        deadline = time.time() + max(0.0, wait_secs)
        while time.time() < deadline:
            time.sleep(0.2)
            if torch.cuda.is_available():
                break

    if torch.cuda.is_available():
        # torchrun/SLURM passano il rank locale: rispettiamolo quando presente.
        lr = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID")
        if lr is not None:
            try:
                return torch.device("cuda", int(lr))
            except (TypeError, ValueError):
                pass
        return torch.device("cuda", 0)
    if allow_cpu or os.environ.get("ALLOW_CPU", "0") == "1":
        return torch.device("cpu")
    raise RuntimeError("No GPU visible (enable experiment.allow_cpu or set ALLOW_CPU=1).")
