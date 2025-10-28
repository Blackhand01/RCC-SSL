"""Device selection utilities."""
from __future__ import annotations

import os
import time
from typing import List

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
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        lr_str = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID")

        # Preferisci sempre LOCAL_RANK (impostato da torchrun)
        if lr_str is not None:
            try:
                lr = int(lr_str)
                tokens: List[str] = [t.strip() for t in cvd.split(",") if t.strip()] if cvd else []
                if tokens:
                    # Mappa per posizione dentro CUDA_VISIBLE_DEVICES
                    try:
                        mapped = int(tokens[lr])
                        return torch.device("cuda", mapped)
                    except (ValueError, IndexError):
                        # token non numerici (es. MIG) o lista corta -> usa indice logico
                        return torch.device("cuda", lr % max(1, torch.cuda.device_count()))
                # niente CVD: usa indice logico
                return torch.device("cuda", lr % max(1, torch.cuda.device_count()))
            except (TypeError, ValueError):
                pass

        # Nessun LOCAL_RANK: prendi il primo token numerico da CVD, altrimenti 0
        if cvd:
            for tok in cvd.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    return torch.device("cuda", int(tok))
                except ValueError:
                    continue
        return torch.device("cuda", 0)
    if allow_cpu or os.environ.get("ALLOW_CPU", "0") == "1":
        return torch.device("cpu")
    raise RuntimeError("No GPU visible (enable experiment.allow_cpu or set ALLOW_CPU=1).")
