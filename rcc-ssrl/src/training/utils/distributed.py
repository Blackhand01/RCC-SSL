# src/training/utils/distributed.py
import os
import torch
import torch.distributed as dist

def is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_enabled() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_enabled() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def get_default_process_group():
    return dist.group.WORLD if is_enabled() else None

# --- Shim for DINOv3 compatibility ---
# DINOv3 uses "subgroup" for training on thousands of GPUs.
# In your case, the subgroup equals the world group (all GPUs work together).

def get_process_subgroup():
    """Return the default group instead of a specific subgroup."""
    return None  # PyTorch uses the default group if this is None

def get_subgroup_size() -> int:
    """Subgroup size is the total world size."""
    return get_world_size()

def get_subgroup_rank() -> int:
    """Rank in the subgroup is the global rank."""
    return get_rank()