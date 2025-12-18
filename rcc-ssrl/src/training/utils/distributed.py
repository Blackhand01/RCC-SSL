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

# --- Shim per compatibilità DINOv3 ---
# DINOv3 usa i "subgroup" per training su migliaia di GPU. 
# Per il tuo caso, il subgroup è uguale al world group (tutte le GPU lavorano insieme).

def get_process_subgroup():
    """Restituisce il gruppo di default invece di un sottogruppo specifico."""
    return None  # PyTorch usa il gruppo di default se questo è None

def get_subgroup_size() -> int:
    """La dimensione del sottogruppo è la dimensione totale del mondo."""
    return get_world_size()

def get_subgroup_rank() -> int:
    """Il rank nel sottogruppo è il rank globale."""
    return get_rank()