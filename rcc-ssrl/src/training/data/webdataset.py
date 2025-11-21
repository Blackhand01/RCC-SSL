# src/training/data/webdataset.py
from __future__ import annotations
from typing import Iterable, List, Optional, Any, Dict
import os, glob, itertools
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

def list_shards(dir_or_glob: str) -> List[str]:
    if os.path.isdir(dir_or_glob):
        return sorted(glob.glob(os.path.join(dir_or_glob, "shard-*.tar")))
    return sorted(glob.glob(dir_or_glob))

def make_wds(shards: List[str], shuffle_shards: int, shuffle_samples: int) -> wds.WebDataset:
    ds = wds.WebDataset(
        shards,
        shardshuffle=shuffle_shards,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
        empty_check=False,
    )
    return ds.shuffle(shuffle_samples).decode("pil").to_tuple("img.jpg;jpg;jpeg;png", "meta.json;json")

def limit_epoch(ds: Iterable, samples_per_epoch: Optional[int]):
    if not samples_per_epoch:
        return ds
    class _Limiter(IterableDataset):
        def __init__(self, base: Iterable, n: int): self.base, self.n = base, int(n)
        def __iter__(self):
            yield from itertools.islice(iter(self.base), self.n)
        def __len__(self): return self.n
    return _Limiter(ds, samples_per_epoch)

def dataloader_args(pin_cuda: bool, batch_size: int, num_workers: int, prefetch_factor: int, collate_fn):
    args: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": bool(num_workers > 0),
        "collate_fn": collate_fn,
    }
    # NOTE: avoid setting 'pin_memory_device' to silence deprecation warnings on some torch builds.
    # CUDA pinning remains enabled via 'pin_memory=True'.
    if num_workers > 0:
        args["prefetch_factor"] = int(prefetch_factor)
    return args
