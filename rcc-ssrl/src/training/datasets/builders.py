"""DataLoader builders for SSL and SL training."""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import webdataset as wds

from .labels import make_class_to_id_norm
from .transforms import (
    collate_multicrop,
    collate_single_image,
    collate_supervised,
    collate_two_views,
    multicrop_transform,
    single_image_transform,
    sl_eval_transforms,
    sl_train_transforms,
    two_view_transform,
)

__all__ = [
    "build_ssl_loader",
    "build_sl_loader",
    "build_sl_loaders",
    "build_ssl_loader_from_cfg",
]


def _list_shards(dir_or_glob: str) -> List[str]:
    if os.path.isdir(dir_or_glob):
        return sorted(glob.glob(os.path.join(dir_or_glob, "shard-*.tar")))
    return sorted(glob.glob(dir_or_glob))


def _make_webdataset(shards: List[str], shuffle_shards: int, shuffle_samples: int) -> wds.WebDataset:
    ds = wds.WebDataset(
        shards,
        shardshuffle=shuffle_shards,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
    )
    return ds.shuffle(shuffle_samples).decode("pil").to_tuple("img.jpg;jpg;jpeg;png", "meta.json;json")


def _limit_epoch(ds: Iterable, samples_per_epoch: Optional[int]):
    if not samples_per_epoch:
        return ds

    class _EpochLimiter:
        def __init__(self, base: Iterable, target: int):
            self.base = base
            self.n = int(target)

        def __iter__(self):
            count = 0
            for sample in self.base:
                yield sample
                count += 1
                if count >= self.n:
                    break

        def __len__(self):
            return self.n

        def __getitem__(self, idx: int):
            if idx < 0 or idx >= self.n:
                raise IndexError("Index out of range")
            for i, sample in enumerate(self.base):
                if i == idx:
                    return sample
            raise IndexError("Index out of range")

    return _EpochLimiter(ds, samples_per_epoch)


def build_ssl_loader(data_cfg: Dict[str, Any], model_cfg: Dict[str, Any], split: str = "train"
                     ) -> torch.utils.data.DataLoader:
    """Construct the SSL dataloader based on the configured model family."""
    if "webdataset" not in data_cfg:
        raise KeyError("Missing data.webdataset configuration.")

    wds_cfg = data_cfg["webdataset"]
    shards = _list_shards(wds_cfg[f"{split}_dir"])
    dataset = _make_webdataset(shards, wds_cfg["shuffle_shards"], wds_cfg["shuffle_samples"])

    img_size = int(data_cfg.get("img_size", 224))
    mode = model_cfg["ssl"]["name"].lower()

    if mode in ("moco_v3", "ibot"):
        transform = two_view_transform(img_size)
        dataset = dataset.map_tuple(transform, lambda meta: meta)
        dataset = _limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))
        collate_fn = collate_two_views
    elif mode == "dino_v3":
        dino_cfg = data_cfg.get("dino_v3", {})
        transform = multicrop_transform(
            int(dino_cfg.get("global_size", img_size)),
            int(dino_cfg.get("local_size", 96)),
            int(dino_cfg.get("n_local", 6)),
        )
        dataset = dataset.map_tuple(transform, lambda meta: meta)
        dataset = _limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))
        collate_fn = collate_multicrop
    elif mode == "i_jepa":
        transform = single_image_transform(img_size)
        dataset = dataset.map_tuple(transform, lambda meta: meta)
        dataset = _limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))
        collate_fn = collate_single_image
    else:
        raise ValueError(f"Unsupported SSL model family '{mode}'.")

    num_workers = wds_cfg["num_workers"]
    persistent = bool(num_workers > 0)
    dl_kwargs: Dict[str, Any] = {
        "batch_size": wds_cfg["batch_size_ssl"],
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": persistent,
        "collate_fn": collate_fn,
    }
    if torch.cuda.is_available():
        dl_kwargs["pin_memory_device"] = "cuda"
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = int(wds_cfg.get("prefetch_factor", 4))
    return torch.utils.data.DataLoader(dataset, **dl_kwargs)


def build_sl_loader(data_cfg: Dict[str, Any], split: str = "train", override_transforms=None
                    ) -> torch.utils.data.DataLoader:
    if "webdataset" not in data_cfg:
        raise KeyError("Missing data.webdataset configuration.")

    wds_cfg = data_cfg["webdataset"]
    shards = _list_shards(wds_cfg[f"{split}_dir"])
    dataset = _make_webdataset(shards, wds_cfg["shuffle_shards"], wds_cfg["shuffle_samples"])

    img_size = int(data_cfg.get("img_size", 224))
    transforms_fn = override_transforms or (sl_train_transforms(img_size) if split == "train" else sl_eval_transforms(img_size))
    dataset = dataset.map_tuple(transforms_fn, lambda meta: meta)
    dataset = _limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))

    class_map = make_class_to_id_norm(wds_cfg["class_to_id"])
    num_workers = wds_cfg["num_workers"]
    persistent = bool(num_workers > 0)
    dl_kwargs: Dict[str, Any] = {
        "batch_size": wds_cfg["batch_size_sl"],
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": persistent,
        "collate_fn": lambda batch: collate_supervised(batch, class_map),
    }
    if torch.cuda.is_available():
        dl_kwargs["pin_memory_device"] = "cuda"
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = int(wds_cfg.get("prefetch_factor", 4))
    return torch.utils.data.DataLoader(dataset, **dl_kwargs)


def build_sl_loaders(cfg: Dict[str, Any], override_transforms=None
                     ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_loader = build_sl_loader(cfg["data"], split="train", override_transforms=override_transforms)
    val_loader = build_sl_loader(cfg["data"], split="val", override_transforms=override_transforms)
    return train_loader, val_loader


def build_ssl_loader_from_cfg(cfg: Dict[str, Any], split: str = "train") -> torch.utils.data.DataLoader:
    return build_ssl_loader(cfg["data"], cfg["model"], split=split)
