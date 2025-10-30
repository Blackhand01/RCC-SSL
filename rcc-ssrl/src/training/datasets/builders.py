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
from ..data.webdataset import list_shards, make_wds, limit_epoch, dataloader_args

__all__ = [
    "build_ssl_loader",
    "build_sl_loader",
    "build_sl_loaders",
    "build_ssl_loader_from_cfg",
]


def build_ssl_loader(data_cfg: Dict[str, Any], model_cfg: Dict[str, Any], split: str = "train"
                     ) -> torch.utils.data.DataLoader:
    """Construct the SSL dataloader based on the configured model family."""
    if "webdataset" not in data_cfg:
        raise KeyError("Missing data.webdataset configuration.")

    wds_cfg = data_cfg["webdataset"]
    shards = list_shards(wds_cfg[f"{split}_dir"])
    dataset = make_wds(shards, wds_cfg["shuffle_shards"], wds_cfg["shuffle_samples"])

    img_size = int(data_cfg.get("img_size", 224))
    mode = model_cfg["ssl"]["name"].lower()

    if mode in ("moco_v3", "ibot"):
        transform = two_view_transform(img_size)
        dataset = dataset.map_tuple(transform, lambda meta: meta)
        dataset = limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))
        collate_fn = collate_two_views
    elif mode == "dino_v3":
        dino_cfg = data_cfg.get("dino_v3", {})
        transform = multicrop_transform(
            int(dino_cfg.get("global_size", img_size)),
            int(dino_cfg.get("local_size", 96)),
            int(dino_cfg.get("n_local", 6)),
            float(dino_cfg.get("jitter", 0.4)),
            global_scale=tuple(dino_cfg.get("global_scale", (0.14, 1.0))),
            local_scale=tuple(dino_cfg.get("local_scale", (0.05, 0.14))),
            blur_prob=float(dino_cfg.get("blur_prob", 0.5)),
            solarize_prob=float(dino_cfg.get("solarize_prob", 0.0)),
        )
        dataset = dataset.map_tuple(transform, lambda meta: meta)
        dataset = limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))
        collate_fn = collate_multicrop
    elif mode == "i_jepa":
        transform = single_image_transform(img_size)
        dataset = dataset.map_tuple(transform, lambda meta: meta)
        dataset = limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))
        collate_fn = collate_single_image
    else:
        raise ValueError(f"Unsupported SSL model family '{mode}'.")

    dl_kwargs = dataloader_args(
        pin_cuda=True,
        batch_size=wds_cfg["batch_size_ssl"],
        num_workers=wds_cfg["num_workers"],
        prefetch_factor=int(wds_cfg.get("prefetch_factor", 4)),
        collate_fn=collate_fn,
    )
    return torch.utils.data.DataLoader(dataset, **dl_kwargs)


def build_sl_loader(data_cfg: Dict[str, Any], split: str = "train", override_transforms=None
                    ) -> torch.utils.data.DataLoader:
    if "webdataset" not in data_cfg:
        raise KeyError("Missing data.webdataset configuration.")

    wds_cfg = data_cfg["webdataset"]
    shards = list_shards(wds_cfg[f"{split}_dir"])
    dataset = make_wds(shards, wds_cfg["shuffle_shards"], wds_cfg["shuffle_samples"])

    img_size = int(data_cfg.get("img_size", 224))
    transforms_fn = override_transforms or (sl_train_transforms(img_size) if split == "train" else sl_eval_transforms(img_size))
    dataset = dataset.map_tuple(transforms_fn, lambda meta: meta)
    dataset = limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))

    class_map = make_class_to_id_norm(wds_cfg["class_to_id"])
    dl_kwargs = dataloader_args(
        pin_cuda=True,
        batch_size=wds_cfg["batch_size_sl"],
        num_workers=wds_cfg["num_workers"],
        prefetch_factor=int(wds_cfg.get("prefetch_factor", 4)),
        collate_fn=lambda batch: collate_supervised(batch, class_map),
    )
    return torch.utils.data.DataLoader(dataset, **dl_kwargs)


def build_sl_loaders(cfg: Dict[str, Any], override_transforms=None
                     ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_loader = build_sl_loader(cfg["data"], split="train", override_transforms=override_transforms)
    val_loader = build_sl_loader(cfg["data"], split="val", override_transforms=override_transforms)
    return train_loader, val_loader


def build_ssl_loader_from_cfg(cfg: Dict[str, Any], split: str = "train") -> torch.utils.data.DataLoader:
    return build_ssl_loader(cfg["data"], cfg["model"], split=split)
