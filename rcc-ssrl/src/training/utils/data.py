# utils/data.py
from __future__ import annotations

import glob
import os
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torchvision import transforms

__all__ = [
    "device_from_env",
    "build_ssl_loader",
    "build_sl_loader",
    "build_sl_loaders",
    "build_ssl_loader_from_cfg",
    "class_labels_from_cfg",
]


def device_from_env(allow_cpu: bool = False) -> torch.device:
    """
    Resolve the preferred torch.device by respecting CUDA availability and the
    configuration/env escape hatches for CPU-only dry runs.
    """
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if allow_cpu or os.environ.get("ALLOW_CPU", "0") == "1":
        return torch.device("cpu")
    raise RuntimeError("No GPU visible (enable experiment.allow_cpu or set ALLOW_CPU=1).")


# ---------------------------------------------------------------------------
# Label normalisation and class mapping helpers
# ---------------------------------------------------------------------------
_ALIAS_TABLE = {
    "CCRCC": ["CCRCC", "CC_RCC", "CLEAR_CELL_RCC", "CLEARCELL"],
    "PRCC": ["PRCC", "P_RCC", "PAPILLARY_RCC"],
    "CHROMO": ["CHROMO", "CHROMOPHOBE", "CHR", "CHRCC"],
    "ONCO": ["ONCO", "ONCOCYTOMA"],
    "NOT_TUMOR": ["NOT_TUMOR", "NON_TUMOR", "NONTUMOR", "NORMAL", "BACKGROUND"],
}


def _norm_label(s: Any) -> str:
    return str(s).strip().upper().replace(" ", "_").replace("-", "_")


def _make_class_to_id_norm(user_map: Dict[str, int]) -> Dict[str, int]:
    """
    Build a normalised class mapping, adding the canonical alias table when the
    canonical class is present in the user-provided YAML.
    """
    base = {_norm_label(k): v for k, v in user_map.items()}
    out = dict(base)
    for canon, synonyms in _ALIAS_TABLE.items():
        if canon in base:
            for synonym in synonyms:
                out[_norm_label(synonym)] = base[canon]
    return out


# ---------------------------------------------------------------------------
# Image coercion helpers
# ---------------------------------------------------------------------------
def _coerce_pil_rgb(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB") if img.mode != "RGB" else img
    if isinstance(img, (bytes, bytearray)):
        return Image.open(BytesIO(img)).convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return Image.fromarray(img).convert("RGB")
    try:
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return Image.fromarray(arr).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"Unsupported image payload {type(img)}") from exc


def _pil_to_uint8_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
    tensor = torch.tensor(arr, dtype=torch.uint8)
    tensor = tensor.permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
    return tensor


# ---------------------------------------------------------------------------
# WebDataset helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# SSL augmentations and collates
# ---------------------------------------------------------------------------
def _two_views_transforms(size: int, jitter: float = 0.4) -> Callable[[Image.Image], Tuple[torch.Tensor, torch.Tensor]]:
    aug = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(jitter, jitter, 0.2, 0.1),
        transforms.Lambda(lambda im: _pil_to_uint8_tensor(_coerce_pil_rgb(im))),
    ])
    return lambda img: (aug(img), aug(img))


def _multicrop_transforms(global_size: int, local_size: int, n_local: int, jitter: float = 0.4
                          ) -> Callable[[Image.Image], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    global_aug = transforms.Compose([
        transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(jitter, jitter, 0.2, 0.1),
        transforms.Lambda(lambda im: _pil_to_uint8_tensor(_coerce_pil_rgb(im))),
    ])
    local_aug = transforms.Compose([
        transforms.RandomResizedCrop(local_size, scale=(0.05, 0.4)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(jitter, jitter, 0.2, 0.1),
        transforms.Lambda(lambda im: _pil_to_uint8_tensor(_coerce_pil_rgb(im))),
    ])
    return lambda img: ([global_aug(img), global_aug(img)], [local_aug(img) for _ in range(n_local)])


def _single_image_transforms(size: int) -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda im: _pil_to_uint8_tensor(_coerce_pil_rgb(im))),
    ])


def _collate_two_views(batch):
    x1 = torch.stack([sample[0][0] for sample in batch], 0)
    x2 = torch.stack([sample[0][1] for sample in batch], 0)
    meta = [sample[1] for sample in batch]
    return {"images": [x1, x2], "meta": meta}


def _collate_multicrop(batch):
    g_all = [g for sample in batch for g in sample[0][0]]
    l_all = [l for sample in batch for l in sample[0][1]]
    meta = [sample[1] for sample in batch]
    return {"images": [torch.stack(g_all, 0), torch.stack(l_all, 0)], "meta": meta}


def _collate_single_image(batch):
    images = torch.stack([sample[0] for sample in batch], 0)
    meta = [sample[1] for sample in batch]
    return {"images": [images], "meta": meta}


# ---------------------------------------------------------------------------
# SL transforms and collate
# ---------------------------------------------------------------------------
def _sl_train_transforms(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda im: _pil_to_uint8_tensor(_coerce_pil_rgb(im))),
    ])


def _sl_eval_transforms(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.Lambda(lambda im: _pil_to_uint8_tensor(_coerce_pil_rgb(im))),
    ])


def _collate_sl(batch, class_to_id: Dict[str, int]):
    images: List[torch.Tensor] = []
    labels: List[int] = []
    meta: List[Dict[str, Any]] = []
    missing: List[str] = []
    for image, info in batch:
        images.append(_pil_to_uint8_tensor(_coerce_pil_rgb(image)) if isinstance(image, Image.Image) else image)
        label_str = _norm_label(info.get("class_label", ""))
        if label_str not in class_to_id:
            missing.append(label_str)
        labels.append(class_to_id.get(label_str, -1))
        meta.append(info)
    if missing:
        raise KeyError(
            f"[SL] Labels not present in class_to_id after normalisation: {sorted(set(missing))}. "
            "Ensure data.webdataset.class_to_id is aligned with the normalisation scheme."
        )
    inputs = torch.stack(images, 0)
    targets = torch.tensor(labels, dtype=torch.long)
    return {"inputs": inputs, "targets": targets, "meta": meta}


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def build_ssl_loader(dc: Dict[str, Any], mc: Dict[str, Any], split: str = "train") -> torch.utils.data.DataLoader:
    """
    Build a WebDataset-based self-supervised loader using the augmentation and
    collate strategy dictated by the SSL model family.
    """
    if "webdataset" not in dc:
        raise KeyError("Missing data.webdataset configuration.")
    wds_cfg = dc["webdataset"]
    shards = _list_shards(wds_cfg[f"{split}_dir"])
    dataset = _make_webdataset(shards, wds_cfg["shuffle_shards"], wds_cfg["shuffle_samples"])
    img_size = int(dc.get("img_size", 224))
    mode = mc["ssl"]["name"].lower()

    if mode in ("moco_v3", "ibot"):
        aug = _two_views_transforms(img_size)
        dataset = dataset.map_tuple(lambda im: aug(_coerce_pil_rgb(im)), lambda meta: meta)
        dataset = _limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))
        collate_fn = _collate_two_views
        batch_size = wds_cfg["batch_size_ssl"]
    elif mode == "dino_v3":
        dino_cfg = dc.get("dino_v3", {})
        aug = _multicrop_transforms(
            int(dino_cfg.get("global_size", img_size)),
            int(dino_cfg.get("local_size", 96)),
            int(dino_cfg.get("n_local", 6)),
        )
        dataset = dataset.map_tuple(lambda im: aug(_coerce_pil_rgb(im)), lambda meta: meta)
        dataset = _limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))
        collate_fn = _collate_multicrop
        batch_size = wds_cfg["batch_size_ssl"]
    elif mode == "i_jepa":
        aug = _single_image_transforms(img_size)
        dataset = dataset.map_tuple(lambda im: aug(_coerce_pil_rgb(im)), lambda meta: meta)
        dataset = _limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))
        collate_fn = _collate_single_image
        batch_size = wds_cfg["batch_size_ssl"]
    else:
        raise ValueError(f"Unsupported SSL model family '{mode}'.")

    num_workers = wds_cfg["num_workers"]
    persistent = bool(wds_cfg.get("num_workers", 0) > 0)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        collate_fn=collate_fn,
    )


def build_sl_loader(dc: Dict[str, Any], split: str = "train",
                    override_transforms: Optional[transforms.Compose] = None) -> torch.utils.data.DataLoader:
    if "webdataset" not in dc:
        raise KeyError("Missing data.webdataset configuration.")
    wds_cfg = dc["webdataset"]
    shards = _list_shards(wds_cfg[f"{split}_dir"])
    dataset = _make_webdataset(shards, wds_cfg["shuffle_shards"], wds_cfg["shuffle_samples"])

    img_size = int(dc.get("img_size", 224))
    transforms_fn = override_transforms or (_sl_train_transforms(img_size) if split == "train" else _sl_eval_transforms(img_size))
    dataset = dataset.map_tuple(transforms_fn, lambda meta: meta)
    dataset = _limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))

    class_map = _make_class_to_id_norm(wds_cfg["class_to_id"])

    num_workers = wds_cfg["num_workers"]
    persistent = bool(wds_cfg.get("num_workers", 0) > 0)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=wds_cfg["batch_size_sl"],
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        collate_fn=lambda batch: _collate_sl(batch, class_map),
    )


# Convenience wrappers for the previous config-level API --------------------
def build_ssl_loader_from_cfg(cfg: Dict[str, Any], split: str = "train") -> torch.utils.data.DataLoader:
    return build_ssl_loader(cfg["data"], cfg["model"], split=split)


def build_sl_loaders(cfg: Dict[str, Any], override_transforms: Optional[transforms.Compose] = None
                     ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_loader = build_sl_loader(cfg["data"], split="train", override_transforms=override_transforms)
    val_loader = build_sl_loader(cfg["data"], split="val", override_transforms=override_transforms)
    return train_loader, val_loader


def class_labels_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    mapping = (cfg.get("data", {}).get("webdataset", {}) or {}).get("class_to_id", {})
    if not mapping:
        return []
    inverse = {idx: name for name, idx in mapping.items()}
    return [inverse[idx] for idx in sorted(inverse)]
