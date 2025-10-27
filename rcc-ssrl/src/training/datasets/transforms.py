"""Augmentations and collate utilities used by SSL/SL loaders."""
from __future__ import annotations

from io import BytesIO
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .labels import normalise_label


__all__ = [
    "coerce_to_pil_rgb",
    "pil_to_unit_tensor",
    "two_view_transform",
    "multicrop_transform",
    "single_image_transform",
    "sl_train_transforms",
    "sl_eval_transforms",
    "collate_two_views",
    "collate_multicrop",
    "collate_single_image",
    "collate_supervised",
]


def coerce_to_pil_rgb(img: Any) -> Image.Image:
    """Coerce the given payload to a PIL.Image in RGB space."""
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
    except Exception as exc:  # pragma: no cover - defensive path
        raise TypeError(f"Unsupported image payload {type(img)}") from exc


def pil_to_unit_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to float32 [0,1] CHW evitando copie inutili."""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()  # ensure writable buffer for torch.from_numpy
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.to(dtype=torch.float32).div_(255.0)


def two_view_transform(size: int, jitter: float = 0.4) -> Callable[[Image.Image], Tuple[torch.Tensor, torch.Tensor]]:
    augment = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(jitter, jitter, 0.2, 0.1),
            transforms.Lambda(lambda im: pil_to_unit_tensor(coerce_to_pil_rgb(im))),
        ]
    )
    return lambda img: (augment(img), augment(img))


def multicrop_transform(
    global_size: int,
    local_size: int,
    n_local: int,
    jitter: float = 0.4,
) -> Callable[[Image.Image], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    global_aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(jitter, jitter, 0.2, 0.1),
            transforms.Lambda(lambda im: pil_to_unit_tensor(coerce_to_pil_rgb(im))),
        ]
    )
    local_aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(jitter, jitter, 0.2, 0.1),
            transforms.Lambda(lambda im: pil_to_unit_tensor(coerce_to_pil_rgb(im))),
        ]
    )
    return lambda img: ([global_aug(img), global_aug(img)], [local_aug(img) for _ in range(n_local)])


def single_image_transform(size: int) -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda im: pil_to_unit_tensor(coerce_to_pil_rgb(im))),
        ]
    )


def sl_train_transforms(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda im: pil_to_unit_tensor(coerce_to_pil_rgb(im))),
        ]
    )


def sl_eval_transforms(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(size * 256 / 224)),
            transforms.CenterCrop(size),
            transforms.Lambda(lambda im: pil_to_unit_tensor(coerce_to_pil_rgb(im))),
        ]
    )


def collate_two_views(batch):
    x1 = torch.stack([sample[0][0] for sample in batch], 0)
    x2 = torch.stack([sample[0][1] for sample in batch], 0)
    meta = [sample[1] for sample in batch]
    return {"images": [x1, x2], "meta": meta}


def collate_multicrop(batch):
    global_crops = [g for sample in batch for g in sample[0][0]]
    local_crops = [l for sample in batch for l in sample[0][1]]
    meta = [sample[1] for sample in batch]
    return {
        "images": [torch.stack(global_crops, 0), torch.stack(local_crops, 0)],
        "meta": meta,
    }


def collate_single_image(batch):
    images = torch.stack([sample[0] for sample in batch], 0)
    meta = [sample[1] for sample in batch]
    return {"images": [images], "meta": meta}


def collate_supervised(batch, class_to_id: Dict[str, int]):
    images: List[torch.Tensor] = []
    labels: List[int] = []
    meta: List[Dict[str, Any]] = []
    missing: List[str] = []

    for image, info in batch:
        if not isinstance(image, torch.Tensor):
            image = pil_to_unit_tensor(coerce_to_pil_rgb(image))
        images.append(image)
        label_str = normalise_label(info.get("class_label", ""))
        if label_str not in class_to_id:
            missing.append(label_str)
        labels.append(class_to_id.get(label_str, -1))
        meta.append(info)

    if missing:
        raise KeyError(
            f"[SL] Labels not present in class_to_id after normalisation: {sorted(set(missing))}. "
            "Ensure data.webdataset.class_to_id is aligned with the normalisation scheme."
        )

    return {
        "inputs": torch.stack(images, 0),
        "targets": torch.tensor(labels, dtype=torch.long),
        "meta": meta,
    }
