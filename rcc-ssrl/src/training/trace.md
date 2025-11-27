datasets/builders.py codice <<
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
    collate_supervised_with_mixing,
    collate_two_views,
    estimate_tissue_fraction,
    ijepa_input_transform,
    multicrop_transform,
    multiscale_transform_or_none,
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


def _maybe_filter_tissue(dataset, sampler_cfg: Dict[str, Any]):
    if not bool((sampler_cfg or {}).get("tissue_aware", {}).get("enable", False)):
        return dataset
    tau = float(sampler_cfg["tissue_aware"].get("min_tissue_frac", 0.6))
    def _keep(sample):
        img, meta = sample[0], sample[1] if len(sample) > 1 else {}
        try:
            frac = estimate_tissue_fraction(img)
            return frac >= tau
        except Exception:
            return True
    return dataset.select(_keep)


def build_ssl_loader(
    data_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    split: str = "train",
    *,
    sampler_cfg: Dict[str, Any] | None = None,
    cfg_aug_top: Dict[str, Any] | None = None,
) -> torch.utils.data.DataLoader:
    """Construct the SSL dataloader based on the configured model family."""
    if "webdataset" not in data_cfg:
        raise KeyError("Missing data.webdataset configuration.")

    wds_cfg = data_cfg["webdataset"]
    shards = list_shards(wds_cfg[f"{split}_dir"])
    dataset = make_wds(shards, wds_cfg["shuffle_shards"], wds_cfg["shuffle_samples"])
    # filtro tessuto (se richiesto)
    dataset = _maybe_filter_tissue(dataset, sampler_cfg or {})

    img_size = int(data_cfg.get("img_size", 224))
    mode = model_cfg["ssl"]["name"].lower()
    use_mc_ibot = bool((model_cfg.get("ssl", {}) or {}).get("use_multicrop", False))
    use_mc_moco = bool((model_cfg.get("ssl", {}) or {}).get("use_multicrop", False)) if mode == "moco_v3" else False

    if (mode in ("moco_v3",) and not use_mc_moco) or (mode == "ibot" and not use_mc_ibot):
        # Preferisci le aug top-level se fornite (Macenko/HED/rotate90 ecc.)
        ssl_aug_legacy = (model_cfg.get("ssl", {}) or {}).get("aug", {})
        if cfg_aug_top:
            transform = two_view_transform(
                img_size, float(ssl_aug_legacy.get("jitter", 0.4)),
                blur_prob=float(ssl_aug_legacy.get("blur_prob", 0.1)),
                gray_prob=float(ssl_aug_legacy.get("gray_prob", 0.2)),
                solarize_prob=float(ssl_aug_legacy.get("solarize_prob", 0.0)),
                cfg_aug=cfg_aug_top,
            )
        else:
            transform = two_view_transform(
                img_size,
                float(ssl_aug_legacy.get("jitter", 0.4)),
                blur_prob=float(ssl_aug_legacy.get("blur_prob", 0.1)),
                gray_prob=float(ssl_aug_legacy.get("gray_prob", 0.2)),
                solarize_prob=float(ssl_aug_legacy.get("solarize_prob", 0.0)),
            )
        dataset = dataset.map_tuple(transform, lambda meta: meta)
        dataset = limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))
        collate_fn = collate_two_views
    elif mode == "dino_v3" or (mode == "ibot" and use_mc_ibot) or (mode == "moco_v3" and use_mc_moco):
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
            solarize_prob_g2=float(dino_cfg.get("solarize_prob_g2", 0.2)),
            cfg_aug=cfg_aug_top,
        )
        dataset = dataset.map_tuple(transform, lambda meta: meta)
        dataset = limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))
        collate_fn = collate_multicrop
    elif mode == "i_jepa":
        # No view-aug; optional stain normalization/jitter from top-level cfg.
        transform = ijepa_input_transform(img_size, cfg_aug_top)
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


def build_ssl_dataset(cfg: Dict[str, Any], mode: str, *, use_mc_moco: bool = False, use_mc_ibot: bool = False):
    ds_cfg = cfg.get("data", {})
    wds_cfg = ds_cfg.get("webdataset", {})
    img_size = int(ds_cfg.get("img_size", 224))
    model_cfg = cfg.get("model", {})
    aug_top = cfg.get("aug", {}) or {}

    shards = list_shards(wds_cfg["train_dir"])
    dataset = make_wds(shards, wds_cfg["shuffle_shards"], wds_cfg["shuffle_samples"])

    if (mode in ("moco_v3",) and not use_mc_moco) or (mode == "ibot" and not use_mc_ibot):
        # preferenza per aug top-level; retro-compat con model.ssl.aug
        ssl_aug = (model_cfg.get("ssl", {}) or {}).get("aug", {})
        cfg_aug = aug_top if aug_top else {"base": {}, "stain": {}}
        # merge semplice: i key presenti in model.ssl.aug sovrascrivono
        for k, v in (ssl_aug or {}).items():
            cfg_aug.setdefault(k, v)
        transform = two_view_transform(img_size, 0.4, cfg_aug=cfg_aug)
        dataset = dataset.map_tuple(transform, lambda meta: meta)
        dataset = limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))
        collate_fn = collate_two_views
    elif mode == "dino_v3" or (mode == "ibot" and use_mc_ibot) or (mode == "moco_v3" and use_mc_moco):
        dino_cfg = ds_cfg.get("dino_v3", {})
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
        transform = ijepa_input_transform(img_size, aug_top)
        dataset = dataset.map_tuple(transform, lambda meta: meta)
        dataset = limit_epoch(dataset, wds_cfg.get("samples_per_epoch"))
        collate_fn = collate_single_image
    else:
        raise ValueError(f"Unsupported SSL model family '{mode}'.")

    dataset = _maybe_filter_tissue(dataset, cfg.get("sampler", {}) or {})
    return dataset, collate_fn


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
    return build_ssl_loader(
        cfg["data"], cfg["model"], split=split,
        sampler_cfg=cfg.get("sampler", {}) or {},
        cfg_aug_top=cfg.get("aug", {}) or {},
    )
>>

datasets/device.py codice <<
"""Device selection utilities."""
from __future__ import annotations

import os
import time
from typing import List

import torch

__all__ = ["device_from_env"]


def device_from_env(allow_cpu: bool = False) -> torch.device:
    """
    Resolve the preferred torch.device respecting CUDA availability and config/env.
    Config (allow_cpu) takes precedence over env ALLOW_CPU=1.
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
>>

datasets/labels.py codice <<
"""Label normalisation and metadata helpers for WebDataset-backed loaders."""
from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["make_class_to_id_norm", "class_labels_from_cfg", "normalise_label"]

_ALIAS_TABLE = {
    "CCRCC": ["CCRCC", "CC_RCC", "CLEAR_CELL_RCC", "CLEARCELL"],
    "PRCC": ["PRCC", "P_RCC", "PAPILLARY_RCC"],
    "CHROMO": ["CHROMO", "CHROMOPHOBE", "CHR", "CHRCC"],
    "ONCO": ["ONCO", "ONCOCYTOMA"],
    "NOT_TUMOR": ["NOT_TUMOR", "NON_TUMOR", "NONTUMOR", "NORMAL", "BACKGROUND"],
}


def normalise_label(value: Any) -> str:
    """Normalise class labels by uppercasing and removing separators."""
    return str(value).strip().upper().replace(" ", "_").replace("-", "_")


def make_class_to_id_norm(user_map: Dict[str, int]) -> Dict[str, int]:
    """
    Build a normalized class mapping, appending canonical aliases when present
    in the user-provided YAML.
    """
    base = {normalise_label(k): v for k, v in user_map.items()}
    out = dict(base)
    for canon, synonyms in _ALIAS_TABLE.items():
        if canon in base:
            for synonym in synonyms:
                out[normalise_label(synonym)] = base[canon]
    return out


def class_labels_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    mapping = (cfg.get("data", {}).get("webdataset", {}) or {}).get("class_to_id", {})
    if not mapping:
        return []
    inverse = {idx: name for name, idx in mapping.items()}
    return [inverse[idx] for idx in sorted(inverse)]
>>

datasets/transforms.py codice <<
"""Augmentations and collate utilities used by SSL/SL loaders."""
from __future__ import annotations

from io import BytesIO
from typing import Any, Callable, Dict, List, Tuple, Optional
import io
import random
import math
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
    "ijepa_input_transform",
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


def _rotate90_multi() -> transforms.RandomChoice:
    # scelte 0/90/180/270, preservano morfologia
    return transforms.RandomChoice([
        transforms.Lambda(lambda im: im),
        transforms.Lambda(lambda im: im.rotate(90, expand=True)),
        transforms.Lambda(lambda im: im.rotate(180, expand=True)),
        transforms.Lambda(lambda im: im.rotate(270, expand=True)),
    ])

class _JPEGArtifacts:
    def __init__(self, p: float = 0.1, qmin: int = 40, qmax: int = 80):
        self.p, self.qmin, self.qmax = float(p), int(qmin), int(qmax)
    def __call__(self, im: Image.Image) -> Image.Image:
        if random.random() > self.p: return im
        q = random.randint(self.qmin, self.qmax)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q, optimize=True)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

class _RandomAdjustSharpness(transforms.RandomAdjustSharpness):
    def __init__(self, p: float = 0.2):
        super().__init__(sharpness_factor=1.5, p=float(p))

def _pil_to_unit(im: Image.Image) -> torch.Tensor:
    return pil_to_unit_tensor(coerce_to_pil_rgb(im))

# ---- Stain Normalization / Jitter (fallback se lib non presenti) ----
class StainNormalizer:
    def __init__(self, method: str = "macenko", enable: bool = False):
        self.enable = bool(enable)
        self.method = (method or "macenko").lower()
        self._impl = None
        if not self.enable: return
        # prova staintools / torchstain
        try:
            import staintools  # type: ignore
            self._lib = "staintools"
            self._impl = staintools
        except Exception:
            try:
                import torchstain  # type: ignore
                self._lib = "torchstain"
                self._impl = torchstain
            except Exception:
                self._lib = None
                self.enable = False  # fallback no-op
    def __call__(self, im: Image.Image) -> Image.Image:
        if not self.enable or self._impl is None: return im
        try:
            if self._lib == "staintools":
                import staintools
                tgt = np.asarray(im)
                if self.method == "vahadane":
                    N = staintools.StainNormalizer(method=staintools.StainNormalizer.METHOD_VAHADANE)
                else:
                    N = staintools.StainNormalizer(method=staintools.StainNormalizer.METHOD_MACENKO)
                # Nota: in assenza di "fit" sul target, si usa auto-fit per immagine
                N.fit(tgt)
                out = N.transform(tgt)
                return Image.fromarray(out.astype(np.uint8))
            elif self._lib == "torchstain":
                # torchstain richiede torch tensori in OD/HE; per brevitÃ : no-op se mancano parametri
                return im
        except Exception:
            return im
        return im

class HEDColorJitter:
    """Jitter lieve in spazio HED; fallback a ColorJitter se skimage non disponibile."""
    def __init__(self, delta: float = 0.02, enable: bool = False):
        self.delta, self.enable = float(delta), bool(enable)
        try:
            from skimage import color  # type: ignore
            self._sk_color = color
        except Exception:
            self._sk_color = None
            self._cj = transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)
    def __call__(self, im: Image.Image) -> Image.Image:
        if not self.enable: return im
        if self._sk_color is None:
            return self._cj(im)  # fallback
        arr = np.asarray(im).astype(np.float32) / 255.0
        hed = self._sk_color.rgb2hed(arr)
        noise = np.random.uniform(-self.delta, self.delta, size=hed.shape).astype(np.float32)
        hed2 = hed + noise
        rgb = np.clip(self._sk_color.hed2rgb(hed2), 0.0, 1.0)
        return Image.fromarray((rgb * 255.0).astype(np.uint8))

# ---- Multi-Scale ----
class MultiScaleCrops:
    def __init__(self, size: int, scales: List[float], n_per_scale: int = 1):
        self.size = int(size)
        self.scales = [float(s) for s in (scales or [1.0])]
        self.n_per_scale = int(max(1, n_per_scale))
    def __call__(self, im: Image.Image) -> List[Image.Image]:
        outs: List[Image.Image] = []
        w, h = im.size
        for s in self.scales:
            sw, sh = int(w / s), int(h / s)
            if sw <= 0 or sh <= 0:
                continue
            for _ in range(self.n_per_scale):
                if sw < self.size or sh < self.size:
                    crop = im.resize((max(self.size, sw), max(self.size, sh)))
                else:
                    x = random.randint(0, max(0, sw - self.size))
                    y = random.randint(0, max(0, sh - self.size))
                    crop = im.crop((x, y, x + self.size, y + self.size))
                outs.append(crop)
        return outs or [im.resize((self.size, self.size))]

# ---- Tissue fraction (per filtro sampler) ----
def estimate_tissue_fraction(im: Image.Image, sat_thr: int = 25) -> float:
    hsv = im.convert("HSV")
    s = np.asarray(hsv)[:, :, 1].astype(np.uint8)
    frac = float((s > sat_thr).sum()) / float(s.size)
    return frac

# ---- Builder Augment (core + stain) ----
def build_base_augment(size: int, base_cfg: Dict[str, Any]) -> transforms.Compose:
    scale_lo, scale_hi = (base_cfg.get("random_resized_crop", {}) or {}).get("scale", [0.6, 1.0])
    ratio_lo, ratio_hi = (base_cfg.get("random_resized_crop", {}) or {}).get("ratio", [0.75, 1.33])
    ops = []
    if base_cfg.get("rotate90", False):
        ops.append(_rotate90_multi())
    ops.extend([
        transforms.RandomHorizontalFlip() if base_cfg.get("hflip", True) else transforms.Lambda(lambda x: x),
        transforms.RandomVerticalFlip() if base_cfg.get("vflip", True) else transforms.Lambda(lambda x: x),
        transforms.RandomResizedCrop(size, scale=(float(scale_lo), float(scale_hi)), ratio=(float(ratio_lo), float(ratio_hi))),
    ])
    # Color jitter
    cj_cfg = base_cfg.get("color_jitter", {})
    if cj_cfg.get("enable", True):
        brightness = cj_cfg.get("brightness", 0.4)
        contrast = cj_cfg.get("contrast", 0.4)
        saturation = cj_cfg.get("saturation", 0.2)
        hue = cj_cfg.get("hue", 0.1)
        ops.append(transforms.ColorJitter(brightness, contrast, saturation, hue))
    # Grayscale
    grayscale_p = base_cfg.get("grayscale_p", 0.2)
    if grayscale_p > 0:
        ops.append(transforms.RandomGrayscale(p=grayscale_p))
    # Gaussian blur
    blur_p = base_cfg.get("gaussian_blur_p", 0.2)
    if blur_p > 0:
        ops.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=blur_p))
    # Sharpen
    sharpen_p = base_cfg.get("sharpen_p", 0.2)
    if sharpen_p > 0:
        ops.append(transforms.RandomApply([_RandomAdjustSharpness(p=1.0)], p=sharpen_p))
    # JPEG artifacts
    jpeg_p = base_cfg.get("jpeg_artifacts_p", 0.1)
    if jpeg_p > 0:
        ops.append(transforms.RandomApply([_JPEGArtifacts(p=1.0)], p=jpeg_p))
    # Solarize
    solarize_p = base_cfg.get("solarize_p", 0.0)
    if solarize_p > 0 and hasattr(transforms, "RandomSolarize"):
        ops.append(transforms.RandomApply([transforms.RandomSolarize(threshold=128)], p=solarize_p))
    ops.append(transforms.Lambda(_pil_to_unit))
    return transforms.Compose(ops)

def build_stain_ops(stain_cfg: Dict[str, Any]) -> List[Callable[[Image.Image], Image.Image]]:
    ops: List[Callable] = []
    norm = stain_cfg.get("normalize", {}) or {}
    if bool(norm.get("enable", False)):
        ops.append(StainNormalizer(method=str(norm.get("method", "macenko")), enable=True))
    jitter = stain_cfg.get("jitter", {}) or {}
    if bool(jitter.get("enable", False)):
        ops.append(HEDColorJitter(delta=float(jitter.get("delta", 0.02)), enable=True))
    # randstainna stub: se vuoi usare una lib esterna, qui resta disattivato per design no-deps
    return ops

def ijepa_input_transform(size: int, cfg_aug: Optional[Dict[str, Any]] = None) -> Callable[[Image.Image], torch.Tensor]:
    """
    Deterministic, no view-augmentation input pipeline for I-JEPA.
    Optionally applies stain normalization / light HED jitter if provided
    in cfg_aug['stain'], but avoids flips/solarize/blur to respect JEPA's
    "no manual aug" principle (masking happens inside the model).
    """
    base_resize = transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
    ])
    stain_ops = build_stain_ops((cfg_aug or {}).get("stain", {}) if cfg_aug else {})
    def _apply(im: Image.Image) -> torch.Tensor:
        out = base_resize(coerce_to_pil_rgb(im))
        for op in stain_ops:  # optional, safe to keep empty
            out = op(out)
        return pil_to_unit_tensor(out)
    return _apply

def two_view_transform(
    size: int,
    jitter: float = 0.4,
    *,
    blur_prob: float = 0.1,
    gray_prob: float = 0.2,
    solarize_prob: float = 0.0,
    cfg_aug: Optional[Dict[str, Any]] = None,
) -> Callable[[Image.Image], Tuple[torch.Tensor, torch.Tensor]]:
    # Backward compatibility: if cfg_aug is provided, use new system
    if cfg_aug is not None:
        base = build_base_augment(size, cfg_aug.get("base", {}) or {"hflip": True, "vflip": True, "random_resized_crop": {"scale": [0.6, 1.0], "ratio": [0.75, 1.33]}})
        stain_ops = build_stain_ops(cfg_aug.get("stain", {}) or {})
        def _apply(img: Image.Image) -> torch.Tensor:
            for op in stain_ops:
                img = op(img)
            return base(img)
        return lambda img: (_apply(img), _apply(img))
    # Legacy path: use old parameters
    def _blur():
        k = int(max(3, (size // 20) * 2 + 1))
        return transforms.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))
    ops = [
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(jitter, jitter, 0.2, 0.1),
        transforms.RandomGrayscale(p=float(gray_prob)),
        transforms.RandomApply([_blur()], p=float(blur_prob)),
    ]
    if solarize_prob > 0.0 and hasattr(transforms, "RandomSolarize"):
        ops.append(transforms.RandomApply([transforms.RandomSolarize(threshold=128)], p=float(solarize_prob)))
    ops.append(transforms.Lambda(lambda im: pil_to_unit_tensor(coerce_to_pil_rgb(im))))
    augment = transforms.Compose(ops)
    return lambda img: (augment(img), augment(img))

def multiscale_transform_or_none(size: int, ms_cfg: Dict[str, Any]) -> Optional[Callable[[Image.Image], List[torch.Tensor]]]:
    if not bool(ms_cfg.get("enable", False)): return None
    scales = [float(s) for s in ms_cfg.get("scales", [1.0, 1.5, 2.0])]
    nps = int(ms_cfg.get("n_per_scale", 1))
    MSC = MultiScaleCrops(size, scales, n_per_scale=nps)
    base = build_base_augment(size, (ms_cfg.get("base") or ({})))
    stain_ops = build_stain_ops((ms_cfg.get("stain") or ({})))
    def _call(img: Image.Image) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        for crop in MSC(img):
            im = crop
            for op in stain_ops: im = op(im)
            outs.append(base(im))
        return outs
    return _call


def multicrop_transform(
    global_size: int,
    local_size: int,
    n_local: int,
    jitter: float = 0.4,
    *,
    global_scale: Tuple[float, float] = (0.14, 1.0),
    local_scale: Tuple[float, float] = (0.05, 0.14),
    blur_prob: float = 0.5,
    solarize_prob: float = 0.0,
    solarize_prob_g2: float = 0.2,
    cfg_aug: Optional[Dict[str, Any]] = None,
) -> Callable[[Image.Image], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    """
    Multi-crop alla DINO:
      - 2 global crops (scale ampie)
      - n_local local crops (field-of-view ridotto)
      - augment: jitter, blur opzionale, solarization opzionale
    """
    # opzionale: stain ops + rotate90 dalle aug top-level
    stain_ops = build_stain_ops((cfg_aug or {}).get("stain", {}) if cfg_aug else {})
    rotate90_flag = bool(((cfg_aug or {}).get("base", {}) or {}).get("rotate90", False)) if cfg_aug else False

    def _apply_stain(im: Image.Image) -> Image.Image:
        for op in stain_ops:
            im = op(im)
        return im

    def _blur(size: int):
        k = int(max(3, (size // 20) * 2 + 1))  # kernel dispari ~ size/20
        return transforms.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))

    solarize_base = (
        transforms.RandomSolarize(threshold=128)
        if hasattr(transforms, "RandomSolarize")
        else transforms.Lambda(lambda im: im)
    )

    def _build_aug(size: int, scale: Tuple[float, float], do_blur: bool, do_solar: bool):
        ops: List[Any] = [
            transforms.Lambda(_apply_stain),
            _rotate90_multi() if rotate90_flag else transforms.Lambda(lambda im: im),
            transforms.RandomResizedCrop(size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(jitter, jitter, 0.2, 0.1),
        ]
        if do_blur and blur_prob > 0:
            ops.append(transforms.RandomApply([_blur(size)], p=float(blur_prob)))
        if do_solar and solarize_prob > 0:
            ops.append(transforms.RandomApply([solarize_base], p=float(solarize_prob)))
        ops.append(transforms.Lambda(lambda im: pil_to_unit_tensor(coerce_to_pil_rgb(im))))
        return transforms.Compose(ops)

    # Global #1: NO solarize; Global #2: solarize con p dedicata
    global_aug1 = _build_aug(global_size, global_scale, do_blur=True, do_solar=False)
    # usa p dedicata per la seconda global (stile DINO: solarize solo su una vista)
    old_sp = float(solarize_prob)
    solarize_prob = float(solarize_prob_g2)
    global_aug2 = _build_aug(global_size, global_scale, do_blur=True, do_solar=True)
    # ripristina (per sicurezza in caso di chiusure)
    solarize_prob = old_sp
    local_aug  = _build_aug(local_size,  local_scale,  do_blur=True, do_solar=False)
    return lambda img: ([global_aug1(img), global_aug2(img)], [local_aug(img) for _ in range(n_local)])


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
    x1 = torch.stack([sample[0][0] for sample in batch], 0).contiguous(memory_format=torch.channels_last)
    x2 = torch.stack([sample[0][1] for sample in batch], 0).contiguous(memory_format=torch.channels_last)
    meta = [sample[1] for sample in batch]
    return {"images": [x1, x2], "meta": meta}


def collate_multicrop(batch):
    # NOTE: 'g' e 'l' sono tensori 3D (C,H,W). 'channels_last' richiede 4D.
    # Primo: stack -> 4D (N,C,H,W). Poi: applica channels_last in modo sicuro.
    g_list = [g for sample in batch for g in sample[0][0]]
    l_list = [l for sample in batch for l in sample[0][1]]

    G = torch.stack(g_list, 0).contiguous(memory_format=torch.channels_last)
    L = torch.stack(l_list, 0).contiguous(memory_format=torch.channels_last)

    meta = [sample[1] for sample in batch]
    return {"images": [G, L], "meta": meta}


def collate_single_image(batch):
    images = torch.stack([sample[0] for sample in batch], 0).contiguous(memory_format=torch.channels_last)
    meta = [sample[1] for sample in batch]
    return {"images": [images], "meta": meta}


def _mixup_cutmix(x: torch.Tensor, y: torch.Tensor, mixing_cfg: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    C = int(y.max().item() + 1) if y.ndim == 1 else y.size(1)
    one_hot = (torch.nn.functional.one_hot(y, num_classes=C).float() if y.ndim == 1 else y)
    B = x.size(0)
    idx = torch.randperm(B, device=x.device)
    x2, y2 = x[idx], one_hot[idx]
    out_x, out_y = x, one_hot
    # MixUp
    mx = (mixing_cfg.get("mixup") or {})
    if bool(mx.get("enable", False)):
        import numpy as _np
        alpha = float(mx.get("alpha", 0.3))
        lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
        out_x = lam * out_x + (1 - lam) * x2
        out_y = lam * out_y + (1 - lam) * y2
    # CutMix
    cm = (mixing_cfg.get("cutmix") or {})
    if bool(cm.get("enable", False)) and (random.random() < float(cm.get("p", 0.5))):
        beta = float(cm.get("beta", 1.0))
        lam = np.random.beta(beta, beta) if beta > 0 else 1.0
        H, W = x.size(2), x.size(3)
        rx, ry = np.random.uniform(0, W), np.random.uniform(0, H)
        rw, rh = W * math.sqrt(1 - lam), H * math.sqrt(1 - lam)
        x1, y1 = int(np.clip(rx - rw / 2, 0, W)), int(np.clip(ry - rh / 2, 0, H))
        x2_, y2_ = int(np.clip(rx + rw / 2, 0, W)), int(np.clip(ry + rh / 2, 0, H))
        out_x[:, :, y1:y2_, x1:x2_] = x2[:, :, y1:y2_, x1:x2_]
        lam2 = 1 - ((x2_ - x1) * (y2_ - y1) / (W * H))
        out_y = lam2 * out_y + (1 - lam2) * y2
    return out_x, out_y

def collate_supervised_with_mixing(batch: List[Tuple[torch.Tensor, int, Dict[str, Any]]], *, mixing_cfg: Dict[str, Any]):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
    if mixing_cfg:
        try:
            imgs, labels = _mixup_cutmix(imgs, labels, mixing_cfg)  # labels diventa soft se mix attivo
        except Exception:
            pass
    metas = [b[2] if len(b) > 2 else {} for b in batch]
    return {"images": imgs, "labels": labels, "meta": metas}

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
        "inputs": torch.stack(images, 0).contiguous(memory_format=torch.channels_last),
        "targets": torch.tensor(labels, dtype=torch.long),
        "meta": meta,
    }
>>

data/webdataset.py codice <<
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
>>

.DS_Store codice <<
   Bud1                                                                      i g sdsclbo                                                                                                                                                                           c o n f i g sdsclbool    e n vdsclbool    m o d e l sdsclbool    s c r i p t sdsclbool    s l u r mdsclbool    u t i l sdsclbool                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  @      €                                        @      €                                          @      €                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   E                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         DSDB                                 `          €                                         @      €                                          @      €                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              >>

launch_training.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------
MODULE_ROOT = Path(__file__).resolve().parent          # .../src/training
REPO_ROOT = MODULE_ROOT.parent.parent                  # .../
SRC_ROOT = REPO_ROOT / "src"

for p in (REPO_ROOT, SRC_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from src.training.orchestrator import Orchestrator
from src.training.utils.io import append_row_csv, make_exp_id
from src.training.utils.reproducibility import set_global_seed, copy_code_snapshot
from src.training.utils.paths import CONFIG_PATH as DEFAULT_CONFIG_PATH, RUN_INDEX as DEFAULT_RUN_INDEX, _as_abs

# ---------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------
def _merge_dicts(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def read_yaml(path: Path | str) -> Dict[str, Any]:
    """
    Read a YAML file, honoring optional `_include` entries for shared fragments.
    Includes are resolved relative to the current file and merged depth-first.
    """
    path = _as_abs(path)
    with path.open("r") as handle:
        cfg = yaml.safe_load(handle) or {}

    includes = cfg.pop("_include", []) or []
    if isinstance(includes, str):
        includes = [includes]

    merged: Dict[str, Any] = {}
    for inc in includes:
        inc_path = (path.parent / inc).resolve()
        merged = _merge_dicts(merged, read_yaml(inc_path))

    merged = _merge_dicts(merged, cfg)
    return merged

# ---------------------------------------------------------------------
# paths(): centralization of resolved project paths
# ---------------------------------------------------------------------
def _is_rank_zero() -> bool:
    return os.environ.get("RANK", "0") == "0"

def _stringify_paths(tree: Dict[str, Any]) -> Dict[str, Any]:
    def _convert(val: Any) -> Any:
        if isinstance(val, Path):
            return str(val)
        if isinstance(val, dict):
            return {k: _convert(v) for k, v in val.items()}
        return val
    return {key: _convert(val) for key, val in tree.items()}

def paths() -> Dict[str, Any]:
    from src.training.utils import paths as pathmod

    resolved = pathmod.get_all()
    project_root = resolved["project_root"]
    outputs_root = resolved["outputs_root"]
    if _is_rank_zero():
        print(f"[paths] project_root={project_root} outputs_root={outputs_root}")

    webdataset = resolved.get("webdataset", {})
    if not webdataset:
        raise KeyError("[paths] missing or empty 'webdataset' section")

    for key, section in webdataset.items():
        for field in ("train_dir", "val_dir", "test_dir"):
            candidate = section.get(field)
            if candidate is None:
                raise KeyError(f"[paths] webdataset.{key}.{field} missing")
            if not Path(candidate).exists():
                raise FileNotFoundError(f"[paths] missing {key}.{field}: {candidate}")
        if _is_rank_zero():
            print(f"[paths] webdataset.{key}: train={section['train_dir']} "
                  f"val={section['val_dir']} test={section['test_dir']}")

    return resolved

# ---------------------------------------------------------------------
# Merge utils
# ---------------------------------------------------------------------
def deep_update(base: Dict, override: Optional[Dict]) -> Dict:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged

def compose_run_config(base_cfg: Dict[str, Any], run_block: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    """Compose a single run config from a base config + a 'runs' entry override."""
    cfg = deep_update(base_cfg, run_block.get("override", {}))
    experiment = cfg.setdefault("experiment", {})
    experiment["name"] = base_cfg["experiment"]["name"]  # keep main name
    runtime = cfg.setdefault("_runtime", {})
    runtime["mode"] = run_block.get("mode", base_cfg.get("model", {}).get("type", "ssl"))
    runtime["config_path"] = str(cfg_path)

    # Prefer RUN_NAME from environment if provided, otherwise runs[].name
    env_run_name = os.environ.get("RUN_NAME", "").strip()
    runtime["run_name"] = env_run_name if env_run_name else run_block["name"]
    # Thread-through of ablation id / explicit run subdir from env (set by sbatch launcher)
    abl_id = os.environ.get("ABLATION_ID", "").strip()
    if abl_id:
        runtime["ablation_id"] = abl_id
    subdir = os.environ.get("EXP_SUBDIR", "").strip()
    if subdir:
        runtime["run_subdir"] = subdir
        # Force experiment.name to the canonical leaf dir (no "pretty" names)
        experiment["name"] = subdir

    # Drop top-level 'runs' to freeze config
    cfg.pop("runs", None)
    return cfg

def expand_runs(base_cfg: Dict[str, Any], cfg_path: Path, run_index: int) -> List[Dict[str, Any]]:
    """Expand 'runs' block to a list of concrete run configs respecting RUN_INDEX."""
    runs = base_cfg.get("runs", [])
    if run_index >= 0:
        if not runs:
            if _is_rank_zero():
                print(f"[runs] Ignoring RUN_INDEX={run_index}: config '{cfg_path}' has no runs.")
        elif run_index >= len(runs):
            raise IndexError(
                f"RUN_INDEX={run_index} out of range for config '{cfg_path}' "
                f"(available runs: {len(runs)})"
            )
        else:
            runs = [runs[run_index]]
    if runs:
        return [compose_run_config(base_cfg, block, cfg_path) for block in runs]

    # Single-run config (no 'runs' block)
    single = copy.deepcopy(base_cfg)
    runtime = single.setdefault("_runtime", {})
    runtime["mode"] = runtime.get("mode", single.get("model", {}).get("type", "ssl"))
    runtime["config_path"] = str(cfg_path)
    env_run_name = os.environ.get("RUN_NAME", "").strip()
    runtime["run_name"] = env_run_name if env_run_name else single.get("experiment", {}).get("name", "default")
    single.pop("runs", None)
    return [single]

# ---------------------------------------------------------------------
# Inject resolved paths into run config
# ---------------------------------------------------------------------
def inject_paths_into_cfg(cfg: Dict[str, Any], resolved_paths: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    experiment = cfg.setdefault("experiment", {})
    experiment["outputs_root"] = str(resolved_paths["outputs_root"])
    experiment["project_root"] = str(resolved_paths["project_root"])

    # Optional MLflow experiment override from env
    mlflow_env = os.environ.get("MLFLOW_EXPERIMENT_NAME", "").strip()
    if mlflow_env:
        experiment["mlflow_experiment"] = mlflow_env

    wds_cfg = cfg.setdefault("data", {}).setdefault("webdataset", {})
    dataset_key = wds_cfg.get("dataset_key")
    if not dataset_key:
        raise KeyError("data.webdataset.dataset_key missing: it must match a key in src/training/paths.py")
    if dataset_key not in resolved_paths["webdataset"]:
        raise KeyError(f"dataset_key='{dataset_key}' not present in src/training/paths.py")

    selected = resolved_paths["webdataset"][dataset_key]
    wds_cfg["train_dir"] = str(selected["train_dir"])
    wds_cfg["val_dir"] = str(selected["val_dir"])
    wds_cfg["test_dir"] = str(selected["test_dir"])
    print(f"[paths] dataset_key={dataset_key} -> train={selected['train_dir']} "
          f"val={selected['val_dir']} test={selected['test_dir']}")

    runtime = cfg.setdefault("_runtime", {})
    runtime["paths"] = _stringify_paths(resolved_paths)
    return cfg

# ---------------------------------------------------------------------
# CSV Summary utilities
# ---------------------------------------------------------------------
def _summary_csv_path(run_root: Path, mode: str) -> Path:
    exp_folder = run_root.parents[1]
    exp_folder.mkdir(parents=True, exist_ok=True)
    return exp_folder / f"runs_summary_{mode}.csv"

def _record_summary(orch: Orchestrator, metrics: Dict[str, Any], elapsed_s: float) -> Path:
    run_label = orch.cfg.get("_runtime", {}).get("run_name", orch.cfg["experiment"]["name"])
    row = {
        "exp_id": orch.exp_id,
        "run_name": run_label,
        "mode": orch.mode,
        "model": orch.model_key,
        "elapsed_s": round(elapsed_s, 2),
        **metrics,
    }
    return append_row_csv(_summary_csv_path(orch.run_dirs["root"], orch.mode), row)

# ---------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------
def _env_exp_datetime() -> str | None:
    """Return EXP_DATETIME if valid (YYYYMMDD-HHMMSS)."""
    v = os.environ.get("EXP_DATETIME", "").strip()
    return v if re.match(r"^\d{8}-\d{6}$", v) else None

def _env_exp_group() -> str | None:
    """Return EXP_GROUP if non-empty (used as shared exp_id)."""
    v = os.environ.get("EXP_GROUP", "").strip()
    return v or None

def _resolve_config_path(cli_cfg: Optional[str]) -> Path:
    """
    Resolve the configuration file path with this precedence:
    1) --config CLI
    2) $CONFIG_PATH
    3) $RUN_CFG
    4) $TRAIN_CONFIG
    5) DEFAULT_CONFIG_PATH (from utils.paths)
    """
    candidates = [
        cli_cfg,
        os.environ.get("CONFIG_PATH"),
        os.environ.get("RUN_CFG"),
        os.environ.get("TRAIN_CONFIG"),
        os.environ.get("EXPERIMENT_CONFIG_PATH"),
        str(DEFAULT_CONFIG_PATH),
    ]
    for c in candidates:
        if c:
            p = Path(c).expanduser()
            if p.is_file():
                return p
    # Last resort: show where we looked
    raise FileNotFoundError(
        "No valid config file found. Tried (in order): "
        f"--config, $CONFIG_PATH, $RUN_CFG, $TRAIN_CONFIG, $EXPERIMENT_CONFIG_PATH, DEFAULT_CONFIG_PATH={DEFAULT_CONFIG_PATH}"
    )

def _resolve_run_index(cli_idx: Optional[int]) -> int:
    """
    Resolve RUN_INDEX with precedence:
    1) --run-index CLI
    2) $RUN_INDEX env
    3) DEFAULT_RUN_INDEX (from utils.paths)
    """
    if cli_idx is not None:
        return cli_idx
    env_val = os.environ.get("RUN_INDEX", "").strip()
    if env_val != "":
        try:
            return int(env_val)
        except ValueError:
            raise ValueError(f"Invalid RUN_INDEX env value: '{env_val}'")
    return int(DEFAULT_RUN_INDEX)

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SSL/SL training launcher")
    parser.add_argument("--config", type=str, help="Path to YAML config", default=None)
    parser.add_argument("--run-index", type=int, help="Index into 'runs' list (>=0) or -1 for all/single", default=None)
    args = parser.parse_args(argv)

    # Resolve config path and run index with clear precedence
    cfg_path = _resolve_config_path(args.config)
    run_index = _resolve_run_index(args.run_index)

    if _is_rank_zero():
        print(f"[config] Using config: {cfg_path}")
        print(f"[config] RUN_INDEX={run_index}")

    base_cfg = read_yaml(cfg_path)
    resolved_paths = paths()
    all_runs = expand_runs(base_cfg, cfg_path, run_index)

    # Decide shared experiment id (directory name under outputs/experiments/<exp_id>)
    shared_exp_id: Optional[str] = None
    exp_dt = _env_exp_datetime()
    exp_group = _env_exp_group()

    for cfg_run in all_runs:
        cfg_run = inject_paths_into_cfg(cfg_run, resolved_paths)

        runtime = cfg_run.setdefault("_runtime", {})

        # Prefer EXP_GROUP (exact name), then EXP_DATETIME (prefixed), else autogenerated
        if shared_exp_id is None:
            if exp_group:
                shared_exp_id = exp_group
            elif exp_dt:
                shared_exp_id = f"exp_{exp_dt}"
            else:
                shared_exp_id = make_exp_id(cfg_run["experiment"]["outputs_root"], exp_dt)

        runtime["exp_id"] = shared_exp_id
        cfg_run.setdefault("experiment", {})["id"] = shared_exp_id

        # Also surface the canonical run_subdir on the experiment section (for downstream tools)
        rsd = cfg_run.get("_runtime", {}).get("run_subdir")
        if rsd:
            cfg_run["experiment"]["name"] = rsd

        # Expose for downstream tooling
        os.environ["EXP_ID"] = shared_exp_id

        # Seed and snapshot
        set_global_seed(cfg_run["experiment"].get("seed", 1337))
        orchestrator = Orchestrator(cfg_run)

        # Best-effort code snapshot (non-fatal)
        try:
            # Save code under .../records/code_snapshot (no virtualenv / large artifacts).
            snap_dst = os.path.join(str(orchestrator.run_dirs["records"]), "code_snapshot")
            copy_code_snapshot(
                str(SRC_ROOT / "training"),
                snap_dst,
                excludes=(),  # default excludes already skip .venv/site-packages/etc.
            )
        except Exception:
            pass

        try:
            start_time = time.time()
            metrics = orchestrator.fit()
            _record_summary(orchestrator, metrics, time.time() - start_time)
        except Exception as e:
            # Always emit a reporting stub so downstream tooling finds the folder.
            try:
                rep_dir = orchestrator.run_dirs["root"] / "reporting"
                rep_dir.mkdir(parents=True, exist_ok=True)
                with (rep_dir / "FAILED.txt").open("w") as fh:
                    fh.write(f"{type(e).__name__}: {e}\n")
            except Exception:
                pass
            raise

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
>>

models/dino_v3.py codice <<
# models/dino_v3.py
from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.trainer.backbones import get_backbone, mlp_head, resolve_backbone_from_model_cfg
from src.training.utils.torch_ops import copy_weights_and_freeze, ema_update, l2n
from src.training.trainer.loops import SSLBaseModel

def dino_distill_loss(
    s: torch.Tensor,
    t: torch.Tensor,
    t_temp: float = 0.04,
    s_temp: float = 0.1,
    center: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if center is None:
        center = t.mean(0, keepdim=True)
    t_norm = (t - center) / max(t_temp, 1e-8)
    s_norm = s / max(s_temp, 1e-8)
    pt = t_norm.softmax(dim=-1)
    ls = s_norm.log_softmax(dim=-1)
    return -(pt * ls).sum(dim=-1).mean()

def gram_loss(tokens_s: torch.Tensor, tokens_t: torch.Tensor) -> torch.Tensor:
    ts = l2n(tokens_s); tt = l2n(tokens_t)
    Gs = ts @ ts.transpose(1,2); Gt = tt @ tt.transpose(1,2)
    return F.mse_loss(Gs, Gt)

class DINOv3(SSLBaseModel):
    def __init__(self, stu: torch.nn.Module, tea: torch.nn.Module,
                 head_s_g: nn.Module, head_t_g: nn.Module,
                 head_s_l: nn.Module, head_t_l: nn.Module,
                 t_t: float=0.04, t_s: float=0.1, w_gram: float=1.0, ema_m: float=0.996,
                 center_m: float = 0.9,
                 t_warmup_min: Optional[float] = None,
                 t_warmup_max: Optional[float] = None,
                 t_warmup_steps: int = 0):
        super().__init__()
        self.stu, self.tea = stu, tea
        self.hs_g, self.ht_g = head_s_g, head_t_g
        self.hs_l, self.ht_l = head_s_l, head_t_l
        self.t_t, self.t_s, self.wg, self.m = t_t, t_s, w_gram, ema_m
        self.center_m = float(center_m)
        self.t_min = float(t_warmup_min) if t_warmup_min is not None else float(t_t)
        self.t_max = float(t_warmup_max) if t_warmup_max is not None else float(t_t)
        self.t_warmup_steps = int(max(0, t_warmup_steps))
        self.center = None
        self._bootstrap()

    @classmethod
    def from_config(cls, cfg: Dict[str,Any]) -> "DINOv3":
        m = cfg["model"]["ssl"]
        bname, bopts = resolve_backbone_from_model_cfg(cfg["model"])
        stu = get_backbone(name=bname, pretrained=False, **bopts)
        tea = get_backbone(name=bname, pretrained=False, **bopts)
        dim = stu.out_dim
        head_s_g = mlp_head(dim, 4096, 1024, bn_last_affine=True); head_t_g = mlp_head(dim, 4096, 1024, bn_last_affine=True)
        head_s_l = mlp_head(dim, 2048, 256, bn_last_affine=True); head_t_l = mlp_head(dim, 2048, 256, bn_last_affine=True)
        # Accept both 'teacher_temp_schedule' and 'temp_teacher_schedule'
        # and both naming schemes: {min,max,warmup_steps} or {start,end,warmup_frac}.
        sched = (m.get("temp_teacher_schedule") or m.get("teacher_temp_schedule") or {})
        # Derive warmup steps if only a fraction is provided.
        tr_ssl = (cfg.get("train", {}).get("ssl", {}) or {})
        total_steps = int(tr_ssl.get("epochs", 1)) * int(tr_ssl.get("steps_per_epoch", 1))
        def _get(name, alt, default):
            v = sched.get(name, None)
            if v is None:
                v = sched.get(alt, default)
            return v
        t_min = _get("min",   "start", m.get("temp_teacher", 0.04))
        t_max = _get("max",   "end",   m.get("temp_teacher", 0.04))
        wu_fr = sched.get("warmup_frac", None)
        wu_st = sched.get("warmup_steps", None)
        if wu_st is None and wu_fr is not None:
            try:
                wu_st = int(float(wu_fr) * max(1, total_steps))
            except Exception:
                wu_st = 0
        if wu_st is None:
            wu_st = 0
        return cls(
            stu, tea, head_s_g, head_t_g, head_s_l, head_t_l,
            t_t=m.get("temp_teacher", 0.04),
            t_s=m.get("temp_student", 0.1),
            w_gram=m.get("gram_lambda", 1.0),
            ema_m=cfg["train"]["ssl"].get("ema_momentum", 0.996),
            center_m=m.get("center_momentum", 0.9),
            t_warmup_min=float(t_min),
            t_warmup_max=float(t_max),
            t_warmup_steps=int(wu_st),
        )

    @torch.no_grad()
    def _bootstrap(self):
        copy_weights_and_freeze(self.tea, self.stu)

    def _teacher_temp(self, global_step: int) -> float:
        if self.t_warmup_steps <= 0 or abs(self.t_max - self.t_min) < 1e-8:
            return self.t_max
        alpha = min(1.0, max(0.0, float(global_step) / float(self.t_warmup_steps)))
        return self.t_min + (self.t_max - self.t_min) * alpha

    def training_step(self, batch: Dict[str,Any], global_step: int) -> Dict[str,Any]:
        # images = [G, L] (multi-crop) oppure [G] se non hai locali
        images = batch["images"]
        if len(images) == 2:
            G, L = images[0], images[1]      # G: concat global, L: concat local
        else:
            G, L = images[0], images[0]      # fallback senza locali

        student_global = self.hs_g(self.stu.forward_global(G))
        student_tokens_raw = self.stu.forward_tokens(L)

        with torch.no_grad():
            ema_update(self.tea, self.stu, self.m)
            teacher_global = self.ht_g(self.tea.forward_global(G)).detach()
            teacher_tokens_raw = self.tea.forward_tokens(L)

        mean_t = teacher_global.mean(0, keepdim=True)
        self.center = mean_t if self.center is None else self.center_m * self.center + (1.0 - self.center_m) * mean_t
        t_temp = self._teacher_temp(global_step)
        loss_global = dino_distill_loss(student_global, teacher_global, t_temp, self.t_s, self.center)

        b_tokens, t_tokens, c_tokens = student_tokens_raw.shape
        student_tokens = self.hs_l(student_tokens_raw.reshape(b_tokens * t_tokens, c_tokens)).view(b_tokens, t_tokens, -1)
        teacher_tokens = self.ht_l(teacher_tokens_raw.reshape(b_tokens * t_tokens, c_tokens)).view(b_tokens, t_tokens, -1).detach()
        loss_gram = gram_loss(student_tokens, teacher_tokens)

        loss = loss_global + self.wg * loss_gram
        return {
            "loss_total": loss,
            "loss_components": {
                "loss_global": float(loss_global.detach()),
                "loss_gram": float(loss_gram.detach()),
                "teacher_temp": float(t_temp),
            },
        }
>>

models/.DS_Store codice <<
   Bud1            %                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 @      €                                        @      €                                          @      €                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   E   %                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       DSDB                             `          €                                           @      €                                          @      €                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              >>

models/ibot.py codice <<
# models/ibot.py
from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.trainer.backbones import get_backbone, mlp_head, resolve_backbone_from_model_cfg
from src.training.utils.torch_ops import copy_weights_and_freeze, ema_update, l2n
from src.training.trainer.loops import SSLBaseModel

@torch.no_grad()
def sinkhorn(logits: torch.Tensor, iters: int=3, eps: float=1e-6) -> torch.Tensor:
    # subtract max per row before exp to avoid infs on CPU runs
    x = logits - logits.max(dim=1, keepdim=True).values
    Q = torch.exp(x).t()  # [K,B]
    s = Q.sum()
    if not torch.isfinite(s):
        Q = torch.nan_to_num(Q, nan=0.0, posinf=1e6, neginf=0.0)
        s = Q.sum().clamp_min(eps)
    Q /= s; K,B = Q.shape
    for _ in range(iters):
        Q /= (Q.sum(dim=1, keepdim=True) + eps)
        Q /= (Q.sum(dim=0, keepdim=True) + eps)
    Q = (Q.t() * K)
    Q = torch.nan_to_num(Q, nan=0.0, posinf=1e6, neginf=0.0)
    return Q.detach()

def make_mask(B:int, T:int, ratio:float, device) -> torch.Tensor:
    k = max(1, int(T*ratio))
    idx = torch.rand(B, T, device=device).argsort(dim=1)[:, :k]
    m = torch.zeros(B, T, dtype=torch.bool, device=device); m.scatter_(1, idx, True); return m

class IBOT(SSLBaseModel):
    def __init__(self, stu: ResNetBackbone, tea: ResNetBackbone,
                 head_cls_s: nn.Module, head_cls_t: nn.Module,
                 head_tok_s: nn.Module, head_tok_t: nn.Module,
                 prototypes: nn.Parameter, t_s: float=0.1, t_t: float=0.04, mask_ratio: float=0.5, ema_m: float=0.996):
        super().__init__()
        self.stu, self.tea = stu, tea
        self.hc_s, self.hc_t = head_cls_s, head_cls_t
        self.hp_s, self.hp_t = head_tok_s, head_tok_t
        self.prototypes = prototypes
        self.t_s, self.t_t, self.r, self.m = t_s, t_t, mask_ratio, ema_m
        self._bootstrap()

    @classmethod
    def from_config(cls, cfg: Dict[str,Any]) -> "IBOT":
        m = cfg["model"]["ssl"]
        bname, bopts = resolve_backbone_from_model_cfg(cfg["model"])
        stu = get_backbone(bname, pretrained=False, **bopts)
        tea = get_backbone(bname, pretrained=False, **bopts)
        dim = stu.out_dim
        head_cls_s = mlp_head(dim, 4096, 256); head_cls_t = mlp_head(dim, 4096, 256)
        head_tok_s = mlp_head(stu.out_dim, 2048, 256); head_tok_t = mlp_head(stu.out_dim, 2048, 256)
        # accept both 'num_prototypes' and 'prototypes'
        K = int(m.get("num_prototypes", m.get("prototypes", 8192)))
        proto = nn.Parameter(l2n(torch.randn(K, 256)), requires_grad=True)
        return cls(stu, tea, head_cls_s, head_cls_t, head_tok_s, head_tok_t, proto,
                   t_s=m.get("temp_student",0.1), t_t=m.get("temp_teacher",0.04),
                   mask_ratio=m.get("mask_ratio",0.5), ema_m=cfg["train"]["ssl"].get("ema_momentum",0.996))

    @torch.no_grad()
    def _bootstrap(self):
        copy_weights_and_freeze(self.tea, self.stu)

    def _loss_cls(self, xg: torch.Tensor) -> torch.Tensor:
        s = self.hc_s(self.stu.forward_global(xg)) / self.t_s
        with torch.no_grad():
            t = self.hc_t(self.tea.forward_global(xg)).detach() / self.t_t
            y = sinkhorn(t)  # soft assignments
        ce = -(y * F.log_softmax(s, dim=-1)).sum(dim=-1)
        return (ce / y.size(-1)).mean()  # <-- divide per K

    def _loss_tok(self, xl: torch.Tensor) -> torch.Tensor:
        ts = self.stu.forward_tokens(xl)         # [B,T,C]
        with torch.no_grad():
            tt = self.tea.forward_tokens(xl).detach()
        B,T,C = ts.shape; mask = make_mask(B,T,self.r,ts.device)   # [B,T]
        s = self.hp_s(ts[mask])                                    # [B*Tmask, 256]
        t = self.hp_t(tt[mask]).detach()                           # [B*Tmask, 256]
        s = F.normalize(s, dim=-1)
        t = F.normalize(t, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        s = (s @ proto.t()) / max(self.t_s, 1e-6)
        t = (t @ proto.t()) / max(self.t_t, 1e-6)
        s = torch.clamp(s, -50.0, 50.0)
        t = torch.clamp(t, -50.0, 50.0)
        y = sinkhorn(t)
        ce = -(y * F.log_softmax(s, dim=-1)).sum(dim=-1)
        return (ce / y.size(-1)).mean()  # <-- divide per K

    def training_step(self, batch: Dict[str,Any], global_step: int) -> Dict[str,Any]:
        images = batch["images"]
        if len(images) < 2:
            raise ValueError("iBOT requires two global views: got len(images) < 2.")
        xg1, xg2 = images[0], images[1]
        xl = xg1  # se non hai locali separati, usa una globale per il token loss

        with torch.no_grad():
            ema_update(self.tea, self.stu, self.m)

        loss_cls = 0.5 * (self._loss_cls(xg1) + self._loss_cls(xg2))
        loss_tok = self._loss_tok(xl)
        loss = loss_cls + loss_tok
        return {"loss_total": loss, "loss_components": {"loss_cls": float(loss_cls.detach()), "loss_tok": float(loss_tok.detach())}}
>>

models/i_jepa.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.trainer.backbones import resolve_backbone_from_model_cfg, get_backbone

class IJEPA(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        backbone_params: Dict[str, Any],
        predictor_depth: int = 6,
        predictor_embed_dim: int = 384,
        predictor_num_heads: int = 12,
        num_target_masks: int = 4,
        target_scale_range: Tuple[float, float] = (0.15, 0.2),
        context_scale_range: Tuple[float, float] = (0.85, 1.0),
        ema_decay: float = 0.996,
        img_size: int = 224,
        patch_size: int = 16,
        # Nuovo parametro per il filtraggio sfondo
        background_std_threshold: float = 0.02 
    ):
        super().__init__()
        
        # 1. Student (Context Encoder)
        self.stu = get_backbone(backbone_name, pretrained=False, **backbone_params)
        self.embed_dim = self.stu.out_dim
        
        # 2. Teacher (Target Encoder)
        self.tea = copy.deepcopy(self.stu)
        for p in self.tea.parameters():
            p.requires_grad = False
            
        # 3. Predictor
        self.predictor = IJEPA_Predictor(
            input_dim=self.embed_dim,
            depth=predictor_depth,
            embed_dim=predictor_embed_dim,
            num_heads=predictor_num_heads
        )
        
        # 4. Positional Embeddings
        self.num_patches = (img_size // patch_size) ** 2
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, predictor_embed_dim)
        )
        torch.nn.init.trunc_normal_(self.predictor_pos_embed, std=0.02)
        
        # Parametri
        self.num_target_masks = num_target_masks
        self.target_scale_range = target_scale_range
        self.context_scale_range = context_scale_range
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.ema_decay = ema_decay
        
        # Soglia per considerare un patch "vuoto" (solo sfondo)
        self.background_std_threshold = background_std_threshold

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> IJEPA:
        model_cfg = cfg["model"]
        ssl_cfg = model_cfg.get("ssl", {})
        backbone_name, backbone_opts = resolve_backbone_from_model_cfg(model_cfg)
        ijepa_cfg = ssl_cfg.get("i_jepa", {})
        
        img_size = cfg.get("data", {}).get("webdataset", {}).get("transform_train", {}).get("resize", 224)
        if isinstance(img_size, list): img_size = img_size[0]
        
        return cls(
            backbone_name=backbone_name,
            backbone_params=backbone_opts,
            predictor_depth=ijepa_cfg.get("predictor_depth", 6),
            predictor_embed_dim=ijepa_cfg.get("predictor_embed_dim", 384),
            predictor_num_heads=ijepa_cfg.get("predictor_num_heads", 6),
            num_target_masks=ijepa_cfg.get("num_target_masks", 4),
            ema_decay=ssl_cfg.get("ema_m", 0.996),
            img_size=int(img_size),
            patch_size=backbone_opts.get("patch_size", 16),
            # Puoi aggiungere questo parametro al config yaml sotto i_jepa se vuoi tunarlo
            background_std_threshold=ijepa_cfg.get("background_std_threshold", 0.02)
        )

    @torch.no_grad()
    def update_teacher(self):
        for param_q, param_k in zip(self.stu.parameters(), self.tea.parameters()):
            param_k.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * param_q.data)
            
    def training_step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, Any]:
        images = batch["images"]
        if isinstance(images, list):
            images = images[0]
        metrics = self(images)
        return {
            "loss_total": metrics["loss"],
            "loss_components": metrics
        }

    def _get_tissue_mask(self, images: torch.Tensor) -> torch.Tensor:
        """
        Analizza i pixel dell'immagine per creare una maschera che vale:
        1.0 se il patch contiene tessuto (informazione utile)
        0.0 se il patch Ã¨ sfondo piatto (bianco/vuoto)
        
        Input: [B, 3, H, W] -> Output: [B, Num_Patches]
        """
        B, C, H, W = images.shape
        P = self.patch_size
        
        # 1. Dividi in patch [B, 3, H//P, P, W//P, P] -> [B, N, 3*P*P]
        # Unfold manuale efficiente
        patches = images.unfold(2, P, P).unfold(3, P, P) 
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, self.num_patches, -1)
        
        # 2. Calcola Deviazione Standard per ogni patch
        # Se std Ã¨ basso, il patch Ã¨ piatto (sfondo bianco o nero uniforme)
        # Se std Ã¨ alto, c'Ã¨ texture (tessuto)
        patch_std = patches.std(dim=-1)
        
        # 3. Crea maschera binaria (float per moltiplicazione)
        tissue_mask = (patch_std > self.background_std_threshold).float()
        
        return tissue_mask

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = images.device
        B = images.shape[0]
        
        if self.training:
            self.update_teacher()

        # 1. Calcola Maschera Tessuto vs Sfondo
        tissue_mask = self._get_tissue_mask(images) # [B, N_patches]

        # 2. Generazione Maschere (Context e Target spaziali)
        context_masks, target_masks_list = self._generate_masks(B, device)

        # 3. Forward Teacher (Target Encoder)
        with torch.no_grad():
            full_teacher_tokens = self.tea.forward_tokens(images) 
            full_teacher_tokens = F.layer_norm(full_teacher_tokens, (full_teacher_tokens.size(-1),))
            if full_teacher_tokens.shape[1] == self.num_patches + 1:
                spatial_teacher_tokens = full_teacher_tokens[:, 1:, :]
            else:
                spatial_teacher_tokens = full_teacher_tokens

        # 4. Forward Student (Context Encoder)
        full_student_tokens = self.stu.forward_tokens(images)
        if full_student_tokens.shape[1] == self.num_patches + 1:
            spatial_student_tokens = full_student_tokens[:, 1:, :]
        else:
            spatial_student_tokens = full_student_tokens

        # Selezione Context Tokens
        context_tokens = spatial_student_tokens[context_masks].view(B, -1, self.embed_dim)

        # 5. Predictor Loop con Filtraggio Sfondo
        loss = 0.0
        valid_targets_count = 0 # Per evitare divisione per zero se tutto Ã¨ sfondo
        
        for i, target_mask in enumerate(target_masks_list):
            # Estrai target token
            target_tokens = spatial_teacher_tokens[target_mask].view(B, -1, self.embed_dim)
            
            # Estrai pos embeddings
            target_pos_embeds = self.predictor_pos_embed.expand(B, -1, -1)
            curr_target_pos = target_pos_embeds[target_mask].view(B, -1, self.predictor.embed_dim)
            
            # Predizione
            pred_tokens = self.predictor(context_tokens, curr_target_pos)
            
            # --- CALCOLO LOSS CON FILTRO TESSUTO ---
            # 1. Recuperiamo la maschera tessuto per QUESTI specifici target
            # [B, N_patches] -> [B, N_patches_kept] (appiattito o view dipendente dalla maschera)
            # PoichÃ© usiamo Batch Unified, possiamo usare .view()
            curr_tissue_mask = tissue_mask[target_mask].view(B, -1) # [B, N_targets_per_batch]
            
            # 2. Calcola MSE element-wise (senza mediare ancora)
            # [B, N_targets, D] -> mean su D -> [B, N_targets]
            mse_per_token = F.mse_loss(pred_tokens, target_tokens, reduction='none').mean(dim=-1)
            
            # 3. Applica filtro: Azzera loss dove Ã¨ sfondo
            masked_loss = mse_per_token * curr_tissue_mask
            
            # 4. Somma e normalizza solo per i token validi (tessuto)
            num_tissue_tokens = curr_tissue_mask.sum()
            
            if num_tissue_tokens > 0:
                loss += masked_loss.sum() / num_tissue_tokens
                valid_targets_count += 1
            
            # Se num_tissue_tokens Ã¨ 0 (tutto sfondo), loss += 0 (corretto, saltiamo questo target)

        # Media finale sui blocchi target validi
        if valid_targets_count > 0:
            loss = loss / valid_targets_count
        else:
            # Caso limite: immagine interamente bianca/sfondo -> Loss 0 ma con gradiente
            # Usiamo una dummy loss per non rompere DDP
            loss = (pred_tokens * 0.0).sum()

        return {"loss": loss, "ssl_loss": loss}

    def _generate_masks(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Batch Unified Masking Strategy"""
        h = w = self.grid_size
        
        def _get_block_mask_batch_unified(scale_range):
            min_s, max_s = scale_range
            scale = min_s + torch.rand(1).item() * (max_s - min_s)
            aspect = 0.75 + torch.rand(1).item() * 0.75
            num_patches = int(self.num_patches * scale)
            bh = max(min(int(math.sqrt(num_patches * aspect)), h), 1)
            bw = max(min(int(math.sqrt(num_patches / aspect)), w), 1)
            top = torch.randint(0, h - bh + 1, (1,), device=device).item()
            left = torch.randint(0, w - bw + 1, (1,), device=device).item()
            mask = torch.zeros((h, w), dtype=torch.bool, device=device)
            mask[top:top+bh, left:left+bw] = True
            return mask.view(1, -1).expand(batch_size, -1)

        target_masks = []
        union_target_mask = torch.zeros((batch_size, self.num_patches), dtype=torch.bool, device=device)
        
        for _ in range(self.num_target_masks):
            tm = _get_block_mask_batch_unified(self.target_scale_range)
            target_masks.append(tm)
            union_target_mask = union_target_mask | tm
            
        raw_context_mask = _get_block_mask_batch_unified(self.context_scale_range)
        context_mask = raw_context_mask & (~union_target_mask)
        
        if context_mask[0].sum() == 0:
            context_mask = raw_context_mask
        
        return context_mask, target_masks


class IJEPA_Predictor(nn.Module):
    def __init__(self, input_dim, depth, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_in = nn.Linear(input_dim, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * 4),
            dropout=0.0,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.proj_out = nn.Linear(embed_dim, input_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_tokens, target_pos_embeds):
        x = self.proj_in(context_tokens)
        full_seq = torch.cat([x, target_pos_embeds], dim=1)
        out = self.blocks(full_seq)
        out = self.norm(out)
        target_out = out[:, context_tokens.shape[1]:, :]
        return self.proj_out(target_out)>>

models/moco_v3.py codice <<
# models/moco_v3.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from src.training.trainer.backbones import get_backbone, mlp_head, predictor_head, resolve_backbone_from_model_cfg
from src.training.utils.torch_ops import copy_weights_and_freeze, cosine_logits, ema_update
from src.training.trainer.loops import SSLBaseModel

class CosineWithWarmup:
    """Cosine annealing from start->end with warmup in [0, warmup_frac]."""
    def __init__(self, start: float, end: float, warmup_frac: float = 0.1):
        self.start, self.end, self.warmup = float(start), float(end), float(warmup_frac)

    def at(self, t: float) -> float:
        t = min(max(t, 0.0), 1.0)
        if t < self.warmup:
            return self.start + (self.end - self.start) * (t / max(self.warmup, 1e-8))
        # cosine on the remaining segment
        import math
        tc = (t - self.warmup) / max(1.0 - self.warmup, 1e-8)
        return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * tc))
import math

class MoCoV3(SSLBaseModel):
    """
    MoCo v3 â€œno-queueâ€: due viste globali, encoder a momentum (teacher) + predictor.
    Negativi = in-batch. Loss simmetrizzata: ctr(q1,k2) + ctr(q2,k1).
    """
    def __init__(self, backbone_q: nn.Module, backbone_k: nn.Module,
                 proj_q: nn.Module, proj_k: nn.Module, pred_q: nn.Module,
                 tau: float = 0.2, momentum: float = 0.996,
                 *,
                 temp_teacher_sched: Optional[CosineWithWarmup]=None,
                 ema_to_one: bool=True,
                 use_multicrop: bool=False,
                 total_steps: int=10000,
                 clip_qk: float = 50.0,
                 sync_bn: bool = False):
        super().__init__()
        self.backbone_q, self.backbone_k = backbone_q, backbone_k
        self.proj_q, self.proj_k, self.pred_q = proj_q, proj_k, pred_q
        self.Ts = tau
        self.m0 = momentum
        self.m = momentum
        self.Tsched = temp_teacher_sched
        self.ema_to_one = bool(ema_to_one)
        self.use_multicrop = bool(use_multicrop)
        self.total_steps = int(max(1, total_steps))
        self._step = 0
        self.clip_qk = float(clip_qk)
        self._sync_bn_enabled = bool(sync_bn)

        self._bootstrap()

    @classmethod
    def from_config(cls, cfg: Dict[str,Any]) -> "MoCoV3":
        mcfg = cfg["model"]["ssl"]
        bname, bopts = resolve_backbone_from_model_cfg(cfg["model"])
        bb_q = get_backbone(name=bname, pretrained=False, **bopts)
        bb_k = get_backbone(name=bname, pretrained=False, **bopts)
        dim = bb_q.out_dim
        proj_dim = int(mcfg.get("proj_dim", 256))
        hid = int(mcfg.get("hidden_dim", 4096))
        proj_q = mlp_head(dim, hid, proj_dim); proj_k = mlp_head(dim, hid, proj_dim)
        pred_q = predictor_head(proj_dim, hid)
        tr_ssl = (cfg.get("train",{}).get("ssl",{}) or {})
        steps_per_epoch = int(tr_ssl.get("steps_per_epoch", 1000))
        epochs = int(tr_ssl.get("epochs", 10))
        total_steps = max(1, steps_per_epoch * epochs)
        # ---- Robust scheduling & numeric coercions (defensive vs. null/None) ----
        sched = (mcfg.get("temp_teacher_schedule") or {})
        Ts = None
        if sched:
            t_default = mcfg.get("temperature", 0.2)
            t_start  = sched.get("start", t_default)
            t_end    = sched.get("end",   t_default)
            t_warm   = sched.get("warmup_frac", 0.0)
            # Coercioni sicure: se i campi sono None o stringhe vuote, usa i default
            def _safe_float(v, d): 
                try:
                    return float(v if v is not None and v != "" else d)
                except (TypeError, ValueError):
                    return float(d)
            Ts = CosineWithWarmup(
                _safe_float(t_start, t_default),
                _safe_float(t_end,   t_default),
                warmup_frac=_safe_float(t_warm, 0.0),
            )
        clip_qk_val = mcfg.get("clip_qk", 50.0)
        # Se Ã¨ None/invalid, ripiega su 50.0 per stabilitÃ  numerica
        try:
            clip_qk_val = 50.0 if clip_qk_val is None else float(clip_qk_val)
        except (TypeError, ValueError):
            clip_qk_val = 50.0
        return cls(
            bb_q, bb_k, proj_q, proj_k, pred_q,
            tau=float(mcfg.get("temperature", 0.2) or 0.2),
            momentum=cfg["train"]["ssl"].get("ema_momentum",0.996),
            temp_teacher_sched=Ts,
            ema_to_one=bool(mcfg.get("ema_to_one", True)),
            use_multicrop=bool(mcfg.get("use_multicrop", False)),  # accettato ma ignoriamo le local per MoCo
            total_steps=total_steps,
            clip_qk=float(clip_qk_val),
            sync_bn=bool(mcfg.get("sync_bn", False)),
        )

    @torch.no_grad()
    def _bootstrap(self) -> None:
        copy_weights_and_freeze(self.backbone_k, self.backbone_q)
        copy_weights_and_freeze(self.proj_k, self.proj_q)
        # Converti BN in SyncBN solo su projector/predictor quando in DDP
        if self._sync_bn_enabled:
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    self.proj_q = nn.SyncBatchNorm.convert_sync_batchnorm(self.proj_q)
                    self.proj_k = nn.SyncBatchNorm.convert_sync_batchnorm(self.proj_k)
                    self.pred_q = nn.SyncBatchNorm.convert_sync_batchnorm(self.pred_q)
            except Exception:
                pass

    @torch.no_grad()
    def _ema_momentum(self) -> float:
        if not self.ema_to_one or self.total_steps <= 0:
            return self.m
        t = min(1.0, self._step / float(self.total_steps))
        # cosine to 1.0
        return 1.0 - (1.0 - self.m0) * 0.5 * (1.0 + math.cos(math.pi * t))

    def _teacher_temp(self) -> float:
        if self.Tsched is None or self.total_steps <= 0:
            return self.Ts
        t = min(1.0, self._step / float(self.total_steps))
        return float(self.Tsched.at(t))

    def _info_nce_sym(self, q1, q2, k1, k2) -> torch.Tensor:
        lab = torch.arange(q1.size(0), device=q1.device)
        Tt = self._teacher_temp()
        # clamp per stabilitÃ  numerica
        l12 = F.cross_entropy(torch.clamp(cosine_logits(q1, k2, Tt), -self.clip_qk, self.clip_qk), lab)
        l21 = F.cross_entropy(torch.clamp(cosine_logits(q2, k1, Tt), -self.clip_qk, self.clip_qk), lab)
        return (l12 + l21)

    def training_step(self, batch: Dict[str,Any], global_step: int) -> Dict[str,Any]:
        self._step = global_step
        imgs, meta = batch["images"], batch.get("meta", None)
        # Two global views required. If multicrop is enabled, imgs == [G, L] where
        # G is the stack of BOTH globals (shape: 2*B, C, H, W) and L are local crops.
        if self.use_multicrop:
            G = imgs[0]
            if G.dim() != 4 or (G.size(0) % 2 != 0):
                raise ValueError(f"MoCo v3 (multicrop) expects stacked 2 globals; got {tuple(G.shape)}.")
            # Split stacked globals into x1/x2
            x1, x2 = torch.chunk(G, 2, dim=0)
        else:
            if len(imgs) < 2:
                raise ValueError("MoCo v3 requires two global views.")
            x1, x2 = imgs[0], imgs[1]

        q1 = self.pred_q(self.proj_q(self.backbone_q.forward_global(x1)))
        q2 = self.pred_q(self.proj_q(self.backbone_q.forward_global(x2)))
        with torch.no_grad():
            ema_update(self.backbone_k, self.backbone_q, self._ema_momentum())
            k1 = self.proj_k(self.backbone_k.forward_global(x1))
            k2 = self.proj_k(self.backbone_k.forward_global(x2))
        # normalizza e clampa per robustezza
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1.detach(), dim=-1)
        k2 = torch.nn.functional.normalize(k2.detach(), dim=-1)
        loss_main = self._info_nce_sym(q1, q2, k1, k2)
        loss = loss_main

        # metriche diagnostiche
        with torch.no_grad():
            Tt = self._teacher_temp()
            pos_sim = float((q1 * k2).sum(dim=-1).mean().item())
            # media SOLO sugli off-diagonali (negativi)
            sim_mat = q1 @ k2.t()
            off = ~torch.eye(sim_mat.shape[0], dtype=torch.bool, device=sim_mat.device)
            neg_sim = float(sim_mat[off].mean().item())

        return {
            "loss_total": loss,
            "loss_components": {
                "loss_main": float(loss_main.detach()),
                "t_teacher": float(Tt),
                "ema_m": float(self._ema_momentum()),
                "pos_sim": pos_sim,
                "neg_sim": neg_sim,
            },
        }

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone_q.forward_global(x)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone_q.forward_tokens(x)
>>

models/supervised.py codice <<
# models/supervised.py
from __future__ import annotations
import torch.nn as nn
from torchvision import models

def _head(in_f: int, num_classes: int, p: float=0.0) -> nn.Module:
    layers = ([nn.Dropout(p)] if p>0 else []) + [nn.Linear(in_f, num_classes)]
    return nn.Sequential(*layers)

def build_resnet_scratch(name: str, num_classes: int, dropout_p: float=0.0):
    if name=="resnet34_scratch": m = models.resnet34(weights=None)
    elif name=="resnet50_scratch": m = models.resnet50(weights=None)
    else: raise ValueError("name must be resnet34_scratch|resnet50_scratch")
    m.fc = _head(m.fc.in_features, num_classes, dropout_p)
    return m, None  # (model, transforms) per simmetria con transfer
>>

models/transfer.py codice <<
# models/transfer.py
from __future__ import annotations
import torch.nn as nn
from torchvision import models

def _freeze_except_head(m: nn.Module) -> None:
    for n,p in m.named_parameters():
        if not n.startswith("fc."): p.requires_grad=False

def build_resnet_transfer(name: str, num_classes: int, weights_tag="DEFAULT",
                          freeze_backbone=False, dropout_p: float=0.0, bn_eval_freeze=False):
    if "34" in name:
        weights_enum = models.ResNet34_Weights
        weights = getattr(weights_enum, weights_tag) if weights_tag else None
        m = models.resnet34(weights=weights)
    elif "50" in name:
        weights_enum = models.ResNet50_Weights
        weights = getattr(weights_enum, weights_tag) if weights_tag else None
        m = models.resnet50(weights=weights)
    else:
        raise ValueError("name must contain 34 or 50")
    in_f = m.fc.in_features
    m.fc = nn.Sequential(*( [nn.Dropout(dropout_p)] if dropout_p>0 else [] ), nn.Linear(in_f, num_classes))
    tfm = weights.transforms()
    if freeze_backbone: _freeze_except_head(m)
    if bn_eval_freeze and freeze_backbone:
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d): mod.eval()
    return m, tfm
>>

orchestrator.py codice <<

# src/training/orchestrator.py
from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

import torch

try:  # pragma: no cover - optional dependency
    from sklearn.metrics import confusion_matrix
except ImportError:  # pragma: no cover
    def confusion_matrix(y_true, y_pred):
        return [[0]]

def _f1_macro_np(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    if yt.size == 0 or yp.size == 0:
        return 0.0
    classes = np.unique(np.concatenate([yt, yp]))
    f1s = []
    for c in classes:
        tp = ((yp == c) & (yt == c)).sum()
        fp = ((yp == c) & (yt != c)).sum()
        fn = ((yp != c) & (yt == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0

from .datasets import build_sl_loaders, build_ssl_loader_from_cfg, class_labels_from_cfg, device_from_env
from .trainer.features import save_features, train_linear_probe_torch, visualize_features_umap_pca
from .utils.io import append_row_csv, copy_yaml_config, dump_json, make_exp_id, make_run_dirs, prefixed
try:
    from .utils.torch_ops import safe_state_dict
except ModuleNotFoundError:  # pragma: no cover - fallback when namespace packages misbehave
    import importlib.util
    import sys
    from pathlib import Path as _Path

    _torch_ops_path = _Path(__file__).resolve().parent / "utils" / "torch_ops.py"
    spec = importlib.util.spec_from_file_location("src.training.utils.torch_ops", _torch_ops_path)
    module = importlib.util.module_from_spec(spec) if spec and spec.loader else None
    if module and spec and spec.loader:
        spec.loader.exec_module(module)
        sys.modules.setdefault("src.training.utils.torch_ops", module)
        safe_state_dict = module.safe_state_dict  # type: ignore[attr-defined]
    else:  # pragma: no cover
        raise
from .trainer.loops import SLTrainer, SSLTrainer
from .utils.viz import plot_confusion, render_all_sl, render_all_ssl, render_ssl_classifier

# ---- modelli (riuso tuoi) ----
try:
    from .models.moco_v3 import MoCoV3
    from .models.dino_v3 import DINOv3
    from .models.ibot import IBOT
    from .models.i_jepa import IJEPA
    from .models.supervised import build_resnet_scratch
    from .models.transfer import build_resnet_transfer
except ImportError:  # pragma: no cover - lightweight fallback
    def MoCoV3(cfg): return None  # type: ignore[override]
    def DINOv3(cfg): return None  # type: ignore[override]
    def IBOT(cfg): return None  # type: ignore[override]
    def IJEPA(cfg): return None  # type: ignore[override]
    def build_resnet_scratch(*args): return None  # type: ignore[override]
    def build_resnet_transfer(*args): return None  # type: ignore[override]


def _with_context(tag: str, fn: Callable, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"[{tag}] {exc.__class__.__name__}: {exc}") from exc


def _validate_config(cfg: Dict[str, Any]) -> None:
    wds = (cfg.get("data", {}).get("webdataset", {}) or {})
    validate_cfg = (cfg.get("experiment", {}).get("validate", {}) or {})
    if validate_cfg.get("paths", True):
        missing = [key for key in ("train_dir", "val_dir") if wds.get(key) and not Path(wds[key]).exists()]
        if missing:
            raise FileNotFoundError(f"Missing WebDataset directories: {missing}")
    if not wds.get("class_to_id"):
        raise ValueError("data.webdataset.class_to_id must be populated.")
    if validate_cfg.get("steps_per_epoch", True):
        steps = ((cfg.get("train", {}) or {}).get("ssl", {}) or {}).get("steps_per_epoch")
        if steps is not None and steps <= 0:
            raise ValueError("train.ssl.steps_per_epoch must be > 0.")
    if validate_cfg.get("batch_sizes", True):
        for key in ("batch_size_ssl", "batch_size_sl"):
            bs = wds.get(key)
            if bs is not None and bs <= 0:
                raise ValueError(f"data.webdataset.{key} must be > 0.")


def _log_every_steps(cfg: Dict[str, Any]) -> int:
    return int((cfg.get("logging", {}) or {}).get("log_every_steps", 0))


def _ssl_model_factory(cfg: Dict[str, Any]) -> torch.nn.Module:
    name = cfg["model"]["ssl"]["name"].lower()
    mapping = {
        "moco_v3": MoCoV3.from_config,
        "dino_v3": DINOv3.from_config,
        "ibot": IBOT.from_config,
        "i_jepa": IJEPA.from_config,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported SSL model family '{name}'.")
    return mapping[name](cfg)


def _sl_model_factory(cfg: Dict[str, Any]) -> tuple[torch.nn.Module, Any]:
    name = cfg["model"]["sl"]["name"]
    numc = len(cfg["data"]["webdataset"]["class_to_id"])
    if name.endswith("_scratch"):
        model, tfm = build_resnet_scratch(name, numc, cfg["model"]["sl"].get("dropout_p", 0.0))
        return model, tfm
    model, tfm = build_resnet_transfer(
        name,
        numc,
        cfg["model"]["sl"].get("imagenet_weights", "DEFAULT"),
        cfg["model"]["sl"].get("freeze_backbone", False),
        cfg["model"]["sl"].get("dropout_p", 0.0),
        cfg["model"]["sl"].get("bn_eval_freeze", False),
    )
    return model, tfm


class Orchestrator:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = copy.deepcopy(cfg)
        _validate_config(self.cfg)
        allow_cpu = bool(self.cfg.get("experiment", {}).get("allow_cpu", False))
        self.device = device_from_env(allow_cpu=allow_cpu)
        self.mode = self.cfg.get("_runtime", {}).get("mode") or self.cfg.get("model", {}).get("type", "ssl")
        self.model_key = self.cfg["model"]["ssl"]["name"] if self.mode == "ssl" else self.cfg["model"]["sl"]["name"]
        runtime = self.cfg.setdefault("_runtime", {})
        provided_exp_id = runtime.get("exp_id") or self.cfg.get("experiment", {}).get("id")
        outputs_root = self.cfg["experiment"]["outputs_root"]
        if provided_exp_id:
            self.exp_id = provided_exp_id
        else:
            self.exp_id = make_exp_id(outputs_root)
        runtime["exp_id"] = self.exp_id
        self.cfg.setdefault("experiment", {})["id"] = self.exp_id
        # Honor explicit group/leaf overrides set by the sbatch launcher:
        subdir_override = os.environ.get("EXP_SUBDIR", "") or self.cfg.get("_runtime", {}).get("run_subdir", "")
        group_dir_env   = os.environ.get("OUTPUTS_GROUP_DIR", "")
        if subdir_override:
            # New behavior: run root = <outputs>/experiments/<exp_id>/<exp_subdir>
            # No extra "model_key" folder unless make_run_dirs chooses to add it.
            self.run_dirs = make_run_dirs(
                outputs_root,
                self.exp_id,
                subdir_override,
                self.model_key,
                override_leaf=True,
                outputs_group_dir=group_dir_env or None,
            )
        else:
            self.run_dirs = make_run_dirs(outputs_root, self.exp_id, self.cfg["experiment"]["name"], self.model_key)
        config_path = self.cfg.get("_runtime", {}).get("config_path") or os.environ.get("EXPERIMENT_CONFIG_PATH")
        copy_yaml_config(config_path, self.run_dirs["configuration"])
        self.override_transforms: Optional[Any] = None

    def fit(self) -> Dict[str, Any]:
        if self.mode == "ssl":
            metrics = self._fit_ssl()
        elif self.mode == "sl":
            metrics = self._fit_sl()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        self._finalize_run(metrics)
        return metrics

    def _finalize_run(self, metrics: Dict[str, Any]) -> None:
        serializable: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, Path):
                serializable[key] = str(value)
            elif isinstance(value, (int, float, str, bool)) or value is None:
                serializable[key] = value
            else:
                try:
                    serializable[key] = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    serializable[key] = str(value)
        dump_json(self.run_dirs["metrics"] / "final_metrics.json", serializable)

    # ------------------------------------------------------------------ helpers
    def _build_optimizer(self, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        conf = self.cfg["train"]["optim"]
        name = conf.get("name", "adamw").lower()
        lr = conf["lr"]
        weight_decay = conf.get("weight_decay", 5e-2)
        if name == "adamw":
            betas = tuple(conf.get("betas", (0.9, 0.999)))
            extra: Dict[str, Any] = {}
            try:
                if torch.cuda.is_available():
                    extra["fused"] = True
            except TypeError:
                pass
            return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay, **extra)
        if name == "sgd":
            momentum = conf.get("momentum", 0.9)
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer '{name}'.")

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, total_units: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        sched_cfg = (self.cfg["train"].get("scheduler") or {})
        name = sched_cfg.get("name", "").lower()
        if name == "cosine":
            # Nota: in SSL usiamo 'steps' come unitÃ ; in SL 'epochs'
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_units)
        return None

    # ------------------------------------------------------------------ SSL path
    def _fit_ssl(self) -> Dict[str, float]:
        # --- OPTIONAL: dump a few augmented samples before training ---
        try:
            viz_cfg = ((self.cfg.get("viz", {}) or {}).get("dump_augmentations", {}) or {})
            env_switch = os.environ.get("DUMP_AUGS", "0") == "1"
            if bool(viz_cfg.get("enable", False)) or env_switch:
                from src.training.tools.dump_augmentations import dump_from_config
                out_root = viz_cfg.get(
                    "out_root",
                    "/home/mla_group_01/rcc-ssrl/src/training/configs/ablations/augms"
                )
                per_class = int(viz_cfg.get("per_class", 2))
                dump_from_config(self.cfg, out_root=out_root, per_class=per_class, seed=self.cfg["experiment"].get("seed"))
        except Exception as e:
            print(f"[viz] dump_augmentations failed (non-fatal): {e}")
        # ----------------------------------------------------------------
        model = _with_context("build_ssl_model", _ssl_model_factory, self.cfg)
        loader = _with_context("build_ssl_loader", build_ssl_loader_from_cfg, self.cfg, "train")
        print(f"[RUN][{self.model_key}] device={self.device.type}")

        ssl_cfg = self.cfg["train"]["ssl"]
        optimizer = self._build_optimizer(model.parameters())

        # --- Determine steps_per_epoch once (never call len(loader) if IterableDataset) ---
        steps_per_epoch = ssl_cfg.get("steps_per_epoch")
        if steps_per_epoch is None:
            wds_cfg = self.cfg["data"]["webdataset"]
            samples = wds_cfg.get("samples_per_epoch")
            if samples:
                import math
                bs = int(wds_cfg["batch_size_ssl"])
                steps_per_epoch = max(1, math.ceil(int(samples) / max(1, bs)))
            else:
                try:
                    steps_per_epoch = len(loader)
                except TypeError as e:
                    raise ValueError(
                        "Missing train.ssl.steps_per_epoch and data.webdataset.samples_per_epoch "
                        "for an IterableDataset/WebDataset."
                    ) from e
        steps_per_epoch = max(1, int(steps_per_epoch))
        epochs = int(ssl_cfg["epochs"])
        total_steps = steps_per_epoch * epochs

        # Cosine su 'unitÃ ' del programma (per SSL=steps, per SL=epochs)
        scheduler = self._build_scheduler(optimizer, total_steps)
        trainer = SSLTrainer(
            model,
            optimizer,
            scheduler=scheduler,
            ema_m=float(ssl_cfg.get("ema_m", 0.0)),
            device=self.device,
            log_every_steps=_log_every_steps(self.cfg),
            log_tag=self.model_key,
            grad_clip_max=float(ssl_cfg.get("grad_clip_max", 0.0)),
            accumulate_steps=int(ssl_cfg.get("accumulate_steps", 1)),
            amp=bool(ssl_cfg.get("amp", True)),
        )

        t0_run = time.time()

        from pathlib import Path as _P
        csv_stem = _P((self.cfg.get("logging", {}) or {}).get("metrics_csv_name", "ssl_timeseries.csv")).stem
        csv_path = prefixed(self.run_dirs["metrics"], self.model_key, csv_stem, "csv")
        best_loss = float("inf")
        best_epoch = -1
        best_state = None
        backbone_ckpt = prefixed(self.run_dirs["checkpoints"], self.model_key, "ssl_best", "pt")
        global_step_offset = 0
        log_every = max(1, _log_every_steps(self.cfg))

        def _epoch_mode() -> Callable:
            name = self.cfg["model"]["ssl"]["name"].lower()
            if name == "moco_v3":
                use_mc = bool((self.cfg.get("model",{}).get("ssl",{}) or {}).get("use_multicrop", False))
                return trainer.train_epoch_multicrop if use_mc else trainer.train_epoch_two_views
            if name == "dino_v3":
                return trainer.train_epoch_multicrop
            if name == "ibot":
                use_mc = bool((self.cfg.get("model",{}).get("ssl",{}) or {}).get("use_multicrop", False))
                return trainer.train_epoch_multicrop if use_mc else trainer.train_epoch_two_views
            if name == "i_jepa":
                return trainer.train_epoch_single_image
            raise ValueError(f"Unsupported SSL model '{name}'.")

        run_epoch = _epoch_mode()

        for epoch in range(epochs):
            def _log_step(global_step: int, stats: Dict[str, float], epoch_idx: int = epoch) -> None:
                row = {
                    "epoch": epoch_idx,
                    "step": global_step,
                    "lr": optimizer.param_groups[0].get("lr", 0.0),
                }
                # ETA globale (fino a fine run)
                done = min(global_step, total_steps)
                elapsed = time.time() - t0_run
                frac = max(1e-9, float(done) / float(total_steps))
                eta_s = max(0.0, elapsed * (1.0 - frac) / frac)
                row.update({"elapsed_s": round(elapsed, 2), "eta_s": round(eta_s, 2)})
                # mantieni sottochiavi piatte per CSV
                for k,v in stats.items():
                    row[k] = v
                append_row_csv(csv_path, row)
                # Log solo da rank 0 per evitare rumore
                if os.environ.get("RANK", "0") != "0":
                    return
                # Log only every log_every steps or at the last step to avoid double logging
                if (global_step - 1) % log_every != 0 and global_step != total_steps:
                    return
                # Log leggibile sullo stdout SLURM
                step_in_epoch = ((global_step - epoch_idx * steps_per_epoch - 1) % steps_per_epoch) + 1
                metrics_msg = " ".join(
                    f"{key}={float(value):.4f}" if isinstance(value, (int, float)) else f"{key}={value}"
                    for key, value in sorted(stats.items())
                )
                # Stima oraria di fine
                eta_h = int(eta_s // 3600); eta_m = int((eta_s % 3600) // 60); eta_sec = int(eta_s % 60)
                print(
                    f"[{self.model_key}][epoch {epoch_idx + 1}/{epochs}] "
                    f"step {step_in_epoch}/{steps_per_epoch} (global {global_step}/{total_steps}) "
                    f"{metrics_msg} | ETA={eta_h:02d}:{eta_m:02d}:{eta_sec:02d}",
                    flush=True,
                )

            epoch_stats = run_epoch(loader, steps_per_epoch, start_step=global_step_offset, step_callback=_log_step)
            global_step_offset += steps_per_epoch
            # Be tolerant to different metric keys from the trainer:
            loss_epoch = float(
                epoch_stats.get(
                    "loss_total",
                    epoch_stats.get("ssl_loss_ema", epoch_stats.get("ssl_loss", float("inf")))
                )
            )
            if loss_epoch < best_loss:
                best_loss = loss_epoch
                best_epoch = epoch
                best_state = safe_state_dict(model)

            if hasattr(model, "on_epoch_end"):
                model.on_epoch_end(epoch)

        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
            model.to(self.device)
            torch.save(best_state, backbone_ckpt)

        # --- After writing metrics CSV, emit derived CSV with smoothing ---
        try:
            from src.training.utils.viz import write_derived_csv
            log_cfg = (self.cfg.get("logging", {}) or {})
            window = int(log_cfg.get("smoothing_window", 50))
            ema_m = float((self.cfg.get("train", {}).get("ssl", {}) or {}).get("ema_m", 0.0))
            derived_csv_path = write_derived_csv(str(csv_path), target_col="ssl_loss", sma_window=window,
                              ema_m=(ema_m if ema_m > 0 else None))
            print(f"[viz] Derived CSV written to: {derived_csv_path}")
        except Exception as e:
            print(f"[viz] Failed to write derived CSV: {e}")
            pass

        # Render SSL plots; never fail the whole run because of viz
        try:
            render_all_ssl(csv_path, self.run_dirs["plots"], self.model_key)
        except Exception as e:
            import sys, traceback
            print(f"[viz][WARNING] Plotting failed: {e}", file=sys.stderr)
            traceback.print_exc()

        backbone_rel = str(backbone_ckpt.relative_to(self.run_dirs["root"])) if backbone_ckpt.exists() else ""
        metrics_path = prefixed(self.run_dirs["metrics"], self.model_key, "ssl_summary", "json")
        dump_json(
            metrics_path,
            {
                "best_epoch": int(best_epoch),
                "ssl_loss": float(best_loss),
                "ssl_backbone_path": backbone_rel,
            },
        )

        ssl_summary: Dict[str, float | str] = {
            "ssl_best_epoch": int(best_epoch),
            "ssl_loss": float(best_loss),
            "ssl_backbone_path": backbone_rel,
        }

        # ------------------------------------------------------------------ feature extraction + linear probe
        train_loader, val_loader = _with_context(
            "build_sl_loaders",
            build_sl_loaders,
            self.cfg,
            override_transforms=self.override_transforms,
        )
        loaders = {"train": train_loader, "val": val_loader}
        backbone_module = model.stu if hasattr(model, "stu") else model

        feature_paths = save_features(backbone_module, loaders, self.device, self.run_dirs["checkpoints"], self.model_key)

        clf_summary: Dict[str, object] = {"ssl_linear_status": "skipped"}
        required_keys = {"train_X", "train_y", "val_X", "val_y"}
        if required_keys.issubset(feature_paths.keys()):
            Xtr = np.load(feature_paths["train_X"], allow_pickle=False)
            ytr = np.load(feature_paths["train_y"], allow_pickle=False)
            Xva = np.load(feature_paths["val_X"], allow_pickle=False)
            yva = np.load(feature_paths["val_y"], allow_pickle=False)

            if Xtr.size and Xva.size:
                visualize_features_umap_pca(
                    np.vstack([Xtr, Xva]),
                    np.hstack([ytr, yva]),
                    prefixed(self.run_dirs["plots"], self.model_key, "ssl_features_umap", "png"),
                    labels=class_labels_from_cfg(self.cfg),
                )

                probe_cfg = (self.cfg.get("train", {}).get("ssl", {}).get("probe", {}) or {})
                lin_metrics, lin_ckpt = train_linear_probe_torch(
                    Xtr,
                    ytr,
                    Xva,
                    yva,
                    n_epochs=int(probe_cfg.get("epochs", 5)),
                    lr=float(probe_cfg.get("lr", 0.01)),
                    wd=float(probe_cfg.get("weight_decay", 0.0)),
                    batch_size=int(probe_cfg.get("batch_size", 128)),
                    out_dirs=self.run_dirs,
                    model_key=self.model_key,
                )

                lin_csv = prefixed(self.run_dirs["metrics"], self.model_key, "ssl_linear_timeseries", "csv")
                if lin_csv.exists():
                    render_ssl_classifier(lin_csv, self.run_dirs["plots"], self.model_key)

                lin_ckpt_rel = str(Path(lin_ckpt).relative_to(self.run_dirs["root"])) if lin_ckpt else ""
                feature_paths_rel: Dict[str, str] = {}
                for key, path in feature_paths.items():
                    candidate = Path(path)
                    try:
                        feature_paths_rel[key] = str(candidate.relative_to(self.run_dirs["root"]))
                    except ValueError:
                        feature_paths_rel[key] = str(candidate)
                dump_json(
                    prefixed(self.run_dirs["metrics"], self.model_key, "ssl_linear_summary", "json"),
                    {
                        "val_acc": float(lin_metrics.get("val_acc", float("nan"))),
                        "checkpoint": lin_ckpt_rel,
                        "features": feature_paths_rel,
                    },
                )

                try:
                    lin_csv_rel = str(lin_csv.relative_to(self.run_dirs["root"]))
                except ValueError:
                    lin_csv_rel = str(lin_csv)
                clf_summary = {
                    "probe_linear_val_acc": float(lin_metrics.get("val_acc", float("nan"))),
                    "ssl_linear_ckpt_path": lin_ckpt_rel,
                    "ssl_linear_features": feature_paths_rel,
                    "ssl_linear_timeseries": lin_csv_rel if lin_csv.exists() else "",
                }

        return {**ssl_summary, **clf_summary}

    # ------------------------------------------------------------------ SL path
    def _fit_sl(self) -> Dict[str, float]:
        model, self.override_transforms = _with_context("build_sl_model", _sl_model_factory, self.cfg)
        train_loader, val_loader = _with_context(
            "build_sl_loaders", build_sl_loaders, self.cfg, override_transforms=self.override_transforms
        )
        loaders = {"train": train_loader, "val": val_loader}
        print(f"[RUN][{self.model_key}] device={self.device.type}")

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params)
        scheduler = self._build_scheduler(optimizer, int(self.cfg["train"]["sl"]["epochs"]))
        criterion = torch.nn.CrossEntropyLoss()
        trainer = SLTrainer(
            model,
            criterion,
            optimizer,
            scheduler=scheduler,
            amp=bool(self.cfg["train"]["sl"].get("amp", True)),
            log_tag=self.model_key,
            log_every_steps=_log_every_steps(self.cfg),
        )

        csv_path = prefixed(self.run_dirs["metrics"], self.model_key, "sl_timeseries", "csv")
        best_state = None
        best_epoch = -1
        best_acc = -1.0
        best_loss = float("inf")
        classifier_ckpt = prefixed(self.run_dirs["checkpoints"], self.model_key, "sl_best_classifier", "pt")
        epochs = int(self.cfg["train"]["sl"]["epochs"])
        t0_run = time.time()

        for epoch in range(epochs):
            train_metrics = trainer.run_epoch(loaders["train"], self.device, train=True)
            val_metrics = trainer.run_epoch(loaders["val"], self.device, train=False)
            lr = optimizer.param_groups[0].get("lr", 0.0)
            append_row_csv(
                csv_path,
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                    "lr": lr,
                    "elapsed_s": round(time.time() - t0_run, 2),
                },
            )
            # Stima ETA fine training (lineare sui tempi per epoca)
            if os.environ.get("RANK", "0") == "0":
                done_epochs = epoch + 1
                elapsed = time.time() - t0_run
                frac = max(1e-9, done_epochs / float(epochs))
                eta_s = max(0.0, elapsed * (1.0 - frac) / frac)
                eta_h = int(eta_s // 3600); eta_m = int((eta_s % 3600) // 60); eta_sec = int(eta_s % 60)
                print(
                    f"[{self.model_key}][epoch {done_epochs}/{epochs}] "
                    f"val_acc={val_metrics['acc']:.4f} val_loss={val_metrics['loss']:.4f} "
                    f"| ETA={eta_h:02d}:{eta_m:02d}:{eta_sec:02d}",
                    flush=True,
                )

            if val_metrics["acc"] > best_acc:
                best_acc = val_metrics["acc"]
                best_loss = val_metrics["loss"]
                best_epoch = epoch
                best_state = safe_state_dict(model)

        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
            model.to(self.device)
            torch.save(best_state, classifier_ckpt)

        render_all_sl(csv_path, self.run_dirs["plots"], self.model_key)

        final_metrics = self._evaluate_sl(model, loaders["val"], criterion)
        metrics_path = prefixed(self.run_dirs["metrics"], self.model_key, "sl_summary", "json")
        dump_json(
            metrics_path,
            {
                "best_epoch": int(best_epoch),
                "val_acc": float(final_metrics["val_acc"]),
                "val_f1_macro": float(final_metrics["val_f1_macro"]),
                "val_loss": float(final_metrics["val_loss"]),
                "sl_classifier_path": str(classifier_ckpt.relative_to(self.run_dirs["root"])) if classifier_ckpt.exists() else "",
            },
        )

        return {
            "sl_best_epoch": int(best_epoch),
            "sl_val_acc": float(final_metrics["val_acc"]),
            "sl_val_f1_macro": float(final_metrics["val_f1_macro"]),
            "sl_val_loss": float(final_metrics["val_loss"]),
            "sl_classifier_path": str(classifier_ckpt.relative_to(self.run_dirs["root"])) if classifier_ckpt.exists() else "",
        }

    def _evaluate_sl(self, model: torch.nn.Module, loader, criterion: torch.nn.Module) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.device, non_blocking=True)
                targets = batch["targets"].to(self.device, non_blocking=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                preds = logits.argmax(1)
                total_loss += float(loss.detach()) * targets.size(0)
                total_correct += float((preds == targets).sum().item())
                total_samples += targets.size(0)
                y_true.extend(targets.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        avg_loss = total_loss / max(1, total_samples)
        avg_acc = total_correct / max(1, total_samples)
        cm = confusion_matrix(y_true, y_pred)
        labels = class_labels_from_cfg(self.cfg)
        if not labels:
            labels = [str(i) for i in range(len(cm))]
        plot_confusion(cm, labels, prefixed(self.run_dirs["plots"], self.model_key, "sl_confusion_val", "png"))
        return {
            "val_loss": avg_loss,
            "val_acc": avg_acc,
            "val_f1_macro": float(_f1_macro_np(y_true, y_pred)),
        }
>>

requirements.txt codice <<
torch>=2.5.0
torchvision>=0.20.0
torchaudio>=2.5.0
webdataset>=0.2.100,<1.0
braceexpand>=0.1.7
pyarrow>=15.0.2
pandas>=2.3.3
scikit-learn>=1.4.2
pyyaml>=6.0
matplotlib>=3.9.0
seaborn>=0.13.0
mlflow>=3.1.0
umap-learn>=0.5.6
timm>=1.0.7
>>

slurm/launch_train_job.sbatch codice <<
#!/usr/bin/env bash
#SBATCH --job-name=rcc-train
#SBATCH --account=mla_group_01
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1                    # 3 max che puoi usare ora
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/%x.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/%x.%j.err
set -euo pipefail

# Standardized roots for HPC
export HOME_ROOT="${HOME_ROOT:-/home/mla_group_01}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/beegfs-scratch/mla_group_01/workspace/mla_group_01}"
export PROJECT_ROOT="${PROJECT_ROOT:-${HOME_ROOT}/rcc-ssrl}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs/mlruns}"
export RCC_DATASET_ROOT="${RCC_DATASET_ROOT:-${SCRATCH_ROOT}/wsi-ssrl-rcc_project/data/processed}"
export RCC_WDS_ROOT="${RCC_WDS_ROOT:-${RCC_DATASET_ROOT}/rcc_webdataset_final}"
export WDS_ROOT="${WDS_ROOT:-${RCC_WDS_ROOT}}"
export WEB_DATASET_ROOT="${WEB_DATASET_ROOT:-${RCC_WDS_ROOT}}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# 1) Workdir
WORKDIR="${SLURM_SUBMIT_DIR:-$PROJECT_ROOT}"
if [[ "$WORKDIR" == *"/src/training" ]]; then
  WORKDIR="$(dirname "$(dirname "$WORKDIR")")"
fi
cd "$WORKDIR"

# 2) Env
source "${WORKDIR}/src/training/scripts/bootstrap_env.sh"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:$PYTHONPATH}"

# NCCL (single-node, A40)
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0

# 3) Staging dataset su scratch locale (se disponibile)
STAGING_DIR=""
if [[ -n "${SLURM_TMPDIR:-}" && -d "$SLURM_TMPDIR" ]]; then
  STAGING_DIR="$SLURM_TMPDIR"
elif [[ -n "${TMPDIR:-}" && -d "$TMPDIR" ]]; then
  STAGING_DIR="$TMPDIR"
elif [[ -n "${SCRATCH:-}" && -d "${SCRATCH:-}" ]]; then
  STAGING_DIR="${SCRATCH:-}/$USER"
fi


# 4) Sanity log
nvidia-smi -L || true
python - <<'PY'
import torch
print("Torch", torch.__version__, "CUDA", torch.cuda.is_available(), "NGPU", torch.cuda.device_count())
PY

# Dataset visibility check on the allocated node
srun -N1 -n1 bash -lc 'hostname; ls -ld ${RCC_DATASET_ROOT}/rcc_webdataset_final/train' || {
  echo "[sbatch] Dataset path not visible on this node" >&2
  exit 4
}

# Fail-fast preflight (dataset presence + shards)
PYTHONPATH="${PYTHONPATH}" python -m src.training.utils.preflight || {
  echo "[sbatch] Preflight failed" >&2
  exit 5
}

# 5) Project paths
# Cambia questa variabile se vuoi un'altra config
export EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-${WORKDIR}/src/training/configs/exp_debug_pipeline.yaml}"
export RUN_INDEX="${RUN_INDEX:--1}"

# 6) DDP: lancia 1 processo per GPU (3 proc)
NPROC="${SLURM_GPUS_ON_NODE:-3}"

# 7) Run con torchrun (standalone su singolo nodo)
exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  src/training/launch_training.py
>>

slurm/train_single_node.sbatch codice <<
#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/%x.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/%x.%j.err
# The job name and log paths are overridden by --job-name/--output/--error from sbatch

set -euo pipefail

# -------- Standardized roots (HPC-safe) --------------------------------------
export HOME_ROOT="${HOME_ROOT:-/home/mla_group_01}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/beegfs-scratch/mla_group_01/workspace/mla_group_01}"
export PROJECT_ROOT="${PROJECT_ROOT:-${HOME_ROOT}/rcc-ssrl}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs/mlruns}"
export RCC_DATASET_ROOT="${RCC_DATASET_ROOT:-${SCRATCH_ROOT}/wsi-ssrl-rcc_project/data/processed}"
export RCC_WDS_ROOT="${RCC_WDS_ROOT:-${RCC_DATASET_ROOT}/rcc_webdataset_final}"
export WDS_ROOT="${WDS_ROOT:-${RCC_WDS_ROOT}}"
export WEB_DATASET_ROOT="${WEB_DATASET_ROOT:-${RCC_WDS_ROOT}}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# -------- User/Repo paths -----------------------------------------------------
ROOT="${PROJECT_ROOT}"
TRAIN_DIR="${ROOT}/src/training"

cd "${TRAIN_DIR}"

# -------- Config argument (REQUIRED) ------------------------------------------
# Accept config YAML as first CLI arg; fail fast if missing
CONFIG_PATH="${1:-}"
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "[sbatch] ERROR: missing CONFIG_PATH argument (YAML file)." >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[sbatch] ERROR: config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

# -------- Environment (venv/modules) ------------------------------------------
if [[ -d "${ROOT}/.venv" ]]; then
  source "${ROOT}/.venv/bin/activate"
fi

# Optional: user can set GPUS_PER_NODE externally (defaults to 1)
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

echo "[sbatch] Host: $(hostname)"
echo "[sbatch] GPUs per node: ${GPUS_PER_NODE}"
echo "[sbatch] Config: ${CONFIG_PATH}"
echo "[sbatch] RUN_NAME=${RUN_NAME:-}"
echo "[sbatch] EXP_GROUP=${EXP_GROUP:-}"
echo "[sbatch] OUTPUTS_GROUP_DIR=${OUTPUTS_GROUP_DIR:-}"
echo "[sbatch] RCC_DATASET_ROOT=${RCC_DATASET_ROOT}"

# Export several conventional env vars so Python can locate the config
export CONFIG_PATH="${CONFIG_PATH}"
export TRAIN_CONFIG="${CONFIG_PATH}"
export RUN_CFG="${CONFIG_PATH}"

# Quick sanity check on the target node to confirm visibility of the dataset
srun -N1 -n1 bash -lc 'hostname; ls -ld ${RCC_DATASET_ROOT}/rcc_webdataset_final/train' || {
  echo "[sbatch] Dataset path not visible on this node" >&2
  exit 4
}

# Fail-fast preflight (dataset presence + shards)
PYTHONPATH="${PYTHONPATH}" python -m src.training.utils.preflight || {
  echo "[sbatch] Preflight failed" >&2
  exit 5
}

# -------- Launch ---------------------------------------------------------------
# torchrun is used as in your previous runs; --standalone for single-node
command -v torchrun >/dev/null 2>&1 || { echo "[sbatch] ERROR: torchrun not found"; exit 3; }

# Prefer an explicit --config ARG if your launcher supports it; otherwise the env var above is used.
torchrun --nproc_per_node="${GPUS_PER_NODE}" --standalone \
  "${TRAIN_DIR}/launch_training.py" --config "${CONFIG_PATH}"
>>

trainer/backbones.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    import timm
except Exception:
    timm = None

__all__ = ["ResNetBackbone", "ViTBackbone", "mlp_head", "predictor_head", "get_backbone", "resolve_backbone_from_model_cfg"]


def _get_resnet_factory(name: str):
    factories = {
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    if name not in factories:
        raise ValueError(f"Unsupported ResNet backbone '{name}'.")
    return factories[name]


def _resolve_torchvision_weights(name: str, pretrained: bool):
    if not pretrained:
        return None
    enum_name = "ResNet34_Weights" if "34" in name else "ResNet50_Weights"
    weights_enum = getattr(models, enum_name, None)
    if weights_enum is None:
        raise RuntimeError(f"Pretrained weights for '{name}' not available in this torchvision version.")
    return weights_enum.DEFAULT


class ResNetBackbone(nn.Module):
    """
    ResNet come estrattore di feature con:
      - forward_global: pooled feature [B, D]
      - forward_tokens: token spatiali [B, T, C] da un blocco selezionato
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = False, return_tokens_from: str = "layer4"):
        super().__init__()
        factory = _get_resnet_factory(name)
        weights = _resolve_torchvision_weights(name, pretrained)
        model = factory(weights=weights)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.out_dim = model.fc.in_features
        self.tokens_source = return_tokens_from

    def _forward_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return {"layer1": l1, "layer2": l2, "layer3": l3, "layer4": l4}

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._forward_stages(x)["layer4"]
        return torch.flatten(F.adaptive_avg_pool2d(feats, 1), 1)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._forward_stages(x)[self.tokens_source]
        b, c, h, w = feats.shape
        return feats.flatten(2).transpose(1, 2).contiguous().view(b, h * w, c)


class ViTBackbone(nn.Module):
    """Light wrapper for ViT-S/16 using timm (features only)."""
    def __init__(
        self,
        name: str = "vit_small_patch16_224",
        pretrained: bool = False,
        *,
        patch_size: Optional[int] = None,
        freeze_patch_embed: bool = False,
        random_patch_proj: bool = False,
        bn_in_patch: bool = False,
        input_size: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm not available: pip install timm")
        # Accept 'input_size' from configs (e.g., i_jepa ablations). Not all timm models
        # support 'img_size', so fall back gracefully.
        try:
            self.vit = timm.create_model(
                name,
                pretrained=pretrained,
                num_classes=0,
                img_size=input_size,
                dynamic_img_size=True,
            )
        except TypeError:
            self.vit = timm.create_model(
                name,
                pretrained=pretrained,
                num_classes=0,
                dynamic_img_size=True,
            )
        self.out_dim = self.vit.num_features
        patch_embed = getattr(self.vit, "patch_embed", None)
        patch_sz = None
        if patch_embed is not None and hasattr(patch_embed, "patch_size"):
            patch_sz = patch_embed.patch_size
            if isinstance(patch_sz, (tuple, list)):
                patch_sz = patch_sz[0]
        if patch_sz is None:
            patch_sz = patch_size or 16
        self.patch_size = int(patch_sz)
        # Keep a record of intended input size (for downstream logic if needed)
        self.default_img_size = input_size or getattr(self.vit, "img_size", None)
        # opzionale: congela l'embed dei patch per stabilizzare i primi step
        if freeze_patch_embed and hasattr(self.vit, "patch_embed"):
            for p in self.vit.patch_embed.parameters():
                p.requires_grad = False
        # (gli altri flag sono placeholder compatibili)
        # Previeni NaN: clamp LayerScale / attn se presenti
        if hasattr(self.vit, "blocks"):
            for blk in self.vit.blocks:
                if hasattr(blk, "ls1") and hasattr(blk.ls1, "gamma"):
                    blk.ls1.gamma.data.clamp_(min=1e-4)
                if hasattr(blk, "ls2") and hasattr(blk.ls2, "gamma"):
                    blk.ls2.gamma.data.clamp_(min=1e-4)

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vit.forward_features(x)  # [B,T,C] for ViT
        return feats.mean(dim=1) if feats.dim() == 3 else torch.flatten(
            torch.nn.functional.adaptive_avg_pool2d(feats, 1), 1
        )

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vit.forward_features(x)
        return feats if feats.dim() == 3 else feats.flatten(2).transpose(1, 2)


def get_backbone(name: str, pretrained: bool, **kwargs) -> nn.Module:
    n = name.lower()
    if n in ("resnet34","resnet50"):
        return ResNetBackbone(n, pretrained)
    # add common aliases to be robust to configs
    if n in ("vit_s16", "vit_small", "vit_small_16", "vit_small_patch16", "vit_small_patch16_224", "vit_base_patch16_224"):
        return ViTBackbone(name, pretrained, **kwargs)
    raise ValueError(f"Unsupported backbone: {name}")

def resolve_backbone_from_model_cfg(model_cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Accept both:
      model.backbone: "vit_small_patch16_224"
    or:
      model.backbone: { name: "vit_small_patch16_224", patch_size: 16, ... }
    Fallback to model.backbone_opts for the string form.
    """
    spec = model_cfg.get("backbone", "resnet50")
    if isinstance(spec, dict):
        name = spec.get("name") or spec.get("type")
        if not isinstance(name, str) or not name:
            raise ValueError("model.backbone must be a string or a dict with a 'name' (or 'type') field.")
        opts = {k: v for k, v in spec.items() if k not in ("name", "type")}
        return name, opts
    # string form + optional legacy options
    return str(spec), (model_cfg.get("backbone_opts", {}) or {})


def _bn1d(dim: int, affine: bool = True) -> nn.BatchNorm1d:
    return nn.BatchNorm1d(dim, affine=affine)


def _relu() -> nn.ReLU:
    return nn.ReLU(inplace=True)


def _linear(in_f: int, out_f: int, bias: bool = False) -> nn.Linear:
    return nn.Linear(in_f, out_f, bias=bias)


def mlp_head(in_dim: int, hidden: int, out_dim: int, bn_last_affine: bool = False) -> nn.Sequential:
    """MLP 3-layer con BN e ReLU; ultima BN opzionale affine."""
    return nn.Sequential(
        _linear(in_dim, hidden, bias=False),
        _bn1d(hidden),
        _relu(),
        _linear(hidden, hidden, bias=False),
        _bn1d(hidden),
        _relu(),
        _linear(hidden, out_dim, bias=False),
        _bn1d(out_dim, affine=bn_last_affine),
    )


def predictor_head(dim: int, hidden: int = 4096) -> nn.Sequential:
    """MLP predittore stile BYOL/MoCoV3."""
    return nn.Sequential(
        _linear(dim, hidden, bias=False),
        _bn1d(hidden),
        _relu(),
        _linear(hidden, dim),
    )
>>

trainer/features.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.io import append_row_csv, ensure_dir, prefixed
try:
    from ..utils.torch_ops import move_to
except ModuleNotFoundError:  # pragma: no cover - fallback for namespace issues
    import importlib.util
    import sys
    from pathlib import Path as _Path

    _torch_ops_path = _Path(__file__).resolve().parent.parent / "utils" / "torch_ops.py"
    spec = importlib.util.spec_from_file_location("src.training.utils.torch_ops", _torch_ops_path)
    module = importlib.util.module_from_spec(spec) if spec and spec.loader else None
    if module and spec and spec.loader:
        spec.loader.exec_module(module)
        sys.modules.setdefault("src.training.utils.torch_ops", module)
        move_to = module.move_to  # type: ignore[attr-defined]
    else:  # pragma: no cover
        raise
from ..utils.viz import plot_confusion

__all__ = [
    "extract_features",
    "save_features",
    "visualize_features_umap_pca",
    "train_linear_probe_torch",
    "extract_split",
    "save_parquet",
]

try:  # pragma: no cover - optional dependency
    from sklearn.metrics import confusion_matrix
except Exception:  # pragma: no cover
    confusion_matrix = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None  # type: ignore


@torch.no_grad()
def extract_features(backbone: torch.nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estrae feature normalizzate dal backbone per l'intero loader.
    Restituisce tuple (features, labels) come matrici NumPy.
    """
    backbone.eval().to(device)
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for batch in loader:
        batch = move_to(batch, device)
        inputs = batch["inputs"]
        targets = batch["targets"]
        if hasattr(backbone, "forward_global"):
            embeddings = backbone.forward_global(inputs)
        else:
            embeddings = backbone(inputs)
        embeddings = F.normalize(embeddings, dim=-1)
        feats.append(embeddings.detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())

    if not feats:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    features = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(labels, axis=0).astype(np.int64, copy=False)
    return features, y


def save_features(backbone, loaders, device: torch.device, out_dir: Path, tag: str) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    feature_dir = ensure_dir(Path(out_dir) / "features")
    for split_name in ("train", "val"):
        loader = loaders.get(split_name)
        if loader is None:
            continue
        X, y = extract_features(backbone, loader, device)
        np.save(feature_dir / f"{tag}_{split_name}_X.npy", X)
        np.save(feature_dir / f"{tag}_{split_name}_y.npy", y)
        paths[f"{split_name}_X"] = str(feature_dir / f"{tag}_{split_name}_X.npy")
        paths[f"{split_name}_y"] = str(feature_dir / f"{tag}_{split_name}_y.npy")
    return paths


def visualize_features_umap_pca(X: np.ndarray, y: np.ndarray, out_png: Path, labels=None) -> None:
    if X.size == 0 or y.size == 0:
        return
    emb = None
    try:  # pragma: no cover - optional dependency
        import warnings
        import umap  # type: ignore
        warnings.filterwarnings(
            "ignore",
            message="n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.",
            category=UserWarning,
            module="umap.umap_"
        )
        emb = umap.UMAP(n_components=2, random_state=1337).fit_transform(X)
    except Exception:
        try:
            from sklearn.decomposition import PCA
            emb = PCA(n_components=2, random_state=1337).fit_transform(X)
        except Exception:
            return
    if emb is None:
        return
    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    fig = plt.figure()
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=y, s=4, alpha=0.7, cmap="tab10")
    if labels is not None and len(labels) > 0:
        # crea una legenda con i nomi delle classi
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(i)), label=labels[i]) for i in range(len(labels))]
        plt.legend(handles=handles, title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _as_numpy(arr: np.ndarray) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr)


def train_linear_probe_torch(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    n_epochs: int,
    lr: float,
    wd: float,
    batch_size: int,
    out_dirs: Dict[str, Path],
    model_key: str,
) -> Tuple[Dict[str, float], str]:
    Xtr = _as_numpy(Xtr).astype(np.float32, copy=False)
    Xva = _as_numpy(Xva).astype(np.float32, copy=False)
    ytr = _as_numpy(ytr).astype(np.int64, copy=False)
    yva = _as_numpy(yva).astype(np.int64, copy=False)

    if Xtr.size == 0 or Xva.size == 0:
        return {"val_acc": float("nan")}, ""

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    in_dim = Xtr.shape[1]
    classes = np.unique(np.concatenate([ytr, yva])) if ytr.size or yva.size else np.arange(Xtr.shape[1])
    num_classes = int(len(classes)) if len(classes) > 0 else 1

    head = nn.Linear(in_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    # Pesi di classe = inv. frequenza (normalizzati)
    _, counts = np.unique(ytr, return_counts=True)
    w = counts.max() / np.maximum(counts, 1)
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    def _iter_batches(X: np.ndarray, y: np.ndarray):
        N = X.shape[0]
        if N == 0:
            return
        for start in range(0, N, batch_size):
            stop = min(N, start + batch_size)
            xb = torch.from_numpy(X[start:stop]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(y[start:stop]).to(device=device, dtype=torch.long)
            yield xb, yb

    best_acc = -math.inf
    best_path = str(prefixed(out_dirs["checkpoints"], model_key, "ssl_linear_best", "pt"))
    csv_path = prefixed(out_dirs["metrics"], model_key, "ssl_linear_timeseries", "csv")

    for epoch in range(max(1, n_epochs)):
        # train
        head.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in _iter_batches(Xtr, ytr):
            optimizer.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * yb.size(0)
            total_count += yb.size(0)

        # validation
        head.eval()
        val_loss = 0.0
        val_count = 0
        correct = 0
        preds: list[int] = []
        with torch.no_grad():
            for xb, yb in _iter_batches(Xva, yva):
                logits = head(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.detach()) * yb.size(0)
                val_count += yb.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum())
                preds.extend(pred.detach().cpu().tolist())

        train_loss = total_loss / max(1, total_count)
        val_loss_avg = val_loss / max(1, val_count)
        val_acc = correct / max(1, val_count)
        append_row_csv(
            csv_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss_avg,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0].get("lr", 0.0),
            },
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"state_dict": head.state_dict(), "in_dim": in_dim, "num_classes": num_classes},
                best_path,
            )
            if confusion_matrix is not None and val_count > 0:
                cm = confusion_matrix(yva.tolist(), preds)  # type: ignore[arg-type]
                labels = [str(cls) for cls in sorted(set(int(c) for c in classes))]
                plot_confusion(
                    cm,
                    labels,
                    prefixed(out_dirs["plots"], model_key, "ssl_linear_confusion_val", "png"),
                )

    return {"val_acc": best_acc}, best_path


# ------------------------------------------------------------------ retro-compatibility helpers
@torch.no_grad()
def extract_split(backbone: torch.nn.Module, loader, device: torch.device) -> Dict[str, np.ndarray]:
    X, y = extract_features(backbone, loader, device)
    return {"X": X, "y": y}


def save_parquet(features: np.ndarray, labels: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(features.astype("float32"))
    df["label"] = labels.astype("int32")
    df.to_parquet(out_path)
    return out_path
>>

trainer/heads.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn

try:  # pragma: no cover - optional dependency
    from sklearn.metrics import f1_score
except ImportError:  # pragma: no cover
    def f1_score(y_true, y_pred, average="macro"):
        return 0.0

__all__ = ["train_linear_head"]


def _load_parquet(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Carica un file parquet prodotto da save_parquet e restituisce tensori (features, labels).
    """
    df = pd.read_parquet(path)
    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    features = torch.tensor(df.drop(columns=["label"]).to_numpy(), dtype=torch.float32)
    return features, labels


def train_linear_head(
    train_pq: Path,
    val_pq: Path,
    num_classes: int,
    epochs: int = 50,
    lr: float = 1e-2,
) -> Dict[str, object]:
    """
    Addestra una testa lineare (torch.nn.Linear) sulle feature salvate.
    Ritorna statistiche per logging e lo state_dict del modello migliore.
    """
    Xtr, Ytr = _load_parquet(train_pq)
    Xva, Yva = _load_parquet(val_pq)

    head = nn.Linear(Xtr.shape[1], num_classes)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    history: list[Dict[str, float]] = []
    best = {"epoch": -1, "val_acc": -1.0, "val_loss": float("inf"), "val_f1_macro": 0.0, "state_dict": None}

    for epoch in range(epochs):
        head.train()
        optimizer.zero_grad()
        logits = head(Xtr)
        loss = criterion(logits, Ytr)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(Xva)
            val_loss = float(criterion(val_logits, Yva))
            val_pred = val_logits.argmax(dim=1)
            val_acc = float((val_pred == Yva).float().mean())
            y_true_np = Yva.cpu().numpy()
            y_pred_np = val_pred.cpu().numpy()
            val_f1 = float(f1_score(y_true_np, y_pred_np, average="macro")) if y_true_np.size else 0.0

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.detach()),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
            }
        )

        if val_acc > best["val_acc"]:
            best["epoch"] = epoch
            best["val_acc"] = val_acc
            best["val_loss"] = val_loss
            best["val_f1_macro"] = val_f1
            best["state_dict"] = {k: v.detach().cpu() for k, v in head.state_dict().items()}

    if best["state_dict"] is None:
        best["state_dict"] = {k: v.detach().cpu() for k, v in head.state_dict().items()}

    head.load_state_dict(best["state_dict"], strict=False)
    head.eval()
    with torch.no_grad():
        logits = head(Xva)
        val_pred = logits.argmax(dim=1)
        predictions = val_pred.cpu().numpy()
        targets = Yva.cpu().numpy()
        best["val_acc"] = float((val_pred == Yva).float().mean())
        best["val_f1_macro"] = float(f1_score(targets, predictions, average="macro")) if targets.size else 0.0

    return {
        "rows": history,
        "best": best,
        "state_dict": best["state_dict"],
        "val_targets": targets,
        "val_predictions": predictions,
    }
>>

trainer/loops.py codice <<
# trainer/loops.py
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from ..utils.torch_ops import move_to

__all__ = ["SSLBaseModel", "SLBaseModel", "SSLTrainer", "SLTrainer"]


# -----------------------------------------------------------------------------
# Base model contracts
# -----------------------------------------------------------------------------
class SSLBaseModel(nn.Module):
    """
    Contratto base per modelli SSL.
    Richiede:
      - from_config(cls, cfg) -> model
      - training_step(batch, global_step) -> {'loss_total': Tensor, 'loss_components': dict}
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SSLBaseModel":  # pragma: no cover - interfaccia
        raise NotImplementedError

    def training_step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def on_epoch_end(self, epoch: int) -> None:
        pass

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        torch.save({"state_dict": self.state_dict(), "extra": extra or {}}, path)

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu")
        self.load_state_dict(payload["state_dict"], strict=False)


class SLBaseModel(nn.Module):
    """Contratto base per modelli supervisionati."""

    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SLBaseModel":  # pragma: no cover - interfaccia
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def training_step(self, batch: Dict[str, Any], global_step: int, criterion: nn.Module) -> Dict[str, Any]:
        inputs, targets = batch["inputs"], batch["targets"]
        logits = self(inputs)
        loss = criterion(logits, targets)
        acc = (logits.argmax(1) == targets).float().mean().item()
        return {"loss_total": loss, "metrics": {"acc": acc}}


# -----------------------------------------------------------------------------
# Logging & progress utils
# -----------------------------------------------------------------------------
def _eta_hms(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _eta_secs(start: float, done: int, total: int) -> float:
    rate = (time.time() - start) / max(1, done)
    return (total - done) * rate


def _should_log(idx: int, total: Optional[int], every: Optional[int]) -> bool:
    if every is None:
        return True
    return idx == 1 or (every and idx % every == 0) or (total is not None and idx == total)


class _EMAMetrics:
    """Accumulatore semplice di medie ed EMA."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.ema: Dict[str, float] = {}
        self.sum: Dict[str, float] = {}

    def update(self, stats: Dict[str, float]) -> None:
        for k, v in stats.items():
            self.ema[k] = (1 - self.alpha) * self.ema.get(k, v) + self.alpha * v
            self.sum[k] = self.sum.get(k, 0.0) + v

    def averaged(self, denom: int) -> Dict[str, float]:
        return {k: self.sum[k] / max(1, denom) for k in self.sum}


# -----------------------------------------------------------------------------
# SSL Trainer
# -----------------------------------------------------------------------------
class SSLTrainer:
    """Loop di training per SSL con callback step-level e logging compatto."""

    def __init__(
        self,
        model: SSLBaseModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        ema_m: float = 0.0,
        device: Optional[torch.device] = None,
        log_every_steps: int = 0,
        log_tag: Optional[str] = None,
        grad_clip_max: float = 0.0,
        accumulate_steps: int = 1,
        amp: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # EMA on log-loss for smooth logging; disabled if ema_m <= 0
        self.ema_m = float(ema_m or 0.0)
        self._logloss_ema: float | None = None
        self.device = device or (torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"))
        self.log_every = int(log_every_steps)
        self.log_tag = log_tag or model.__class__.__name__
        self.grad_clip_max = float(max(0.0, grad_clip_max))
        self.accumulate = int(max(1, accumulate_steps))
        self._acc_counter = 0
        self._pending_grads = False
        self.model.to(self.device)
        # AMP (come SLTrainer): autocast + GradScaler
        self._amp_enabled = bool(amp and torch.cuda.is_available())
        try:
            import torch.amp as _amp  # torch>=2
            self.scaler = _amp.GradScaler("cuda", enabled=self._amp_enabled)
            self._autocast = lambda: _amp.autocast(device_type="cuda", enabled=self._amp_enabled)
        except Exception:
            from torch.cuda import amp as _amp
            self.scaler = _amp.GradScaler(enabled=self._amp_enabled)
            self._autocast = lambda: _amp.autocast(enabled=self._amp_enabled)

    # ---- internals ----------------------------------------------------------
    def _run_step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, float]:
        batch = move_to(batch, self.device)

        # Ottimizza layout memoria: channels_last per tensori 4D (NCHW).
        def _as_channels_last(obj: Any) -> Any:
            if torch.is_tensor(obj) and obj.dim() == 4:
                return obj.to(memory_format=torch.channels_last)
            if isinstance(obj, list):
                return [_as_channels_last(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_as_channels_last(v) for v in obj)
            if isinstance(obj, dict):
                return {k: _as_channels_last(v) for k, v in obj.items()}
            return obj

        batch = _as_channels_last(batch)
        # forward con autocast
        with self._autocast():
            out = self.model.training_step(batch, global_step)
        raw_loss = out["loss_total"]
        loss = raw_loss / float(self.accumulate)

        # grad accumulation
        if (self._acc_counter % self.accumulate) == 0:
            self.optimizer.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._pending_grads = True
        self._acc_counter += 1
        if (self._acc_counter % self.accumulate) == 0:
            if self.grad_clip_max > 0.0:
                clip_grad_norm_(self.model.parameters(), self.grad_clip_max)
            if self._amp_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self._pending_grads = False

        comp = {k: float(v) for k, v in out.get("loss_components", {}).items()}
        val = float(raw_loss.detach())
        # Maintain EMA of log-loss for more stable charts
        if self.ema_m > 0.0 and math.isfinite(val) and val > 0.0:
            logv = math.log(max(val, 1e-12))
            if self._logloss_ema is None:
                self._logloss_ema = logv
            else:
                self._logloss_ema = self.ema_m * self._logloss_ema + (1.0 - self.ema_m) * logv
            ema_linear = math.exp(self._logloss_ema)
        else:
            ema_linear = None
        comp["ssl_loss"] = val
        if ema_linear is not None:
            comp["ssl_loss_ema"] = float(ema_linear)   # smoothed on linear scale
            comp["ssl_logloss_ema"] = float(self._logloss_ema)  # optional: keep also the log-space value
        return comp

    def _train_steps(
        self,
        loader: Iterable,
        steps: int,
        start_step: int,
        step_callback: Optional[Callable[[int, Dict[str, float]], None]],
    ) -> Dict[str, float]:
        self.model.train()
        metrics = _EMAMetrics(alpha=0.1)
        it = iter(loader)
        t0 = time.time()
        last = t0

        for s in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            gstep = start_step + s + 1
            stats = self._run_step(batch, gstep)
            metrics.update(stats)
            if step_callback:
                step_callback(gstep, stats)

        # flush step finale se rimangono grad non applicati
        if self._pending_grads:
            if self.grad_clip_max > 0.0:
                clip_grad_norm_(self.model.parameters(), self.grad_clip_max)
            if self._amp_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self._pending_grads = False

        avg = metrics.averaged(steps)
        avg["steps"] = steps
        avg.update({f"{k}_ema": v for k, v in metrics.ema.items()})
        # Ensure orchestrator can track best model:
        # use averaged ssl_loss as epoch-level loss_total.
        avg["loss_total"] = float(avg.get("ssl_loss", float("inf")))
        return avg

    # ---- public API (compat) ------------------------------------------------
    def train_epoch_two_views(
        self,
        loader: Iterable,
        steps: int,
        start_step: int = 0,
        step_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> Dict[str, float]:
        return self._train_steps(loader, steps, start_step, step_callback)

    def train_epoch_multicrop(
        self,
        loader: Iterable,
        steps: int,
        start_step: int = 0,
        step_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> Dict[str, float]:
        return self._train_steps(loader, steps, start_step, step_callback)

    def train_epoch_single_image(
        self,
        loader: Iterable,
        steps: int,
        start_step: int = 0,
        step_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> Dict[str, float]:
        return self._train_steps(loader, steps, start_step, step_callback)


# -----------------------------------------------------------------------------
# SL Trainer (AMP-friendly)
# -----------------------------------------------------------------------------
class SLTrainer:
    """Loop SL con AMP opzionale, logging a ETA e scheduler per-epoch."""

    def __init__(
        self,
        model: SLBaseModel,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        amp: bool = True,
        log_tag: str = "SL",
        log_every_steps: Optional[int] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_tag = log_tag
        self.log_every = max(1, int(log_every_steps)) if log_every_steps else None
        self._amp_enabled = bool(amp and torch.cuda.is_available())
        self._current_device: Optional[torch.device] = None
        self.scaler, self._autocast_ctx = self._init_amp(self._amp_enabled)

    # ---- AMP helpers --------------------------------------------------------
    def _init_amp(self, enabled: bool):
        """Crea GradScaler e context manager autocast (torch>=2: torch.amp)."""
        try:
            import torch.amp as _amp  # PyTorch â‰¥ 2

            scaler = _amp.GradScaler("cuda", enabled=enabled)
            ctx = lambda: _amp.autocast(device_type="cuda", enabled=enabled)
            return scaler, ctx
        except Exception:
            from torch.cuda import amp as _amp  # fallback compat

            scaler = _amp.GradScaler(enabled=enabled)
            ctx = lambda: _amp.autocast(enabled=enabled)
            return scaler, ctx

    def _ensure_device(self, device: torch.device) -> None:
        """Sposta componenti su device e adegua AMP se CPU."""
        if self._current_device == device:
            return
        self.model = self.model.to(device, non_blocking=True)
        self.criterion = self.criterion.to(device)
        self._current_device = device
        if device.type != "cuda":
            self._amp_enabled = False
            self.scaler, self._autocast_ctx = self._init_amp(False)

    # ---- batch & step -------------------------------------------------------
    def _unpack_batch(self, batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Supporta sia dict SL canonico sia un fallback compat."""
        if "inputs" in batch and "targets" in batch:
            x = batch["inputs"].to(memory_format=torch.channels_last)
            return x.to(device, non_blocking=True), batch["targets"].to(device, non_blocking=True)
        # compat: alcuni loader forniscono "images"/"label"
        x = batch["images"][0].to(memory_format=torch.channels_last)
        return x.to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)

    def _update_optim(self, loss: torch.Tensor) -> None:
        """Applica step con/without GradScaler a seconda di AMP."""
        self.optimizer.zero_grad(set_to_none=True)
        if self._amp_enabled and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def _run_step(self, batch: Dict[str, Any], device: torch.device, train: bool) -> Tuple[float, float, int]:
        """Esegue un passo (fw/bw opz.) e restituisce (loss, acc, n)."""
        inputs, targets = self._unpack_batch(batch, device)
        autocast = self._autocast_ctx()
        with torch.set_grad_enabled(train):
            with autocast:
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
            if train:
                self._update_optim(loss)
        logits = logits.float()  # per argmax stabile anche in half
        acc = (logits.argmax(1) == targets).float().mean().item()
        return float(loss.detach()), acc, targets.size(0)

    # ---- epoch loop ---------------------------------------------------------
    def run_epoch(
        self,
        loader: Iterable,
        device: torch.device,
        train: bool = True,
        expected_total: Optional[int] = None,
    ) -> Dict[str, float]:
        """Esegue un'epoch su loader; logga ETA e restituisce medie pesate."""
        self._ensure_device(device)
        self.model.train(mode=train)

        try:
            total_batches = len(loader) if expected_total is None else expected_total
        except TypeError:
            total_batches = expected_total

        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        start = time.time()

        for idx, batch in enumerate(loader, 1):
            t0 = time.time()
            loss, acc, n = self._run_step(batch, device, train)
            total_loss += loss * n
            total_acc += acc * n
            total_samples += n

            if _should_log(idx, total_batches, self.log_every):
                if total_batches is not None:
                    eta = _eta_hms(_eta_secs(start, min(idx, total_batches), total_batches))
                    print(
                        f"[{self.log_tag}][{'train' if train else 'val'}] "
                        f"[{min(idx, total_batches)}/{total_batches}] ETA={eta} "
                        f"loss={loss:.4f} acc={acc:.4f} dt/step={time.time() - t0:.2f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[{self.log_tag}][{'train' if train else 'val'}] "
                        f"[step {idx}] loss={loss:.4f} acc={acc:.4f} dt/step={time.time() - t0:.2f}s",
                        flush=True,
                    )

        if train and self.scheduler is not None:
            self.scheduler.step()

        denom = max(1, total_samples)
        return {"loss": total_loss / denom, "acc": total_acc / denom}
>>

utils/device.py codice <<
"""Device selection utilities."""
from __future__ import annotations

import os
import time
from typing import List

import torch

__all__ = ["device_from_env"]


def device_from_env(allow_cpu: bool = False) -> torch.device:
    """
    Resolve the preferred torch.device respecting CUDA availability and config/env.
    Config (allow_cpu) takes precedence over env ALLOW_CPU=1.
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
>>

utils/.DS_Store codice <<
   Bud1            %                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 @      €                                        @      €                                          @      €                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   E   %                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       DSDB                             `          €                                           @      €                                          @      €                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              >>

utils/io.py codice <<
# utils/io.py
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict

__all__ = [
    "ensure_dir",
    "make_exp_id",
    "make_run_dirs",
    "prefixed",
    "append_row_csv",
    "dump_json",
    "load_json",
    "copy_yaml_config",
]


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (recursively) if missing and return it as a Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def make_exp_id(outputs_root: str, ts: str | None = None) -> str:
    """
    Build a canonical experiment id. If `ts` (YYYYMMDD-HHMMSS) is provided,
    use it verbatim to ensure cross-tool consistency (e.g., SLURM log naming).
    """
    root = ensure_dir(outputs_root)
    if ts is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"exp_{ts}"


def make_run_dirs(outputs_root: str, exp_id: str, exp_name: str, model_key: str, *, override_leaf: bool = False, outputs_group_dir: str | None = None):
    """
    Create (and return) a dict of run directories.
    If override_leaf=True, the run root becomes:
      <outputs>/experiments/<exp_id>/<exp_name>
    ignoring model_key (no extra descriptive level).
    If outputs_group_dir is provided, it is used as the absolute group directory
    (<outputs>/experiments/<exp_id>) to guard against mismatches across hosts.
    """
    from pathlib import Path
    base = Path(outputs_root) / "experiments" / exp_id
    if outputs_group_dir:
        base = Path(outputs_group_dir)
    if override_leaf:
        run_root = base / exp_name
    else:
        # Legacy layout (kept for backward compatibility)
        run_root = base / exp_name / model_key
    (run_root / "metrics").mkdir(parents=True, exist_ok=True)
    (run_root / "plots").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_root / "records").mkdir(parents=True, exist_ok=True)
    (run_root / "configuration").mkdir(parents=True, exist_ok=True)
    return {
        "root": run_root,
        "metrics": run_root / "metrics",
        "plots": run_root / "plots",
        "checkpoints": run_root / "checkpoints",
        "records": run_root / "records",
        "configuration": run_root / "configuration",
    }


def prefixed(path_dir: Path | str, model_key: str, stem: str, ext: str) -> Path:
    directory = ensure_dir(path_dir)
    extension = ext.lstrip(".")
    return directory / f"{model_key}__{stem}.{extension}"


def append_row_csv(path: Path | str, row: Dict[str, Any]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    write_header = not target.exists()
    with target.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return target


def dump_json(path: Path | str, payload: Dict[str, Any]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return target


def load_json(path: Path | str) -> Dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"JSON file not found: {target}")
    return json.loads(target.read_text())


def copy_yaml_config(src: str | None, dst_dir: Path | str) -> Path | None:
    if not src:
        return None
    source = Path(src)
    if not source.exists():
        return None
    destination = ensure_dir(dst_dir) / "experiment_snapshot.yaml"
    shutil.copy2(source, destination)
    return destination
>>

utils/paths.py codice <<
"""Centralized path resolution utilities for the training pipeline."""
from pathlib import Path
import os
from typing import Dict, Optional, Union

import yaml

# ---------------------------------------------------------------------
# Config path & RUN_INDEX (leggibili da ENV, con default sensati)
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CFG = REPO_ROOT / "src" / "training" / "configs" / "ablations" / "exp_debug_pipeline.yaml"
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(DEFAULT_CFG))).resolve()
RUN_INDEX = int(os.environ.get("RUN_INDEX", "-1"))

# Roots can be overridden from the environment for HPC/SLURM consistency
HOME_ROOT = Path(os.environ.get("HOME_ROOT", str(Path.home())))
SCRATCH_ROOT = Path(
    os.environ.get("SCRATCH_ROOT", "/beegfs-scratch/mla_group_01/workspace/mla_group_01")
)

def _as_abs(p: Union[Path, str]) -> Path:
    return Path(p).resolve() if not isinstance(p, Path) else p.resolve()

def _first_existing(*candidates: Path) -> Optional[Path]:
    for c in candidates:
        if c and c.exists():
            return c
    return None

def _env_path(name: str) -> Optional[Path]:
    val = os.environ.get(name, "").strip()
    return Path(val).expanduser() if val else None


def _expand_env_in_tree(tree: Dict) -> Dict:
    """Recursively expand environment variables in a mapping."""
    expanded: Dict = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            expanded[key] = _expand_env_in_tree(value)
        elif isinstance(value, str):
            expanded[key] = os.path.expandvars(value)
        else:
            expanded[key] = value
    return expanded


def _load_wds_from_yaml(include_path: Path) -> Optional[Dict[str, Dict[str, str]]]:
    """
    Optional override: load WebDataset paths from a YAML include file.
    The YAML is expected to contain:
    data:
      <dataset_key>:
        train_dir: ...
        val_dir: ...
        test_dir: ...
    """
    if not include_path.exists():
        return None
    raw = yaml.safe_load(include_path.read_text()) or {}
    data = raw.get("data") or {}
    data = _expand_env_in_tree(data)
    result: Dict[str, Dict[str, str]] = {}
    for key, section in data.items():
        if not isinstance(section, dict):
            continue
        result[key] = {
            "train_dir": str(Path(section.get("train_dir", "")).expanduser()),
            "val_dir": str(Path(section.get("val_dir", "")).expanduser()),
            "test_dir": str(Path(section.get("test_dir", "")).expanduser()),
        }
    return result or None

def _infer_from_outputs_group() -> tuple[Optional[Path], Optional[Path]]:
    """
    Se OUTPUTS_GROUP_DIR Ã¨ settata (da launch_ssl_ablations.sh),
    prova a risalire a outputs_root e project_root.
    Atteso: .../outputs/mlruns/experiments/<group_name>
    """
    og = _env_path("OUTPUTS_GROUP_DIR")
    if not og:
        return None, None
    try:
        outputs_root = og.parents[1]   # .../outputs/mlruns
        project_root = og.parents[3]   # .../wsi-ssrl-rcc_project
    except IndexError:
        return None, None
    if outputs_root.name != "mlruns" or outputs_root.parent.name != "outputs":
        return None, None
    return outputs_root, project_root


def get_all():
    """
    Ritorna tutti i path risolti, scegliendo prima BeeGFS (veloce, condiviso),
    con fallback alla copia locale nella repo solo se necessario.
    """
    env_project_root, env_outputs_root = _env_path("PROJECT_ROOT"), _env_path("OUTPUTS_ROOT")
    og_outputs_root, og_project_root   = _infer_from_outputs_group()

    # Roots preferite su BeeGFS, con fallback locale
    beegfs_project = SCRATCH_ROOT / "wsi-ssrl-rcc_project"
    home_project   = HOME_ROOT / "rcc-ssrl"  # fallback minimale

    project_root = _first_existing(
        env_project_root,
        og_project_root,
        beegfs_project,
        home_project,
    ) or home_project
    outputs_root = _first_existing(
        env_outputs_root,
        og_outputs_root,
        project_root / "outputs" / "mlruns",
    ) or (project_root / "outputs" / "mlruns")

    # WebDataset shards: prima BeeGFS, poi fallback locale nella repo
    wds_env_root = (
        _env_path("RCC_DATASET_ROOT")
        or _env_path("RCC_WDS_ROOT")
        or _env_path("WDS_ROOT")
        or _env_path("WEB_DATASET_ROOT")
    )
    if not wds_env_root and (env_project_root or og_project_root):
        base = env_project_root or og_project_root
        wds_env_root = base / "data" / "processed"

    wds_beegfs_root = beegfs_project / "data" / "processed"
    wds_home_root   = REPO_ROOT / "src" / "data" / "processed"

    def _dataset_root(base: Path) -> Path:
        base = Path(base)
        return base if base.name == "rcc_webdataset_final" else base / "rcc_webdataset_final"

    wds_env_root = _dataset_root(wds_env_root) if wds_env_root else _dataset_root(wds_beegfs_root)

    def _wds_dir(name: str) -> Optional[Path]:
        return _first_existing(
            wds_env_root and Path(wds_env_root) / name,
            _dataset_root(wds_beegfs_root) / name,
            _dataset_root(wds_home_root) / name,
        )

    # Usata come hint se tutte le dir mancano: preferiamo rispettare eventuali override da ENV
    wds_root_hint = wds_env_root or _dataset_root(wds_beegfs_root) or _dataset_root(wds_home_root)

    wds_train = _wds_dir("train")
    wds_val   = _wds_dir("val")
    wds_test  = _wds_dir("test")

    # Se mancano gli shard ovunque, lasciamo che l'alto livello tiri un errore chiaro
    wds_map = {
        # "rcc_v2": {
        #     "train_dir": str(wds_train or wds_beegfs / "train"),
        #     "val_dir":   str(wds_val   or wds_beegfs / "val"),
        #     "test_dir":  str(wds_test  or wds_beegfs / "test"),
        # },
        "rcc_final_ablation": {
            "train_dir": str(wds_train or wds_root_hint / "train"),
            "val_dir":   str(wds_val   or wds_root_hint / "val"),
            "test_dir":  str(wds_test  or wds_root_hint / "test"),
        },
    }

    # Optional override from YAML include (centralized dataset mapping)
    include_path = REPO_ROOT / "src" / "training" / "configs" / "includes" / "data_paths.yaml"
    yaml_map = _load_wds_from_yaml(include_path) or {}
    wds_map.update(yaml_map)

    return {
        "project_root": str(project_root),
        "outputs_root": str(outputs_root),
        "webdataset": wds_map,
    }
>>

utils/preflight.py codice <<
"""Fail fast on missing dataset paths and shards before launching training."""
import glob
import os
import sys
from pathlib import Path


def _require_dir(path: str) -> int:
    """Ensure the directory exists and contains at least one shard; return shard count."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing directory: {path}")
    shards = glob.glob(os.path.join(path, "*.tar*"))
    if not shards:
        raise FileNotFoundError(f"No shards in {path}")
    return len(shards)


def main() -> int:
    root = os.environ.get("RCC_DATASET_ROOT", "").strip()
    if not root:
        raise FileNotFoundError("RCC_DATASET_ROOT is not set")

    dataset_root = Path(root) / "rcc_webdataset_final"
    required = {
        "train": str(dataset_root / "train"),
        "val": str(dataset_root / "val"),
    }

    counts = {name: _require_dir(path) for name, path in required.items()}
    print(f"[preflight] OK: train={required['train']} ({counts['train']} shards) "
          f"val={required['val']} ({counts['val']} shards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
>>

utils/torch_ops.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

__all__ = [
    "l2n",
    "cosine_logits",
    "ema_update",
    "copy_weights_and_freeze",
    "move_to",
    "safe_state_dict",
]


def l2n(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalizza sull'ultima dimensione evitando divisioni per zero."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _safe_tau(tau: float) -> float:
    return max(tau, 1e-8)


def cosine_logits(q: torch.Tensor, k: torch.Tensor, tau: float) -> torch.Tensor:
    """Logit di similaritÃ  coseno con temperatura."""
    return (l2n(q) @ l2n(k).t()) / _safe_tau(tau)


def move_to(obj: Any, device: torch.device) -> Any:
    """Sposta ricorsivamente tensori su device, mantenendo struttura."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = [move_to(v, device) for v in obj]
        return type(obj)(items) if isinstance(obj, tuple) else items
    return obj


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    """Aggiorna teacher = m*teacher + (1-m)*student (in-place, no grad)."""
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)


def copy_weights_and_freeze(dst: nn.Module, src: nn.Module) -> None:
    """Copia pesi e disabilita i gradienti del modulo di destinazione."""
    for p_dst, p_src in zip(dst.parameters(), src.parameters()):
        p_dst.data.copy_(p_src.data)
        p_dst.requires_grad = False


def safe_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """State dict pronto al salvataggio: tensori dettacchi e su CPU."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}
>>

scripts/launch_ssl_ablations.sh codice <<
#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Launch all SSL ablation jobs for a given model (dino_v3 | moco_v3 | ibot | i_jepa)
#
# Usage:
#   ./launch_ssl_ablations.sh <ssl_model>
#
# Example:
#   ./launch_ssl_ablations.sh dino_v3
#
# What it does:
#   - Scans configs/ablations/<ssl_model>/ for ablation YAMLs (exp_*_<ssl_model>_abl*.yaml)
#   - Creates an experiment group directory:
#       /beegfs-scratch/.../outputs/mlruns/experiments/exp_{DATETIME}_{ssl_model}/
#   - Creates a subfolder per ablation inside that group
#   - Submits one Slurm job per YAML via train_single_node.sbatch
#   - Writes Slurm logs to /home/mla_group_01/rcc-ssrl/src/logs/<ssl_model>/
#   - Saves a jobs manifest (job id, yaml, subdir) in both the logs dir and the group dir
# -----------------------------------------------------------------------------

if [[ $# -lt 1 ]]; then
  echo "ERROR: Missing <ssl_model> (dino_v3 | moco_v3 | ibot | i_jepa)" >&2
  exit 2
fi

MODEL="$1"
case "$MODEL" in
  dino_v3|moco_v3|ibot|i_jepa) ;;
  *) echo "ERROR: Unsupported model '$MODEL'"; exit 2 ;;
esac

# Map model -> code letter for job_name
case "$MODEL" in
  dino_v3) CODE="D" ;;
  moco_v3) CODE="M" ;;
  ibot)    CODE="B" ;;
  i_jepa)  CODE="J" ;;
esac

ROOT="/home/mla_group_01/rcc-ssrl"
TRAIN_DIR="$ROOT/src/training"
CFG_DIR="$TRAIN_DIR/configs/ablations/${MODEL}"
SBATCH_SCRIPT="$TRAIN_DIR/slurm/train_single_node.sbatch"

OUT_BASE="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments"
DATETIME="$(date +%Y%m%d_%H%M%S)"
EXP_GROUP="exp_${DATETIME}_${MODEL}"
EXP_ROOT="${OUT_BASE}/${EXP_GROUP}"

LOGDIR="$ROOT/src/logs/${MODEL}"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found"; exit 3; }
[[ -f "$SBATCH_SCRIPT" ]] || { echo "ERROR: sbatch script not found: $SBATCH_SCRIPT"; exit 3; }
[[ -d "$CFG_DIR" ]] || { echo "ERROR: config dir not found: $CFG_DIR"; exit 3; }

# Sort YAMLs deterministically (natural sort on ablNN)
mapfile -t CFGS < <(find "$CFG_DIR" -maxdepth 1 -type f -name "exp_${MODEL}_abl*.yaml" | sort -V)
[[ ${#CFGS[@]} -gt 0 ]] || { echo "ERROR: No ablation YAMLs in $CFG_DIR"; exit 4; }

mkdir -p "$EXP_ROOT" "$LOGDIR"

MANIFEST_LOG="${LOGDIR}/jobs_manifest_${EXP_GROUP}.tsv"
MANIFEST_EXP="${EXP_ROOT}/jobs_manifest.tsv"
echo -e "job_id\tjob_name\tmodel\tyaml_path\trun_name\texp_group\tslurm_log" | tee "$MANIFEST_LOG" > "$MANIFEST_EXP"

echo "==============================================================="
echo " Model           : $MODEL"
echo " Ablations       : ${#CFGS[@]}"
echo " Group (MLflow)  : ${EXP_GROUP}"
echo " Group dir       : ${EXP_ROOT}"
echo " Logs dir        : ${LOGDIR}"
echo " Sbatch script   : ${SBATCH_SCRIPT}"
echo "==============================================================="

for CFG in "${CFGS[@]}"; do
  BASE="$(basename "$CFG" .yaml)"                  # e.g., exp_moco_v3_abl01
  SUBDIR="${BASE}"
  # Extract ablation number (NN) robustly (works with/without .yaml)
  FNAME="$(basename "$CFG")"
  ABL_NUM="$(sed -nE 's/.*_abl([0-9]+)\.yaml$/\1/p' <<< "${FNAME}")"
  [[ -n "$ABL_NUM" ]] || ABL_NUM="$(sed -nE 's/.*_abl([0-9]+)$/\1/p' <<< "${BASE}")"
  [[ -n "$ABL_NUM" ]] || ABL_NUM="00"
  # Force decimal interpretation in case YAML uses leading zeros (08, 09, ...)
  ABL_NUM_FMT="$(printf "%02d" "$((10#${ABL_NUM}))")"

  JOB_NAME="tr${CODE}abl${ABL_NUM_FMT}"            # e.g., trMabl01

  # Create per-ablation folder inside the experiment group
  mkdir -p "${EXP_ROOT}/${SUBDIR}"

  # Slurm log files (coded)
  SLURM_LOG_OUT="${LOGDIR}/${JOB_NAME}_%j.out"
  SLURM_LOG_ERR="${LOGDIR}/${JOB_NAME}_%j.err"

  # Export ablation + naming hints for Python (consumed by launch_training/orchestrator)
  EXPORTS="ALL,MLFLOW_EXPERIMENT_NAME=${EXP_GROUP},EXP_GROUP=${EXP_GROUP},EXP_SUBDIR=${SUBDIR},OUTPUTS_GROUP_DIR=${EXP_ROOT},RUN_NAME=${JOB_NAME},ABLATION_ID=${ABL_NUM_FMT},MODEL_KEY=${MODEL}"

  # Submit and capture job id, passing the YAML path as first argument
  JOBID=$(sbatch \
    --job-name="${JOB_NAME}" \
    --output="${SLURM_LOG_OUT}" \
    --error="${SLURM_LOG_ERR}" \
    --export="${EXPORTS}" \
    --parsable \
    "${SBATCH_SCRIPT}" "${CFG}")

  # Concrete stdout log path for the manifest
  SLURM_LOG_CONCRETE="${SLURM_LOG_OUT//%j/${JOBID}}"
  echo -e "${JOBID}\t${JOB_NAME}\t${MODEL}\t${CFG}\t${JOB_NAME}\t${EXP_GROUP}\t${SLURM_LOG_CONCRETE}" | tee -a "$MANIFEST_LOG" >> "$MANIFEST_EXP"

  echo "Submitted: ${SUBDIR} as ${JOB_NAME} -> JobID ${JOBID}"
done

echo "---------------------------------------------------------------"
echo "All jobs submitted."
echo "Manifest:"
echo "  - ${MANIFEST_LOG}"
echo "  - ${MANIFEST_EXP}"
echo
echo "Monitor with:  watch -n 5 'squeue -u $USER -o \"%.18i %.12j %.8T %.10M %.6D %R\" | grep -E \"tr[DMJB]abl\"'"
echo "Tail a log:     tail -n 200 -f ${LOGDIR}/<JOB_NAME>_<JOBID>.out"
>>

scripts/generate_ssl_ablation_configs.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate SSL ablation experiment YAMLs from a shared template and per-model JSON definitions.

Usage examples:
  # Generate all models found in .../ablations/*_ablations.json
  python src/training/configs/tools/generate_ssl_ablation_configs.py

  # Only for one model (e.g., moco_v3)
  python src/training/configs/tools/generate_ssl_ablation_configs.py --model moco_v3

  # Also emit sbatch launchers
  python src/training/configs/tools/generate_ssl_ablation_configs.py --emit-launchers

This script:
  1) Loads a shared template YAML (exp_template_ssl.yaml).
  2) Optionally merges a per-suite base_config YAML (if provided in the JSON).
  3) Applies each experiment "override" (deep-merge) and enriches metadata (name, tags, date).
  4) Writes each final YAML into configs/ablations/{model_name}/exp_{model_name}_ablXX.yaml.
  5) Creates a single exp_debug_pipeline_smoke.yaml to quickly smoke-test the pipeline.

All comments/docstrings are in English as requested.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

try:
    import yaml  # PyYAML
except Exception as e:  # pragma: no cover
    print("ERROR: PyYAML is required. Try: pip install pyyaml", file=sys.stderr)
    raise

# --- Repository-absolute defaults (adapt to your layout if needed) -----------------

REPO_ROOT = Path("/home/mla_group_01/rcc-ssrl").resolve()
CFG_ROOT = REPO_ROOT / "src" / "training" / "configs"
TEMPLATE_PATH = CFG_ROOT / "templates" / "exp_template_ssl.yaml"
ABLATIONS_ROOT = CFG_ROOT / "ablations"
ABLATIONS_JSON = CFG_ROOT / "ablations" / "json_abl_exp"

SMOKE_CFG_PATH = ABLATIONS_ROOT / "exp_debug_pipeline_smoke.yaml"

# Pattern for per-model JSON files:
#   /.../configs/ablations/json_abl_exp/{model_name}_ablations.json
JSON_GLOB = (ABLATIONS_ROOT / "json_abl_exp").glob("*_ablations.json")

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def read_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file as a Python dict (empty dict if missing)."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(data: Mapping[str, Any], path: Path) -> None:
    """Write dict to YAML file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def deep_update(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """
    Recursively merge src into dst.
    - Dicts are merged key-by-key (src values override).
    - Lists and scalars are replaced by src.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def today_iso() -> str:
    """Return today's date in ISO format."""
    return dt.date.today().isoformat()


def ensure_list(x: Optional[Any]) -> List[Any]:
    """Return x as list; scalar -> [scalar], None -> []."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def enrich_experiment_meta(cfg: MutableMapping[str, Any],
                           suite: Mapping[str, Any],
                           exp: Mapping[str, Any],
                           model_name: str) -> None:
    """
    Set/augment experiment metadata:
    - experiment.name
    - experiment.date
    - experiment.paper_tags (union of existing + suite/model/exp tags)
    """
    # Canonical experiment name, independent of any "pretty" name in JSON:
    abl_id = int(exp.get("id", 0))
    exp_name = f"exp_{model_name}_abl{abl_id:02d}"
    exp_tags = ensure_list(exp.get("tags"))
    suite_name = suite.get("suite_name", "")
    extra_tags = [t for t in [suite_name, model_name, "ablation"] if t]

    # Ensure 'experiment' section exists
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = exp_name
    cfg["experiment"]["date"] = today_iso()

    # Merge/unique-ify paper_tags
    existing = ensure_list(cfg["experiment"].get("paper_tags"))
    merged = list(dict.fromkeys(existing + extra_tags + exp_tags))  # order-preserving unique
    cfg["experiment"]["paper_tags"] = merged


def apply_override(base_cfg: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a new dict with override deep-merged on top of base."""
    final = json.loads(json.dumps(base_cfg))  # deep copy via roundtrip
    return deep_update(final, override or {})


def load_suite_json(json_path: Path) -> Dict[str, Any]:
    """Load and minimally validate a suite JSON."""
    with json_path.open("r", encoding="utf-8") as f:
        suite = json.load(f)
    if "model" not in suite:
        raise ValueError(f"Missing 'model' in {json_path}")
    if "experiments" not in suite or not isinstance(suite["experiments"], list):
        raise ValueError(f"Missing/invalid 'experiments' in {json_path}")
    return suite


def build_full_smoke_config_from_template(template_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a full smoke config with shared fields and a runs section for each SSL family.
    """
    # Shared config fields (copied and pruned as needed)
    base = json.loads(json.dumps(template_cfg))

    # Overwrite/force shared fields to match requested output
    base["experiment"] = {
        "name": "exp_debug_pipeline_smoke",
        "date": today_iso(),
        "seed": 1337,
        "default_run_index": -1,
        "paper_tags": ["ssl", "histopathology", "smoke", "pipeline"],
        "mlflow_experiment": "RCC_SSL",
        "outputs_layout": "v2",
        "validate": {"paths": True, "steps_per_epoch": True, "batch_sizes": True},
        "allow_cpu": True,
    }

    # Data section
    base["data"] = {
        "sampler": {"limit_per_epoch": 128},
        "img_size": 224,
        "webdataset": {
            "dataset_key": "rcc_final_ablation",
            "shuffle_shards": 64,
            "shuffle_samples": 2000,
            "prefetch_factor": 2,
            "batch_size_ssl": 64,
            "batch_size_sl": 64,
            "num_workers": 4,
            "class_to_id": {"ccRCC": 0, "pRCC": 1, "CHROMO": 2, "ONCO": 3, "NOT_TUMOR": 4},
        },
    }

    # Model section
    base["model"] = {
        "backbone": {"name": "vit_small_patch16_224", "patch_size": 16},
        "ssl": {
            "name": "dino_v3",
            "temperature": 0.2,
            "hidden_dim": 4096,
            "proj_dim": 256,
            "use_multicrop": False,
            "temp_teacher_schedule": None,
            "ema_to_one": True,
            "clip_qk": None,
            "sync_bn": False,
            "aug": {
                "jitter": 0.4,
                "blur_prob": 0.1,
                "gray_prob": 0.2,
                "solarize_prob": 0.0,
            },
        },
    }

    # Train section
    base["train"] = {
        "optim": {
            "name": "adamw",
            "lr": 0.0003,
            "weight_decay": 0.05,
            "betas": [0.9, 0.999],
        },
        "scheduler": {"name": "cosine", "T_max": None},
        "ssl": {
            "epochs": 2,
            "steps_per_epoch": 2,
            "batch_size": 64,
            "accumulate_steps": 1,
            "num_workers": 4,
            "ema_momentum": 0.996,
            "amp": True,
            "grad_clip_max": 1.0,
            "probe": {
                "enabled": True,
                "epochs": 2,
                "lr": 0.05,
                "weight_decay": 0.0,
                "batch_size": 512,
            },
        },
    }

    # Aug section
    base["aug"] = {
        "base": {
            "rotate90": True,
            "hflip": True,
            "vflip": True,
            "random_resized_crop": {"scale": [0.6, 1.0], "ratio": [0.75, 1.33]},
            "gaussian_blur_p": 0.2,
            "sharpen_p": 0.2,
            "jpeg_artifacts_p": 0.1,
        },
        "stain": {
            "normalize": {"enable": False, "method": "macenko"},
            "jitter": {"enable": True, "mode": "HED", "delta": 0.02},
        },
        "mixing": {
            "mixup": {"enable": True, "alpha": 0.3},
            "cutmix": {"enable": True, "beta": 1.0, "p": 0.5},
        },
    }

    # Multiscale section
    base["multiscale"] = {
        "enable": True,
        "scales": [1.0, 1.5, 2.0],
        "n_per_scale": 1,
    }

    # Sampler section
    base["sampler"] = {
        "tissue_aware": {"enable": True, "min_tissue_frac": 0.6},
    }

    # Logging section
    base["logging"] = {
        "log_every_steps": 10,
        "metrics_csv_name": "ssl_timeseries.csv",
        "smoothing_window": 100,
    }

    # Artifacts section
    base["artifacts"] = {
        "save_best_model": True,
        "save_ckpt_every": 1,
        "export_csv": ["metrics"],
        "report_md": True,
    }

    # Runs section: one per SSL family
    base["runs"] = [
        {
            "name": "smoke_dino_v3_vit_s16",
            "mode": "ssl",
            "override": {
                "model": {"ssl": {"name": "dino_v3"}},
                "train": {"ssl": {"epochs": 2, "steps_per_epoch": 2, "accumulate_steps": 1, "probe": {"epochs": 2}}},
            },
        },
        {
            "name": "smoke_moco_v3_vit_s16",
            "mode": "ssl",
            "override": {
                "model": {"ssl": {"name": "moco_v3", "clip_qk": 50.0}},
                "train": {"ssl": {"epochs": 2, "steps_per_epoch": 2, "accumulate_steps": 1, "probe": {"epochs": 2}}},
            },
        },
        {
            "name": "smoke_ibot_vit_s16",
            "mode": "ssl",
            "override": {
                "model": {"ssl": {"name": "ibot", "num_prototypes": 8192, "temp_student": 0.10, "temp_teacher": 0.07, "use_multicrop": False}},
                "train": {"ssl": {"epochs": 2, "steps_per_epoch": 2, "accumulate_steps": 1, "probe": {"epochs": 2}}},
            },
        },
        {
            "name": "smoke_i_jepa_vit_s16",
            "mode": "ssl",
            "override": {
                "model": {"ssl": {"name": "i_jepa", "prediction_space": "global_mean", "ema_to_one": True, "jepa": {"context": {"scale": [0.85, 1.0], "aspect_ratio": [0.9, 1.1]}, "target": {"n": 2, "scale": [0.15, 0.20], "aspect_ratio": [0.75, 1.5], "no_overlap": True}}}},
                "train": {"ssl": {"epochs": 2, "steps_per_epoch": 2, "accumulate_steps": 1, "probe": {"epochs": 2}}},
            },
        },
    ]

    return base

# ------------------------------------------------------------------------------
# Main generation logic
# ------------------------------------------------------------------------------

def generate_for_suite(json_path: Path,
                       template_cfg: Dict[str, Any],
                       emit_launchers: bool = False) -> Tuple[str, List[Path]]:
    """
    Generate YAMLs for one suite JSON.
    Returns (model_name, [written_paths])
    """
    suite = load_suite_json(json_path)
    model_name: str = suite["model"]
    out_dir = ABLATIONS_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base config: merge (template <- base_config <- experiment override)
    base_cfg = json.loads(json.dumps(template_cfg))  # deep copy of template
    # Set the actual model name from suite
    base_cfg["model"]["ssl"]["name"] = model_name
    base_cfg_path = suite.get("base_config")
    if base_cfg_path:
        # Allow relative path from repo root
        base_path = (REPO_ROOT / base_cfg_path).resolve()
        if base_path.exists():
            deep_update(base_cfg, read_yaml(base_path))
        else:
            print(f"[WARN] base_config not found: {base_path}", file=sys.stderr)

    written: List[Path] = []

    # Create YAMLs per experiment
    for exp in suite["experiments"]:
        cfg_filename = exp.get("config_filename")
        if not cfg_filename:
            # Fallback to a conventional name
            cfg_filename = f"exp_{model_name}_abl{int(exp.get('id', 0)):02d}.yaml"

        override = exp.get("override", {}) or {}
        final_cfg = apply_override(base_cfg, override)
        enrich_experiment_meta(final_cfg, suite, exp, model_name)

        out_path = out_dir / cfg_filename
        write_yaml(final_cfg, out_path)
        written.append(out_path)

    # Optionally emit a launcher script using sbatch_script
    if emit_launchers:
        sbatch_script = suite.get("sbatch_script", "slurm/train_single_node.sbatch")
        launch_path = out_dir / f"launch_all_{model_name}.sh"
        with launch_path.open("w", encoding="utf-8") as sh:
            sh.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
            sh.write(f'echo "Launching {model_name} ablations..." \n\n')
            for p in written:
                rel = p.relative_to(REPO_ROOT)
                sh.write(f"sbatch {sbatch_script} {rel}\n")
            sh.write('\necho "Done."\n')
        os.chmod(launch_path, 0o755)

    return model_name, written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SSL ablation experiment configs.")
    parser.add_argument("--model", type=str, default=None,
                        help="Only generate for this model name (e.g., moco_v3).")
    parser.add_argument("--emit-launchers", action="store_true",
                        help="Also create per-model launch_all_{model}.sh with sbatch commands.")
    parser.add_argument("--no-smoke", action="store_true",
                        help="Do not (re)generate the pipeline smoke config.")
    args = parser.parse_args(argv)

    # Load shared template
    if not TEMPLATE_PATH.exists():
        print(f"ERROR: Template not found: {TEMPLATE_PATH}", file=sys.stderr)
        return 2
    template_cfg = read_yaml(TEMPLATE_PATH)

    # Discover JSONs
    json_files = sorted(JSON_GLOB)
    if args.model:
        json_files = [p for p in json_files if p.name == f"{args.model}_ablations.json"]

    if not json_files:
        print("ERROR: No *_ablations.json found matching the selection.", file=sys.stderr)
        return 3

    print(f"[INFO] Using template: {TEMPLATE_PATH}")
    for j in json_files:
        print(f"[INFO] Suite: {j}")

    # Generate per-suite
    summary: List[Tuple[str, int]] = []
    for json_path in json_files:
        model, written = generate_for_suite(json_path, template_cfg, emit_launchers=args.emit_launchers)
        summary.append((model, len(written)))

    # Generate/update smoke config unless disabled
    if not args.no_smoke:
        smoke_cfg = build_full_smoke_config_from_template(template_cfg)
        write_yaml(smoke_cfg, SMOKE_CFG_PATH)
        print(f"[INFO] Smoke config written: {SMOKE_CFG_PATH}")

    # Report
    print("\n[SUMMARY]")
    for model, count in summary:
        print(f"  {model}: {count} YAMLs written in {ABLATIONS_ROOT / model}")
    if not args.no_smoke:
        print(f"  + smoke: {SMOKE_CFG_PATH}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
>>

