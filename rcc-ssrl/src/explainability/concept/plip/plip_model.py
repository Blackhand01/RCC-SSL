#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from transformers import CLIPModel, AutoProcessor

try:
    from huggingface_hub import snapshot_download
    HAVE_HF_HUB = True
except Exception:
    HAVE_HF_HUB = False

try:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
except Exception as e:
    raise RuntimeError("torchvision is required for PLIP preprocessing") from e


def _as_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _dtype_from_precision(precision: str) -> torch.dtype:
    precision = (precision or "fp16").lower()
    if precision in ("fp16", "float16"):
        return torch.float16
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def ensure_model_local(model_id: str, model_local_dir: Path, hf_cache_dir: Optional[Path] = None) -> Path:
    model_local_dir.mkdir(parents=True, exist_ok=True)
    # Heuristic: if config.json exists, assume it's a valid snapshot.
    if (model_local_dir / "config.json").exists():
        return model_local_dir
    if not HAVE_HF_HUB:
        raise RuntimeError("huggingface_hub not available; cannot snapshot_download")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_local_dir),
        cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
    )
    return model_local_dir


def build_clip_preprocess(proc) -> transforms.Compose:
    # Use processor image stats for correctness.
    ip = getattr(proc, "image_processor", None)
    if ip is None:
        # Fallback: simple 224 center crop, CLIP-like mean/std unknown.
        size = 224
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        size = ip.size["shortest_edge"] if isinstance(ip.size, dict) and "shortest_edge" in ip.size else ip.size
        crop_h = ip.crop_size["height"] if isinstance(ip.crop_size, dict) else ip.crop_size
        crop_w = ip.crop_size["width"] if isinstance(ip.crop_size, dict) else ip.crop_size
        size = int(size)
        crop_h = int(crop_h)
        crop_w = int(crop_w)
        mean = list(ip.image_mean)
        std = list(ip.image_std)

        # If crop differs, respect crop size
        if crop_h != size or crop_w != size:
            crop = (crop_h, crop_w)
        else:
            crop = size

    # default CLIP uses bicubic resize
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(crop if "crop" in locals() else size),
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@dataclass
class PLIP:
    model: CLIPModel
    processor: any
    device: torch.device
    dtype: torch.dtype
    max_text_len: int
    score_scale: float
    preprocess: transforms.Compose
    model_id: str
    model_path: str


def load_plip(
    model_id: str,
    model_local_dir: Optional[Union[str, Path]] = None,
    device: str = "cuda",
    precision: str = "fp16",
    score_scale: Optional[float] = 100.0,
    hf_cache_dir: Optional[Union[str, Path]] = None,
) -> PLIP:
    dev = _as_device(device)
    dtype = _dtype_from_precision(precision)
    hf_cache_dir = Path(hf_cache_dir) if hf_cache_dir else None

    model_path: Union[str, Path] = model_id
    if model_local_dir is not None:
        model_local_dir = Path(model_local_dir)
        model_path = ensure_model_local(model_id, model_local_dir, hf_cache_dir=hf_cache_dir)

    model = CLIPModel.from_pretrained(model_path)
    proc = AutoProcessor.from_pretrained(model_path)

    model.eval()
    model.to(dev)

    max_text_len = int(getattr(model.config.text_config, "max_position_embeddings", 77))

    # If score_scale is None, use learned CLIP logit_scale.
    if score_scale is None:
        score_scale = float(model.logit_scale.exp().detach().cpu().item())

    preprocess = build_clip_preprocess(proc)

    return PLIP(
        model=model,
        processor=proc,
        device=dev,
        dtype=dtype,
        max_text_len=max_text_len,
        score_scale=float(score_scale),
        preprocess=preprocess,
        model_id=model_id,
        model_path=str(model_path),
    )


@torch.inference_mode()
def encode_text(plip: PLIP, prompts: List[str]) -> torch.Tensor:
    # Force max_length to avoid "tokenizer.model_max_length is huge" truncation warnings.
    inputs = plip.processor(
        text=prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=plip.max_text_len,
    )
    inputs = {k: v.to(plip.device) for k, v in inputs.items()}
    if plip.device.type == "cuda" and plip.dtype in (torch.float16, torch.bfloat16):
        with torch.autocast(device_type="cuda", dtype=plip.dtype):
            feats = plip.model.get_text_features(**inputs)
    else:
        feats = plip.model.get_text_features(**inputs)
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    out_dtype = plip.dtype if plip.device.type == "cuda" else torch.float32
    return feats.to(dtype=out_dtype)


@torch.inference_mode()
def encode_images(plip: PLIP, images: torch.Tensor) -> torch.Tensor:
    # images: [B,3,H,W] already normalized
    images = images.to(plip.device, non_blocking=True)
    if plip.device.type == "cuda" and plip.dtype in (torch.float16, torch.bfloat16):
        with torch.autocast(device_type="cuda", dtype=plip.dtype):
            feats = plip.model.get_image_features(pixel_values=images)
    else:
        feats = plip.model.get_image_features(pixel_values=images)
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    out_dtype = plip.dtype if plip.device.type == "cuda" else torch.float32
    return feats.to(dtype=out_dtype)


@torch.inference_mode()
def score(plip: PLIP, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    # logits: [B,C]
    if image_features.dtype != text_features.dtype:
        text_features = text_features.to(dtype=image_features.dtype)
    return plip.score_scale * (image_features @ text_features.T)
