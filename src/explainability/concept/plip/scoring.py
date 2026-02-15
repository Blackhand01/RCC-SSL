from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import torch

from .plip_model import PLIP, encode_images, encode_text, score


@torch.inference_mode()
def encode_text_cached(
    plip: PLIP,
    prompts: List[str],
    cache_dir: Optional[Union[str, Path]] = None,
    cache_key: Optional[str] = None,
) -> torch.Tensor:
    """
    Encode prompts with optional on-disk caching to avoid recomputation across runs.
    """
    cache_path: Optional[Path] = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fname = cache_key if cache_key else "text_features.pt"
        cache_path = cache_dir / fname
        if cache_path.exists():
            saved = torch.load(cache_path, map_location="cpu")
            feats = saved["text_features"] if isinstance(saved, dict) and "text_features" in saved else saved
            return feats.to(device=plip.device, dtype=plip.dtype)

    feats = encode_text(plip, prompts)
    if cache_path:
        torch.save(
            {
                "text_features": feats.detach().cpu(),
                "prompts": prompts,
                "model_id": plip.model_id,
                "max_text_len": plip.max_text_len,
            },
            cache_path,
        )
    return feats


@torch.inference_mode()
def score_batch(
    plip: PLIP, images: Union[torch.Tensor, Iterable[torch.Tensor]], text_features: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode images and return (logits, probs) against provided text features.
    """
    if isinstance(images, Iterable) and not isinstance(images, torch.Tensor):
        images = torch.stack(list(images), dim=0)
    img_feats = encode_images(plip, images)
    logits = score(plip, img_feats, text_features)
    probs = torch.softmax(logits, dim=1)
    return logits, probs
