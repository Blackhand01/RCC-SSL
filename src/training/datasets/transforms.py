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
    # choices 0/90/180/270, preserve morphology
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

# ---- Stain Normalization / Jitter (fallback if libs not available) ----
class StainNormalizer:
    def __init__(self, method: str = "macenko", enable: bool = False):
        self.enable = bool(enable)
        self.method = (method or "macenko").lower()
        self._impl = None
        if not self.enable: return
        # try staintools / torchstain
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
                # Note: in absence of "fit" on target, use auto-fit per image
                N.fit(tgt)
                out = N.transform(tgt)
                return Image.fromarray(out.astype(np.uint8))
            elif self._lib == "torchstain":
                # torchstain requires torch tensors in OD/HE; for brevity: no-op if missing params
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

# ---- Tissue fraction (for sampler filter) ----
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
    # randstainna stub: if you want to use an external lib, it remains disabled here for design no-deps
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

    # Global #1: NO solarize; Global #2: solarize with dedicated probability
    global_aug1 = _build_aug(global_size, global_scale, do_blur=True, do_solar=False)
    # use dedicated probability for the second global (DINO style: solarize only on one view)
    old_sp = float(solarize_prob)
    solarize_prob = float(solarize_prob_g2)
    global_aug2 = _build_aug(global_size, global_scale, do_blur=True, do_solar=True)
    # restore (for safety in case of closures)
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
    # NOTE: 'g' and 'l' are 3D tensors (C,H,W). 'channels_last' requires 4D.
    # First: stack -> 4D (N,C,H,W). Then: safely apply channels_last.
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
